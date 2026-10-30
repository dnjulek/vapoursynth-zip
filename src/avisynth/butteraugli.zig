//! avszip_Butteraugli(clip reference, clip distorted, float "intensity_target",
//! float "hf_asymmetry", float "xmul", int "distmap")
//!
//! Planar RGB (8-16 bit int, 32-bit float), bit-exact with vszip. Sets the
//! per-frame props Butteraugli and Butteraugli3Norm on: the reference itself
//! (distmap=0), butteraugli_main's heatmap in the reference's format
//! (distmap=1, any reference alpha copied through), or the raw diffmap as Y32
//! (distmap=2). The sRGB transfer is undone unless frame 0 of that clip
//! carries `_Transfer` = 8 (linear). The distorted clip's alpha is ignored.

const std = @import("std");
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/butteraugli.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_Butteraugli";
pub const params = "c[distorted]c[intensity_target]f[hf_asymmetry]f[xmul]f[distmap]i";
const Scratch = []align(common.vec_len) f32;
const rgb_ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };

const Source = struct {
    srgb: bool = true,
    lut: []f32 = &.{},
};

const Data = struct {
    other: avs.OwnedClip,
    kind: filter.Input.Kind = .f32,
    bits: u8 = 32,
    src1: Source = .{},
    src2: Source = .{},
    params: filter.Params = .{},
    distmap: u2 = 0,
    alpha: bool = false,
    good: f64 = 0,
    bad: f64 = 0,
    w: u32 = 0,
    h: u32 = 0,
    // Reusable scratch blocks; concurrent calls each pop their own block (or
    // allocate one) and push it back when done.
    pool: std.ArrayList(Scratch) = .empty,
    pool_mutex: std.Io.Mutex = .init,

    fn freeLuts(self: *Data) void {
        allocator.free(self.src1.lut);
        allocator.free(self.src2.lut);
    }

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |buf| allocator.free(buf);
        self.pool.deinit(allocator);
        self.freeLuts();
        self.other.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn acquireScratch(d: *Data) !Scratch {
    {
        d.pool_mutex.lockUncancelable(common.io);
        defer d.pool_mutex.unlock(common.io);
        if (d.pool.pop()) |buf| return buf;
    }
    const buf = try allocator.alignedAlloc(f32, common.alignment, filter.scratchSize(d.w, d.h));
    // the kernel's padding columns and Malta border are never written
    @memset(buf, 0);
    return buf;
}

fn releaseScratch(d: *Data, buf: Scratch) void {
    d.pool_mutex.lockUncancelable(common.io);
    defer d.pool_mutex.unlock(common.io);
    d.pool.append(allocator, buf) catch allocator.free(buf);
}

/// Input view of a frame's R, G, B planes with one shared byte stride (a
/// mismatching or unaligned plane is repacked).
fn input(d: *const Data, src: *const Source, frame: avs.Frame, scratch: *hz.Scratch) !filter.Input {
    var planes: [3][*]const u8 = undefined;
    var pitch: usize = 0;
    inline for (0..3) |p| {
        const raw = frame.plane(u8, rgb_ids[p]);
        if (p == 0) pitch = raw.pitch;
        planes[p] = (try hz.canonical(u8, raw, pitch, scratch)).ptr;
    }
    return .{ .kind = d.kind, .planes = planes, .stride = pitch, .lut = src.lut, .srgb = src.srgb };
}

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const env = ctx.env();
    var scratch: hz.Scratch = .{};
    defer scratch.deinit();

    const ref = try ctx.childFrame(n);
    defer ref.release();
    const dist = try ctx.getFrame(d.other.get(), n);
    defer dist.release();

    const buf = acquireScratch(d) catch return ctx.fail("out of memory", .{});
    defer releaseScratch(d, buf);
    const in1 = input(d, &d.src1, ref, &scratch) catch return ctx.fail("out of memory", .{});
    const in2 = input(d, &d.src2, dist, &scratch) catch return ctx.fail("out of memory", .{});
    var bctx = filter.Context.init(buf, d.w, d.h);
    const res = filter.run(&bctx, in1, in2, d.params);

    var out = if (d.distmap == 0) ref.retain() else try ctx.newFrame(ref);
    errdefer out.release();
    switch (d.distmap) {
        0 => try ctx.makePropertyWritable(&out),
        1 => {
            writeHeatmap(d, out, &bctx);
            if (d.alpha) avs.copyPlane(env, out, ref, c.AVS_PLANAR_A);
        },
        else => writeRaw(d, out, &bctx),
    }

    const props = out.propsRW(env);
    switch (d.distmap) {
        1 => {
            // butteraugli_main saves these colors as an sRGB image; the
            // reference's color tags (e.g. a linear _Transfer) don't apply
            props.setInt("_Transfer", 13, .replace) catch {}; // IEC 61966-2-1 (sRGB)
            props.setInt("_Primaries", 1, .replace) catch {}; // BT.709
            props.setInt("_Matrix", 0, .replace) catch {}; // RGB
            props.setInt("_ColorRange", 0, .replace) catch {}; // full
        },
        2 => {
            // plain numbers, not light: drop the inherited color tags
            _ = props.delete("_Transfer");
            _ = props.delete("_Primaries");
            _ = props.delete("_Matrix");
        },
        else => {},
    }
    props.setFloat("Butteraugli", res.max, .replace) catch return ctx.fail("could not set Butteraugli.", .{});
    props.setFloat("Butteraugli3Norm", res.norm3, .replace) catch return ctx.fail("could not set Butteraugli3Norm.", .{});
    return out;
}

/// butteraugli_main's --distmap colors, in the reference's RGB format.
fn writeHeatmap(d: *const Data, dst: avs.Frame, bctx: *const filter.Context) void {
    const map = bctx.diffmap();
    const ms = bctx.full.stride;
    switch (d.kind) {
        inline else => |k| {
            const T = switch (k) {
                .u8 => u8,
                .u16 => u16,
                .f32 => f32,
            };
            var planes: [3]avs.PlaneMut(T) = undefined;
            for (&planes, rgb_ids) |*p, id| p.* = dst.planeMut(T, id);
            const peak: f32 = if (T == f32) 1.0 else @floatFromInt((@as(u32, 1) << @intCast(d.bits)) - 1);
            for (0..d.h) |y| {
                const rows = [3][]T{ planes[0].row(y), planes[1].row(y), planes[2].row(y) };
                for (0..d.w) |x| {
                    const rgb = filter.heatmapRgb(map[y * ms + x], d.good, d.bad);
                    for (0..3) |ch| {
                        rows[ch][x] = if (T == f32) rgb[ch] else @intFromFloat(@round(rgb[ch] * peak));
                    }
                }
            }
        },
    }
}

fn writeRaw(d: *const Data, dst: avs.Frame, bctx: *const filter.Context) void {
    const map = bctx.diffmap();
    const ms = bctx.full.stride;
    const out = dst.planeMut(f32, c.AVS_PLANAR_Y);
    for (0..d.h) |y| @memcpy(out.row(y)[0..d.w], map[y * ms ..][0..d.w]);
}

/// Frame 0's `_Transfer`: linear (8) input skips the sRGB decode.
fn frame0Srgb(ctx: *avs.Create, clip: avs.Clip, comptime which: []const u8) !bool {
    const frame = clip.getFrame(0) catch return ctx.fail(which ++ " frame 0 failed.", .{});
    defer frame.release();
    return frame.propsRO(ctx.env).intOr("_Transfer", 2) != 8;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const ref = try ctx.clip(0, "reference");
    defer ref.release();
    const dist_in = try arg.clip("distorted");
    defer dist_in.release();
    const vi1 = ref.videoInfo();
    const vi2 = dist_in.videoInfo();
    try hz.compareClips(ctx, &.{ vi1, vi2 }, .SAME_LEN, .color_only);
    const f = try hz.planarFormat(ctx, vi1);
    if (f.family != .rgb) return ctx.fail("only RGB input is supported.", .{});
    if (vi1.width < 64 or vi1.height < 64) return ctx.fail("clips must be at least 64x64.", .{});

    var d: Data = .{ .other = .{} };
    // validated as f64 against the decimal bounds, then narrowed: an f32
    // bound (0.001f > 0.001) would reject the documented endpoint
    d.params.intensity_target = @floatCast(try arg.float(f64, "intensity_target", 80.0, 0.001, 10000.0));
    d.params.hf_asymmetry = @floatCast(try arg.float(f64, "hf_asymmetry", 1.0, 0.001, 1000.0));
    d.params.xmul = @floatCast(try arg.float(f64, "xmul", 1.0, 0.0, 1000.0));
    d.distmap = try arg.int(u2, "distmap", 0, 0, 2);
    d.src1.srgb = try frame0Srgb(ctx, ref, "reference");
    d.src2.srgb = try frame0Srgb(ctx, dist_in, "distorted");

    d.w = @intCast(vi1.width);
    d.h = @intCast(vi1.height);
    d.bits = f.bits;
    d.alpha = f.has_alpha;
    d.kind = switch (f.sample) {
        .u8 => .u8,
        .u16 => .u16,
        .f32 => .f32,
    };
    errdefer d.freeLuts();
    if (d.kind != .f32) {
        const lut_len: usize = if (d.kind == .u8) 256 else 65536;
        for ([_]*Source{ &d.src1, &d.src2 }) |s| {
            s.lut = try allocator.alloc(f32, lut_len);
            filter.buildLut(s.lut, @intCast(d.bits), s.srgb);
        }
    }
    d.good = filter.fuzzyInverse(1.5);
    d.bad = filter.fuzzyInverse(0.5);

    const out_vi: ?c.AVS_VideoInfo = if (d.distmap == 2)
        hz.withFormat(vi1.*, .{ .family = .gray, .sample = .f32, .bits = 32, .color_planes = 1, .has_alpha = false, .ssw = 0, .ssh = 0 })
    else
        null;
    const distorted = dist_in.retain();
    errdefer distorted.release();
    d.other = .{ .clip = distorted };
    const out = try F.create(ctx.env, ref, d, F.getFrame(getFrame), .{ .mt_mode = .nice, .vi = out_vi });
    return avs.Create.returnClip(out);
}
