//! vszip_SSIMULACRA2(clip reference, clip distorted)
//!
//! Returns `reference` with the per-frame prop SSIMULACRA2. Planar RGB inputs
//! (8-16 bit int, 32-bit float) are read natively, bit-exact with vszip.
//! YUV/Gray inputs are converted to RGBPS first with AviSynth's own
//! ConvertBits(32) + ConvertToPlanarRGB (matrix Rec709 above 650 lines, else
//! Rec601, like vszip's choice); vszip uses zimg (resize.Bicubic) there, so
//! scores of YUV inputs differ slightly (chroma resampler details). The sRGB transfer is undone in
//! the filter unless frame 0 carries `_Transfer` = 8 (linear).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter_ssim = @import("../filters/ssimulacra2.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_SSIMULACRA2";
pub const params = "c[distorted]c";
const Scratch = []align(common.vec_len) f32;

/// How one clip's frames are read: native RGB (int via LUT, or f32).
const Source = struct {
    kind: filter_ssim.Input.Kind = .f32,
    srgb: bool = true,
    lut: []f32 = &.{},
    format: avs.Format = undefined,
};

const Data = struct {
    other: avs.OwnedClip,
    src1: Source = .{},
    src2: Source = .{},
    // Reusable scratch buffers; concurrent calls each pop their own buffer
    // (or allocate one) and push it back when done.
    pool: std.ArrayList(Scratch) = .empty,
    pool_mutex: std.Io.Mutex = .init,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |buf| allocator.free(buf);
        self.pool.deinit(allocator);
        allocator.free(self.src1.lut);
        allocator.free(self.src2.lut);
        self.other.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn acquireScratch(d: *Data, size: usize) !Scratch {
    {
        d.pool_mutex.lockUncancelable(common.io);
        defer d.pool_mutex.unlock(common.io);
        if (d.pool.pop()) |buf| return buf;
    }
    return allocator.alignedAlloc(f32, common.alignment, size);
}

fn releaseScratch(d: *Data, buf: Scratch) void {
    d.pool_mutex.lockUncancelable(common.io);
    defer d.pool_mutex.unlock(common.io);
    d.pool.append(allocator, buf) catch allocator.free(buf);
}

/// Input view of a frame's R, G, B planes with one shared byte stride (the
/// planes of a planar RGB frame share a pitch; a mismatching or unaligned
/// one is repacked).
fn input(src: *const Source, frame: avs.Frame, scratch: *hz.Scratch) !filter_ssim.Input {
    const ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };
    var planes: [3][*]const u8 = undefined;
    var pitch: usize = 0;
    inline for (0..3) |p| {
        const raw = frame.plane(u8, ids[p]);
        if (p == 0) pitch = raw.pitch;
        const can = try hz.canonical(u8, raw, pitch, scratch);
        planes[p] = can.ptr;
    }
    return .{ .kind = src.kind, .planes = planes, .stride = pitch, .lut = src.lut, .srgb = src.srgb };
}

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    var frame = try ctx.childFrame(n); // reference: only the prop is added
    errdefer frame.release();
    const src2 = try ctx.getFrame(d.other.get(), n);
    defer src2.release();
    var scratch: hz.Scratch = .{};
    defer scratch.deinit();

    const w: u32 = @intCast(ctx.vi().width);
    const h: u32 = @intCast(ctx.vi().height);
    const buf = acquireScratch(d, filter_ssim.scratchSize(w, h)) catch return ctx.fail("out of memory", .{});
    defer releaseScratch(d, buf);
    const val = filter_ssim.process(try input(&d.src1, frame, &scratch), try input(&d.src2, src2, &scratch), w, h, buf);

    const env = ctx.env();
    try ctx.makePropertyWritable(&frame);
    frame.propsRW(env).setFloat("SSIMULACRA2", val, .replace) catch return ctx.fail("could not set SSIMULACRA2.", .{});
    return frame;
}

fn invokeClip(ctx: *avs.Create, name: [:0]const u8, args: []const c.AVS_Value, names: []const ?[*:0]const u8) !avs.Clip {
    var msg: ?[:0]const u8 = null;
    const v = ctx.env.invoke(name, avs.Value.arrayView(args), names, &msg) catch
        return ctx.fail("internal {s} failed: {s}", .{ name, msg orelse "unknown error" });
    defer v.release();
    if (!v.isClip()) return ctx.fail("internal {s} did not return a clip.", .{name});
    return v.takeClip(ctx.env);
}

/// Owned clip readable natively: planar RGB as-is, else converted to RGBPS.
fn prepare(ctx: *avs.Create, clip: avs.Clip, src: *Source) !avs.Clip {
    const vi = clip.videoInfo();
    const f = try hz.planarFormat(ctx, vi);
    var out = clip.retain();
    errdefer out.release();
    if (f.family != .rgb) {
        // The matrix comes from the clip's own `_Matrix` (and the range from
        // `_ColorRange`), as zimg does on the VapourSynth side. Only an absent
        // or unspecified `_Matrix` falls back to vszip's resolution guess
        // (BT.709 above 650 lines, else BT.601), which is tagged on the clip
        // instead of passed as `matrix=`: an explicit matrix would override
        // the tag, and the host still applies `_ColorRange` to a tagged clip.
        const tagged = blk: {
            const frame = out.getFrame(0) catch break :blk false;
            defer frame.release();
            const m = frame.propsRO(ctx.env).intOr("_Matrix", 2);
            break :blk m != 2;
        };
        if (!tagged) {
            const cv = avs.Value.fromClip(out);
            defer cv.release();
            const a0 = [_]c.AVS_Value{ cv.raw, c.avs_new_value_string("_Matrix"), c.avs_new_value_int(if (vi.height > 650) 1 else 6) };
            const t = try invokeClip(ctx, "propSet", &a0, &.{ null, null, null });
            out.release();
            out = t;
        }
        // Float first, then the matrix: like zimg, no intermediate integer
        // RGB rounding/clipping (measured: max |diff| vs zimg 0.003 instead
        // of 0.9 when converting through 8-bit RGB).
        const cv = avs.Value.fromClip(out);
        defer cv.release();
        const a1 = [_]c.AVS_Value{ cv.raw, c.avs_new_value_int(32) };
        const flt = try invokeClip(ctx, "ConvertBits", &a1, &.{ null, null });
        defer flt.release();
        const fv = avs.Value.fromClip(flt);
        defer fv.release();
        const a2 = [_]c.AVS_Value{fv.raw};
        const conv = try invokeClip(ctx, "ConvertToPlanarRGB", &a2, &.{null});
        out.release();
        out = conv;
    }

    const fmt = try hz.planarFormat(ctx, out.videoInfo());
    src.format = fmt;
    src.srgb = blk: {
        const frame = out.getFrame(0) catch break :blk true;
        defer frame.release();
        break :blk frame.propsRO(ctx.env).intOr("_Transfer", 2) != 8;
    };
    if (!fmt.isFloat()) {
        src.kind = if (fmt.sample == .u8) .u8 else .u16;
        src.lut = try allocator.alloc(f32, if (fmt.sample == .u8) 256 else 65536);
        filter_ssim.buildLut(src.lut, @intCast(fmt.bits), src.srgb);
    } else {
        src.kind = .f32;
    }
    return out;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const ref_in = try ctx.clip(0, "reference");
    defer ref_in.release();
    const dist_in = try ctx.clip(1, "distorted");
    defer dist_in.release();
    const vi1 = ref_in.videoInfo();
    const vi2 = dist_in.videoInfo();
    if (vi1.width != vi2.width or vi1.height != vi2.height) return ctx.fail("clips must have the same dimensions.", .{});
    if (vi1.num_frames != vi2.num_frames) return ctx.fail("clips must have the same length.", .{});

    var d: Data = .{ .other = .{} };
    errdefer {
        allocator.free(d.src1.lut);
        allocator.free(d.src2.lut);
    }
    const reference = try prepare(ctx, ref_in, &d.src1);
    defer reference.release();
    const distorted = try prepare(ctx, dist_in, &d.src2);
    errdefer distorted.release();
    d.other = .{ .clip = distorted };

    const out = try F.create(ctx.env, reference, d, F.getFrame(getFrame), .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
