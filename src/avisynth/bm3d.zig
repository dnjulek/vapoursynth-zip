//! avszip_BM3D(clip, clip "ref", float[] "sigma", int[] "block_step",
//!   int[] "bm_range", int "radius", int[] "ps_num", int[] "ps_range", bool "chroma",
//!   string[] "bm_error_s", string[] "transform_2d_s", string[] "transform_1d_s")
//!
//! vszipcu's BM3D on the CPU (filters/bm3d.zig), bit-exact with vszip.
//! 32-bit float only; alpha passes through. radius > 0 aggregates inside the
//! filter through the same `bm3d_common.TemporalCache` as vszip (no stacked
//! clip, same summation order), so the returned clip is the final one.
//!
//! Planes go to the kernel in place at their own pitch: 8x8 blocks never
//! read past a plane's width, so cropped sources need no canonical copy.
//!
//! Scratch: vszip keys it by thread id; here each request checks a block out
//! of a mutex-guarded pool (exclusive per request).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/bm3d.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;

pub const filter_name = "avszip_BM3D";
pub const params = "c[ref]c[sigma]f*[block_step]i*[bm_range]i*[radius]i[ps_num]i*[ps_range]i*[chroma]b" ++
    "[bm_error_s]s*[transform_2d_s]s*[transform_1d_s]s*";

const PlaneEntry = struct {
    p: filter.Params,
    /// logical plane of each entry plane
    planes: [3]u32,
};

const Data = struct {
    format: avs.Format,
    num_frames: c_int,
    ref: avs.OwnedClip = .{},
    radius: u32,
    process: [3]bool,
    entries: [3]PlaneEntry,
    n_entries: usize,
    cache: ?bc.TemporalCache = null,
    lock: std.Io.Mutex = .init,
    pool: std.ArrayList(*filter.Scratch) = .empty,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |sc| sc.destroy();
        self.pool.deinit(allocator);
        if (self.cache) |*cache| cache.deinit();
        self.ref.deinit();
    }

    fn acquire(self: *Data) !*filter.Scratch {
        {
            self.lock.lockUncancelable(common.io);
            defer self.lock.unlock(common.io);
            if (self.pool.pop()) |sc| return sc;
        }
        return filter.Scratch.create();
    }

    fn release(self: *Data, sc: *filter.Scratch) void {
        self.lock.lockUncancelable(common.io);
        defer self.lock.unlock(common.io);
        self.pool.append(allocator, sc) catch sc.destroy();
    }

    fn process4(self: *const Data) [4]bool {
        return .{ self.process[0], self.process[1], self.process[2], false };
    }
};

fn planeIn(f: avs.Frame, format: avs.Format, logical: u32) filter.PlaneIn {
    const p = f.plane(f32, format.planeId(logical));
    return .{ .data = @ptrCast(@alignCast(p.ptr)), .stride = p.stride() };
}

fn Bm3d(comptime final: bool, comptime name: []const u8) type {
    return struct {
        const F = avs.Filter(Data, name);

        fn spatial(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const ref = if (final) try ctx.getFrame(d.ref.get(), n) else src.retain();
            defer ref.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.process4());

            const sc = d.acquire() catch return ctx.fail("out of memory", .{});
            defer d.release(sc);
            for (d.entries[0..d.n_entries], 0..) |*e, ei| {
                const ring = sc.ringFor(ei, &e.p) catch return ctx.fail("out of memory", .{});
                var srcs: [3]filter.PlaneIn = undefined;
                var refs: [3]filter.PlaneIn = undefined;
                var sink: filter.DivideSink = .{ .dst = @splat(null), .stride = @splat(0) };
                for (0..e.p.n_planes) |i| {
                    const pl = e.planes[i];
                    srcs[i] = planeIn(src, d.format, pl);
                    if (final) refs[i] = planeIn(ref, d.format, pl);
                    if (e.p.process[i]) {
                        const dp = dst.planeMut(f32, d.format.planeId(pl));
                        sink.dst[i] = @ptrCast(@alignCast(dp.ptr));
                        sink.stride[i] = dp.stride();
                    }
                }
                filter.stage(final, &e.p, srcs[0..e.p.n_planes], refs[0..e.p.n_planes], ring, &sink);
            }
            return dst;
        }

        const max_span = 4 * filter.max_radius + 1;

        /// Source frames n - 2r .. n + 2r (clamped): every stage n - r ..
        /// n + r reads within them.
        const Frames = struct {
            lo: i32,
            src: [max_span]avs.Frame = undefined,
            ref: [max_span]avs.Frame = undefined,
            count: usize = 0,
        };

        const Stages = struct {
            d: *Data,
            fr: *const Frames,
            sc: *filter.Scratch,

            pub fn compute(self: *const Stages, m: i32) error{OutOfMemory}!void {
                const d = self.d;
                const r: i32 = @intCast(d.radius);
                const tw = 2 * d.radius + 1;
                for (d.entries[0..d.n_entries], 0..) |*e, ei| {
                    const ring = try self.sc.ringFor(ei, &e.p);
                    var srcs: [3 * (2 * filter.max_radius + 1)]filter.PlaneIn = undefined;
                    var refs: [3 * (2 * filter.max_radius + 1)]filter.PlaneIn = undefined;
                    for (0..e.p.n_planes) |i| {
                        const pl = e.planes[i];
                        for (0..tw) |z| {
                            const f = bc.clampFrame(m, @as(i64, @intCast(z)) - r, d.num_frames);
                            const k: usize = @intCast(f - self.fr.lo);
                            srcs[i * tw + z] = planeIn(self.fr.src[k], d.format, pl);
                            if (final) refs[i * tw + z] = planeIn(self.fr.ref[k], d.format, pl);
                        }
                    }
                    const sink: FoldSink = .{ .cache = &d.cache.?, .m = m, .planes = e.planes };
                    const n = e.p.n_planes * tw;
                    filter.stage(final, &e.p, srcs[0..n], refs[0..n], ring, &sink);
                }
            }
        };

        const FoldSink = struct {
            cache: *bc.TemporalCache,
            m: i32,
            planes: [3]u32,

            pub fn rows(self: *const FoldSink, slot: u32, plane: u32, y: u32, num: []const f32, den: []const f32) void {
                self.cache.fold(self.m, slot, self.planes[plane], y, num, den);
            }
        };

        const Out = struct {
            dst: avs.Frame,
            format: avs.Format,

            pub fn plane(self: *const Out, p: usize, num: []const f32, den: []const f32) void {
                const dp = self.dst.planeMut(f32, self.format.planeId(p));
                const w = dp.width;
                for (0..dp.height) |y| {
                    filter.divideRow(dp.row(y), num[y * w ..][0..w], den[y * w ..][0..w]);
                }
            }
        };

        fn temporal(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const span: i32 = 2 * @as(i32, @intCast(d.radius));
            const lo = bc.clampFrame(n, -span, d.num_frames);
            const hi = bc.clampFrame(n, span, d.num_frames);
            var fr: Frames = .{ .lo = lo };
            var got_ref: usize = 0;
            defer {
                for (fr.src[0..fr.count]) |f| f.release();
                for (fr.ref[0..got_ref]) |f| f.release();
            }
            var i = lo;
            while (i <= hi) : (i += 1) {
                fr.src[fr.count] = try ctx.childFrame(i);
                fr.count += 1;
                if (final) {
                    fr.ref[got_ref] = try ctx.getFrame(d.ref.get(), i);
                    got_ref += 1;
                }
            }

            const center = fr.src[@intCast(n - lo)];
            const dst = try ctx.newFrame(center);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, center, d.format, &d.process4());

            const sc = d.acquire() catch return ctx.fail("out of memory", .{});
            defer d.release(sc);
            const stages: Stages = .{ .d = d, .fr = &fr, .sc = sc };
            const out: Out = .{ .dst = dst, .format = d.format };
            d.cache.?.produce(n, &stages, &out) catch return ctx.fail("out of memory", .{});
            return dst;
        }
    };
}

fn selectGetFrame(comptime name: []const u8, final: bool, temporal: bool) avs.filter.RawGetFrame {
    return switch (final) {
        inline else => |f| blk: {
            const B = Bm3d(f, name);
            break :blk if (temporal) B.F.getFrame(B.temporal) else B.F.getFrame(B.spatial);
        },
    };
}

/// vszipcu's per-plane string rule: index i given, else the previous value
/// (the default for plane 0).
fn perPlaneEnum(comptime T: type, ctx: *avs.Create, arg: anytype, comptime key: []const u8, base: T) ![3]T {
    var v: [3]T = .{ base, base, base };
    const a = arg.arg(key);
    if (!a.isDefined()) return v;
    const len = a.len();
    if (len == 0) return ctx.fail(key ++ " must not be empty.", .{});
    if (len > 3) return ctx.fail(key ++ " has too many elements (got {d}, max 3).", .{len});
    for (0..3) |k| {
        if (k < len) {
            const e = a.at(k);
            if (!e.isString()) return ctx.fail(key ++ "[{d}] must be a string.", .{k});
            v[k] = T.parse(e.asString()) orelse return ctx.fail("invalid \"" ++ key ++ "\".", .{});
        } else v[k] = v[k - 1];
    }
    return v;
}

fn checkFloat32(ctx: *avs.Create, vi: *const c.AVS_VideoInfo) !avs.Format {
    const format = try hz.planarFormat(ctx, vi);
    if (!format.isFloat()) return ctx.fail("only constant format 32 bit float input supported.", .{});
    return format;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const name = filter_name;
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try checkFloat32(ctx, vi);
    if (vi.width < 64 or vi.height < 64) return ctx.fail("clip must be at least 64x64.", .{});

    const ref = try arg.optClip("ref");
    errdefer if (ref) |r| r.release();
    if (ref) |r| {
        const rvi = r.videoInfo();
        const rf = hz.planarFormat(ctx, rvi) catch return ctx.fail("\"ref\" must be of the same format as \"clip\".", .{});
        if (rf.family != format.family or rf.bits != format.bits or rf.ssw != format.ssw or rf.ssh != format.ssh) {
            return ctx.fail("\"ref\" must be of the same format as \"clip\".", .{});
        }
        if (rvi.width != vi.width or rvi.height != vi.height) return ctx.fail("\"ref\" must be of the same dimensions as \"clip\".", .{});
        if (rvi.num_frames != vi.num_frames) return ctx.fail("\"ref\" must be of the same number of frames as \"clip\".", .{});
    }
    const final = ref != null;

    // sigma: validated as f64, used as f32 like the GPU plugin
    var sigma = try arg.perPlaneFloat(f32, "sigma", 3.0, 0, math.floatMax(f32));
    var process: [3]bool = undefined;
    for (0..3) |i| process[i] = !(sigma[i] < math.floatEps(f32));
    const factor = if (final) filter.sigma_factor_final else filter.sigma_factor_basic;
    for (&sigma) |*sv| sv.* *= factor;

    // larger ranges select the same (clamped) windows; the cap keeps the
    // search arithmetic in i32
    const block_step = try arg.perPlaneInt(i64, "block_step", 8, 1, 8);
    const bm_range = try arg.perPlaneInt(i64, "bm_range", 9, 1, 1 << 20);
    const radius = try arg.int(u32, "radius", 0, 0, filter.max_radius);
    const ps_num = try arg.perPlaneInt(i64, "ps_num", 2, 1, 8);
    const ps_range = try arg.perPlaneInt(i64, "ps_range", 4, 1, 1 << 20);

    const chroma = try arg.boolean("chroma", false);
    if (chroma and (format.family != .yuv or format.ssw != 0 or format.ssh != 0)) {
        return ctx.fail("clip format must be YUV444 when \"chroma\" is true.", .{});
    }

    const bm_error = try perPlaneEnum(filter.BmError, ctx, arg, "bm_error_s", .ssd);
    const t2d = try perPlaneEnum(filter.Transform, ctx, arg, "transform_2d_s", .dct);
    const t1d = try perPlaneEnum(filter.Transform, ctx, arg, "transform_1d_s", .dct);

    const np: usize = format.color_planes;
    for (np..3) |i| process[i] = false;
    var any = false;
    for (0..np) |i| any = any or process[i];
    if (!any) {
        // nothing to process (every sigma 0): pass the clip through, like the
        // other filters
        if (ref) |r| r.release();
        return avs.Create.returnClip(clip.retain());
    }

    const base = struct {
        fn f(i: usize, w: u32, h: u32, np_: u32, sg: [3]f32, pr: [3]bool, bs: [3]i64, br: [3]i64, pn: [3]i64, prg: [3]i64, r: u32, be: [3]filter.BmError, a: [3]filter.Transform, b: [3]filter.Transform) filter.Params {
            return .{
                .sigma = sg,
                .process = pr,
                .n_planes = np_,
                .width = w,
                .height = h,
                .block_step = @intCast(bs[i]),
                .bm_range = @intCast(br[i]),
                .ps_num = @intCast(pn[i]),
                .ps_range = @intCast(prg[i]),
                .radius = r,
                .bm_error = be[i],
                .t2d = a[i],
                .t1d = b[i],
            };
        }
    }.f;

    var d: Data = .{
        .format = format,
        .num_frames = vi.num_frames,
        .radius = radius,
        .process = process,
        .entries = undefined,
        .n_entries = 0,
    };
    const w: u32 = @intCast(vi.width);
    const h: u32 = @intCast(vi.height);
    var pw: [3]u32 = @splat(0);
    var ph: [3]u32 = @splat(0);
    for (0..np) |p| {
        pw[p] = format.planeWidth(w, p);
        ph[p] = format.planeHeight(h, p);
    }
    if (chroma) {
        d.entries[0] = .{
            .p = base(0, w, h, 3, sigma, process, block_step, bm_range, ps_num, ps_range, radius, bm_error, t2d, t1d),
            .planes = .{ 0, 1, 2 },
        };
        d.n_entries = 1;
    } else {
        for (0..np) |p| {
            if (!process[p]) continue;
            if (pw[p] < 8 or ph[p] < 8) return ctx.fail("every processed plane must be at least 8x8.", .{});
            d.entries[d.n_entries] = .{
                .p = base(p, pw[p], ph[p], 1, .{ sigma[p], sigma[p], sigma[p] }, .{ true, false, false }, block_step, bm_range, ps_num, ps_range, radius, bm_error, t2d, t1d),
                .planes = .{ @intCast(p), 0, 0 },
            };
            d.n_entries += 1;
        }
    }

    // The cache's eviction threshold counts requests in flight; AviSynth
    // doesn't report its Prefetch thread count, so assume one per CPU.
    if (radius > 0) {
        const threads: u32 = @intCast(std.Thread.getCpuCount() catch 8);
        d.cache = bc.TemporalCache.init(radius, vi.num_frames, process, pw, ph, threads);
    }
    d.ref = .{ .clip = ref };
    errdefer if (d.cache) |*cache| cache.deinit();

    const gf = selectGetFrame(name, final, radius > 0);
    const out = try Bm3d(false, name).F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
