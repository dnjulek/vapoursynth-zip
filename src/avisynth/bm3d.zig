//! avszip_BM3D / avszip_BM3Dv2(clip, clip "ref", float[] "sigma", int[] "block_step",
//!   int[] "bm_range", int "radius", int[] "ps_num", int[] "ps_range", bool "chroma",
//!   int "extractor_exp", string[] "bm_error_s", string[] "transform_2d_s",
//!   string[] "transform_1d_s", bool "zero_init")
//! avszip_VAggregate(clip, clip src, int[] planes)
//!
//! vszipcu's BM3D on the CPU (filters/bm3d_cuda.zig), bit-exact with vszip.
//! 32-bit float only; alpha passes through. radius > 0 aggregates inside the
//! filter through the same `bm3d_common.TemporalCache` as vszip (no stacked
//! clip, same summation order), so BM3D and BM3Dv2 return the same clip.
//! VAggregate aggregates stacked clips made by the GPU plugins.
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

const filter = @import("../filters/bm3d_cuda.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;

const shared_params = "c[ref]c[sigma]f*[block_step]i*[bm_range]i*[radius]i[ps_num]i*[ps_range]i*[chroma]b" ++
    "[extractor_exp]i[bm_error_s]s*[transform_2d_s]s*[transform_1d_s]s*[zero_init]b";

pub const BM3D = Entry(false);
pub const BM3Dv2 = Entry(true);

fn Entry(comptime v2: bool) type {
    return struct {
        pub const filter_name = if (v2) "avszip_BM3Dv2" else "avszip_BM3D";
        pub const params = shared_params;

        pub fn create(ctx: *avs.Create) !avs.Value {
            return createImpl(v2, filter_name, ctx);
        }
    };
}

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
fn perPlaneEnum(comptime T: type, ctx: *avs.Create, i: usize, comptime key: []const u8, base: T) ![3]T {
    var v: [3]T = .{ base, base, base };
    const a = ctx.arg(i);
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

fn createImpl(comptime v2: bool, comptime name: []const u8, ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try checkFloat32(ctx, vi);
    if (vi.width < 64 or vi.height < 64) return ctx.fail("clip must be at least 64x64.", .{});

    const ref = try ctx.optClip(1, "ref");
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
    var sigma = try ctx.perPlaneFloat(f32, 2, "sigma", 3.0, 0, math.floatMax(f32));
    var process: [3]bool = undefined;
    for (0..3) |i| process[i] = !(sigma[i] < math.floatEps(f32));
    const factor = if (final) filter.sigma_factor_final else filter.sigma_factor_basic;
    for (&sigma) |*sv| sv.* *= factor;

    // larger ranges select the same (clamped) windows; the cap keeps the
    // search arithmetic in i32
    const block_step = try ctx.perPlaneInt(i64, 3, "block_step", 8, 1, 8);
    const bm_range = try ctx.perPlaneInt(i64, 4, "bm_range", 9, 1, 1 << 20);
    const radius = try ctx.int(u32, 5, "radius", 0, 0, filter.max_radius);
    const ps_num = try ctx.perPlaneInt(i64, 6, "ps_num", 2, 1, 8);
    const ps_range = try ctx.perPlaneInt(i64, 7, "ps_range", 4, 1, 1 << 20);

    const chroma = try ctx.boolean(8, "chroma", false);
    if (chroma and (format.family != .yuv or format.ssw != 0 or format.ssh != 0)) {
        return ctx.fail("clip format must be YUV444 when \"chroma\" is true.", .{});
    }
    const extractor_exp = try ctx.int(i32, 9, "extractor_exp", 0, -149, 127);
    const extractor: f32 = if (extractor_exp != 0) math.ldexp(@as(f32, 1.0), extractor_exp) else 0.0;

    const bm_error = try perPlaneEnum(filter.BmError, ctx, 10, "bm_error_s", .ssd);
    const t2d = try perPlaneEnum(filter.Transform, ctx, 11, "transform_2d_s", .dct);
    const t1d = try perPlaneEnum(filter.Transform, ctx, 12, "transform_1d_s", .dct);
    _ = try ctx.boolean(13, "zero_init", true); // stacked-clip option of the GPU plugin; no stacked clip here

    const np: usize = format.color_planes;
    for (np..3) |i| process[i] = false;
    var any = false;
    for (0..np) |i| any = any or process[i];
    if (!any) {
        if (v2) {
            // BM3Dv2 passes such a clip through
            if (ref) |r| r.release();
            return avs.Create.returnClip(clip.retain());
        }
        return ctx.fail("all planes have sigma < FLT_EPSILON (nothing to process).", .{});
    }

    const base = struct {
        fn f(i: usize, w: u32, h: u32, np_: u32, sg: [3]f32, pr: [3]bool, bs: [3]i64, br: [3]i64, pn: [3]i64, prg: [3]i64, r: u32, be: [3]filter.BmError, a: [3]filter.Transform, b: [3]filter.Transform, ex: f32) filter.Params {
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
                .extractor = ex,
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
            .p = base(0, w, h, 3, sigma, process, block_step, bm_range, ps_num, ps_range, radius, bm_error, t2d, t1d, extractor),
            .planes = .{ 0, 1, 2 },
        };
        d.n_entries = 1;
    } else {
        for (0..np) |p| {
            if (!process[p]) continue;
            if (pw[p] < 8 or ph[p] < 8) return ctx.fail("every processed plane must be at least 8x8.", .{});
            d.entries[d.n_entries] = .{
                .p = base(p, pw[p], ph[p], 1, .{ sigma[p], sigma[p], sigma[p] }, .{ true, false, false }, block_step, bm_range, ps_num, ps_range, radius, bm_error, t2d, t1d, extractor),
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

/// Aggregates a stacked clip made by vszipcu / BM3DCUDA `BM3D(radius > 0)`
/// (vszipcu's VAggregate: same checks, same order of summation).
pub const VAggregate = struct {
    pub const filter_name = "avszip_VAggregate";
    pub const params = "c[src]c[planes]i*";

    const AggData = struct {
        format: avs.Format,
        stack: avs.OwnedClip,
        num_frames: c_int,
        radius: i32,
        process: [4]bool,

        pub fn deinit(self: *AggData) void {
            self.stack.deinit();
        }
    };

    const F = avs.Filter(AggData, filter_name);

    fn getFrame(d: *AggData, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
        const r = d.radius;
        const nf = d.num_frames;
        const src = try ctx.childFrame(n);
        defer src.release();
        const tw: usize = @intCast(2 * r + 1);
        var stack: [2 * filter.max_radius + 1]avs.Frame = undefined;
        var got: usize = 0;
        defer for (stack[0..got]) |f| f.release();
        for (0..tw) |i| {
            stack[i] = try ctx.getFrame(d.stack.get(), bc.clampFrame(n, @as(i64, @intCast(i)) - r, nf));
            got += 1;
        }

        const dst = try ctx.newFrame(src);
        errdefer dst.release();
        hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.process);
        for (0..d.format.color_planes) |p| {
            if (!d.process[p]) continue;
            const id = d.format.planeId(p);
            const dp = dst.planeMut(f32, id);
            const h = dp.height;
            var rows: [2 * filter.max_radius + 1][*]const f32 = undefined;
            var wts: [2 * filter.max_radius + 1][*]const f32 = undefined;
            for (0..h) |y| {
                for (0..tw) |i| {
                    const z: usize = @intCast(bc.aggZ(@intCast(i), n, nf, r));
                    const sp = stack[i].plane(f32, id);
                    rows[i] = sp.row(z * 2 * h + y).ptr;
                    wts[i] = sp.row((z * 2 + 1) * h + y).ptr;
                }
                bc.aggRow(dp.row(y), rows[0..tw], wts[0..tw]);
            }
        }
        return dst;
    }

    pub fn create(ctx: *avs.Create) !avs.Value {
        const stack = try ctx.clip(0, "clip");
        defer stack.release();
        const src = try ctx.clip(1, "src");
        defer src.release();
        const vi = stack.videoInfo();
        const svi = src.videoInfo();
        const sf = hz.planarFormat(ctx, svi) catch return ctx.fail("\"src\" must be 32 bit float.", .{});
        if (!sf.isFloat()) return ctx.fail("\"src\" must be 32 bit float.", .{});
        if (svi.width < 64 or svi.height < 64) return ctx.fail("\"src\" must be at least 64x64.", .{});
        const cf = hz.planarFormat(ctx, vi) catch return ctx.fail("\"clip\" must have the same format and frame count as \"src\".", .{});
        // color planes only: the stack's alpha is never read (output alpha
        // comes from `src`), so alpha presence may differ
        var cf_color = cf;
        cf_color.has_alpha = sf.has_alpha;
        if (!cf_color.eql(sf) or vi.num_frames != svi.num_frames) {
            return ctx.fail("\"clip\" must have the same format and frame count as \"src\".", .{});
        }
        const ratio = @divTrunc(vi.height, svi.height);
        if (ratio < 6 or @rem(ratio - 2, 4) != 0 or @rem(vi.height, svi.height) != 0 or vi.width != svi.width) {
            return ctx.fail("\"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).", .{});
        }
        const radius = @divTrunc(ratio - 2, 4);
        if (radius > filter.max_radius) return ctx.fail("radius too large.", .{});
        // stacked chroma planes must still divide evenly per slot
        if (cf.ssh != 0 and @rem(vi.height >> cf.ssh, ratio) != 0) {
            return ctx.fail("\"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).", .{});
        }

        if (!ctx.defined(2)) return ctx.fail("\"planes\" is required.", .{});
        var process: [4]bool = @splat(false);
        const pv = ctx.arg(2);
        if (pv.len() == 0) return ctx.fail("\"planes\" is required.", .{});
        for (0..pv.len()) |k| {
            const e = pv.at(k);
            if (!e.isInt()) return ctx.fail("planes[{d}] must be an integer.", .{k});
            const pl = e.asInt();
            if (pl < 0 or pl >= sf.color_planes) return ctx.fail("\"planes\" contains an out-of-range plane index.", .{});
            if (process[@intCast(pl)]) return ctx.fail("\"planes\" contains a duplicate plane index.", .{});
            process[@intCast(pl)] = true;
        }

        const d: AggData = .{
            .format = sf,
            .stack = .{ .clip = stack.retain() },
            .num_frames = svi.num_frames,
            .radius = radius,
            .process = process,
        };
        const out = F.create(ctx.env, src, d, F.getFrame(getFrame), .{ .mt_mode = .nice }) catch |e| {
            stack.release();
            return e;
        };
        return avs.Create.returnClip(out);
    }
};
