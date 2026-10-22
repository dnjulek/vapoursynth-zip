//! BM3D / BM3Dv2 / VAggregate: the CUDA-output BM3D (vszipcu's arguments and
//! output; `filters/bm3d_cuda.zig`). radius > 0 aggregates internally (no
//! stacked clip), so BM3D and BM3Dv2 return the same clip; VAggregate
//! aggregates stacked clips made by the GPU plugins.

const std = @import("std");
const math = std.math;

const filter = @import("../filters/bm3d_cuda.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

pub const BM3D = Entry(false);
pub const BM3Dv2 = Entry(true);

fn Entry(comptime v2: bool) type {
    return struct {
        pub const filter_name = if (v2) "BM3Dv2" else "BM3D";
        pub const params = "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_step:int[]:opt;bm_range:int[]:opt;" ++
            "radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;chroma:int:opt;extractor_exp:int:opt;" ++
            "bm_error_s:data[]:opt;transform_2d_s:data[]:opt;transform_1d_s:data[]:opt;zero_init:int:opt;";

        pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
            createImpl(v2, filter_name, in, out, core, vsapi);
        }
    };
}

const PlaneEntry = struct {
    p: filter.Params,
    /// VapourSynth plane of each entry plane
    planes: [3]u32,
};

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    final: bool = false,
    radius: u32 = 0,
    process: [3]bool = @splat(false),
    entries: [3]PlaneEntry = undefined,
    n_entries: usize = 0,
    rcp: [*]const u8 = undefined,
    cache: bc.TemporalCache = undefined,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *filter.Scratch) = undefined,
};

fn acquireScratch(d: *Data) ?*filter.Scratch {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |sc| return sc;
    const sc = filter.Scratch.create() catch return null;
    d.pool.put(tid, sc) catch {
        sc.destroy();
        return null;
    };
    return sc;
}

fn Spatial(comptime final: bool, comptime name: []const u8) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (final) zapi.requestFrameFilter(n, d.ref_node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            const ref = if (final) zapi.initZFrame(d.ref_node, n) else src;
            defer if (final) ref.deinit();
            const dst = src.newVideoFrame2(d.process);

            const sc = acquireScratch(d) orelse return hz.frameOom(&zapi, name, dst);
            for (d.entries[0..d.n_entries], 0..) |*e, ei| {
                const ring = sc.ringFor(ei, &e.p) catch return hz.frameOom(&zapi, name, dst);
                var srcs: [3]filter.PlaneIn = undefined;
                var refs: [3]filter.PlaneIn = undefined;
                var sink: filter.DivideSink = .{ .dst = @splat(null), .stride = @splat(0) };
                for (0..e.p.n_planes) |i| {
                    const pl = e.planes[i];
                    srcs[i] = .{ .data = src.getReadSlice2(f32, pl).ptr, .stride = src.getStride2(f32, pl) };
                    if (final) refs[i] = .{ .data = ref.getReadSlice2(f32, pl).ptr, .stride = ref.getStride2(f32, pl) };
                    if (e.p.process[i]) {
                        sink.dst[i] = dst.getWriteSlice2(f32, pl).ptr;
                        sink.stride[i] = dst.getStride2(f32, pl);
                    }
                }
                filter.stage(final, &e.p, srcs[0..e.p.n_planes], refs[0..e.p.n_planes], ring, d.rcp, &sink);
            }
            return dst.frame;
        }
    };
}


const max_span = 4 * filter.max_radius + 1;

fn Temporal(comptime final: bool, comptime name: []const u8) type {
    return struct {
        const ZF = ZAPI.ZFrame(*const vs.Frame);

        /// Source frames n - 2r .. n + 2r (clamped): every stage n - r ..
        /// n + r reads within them.
        const Frames = struct {
            lo: i32,
            src: [max_span]ZF = undefined,
            ref: [max_span]ZF = undefined,
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
                            const f = math.clamp(m - r + @as(i32, @intCast(z)), 0, d.vi.numFrames - 1);
                            const k: usize = @intCast(f - self.fr.lo);
                            const sf = self.fr.src[k];
                            srcs[i * tw + z] = .{ .data = sf.getReadSlice2(f32, pl).ptr, .stride = sf.getStride2(f32, pl) };
                            if (final) {
                                const rf = self.fr.ref[k];
                                refs[i * tw + z] = .{ .data = rf.getReadSlice2(f32, pl).ptr, .stride = rf.getStride2(f32, pl) };
                            }
                        }
                    }
                    const sink: FoldSink = .{ .cache = &d.cache, .m = m, .planes = e.planes };
                    const n = e.p.n_planes * tw;
                    filter.stage(final, &e.p, srcs[0..n], refs[0..n], ring, d.rcp, &sink);
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
            dst: ZAPI.ZFrame(*vs.Frame),

            pub fn plane(self: *const Out, p: usize, num: []const f32, den: []const f32) void {
                const w, const h, const stride = self.dst.getDimensions2(f32, p);
                const out = self.dst.getWriteSlice2(f32, p);
                for (0..h) |y| {
                    filter.divideRow(out[y * stride ..][0..w], num[y * w ..][0..w], den[y * w ..][0..w]);
                }
            }
        };

        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const span: i32 = 2 * @as(i32, @intCast(d.radius));
            const lo = @max(n - span, 0);
            const hi = @min(n + span, d.vi.numFrames - 1);
            if (activation_reason == .Initial) {
                var i = lo;
                while (i <= hi) : (i += 1) {
                    zapi.requestFrameFilter(i, d.node);
                    if (final) zapi.requestFrameFilter(i, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            var fr: Frames = .{ .lo = lo };
            defer for (0..fr.count) |i| {
                fr.src[i].deinit();
                if (final) fr.ref[i].deinit();
            };
            var i = lo;
            while (i <= hi) : (i += 1) {
                fr.src[fr.count] = zapi.initZFrame(d.node, i);
                if (final) fr.ref[fr.count] = zapi.initZFrame(d.ref_node, i);
                fr.count += 1;
            }

            const dst = fr.src[@intCast(n - lo)].newVideoFrame2(d.process);
            const sc = acquireScratch(d) orelse return hz.frameOom(&zapi, name, dst);
            const stages: Stages = .{ .d = d, .fr = &fr, .sc = sc };
            const out: Out = .{ .dst = dst };
            d.cache.produce(n, &stages, &out) catch return hz.frameOom(&zapi, name, dst);
            return dst.frame;
        }
    };
}

fn bm3dFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    var it = d.pool.valueIterator();
    while (it.next()) |sc| sc.*.destroy();
    d.pool.deinit();
    if (d.radius > 0) d.cache.deinit();
    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

/// vszipcu's per-plane array rule: index i given, else the previous value
/// (the default for plane 0).
fn perPlaneInt(map_in: anytype, comptime key: [:0]const u8, base: i64) [3]i64 {
    var v: [3]i64 = undefined;
    for (0..3) |i| {
        v[i] = map_in.getValue2(i64, key, i) orelse if (i == 0) base else v[i - 1];
    }
    return v;
}

fn perPlaneStr(comptime T: type, map_in: anytype, comptime key: [:0]const u8, base: T) ?[3]T {
    var v: [3]T = undefined;
    for (0..3) |i| {
        if (map_in.getData(key, @intCast(i))) |s| {
            v[i] = T.parse(s) orelse return null;
        } else v[i] = if (i == 0) base else v[i - 1];
    }
    return v;
}

fn createImpl(comptime v2: bool, comptime name: [:0]const u8, in: ?*const vs.Map, out: ?*vs.Map, core: ?*vs.Core, vsapi: ?*const vs.API) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;
    d.ref_node = map_in.getNode("ref");
    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.ref_node);
    };
    const fail = struct {
        fn f(mo: anytype, comptime msg: []const u8) void {
            mo.setError(name ++ ": " ++ msg);
        }
    }.f;

    const fmt = d.vi.format;
    if (!vapoursynth.vshelper.isConstantVideoFormat(d.vi) or fmt.sampleType != .Float or fmt.bitsPerSample != 32) {
        return fail(map_out, "only constant format 32 bit float input supported.");
    }
    if (d.ref_node) |rn| {
        const rvi = zapi.getVideoInfo(rn);
        const rf = rvi.format;
        if (rf.colorFamily != fmt.colorFamily or rf.sampleType != fmt.sampleType or rf.bitsPerSample != fmt.bitsPerSample or
            rf.subSamplingW != fmt.subSamplingW or rf.subSamplingH != fmt.subSamplingH)
        {
            return fail(map_out, "\"ref\" must be of the same format as \"clip\".");
        }
        if (rvi.width != d.vi.width or rvi.height != d.vi.height) return fail(map_out, "\"ref\" must be of the same dimensions as \"clip\".");
        if (rvi.numFrames != d.vi.numFrames) return fail(map_out, "\"ref\" must be of the same number of frames as \"clip\".");
        d.final = true;
    }

    // sigma: validated as f64, used as f32 like the GPU plugin
    hz.checkArrayLen(map_in, "sigma", 3, name, map_out) catch return;
    var sigma: [3]f32 = undefined;
    for (0..3) |i| {
        if (map_in.getValue2(f64, "sigma", i)) |sv| {
            hz.checkFinite(sv, name, "sigma", i, map_out) catch return;
            if (!(sv >= 0)) return fail(map_out, "\"sigma\" must be non-negative.");
            if (sv > math.floatMax(f32)) return fail(map_out, "\"sigma\" is too large.");
            sigma[i] = @floatCast(sv);
        } else sigma[i] = if (i == 0) 3.0 else sigma[i - 1];
    }
    for (0..3) |i| d.process[i] = !(sigma[i] < math.floatEps(f32));
    const factor = if (d.final) filter.sigma_factor_final else filter.sigma_factor_basic;
    for (&sigma) |*sv| sv.* *= factor;

    inline for (.{ "block_step", "bm_range", "ps_num", "ps_range", "bm_error_s", "transform_2d_s", "transform_1d_s" }) |key| {
        hz.checkArrayLen(map_in, key, 3, name, map_out) catch return;
    }
    const block_step = perPlaneInt(map_in, "block_step", 8);
    const bm_range = perPlaneInt(map_in, "bm_range", 9);
    const ps_num = perPlaneInt(map_in, "ps_num", 2);
    const ps_range = perPlaneInt(map_in, "ps_range", 4);
    for (block_step) |v| if (v <= 0 or v > 8) return fail(map_out, "\"block_step\" must be in range [1, 8].");
    // larger ranges select the same (clamped) windows; the cap keeps the
    // search arithmetic in i32
    for (bm_range) |v| if (v <= 0 or v > 1 << 20) return fail(map_out, "\"bm_range\" must be positive.");
    for (ps_num) |v| if (v <= 0 or v > 8) return fail(map_out, "\"ps_num\" must be in range [1, 8].");
    for (ps_range) |v| if (v <= 0 or v > 1 << 20) return fail(map_out, "\"ps_range\" must be positive.");

    const radius = map_in.getValue(i64, "radius") orelse 0;
    if (radius < 0) return fail(map_out, "\"radius\" must be non-negative.");
    if (radius > filter.max_radius) return fail(map_out, "\"radius\" must be <= 16.");
    d.radius = @intCast(radius);

    const chroma = (map_in.getValue(i64, "chroma") orelse 0) != 0;
    if (chroma and (fmt.colorFamily != .YUV or fmt.subSamplingW != 0 or fmt.subSamplingH != 0)) {
        return fail(map_out, "clip format must be YUV444 when \"chroma\" is true.");
    }
    _ = map_in.getValue(i64, "zero_init"); // stacked-clip option of the GPU plugin; no stacked clip here

    const extractor_exp = map_in.getValue(i64, "extractor_exp") orelse 0;
    if (extractor_exp < -149 or extractor_exp > 127) return fail(map_out, "\"extractor_exp\" must be in range [-149, 127].");
    const extractor: f32 = if (extractor_exp != 0) math.ldexp(@as(f32, 1.0), @intCast(extractor_exp)) else 0.0;

    const bm_error = perPlaneStr(filter.BmError, map_in, "bm_error_s", .ssd) orelse return fail(map_out, "invalid \"bm_error_s\".");
    const t2d = perPlaneStr(filter.Transform, map_in, "transform_2d_s", .dct) orelse return fail(map_out, "invalid \"transform_2d_s\".");
    const t1d = perPlaneStr(filter.Transform, map_in, "transform_1d_s", .dct) orelse return fail(map_out, "invalid \"transform_1d_s\".");

    const np: usize = @intCast(fmt.numPlanes);
    for (np..3) |i| d.process[i] = false;
    var any = false;
    for (0..np) |i| any = any or d.process[i];
    if (!any) {
        if (v2) {
            // BM3Dv2 passes such a clip through
            _ = map_out.setNode("clip", d.node, .Replace);
            return;
        }
        return fail(map_out, "all planes have sigma < FLT_EPSILON (nothing to process).");
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

    const w: u32 = @intCast(d.vi.width);
    const h: u32 = @intCast(d.vi.height);
    if (chroma) {
        if (w < 8 or h < 8) return fail(map_out, "every processed plane must be at least 8x8.");
        d.entries[0] = .{
            .p = base(0, w, h, 3, sigma, d.process, block_step, bm_range, ps_num, ps_range, d.radius, bm_error, t2d, t1d, extractor),
            .planes = .{ 0, 1, 2 },
        };
        d.n_entries = 1;
    } else {
        for (0..np) |p| {
            if (!d.process[p]) continue;
            const ssw: u5 = if (p == 0) 0 else @intCast(fmt.subSamplingW);
            const ssh: u5 = if (p == 0) 0 else @intCast(fmt.subSamplingH);
            const pw = w >> ssw;
            const ph = h >> ssh;
            if (pw < 8 or ph < 8) return fail(map_out, "every processed plane must be at least 8x8.");
            d.entries[d.n_entries] = .{
                .p = base(p, pw, ph, 1, .{ sigma[p], sigma[p], sigma[p] }, .{ true, false, false }, block_step, bm_range, ps_num, ps_range, d.radius, bm_error, t2d, t1d, extractor),
                .planes = .{ @intCast(p), 0, 0 },
            };
            d.n_entries += 1;
        }
    }

    d.rcp = bc.rcpTable() catch return fail(map_out, "out of memory.");

    if (d.radius > 0) {
        var info: vs.CoreInfo = undefined;
        zapi.getCoreInfo(core, &info);
        var pw: [3]u32 = @splat(0);
        var ph: [3]u32 = @splat(0);
        for (0..np) |p| {
            const ssw: u5 = if (p == 0) 0 else @intCast(fmt.subSamplingW);
            const ssh: u5 = if (p == 0) 0 else @intCast(fmt.subSamplingH);
            pw[p] = w >> ssw;
            ph[p] = h >> ssh;
        }
        d.cache = bc.TemporalCache.init(.cuda, d.radius, d.vi.numFrames, d.process, pw, ph, @intCast(@max(info.numThreads, 1)));
    }

    const data = allocator.create(Data) catch return fail(map_out, "out of memory.");
    data.* = d;
    data.pool_lock = .init;
    data.pool = std.AutoHashMap(std.Thread.Id, *filter.Scratch).init(allocator);
    keep = true;

    const gf: vs.FilterGetFrame = if (d.radius == 0)
        (if (d.final) &Spatial(true, name).getFrame else &Spatial(false, name).getFrame)
    else
        (if (d.final) &Temporal(true, name).getFrame else &Temporal(false, name).getFrame);
    const rp: vs.RequestPattern = if (d.radius > 0) .General else .StrictSpatial;
    var deps: [2]vs.FilterDependency = .{
        .{ .source = d.node, .requestPattern = rp },
        .{ .source = d.ref_node, .requestPattern = rp },
    };
    zapi.createVideoFilter(out, name, d.vi, gf, bm3dFree, .Parallel, deps[0..if (d.final) 2 else 1], data);
}

/// Aggregates a stacked clip made by vszipcu / BM3DCUDA `BM3D(radius > 0)`
/// (vszipcu's VAggregate: same checks, same order of summation).
pub const VAggregate = struct {
    pub const filter_name = "VAggregate";
    pub const params = "clip:vnode;src:vnode;planes:int[];";

    const AggData = struct {
        node: ?*vs.Node = null,
        src_node: ?*vs.Node = null,
        src_vi: *const vs.VideoInfo = undefined,
        radius: i32 = 0,
        process: [3]bool = @splat(false),
    };

    fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
        const d: *AggData = @ptrCast(@alignCast(instance_data));
        const zapi = ZAPI.init(vsapi, core, frame_ctx);
        const r = d.radius;
        const nf = d.src_vi.numFrames;
        if (activation_reason == .Initial) {
            var i = @max(n - r, 0);
            while (i <= @min(n + r, nf - 1)) : (i += 1) zapi.requestFrameFilter(i, d.node);
            zapi.requestFrameFilter(n, d.src_node);
            return null;
        }
        if (activation_reason != .AllFramesReady) return null;

        const src = zapi.initZFrame(d.src_node, n);
        defer src.deinit();
        const tw: usize = @intCast(2 * r + 1);
        var stack: [2 * filter.max_radius + 1]ZAPI.ZFrame(*const vs.Frame) = undefined;
        for (0..tw) |i| stack[i] = zapi.initZFrame(d.node, math.clamp(n - r + @as(i32, @intCast(i)), 0, nf - 1));
        defer for (stack[0..tw]) |f| f.deinit();

        const dst = src.newVideoFrame2(d.process);
        for (0..3) |p| {
            if (!d.process[p]) continue;
            const w, const h, const stride = dst.getDimensions2(f32, p);
            const out = dst.getWriteSlice2(f32, p);
            var rows: [2 * filter.max_radius + 1][*]const f32 = undefined;
            var wts: [2 * filter.max_radius + 1][*]const f32 = undefined;
            for (0..h) |y| {
                for (0..tw) |i| {
                    const z = bc.aggZ(@intCast(i), n, nf, r);
                    const sp = stack[i].getReadSlice2(f32, p);
                    const sstride = stack[i].getStride2(f32, p);
                    const base = (@as(usize, @intCast(z)) * 2 * h + y) * sstride;
                    rows[i] = sp[base..].ptr;
                    wts[i] = sp[base + h * sstride ..].ptr;
                }
                aggRow(out[y * stride ..][0..w], rows[0..tw], wts[0..tw]);
            }
        }
        return dst.frame;
    }

    fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
        const d: *AggData = @ptrCast(@alignCast(instance_data));
        const zapi = ZAPI.init(vsapi, core, null);
        zapi.freeNode(d.node);
        zapi.freeNode(d.src_node);
        allocator.destroy(d);
    }

    pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
        const zapi = ZAPI.init(vsapi, core, null);
        const map_in = zapi.initZMap(in);
        const map_out = zapi.initZMap(out);
        var d: AggData = .{};
        const node, const vi = map_in.getNodeVi("clip").?;
        d.node = node;
        d.src_node, d.src_vi = map_in.getNodeVi("src").?;
        var keep = false;
        defer if (!keep) {
            zapi.freeNode(d.node);
            zapi.freeNode(d.src_node);
        };
        const sf = d.src_vi.format;
        if (!vapoursynth.vshelper.isConstantVideoFormat(d.src_vi) or d.src_vi.height <= 0 or sf.sampleType != .Float or sf.bitsPerSample != 32) {
            return map_out.setError("VAggregate: \"src\" must be 32 bit float.");
        }
        const cf = vi.format;
        if (cf.colorFamily != sf.colorFamily or cf.sampleType != sf.sampleType or cf.bitsPerSample != sf.bitsPerSample or
            cf.subSamplingW != sf.subSamplingW or cf.subSamplingH != sf.subSamplingH or vi.numFrames != d.src_vi.numFrames)
        {
            return map_out.setError("VAggregate: \"clip\" must have the same format and frame count as \"src\".");
        }
        const ratio = @divTrunc(vi.height, d.src_vi.height);
        if (ratio < 6 or @rem(ratio - 2, 4) != 0 or @rem(vi.height, d.src_vi.height) != 0 or vi.width != d.src_vi.width) {
            return map_out.setError("VAggregate: \"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).");
        }
        d.radius = @divTrunc(ratio - 2, 4);
        if (d.radius > filter.max_radius) return map_out.setError("VAggregate: radius too large.");
        // stacked chroma planes must still divide evenly per slot
        if (@rem(vi.height >> @intCast(cf.subSamplingH), ratio) != 0 and cf.subSamplingH != 0) {
            return map_out.setError("VAggregate: \"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).");
        }

        const np: i64 = sf.numPlanes;
        const len = map_in.numElements("planes") orelse 0;
        if (len == 0) return map_out.setError("VAggregate: \"planes\" is required.");
        for (0..len) |i| {
            const pl = map_in.getValue2(i64, "planes", i).?;
            if (pl < 0 or pl >= np) return map_out.setError("VAggregate: \"planes\" contains an out-of-range plane index.");
            if (d.process[@intCast(pl)]) return map_out.setError("VAggregate: \"planes\" contains a duplicate plane index.");
            d.process[@intCast(pl)] = true;
        }

        const data = allocator.create(AggData) catch return map_out.setError("VAggregate: out of memory.");
        data.* = d;
        keep = true;
        var deps = [_]vs.FilterDependency{
            .{ .source = d.node, .requestPattern = .General },
            .{ .source = d.src_node, .requestPattern = .StrictSpatial },
        };
        zapi.createVideoFilter(out, "VAggregate", d.src_vi, getFrame, free, .Parallel, &deps, data);
    }
};

/// Ascending-t sums of the stacked numerators and weights, then a correctly
/// rounded divide (the GPU's vaggregate kernel).
fn aggRow(out: []f32, rows: []const [*]const f32, wts: []const [*]const f32) void {
    const V8 = bc.V8;
    var x: usize = 0;
    while (x + 8 <= out.len) : (x += 8) {
        var sv: V8 = @splat(0);
        var sw: V8 = @splat(0);
        for (rows, wts) |rp, wp| {
            sv += @as(V8, rp[x..][0..8].*);
            sw += @as(V8, wp[x..][0..8].*);
        }
        out[x..][0..8].* = sv / sw;
    }
    while (x < out.len) : (x += 1) {
        var sv: f32 = 0;
        var sw: f32 = 0;
        for (rows, wts) |rp, wp| {
            sv += rp[x];
            sw += wp[x];
        }
        out[x] = sv / sw;
    }
}
