//! BM3D on the CPU (vszipcu's algorithm and arguments; `filters/bm3d.zig`).
//! radius > 0 aggregates internally, so the
//! returned clip is the final one (no stacked clip, no VAggregate step).

const std = @import("std");
const math = std.math;

const filter = @import("../filters/bm3d.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

pub const filter_name = "BM3D";
pub const params = "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_step:int[]:opt;bm_range:int[]:opt;" ++
    "radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;chroma:int:opt;" ++
    "bm_error_s:data[]:opt;transform_2d_s:data[]:opt;transform_1d_s:data[]:opt;";

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
                filter.stage(final, &e.p, srcs[0..e.p.n_planes], refs[0..e.p.n_planes], ring, &sink);
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
                            const f = bc.clampFrame(m, @as(i64, @intCast(z)) - r, d.vi.numFrames);
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
            const lo = bc.clampFrame(n, -span, d.vi.numFrames);
            const hi = bc.clampFrame(n, span, d.vi.numFrames);
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

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const name = filter_name;
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
    if (d.vi.width < 64 or d.vi.height < 64) return fail(map_out, "clip must be at least 64x64.");
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

    const bm_error = perPlaneStr(filter.BmError, map_in, "bm_error_s", .ssd) orelse return fail(map_out, "invalid \"bm_error_s\".");
    const t2d = perPlaneStr(filter.Transform, map_in, "transform_2d_s", .dct) orelse return fail(map_out, "invalid \"transform_2d_s\".");
    const t1d = perPlaneStr(filter.Transform, map_in, "transform_1d_s", .dct) orelse return fail(map_out, "invalid \"transform_1d_s\".");

    const np: usize = @intCast(fmt.numPlanes);
    for (np..3) |i| d.process[i] = false;
    var any = false;
    for (0..np) |i| any = any or d.process[i];
    if (!any) {
        // nothing to process (every sigma 0): pass the clip through, like the
        // other filters (setNode adds its own reference; ours are freed below)
        _ = map_out.setNode("clip", d.node, .Replace);
        return;
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

    const w: u32 = @intCast(d.vi.width);
    const h: u32 = @intCast(d.vi.height);
    if (chroma) {
        if (w < 8 or h < 8) return fail(map_out, "every processed plane must be at least 8x8.");
        d.entries[0] = .{
            .p = base(0, w, h, 3, sigma, d.process, block_step, bm_range, ps_num, ps_range, d.radius, bm_error, t2d, t1d),
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
                .p = base(p, pw, ph, 1, .{ sigma[p], sigma[p], sigma[p] }, .{ true, false, false }, block_step, bm_range, ps_num, ps_range, d.radius, bm_error, t2d, t1d),
                .planes = .{ @intCast(p), 0, 0 },
            };
            d.n_entries += 1;
        }
    }

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
        d.cache = bc.TemporalCache.init(d.radius, d.vi.numFrames, d.process, pw, ph, @intCast(@max(info.numThreads, 1)));
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
