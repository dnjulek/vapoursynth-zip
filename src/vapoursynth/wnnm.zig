const std = @import("std");
const math = std.math;

const filter = @import("../filters/wnnm.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "WNNM";
pub const params = "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;" ++
    "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;" ++
    "residual:int:opt;adaptive_aggregation:int:opt;rclip:vnode:opt;";

const max_radius = filter.max_radius;
const max_window = filter.max_window;
const Config = filter.Config;
const Bufs = filter.Bufs;
const processPlane = filter.processPlane;

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    cfg: Config = .{},
    /// radius > 0: in-filter temporal aggregation (BM3D's fused scheme)
    cache: bc.TemporalCache = undefined,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *Bufs) = undefined,
};

fn acquireBufs(d: *Data) ?*Bufs {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |b| return b;
    const b = Bufs.create(&d.cfg) catch return null;
    d.pool.put(tid, b) catch {
        b.destroy();
        return null;
    };
    return b;
}

fn Wnnm(comptime BS: ?u32) type {
    return struct {
        pub fn getFrameSpatial(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (d.ref_node != null) zapi.requestFrameFilter(n, d.ref_node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            const ref = if (d.ref_node != null) zapi.initZFrame(d.ref_node, n) else src.addFrameRef();
            defer ref.deinit();
            const dst = src.newVideoFrame2(d.cfg.process);

            const bufs = acquireBufs(d) orelse return hz.frameOom(&zapi, filter_name, dst);

            var plane: u32 = 0;
            while (plane < d.vi.format.numPlanes) : (plane += 1) {
                if (!d.cfg.process[plane]) continue;

                const w, const h, const stride = src.getDimensions2(f32, plane);
                const srcp = src.getReadSlice2(f32, plane);
                const refp = ref.getReadSlice2(f32, plane);
                const dstp = dst.getWriteSlice2(f32, plane);

                const intermediate = bufs.intermediate[0 .. 2 * h * w];
                @memset(intermediate, 0.0);

                processPlane(BS, &d.cfg, bufs, &.{srcp}, &.{refp}, intermediate.ptr, w, h, w, h, stride, d.cfg.sigma[plane], {}) catch return hz.frameOom(&zapi, filter_name, dst);
                filter.aggregation(dstp, intermediate, w, h, stride);
            }

            return dst.frame;
        }

        /// Source and reference frames n - 2r .. n + 2r (clamped): every
        /// stage m = n - r .. n + r reads frames m - r .. m + r.
        const Frames = struct {
            lo: i32,
            count: usize = 0,
            src: [2 * max_window]ZAPI.ZFrame(*const vs.Frame) = undefined,
            ref: [2 * max_window]ZAPI.ZFrame(*const vs.Frame) = undefined,
        };

        /// One WNNM pass centered on frame m (VAggregate's "stage" m): the
        /// window frames' accumulators stream through a per-thread row ring
        /// into the temporal cache instead of a stacked raw frame of
        /// 2 * (2r + 1) planes per plane.
        const Stages = struct {
            d: *Data,
            fr: *const Frames,
            bufs: *Bufs,

            pub fn compute(self: *const Stages, m: i32) error{OutOfMemory}!void {
                const d = self.d;
                const r: i32 = @intCast(d.cfg.radius);
                const nf: usize = 2 * d.cfg.radius + 1;
                var srcps: [max_window][]const f32 = undefined;
                var refps: [max_window][]const f32 = undefined;
                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!d.cfg.process[plane]) continue;
                    const any = self.fr.src[0];
                    const w, const h, const stride = any.getDimensions2(f32, plane);
                    for (0..nf) |z| {
                        const f = bc.clampFrame(m, @as(i64, @intCast(z)) - r, d.vi.numFrames);
                        const k: usize = @intCast(f - self.fr.lo);
                        srcps[z] = self.fr.src[k].getReadSlice2(f32, plane);
                        refps[z] = self.fr.ref[k].getReadSlice2(f32, plane);
                    }
                    const rows = filter.ringRows(&d.cfg, h);
                    const ring = try self.bufs.ensureRing(nf * 2 * @as(usize, filter.ringRows(&d.cfg, d.cfg.max_ih)) * d.cfg.max_iw);
                    const sink: FoldSink = .{ .cache = &d.cache, .m = m, .plane = plane };
                    processPlane(BS, &d.cfg, self.bufs, srcps[0..nf], refps[0..nf], ring.ptr, w, rows, w, h, stride, d.cfg.sigma[plane], &sink) catch |e| {
                        @memset(ring, 0); // keep the all-zero invariant
                        return e;
                    };
                }
            }
        };

        const FoldSink = struct {
            cache: *bc.TemporalCache,
            m: i32,
            plane: u32,

            pub fn rows(self: *const FoldSink, t: u32, y: u32, num: []const f32, den: []const f32) void {
                self.cache.fold(self.m, t, self.plane, y, num, den);
            }
        };

        const Out = struct {
            dst: ZAPI.ZFrame(*vs.Frame),

            pub fn plane(self: *const Out, p: usize, num: []const f32, den: []const f32) void {
                const w, const h, const stride = self.dst.getDimensions2(f32, p);
                const out = self.dst.getWriteSlice2(f32, p);
                for (0..h) |y| {
                    filter.vagDivide(out[y * stride ..][0..w], num[y * w ..][0..w], den[y * w ..][0..w]);
                }
            }
        };

        /// radius > 0. The cache sums the stages' window slices per output
        /// frame in the order of the stacked clip + VAggregate it replaces
        /// (bm3d_common.contribs == the old slab choice), so the output is
        /// the same float sum.
        pub fn getFrameTemporal(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const span: i32 = 2 * @as(i32, @intCast(d.cfg.radius));
            const lo = bc.clampFrame(n, -span, d.vi.numFrames);
            const hi = bc.clampFrame(n, span, d.vi.numFrames);

            if (activation_reason == .Initial) {
                var i = lo;
                while (i <= hi) : (i += 1) {
                    zapi.requestFrameFilter(i, d.node);
                    if (d.ref_node != null) zapi.requestFrameFilter(i, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            var fr: Frames = .{ .lo = lo };
            defer for (0..fr.count) |i| {
                fr.src[i].deinit();
                fr.ref[i].deinit();
            };
            var i = lo;
            while (i <= hi) : (i += 1) {
                fr.src[fr.count] = zapi.initZFrame(d.node, i);
                fr.ref[fr.count] = if (d.ref_node != null) zapi.initZFrame(d.ref_node, i) else fr.src[fr.count].addFrameRef();
                fr.count += 1;
            }

            const dst = fr.src[@intCast(n - lo)].newVideoFrame2(d.cfg.process);
            const bufs = acquireBufs(d) orelse return hz.frameOom(&zapi, filter_name, dst);
            const stages: Stages = .{ .d = d, .fr = &fr, .bufs = bufs };
            const out: Out = .{ .dst = dst };
            d.cache.produce(n, &stages, &out) catch return hz.frameOom(&zapi, filter_name, dst);
            return dst.frame;
        }
    };
}

fn wnnmFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    var it = d.pool.valueIterator();
    while (it.next()) |b| Bufs.destroy(b.*);
    d.pool.deinit();
    if (d.cfg.radius > 0) d.cache.deinit();

    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

fn selectGetFrame(bs: u32, radius: u32) vs.FilterGetFrame {
    return switch (bs) {
        4 => if (radius == 0) &Wnnm(4).getFrameSpatial else &Wnnm(4).getFrameTemporal,
        6 => if (radius == 0) &Wnnm(6).getFrameSpatial else &Wnnm(6).getFrameTemporal,
        8 => if (radius == 0) &Wnnm(8).getFrameSpatial else &Wnnm(8).getFrameTemporal,
        16 => if (radius == 0) &Wnnm(16).getFrameSpatial else &Wnnm(16).getFrameTemporal,
        else => if (radius == 0) &Wnnm(null).getFrameSpatial else &Wnnm(null).getFrameTemporal,
    };
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;

    if (!vapoursynth.vshelper.isConstantVideoFormat(d.vi) or d.vi.format.sampleType != .Float or d.vi.format.bitsPerSample != 32) {
        map_out.setError(filter_name ++ ": only constant format 32 bit float input supported.");
        zapi.freeNode(d.node);
        return;
    }

    // Shared parser: <= 3 values (last repeats, default 3.0), finite and
    // bounded to f32 before narrowing. The sign keeps its own message.
    const fmax = math.floatMax(f32);
    const raw_sigma = hz.getArray(f32, 3.0, -fmax, fmax, "sigma", filter_name, map_in, map_out, &.{d.node}, &zapi) catch return;

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);
    var plane_w: [3]u32 = @splat(0);
    var plane_h: [3]u32 = @splat(0);
    {
        const ssw: u5 = @intCast(d.vi.format.subSamplingW);
        const ssh: u5 = @intCast(d.vi.format.subSamplingH);
        for (0..num_planes) |i| {
            plane_w[i] = if (i == 0) @intCast(d.vi.width) else @as(u32, @intCast(d.vi.width)) >> ssw;
            plane_h[i] = if (i == 0) @intCast(d.vi.height) else @as(u32, @intCast(d.vi.height)) >> ssh;
        }
    }

    const cfg = Config.init(.{
        .sigma = raw_sigma,
        .block_size = map_in.getValue(i32, "block_size") orelse 8,
        .block_step = if (map_in.getValue(i32, "block_step")) |v| v else null,
        .group_size = map_in.getValue(i32, "group_size") orelse 8,
        .bm_range = map_in.getValue(i32, "bm_range") orelse 7,
        .radius = map_in.getValue(i32, "radius") orelse 0,
        .ps_num = map_in.getValue(i32, "ps_num") orelse 2,
        .ps_range = map_in.getValue(i32, "ps_range") orelse 4,
        .residual = map_in.getBool("residual") orelse false,
        .adaptive = map_in.getBool("adaptive_aggregation") orelse true,
        .num_planes = num_planes,
        .plane_w = plane_w,
        .plane_h = plane_h,
    }) catch |e| {
        hz.setArgError(map_out, "{s}: {s}", .{ filter_name, Config.message(e) });
        zapi.freeNode(d.node);
        return;
    };
    d.cfg = cfg orelse {
        _ = map_out.setNode("clip", d.node, .Replace);
        zapi.freeNode(d.node);
        return;
    };

    d.ref_node = map_in.getNode("rclip");
    if (d.ref_node != null) {
        const ref_vi = zapi.getVideoInfo(d.ref_node);
        if (!vapoursynth.vshelper.isSameVideoInfo(d.vi, ref_vi) or ref_vi.numFrames != d.vi.numFrames) {
            map_out.setError(filter_name ++ ": \"rclip\" must be of the same format, dimensions and number of frames as \"clip\".");
            zapi.freeNode(d.node);
            zapi.freeNode(d.ref_node);
            return;
        }
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    data.pool_lock = .init;
    data.pool = std.AutoHashMap(std.Thread.Id, *Bufs).init(allocator);

    if (d.cfg.radius == 0) {
        var deps_buf: [2]vs.FilterDependency = undefined;
        var n_deps: usize = 1;
        deps_buf[0] = .{ .source = d.node, .requestPattern = .StrictSpatial };
        if (d.ref_node != null) {
            deps_buf[1] = .{ .source = d.ref_node, .requestPattern = .StrictSpatial };
            n_deps = 2;
        }
        zapi.createVideoFilter(out, filter_name, d.vi, selectGetFrame(d.cfg.bs, 0), wnnmFree, .Parallel, deps_buf[0..n_deps], data);
        if (map_out.getError() != null) wnnmFree(data, core, vsapi);
        return;
    }

    var info: vs.CoreInfo = undefined;
    zapi.getCoreInfo(core, &info);
    // Config.process flags planes past numPlanes too (sigma repeats)
    var cache_planes: [3]bool = @splat(false);
    for (0..num_planes) |p| cache_planes[p] = d.cfg.process[p];
    data.cache = bc.TemporalCache.init(d.cfg.radius, d.vi.numFrames, cache_planes, plane_w, plane_h, @intCast(@max(info.numThreads, 1)));

    var deps_buf: [2]vs.FilterDependency = undefined;
    var n_deps: usize = 1;
    deps_buf[0] = .{ .source = d.node, .requestPattern = .General };
    if (d.ref_node != null) {
        deps_buf[1] = .{ .source = d.ref_node, .requestPattern = .General };
        n_deps = 2;
    }
    zapi.createVideoFilter(out, filter_name, d.vi, selectGetFrame(d.cfg.bs, d.cfg.radius), wnnmFree, .Parallel, deps_buf[0..n_deps], data);
    if (map_out.getError() != null) wnnmFree(data, core, vsapi);
}
