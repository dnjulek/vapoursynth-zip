const std = @import("std");
const math = std.math;

const filter = @import("../filters/wnnm.zig");
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
    vi_raw: vs.VideoInfo = undefined,
    cfg: Config = .{},

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

                processPlane(BS, &d.cfg, bufs, &.{srcp}, &.{refp}, intermediate.ptr, w, w, h, stride, d.cfg.sigma[plane]) catch return hz.frameOom(&zapi, filter_name, dst);
                filter.aggregation(dstp, intermediate, w, h, stride);
            }

            return dst.frame;
        }

        pub fn getFrameRaw(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const radius: i32 = @intCast(d.cfg.radius);
            const num_frames = 2 * d.cfg.radius + 1;

            if (activation_reason == .Initial) {
                var i: i32 = -radius;
                while (i <= radius) : (i += 1) {
                    const fid = math.clamp(n + i, 0, d.vi.numFrames - 1);
                    zapi.requestFrameFilter(fid, d.node);
                    if (d.ref_node != null) zapi.requestFrameFilter(fid, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            comptime std.debug.assert(max_window >= 2 * max_radius + 1);
            var srcs: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
            var refs: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
            for (0..num_frames) |i| {
                const fid = math.clamp(n - radius + @as(i32, @intCast(i)), 0, d.vi.numFrames - 1);
                srcs[i] = zapi.initZFrame(d.node, fid);
                refs[i] = if (d.ref_node != null) zapi.initZFrame(d.ref_node, fid) else srcs[i].addFrameRef();
            }
            defer for (0..num_frames) |i| {
                srcs[i].deinit();
                refs[i].deinit();
            };

            const dst = zapi.initZFrameFromVi(&d.vi_raw, srcs[d.cfg.radius].frame);

            const bufs = acquireBufs(d) orelse return hz.frameOom(&zapi, filter_name, dst);

            var srcps: [max_window][]const f32 = undefined;
            var refps: [max_window][]const f32 = undefined;

            var plane: u32 = 0;
            while (plane < d.vi.format.numPlanes) : (plane += 1) {
                if (!d.cfg.process[plane]) continue;

                const w, const h, const stride = srcs[d.cfg.radius].getDimensions2(f32, plane);
                for (0..num_frames) |i| {
                    srcps[i] = srcs[i].getReadSlice2(f32, plane);
                    refps[i] = refs[i].getReadSlice2(f32, plane);
                }
                const dstp = dst.getWriteSlice2(f32, plane);
                @memset(dstp[0 .. 2 * num_frames * h * stride], 0.0);

                processPlane(BS, &d.cfg, bufs, srcps[0..num_frames], refps[0..num_frames], dstp.ptr, stride, w, h, stride, d.cfg.sigma[plane]) catch return hz.frameOom(&zapi, filter_name, dst);
            }

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

    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

const VagData = struct {
    node: ?*vs.Node = null,
    src_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    radius: u32 = 0,
    process: [3]bool = undefined,
    max_w: u32 = 0,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, []align(64) f32) = undefined,
};

fn vagAcquireBuffer(d: *VagData) ?[]align(64) f32 {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |b| return b;
    const b: []align(64) f32 = allocator.alignedAlloc(f32, comptime std.mem.Alignment.fromByteUnits(64), 2 * d.max_w) catch return null;
    d.pool.put(tid, b) catch {
        allocator.free(b);
        return null;
    };
    return b;
}

fn vagGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *VagData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);
    const radius: i32 = @intCast(d.radius);
    const num_frames = 2 * d.radius + 1;

    if (activation_reason == .Initial) {
        var i: i32 = -radius;
        while (i <= radius) : (i += 1) {
            zapi.requestFrameFilter(math.clamp(n + i, 0, d.vi.numFrames - 1), d.node);
        }
        zapi.requestFrameFilter(n, d.src_node);
        return null;
    }
    if (activation_reason != .AllFramesReady) return null;

    const src = zapi.initZFrame(d.src_node, n);
    defer src.deinit();

    var raws: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
    for (0..num_frames) |i| {
        const fid = math.clamp(n - radius + @as(i32, @intCast(i)), 0, d.vi.numFrames - 1);
        raws[i] = zapi.initZFrame(d.node, fid);
    }
    defer for (0..num_frames) |i| raws[i].deinit();

    const dst = src.newVideoFrame2(d.process);

    const buffer = vagAcquireBuffer(d) orelse return hz.frameOom(&zapi, filter_name, dst);

    var plane: u32 = 0;
    while (plane < d.vi.format.numPlanes) : (plane += 1) {
        if (!d.process[plane]) continue;

        const w, const h, const stride = src.getDimensions2(f32, plane);
        const dstp = dst.getWriteSlice2(f32, plane);
        const num = buffer[0..w];
        const den = buffer[d.max_w..][0..w];

        var y: u32 = 0;
        while (y < h) : (y += 1) {
            @memset(num, 0.0);
            @memset(den, 0.0);
            for (0..num_frames) |i| {
                const raw = raws[i].getReadSlice2(f32, plane);

                const slab = math.clamp(
                    2 * radius - @as(i32, @intCast(i)),
                    n - d.vi.numFrames + 1 + radius,
                    n + radius,
                );
                const base = (@as(usize, @intCast(slab)) * 2 * h + y) * stride;
                const nrow = raw[base..][0..w];
                const drow = raw[base + h * stride ..][0..w];
                filter.vagAccumulate(num, nrow);
                filter.vagAccumulate(den, drow);
            }
            filter.vagDivide(dstp[y * stride ..][0..w], num, den);
        }
    }

    return dst.frame;
}

fn vagFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *VagData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    var it = d.pool.valueIterator();
    while (it.next()) |b| allocator.free(b.*);
    d.pool.deinit();

    zapi.freeNode(d.node);
    zapi.freeNode(d.src_node);
    allocator.destroy(d);
}

fn selectGetFrame(bs: u32, radius: u32) vs.FilterGetFrame {
    return switch (bs) {
        4 => if (radius == 0) &Wnnm(4).getFrameSpatial else &Wnnm(4).getFrameRaw,
        6 => if (radius == 0) &Wnnm(6).getFrameSpatial else &Wnnm(6).getFrameRaw,
        8 => if (radius == 0) &Wnnm(8).getFrameSpatial else &Wnnm(8).getFrameRaw,
        16 => if (radius == 0) &Wnnm(16).getFrameSpatial else &Wnnm(16).getFrameRaw,
        else => if (radius == 0) &Wnnm(null).getFrameSpatial else &Wnnm(null).getFrameRaw,
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

    data.vi_raw = d.vi.*;
    data.vi_raw.height *= @intCast(2 * (2 * d.cfg.radius + 1));

    var deps_buf: [2]vs.FilterDependency = undefined;
    var n_deps: usize = 1;
    deps_buf[0] = .{ .source = d.node, .requestPattern = .General };
    if (d.ref_node != null) {
        deps_buf[1] = .{ .source = d.ref_node, .requestPattern = .General };
        n_deps = 2;
    }

    const src_node2 = zapi.addNodeRef(d.node);

    const raw_node = zapi.createVideoFilter2("WNNMRaw", &data.vi_raw, selectGetFrame(d.cfg.bs, d.cfg.radius), wnnmFree, .Parallel, deps_buf[0..n_deps], data);
    if (raw_node == null) {
        map_out.setError(filter_name ++ ": failed to create internal WNNMRaw filter.");
        zapi.freeNode(src_node2);
        wnnmFree(data, core, vsapi);
        return;
    }

    const vag: *VagData = allocator.create(VagData) catch unreachable;
    vag.* = .{
        .node = raw_node,
        .src_node = src_node2,
        .vi = data.vi,
        .radius = d.cfg.radius,
        .process = d.cfg.process,
        .max_w = d.cfg.max_iw,
    };
    vag.pool_lock = .init;
    vag.pool = std.AutoHashMap(std.Thread.Id, []align(64) f32).init(allocator);

    const vag_deps = [_]vs.FilterDependency{
        .{ .source = raw_node, .requestPattern = .General },
        .{ .source = src_node2, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "WNNMVAggregate", data.vi, &vagGetFrame, vagFree, .Parallel, &vag_deps, vag);
    if (map_out.getError() != null) vagFree(vag, core, vsapi);
}
