const std = @import("std");
const vszip = @import("vszip.zig");
const helper = @import("helper.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const math = std.math;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const pe = vs.MapPropertyError;
const ma = vs.MapAppendMode;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneMinMax";

const PlaneMinMaxData = struct {
    node: *vs.Node,
    node2: ?*vs.Node,
    peak: u16,
    minthr: f32,
    maxthr: f32,
    hist_size: u32,
    process: [3]bool,
    dt: helper.DataType,
};

const StatsFloat = struct {
    max: f32,
    min: f32,
    diff: f64,
};

const StatsInt = struct {
    max: u16,
    min: u16,
    diff: f64,
};

const Stats = union(enum) {
    f: StatsFloat,
    i: StatsInt,
};

fn minMaxInt(comptime T: type, src: [*]const u8, _stride: usize, w: usize, h: usize, d: *PlaneMinMaxData) Stats {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            accum_buf[v] += 1;
        }
        srcp += stride;
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    return .{ .i = .{ .max = retvalmax, .min = retvalmin, .diff = undefined } };
}

fn minMaxFloat(comptime T: type, src: [*]const u8, _stride: usize, w: usize, h: usize, d: *PlaneMinMaxData) Stats {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            accum_buf[math.lossyCast(u16, (v * 65535.0 + 0.5))] += 1;
        }
        srcp += stride;
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    const retvalmaxf = @as(f32, @floatFromInt(retvalmax)) / 65535;
    const retvalminf = @as(f32, @floatFromInt(retvalmin)) / 65535;
    return .{ .f = .{ .max = retvalmaxf, .min = retvalminf, .diff = undefined } };
}

fn minMaxIntRef(comptime T: type, src: [*]const u8, ref: [*]const u8, _stride: usize, w: usize, h: usize, d: *PlaneMinMaxData) Stats {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    var refp: [*]const T = @ptrCast(@alignCast(ref));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const total: f64 = @floatFromInt(w * h);
    var diffacc: u64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[v] += 1;
            diffacc += helper.absDiff(v, j);
        }
        srcp += stride;
        refp += stride;
    }

    const diff: f64 = @as(f64, @floatFromInt(diffacc)) / total / @as(f64, @floatFromInt(d.peak));
    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    return .{ .i = .{ .max = retvalmax, .min = retvalmin, .diff = diff } };
}

fn minMaxFloatRef(comptime T: type, src: [*]const u8, ref: [*]const u8, _stride: usize, w: usize, h: usize, d: *PlaneMinMaxData) Stats {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    var refp: [*]const T = @ptrCast(@alignCast(ref));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[math.lossyCast(u16, (v * 65535.0 + 0.5))] += 1;
            diffacc += helper.absDiff(v, j);
        }
        srcp += stride;
        refp += stride;
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    const retvalmaxf = @as(f32, @floatFromInt(retvalmax)) / 65535;
    const retvalminf = @as(f32, @floatFromInt(retvalmin)) / 65535;
    return .{ .f = .{ .max = retvalmaxf, .min = retvalminf, .diff = (diffacc / total) } };
}

export fn planeMinMaxGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *PlaneMinMaxData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
        if (d.node2) |node| {
            vsapi.?.requestFrameFilter.?(n, node, frame_ctx);
        }
    } else if (activation_reason == ar.AllFramesReady) {
        const src = vsapi.?.getFrameFilter.?(n, d.node, frame_ctx);
        defer vsapi.?.freeFrame.?(src);
        var ref: ?*const vs.Frame = null;
        if (d.node2) |node| {
            ref = vsapi.?.getFrameFilter.?(n, node, frame_ctx);
            defer vsapi.?.freeFrame.?(ref);
        }

        const fi = vsapi.?.getVideoFrameFormat.?(src);
        const dst = vsapi.?.copyFrame.?(src, core).?;
        const props = vsapi.?.getFramePropertiesRW.?(dst);

        var plane: c_int = 0;
        while (plane < fi.numPlanes) : (plane += 1) {
            if (!(d.process[@intCast(plane)])) {
                continue;
            }

            const srcp: [*]const u8 = vsapi.?.getReadPtr.?(src, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: usize = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: usize = @intCast(vsapi.?.getFrameWidth.?(src, plane));

            var stats: Stats = undefined;
            if (ref == null) {
                stats = switch (d.dt) {
                    .U8 => minMaxInt(u8, srcp, stride, w, h, d),
                    .U16 => minMaxInt(u16, srcp, stride, w, h, d),
                    .F32 => minMaxFloat(f32, srcp, stride, w, h, d),
                };
            } else {
                const refp: [*]const u8 = vsapi.?.getReadPtr.?(ref, plane);
                stats = switch (d.dt) {
                    .U8 => minMaxIntRef(u8, srcp, refp, stride, w, h, d),
                    .U16 => minMaxIntRef(u16, srcp, refp, stride, w, h, d),
                    .F32 => minMaxFloatRef(f32, srcp, refp, stride, w, h, d),
                };

                _ = switch (stats) {
                    .i => vsapi.?.mapSetFloat.?(props, "psmDiff", stats.i.diff, ma.Append),
                    .f => vsapi.?.mapSetFloat.?(props, "psmDiff", stats.f.diff, ma.Append),
                };
            }

            switch (stats) {
                .i => {
                    _ = vsapi.?.mapSetInt.?(props, "psmMax", stats.i.max, ma.Append);
                    _ = vsapi.?.mapSetInt.?(props, "psmMin", stats.i.min, ma.Append);
                },
                .f => {
                    _ = vsapi.?.mapSetFloat.?(props, "psmMax", stats.f.max, ma.Append);
                    _ = vsapi.?.mapSetFloat.?(props, "psmMin", stats.f.min, ma.Append);
                },
            }
        }

        return dst;
    }

    return null;
}

export fn planeMinMaxFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *PlaneMinMaxData = @ptrCast(@alignCast(instance_data));

    if (d.node2) |node| {
        vsapi.?.freeNode.?(node);
    }

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn planeMinMaxCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: PlaneMinMaxData = undefined;
    var err: pe = undefined;

    d.node = vsapi.?.mapGetNode.?(in, "clipa", 0, &err).?;
    d.node2 = vsapi.?.mapGetNode.?(in, "clipb", 0, &err);
    helper.compareNodes(out, d.node, d.node2, filter_name, vsapi) catch return;

    const vi = vsapi.?.getVideoInfo.?(d.node);
    d.dt = @enumFromInt(vi.format.bytesPerSample);
    var nodes = [_]?*vs.Node{ d.node, d.node2 };
    var process = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &process, vi.format.numPlanes, filter_name, vsapi) catch return;
    d.process = process;
    d.hist_size = if (d.dt == .F32) 65536 else math.shl(u32, 1, vi.format.bitsPerSample);
    d.peak = @intCast(d.hist_size - 1);
    d.maxthr = getThr(in, out, &nodes, "maxthr", vsapi) catch return;
    d.minthr = getThr(in, out, &nodes, "minthr", vsapi) catch return;

    const data: *PlaneMinMaxData = allocator.create(PlaneMinMaxData) catch unreachable;
    data.* = d;

    var deps1 = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = rp.StrictSpatial,
        },
    };

    var deps_len: c_int = deps1.len;
    var deps: [*]const vs.FilterDependency = &deps1;
    if (d.node2 != null) {
        var deps2 = [_]vs.FilterDependency{
            deps1[0],
            vs.FilterDependency{
                .source = d.node2,
                .requestPattern = if (vi.numFrames <= vsapi.?.getVideoInfo.?(d.node2).numFrames) rp.StrictSpatial else rp.General,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    vsapi.?.createVideoFilter.?(out, filter_name, vi, planeMinMaxGetFrame, planeMinMaxFree, fm.Parallel, deps, deps_len, data, core);
}

pub fn getThr(in: ?*const vs.Map, out: ?*vs.Map, nodes: []?*vs.Node, comptime key: [*]const u8, vsapi: ?*const vs.API) !f32 {
    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        for (nodes) |node| {
            vsapi.?.freeNode.?(node);
        }
    }

    const thr = vsh.mapGetN(f32, in, key, 0, vsapi) orelse 0;
    if (thr < 0 or thr > 1) {
        err_msg = filter_name ++ ": " ++ key ++ " should be a float between 0.0 and 1.0";
        return error.ValidationError;
    }

    return thr;
}
