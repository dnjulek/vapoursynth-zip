const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process = @import("process/planeaverage.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const math = std.math;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const pe = vs.MapPropertyError;
const ma = vs.MapAppendMode;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneAverage";

const PlaneAverageData = struct {
    node: *vs.Node,
    node2: ?*vs.Node,
    exclude: process.Exclude,
    dt: helper.DataType,
    peak: f32,
    planes: [3]bool,
};

export fn planeAverageGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *PlaneAverageData = @ptrCast(@alignCast(instance_data));

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
            if (!(d.planes[@intCast(plane)])) {
                continue;
            }

            const srcp: [*]const u8 = vsapi.?.getReadPtr.?(src, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: usize = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: usize = @intCast(vsapi.?.getFrameWidth.?(src, plane));
            var avg: f64 = undefined;

            if (ref == null) {
                avg = switch (d.dt) {
                    .U8 => process.average(u8, srcp, stride, w, h, d.exclude, d.peak),
                    .U16 => process.average(u16, srcp, stride, w, h, d.exclude, d.peak),
                    .F32 => process.average(f32, srcp, stride, w, h, d.exclude, d.peak),
                };
            } else {
                const refp: [*]const u8 = vsapi.?.getReadPtr.?(ref, plane);
                const stats = switch (d.dt) {
                    .U8 => process.averageRef(u8, srcp, refp, stride, w, h, d.exclude, d.peak),
                    .U16 => process.averageRef(u16, srcp, refp, stride, w, h, d.exclude, d.peak),
                    .F32 => process.averageRef(f32, srcp, refp, stride, w, h, d.exclude, d.peak),
                };
                _ = vsapi.?.mapSetFloat.?(props, "psmDiff", stats.diff, ma.Append);
                avg = stats.avg;
            }
            _ = vsapi.?.mapSetFloat.?(props, "psmAvg", avg, ma.Append);
        }

        return dst;
    }

    return null;
}

export fn planeAverageFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *PlaneAverageData = @ptrCast(@alignCast(instance_data));
    switch (d.exclude) {
        .i => allocator.free(d.exclude.i),
        .f => allocator.free(d.exclude.f),
    }

    if (d.node2) |node| {
        vsapi.?.freeNode.?(node);
    }

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn planeAverageCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: PlaneAverageData = undefined;
    var err: pe = undefined;

    d.node = vsapi.?.mapGetNode.?(in, "clipa", 0, &err).?;
    d.node2 = vsapi.?.mapGetNode.?(in, "clipb", 0, &err);
    helper.compareNodes(out, d.node, d.node2, filter_name, vsapi) catch return;

    const vi = vsapi.?.getVideoInfo.?(d.node);
    d.dt = @enumFromInt(vi.format.bytesPerSample);
    d.peak = @floatFromInt(math.shl(i32, 1, vi.format.bitsPerSample) - 1);
    var nodes = [_]?*vs.Node{ d.node, d.node2 };
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &planes, vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    const ne: usize = @intCast(vsapi.?.mapNumElements.?(in, "exclude"));
    const exclude_in = vsapi.?.mapGetIntArray.?(in, "exclude", &err);

    if (d.dt == .F32) {
        const buff = allocator.alloc(f32, ne) catch unreachable;
        for (0..ne) |i| {
            buff[i] = @floatFromInt(exclude_in[i]);
        }

        d.exclude = process.Exclude{ .f = buff };
    } else {
        const buff = allocator.alloc(i32, ne) catch unreachable;
        for (0..ne) |i| {
            buff[i] = math.lossyCast(i32, exclude_in[i]);
        }

        d.exclude = process.Exclude{ .i = buff };
    }

    const data: *PlaneAverageData = allocator.create(PlaneAverageData) catch unreachable;
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

    vsapi.?.createVideoFilter.?(out, filter_name, vi, planeAverageGetFrame, planeAverageFree, fm.Parallel, deps, deps_len, data, core);
}
