const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process = @import("process/planeminmax.zig");

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

pub const PlaneMinMaxData = struct {
    node: *vs.Node,
    node2: ?*vs.Node,
    peak: u16,
    minthr: f32,
    maxthr: f32,
    hist_size: u32,
    planes: [3]bool,
    dt: helper.DataType,
};

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
            if (!(d.planes[@intCast(plane)])) {
                continue;
            }

            const srcp: [*]const u8 = vsapi.?.getReadPtr.?(src, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: usize = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: usize = @intCast(vsapi.?.getFrameWidth.?(src, plane));

            var stats: process.Stats = undefined;
            if (ref == null) {
                stats = switch (d.dt) {
                    .U8 => process.minMaxInt(u8, srcp, stride, w, h, d),
                    .U16 => process.minMaxInt(u16, srcp, stride, w, h, d),
                    .F32 => process.minMaxFloat(f32, srcp, stride, w, h, d),
                };
            } else {
                const refp: [*]const u8 = vsapi.?.getReadPtr.?(ref, plane);
                stats = switch (d.dt) {
                    .U8 => process.minMaxIntRef(u8, srcp, refp, stride, w, h, d),
                    .U16 => process.minMaxIntRef(u16, srcp, refp, stride, w, h, d),
                    .F32 => process.minMaxFloatRef(f32, srcp, refp, stride, w, h, d),
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
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &planes, vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;
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
