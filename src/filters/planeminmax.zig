const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process = @import("process/planeminmax.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneMinMax";

pub const PlaneMinMaxData = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    peak: u16,
    minthr: f32,
    maxthr: f32,
    hist_size: u32,
    planes: [3]bool,
    dt: helper.DataType,
};

export fn planeMinMaxGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *PlaneMinMaxData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
        if (d.node2) |node| {
            vsapi.?.requestFrameFilter.?(n, node, frame_ctx);
        }
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node1, n, frame_ctx, core, vsapi);
        defer src.deinit();

        var ref: ?zapi.Frame = null;
        if (d.node2) |node| {
            ref = zapi.Frame.init(node, n, frame_ctx, core, vsapi);
            defer ref.?.deinit();
        }

        const dst = src.copyFrame();
        const props = dst.getPropertiesRW();

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            if (!(d.planes[plane])) {
                continue;
            }

            const srcp = src.getReadPtr(plane);
            const w, const h, const stride = src.getDimensions(plane);
            var stats: process.Stats = undefined;
            if (ref == null) {
                stats = switch (d.dt) {
                    .U8 => process.minMaxInt(u8, srcp, stride, w, h, d),
                    .U16 => process.minMaxInt(u16, srcp, stride, w, h, d),
                    .F32 => process.minMaxFloat(f32, srcp, stride, w, h, d),
                };
            } else {
                const refp = ref.?.getReadPtr(plane);
                stats = switch (d.dt) {
                    .U8 => process.minMaxIntRef(u8, srcp, refp, stride, w, h, d),
                    .U16 => process.minMaxIntRef(u16, srcp, refp, stride, w, h, d),
                    .F32 => process.minMaxFloatRef(f32, srcp, refp, stride, w, h, d),
                };

                _ = switch (stats) {
                    .i => vsapi.?.mapSetFloat.?(props, "psmDiff", stats.i.diff, .Append),
                    .f => vsapi.?.mapSetFloat.?(props, "psmDiff", stats.f.diff, .Append),
                };
            }

            switch (stats) {
                .i => {
                    _ = vsapi.?.mapSetInt.?(props, "psmMax", stats.i.max, .Append);
                    _ = vsapi.?.mapSetInt.?(props, "psmMin", stats.i.min, .Append);
                },
                .f => {
                    _ = vsapi.?.mapSetFloat.?(props, "psmMax", stats.f.max, .Append);
                    _ = vsapi.?.mapSetFloat.?(props, "psmMin", stats.f.min, .Append);
                },
            }
        }

        return dst.frame;
    }

    return null;
}

export fn planeMinMaxFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *PlaneMinMaxData = @ptrCast(@alignCast(instance_data));

    if (d.node2) |node| {
        vsapi.?.freeNode.?(node);
    }

    vsapi.?.freeNode.?(d.node1);
    allocator.destroy(d);
}

pub export fn planeMinMaxCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: PlaneMinMaxData = undefined;

    var map = zapi.Map.init(in, out, vsapi);
    d.node1, d.vi = map.getNodeVi("clipa");
    d.node2, const vi2 = map.getNodeVi("clipb");
    helper.compareNodes(out, d.node1, d.node2, d.vi, vi2, filter_name, vsapi) catch return;

    d.dt = @enumFromInt(d.vi.format.bytesPerSample);
    var nodes = [_]?*vs.Node{ d.node1, d.node2 };
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hist_size = if (d.dt == .F32) 65536 else math.shl(u32, 1, d.vi.format.bitsPerSample);
    d.peak = @intCast(d.hist_size - 1);
    d.maxthr = getThr(in, out, &nodes, "maxthr", vsapi) catch return;
    d.minthr = getThr(in, out, &nodes, "minthr", vsapi) catch return;

    const data: *PlaneMinMaxData = allocator.create(PlaneMinMaxData) catch unreachable;
    data.* = d;

    var deps1 = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node1,
            .requestPattern = .StrictSpatial,
        },
    };

    var deps_len: c_int = deps1.len;
    var deps: [*]const vs.FilterDependency = &deps1;
    if (d.node2 != null) {
        var deps2 = [_]vs.FilterDependency{
            deps1[0],
            vs.FilterDependency{
                .source = d.node2,
                .requestPattern = if (d.vi.numFrames <= vi2.numFrames) .StrictSpatial else .General,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, planeMinMaxGetFrame, planeMinMaxFree, .Parallel, deps, deps_len, data, core);
}

pub fn getThr(in: ?*const vs.Map, out: ?*vs.Map, nodes: []?*vs.Node, comptime key: []const u8, vsapi: ?*const vs.API) !f32 {
    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        for (nodes) |node| {
            vsapi.?.freeNode.?(node);
        }
    }

    const thr = vsh.mapGetN(f32, in, key.ptr, 0, vsapi) orelse 0;
    if (thr < 0 or thr > 1) {
        err_msg = filter_name ++ ": " ++ key ++ " should be a float between 0.0 and 1.0";
        return error.ValidationError;
    }

    return thr;
}
