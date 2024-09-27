const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const ssimulacra2 = @import("../filters/metric_ssimulacra2.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "Metrics";

const Data = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
};

const Mode = enum(i32) {
    SSIMU2 = 0,
    XPSNR = 1,
};

fn ssimulacra2GetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
        vsapi.?.requestFrameFilter.?(n, d.node2, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src1 = zapi.Frame.init(d.node1, n, frame_ctx, core, vsapi);
        var src2 = zapi.Frame.init(d.node2, n, frame_ctx, core, vsapi);
        defer src1.deinit();
        defer src2.deinit();

        const dst = src1.copyFrame();
        const props = dst.getPropertiesRW();
        const w, const h, const stride = src1.getDimensions2(f32, 0);

        var srcp1: [3][]const f32 = undefined;
        var srcp2: [3][]const f32 = undefined;
        for (0..3) |i| {
            srcp1[i] = src1.getReadSlice2(f32, i);
            srcp2[i] = src2.getReadSlice2(f32, i);
        }

        const val = ssimulacra2.process(srcp1, srcp2, stride, w, h);
        _ = vsapi.?.mapSetFloat.?(props, "_SSIMULACRA2", val, .Replace);
        return dst.frame;
    }
    return null;
}

export fn MetricsFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    vsapi.?.freeNode.?(d.node1);
    vsapi.?.freeNode.?(d.node2);
    allocator.destroy(d);
}

pub export fn MetricsCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;

    var map = zapi.Map.init(in, out, vsapi);
    d.node1, const vi1 = map.getNodeVi("reference");
    d.node2, const vi2 = map.getNodeVi("distorted");

    if ((vi1.width != vi2.width) or (vi1.height != vi2.height)) {
        vsapi.?.mapSetError.?(out, filter_name ++ " : clips must have the same dimensions.");
        vsapi.?.freeNode.?(d.node1);
        vsapi.?.freeNode.?(d.node2);
        return;
    }

    if (vi1.numFrames != vi2.numFrames) {
        vsapi.?.mapSetError.?(out, filter_name ++ " : clips must have the same length.");
        vsapi.?.freeNode.?(d.node1);
        vsapi.?.freeNode.?(d.node2);
        return;
    }

    const mode = map.getInt(i32, "mode") orelse 0;
    if (mode != 0) {
        vsapi.?.mapSetError.?(out, filter_name ++ " : only mode=0 is implemented.");
        vsapi.?.freeNode.?(d.node1);
        vsapi.?.freeNode.?(d.node2);
        return;
    }

    d.node1 = helper.toRGBS(d.node1, core, vsapi);
    d.node2 = helper.toRGBS(d.node2, core, vsapi);
    d.node1 = sRGBtoLinearRGB(d.node1, core, vsapi);
    d.node2 = sRGBtoLinearRGB(d.node2, core, vsapi);

    const vi_out = vsapi.?.getVideoInfo.?(d.node1);
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node1,
            .requestPattern = .StrictSpatial,
        },
        vs.FilterDependency{
            .source = d.node2,
            .requestPattern = .StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, vi_out, ssimulacra2GetFrame, MetricsFree, .Parallel, &deps, deps.len, data, core);
}

pub fn sRGBtoLinearRGB(node: ?*vs.Node, core: ?*vs.Core, vsapi: ?*const vs.API) ?*vs.Node {
    var in = node;
    var err: vs.MapPropertyError = undefined;
    const frame = vsapi.?.getFrame.?(0, node, null, 0);
    defer vsapi.?.freeFrame.?(frame);
    const transfer_in = vsapi.?.mapGetInt.?(vsapi.?.getFramePropertiesRO.?(frame), "_Transfer", 0, &err);
    const reszplugin = vsapi.?.getPluginByID.?(vsh.RESIZE_PLUGIN_ID, core);

    const args = vsapi.?.createMap.?();
    var ret: ?*vs.Map = null;

    if (transfer_in != 8) {
        _ = vsapi.?.mapConsumeNode.?(args, "clip", in, .Replace);
        _ = vsapi.?.mapSetData.?(args, "prop", "_Transfer", -1, .Utf8, .Replace);
        _ = vsapi.?.mapSetInt.?(args, "intval", 13, .Replace);
        const stdplugin = vsapi.?.getPluginByID.?(vsh.STD_PLUGIN_ID, core);
        ret = vsapi.?.invoke.?(stdplugin, "SetFrameProp", args);
        in = vsapi.?.mapGetNode.?(ret, "clip", 0, null);
        vsapi.?.freeMap.?(ret);
        vsapi.?.clearMap.?(args);
    }

    _ = vsapi.?.mapConsumeNode.?(args, "clip", in, .Replace);
    _ = vsapi.?.mapSetInt.?(args, "transfer", 8, .Replace);
    ret = vsapi.?.invoke.?(reszplugin, "Bicubic", args);
    const out = vsapi.?.mapGetNode.?(ret, "clip", 0, null);
    vsapi.?.freeMap.?(ret);
    vsapi.?.freeMap.?(args);

    return out;
}
