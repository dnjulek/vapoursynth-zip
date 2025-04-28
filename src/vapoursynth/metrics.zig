const std = @import("std");
const math = std.math;

const filter_ssim = @import("../filters/metric_ssimulacra2.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Metrics";

const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
};

const Mode = enum(i32) {
    SSIMU2 = 0,
    XPSNR = 1,
};

fn ssimulacra2GetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node1, frame_ctx);
        zapi.requestFrameFilter(n, d.node2, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        const src1 = zapi.initZFrame(d.node1, n, frame_ctx, core);
        const src2 = zapi.initZFrame(d.node2, n, frame_ctx, core);
        defer src1.deinit();
        defer src2.deinit();

        const dst = src1.copyFrame();
        const w, const h, const stride = src1.getDimensions2(f32, 0);

        var srcp1: [3][]const f32 = undefined;
        var srcp2: [3][]const f32 = undefined;
        for (0..3) |i| {
            srcp1[i] = src1.getReadSlice2(f32, i);
            srcp2[i] = src2.getReadSlice2(f32, i);
        }

        const val = filter_ssim.process(srcp1, srcp2, stride, w, h);
        const dst_prop = dst.getPropertiesRW();
        dst_prop.setFloat("_SSIMULACRA2", val, .Replace);
        return dst.frame;
    }
    return null;
}

fn MetricsFree(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.destroy(d);
}

pub fn metricsCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node1, const vi1 = map_in.getNodeVi("reference");
    d.node2, const vi2 = map_in.getNodeVi("distorted");

    if ((vi1.width != vi2.width) or (vi1.height != vi2.height)) {
        map_out.setError(filter_name ++ " : clips must have the same dimensions.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    if (vi1.numFrames != vi2.numFrames) {
        map_out.setError(filter_name ++ " : clips must have the same length.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    const mode = map_in.getInt(i32, "mode") orelse 0;
    if (mode != 0) {
        map_out.setError(filter_name ++ " : only mode=0 is implemented.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    d.node1 = hz.toRGBS(d.node1, core, &zapi);
    d.node2 = hz.toRGBS(d.node2, core, &zapi);
    d.node1 = sRGBtoLinearRGB(d.node1, core, &zapi);
    d.node2 = sRGBtoLinearRGB(d.node2, core, &zapi);

    const vi_out = zapi.getVideoInfo(d.node1);
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, vi_out, ssimulacra2GetFrame, MetricsFree, .Parallel, &deps, data, core);
}

pub fn sRGBtoLinearRGB(node: ?*vs.Node, core: ?*vs.Core, zapi: *const ZAPI) ?*vs.Node {
    var in = node;
    const frame = zapi.getFrame(0, node, null, 0);
    defer zapi.freeFrame(frame);

    const map_in = zapi.initZMap(zapi.getFramePropertiesRO(frame));
    const transfer_in = map_in.getTransfer();
    if (transfer_in == .LINEAR) {
        return in;
    }

    const reszplugin = zapi.getPluginByID2(.Resize, core);
    const args = zapi.createZMap();

    _ = args.consumeNode("clip", in, .Replace);
    args.setData("prop", "_Transfer", .Utf8, .Replace);
    args.setInt("intval", @intFromEnum(vsc.TransferCharacteristics.IEC_61966_2_1), .Replace);
    const stdplugin = zapi.getPluginByID2(.Std, core);
    var ret = args.invoke(stdplugin, "SetFrameProp");
    in = ret.getNode("clip");
    ret.free();
    args.clear();

    _ = args.consumeNode("clip", in, .Replace);
    args.setInt("transfer", @intFromEnum(vsc.TransferCharacteristics.LINEAR), .Replace);
    ret = args.invoke(reszplugin, "Bicubic");
    const out = ret.getNode("clip");
    ret.free();
    args.free();
    return out;
}
