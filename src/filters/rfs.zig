const std = @import("std");
const vszip = @import("../vszip.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const st = vs.SampleType;
const cf = vs.ColorFamily;
const ma = vs.MapAppendMode;
const pe = vs.MapPropertyError;

const allocator = std.heap.c_allocator;
pub const filter_name = "RFS";

const RFSData = struct {
    node1: *vs.Node,
    node2: *vs.Node,
    replace: []bool,
};

export fn rfsGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = core;
    _ = frame_data;
    const d: *RFSData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, if (d.replace[@intCast(n)]) d.node2 else d.node1, frame_ctx);
    } else if (activation_reason == ar.AllFramesReady) {
        return vsapi.?.getFrameFilter.?(n, if (d.replace[@intCast(n)]) d.node2 else d.node1, frame_ctx);
    }

    return null;
}

export fn rfsFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *RFSData = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node1);
    vsapi.?.freeNode.?(d.node2);
    allocator.free(d.replace);
    allocator.destroy(d);
}

pub export fn rfsCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: RFSData = undefined;
    var node_err: pe = undefined;

    d.node1 = vsapi.?.mapGetNode.?(in, "clipa", 0, &node_err).?;
    d.node2 = vsapi.?.mapGetNode.?(in, "clipb", 0, &node_err).?;
    var vi = vsapi.?.getVideoInfo.?(d.node1).*;
    const mismatch = vsh.mapGetN(bool, in, "mismatch", 0, vsapi) orelse false;
    rfsValidateInput(out.?, d.node1, d.node2, &vi, mismatch, vsapi.?) catch return;
    d.replace = allocator.alloc(bool, @intCast(vi.numFrames)) catch unreachable;

    const np = vi.format.numPlanes;
    const ne = vsapi.?.mapNumElements.?(in, "planes");
    var ci: c_int = 0;

    if ((ne > 0) and (np > 1)) {
        var process = [3]bool{ false, false, false };
        var nodes = [3]*vs.Node{ d.node1, d.node1, d.node1 };
        ci = 0;
        while (ci < ne) : (ci += 1) {
            const e: i32 = vsh.mapGetN(i32, in, "planes", ci, vsapi).?;
            if ((e < 0) or (e >= np)) {
                vsapi.?.mapSetError.?(out, filter_name ++ ": plane index out of range.");
                vsapi.?.freeNode.?(d.node1);
                vsapi.?.freeNode.?(d.node2);
                return;
            }

            const ue: u32 = @intCast(e);
            process[ue] = true;
            nodes[ue] = d.node2;
        }

        if (!(process[0] and process[1] and process[2])) {
            const pl = [3]i64{ 0, 1, 2 };
            const args = vsapi.?.createMap.?();
            _ = vsapi.?.mapSetNode.?(args, "clips", nodes[0], ma.Append);
            _ = vsapi.?.mapSetNode.?(args, "clips", nodes[1], ma.Append);
            _ = vsapi.?.mapSetNode.?(args, "clips", nodes[2], ma.Append);
            _ = vsapi.?.mapSetIntArray.?(args, "planes", &pl, 3);
            _ = vsapi.?.mapSetInt.?(args, "colorfamily", @intFromEnum(vi.format.colorFamily), ma.Replace);

            const stdplugin = vsapi.?.getPluginByID.?(vsh.STD_PLUGIN_ID, core);
            const ret = vsapi.?.invoke.?(stdplugin, "ShufflePlanes", args);
            vsapi.?.freeMap.?(args);
            vsapi.?.freeNode.?(d.node2);
            d.node2 = vsapi.?.mapGetNode.?(ret, "clip", 0, &node_err).?;
            vsapi.?.freeMap.?(ret);
        }
    }

    for (d.replace) |*b| {
        b.* = false;
    }

    ci = 0;
    while (ci < vsapi.?.mapNumElements.?(in, "frames")) : (ci += 1) {
        d.replace[vsh.mapGetN(usize, in, "frames", ci, vsapi).?] = true;
    }

    const data: *RFSData = allocator.create(RFSData) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node1,
            .requestPattern = rp.General,
        },
        vs.FilterDependency{
            .source = d.node2,
            .requestPattern = rp.General,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, &vi, rfsGetFrame, rfsFree, fm.Parallel, &deps, deps.len, data, core);
}

const rfsInputError = error{
    Dimensions,
    Format,
    FrameRate,
};

fn rfsValidateInput(out: *vs.Map, node1: *vs.Node, node2: *vs.Node, outvi: *vs.VideoInfo, mismatch: bool, vsapi: *const vs.API) rfsInputError!void {
    const vi2 = vsapi.getVideoInfo.?(node2);
    var err_msg: ?[*]const u8 = null;

    errdefer {
        vsapi.mapSetError.?(out, err_msg.?);
        vsapi.freeNode.?(node1);
        vsapi.freeNode.?(node2);
    }

    if ((outvi.width != vi2.width) or (outvi.height != vi2.height)) {
        if (mismatch) {
            outvi.width = 0;
            outvi.height = 0;
        } else {
            err_msg = filter_name ++ ": Clip dimensions don't match, enable mismatch if you want variable format.";
            return rfsInputError.Dimensions;
        }
    }

    if (!vsh.isSameVideoFormat(&outvi.format, &vi2.format)) {
        if (mismatch) {
            outvi.format.colorFamily = cf.Undefined;
        } else {
            err_msg = filter_name ++ ": Clip formats don't match, enable mismatch if you want variable format.";
            return rfsInputError.Format;
        }
    }

    if ((outvi.fpsDen != vi2.fpsDen) or (outvi.fpsNum != vi2.fpsNum)) {
        if (mismatch) {
            outvi.fpsDen = 0;
            outvi.fpsNum = 0;
        } else {
            err_msg = filter_name ++ ": Clip frame rates don't match, enable mismatch if you want variable format.";
            return rfsInputError.FrameRate;
        }
    }
}
