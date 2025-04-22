const std = @import("std");

const helper = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "RFS";

const Data = struct {
    node1: *vs.Node = undefined,
    node2: *vs.Node = undefined,
    replace: []bool = undefined,
};

export fn rfsGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = core;
    _ = frame_data;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, if (d.replace[@intCast(n)]) d.node2 else d.node1, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        return zapi.getFrameFilter(n, if (d.replace[@intCast(n)]) d.node2 else d.node1, frame_ctx);
    }

    return null;
}

export fn rfsFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.free(d.replace);
    allocator.destroy(d);
}

pub export fn rfsCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node1 = map_in.getNode("clipa").?;
    d.node2 = map_in.getNode("clipb").?;
    var vi = zapi.getVideoInfo(d.node1).*;
    const mismatch = map_in.getBool("mismatch") orelse false;
    rfsValidateInput(map_out, d.node1, d.node2, &vi, mismatch, &zapi) catch return;
    d.replace = allocator.alloc(bool, @intCast(vi.numFrames)) catch unreachable;

    const np = vi.format.numPlanes;
    var ne = map_in.numElements("planes") orelse 0;
    var i: u32 = 0;

    if ((ne > 0) and (np > 1)) {
        var process = [3]bool{ false, false, false };
        var nodes = [3]*vs.Node{ d.node1, d.node1, d.node1 };
        i = 0;
        while (i < ne) : (i += 1) {
            const e = map_in.getInt2(i32, "planes", i).?;
            if ((e < 0) or (e >= np)) {
                map_out.setError(filter_name ++ ": plane index out of range.");
                zapi.freeNode(d.node1);
                zapi.freeNode(d.node2);
                return;
            }

            const ue: u32 = @intCast(e);
            process[ue] = true;
            nodes[ue] = d.node2;
        }

        if (!(process[0] and process[1] and process[2])) {
            const pl = [3]i64{ 0, 1, 2 };
            const args = zapi.createZMap();
            _ = args.setNode("clips", nodes[0], .Append);
            _ = args.setNode("clips", nodes[1], .Append);
            _ = args.setNode("clips", nodes[2], .Append);
            args.setIntArray("planes", &pl);
            args.setInt("colorfamily", @intFromEnum(vi.format.colorFamily), .Replace);

            const stdplugin = zapi.getPluginByID2(.Std, core);
            const ret = args.invoke(stdplugin, "ShufflePlanes");
            args.free();
            zapi.freeNode(d.node2);
            d.node2 = ret.getNode("clip").?;
            ret.free();
        }
    }

    @memset(d.replace, false);

    i = 0;
    ne = map_in.numElements("frames") orelse 0;
    while (i < ne) : (i += 1) {
        d.replace[map_in.getInt2(usize, "frames", i).?] = true;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .General },
        .{ .source = d.node2, .requestPattern = .General },
    };

    zapi.createVideoFilter(out, filter_name, &vi, rfsGetFrame, rfsFree, .Parallel, &deps, data, core);
}

const rfsInputError = error{
    Dimensions,
    Format,
    FrameRate,
};

fn rfsValidateInput(out: ZAPI.ZMap(?*vs.Map), node1: *vs.Node, node2: *vs.Node, outvi: *vs.VideoInfo, mismatch: bool, zapi: *const ZAPI) rfsInputError!void {
    const vi2 = zapi.getVideoInfo(node2);
    var err_msg: ?[:0]const u8 = null;

    errdefer {
        out.setError(err_msg.?);
        zapi.freeNode(node1);
        zapi.freeNode(node2);
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
            outvi.format = .{};
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
