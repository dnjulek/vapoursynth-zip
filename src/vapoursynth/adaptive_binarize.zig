const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "AdaptiveBinarize";

const Data = struct {
    node: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    tab: [768]u8,
};

fn adaptiveBinarizeGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
        vsapi.?.requestFrameFilter.?(n, d.node2, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node, n, frame_ctx, core, vsapi);
        var src2 = zapi.Frame.init(d.node2, n, frame_ctx, core, vsapi);
        defer src.deinit();
        defer src2.deinit();
        const dst = src.newVideoFrame();

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            var srcp = src.getReadSlice(plane);
            var srcp2 = src2.getReadSlice(plane);
            var dstp = dst.getWriteSlice(plane);
            const w, const h, const stride = src.getDimensions(plane);

            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    const z: u32 = @as(u32, srcp[x]) + 255 - @as(u32, srcp2[x]);
                    dstp[x] = d.tab[z];
                }

                srcp = srcp[stride..];
                srcp2 = srcp2[stride..];
                dstp = dstp[stride..];
            }
        }

        dst.setInt("_ColorRange", 0);
        return dst.frame;
    }

    return null;
}

export fn adaptiveBinarizeFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.node2);
    allocator.destroy(d);
}

pub export fn adaptiveBinarizeCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;
    var map = zapi.Map.init(in, out, vsapi);

    d.node, d.vi = map.getNodeVi("clip");
    d.node2, const vi2 = map.getNodeVi("clip2");

    helper.compareNodes(out, d.node, d.node2, d.vi, vi2, filter_name, vsapi) catch return;
    if ((d.vi.format.sampleType != .Integer) or (d.vi.format.bitsPerSample != 8)) {
        map.setError(filter_name ++ ": only 8 bit int format supported.");
        vsapi.?.freeNode.?(d.node);
        vsapi.?.freeNode.?(d.node2);
        return;
    }

    if (d.vi.numFrames != vi2.numFrames) {
        vsapi.?.mapSetError.?(out, filter_name ++ " : clips must have the same length.");
        vsapi.?.freeNode.?(d.node);
        vsapi.?.freeNode.?(d.node2);
        return;
    }

    const c_param = map.getInt(i32, "c") orelse 3;
    for (&d.tab, 0..) |*i, n| {
        i.* = if (@as(i32, @intCast(n)) - 255 <= -c_param) 255 else 0;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .StrictSpatial,
        },
        vs.FilterDependency{
            .source = d.node2,
            .requestPattern = .StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, adaptiveBinarizeGetFrame, adaptiveBinarizeFree, .Parallel, &deps, deps.len, data, core);
}
