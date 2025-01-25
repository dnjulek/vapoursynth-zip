const std = @import("std");
const math = std.math;

const filter = @import("../filters/clahe.zig");
const helper = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;

const allocator = std.heap.c_allocator;
pub const filter_name = "CLAHE";

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    limit: u32,
    tiles: [2]u32,
};

fn CLAHE(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.ZFrame.init(d.node, n, frame_ctx, core, vsapi);
                defer src.deinit();
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const width, const height, const stride = src.getDimensions2(T, plane);
                    filter.applyCLAHE(T, srcp, dstp, stride, width, height, d.limit, &d.tiles);
                }

                const dst_prop = dst.getProperties();
                dst_prop.setInt("_ColorRange", 0, .Replace);
                return dst.frame;
            }

            return null;
        }
    };
}

export fn claheFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn claheCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;
    const map_in = zapi.ZMap.init(in, vsapi);
    const map_out = zapi.ZMap.init(out, vsapi);

    d.node, d.vi = map_in.getNodeVi("clip");

    if ((d.vi.format.sampleType != .Integer)) {
        map_out.setError(filter_name ++ ": only 8-16 bit int formats supported.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    d.limit = map_in.getInt(u32, "limit") orelse 7;
    const in_arr = map_in.getIntArray("tiles");
    const df_arr = [2]i64{ 3, 3 };
    const tiles_arr = if ((in_arr == null) or (in_arr.?.len == 0)) &df_arr else in_arr.?;
    d.tiles[0] = @intCast(tiles_arr[0]);
    switch (tiles_arr.len) {
        1 => {
            d.tiles[1] = @intCast(tiles_arr[0]);
        },
        2 => {
            d.tiles[1] = @intCast(tiles_arr[1]);
        },
        else => {
            map_out.setError(filter_name ++ " : tiles array can't have more than 2 values.");
            vsapi.?.freeNode.?(d.node);
            return;
        },
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const getFrame = if (d.vi.format.bytesPerSample == 1) &CLAHE(u8).getFrame else &CLAHE(u16).getFrame;
    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, claheFree, .Parallel, &deps, deps.len, data, core);
}
