const std = @import("std");
const math = std.math;

const filter = @import("../filters/clahe.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "CLAHE";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    limit: u32 = 0,
    tiles: [2]u32 = .{ 0, 0 },
};

fn CLAHE(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const width, const height, const stride = src.getDimensions2(T, plane);
                    filter.applyCLAHE(T, srcp, dstp, stride, width, height, d.limit, &d.tiles);
                }

                const dst_prop = dst.getPropertiesRW();
                dst_prop.setColorRange(.FULL);
                return dst.frame;
            }

            return null;
        }
    };
}

fn claheFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn claheCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;

    if ((d.vi.format.sampleType != .Integer)) {
        map_out.setError(filter_name ++ ": only 8-16 bit int formats supported.");
        zapi.freeNode(d.node);
        return;
    }

    d.limit = map_in.getValue(u32, "limit") orelse 7;
    const df_arr = [2]i64{ 3, 3 };
    const tiles_arr = map_in.getIntArray("tiles") orelse &df_arr;
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
            zapi.freeNode(d.node);
            return;
        },
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const getFrame = if (d.vi.format.bytesPerSample == 1) &CLAHE(u8).getFrame else &CLAHE(u16).getFrame;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, claheFree, .Parallel, &deps, data);
}
