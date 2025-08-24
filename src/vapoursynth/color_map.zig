const std = @import("std");
const math = std.math;

const filter = @import("../filters/color_map.zig");
const vszip = @import("../vszip.zig");

const Colors = filter.Colors;
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "ColorMap";

const Data = struct {
    node: ?*vs.Node = null,
    vi: vs.VideoInfo = .{},
    color: [3][256]u8 = .{.{0} ** 256} ** 3,
};

fn colorMapGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();

        const dst = src.newVideoFrame3(.{ .format = &d.vi.format });
        const w, const h, const stride = src.getDimensions(0);
        const srcp = src.getReadSlice(0);
        const dstp = dst.getWriteSlices();

        var x: u32 = 0;
        while (x < w) : (x += 1) {
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                const idx: u32 = y * stride + x;
                const s: u8 = srcp[idx];
                dstp[0][idx] = d.color[0][s];
                dstp[1][idx] = d.color[1][s];
                dstp[2][idx] = d.color[2][s];
            }
        }

        dst.getPropertiesRW().setColorRange(.FULL);
        return dst.frame;
    }

    return null;
}

fn colorMapFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn colorMapCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const in_vi = map_in.getNodeVi("clip").?;

    if (zapi.getVideoFormatID(in_vi) != .Gray8) {
        map_out.setError(filter_name ++ ": only Gray8 format is supported.");
        zapi.freeNode(d.node);
        return;
    }

    const icolor = map_in.getValue(i32, "color") orelse 20;
    if (icolor < 0 or icolor > 21) {
        map_out.setError(filter_name ++ ": \"color\" should be between 0 and 21.");
        zapi.freeNode(d.node);
        return;
    }

    const color: Colors = @enumFromInt(icolor);
    const color_arr = color.getColor();

    for (0..256) |i| {
        const j: usize = color_arr[0].len * i / 256;
        d.color[0][i] = @intFromFloat(@mulAdd(f32, color_arr[0][j], 255, 0.5));
        d.color[1][i] = @intFromFloat(@mulAdd(f32, color_arr[1][j], 255, 0.5));
        d.color[2][i] = @intFromFloat(@mulAdd(f32, color_arr[2][j], 255, 0.5));
    }

    d.vi = in_vi.*;
    _ = zapi.getVideoFormatByID(&d.vi.format, .RGB24);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, &d.vi, colorMapGetFrame, colorMapFree, .Parallel, &deps, data);
}
