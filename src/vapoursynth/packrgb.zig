const std = @import("std");
const math = std.math;

const vszip = @import("../vszip.zig");
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "PackRGB";

const Data = struct {
    node: ?*vs.Node,
    out_vi: vs.VideoInfo,
};

fn Pack(comptime is_rgb24: bool) type {
    return struct {
        fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.out_vi.format });

                const w: u32 = @intCast(d.out_vi.width);
                const h: u32 = @intCast(d.out_vi.height);

                if (is_rgb24) {
                    const srcp = src.getReadSlices();
                    const src_stride = src.getStride(0);
                    const dst_stride = dst.getStride2(u32, 0);
                    const dstp = dst.getWriteSlice(0);

                    for (0..h) |y| {
                        for (0..w) |x| {
                            const i_src = y * src_stride + x;
                            const i_dst = (y * dst_stride + x) * 4;

                            dstp[i_dst + 0] = srcp[2][i_src];
                            dstp[i_dst + 1] = srcp[1][i_src];
                            dstp[i_dst + 2] = srcp[0][i_src];
                            dstp[i_dst + 3] = 255;
                        }
                    }
                } else {
                    const srcp = src.getReadSlices2(u16);
                    const src_stride = src.getStride2(u16, 0);
                    const dst_stride = dst.getStride2(u32, 0);
                    const dstp = dst.getWriteSlice2(u32, 0);

                    for (0..h) |y| {
                        for (0..w) |x| {
                            const i_src = y * src_stride + x;
                            const i_dst = (y * dst_stride + x);
                            dstp[i_dst] = @as(u32, srcp[2][i_src]) | (@as(u32, srcp[1][i_src]) << 10) | (@as(u32, srcp[0][i_src]) << 20) | (0b11 << 30);
                        }
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn packrgbFree(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn packrgbCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, const in_vi = map_in.getNodeVi("clip").?;
    d.out_vi = in_vi.*;

    const id = zapi.getVideoFormatID(in_vi);
    switch (id) {
        .RGB24, .RGB30 => {},

        else => {
            map_out.setError(filter_name ++ ": only RGB24 and RGB30 inputs are supported!");
            zapi.freeNode(d.node);
            return;
        },
    }

    _ = zapi.getVideoFormatByID(&d.out_vi.format, .Gray32);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = if (id == .RGB24) &Pack(true).getFrame else &Pack(false).getFrame;
    zapi.createVideoFilter(out, filter_name, &d.out_vi, gf, packrgbFree, .Parallel, &deps, data);
}
