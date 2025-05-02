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
        fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n, frame_ctx);
                defer src.deinit();
                const dst = zapi.initZFrameFromVi(&d.out_vi, frame_ctx, src.frame, .{});

                const w: u32 = @intCast(d.out_vi.width);
                const h: u32 = @intCast(d.out_vi.height);

                if (is_rgb24) {
                    const src_r = src.getReadSlice(0);
                    const src_g = src.getReadSlice(1);
                    const src_b = src.getReadSlice(2);
                    const src_stride = src.getStride(0);
                    const dstp = dst.getWriteSlice(0);

                    for (0..h) |y| {
                        for (0..w) |x| {
                            const i_src = y * src_stride + x;
                            const i_dst = (y * w + x) * 4;

                            dstp[i_dst + 0] = src_b[i_src];
                            dstp[i_dst + 1] = src_g[i_src];
                            dstp[i_dst + 2] = src_r[i_src];
                            dstp[i_dst + 3] = 255;
                        }
                    }
                } else {
                    const src_r = src.getReadSlice2(u16, 0);
                    const src_g = src.getReadSlice2(u16, 1);
                    const src_b = src.getReadSlice2(u16, 2);
                    const src_stride = src.getStride2(u16, 0);
                    const dstp = dst.getWriteSlice2(u32, 0);

                    for (0..h) |y| {
                        for (0..w) |x| {
                            const i_src = y * src_stride + x;
                            const i_dst = (y * w + x);
                            dstp[i_dst] = @as(u32, src_b[i_src]) | (@as(u32, src_g[i_src]) << 10) | (@as(u32, src_r[i_src]) << 20) | (0b11 << 30);
                        }
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn packrgbFree(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn packrgbCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, const in_vi = map_in.getNodeVi("clip");
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
