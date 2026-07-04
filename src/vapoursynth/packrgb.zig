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

                // Both packs are purely vertical (widen + shift + or), but LLVM
                // never auto-vectorized the scalar loops (measured: 8-10 scalar
                // instructions per pixel, ~100% of the filter's kernel time), so
                // the vector shape is explicit: 8 u32 lanes per iteration.
                const T = if (is_rgb24) u8 else u16;
                const srcp = src.getReadSlices2(T);
                const src_stride = src.getStride2(T, 0);
                const dst_stride = dst.getStride2(u32, 0);
                const dstp = dst.getWriteSlice2(u32, 0);

                const shifts = if (is_rgb24) [2]comptime_int{ 8, 16 } else [2]comptime_int{ 10, 20 };
                const alpha: u32 = if (is_rgb24) 0xFF00_0000 else 0b11 << 30;

                const nv = comptime (std.simd.suggestVectorLength(u32) orelse 1);
                const wv = if (nv > 1) w - w % nv else 0;
                for (0..h) |y| {
                    const rb = srcp[2][y * src_stride ..];
                    const rg = srcp[1][y * src_stride ..];
                    const rr = srcp[0][y * src_stride ..];
                    const rd = dstp[y * dst_stride ..];

                    var x: usize = 0;
                    if (comptime nv > 1) {
                        const sh_g: @Vector(nv, u5) = @splat(shifts[0]);
                        const sh_r: @Vector(nv, u5) = @splat(shifts[1]);
                        const av: @Vector(nv, u32) = @splat(alpha);
                        while (x < wv) : (x += nv) {
                            const b: @Vector(nv, T) = rb[x..][0..nv].*;
                            const g: @Vector(nv, T) = rg[x..][0..nv].*;
                            const r: @Vector(nv, T) = rr[x..][0..nv].*;
                            rd[x..][0..nv].* = @as(@Vector(nv, u32), b) |
                                (@as(@Vector(nv, u32), g) << sh_g) |
                                (@as(@Vector(nv, u32), r) << sh_r) | av;
                        }
                    }
                    while (x < w) : (x += 1) {
                        rd[x] = @as(u32, rb[x]) | (@as(u32, rg[x]) << shifts[0]) |
                            (@as(u32, rr[x]) << shifts[1]) | alpha;
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
