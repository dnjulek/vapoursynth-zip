const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const hz = @import("../helper.zig");
const plugin = @import("../vapoursynth/deband.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const Mode = plugin.Mode;
const Data = plugin.Data;

const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
const f32v = @Vector(vec_len, f32);
const boolv = @Vector(vec_len, bool);
const vec025: f32v = @splat(0.25);
const vec050: f32v = @splat(0.50);
const vec200: f32v = @splat(2.00);

pub fn F3KDB(comptime mode: Mode, comptime blur_first: bool, comptime add_grain: [3]bool, np: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame2(add_grain);

                inline for (0..np) |plane| {
                    const src_slice = src.getReadSlice2(f32, plane);
                    const dst_slice = dst.getWriteSlice2(f32, plane);
                    const stride = src.getStride(plane) >> 2;
                    const width = src.getWidth(plane);
                    const height = src.getHeight(plane);
                    var grain = if (add_grain[plane]) d.tb.grain_float[plane];
                    if (add_grain[plane] and d.dynamic_grain) {
                        const offset = d.tb.grain_offsets[@intCast(n)];
                        grain = grain[offset..];
                    }

                    processPlane(
                        src_slice,
                        dst_slice,
                        d.tb.ref1[plane],
                        d.tb.ref2[plane],
                        grain,
                        stride,
                        width,
                        height,
                        d.thr.f[plane],
                        d.thr1.f[plane],
                        d.thr2.f[plane],
                        d.angle_boost,
                        d.max_angle,
                        mode,
                        blur_first,
                        add_grain[plane],
                    );
                }

                return dst.frame;
            }
            return null;
        }
    };
}

fn processPlane(
    src: []const f32,
    dst: []f32,
    ref1: []const u16,
    ref2: []const u16,
    grain: anytype,
    stride: u32,
    width: u32,
    height: u32,
    thr: f32,
    thr1: f32,
    thr2: f32,
    angle_boost: f32,
    max_angle: f32,
    comptime mode: Mode,
    comptime blur_first: bool,
    comptime add_grain: bool,
) void {
    _ = angle_boost;
    _ = max_angle;

    var ref1_arr: [vec_len]f32 align(32) = undefined;
    var ref2_arr: [vec_len]f32 align(32) = undefined;
    var ref3_arr: [vec_len]f32 align(32) = undefined;
    var ref4_arr: [vec_len]f32 align(32) = undefined;

    const thr_32: f32v = @splat(thr);
    const thr1_32: f32v = @splat(thr1);
    const thr2_32: f32v = @splat(thr2);

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row: u32 = y * stride;
        const grain_row = if (add_grain) grain[row..];
        const ref1_row = ref1[row..];
        const ref2_row = ref2[row..];

        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            var center: f32v = src[row + x ..][0..vec_len].*;

            inline for (0..vec_len) |i| {
                const cur_xy = row + x + i;
                const idx1 = ref1_row[x + i];
                ref1_arr[i] = src[cur_xy + idx1];
                ref3_arr[i] = src[cur_xy - idx1];
            }

            if (mode != .m1 and mode != .m3) {
                inline for (0..vec_len) |i| {
                    const cur_xy = row + x + i;
                    const idx2 = ref2_row[x + i];
                    ref2_arr[i] = src[cur_xy + idx2];
                    ref4_arr[i] = src[cur_xy - idx2];
                }
            }

            const r1_32: f32v = @bitCast(ref1_arr);
            const r2_32: f32v = @bitCast(ref2_arr);
            const r3_32: f32v = @bitCast(ref3_arr);
            const r4_32: f32v = @bitCast(ref4_arr);

            switch (mode) {
                .m1, .m3 => {
                    const avg_32 = (r1_32 + r3_32) * vec050;
                    const use_original = if (blur_first)
                        (@abs(avg_32 - center) >= thr_32)
                    else
                        (@abs(r1_32 - center) >= thr_32) | (@abs(r3_32 - center) >= thr_32);

                    center = @select(f32, use_original, center, avg_32);
                },
                .m2 => {
                    const avg_32 = (r1_32 + r2_32 + r3_32 + r4_32) * vec025;
                    const use_original = if (blur_first)
                        (@abs(avg_32 - center) >= thr_32)
                    else
                        ((@abs(r1_32 - center) >= thr_32) |
                            (@abs(r2_32 - center) >= thr_32) |
                            (@abs(r3_32 - center) >= thr_32) |
                            (@abs(r4_32 - center) >= thr_32));

                    center = @select(f32, use_original, center, avg_32);
                },
                .m4 => {
                    const avg_v = (r1_32 + r3_32) * vec050;
                    const avg_h = (r2_32 + r4_32) * vec050;
                    const use_orig_v: boolv = if (blur_first)
                        (@abs(avg_v - center) >= thr_32)
                    else
                        ((@abs(r1_32 - center) >= thr_32) | (@abs(r3_32 - center) >= thr_32));

                    const use_orig_h: boolv = if (blur_first)
                        (@abs(avg_h - center) >= thr_32)
                    else
                        ((@abs(r2_32 - center) >= thr_32) | (@abs(r4_32 - center) >= thr_32));

                    const dst_v = @select(f32, use_orig_v, center, avg_v);
                    const dst_h = @select(f32, use_orig_h, center, avg_h);
                    center = (dst_v + dst_h) * vec050;
                },
                .m5 => {
                    const avg_32 = (r1_32 + r2_32 + r3_32 + r4_32) * vec025;
                    const avg_dif = @abs(avg_32 - center);
                    const d1 = @abs(r1_32 - center);
                    const d2 = @abs(r2_32 - center);
                    const d3 = @abs(r3_32 - center);
                    const d4 = @abs(r4_32 - center);
                    const max_dif = @max(@max(d1, d2), @max(d3, d4));
                    const two_src = center * vec200;
                    const mid_dif1 = @abs((r1_32 + r3_32) - two_src);
                    const mid_dif2 = @abs((r2_32 + r4_32) - two_src);
                    const use_original = (avg_dif >= thr_32) |
                        (max_dif >= thr1_32) |
                        (mid_dif1 >= thr2_32) |
                        (mid_dif2 >= thr2_32);

                    center = @select(f32, use_original, center, avg_32);
                },
                .m6, .m7 => {}, // TODO: implement mode 6 and 7 later
            }

            if (add_grain) {
                center += @as(f32v, grain_row[x..][0..vec_len].*);
            }

            dst[row + x ..][0..vec_len].* = center;
        }
    }
}
