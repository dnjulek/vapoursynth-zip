const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const hz = @import("../helper.zig");
const plugin = @import("../vapoursynth/deband.zig");
const vszip = @import("../vszip.zig");
const vcl = @import("../vcl.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const Mode = plugin.Mode;
const Data = plugin.Data;

const vec_len = std.simd.suggestVectorLength(i32) orelse 1;
const i32v = @Vector(vec_len, i32);
const u32v = @Vector(vec_len, u32);
const u16v = @Vector(vec_len, u16);
const i16v = @Vector(vec_len, i16);
const f32v = @Vector(vec_len, f32);
const boolv = @Vector(vec_len, bool);
const vec1_i32: i32v = @splat(1);
const vec2_i32: i32v = @splat(2);
const vec4_i32: i32v = @splat(4);

const vec0_f32: f32v = @splat(0.0);
const vec1_f32: f32v = @splat(1.0);
const vec2_f32: f32v = @splat(2.0);
const vec3_f32: f32v = @splat(3.0);
const vec025_f32: f32v = @splat(0.25);
const vec01_f32: f32v = @splat(0.1);
const vec05_f32: f32v = @splat(0.5);
const vec_eps_f32: f32v = @splat(1e-5);
const vec_pi_f32: f32v = @splat(std.math.pi);
const vec_scaled_eps: f32v = @splat(0.01 * 3.0);

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
                    const src_slice = src.getReadSlice2(u16, plane);
                    const dst_slice = dst.getWriteSlice2(u16, plane);
                    const stride = src.getStride(plane) >> 1;
                    const width = src.getWidth(plane);
                    const height = src.getHeight(plane);
                    var grain = if (add_grain[plane]) d.tb.grain_int[plane];
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
                        d.thr.u[plane],
                        d.thr1.u[plane],
                        d.thr2.u[plane],
                        d.pixel_min[plane],
                        d.pixel_max[plane],
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
    src: []const u16,
    dst: []u16,
    ref1: []const u16,
    ref2: []const u16,
    grain: anytype,
    stride: u32,
    width: u32,
    height: u32,
    thr: u16,
    thr1: u16,
    thr2: u16,
    pixel_min: i32,
    pixel_max: i32,
    angle_boost: f32,
    max_angle: f32,
    comptime mode: Mode,
    comptime blur_first: bool,
    comptime add_grain: bool,
) void {
    var ref1_arr: [vec_len]u16 align(32) = undefined;
    var ref2_arr: [vec_len]u16 align(32) = undefined;
    var ref3_arr: [vec_len]u16 align(32) = undefined;
    var ref4_arr: [vec_len]u16 align(32) = undefined;

    const thr_32: u32v = @splat(@as(u32, thr));
    const thr1_32: u32v = @splat(@as(u32, thr1));
    const thr2_32: u32v = @splat(@as(u32, thr2));
    const minv: i32v = @splat(pixel_min);
    const maxv: i32v = @splat(pixel_max);

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row: u32 = y * stride;
        const grain_row = if (add_grain) grain[row..];
        const ref1_row = ref1[row..];
        const ref2_row = ref2[row..];

        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            const src_16: u16v = src[row + x ..][0..vec_len].*;
            var center: i32v = @intCast(src_16);

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

            const r1_16: u16v = @bitCast(ref1_arr);
            const r2_16: u16v = @bitCast(ref2_arr);
            const r3_16: u16v = @bitCast(ref3_arr);
            const r4_16: u16v = @bitCast(ref4_arr);
            const r1_32: i32v = @intCast(r1_16);
            const r2_32: i32v = @intCast(r2_16);
            const r3_32: i32v = @intCast(r3_16);
            const r4_32: i32v = @intCast(r4_16);

            switch (mode) {
                .m1, .m3 => {
                    const avg_32 = (r1_32 + r3_32 + vec1_i32) >> vec1_i32;
                    const use_original = if (blur_first)
                        (@abs(avg_32 - center) >= thr_32)
                    else
                        (@abs(r1_32 - center) >= thr_32) | (@abs(r3_32 - center) >= thr_32);

                    center = @select(i32, use_original, center, avg_32);
                },
                .m2 => {
                    const avg1 = (r1_32 + r3_32 + vec1_i32) >> vec1_i32;
                    const avg2 = (r2_32 + r4_32 + vec1_i32) >> vec1_i32;
                    const avg_32 = (avg1 + avg2 + vec1_i32) >> vec1_i32;
                    const use_original = if (blur_first)
                        (@abs(avg_32 - center) >= thr_32)
                    else
                        ((@abs(r1_32 - center) >= thr_32) |
                            (@abs(r2_32 - center) >= thr_32) |
                            (@abs(r3_32 - center) >= thr_32) |
                            (@abs(r4_32 - center) >= thr_32));

                    center = @select(i32, use_original, center, avg_32);
                },
                .m4 => {
                    const avg_v = (r1_32 + r3_32 + vec1_i32) >> vec1_i32;
                    const avg_h = (r2_32 + r4_32 + vec1_i32) >> vec1_i32;
                    const use_orig_v: boolv = if (blur_first)
                        (@abs(avg_v - center) >= thr_32)
                    else
                        ((@abs(r1_32 - center) >= thr_32) | (@abs(r3_32 - center) >= thr_32));

                    const use_orig_h: boolv = if (blur_first)
                        (@abs(avg_h - center) >= thr_32)
                    else
                        ((@abs(r2_32 - center) >= thr_32) | (@abs(r4_32 - center) >= thr_32));

                    const dst_v = @select(i32, use_orig_v, center, avg_v);
                    const dst_h = @select(i32, use_orig_h, center, avg_h);
                    center = (dst_v + dst_h + vec1_i32) >> vec1_i32;
                },
                .m5 => {
                    const avg_32 = (r1_32 + r2_32 + r3_32 + r4_32 + vec2_i32) >> vec2_i32;
                    const avg_dif = @abs(avg_32 - center);
                    const d1 = @abs(r1_32 - center);
                    const d2 = @abs(r2_32 - center);
                    const d3 = @abs(r3_32 - center);
                    const d4 = @abs(r4_32 - center);
                    const max_dif = @max(@max(d1, d2), @max(d3, d4));
                    const two_src = center << vec1_i32;
                    const mid_dif1 = @abs((r1_32 + r3_32) - two_src);
                    const mid_dif2 = @abs((r2_32 + r4_32) - two_src);
                    const use_original = (avg_dif >= thr_32) |
                        (max_dif >= thr1_32) |
                        (mid_dif1 >= thr2_32) |
                        (mid_dif2 >= thr2_32);

                    center = @select(i32, use_original, center, avg_32);
                },
                .m6, .m7 => {
                    var t_avg: f32v = @splat(@as(f32, @floatFromInt(thr)));
                    var t_max: f32v = @splat(@as(f32, @floatFromInt(thr1)));
                    var t_mid: f32v = @splat(@as(f32, @floatFromInt(thr2)));

                    if (mode == .m7) {
                        const grad_read_distance: i32 = 20;
                        const angle_boost_v: f32v = @splat(angle_boost);
                        const max_angle_v: f32v = @splat(max_angle);

                        var base_x_coords: i32v = undefined;
                        const base_y_coords: i32v = @splat(@intCast(y));
                        inline for (0..vec_len) |i| {
                            base_x_coords[i] = @intCast(x + i);
                        }

                        var y_offsets: i32v = undefined;
                        var x_offsets: i32v = undefined;
                        inline for (0..vec_len) |i| {
                            const offset_val: i32 = @intCast(ref2_row[x + i]);
                            y_offsets[i] = offset_val;
                            x_offsets[i] = offset_val;
                        }

                        const angle_org = calculateGradientAngle(src, stride, width, height, base_y_coords, base_x_coords, grad_read_distance);
                        const angle_ref1_h = calculateGradientAngle(src, stride, width, height, base_y_coords + y_offsets, base_x_coords, grad_read_distance);
                        const angle_ref2_h = calculateGradientAngle(src, stride, width, height, base_y_coords - y_offsets, base_x_coords, grad_read_distance);
                        const angle_ref1_w = calculateGradientAngle(src, stride, width, height, base_y_coords, base_x_coords + x_offsets, grad_read_distance);
                        const angle_ref2_w = calculateGradientAngle(src, stride, width, height, base_y_coords, base_x_coords - x_offsets, grad_read_distance);

                        var max_angle_diff = @max(@abs(angle_ref1_h - angle_org), @abs(angle_ref2_h - angle_org));
                        max_angle_diff = @max(max_angle_diff, @max(@abs(angle_ref1_w - angle_org), @abs(angle_ref2_w - angle_org)));
                        const use_boost: boolv = max_angle_diff <= max_angle_v;
                        t_avg = @select(f32, use_boost, t_avg * angle_boost_v, t_avg);
                        t_max = @select(f32, use_boost, t_max * angle_boost_v, t_max);
                        t_mid = @select(f32, use_boost, t_mid * angle_boost_v, t_mid);
                    }

                    const src_f: f32v = @floatFromInt(center);
                    const p1: f32v = @floatFromInt(r1_32);
                    const p2: f32v = @floatFromInt(r3_32);
                    const p3: f32v = @floatFromInt(r2_32);
                    const p4: f32v = @floatFromInt(r4_32);

                    const avg_refs: f32v = (p1 + p2 + p3 + p4) * vec025_f32;
                    const diff_avg_src: f32v = avg_refs - src_f;
                    const avg_dif: f32v = @abs(diff_avg_src);

                    const d1: f32v = @abs(p1 - src_f);
                    const d2: f32v = @abs(p2 - src_f);
                    const d3: f32v = @abs(p3 - src_f);
                    const d4: f32v = @abs(p4 - src_f);
                    const max_dif: f32v = @max(@max(d1, d2), @max(d3, d4));

                    const two_src: f32v = src_f * vec2_f32;
                    const mid_dif_v: f32v = @abs((p1 + p2) - two_src);
                    const mid_dif_h: f32v = @abs((p3 + p4) - two_src);

                    const comp_avg: f32v = saturate(vec3_f32 * (vec1_f32 - avg_dif / @max(t_avg, vec_eps_f32)));
                    const comp_max: f32v = saturate(vec3_f32 * (vec1_f32 - max_dif / @max(t_max, vec_eps_f32)));
                    const comp_mid_v: f32v = saturate(vec3_f32 * (vec1_f32 - mid_dif_v / @max(t_mid, vec_eps_f32)));
                    const comp_mid_h: f32v = saturate(vec3_f32 * (vec1_f32 - mid_dif_h / @max(t_mid, vec_eps_f32)));

                    const product: f32v = comp_avg * comp_max * comp_mid_v * comp_mid_h;
                    const factor: f32v = vcl.pow(product, vec01_f32);
                    const blended: f32v = src_f + diff_avg_src * factor;

                    center = @intFromFloat(blended + @as(f32v, @splat(0.5)));
                },
            }

            if (add_grain) {
                center += @as(i16v, grain_row[x..][0..vec_len].*);
            }

            const clamped = @max(minv, @min(center, maxv));
            dst[row + x ..][0..vec_len].* = @as(u16v, @intCast(clamped));
        }
    }
}

fn saturate(x: f32v) f32v {
    return @max(vec0_f32, @min(x, vec1_f32));
}

fn gatherPixelValues(src: []const u16, stride: u32, width: u32, height: u32, y_coords: i32v, x_coords: i32v) f32v {
    const width_i: i32v = @splat(@intCast(width - 1));
    const height_i: i32v = @splat(@intCast(height - 1));
    const stride_i: i32v = @splat(@intCast(stride));
    const zero_i: i32v = @splat(0);
    const clamped_y = @max(zero_i, @min(y_coords, height_i));
    const clamped_x = @max(zero_i, @min(x_coords, width_i));
    const offsets: u32v = @intCast(clamped_y * stride_i + clamped_x);

    var result_arr: [vec_len]f32 align(32) = undefined;
    inline for (0..vec_len) |i| {
        result_arr[i] = @floatFromInt(src[offsets[i]]);
    }
    return result_arr;
}

fn calculateGradientAngle(src: []const u16, stride: u32, width: u32, height: u32, y_coords: i32v, x_coords: i32v, read_distance: comptime_int) f32v {
    const rd: i32v = @splat(read_distance);
    const p00 = gatherPixelValues(src, stride, width, height, y_coords - rd, x_coords - rd);
    const p10 = gatherPixelValues(src, stride, width, height, y_coords - rd, x_coords);
    const p20 = gatherPixelValues(src, stride, width, height, y_coords - rd, x_coords + rd);
    const p01 = gatherPixelValues(src, stride, width, height, y_coords, x_coords - rd);
    const p21 = gatherPixelValues(src, stride, width, height, y_coords, x_coords + rd);
    const p02 = gatherPixelValues(src, stride, width, height, y_coords + rd, x_coords - rd);
    const p12 = gatherPixelValues(src, stride, width, height, y_coords + rd, x_coords);
    const p22 = gatherPixelValues(src, stride, width, height, y_coords + rd, x_coords + rd);

    const gx = (p20 + vec2_f32 * p21 + p22) - (p00 + vec2_f32 * p01 + p02);
    const gy = (p00 + vec2_f32 * p10 + p20) - (p02 + vec2_f32 * p12 + p22);
    const gx_is_small: boolv = @abs(gx) < vec_scaled_eps;

    const angle_raw = vcl.atan(gy / gx);
    const angle_normalized = angle_raw / vec_pi_f32 + vec05_f32;
    return @select(f32, gx_is_small, vec1_f32, angle_normalized);
}
