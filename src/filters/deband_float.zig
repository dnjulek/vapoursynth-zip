const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const hz = @import("../common.zig");
const vcl = @import("../vcl.zig");
const angle = @import("deband_angle.zig");
const Mode = @import("deband_setup.zig").Mode;

const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
const i32v = @Vector(vec_len, i32);
const f32v = @Vector(vec_len, f32);
const boolv = @Vector(vec_len, bool);

const vec025: f32v = @splat(0.25);
const vec050: f32v = @splat(0.50);
const vec200: f32v = @splat(2.00);
const vec0_f32: f32v = @splat(0.0);
const vec1_f32: f32v = @splat(1.0);
const vec2_f32: f32v = @splat(2.0);
const vec3_f32: f32v = @splat(3.0);
const vec01_f32: f32v = @splat(0.1);
const vec05_f32: f32v = @splat(0.5);
const vec_eps_f32: f32v = @splat(1e-5);
const vec_pi_f32: f32v = @splat(std.math.pi);
const vec_scaled_eps: f32v = @splat(0.01 * 3.0);

pub fn processPlane(
    src: []const f32,
    dst: []f32,
    ref1: []const i32,
    ref2: []const i32,
    grain: anytype,
    stride: u32,
    width: u32,
    height: u32,
    thr: f32,
    thr1: f32,
    thr2: f32,
    angle_boost: f32,
    max_angle: f32,
    /// mode 7's scratch (unused by other modes)
    angle_pool: *angle.Pool,
    pixel_min: f32,
    pixel_max: f32,
    comptime mode: Mode,
    comptime blur_first: bool,
    comptime add_grain: bool,
) void {
    const minv: f32v = @splat(pixel_min);
    const maxv: f32v = @splat(pixel_max);

    const thr_32: f32v = @splat(thr);
    const thr1_32: f32v = @splat(thr1);
    const thr2_32: f32v = @splat(thr2);

    // Mode 7: the gradient angle of every pixel the 4 reference reads can
    // address, computed once per plane (see deband_angle.zig).
    const angles: angle.Plane = if (mode == .m7)
        (angle.Plane.init(f32, src, stride, width, height, angle_pool) catch {
            var yy: u32 = 0; // graceful passthrough on OOM
            while (yy < height) : (yy += 1) @memcpy(dst[yy * stride ..][0..width], src[yy * stride ..][0..width]);
            return;
        })
    else
        undefined;
    defer if (mode == .m7) angles.deinit(angle_pool);
    const inv_stride: f32 = 1.0 / @as(f32, @floatFromInt(stride));

    const grain_stride: u32 = hz.ceilN(width, 32 / @sizeOf(f32));
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row: u32 = y * stride;
        const grain_row = if (add_grain) grain[y * grain_stride ..];
        const ref1_row = ref1[row..];
        const ref2_row = ref2[row..];

        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            var center: f32v = src[row + x ..][0..vec_len].*;

            // Gathers straight into vector lanes off one centre pointer (see
            // deband_int: the stack-array form spilled its addresses).
            const cp: [*]const f32 = src[row + x ..].ptr;
            var r1_32: f32v = undefined;
            var r3_32: f32v = undefined;
            inline for (0..vec_len) |i| {
                const idx1: isize = ref1_row[x + i];
                r1_32[i] = (cp + i + @as(usize, @bitCast(idx1)))[0];
                r3_32[i] = (cp + i - @as(usize, @bitCast(idx1)))[0];
            }

            var r2_32: f32v = @splat(0); // m1/m3 never read the horizontal refs
            var r4_32: f32v = @splat(0);
            if (mode != .m1 and mode != .m3) {
                inline for (0..vec_len) |i| {
                    // float path uses |ref2| (unlike deband_int's signed
                    // ref_pos_2) — preserved as-is
                    const idx2: isize = @intCast(@abs(ref2_row[x + i]));
                    r2_32[i] = (cp + i + @as(usize, @bitCast(idx2)))[0];
                    r4_32[i] = (cp + i - @as(usize, @bitCast(idx2)))[0];
                }
            }

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
                .m6, .m7 => {
                    var t_avg: f32v = thr_32;
                    var t_max: f32v = thr1_32;
                    var t_mid: f32v = thr2_32;

                    if (mode == .m7) {
                        const angle_boost_v: f32v = @splat(angle_boost);
                        const use_boost: boolv = angles.boost(y, x, ref1_row[x..][0..vec_len].*, ref2_row[x..][0..vec_len].*, inv_stride, @splat(max_angle));
                        t_avg = @select(f32, use_boost, t_avg * angle_boost_v, t_avg);
                        t_max = @select(f32, use_boost, t_max * angle_boost_v, t_max);
                        t_mid = @select(f32, use_boost, t_mid * angle_boost_v, t_mid);
                    }

                    const p1: f32v = r1_32;
                    const p2: f32v = r3_32;
                    const p3: f32v = r2_32;
                    const p4: f32v = r4_32;

                    const avg_refs: f32v = (p1 + p2 + p3 + p4) * vec025;
                    const diff_avg_src: f32v = avg_refs - center;
                    const avg_dif: f32v = @abs(diff_avg_src);

                    const d1: f32v = @abs(p1 - center);
                    const d2: f32v = @abs(p2 - center);
                    const d3: f32v = @abs(p3 - center);
                    const d4: f32v = @abs(p4 - center);
                    const max_dif: f32v = @max(@max(d1, d2), @max(d3, d4));

                    const two_src: f32v = center * vec2_f32;
                    const mid_dif_v: f32v = @abs((p1 + p2) - two_src);
                    const mid_dif_h: f32v = @abs((p3 + p4) - two_src);

                    const comp_avg: f32v = saturate(vec3_f32 * (vec1_f32 - avg_dif / @max(t_avg, vec_eps_f32)));
                    const comp_max: f32v = saturate(vec3_f32 * (vec1_f32 - max_dif / @max(t_max, vec_eps_f32)));
                    const comp_mid_v: f32v = saturate(vec3_f32 * (vec1_f32 - mid_dif_v / @max(t_mid, vec_eps_f32)));
                    const comp_mid_h: f32v = saturate(vec3_f32 * (vec1_f32 - mid_dif_h / @max(t_mid, vec_eps_f32)));

                    const product: f32v = comp_avg * comp_max * comp_mid_v * comp_mid_h;
                    const factor: f32v = vcl.pow(product, vec01_f32);
                    center = center + diff_avg_src * factor;
                },
            }

            if (add_grain) {
                center += @as(f32v, grain_row[x..][0..vec_len].*);
            }

            center = @max(minv, @min(center, maxv));
            dst[row + x ..][0..vec_len].* = center;
        }
    }
}

fn saturate(x: f32v) f32v {
    return @max(vec0_f32, @min(x, vec1_f32));
}
