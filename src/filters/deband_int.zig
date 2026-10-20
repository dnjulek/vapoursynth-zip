const std = @import("std");
const simd = @import("simd.zig");
const math = std.math;
const allocator = std.heap.c_allocator;

const hz = @import("../common.zig");
const vcl = @import("../vcl.zig");
const angle = @import("deband_angle.zig");
const Mode = @import("deband_setup.zig").Mode;

const vec_len = std.simd.suggestVectorLength(i32) orelse 8;
const i32v = @Vector(vec_len, i32);
const u32v = @Vector(vec_len, u32);
const u16v = @Vector(vec_len, u16);
const i16v = @Vector(vec_len, i16);
const f32v = @Vector(vec_len, f32);
const boolv = @Vector(vec_len, bool);
const vec0_i32: i32v = @splat(0);
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

pub fn processPlane(
    src: []const u16,
    dst: []u16,
    ref1: []const i32,
    ref2: []const i32,
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
    /// mode 7's scratch (unused by other modes)
    angle_pool: *angle.Pool,
    comptime mode: Mode,
    comptime blur_first: bool,
    comptime add_grain: bool,
) void {
    const thr_32: u32v = @splat(@as(u32, thr));
    const minv: i32v = @splat(pixel_min);
    const maxv: i32v = @splat(pixel_max);

    // Mode 7: the gradient angle of every pixel the 4 reference reads can
    // address, computed once per plane (see deband_angle.zig).
    const angles: angle.Plane = if (mode == .m7)
        (angle.Plane.init(u16, src, stride, width, height, angle_pool) catch {
            var yy: u32 = 0; // graceful passthrough on OOM
            while (yy < height) : (yy += 1) @memcpy(dst[yy * stride ..][0..width], src[yy * stride ..][0..width]);
            return;
        })
    else
        undefined;
    defer if (mode == .m7) angles.deinit(angle_pool);
    const inv_stride: f32 = 1.0 / @as(f32, @floatFromInt(stride));

    const grain_stride: u32 = hz.ceilN(width, 32 / @sizeOf(u16));
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row: u32 = y * stride;
        const grain_row = if (add_grain) grain[y * grain_stride ..];
        const ref1_row = ref1[row..];
        const ref2_row = ref2[row..];

        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            const src_16: u16v = src[row + x ..][0..vec_len].*;
            var center: i32v = @intCast(src_16);

            // Gathers straight into vector lanes off one centre pointer: the
            // stack-array form made LLVM materialize all 4*vec_len addresses
            // up front and spill them (rbp-slot reloads were the loop's hot
            // spots). Same reads, same lanes.
            const cp: [*]const u16 = src[row + x ..].ptr;
            var r1_16: u16v = undefined;
            var r3_16: u16v = undefined;
            inline for (0..vec_len) |i| {
                const idx1: isize = ref1_row[x + i];
                r1_16[i] = (cp + i + @as(usize, @bitCast(idx1)))[0];
                r3_16[i] = (cp + i - @as(usize, @bitCast(idx1)))[0];
            }

            var r2_16: u16v = undefined;
            var r4_16: u16v = undefined;
            if (mode != .m1 and mode != .m3) {
                inline for (0..vec_len) |i| {
                    // idx2 may be negative (sample_mode 2): +idx2 reads the
                    // "forward" ref_2, -idx2 the "backward" ref_4 — matching the
                    // signed ref_pos_2 in neo_f3kdb.
                    const idx2: isize = ref2_row[x + i];
                    r2_16[i] = (cp + i + @as(usize, @bitCast(idx2)))[0];
                    r4_16[i] = (cp + i - @as(usize, @bitCast(idx2)))[0];
                }
            } else {
                // m1/m3 never read the horizontal refs
                r2_16 = @splat(0);
                r4_16 = @splat(0);
            }
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
                    // neo's SSE/AVX path (opt>=1, what auto-detect runs) pairs the
                    // two vertical refs (r1=+pos, r3=-pos) and the two horizontal
                    // refs (r2=+pos2, r4=-pos2) — a different avg_4 pairing than the
                    // C scalar path (opt=0).
                    const avg_32 = avg4(r1_32, r3_32, r2_32, r4_32);
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
                    // neo's AVX2 kernel (opt=2) ran m5's diff/threshold math in
                    // 16-bit lanes, which OVERFLOW on high-contrast content: a true
                    // |ref-src| > 32767 wraps to a small value, so a sharp edge looks
                    // flat and gets averaged. That is a genuine SIMD bug — neo's SSE
                    // path (opt=1), C path (opt=0) and our own float path all compute
                    // these diffs in 32-bit and don't overflow, so we do the same.
                    // avg is the FLAT truncated (sum>>2), matching every neo m5 path.
                    // AVX2 ref order (+v,-v,+h,-h) maps to vszip (r1,r3,r2,r4).
                    const avg = (r1_32 + r3_32 + r2_32 + r4_32) >> vec2_i32;
                    const avg_dif = @abs(avg - center);
                    const max_dif = @max(
                        @max(@abs(r1_32 - center), @abs(r3_32 - center)),
                        @max(@abs(r2_32 - center), @abs(r4_32 - center)),
                    );
                    const two_src = center << vec1_i32;
                    const mid_dif1 = @abs((r1_32 + r3_32) - two_src);
                    const mid_dif2 = @abs((r2_32 + r4_32) - two_src);

                    const thr_v: u32v = @splat(@as(u32, thr));
                    const thr1_v: u32v = @splat(@as(u32, thr1));
                    const thr2_v: u32v = @splat(@as(u32, thr2));
                    const use_original = (avg_dif >= thr_v) |
                        (max_dif >= thr1_v) |
                        (mid_dif1 >= thr2_v) |
                        (mid_dif2 >= thr2_v);

                    center = @select(i32, use_original, center, avg);
                },
                .m6, .m7 => {
                    var t_avg: f32v = @splat(@as(f32, @floatFromInt(thr)));
                    var t_max: f32v = @splat(@as(f32, @floatFromInt(thr1)));
                    var t_mid: f32v = @splat(@as(f32, @floatFromInt(thr2)));

                    if (mode == .m7) {
                        const angle_boost_v: f32v = @splat(angle_boost);
                        const use_boost: boolv = angles.boost(y, x, ref1_row[x..][0..vec_len].*, ref2_row[x..][0..vec_len].*, inv_stride, @splat(max_angle));
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

                    center = @trunc(blended + @as(f32v, @splat(0.5)));
                },
            }

            if (add_grain) {
                center += @as(i16v, grain_row[x..][0..vec_len].*);
            }

            const clamped = @max(minv, @min(center, maxv));
            dst[row + x ..][0..vec_len].* = simd.intCastSignedToUnsigned(u16v, clamped);
        }
    }
}

fn saturate(x: f32v) f32v {
    return @max(vec0_f32, @min(x, vec1_f32));
}

/// Bit-exact port of neo_f3kdb's pixel_proc avg_4 (16-bit path): two rounded
/// pair-averages, with the first decremented by one when positive — a quirk
/// the original keeps "consistent with SSE code". Argument order matters: the
/// pairs averaged are (a,b) and (c,d).
inline fn avg4(a: i32v, b: i32v, c: i32v, d: i32v) i32v {
    var avg1 = (a + b + vec1_i32) >> vec1_i32;
    const avg2 = (c + d + vec1_i32) >> vec1_i32;
    avg1 -= @select(i32, avg1 > vec0_i32, vec1_i32, vec0_i32);
    return (avg1 + avg2 + vec1_i32) >> vec1_i32;
}
