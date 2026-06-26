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
    comptime mode: Mode,
    comptime blur_first: bool,
    comptime add_grain: bool,
) void {
    var ref1_arr: [vec_len]u16 align(32) = undefined;
    var ref2_arr: [vec_len]u16 align(32) = undefined;
    var ref3_arr: [vec_len]u16 align(32) = undefined;
    var ref4_arr: [vec_len]u16 align(32) = undefined;

    const thr_32: u32v = @splat(@as(u32, thr));
    const minv: i32v = @splat(pixel_min);
    const maxv: i32v = @splat(pixel_max);

    // Mode 7 evaluates the gradient angle at 5 positions per pixel (org + 4
    // ref-offset reads), each a Sobel kernel over 8 gathered pixels + an atan.
    // Instead, precompute the angle once for every pixel into a padded scratch
    // plane and turn the 5 evaluations into cheap lookups. PAD covers the max
    // |ref offset| (signed-char bound, <=128), so every offset read lands on a
    // precomputed cell with no clamping — making the lookup bit-identical to the
    // per-pixel computation (calculateGradientAngle itself is unchanged).
    const ANGLE_PAD: u32 = 128;
    const ang_stride: u32 = if (mode == .m7) plugin.ceilN(width + 2 * ANGLE_PAD, vec_len) else 0;
    const angle_buf: []f32 = if (mode == .m7)
        (allocator.alloc(f32, (height + 2 * ANGLE_PAD) * ang_stride) catch {
            var yy: u32 = 0; // graceful passthrough on OOM
            while (yy < height) : (yy += 1) @memcpy(dst[yy * stride ..][0..width], src[yy * stride ..][0..width]);
            return;
        })
    else
        &.{};
    defer if (mode == .m7) allocator.free(angle_buf);
    if (mode == .m7) fillAnglePlane(src, stride, width, height, angle_buf, ang_stride, ANGLE_PAD);

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
                const base: isize = @intCast(row + x + i);
                const idx1: isize = ref1_row[x + i];
                ref1_arr[i] = src[@intCast(base + idx1)];
                ref3_arr[i] = src[@intCast(base - idx1)];
            }

            if (mode != .m1 and mode != .m3) {
                inline for (0..vec_len) |i| {
                    const base: isize = @intCast(row + x + i);
                    // idx2 may be negative (sample_mode 2): +idx2 reads the
                    // "forward" ref_2, -idx2 the "backward" ref_4 — matching the
                    // signed ref_pos_2 in neo_f3kdb.
                    const idx2: isize = ref2_row[x + i];
                    ref2_arr[i] = src[@intCast(base + idx2)];
                    ref4_arr[i] = src[@intCast(base - idx2)];
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
                        const max_angle_v: f32v = @splat(max_angle);

                        const stride_i: i32 = @intCast(stride);
                        var y_offsets: i32v = undefined;
                        var x_offsets: i32v = undefined;
                        inline for (0..vec_len) |i| {
                            y_offsets[i] = @divTrunc(ref1_row[x + i], stride_i);
                            x_offsets[i] = ref2_row[x + i];
                        }

                        const astride_i: i32 = @intCast(ang_stride);
                        const pad_i: i32 = @intCast(ANGLE_PAD);
                        const yi: i32 = @intCast(y);
                        const angle_org: f32v = angle_buf[(y + ANGLE_PAD) * ang_stride + (x + ANGLE_PAD) ..][0..vec_len].*;
                        var angle_ref1_h: f32v = undefined;
                        var angle_ref2_h: f32v = undefined;
                        var angle_ref1_w: f32v = undefined;
                        var angle_ref2_w: f32v = undefined;
                        inline for (0..vec_len) |i| {
                            const bx: i32 = @intCast(x + i);
                            const yo = y_offsets[i];
                            const xo = x_offsets[i];
                            const col: i32 = bx + pad_i;
                            const row_y: i32 = (yi + pad_i) * astride_i;
                            angle_ref1_h[i] = angle_buf[@intCast((yi + yo + pad_i) * astride_i + col)];
                            angle_ref2_h[i] = angle_buf[@intCast((yi - yo + pad_i) * astride_i + col)];
                            angle_ref1_w[i] = angle_buf[@intCast(row_y + (bx + xo + pad_i))];
                            angle_ref2_w[i] = angle_buf[@intCast(row_y + (bx - xo + pad_i))];
                        }

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

                    center = @trunc(blended + @as(f32v, @splat(0.5)));
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

/// Fill `buf` with the per-pixel normalized gradient angle for every coordinate
/// in the padded range X,Y in [-PAD, dim-1+PAD], laid out so that the angle for
/// coordinate (Y,X) lives at buf[(Y+PAD)*ang_stride + (X+PAD)]. Uses the exact
/// same calculateGradientAngle as the per-pixel path (which clamps its own
/// neighbour reads), so a lookup is bit-identical to recomputing on the fly.
fn fillAnglePlane(src: []const u16, stride: u32, width: u32, height: u32, buf: []f32, ang_stride: u32, comptime PAD: u32) void {
    const padded_w = width + 2 * PAD;
    const padded_h = height + 2 * PAD;
    var yy: u32 = 0;
    while (yy < padded_h) : (yy += 1) {
        const yc: i32v = @splat(@as(i32, @intCast(yy)) - @as(i32, @intCast(PAD)));
        const row = yy * ang_stride;
        var xx: u32 = 0;
        while (xx < padded_w) : (xx += vec_len) {
            var xc: i32v = undefined;
            inline for (0..vec_len) |i| {
                xc[i] = @as(i32, @intCast(xx + i)) - @as(i32, @intCast(PAD));
            }
            buf[row + xx ..][0..vec_len].* = calculateGradientAngle(src, stride, width, height, yc, xc, 20);
        }
    }
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
