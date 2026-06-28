const std = @import("std");
const math = std.math;
const vcl = @import("../vcl.zig");

const allocator = std.heap.c_allocator;

const ksize = 9;
const radius = 4;
const vec_size: comptime_int = std.simd.suggestVectorLength(f32) orelse 8;
const weight_pruning: f64 = 0.01;

const SkipInfo = struct {
    ssim: bool,
    artifact: bool,
    detailloss: bool,

    fn all(self: SkipInfo) bool {
        return self.ssim and self.artifact and self.detailloss;
    }
};

const skip_table: [3][6]SkipInfo = blk: {
    var table: [3][6]SkipInfo = undefined;
    var plane: usize = 0;
    while (plane < 3) : (plane += 1) {
        var scale: usize = 0;
        while (scale < 6) : (scale += 1) {
            const base = plane * 36 + scale * 6;
            table[plane][scale] = .{
                .ssim = weight[base + 0] <= weight_pruning and weight[base + 3] <= weight_pruning,
                .artifact = weight[base + 1] <= weight_pruning and weight[base + 4] <= weight_pruning,
                .detailloss = weight[base + 2] <= weight_pruning and weight[base + 5] <= weight_pruning,
            };
        }
    }
    break :blk table;
};

/// Scratch buffer size (in f32 elements) needed by `process` for a given plane geometry.
pub fn scratchSize(stride: u32, w: u32, h: u32) usize {
    const wh: usize = @as(usize, stride) * h;
    const b_size: usize = @as(usize, (stride + 1) / 2) * ((h + 1) / 2);
    return 6 * b_size + 10 * wh + w + 64;
}

pub fn process(srcp1: [3][]const f32, srcp2: [3][]const f32, stride: u32, w: u32, h: u32, scratch: []f32) f64 {
    const wh: u32 = stride * h;
    // The downscale chain buffers only ever hold scale >= 1 (half resolution).
    const b_size: u32 = ((stride + 1) / 2) * ((h + 1) / 2);

    const srcp1b = [3][]f32{ scratch[0 * b_size .. 1 * b_size], scratch[1 * b_size .. 2 * b_size], scratch[2 * b_size .. 3 * b_size] };
    const srcp2b = [3][]f32{ scratch[3 * b_size .. 4 * b_size], scratch[4 * b_size .. 5 * b_size], scratch[5 * b_size .. 6 * b_size] };
    const rest = scratch[6 * b_size ..];
    const tmpp1 = [3][]f32{ rest[0 * wh .. 1 * wh], rest[1 * wh .. 2 * wh], rest[2 * wh .. 3 * wh] };
    const tmpp2 = [3][]f32{ rest[3 * wh .. 4 * wh], rest[4 * wh .. 5 * wh], rest[5 * wh .. 6 * wh] };
    const tmpp3 = rest[6 * wh .. 7 * wh];
    const tmpsq = rest[7 * wh .. 8 * wh];
    const tmpps12 = rest[8 * wh .. 9 * wh];
    const tmppmu1 = rest[9 * wh .. 10 * wh];
    const blur_tmp = rest[10 * wh ..][0 .. w + 64];

    var plane_avg_ssim: [6][6]f64 = undefined;
    var plane_avg_edge: [6][12]f64 = undefined;
    var stride2 = stride;
    var w2 = w;
    var h2 = h;

    var scale: u32 = 0;
    while (scale < 6) : (scale += 1) {
        if (scale > 0) {
            if (scale == 1) {
                downscale(srcp1, srcp1b, stride2, w2, h2);
                downscale(srcp2, srcp2b, stride2, w2, h2);
            } else {
                downscale(srcp1b, srcp1b, stride2, w2, h2);
                downscale(srcp2b, srcp2b, stride2, w2, h2);
            }
            stride2 = @divTrunc((stride2 + 1), 2);
            w2 = @divTrunc((w2 + 1), 2);
            h2 = @divTrunc((h2 + 1), 2);
        }

        const one_per_pixels: f64 = 1.0 / @as(f64, @floatFromInt(w2 * h2));
        if (scale == 0) {
            toXYB(srcp1, tmpp1, stride2, w2, h2);
            toXYB(srcp2, tmpp2, stride2, w2, h2);
        } else {
            toXYB(srcp1b, tmpp1, stride2, w2, h2);
            toXYB(srcp2b, tmpp2, stride2, w2, h2);
        }

        var plane: u32 = 0;
        while (plane < 3) : (plane += 1) {
            const skip = skip_table[plane][scale];

            if (skip.all()) {
                plane_avg_ssim[scale][plane * 2] = 0.0;
                plane_avg_ssim[scale][plane * 2 + 1] = 0.0;
                plane_avg_edge[scale][plane * 4 + 0] = 0.0;
                plane_avg_edge[scale][plane * 4 + 1] = 0.0;
                plane_avg_edge[scale][plane * 4 + 2] = 0.0;
                plane_avg_edge[scale][plane * 4 + 3] = 0.0;
                continue;
            }

            if (!skip.ssim) {
                multiply(tmpp1[plane], tmpp2[plane], tmpp3, stride2, w2, h2);
                blur(tmpp3, tmpps12, stride2, w2, h2, blur_tmp);

                addSquare(tmpp1[plane], tmpp2[plane], tmpp3, stride2, w2, h2);
                blur(tmpp3, tmpsq, stride2, w2, h2, blur_tmp);
            }

            blur(tmpp1[plane], tmppmu1, stride2, w2, h2, blur_tmp);
            blur(tmpp2[plane], tmpp3, stride2, w2, h2, blur_tmp);

            if (!skip.ssim) {
                ssimMap(tmpsq, tmpps12, tmppmu1, tmpp3, stride2, w2, h2, plane, one_per_pixels, &plane_avg_ssim[scale]);
            } else {
                plane_avg_ssim[scale][plane * 2] = 0.0;
                plane_avg_ssim[scale][plane * 2 + 1] = 0.0;
            }

            if (!skip.artifact or !skip.detailloss) {
                edgeMap(tmpp1[plane], tmpp2[plane], tmppmu1, tmpp3, stride2, w2, h2, plane, one_per_pixels, &plane_avg_edge[scale]);
            } else {
                plane_avg_edge[scale][plane * 4 + 0] = 0.0;
                plane_avg_edge[scale][plane * 4 + 1] = 0.0;
                plane_avg_edge[scale][plane * 4 + 2] = 0.0;
                plane_avg_edge[scale][plane * 4 + 3] = 0.0;
            }
        }
    }

    return score(plane_avg_ssim, plane_avg_edge);
}

fn downscale(src: [3][]const f32, dst: [3][]f32, src_stride: u32, in_w: u32, in_h: u32) void {
    const fscale: f32 = 2.0;
    const uscale: u32 = 2;
    const out_w = @divTrunc((in_w + uscale - 1), uscale);
    const out_h = @divTrunc((in_h + uscale - 1), uscale);
    const dst_stride = @divTrunc((src_stride + uscale - 1), uscale);
    const normalize: f32 = 1.0 / (fscale * fscale);

    const FV = @Vector(vec_size, f32);
    const normv: FV = @splat(normalize);
    // deinterleave masks: even/odd elements of two consecutive vectors
    const even_mask = comptime blk: {
        var m: [vec_size]i32 = undefined;
        for (0..vec_size) |i| {
            m[i] = if (i < vec_size / 2) @intCast(2 * i) else ~@as(i32, @intCast(2 * (i - vec_size / 2)));
        }
        break :blk m;
    };
    const odd_mask = comptime blk: {
        var m: [vec_size]i32 = undefined;
        for (0..vec_size) |i| {
            m[i] = if (i < vec_size / 2) @intCast(2 * i + 1) else ~@as(i32, @intCast(2 * (i - vec_size / 2) + 1));
        }
        break :blk m;
    };

    // ox values whose 2x2 window needs no edge clamping
    const full_w: u32 = in_w / 2;

    var plane: u32 = 0;
    while (plane < 3) : (plane += 1) {
        const srcp = src[plane];
        var dstp = dst[plane];
        var oy: u32 = 0;
        while (oy < out_h) : (oy += 1) {
            var ox: u32 = 0;

            if (oy * 2 + 1 < in_h) {
                const row0 = srcp[(oy * 2) * src_stride ..];
                const row1 = srcp[(oy * 2 + 1) * src_stride ..];
                while (ox + vec_size <= full_w) : (ox += vec_size) {
                    const a0: FV = row0[ox * 2 ..][0..vec_size].*;
                    const b0: FV = row0[ox * 2 + vec_size ..][0..vec_size].*;
                    const a1: FV = row1[ox * 2 ..][0..vec_size].*;
                    const b1: FV = row1[ox * 2 + vec_size ..][0..vec_size].*;
                    const e0 = @shuffle(f32, a0, b0, even_mask);
                    const o0 = @shuffle(f32, a0, b0, odd_mask);
                    const e1 = @shuffle(f32, a1, b1, even_mask);
                    const o1 = @shuffle(f32, a1, b1, odd_mask);
                    // same add order as the scalar 2x2 loop
                    const sum = ((e0 + o0) + e1) + o1;
                    dstp[ox..][0..vec_size].* = sum * normv;
                }
            }

            while (ox < out_w) : (ox += 1) {
                var sum: f32 = 0.0;
                var iy: u32 = 0;
                while (iy < uscale) : (iy += 1) {
                    var ix: u32 = 0;
                    while (ix < uscale) : (ix += 1) {
                        const x: u32 = @min((ox * uscale + ix), (in_w - 1));
                        const y: u32 = @min((oy * uscale + iy), (in_h - 1));
                        sum += srcp[y * src_stride + x];
                    }
                }
                dstp[ox] = sum * normalize;
            }
            dstp = dstp[dst_stride..];
        }
    }
}

fn multiply(src1: []const f32, src2: []const f32, dst: []f32, stride: u32, w: u32, h: u32) void {
    const Vec = @Vector(vec_size, f32);
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const off = y * stride;
        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const a: Vec = src1[off + x ..][0..vec_size].*;
            const b: Vec = src2[off + x ..][0..vec_size].*;
            dst[off + x ..][0..vec_size].* = a * b;
        }
        while (x < w) : (x += 1) {
            dst[off + x] = src1[off + x] * src2[off + x];
        }
    }
}

fn addSquare(src1: []const f32, src2: []const f32, dst: []f32, stride: u32, w: u32, h: u32) void {
    const Vec = @Vector(vec_size, f32);
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const off = y * stride;
        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const a: Vec = src1[off + x ..][0..vec_size].*;
            const b: Vec = src2[off + x ..][0..vec_size].*;
            const v = a + b;
            dst[off + x ..][0..vec_size].* = v * v;
        }
        while (x < w) : (x += 1) {
            const v = src1[off + x] + src2[off + x];
            dst[off + x] = v * v;
        }
    }
}

inline fn blurH(srcp: []f32, dstp: []f32, kernel: [ksize]f32, w: i32) void {
    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = radius;
    const center_end: i32 = w - @min(w, radius);
    // srcp is the padded blur row buffer, so vector loads past w are safe
    while (j + vec_size <= center_end) : (j += vec_size) {
        const base: u32 = @intCast(j - radius);
        var acc: @Vector(vec_size, f32) = @splat(0.0);
        inline for (0..ksize) |k| {
            const kv: @Vector(vec_size, f32) = @splat(kernel[k]);
            const sv: @Vector(vec_size, f32) = srcp[base + k ..][0..vec_size].*;
            acc = acc + kv * sv;
        }
        dstp[@intCast(j)..][0..vec_size].* = acc;
    }
    while (j < center_end) : (j += 1) {
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += kernel[@intCast(k)] * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = sum;
    }
}

inline fn blurV(src: anytype, dstp: []f32, kernel: [ksize]f32, w: u32) void {
    const Vec = @Vector(vec_size, f32);
    var j: u32 = 0;
    while (j + vec_size <= w) : (j += vec_size) {
        var accum: Vec = @splat(0.0);
        inline for (0..ksize) |k| {
            const kv: Vec = @splat(kernel[k]);
            const sv: Vec = src[k][j..][0..vec_size].*;
            accum = @mulAdd(Vec, kv, sv, accum);
        }
        dstp[j..][0..vec_size].* = accum;
    }
    while (j < w) : (j += 1) {
        var accum: f32 = 0.0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            accum += kernel[k] * src[k][j];
        }
        dstp[j] = accum;
    }
}

fn blur(src: []const f32, dst: []f32, stride: u32, w: u32, h: u32, tmp_buf: []f32) void {
    const kernel = [ksize]f32{
        0.0076144188642501831054687500,
        0.0360749699175357818603515625,
        0.1095860823988914489746093750,
        0.2134445458650588989257812500,
        0.2665599882602691650390625000,
        0.2134445458650588989257812500,
        0.1095860823988914489746093750,
        0.0360749699175357818603515625,
        0.0076144188642501831054687500,
    };

    const tmp = tmp_buf[0..w];

    var i: i32 = 0;
    const ih: i32 = @bitCast(h);
    while (i < h) : (i += 1) {
        const ui: u32 = @bitCast(i);
        var srcp: [ksize][]const f32 = undefined;
        const dstp: []f32 = dst[(ui * stride)..];
        const dist_from_bottom: i32 = ih - 1 - i;

        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const row: i32 = if (i < radius - k) (@min(radius - k - i, ih - 1)) else (i - radius + k);
            const urow: u32 = @bitCast(row);
            srcp[@intCast(k)] = src[(urow * stride)..];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const row: i32 = if (dist_from_bottom < k - radius) (i - @min(k - radius - dist_from_bottom, i)) else (i - radius + k);
            const urow: u32 = @bitCast(row);
            srcp[@intCast(k)] = src[(urow * stride)..];
        }

        blurV(srcp, tmp, kernel, w);
        blurH(tmp, dstp, kernel, @intCast(w));
    }
}

const K_D0: f32 = 0.0037930734;
const K_D1: f32 = math.cbrt(K_D0);

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;

const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;

const K_M20: f32 = 0.24342269;
const K_M21: f32 = 0.20476745;
const K_M22: f32 = 1.0 - K_M20 - K_M21;

const OPSIN_ABSORBANCE_MATRIX = [9]f32{ K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22 };
const OPSIN_ABSORBANCE_BIAS: f32 = K_D0;

fn toXYB(_srcp: [3][]const f32, _dstp: [3][]f32, stride: u32, w: u32, h: u32) void {
    const FV = @Vector(vec_size, f32);
    const zero: FV = @splat(0.0);
    const bias: FV = @splat(OPSIN_ABSORBANCE_BIAS);
    const kd1: FV = @splat(K_D1);
    const half: FV = @splat(0.5);
    const c14: FV = @splat(14.0);
    const c042: FV = @splat(0.42);
    const c055: FV = @splat(0.55);
    const c001: FV = @splat(0.01);
    const m00: FV = @splat(OPSIN_ABSORBANCE_MATRIX[0]);
    const m01: FV = @splat(OPSIN_ABSORBANCE_MATRIX[1]);
    const m02: FV = @splat(OPSIN_ABSORBANCE_MATRIX[2]);
    const m10: FV = @splat(OPSIN_ABSORBANCE_MATRIX[3]);
    const m11: FV = @splat(OPSIN_ABSORBANCE_MATRIX[4]);
    const m12: FV = @splat(OPSIN_ABSORBANCE_MATRIX[5]);
    const m20: FV = @splat(OPSIN_ABSORBANCE_MATRIX[6]);
    const m21: FV = @splat(OPSIN_ABSORBANCE_MATRIX[7]);
    const m22: FV = @splat(OPSIN_ABSORBANCE_MATRIX[8]);

    var srcp = _srcp;
    var dstp = _dstp;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        if (w >= vec_size) {
            // the final vector backsteps to w - vec_size (overlapping recompute)
            // so every pixel takes the identical SIMD path
            var x: u32 = 0;
            while (true) {
                if (x + vec_size > w) x = w - vec_size;

                const r: FV = srcp[0][x..][0..vec_size].*;
                const g: FV = srcp[1][x..][0..vec_size].*;
                const b: FV = srcp[2][x..][0..vec_size].*;

                const ox = @mulAdd(FV, m00, r, @mulAdd(FV, m01, g, @mulAdd(FV, m02, b, bias)));
                const oy = @mulAdd(FV, m10, r, @mulAdd(FV, m11, g, @mulAdd(FV, m12, b, bias)));
                const oz = @mulAdd(FV, m20, r, @mulAdd(FV, m21, g, @mulAdd(FV, m22, b, bias)));
                const cx = vcl.cbrt(@max(ox, zero)) - kd1;
                const cy = vcl.cbrt(@max(oy, zero)) - kd1;
                const cz = vcl.cbrt(@max(oz, zero)) - kd1;

                const xv = half * (cx - cy);
                const yv = half * (cx + cy);
                dstp[0][x..][0..vec_size].* = xv * c14 + c042;
                dstp[1][x..][0..vec_size].* = yv + c001;
                dstp[2][x..][0..vec_size].* = (cz - yv) + c055;

                if (x + vec_size >= w) break;
                x += vec_size;
            }
        } else {
            // degenerate width: single-lane vectors keep cbrt rounding identical
            var x: u32 = 0;
            while (x < w) : (x += 1) {
                const F1 = @Vector(1, f32);
                const r: F1 = @splat(srcp[0][x]);
                const g: F1 = @splat(srcp[1][x]);
                const b: F1 = @splat(srcp[2][x]);
                const b1: F1 = @splat(OPSIN_ABSORBANCE_BIAS);
                const ox = @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[0]), r, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[1]), g, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[2]), b, b1)));
                const oy = @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[3]), r, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[4]), g, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[5]), b, b1)));
                const oz = @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[6]), r, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[7]), g, @mulAdd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[8]), b, b1)));
                const cx = vcl.cbrt(@max(ox, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const cy = vcl.cbrt(@max(oy, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const cz = vcl.cbrt(@max(oz, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const xs = 0.5 * (cx[0] - cy[0]);
                const ys = 0.5 * (cx[0] + cy[0]);
                dstp[0][x] = xs * 14.0 + 0.42;
                dstp[1][x] = ys + 0.01;
                dstp[2][x] = (cz[0] - ys) + 0.55;
            }
        }

        var i: u32 = 0;
        while (i < 3) : (i += 1) {
            srcp[i] = srcp[i][stride..];
            dstp[i] = dstp[i][stride..];
        }
    }
}

fn tothe4th(y: f64) f64 {
    var x = y * y;
    x *= x;
    return x;
}

fn ssimMap(
    sumsquared: []f32,
    s12: []f32,
    mu1: []f32,
    mu2: []f32,
    stride: u32,
    w: u32,
    h: u32,
    plane: u32,
    one_per_pixels: f64,
    plane_avg_ssim: []f64,
) void {
    const FV = @Vector(vec_size, f32);
    const DV = @Vector(vec_size, f64);
    const onef: FV = @splat(1.0);
    const twof: FV = @splat(2.0);
    const e9f: FV = @splat(0.0009);
    const oned: DV = @splat(1.0);
    const zerod: DV = @splat(0.0);

    var sum0v: DV = @splat(0.0);
    var sum1v: DV = @splat(0.0);
    var sum1 = [2]f64{ 0.0, 0.0 };
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const sqp = sumsquared[(y * stride)..];
        const s12p = s12[(y * stride)..];
        const mu1p = mu1[(y * stride)..];
        const mu2p = mu2[(y * stride)..];

        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const m1: FV = mu1p[x..][0..vec_size].*;
            const m2: FV = mu2p[x..][0..vec_size].*;
            const s12v: FV = s12p[x..][0..vec_size].*;
            const sqv: FV = sqp[x..][0..vec_size].*;
            const m11 = m1 * m1;
            const m22 = m2 * m2;
            const m12 = m1 * m2;
            const m_diff = m1 - m2;
            const num_m: DV = @floatCast(@mulAdd(FV, m_diff, -m_diff, onef));
            const num_s: DV = @floatCast(@mulAdd(FV, (s12v - m12), twof, e9f));
            const denom_s: DV = @floatCast(sqv - twof * s12v - m11 - m22 + e9f);
            const d1: DV = @max(oned - ((num_m * num_s) / denom_s), zerod);

            sum0v += d1;
            var t4 = d1 * d1;
            t4 *= t4;
            sum1v += t4;
        }

        while (x < w) : (x += 1) {
            const m1: f32 = mu1p[x];
            const m2: f32 = mu2p[x];
            const m11 = m1 * m1;
            const m22 = m2 * m2;
            const m12 = m1 * m2;
            const m_diff = m1 - m2;
            const num_m: f64 = @mulAdd(f32, m_diff, -m_diff, 1.0);
            const num_s: f64 = @mulAdd(f32, (s12p[x] - m12), 2.0, 0.0009);
            const denom_s: f64 = (sqp[x] - 2.0 * s12p[x] - m11 - m22 + 0.0009);
            const d1: f64 = @max(1.0 - ((num_m * num_s) / denom_s), 0.0);

            sum1[0] += d1;
            sum1[1] += tothe4th(d1);
        }
    }

    sum1[0] += @reduce(.Add, sum0v);
    sum1[1] += @reduce(.Add, sum1v);

    plane_avg_ssim[plane * 2] = one_per_pixels * sum1[0];
    plane_avg_ssim[plane * 2 + 1] = @sqrt(@sqrt(one_per_pixels * sum1[1]));
}

fn edgeMap(
    im1: []f32,
    im2: []f32,
    mu1: []f32,
    mu2: []f32,
    stride: u32,
    w: u32,
    h: u32,
    plane: u32,
    one_per_pixels: f64,
    plane_avg_edge: []f64,
) void {
    const FV = @Vector(vec_size, f32);
    const DV = @Vector(vec_size, f64);
    const oned: DV = @splat(1.0);
    const zerod: DV = @splat(0.0);

    var art0: DV = @splat(0.0);
    var art1: DV = @splat(0.0);
    var det0: DV = @splat(0.0);
    var det1: DV = @splat(0.0);
    var sum2 = [4]f64{ 0.0, 0.0, 0.0, 0.0 };
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const im1p = im1[(y * stride)..];
        const im2p = im2[(y * stride)..];
        const mu1p = mu1[(y * stride)..];
        const mu2p = mu2[(y * stride)..];

        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const v1: FV = im1p[x..][0..vec_size].*;
            const v2: FV = im2p[x..][0..vec_size].*;
            const m1: FV = mu1p[x..][0..vec_size].*;
            const m2: FV = mu2p[x..][0..vec_size].*;
            const n2: DV = @floatCast(@abs(v2 - m2));
            const n1: DV = @floatCast(@abs(v1 - m1));
            const d1: DV = (oned + n2) / (oned + n1) - oned;

            const artifact = @max(d1, zerod);
            art0 += artifact;
            var a4 = artifact * artifact;
            a4 *= a4;
            art1 += a4;

            const detail_lost = @max(-d1, zerod);
            det0 += detail_lost;
            var l4 = detail_lost * detail_lost;
            l4 *= l4;
            det1 += l4;
        }

        while (x < w) : (x += 1) {
            const d1: f64 = (1.0 + @as(f64, @abs(im2p[x] - mu2p[x]))) /
                (1.0 + @as(f64, @abs(im1p[x] - mu1p[x]))) - 1.0;
            const artifact: f64 = @max(d1, 0.0);
            sum2[0] += artifact;
            sum2[1] += tothe4th(artifact);
            const detail_lost: f64 = @max(-d1, 0.0);
            sum2[2] += detail_lost;
            sum2[3] += tothe4th(detail_lost);
        }
    }

    sum2[0] += @reduce(.Add, art0);
    sum2[1] += @reduce(.Add, art1);
    sum2[2] += @reduce(.Add, det0);
    sum2[3] += @reduce(.Add, det1);

    plane_avg_edge[plane * 4] = one_per_pixels * sum2[0];
    plane_avg_edge[plane * 4 + 1] = @sqrt(@sqrt(one_per_pixels * sum2[1]));
    plane_avg_edge[plane * 4 + 2] = one_per_pixels * sum2[2];
    plane_avg_edge[plane * 4 + 3] = @sqrt(@sqrt(one_per_pixels * sum2[3]));
}

fn score(plane_avg_ssim: [6][6]f64, plane_avg_edge: [6][12]f64) f64 {
    var ssim: f64 = 0.0;

    var i: u32 = 0;
    var plane: u32 = 0;

    while (plane < 3) : (plane += 1) {
        var s: u32 = 0;
        while (s < 6) : (s += 1) {
            var n: u32 = 0;
            while (n < 2) : (n += 1) {
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_ssim[s][plane * 2 + n]), ssim);
                i += 1;
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n]), ssim);
                i += 1;
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n + 2]), ssim);
                i += 1;
            }
        }
    }

    ssim *= 0.9562382616834844;
    ssim = (6.248496625763138e-5 * ssim * ssim) * ssim +
        2.326765642916932 * ssim -
        0.020884521182843837 * ssim * ssim;

    if (ssim > 0.0) {
        ssim = math.pow(f64, ssim, 0.6276336467831387) * -10.0 + 100.0;
    } else {
        ssim = 100.0;
    }

    return ssim;
}

const weight = [108]f64{
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
};
