const std = @import("std");
const math = std.math;

const hz = @import("../helper.zig");
const Data = @import("../vapoursynth/bilateral.zig").Data;
const vszip = @import("../vszip.zig");

const allocator = std.heap.c_allocator;
const AlignedF32 = []align(vszip.vec_len) f32;

const VLEN = @max(vszip.vec_len / @sizeOf(f32), 1);
const Vf = @Vector(VLEN, f32);
const Vu = @Vector(VLEN, u32);

inline fn rangeIndex(comptime T: type, a: T, b: T) u32 {
    if (@typeInfo(T) == .float) {
        const ad: f32 = @abs(a - b);
        return @intFromFloat(@trunc(@min(1.0, ad) * 65535 + 0.5));
    } else {
        return hz.absDiff(a, b);
    }
}

/// Pixel value as the accumulation float type (T for float clips, f32 for int).
inline fn valOf(comptime T: type, v: T) (if (@typeInfo(T) == .float) T else f32) {
    return if (@typeInfo(T) == .float) v else @floatFromInt(v);
}

/// Final write: round+clip for integer output, raw cast for float output.
inline fn finalize(comptime T: type, sum: anytype, wsum: anytype, peak: f32) T {
    if (@typeInfo(T) == .float) {
        return @floatCast(sum / wsum);
    } else {
        return @trunc(math.clamp(sum / wsum + 0.5, 0.0, peak));
    }
}

// --- VLEN-wide vector helpers for the alg2 (truncated) interior --------------

inline fn Vec(comptime T: type) type {
    return @Vector(VLEN, T);
}

/// VLEN pixels as the accumulation float vector (matches the scalar valOf: for
/// f16 the value is widened to f32 exactly, same as f16*f32 in scalar).
inline fn vToF(comptime T: type, v: Vec(T)) Vf {
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

/// Per-lane GR_LUT gather of the range weight for VLEN pixels — same index math
/// as the scalar `rangeIndex` (the lane loop may lower to a hardware gather).
inline fn vRangeGather(comptime T: type, gr_lut: []const f32, cx: Vec(T), nb: Vec(T)) Vf {
    var idx: Vu = undefined;
    if (@typeInfo(T) == .float) {
        const ad: Vf = @floatCast(@abs(cx - nb)); // |a-b| in T, widened (== scalar)
        idx = @intFromFloat(@trunc(@min(@as(Vf, @splat(1.0)), ad) * @as(Vf, @splat(65535.0)) + @as(Vf, @splat(0.5))));
    } else {
        const a: Vu = @intCast(cx);
        const b: Vu = @intCast(nb);
        // |a-b| via max-min: a single non-underflowing subtraction. (A
        // @select(a-b, b-a) would evaluate BOTH branches, one of which always
        // underflows u32 — fine when it wraps in release, but a Debug panic.)
        idx = @max(a, b) - @min(a, b);
    }
    var w: Vf = undefined;
    inline for (0..VLEN) |i| w[i] = gr_lut[idx[i]];
    return w;
}

/// Vectorized final write — equivalent to `finalize` applied per lane.
inline fn vFinalize(comptime T: type, sum: Vf, wsum: Vf, peakv: Vf) Vec(T) {
    const r = sum / wsum;
    if (@typeInfo(T) == .float) {
        return @floatCast(r);
    } else {
        const c = @min(@max(r + @as(Vf, @splat(0.5)), @as(Vf, @splat(0.0))), peakv);
        return @intFromFloat(@trunc(c));
    }
}

pub fn bilateral(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, w: u32, h: u32, plane: u32, comptime join: bool, d: *Data) void {
    _ = join; // refp already aliases srcp when there is no joint clip
    if (d.algorithm[plane] == 1) {
        pbfic(T, srcp, refp, dstp, stride, w, h, plane, d);
    } else {
        truncated(T, srcp, refp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane], d.peak);
    }
}

// O(1) cross/joint bilateral (PBFIC) — "Real-Time O(1) Bilateral Filtering", Yang et al.
fn pbfic(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, width: u32, height: u32, plane: u32, d: *Data) void {
    const is_float = @typeInfo(T) == .float;
    const num = d.PBFICnum[plane];
    const pcount: u32 = stride * height;
    const gr_lut = d.gr_lut[plane];

    const pbfick = allocator.alloc(T, num) catch unreachable;
    defer allocator.free(pbfick);
    if (is_float) {
        const denom: T = @floatFromInt(num - 1);
        for (pbfick, 0..) |*v, k| v.* = @as(T, @floatFromInt(k)) / denom;
    } else {
        const numf: f32 = @floatFromInt(num);
        for (pbfick, 0..) |*v, k| {
            const kf: f32 = @floatFromInt(k);
            v.* = math.lossyCast(T, d.peak * kf / (numf - 1) + 0.5);
        }
    }

    var b: f32 = undefined;
    var b1: f32 = undefined;
    var b2: f32 = undefined;
    var b3: f32 = undefined;
    recursiveGaussianParameters(d.sigmaS[plane], &b, &b1, &b2, &b3);

    // One contiguous backing buffer for all PBFICs (num * pcount), plus Wk/Jk scratch.
    const pbfic_buf: AlignedF32 = allocator.alignedAlloc(f32, vszip.alignment, num * pcount) catch unreachable;
    defer allocator.free(pbfic_buf);
    const wk: AlignedF32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    const jk: AlignedF32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    defer allocator.free(wk);
    defer allocator.free(jk);

    for (0..num) |k| {
        const pbfic_k = pbfic_buf[k * pcount ..][0..pcount];
        const pk = pbfick[k];
        // Wk = range weight at this PBFIC level; Jk = Wk*src. (Kept scalar: alg1
        // is bound by the recursive-Gaussian passes, so vectorizing this builds
        // measured no faster.)
        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                wk[i] = gr_lut[rangeIndex(T, pk, refp[i])];
                jk[i] = wk[i] * valOf(T, srcp[i]);
            }
        }

        recursiveGaussian2DHorizontal(wk, wk, height, width, stride, b, b1, b2, b3);
        recursiveGaussian2DVertical(wk, wk, height, width, stride, b, b1, b2, b3);
        recursiveGaussian2DHorizontal(jk, jk, height, width, stride, b, b1, b2, b3);
        recursiveGaussian2DVertical(jk, jk, height, width, stride, b, b1, b2, b3);

        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                pbfic_k[i] = if (wk[i] == 0) 0 else (jk[i] / wk[i]);
            }
        }
    }

    for (0..height) |j| {
        var i = stride * j;
        const upper = i + width;
        while (i < upper) : (i += 1) {
            var k: u32 = 0;
            while (k < (num - 2)) : (k += 1) {
                if ((refp[i] < pbfick[k + 1]) and (refp[i] >= pbfick[k])) break;
            }

            const rf: f32 = valOf(T, refp[i]);
            const p0f: f32 = valOf(T, pbfick[k]);
            const p1f: f32 = valOf(T, pbfick[k + 1]);
            const lo = pbfic_buf[k * pcount + i];
            const hi = pbfic_buf[(k + 1) * pcount + i];
            const vf: f32 = ((p1f - rf) * lo + (rf - p0f) * hi) / (p1f - p0f);
            dstp[i] = finalize(T, vf, @as(f32, 1.0), d.peak);
        }
    }
}

// Bilateral with truncated spatial window and sub-sampling. `ref` is the range
// source (== src when not joint); `src` provides the values being averaged.
// The interior is processed VLEN output pixels at a time along x; the column
// remainder and the 4 edge bands are scalar. Accumulation order is identical to
// the scalar path, so the SIMD output is bit-identical.
fn truncated(comptime T: type, src: []const T, ref: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius: u32, step: u32, peak: f32) void {
    const radius2 = radius + 1;
    const w0 = gs_lut[0] * gr_lut[0];
    const peakv: Vf = @splat(peak);
    const xend = width - radius;

    var y: u32 = radius;
    while (y < height - radius) : (y += 1) {
        const ys = y * stride;

        var x: u32 = radius;
        while (x + VLEN <= xend) : (x += VLEN) {
            const base = ys + x;
            const cxv: Vec(T) = ref[base..][0..VLEN].*;
            var wsum: Vf = @splat(w0);
            var sum: Vf = vToF(T, @as(Vec(T), src[base..][0..VLEN].*)) * wsum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys = yy * stride;
                const a_off = ys - yys; // row above
                const b_off = ys + yys; // row below

                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const swei: Vf = @splat(gs_lut[yy * radius2 + xx]);
                    const xp = x + xx;
                    const xm = x - xx;
                    const rw1 = vRangeGather(T, gr_lut, cxv, ref[a_off + xp ..][0..VLEN].*);
                    const rw2 = vRangeGather(T, gr_lut, cxv, ref[b_off + xp ..][0..VLEN].*);
                    const rw3 = vRangeGather(T, gr_lut, cxv, ref[a_off + xm ..][0..VLEN].*);
                    const rw4 = vRangeGather(T, gr_lut, cxv, ref[b_off + xm ..][0..VLEN].*);
                    wsum += swei * (rw1 + rw2 + rw3 + rw4);
                    const s1 = vToF(T, @as(Vec(T), src[a_off + xp ..][0..VLEN].*));
                    const s2 = vToF(T, @as(Vec(T), src[b_off + xp ..][0..VLEN].*));
                    const s3 = vToF(T, @as(Vec(T), src[a_off + xm ..][0..VLEN].*));
                    const s4 = vToF(T, @as(Vec(T), src[b_off + xm ..][0..VLEN].*));
                    sum += swei * (s1 * rw1 + s2 * rw2 + s3 * rw3 + s4 * rw4);
                }
            }

            dst[base..][0..VLEN].* = vFinalize(T, sum, wsum, peakv);
        }

        // scalar column remainder (still interior, no clamping needed)
        while (x < xend) : (x += 1) {
            truncatedPixel(T, src, ref, dst, gs_lut, gr_lut, stride, ys, x, radius2, step, peak);
        }
    }

    // The 4 edge bands clamp out-of-frame reads (== C++ replicate padding).
    truncatedEdges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, 0, 0, radius, width);
    truncatedEdges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, height - radius, 0, height, width);
    truncatedEdges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, 0, height - radius, radius);
    truncatedEdges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, width - radius, height - radius, width);
}

// One interior output pixel (no edge clamping) — used for the SIMD remainder.
inline fn truncatedPixel(comptime T: type, src: []const T, ref: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, ys: u32, x: u32, radius2: u32, step: u32, peak: f32) void {
    const xy = x + ys;
    const cx: T = ref[xy];
    var weight_sum = gs_lut[0] * gr_lut[0];
    var sum = valOf(T, src[xy]) * weight_sum;

    var yy: u32 = 1;
    while (yy < radius2) : (yy += step) {
        const yys = yy * stride;
        const la = src[(ys - yys)..];
        const lb = src[(ys + yys)..];
        const lar = ref[(ys - yys)..];
        const lbr = ref[(ys + yys)..];

        var xx: u32 = 1;
        while (xx < radius2) : (xx += step) {
            const swei = gs_lut[yy * radius2 + xx];
            const rw1 = gr_lut[rangeIndex(T, cx, lar[x + xx])];
            const rw2 = gr_lut[rangeIndex(T, cx, lbr[x + xx])];
            const rw3 = gr_lut[rangeIndex(T, cx, lar[x - xx])];
            const rw4 = gr_lut[rangeIndex(T, cx, lbr[x - xx])];
            weight_sum += swei * (rw1 + rw2 + rw3 + rw4);
            sum += swei * (valOf(T, la[x + xx]) * rw1 + valOf(T, lb[x + xx]) * rw2 +
                valOf(T, la[x - xx]) * rw3 + valOf(T, lb[x - xx]) * rw4);
        }
    }

    dst[xy] = finalize(T, sum, weight_sum, peak);
}

fn truncatedEdges(comptime T: type, dst: []T, src: []const T, ref: []const T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius2: u32, step: u32, peak: f32, y_start: u32, x_start: u32, y_end: u32, x_end: u32) void {
    const max_line = stride * (height - 1);
    var y: u32 = y_start;
    while (y < y_end) : (y += 1) {
        var x: u32 = x_start;
        while (x < x_end) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = ref[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = valOf(T, src[xy]) * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys = yy * stride;
                const la = src[(ys -| yys)..];
                const lb = src[@min(ys + yys, max_line)..];
                const lar = ref[(ys -| yys)..];
                const lbr = ref[@min(ys + yys, max_line)..];

                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const xa = @min(x + xx, width - 1);
                    const xb = x -| xx;
                    const swei = gs_lut[yy * radius2 + xx];
                    const rw1 = gr_lut[rangeIndex(T, cx, lar[xa])];
                    const rw2 = gr_lut[rangeIndex(T, cx, lbr[xa])];
                    const rw3 = gr_lut[rangeIndex(T, cx, lar[xb])];
                    const rw4 = gr_lut[rangeIndex(T, cx, lbr[xb])];
                    weight_sum += swei * (rw1 + rw2 + rw3 + rw4);
                    sum += swei * (valOf(T, la[xa]) * rw1 + valOf(T, lb[xa]) * rw2 +
                        valOf(T, la[xb]) * rw3 + valOf(T, lb[xb]) * rw4);
                }
            }

            dst[xy] = finalize(T, sum, weight_sum, peak);
        }
    }
}

pub fn gaussianFunctionSpatialLUTGeneration(gs_lut: []f32, upper: u32, sigmaS: f64) void {
    var y: u32 = 0;
    while (y < upper) : (y += 1) {
        var x: u32 = 0;
        while (x < upper) : (x += 1) {
            gs_lut[y * upper + x] = math.lossyCast(f32, @exp(@as(f64, @floatFromInt(x * x + y * y)) / (sigmaS * sigmaS * -2.0)));
        }
    }
}

pub fn gaussianFunctionRangeLUTGeneration(gr_lut: []f32, range: f64, sigmaR: f64) void {
    const upper: u32 = @trunc(@min(range, (sigmaR * 8.0 * range + 0.5)));

    var i: u32 = 0;
    while (i <= upper) : (i += 1) {
        const j: f64 = @as(f64, @floatFromInt(i)) / range;
        gr_lut[i] = math.lossyCast(f32, normalizedGaussianFunction(j, sigmaR));
    }

    if (i < gr_lut.len) {
        const upper_value: f32 = gr_lut[upper];
        while (i < gr_lut.len) : (i += 1) gr_lut[i] = upper_value;
    }
}

fn normalizedGaussianFunction(y: f64, sigma: f64) f64 {
    const x = y / sigma;
    return @exp(x * x / -2) / (math.sqrt(2.0 * math.pi) * sigma);
}

fn recursiveGaussianParameters(sigma: f64, b: *f32, b1: *f32, b2: *f32, b3: *f32) void {
    const q: f64 = if (sigma < 2.5) (3.97156 - 4.14554 * math.sqrt(1 - 0.26891 * sigma)) else 0.98711 * sigma - 0.96330;

    const den: f64 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    const n1: f64 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    const n2: f64 = -(1.4281 * q * q + 1.26661 * q * q * q);
    const n3: f64 = 0.422205 * q * q * q;

    b.* = @floatCast(1 - (n1 + n2 + n3) / den);
    b1.* = @floatCast(n1 / den);
    b2.* = @floatCast(n2 / den);
    b3.* = @floatCast(n3 / den);
}

fn recursiveGaussian2DVertical(output: []f32, input: []const f32, height: u32, width: u32, stride: u32, b: f32, b1: f32, b2: f32, b3: f32) void {
    if (output.ptr != input.ptr) {
        @memcpy(output[0..width], input[0..width]);
    }

    for (0..height) |j| {
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var x0: usize = lower;
        var x1: usize = if (j < 1) x0 else (x0 - stride);
        var x2: usize = if (j < 2) x1 else (x1 - stride);
        var x3: usize = if (j < 3) x2 else (x2 - stride);

        while (x0 < upper) : ({
            x0 += 1;
            x1 += 1;
            x2 += 1;
            x3 += 1;
        }) {
            output[x0] = b * input[x0] + b1 * output[x1] + b2 * output[x2] + b3 * output[x3];
        }
    }

    var i: i32 = @bitCast(height - 1);
    while (i >= 0) : (i -= 1) {
        const j: u32 = @bitCast(i);
        const lower: u32 = stride * j;
        const upper: u32 = lower + width;

        var x0: u32 = lower;
        var x1: u32 = if (j >= height - 1) x0 else (x0 + stride);
        var x2: u32 = if (j >= height - 2) x1 else (x1 + stride);
        var x3: u32 = if (j >= height - 3) x2 else (x2 + stride);

        while (x0 < upper) : ({
            x0 += 1;
            x1 += 1;
            x2 += 1;
            x3 += 1;
        }) {
            output[x0] = b * output[x0] + b1 * output[x1] + b2 * output[x2] + b3 * output[x3];
        }
    }
}

fn recursiveGaussian2DHorizontal(output: []f32, input: []const f32, height: u32, width: u32, stride: u32, b: f32, b1: f32, b2: f32, b3: f32) void {
    for (0..height) |j| {
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var i: usize = lower;
        var p1: f32 = input[i];
        var p2: f32 = p1;
        var p3: f32 = p2;
        output[i] = p3;
        i += 1;

        while (i < upper) : (i += 1) {
            const p0 = b * input[i] + b1 * p1 + b2 * p2 + b3 * p3;
            p3 = p2;
            p2 = p1;
            p1 = p0;
            output[i] = p0;
        }

        i -= 1;
        p1 = output[i];
        p2 = p1;
        p3 = p2;
        if (i == lower) continue;
        i -= 1;
        while (true) : (i -= 1) {
            const p0 = b * output[i] + b1 * p1 + b2 * p2 + b3 * p3;
            p3 = p2;
            p2 = p1;
            p1 = p0;
            output[i] = p0;
            if (i == lower) break;
        }
    }
}
