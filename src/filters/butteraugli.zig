//! Butteraugli, ported from libjxl master at 863caac (2026-10-30), the
//! revision it was validated against (lib/jxl/butteraugli/butteraugli.cc),
//! along the path its `butteraugli_main` tool takes: ButteraugliComparator
//! with one 2x-subsampled level (`Make` + `Diffmap`), the score being the
//! diffmap maximum, plus the tool's 3-norm (lib/extras/metrics.cc
//! ComputeDistanceP).
//!
//! Rounding follows libjxl's AVX2 build: highway `MulAdd`/`MulSub` are fused
//! (`simd.madd`), plain C++ arithmetic is not (libjxl compiles as ISO C++,
//! no FP contraction), and double-precision steps stay f64. The blurs repeat
//! libjxl's per-kernel summation order, including Separable5's scalar
//! fallback on the last pixels of its 8-lane rows. Measured against
//! butteraugli_main on real frames up to 1080p, the distance agrees
//! to about one f32 ulp and each diffmap pixel to ~1e-6; builds without FMA
//! differ further by rounding.

const std = @import("std");
const common = @import("../common.zig");
const simd = @import("simd.zig");
const srgb = @import("srgb.zig");

const VL: usize = @max(4, common.vec_len / 4);
const V = @Vector(VL, f32);
const VD = @Vector(VL, f64);
const VI = @Vector(VL, i32);

inline fn sp(x: f32) V {
    return @splat(x);
}
inline fn ld(s: []const f32, i: usize) V {
    return s[i..][0..VL].*;
}
inline fn st(s: []f32, i: usize, v: V) void {
    s[i..][0..VL].* = v;
}
inline fn madd(a: V, b: V, c: V) V {
    return simd.madd(V, a, b, c);
}

pub const Params = struct {
    hf_asymmetry: f32 = 1.0,
    xmul: f32 = 1.0,
    intensity_target: f32 = 80.0,
};

pub const Result = struct {
    /// butteraugli distance: the diffmap maximum
    max: f64,
    /// butteraugli_main's "3-norm" (ComputeDistanceP with p = 3)
    norm3: f64,
};

// ---------------------------------------------------------------- constants

const wMfMalta = 37.0819870399;
const norm1Mf = 130262059.556;
const wMfMaltaX = 8246.75321353;
const norm1MfX = 1009002.70582;
const wHfMalta = 18.7237414387;
const norm1Hf = 4498534.45232;
const wHfMaltaX = 6923.99476109;
const norm1HfX = 8051.15833247;
const wUhfMalta = 1.10039032555;
const norm1Uhf = 71.7800275169;
const wUhfMaltaX = 173.5;
const norm1UhfX = 5.0;
const wmul = [9]f64{ 400.0, 1.50815703118, 0, 2150.0, 10.6195433239, 16.2176043152, 29.2353797994, 0.844626970982, 0.703646627719 };

// std::log(80.0) / std::log(255.0)
const kIntensityTargetNormalizationHack: f32 = 0.79079917404;
const kInternalGoodQualityThreshold: f32 = 17.83 * kIntensityTargetNormalizationHack;
const kGlobalScale: f32 = @floatCast(1.0 / @as(f64, kInternalGoodQualityThreshold));

// Blur sigmas, as the f32 the C++ `Blur(..., float sigma)` receives.
const sigma_opsin: f32 = 1.2;
const sigma_lf: f32 = 7.15593339443;
const sigma_hf: f32 = 3.22489901262;
const sigma_uhf: f32 = 1.56416327805;
const sigma_mask: f32 = 2.7;

// ---------------------------------------------------------------- geometry

/// One resolution level: 20 planes of `stride * h` plus the Malta diff ring
/// and the blur row ring. Roles change through the pipeline (see
/// `diffmapLevel`).
const Level = struct {
    w: u32,
    h: u32,
    stride: u32,
    xyb: [2][3][]f32,
    /// hf[k][c] = st8[2k + c], uhf[k][c] = st8[4 + 2k + c]; also staging
    st8: [8][]f32,
    dc: [3][]f32,
    /// block_diff_ac X, Y (B is formed in the final combine)
    ac: [2][]f32,
    diffmap: []f32,
    /// MaltaDiffMap's diff rows, 4 zero pixels on each side: a ring of 9
    /// rows plus an all-zero row
    pad: []f32,
    pstride: u32,
    /// `blur`'s horizontally filtered rows, ring_rows of them
    ring: []f32,

    /// rows of the widest kernel (sigma_lf: 33 taps)
    const ring_rows = Gauss(sigma_lf).len;

    fn planeLen(w: u32, h: u32) usize {
        return @as(usize, strideOf(w)) * h;
    }
    fn padStride(w: u32) u32 {
        return @intCast(std.mem.alignForward(usize, strideOf(w) + 8, VL));
    }
    fn size(w: u32, h: u32) usize {
        return 20 * planeLen(w, h) + @as(usize, padStride(w)) * 10 + ring_rows * @as(usize, strideOf(w));
    }

    fn init(buf: []f32, w: u32, h: u32) Level {
        const n = planeLen(w, h);
        var lv: Level = undefined;
        lv.w = w;
        lv.h = h;
        lv.stride = strideOf(w);
        lv.pstride = padStride(w);
        var off: usize = 0;
        const next = struct {
            fn f(b: []f32, o: *usize, len: usize) []f32 {
                defer o.* += len;
                return b[o.*..][0..len];
            }
        }.f;
        for (0..2) |k| {
            for (0..3) |c| lv.xyb[k][c] = next(buf, &off, n);
        }
        for (&lv.st8) |*p| p.* = next(buf, &off, n);
        for (&lv.dc) |*p| p.* = next(buf, &off, n);
        for (&lv.ac) |*p| p.* = next(buf, &off, n);
        lv.diffmap = next(buf, &off, n);
        lv.pad = next(buf, &off, @as(usize, lv.pstride) * 10);
        lv.ring = next(buf, &off, ring_rows * @as(usize, lv.stride));
        return lv;
    }

    inline fn row(lv: *const Level, p: []f32, y: usize) []f32 {
        return p[y * lv.stride ..][0..lv.stride];
    }
    inline fn hf(lv: *const Level, k: usize, c: usize) []f32 {
        return lv.st8[2 * k + c];
    }
    inline fn uhf(lv: *const Level, k: usize, c: usize) []f32 {
        return lv.st8[4 + 2 * k + c];
    }
};

/// Plane row stride: whole vectors, so element-wise passes run on full rows.
fn strideOf(w: u32) u32 {
    return @intCast(std.mem.alignForward(usize, w, VL));
}

fn halfOf(n: u32) u32 {
    return (n + 1) / 2;
}

/// f32 elements of scratch for a w x h comparison. The block must start
/// zeroed (padding columns and the Malta border are never written).
pub fn scratchSize(w: u32, h: u32) usize {
    return Level.size(w, h) + Level.size(halfOf(w), halfOf(h));
}

// ---------------------------------------------------------------- blurs

/// libjxl ComputeKernel(sigma) and ConvolutionWithTranspose's normalization.
fn Gauss(comptime sigma: f32) type {
    return struct {
        const o: usize = @max(1, @as(usize, @intFromFloat(@as(f32, 2.25) * @abs(sigma))));
        const len = 2 * o + 1;
        const k: [len]f32 = blk: {
            @setEvalBranchQuota(100_000);
            const s: f64 = sigma;
            const scaler: f64 = -1.0 / (2.0 * s * s);
            var t: [len]f32 = undefined;
            for (0..len) |j| {
                const i: f64 = @as(f64, @floatFromInt(j)) - @as(f64, @floatFromInt(o));
                t[j] = @floatCast(@exp(scaler * i * i));
            }
            break :blk t;
        };
        /// kernel * (1 / sum), for the unclipped middle
        const sk: [o + 1]f32 = blk: {
            var sum: f32 = 0;
            for (k) |v| sum += v;
            const scale: f32 = 1.0 / sum;
            var t: [o + 1]f32 = undefined;
            for (0..o + 1) |i| t[i] = k[i] * scale;
            break :blk t;
        };
    };
}

/// Middle output of a ConvolutionWithTranspose pass: pairs folded into up to
/// four round-robin accumulators, then center + acc0 + acc1 + ..., the order
/// of libjxl's unrolled 7/13/15/33-tap loops. `at(i)` loads tap i.
inline fn foldTaps(comptime G: type, comptime T: type, at: anytype) T {
    const na = @min(4, G.o);
    var acc: [na]T = undefined;
    inline for (0..G.o) |i| {
        const pair = (at.get(i) + at.get(G.len - 1 - i)) * at.k(i);
        if (i < na) acc[i] = pair else acc[i % 4] = acc[i % 4] + pair;
    }
    var sum = at.get(G.o) * at.k(G.o);
    inline for (0..na) |i| sum = sum + acc[i];
    return sum;
}

/// One horizontal pass row (libjxl's first ConvolutionWithTranspose).
fn blurRowH(comptime G: type, src: []const f32, dst: []f32, w: u32) void {
    const o = G.o;
    const wi: usize = w;
    const b1 = @min(wi, o);
    const b2 = if (wi > o) wi - o else 0;
    var x: usize = b1;
    while (x + VL <= b2) : (x += VL) {
        const At = struct {
            s: []const f32,
            base: usize,
            inline fn get(a: @This(), i: usize) V {
                return ld(a.s, a.base + i);
            }
            inline fn k(_: @This(), i: usize) V {
                return sp(G.sk[i]);
            }
        };
        st(dst, x, foldTaps(G, V, At{ .s = src, .base = x - o }));
    }
    while (x < b2) : (x += 1) {
        const At = struct {
            s: []const f32,
            base: usize,
            inline fn get(a: @This(), i: usize) f32 {
                return a.s[a.base + i];
            }
            inline fn k(_: @This(), i: usize) f32 {
                return G.sk[i];
            }
        };
        dst[x] = foldTaps(G, f32, At{ .s = src, .base = x - o });
    }
    // ConvolveBorderColumn: unnormalized taps over the valid range, then
    // divided by their sum
    for (0..b1) |bl| dst[bl] = borderH(G, src, bl, wi);
    var br = @max(b2, b1);
    while (br < wi) : (br += 1) dst[br] = borderH(G, src, br, wi);
}

inline fn borderH(comptime G: type, src: []const f32, x: usize, n: usize) f32 {
    const lo = if (x < G.o) 0 else x - G.o;
    const hi = @min(n - 1, x + G.o);
    var weight: f32 = 0;
    for (lo..hi + 1) |j| weight += G.k[j + G.o - x];
    const scale: f32 = 1.0 / weight;
    var sum: f32 = 0;
    for (lo..hi + 1) |j| sum += src[j] * G.k[j + G.o - x];
    return sum * scale;
}

/// libjxl Blur() for the large kernels (two ConvolutionWithTranspose passes),
/// streamed like SSIMULACRA2's blur: each source row's horizontal pass goes
/// into a ring of G.len rows, and output row y is the vertical pass over ring
/// rows y-o..y+o (vectorized across columns). Same operations in the same
/// order as a full horizontal plane followed by a vertical one, without
/// writing and re-reading that plane.
///
/// Each blurred vector goes to `epi.put(i, v)` (i: plane index of its first
/// lane) as soon as its row is done, so the pass that consumes it can run
/// while the row is in cache. `put` may rewrite `src` at row y and below:
/// source row r is filtered into the ring (at output row r - o) before
/// output row r is put.
fn blurEpi(comptime sigma: f32, lv: *const Level, src: []const f32, epi: anytype) void {
    const G = Gauss(sigma);
    comptime std.debug.assert(G.len != 5 and G.len <= Level.ring_rows);
    const o = G.o;
    const n = G.len;
    const h: usize = lv.h;
    const s: usize = lv.stride;
    var next: usize = 0; // next source row to filter horizontally
    for (0..h) |y| {
        const hi = @min(h - 1, y + o);
        while (next <= hi) : (next += 1) {
            blurRowH(G, src[next * s ..][0..s], lv.ring[(next % n) * s ..][0..s], lv.w);
        }
        const out = OutRow(@TypeOf(epi)){ .epi = epi, .base = y * s };
        if (y >= o and y + o < h) {
            var rows: [n][]const f32 = undefined;
            inline for (0..n) |i| rows[i] = lv.ring[((y - o + i) % n) * s ..][0..s];
            var x: usize = 0;
            while (x < s) : (x += VL) {
                const At = struct {
                    r: *const [n][]const f32,
                    x: usize,
                    inline fn get(a: @This(), i: usize) V {
                        return ld(a.r[i], a.x);
                    }
                    inline fn k(_: @This(), i: usize) V {
                        return sp(G.sk[i]);
                    }
                };
                out.put(x, foldTaps(G, V, At{ .r = &rows, .x = x }));
            }
        } else {
            // ConvolveBorderColumn on the transposed rows
            const lo = if (y < o) 0 else y - o;
            var weight: f32 = 0;
            for (lo..hi + 1) |j| weight += G.k[j + o - y];
            const scale: V = sp(1.0 / weight);
            var x: usize = 0;
            while (x < s) : (x += VL) {
                var sum: V = sp(0);
                for (lo..hi + 1) |j| sum = sum + ld(lv.ring, (j % n) * s + x) * sp(G.k[j + o - y]);
                out.put(x, sum * scale);
            }
        }
    }
}

fn OutRow(comptime E: type) type {
    return struct {
        epi: E,
        base: usize,
        inline fn put(self: @This(), x: usize, v: V) void {
            self.epi.put(self.base + x, v);
        }
    };
}

/// Plain store of the blurred plane.
const Store = struct {
    dst: []f32,
    inline fn put(self: Store, i: usize, v: V) void {
        st(self.dst, i, v);
    }
};

fn blur(comptime sigma: f32, lv: *const Level, src: []const f32, dst: []f32) void {
    blurEpi(sigma, lv, src, Store{ .dst = dst });
}

/// libjxl's Mirror: -1 -> 0, n -> n - 1.
inline fn mirror(i: isize, n: usize) usize {
    const ni: isize = @intCast(n);
    var v = i;
    while (v < 0 or v >= ni) v = if (v < 0) -v - 1 else 2 * ni - 1 - v;
    return @intCast(v);
}

/// Separable5 weights: center, distance 1, distance 2.
const sep5_w = blk: {
    const G = Gauss(sigma_opsin);
    std.debug.assert(G.len == 5);
    break :blk [3]f32{ G.sk[2], G.sk[1], G.sk[0] };
};

/// Separable5's horizontal pass of one row (mirrored edges):
/// fma(l2 + r2, w2, fma(l1 + r1, w1, c * w0)).
fn sep5Row(r: []const f32, o: []f32, w: usize) void {
    const w0 = sep5_w[0];
    const w1 = sep5_w[1];
    const w2 = sep5_w[2];
    var x: usize = 0;
    while (x < @min(2, w)) : (x += 1) o[x] = h5(r, x, w, w0, w1, w2);
    while (x + VL + 2 <= w) : (x += VL) {
        const m1 = madd(ld(r, x - 1) + ld(r, x + 1), sp(w1), ld(r, x) * sp(w0));
        st(o, x, madd(ld(r, x - 2) + ld(r, x + 2), sp(w2), m1));
    }
    while (x < w) : (x += 1) o[x] = h5(r, x, w, w0, w1, w2);
}

/// Output row y of Separable5 into `out` (full stride): the vertical pass
/// over the horizontal rows of y-2..y+2 (mirrored), `hrow(r)` returning row
/// r's horizontal pass. libjxl's AVX2 Separable5 leaves the last
/// (w % 8 >= 1) pixels of a row to a scalar, non-separable 25-term sum on
/// the source rows; those are repeated so they round as libjxl's do.
fn sep5Out(src: []const f32, hrow: anytype, out: []f32, y: usize, w: usize, h: usize, s: usize) void {
    const w0 = sep5_w[0];
    const w1 = sep5_w[1];
    const w2 = sep5_w[2];
    const yi: isize = @intCast(y);
    const ry = [5]usize{ mirror(yi - 2, h), mirror(yi - 1, h), y, mirror(yi + 1, h), mirror(yi + 2, h) };
    const t2 = hrow.get(ry[0]);
    const t1 = hrow.get(ry[1]);
    const m = hrow.get(ry[2]);
    const b1 = hrow.get(ry[3]);
    const b2 = hrow.get(ry[4]);
    var x: usize = 0;
    while (x < s) : (x += VL) {
        const c1 = madd(ld(t1, x) + ld(b1, x), sp(w1), ld(m, x) * sp(w0));
        st(out, x, madd(ld(t2, x) + ld(b2, x), sp(w2), c1));
    }
    var tx = sep5ScalarStart(w);
    while (tx < w) : (tx += 1) {
        var mul: f32 = 0;
        for (0..5) |dyi| {
            const wy = sep5_w[if (dyi < 2) 2 - dyi else dyi - 2];
            const row = src[ry[dyi] * s ..];
            for (0..5) |dxi| {
                const wx = sep5_w[if (dxi < 2) 2 - dxi else dxi - 2];
                const cx = mirror(@as(isize, @intCast(tx + dxi)) - 2, w);
                mul += row[cx] * wx * wy;
            }
        }
        out[tx] = mul;
    }
}

/// First column libjxl's AVX2 (8-lane) Separable5 computes with its scalar
/// fallback: after the first vector, the unpadded vector loop, and a last
/// mirrored vector when w % 8 < 2. w if there is none.
fn sep5ScalarStart(w: usize) usize {
    const lanes = 8;
    var x: usize = lanes;
    while (x + lanes + 2 <= w) x += lanes;
    if (w % lanes < 2) x += lanes;
    return @min(x, w);
}

inline fn h5(r: []const f32, x: usize, w: usize, w0: f32, w1: f32, w2: f32) f32 {
    const xi: isize = @intCast(x);
    const l1 = r[mirror(xi - 1, w)];
    const r1 = r[mirror(xi + 1, w)];
    const l2 = r[mirror(xi - 2, w)];
    const r2 = r[mirror(xi + 2, w)];
    const m1 = simd.madd(f32, l1 + r1, w1, r[x] * w0);
    return simd.madd(f32, l2 + r2, w2, m1);
}

// ---------------------------------------------------------------- opsin

/// libjxl FastLog2f: (2,2) rational polynomial of log2(1 + m) after a range
/// reduction to [-1/3, 1/3].
inline fn fastLog2f(x: V) V {
    const xb: VI = @bitCast(x);
    const eb = xb -% @as(VI, @splat(0x3f2aaaab));
    const es = eb >> @splat(23);
    const mant: V = @bitCast(xb -% (es << @splat(23)));
    const ev: V = @floatFromInt(es);
    const t = mant - sp(1.0);
    var yp = sp(7.4245873327820566E-01);
    yp = madd(yp, t, sp(1.4287160470083755E+00));
    yp = madd(yp, t, sp(-1.8503833400518310E-06));
    var yq = sp(1.7409343003366853E-01);
    yq = madd(yq, t, sp(1.0096718572241148E+00));
    yq = madd(yq, t, sp(9.9032814277590719E-01));
    return yp / yq + ev;
}

inline fn gamma(v_in: V) V {
    const kInvLog2e: f32 = 0.6931471805599453;
    const ret_mul: f32 = 19.245013259874995 * kInvLog2e;
    const v = @select(f32, v_in < sp(0), sp(0), v_in);
    const biased = v + sp(9.9710635769299145);
    return madd(sp(ret_mul), fastLog2f(biased), sp(-23.16046239805755));
}

const mix = [12]f32{
    0.29956550340058319, 0.63373087833825936,  0.077705617820981968, 1.7557483643287353,
    0.22158691104574774, 0.69391388044116142,  0.0987313588422,      1.7557483643287353,
    0.02,                0.02,                 0.20480129041026129,  12.226454707163354,
};

inline fn absorbance(comptime clamp: bool, r: V, g: V, b: V) [3]V {
    var o: [3]V = undefined;
    inline for (0..3) |i| {
        o[i] = madd(sp(mix[4 * i]), r, madd(sp(mix[4 * i + 1]), g, madd(sp(mix[4 * i + 2]), b, sp(mix[4 * i + 3]))));
        if (clamp) o[i] = @max(o[i], sp(mix[4 * i + 3]));
    }
    return o;
}

/// OpsinDynamicsImage: `rgb` linear (intensity applied here), `out` the XYB
/// planes. The Separable5 blur of the three channels is streamed: each
/// source row's horizontal pass goes into a 5-row ring per channel, output
/// row y's vertical pass into a row buffer, and the opsin math reads the
/// buffers while they are in cache (same operations as blurred planes).
fn opsin(lv: *const Level, rgb: [3][]f32, out: [3][]f32, intensity: f32) void {
    const w: usize = lv.w;
    const h: usize = lv.h;
    const s: usize = lv.stride;
    comptime std.debug.assert(Level.ring_rows >= 3 * 5 + 3);
    const Ring = struct {
        r: []f32,
        s: usize,
        inline fn get(self: @This(), row: usize) []const f32 {
            return self.r[(row % 5) * self.s ..][0..self.s];
        }
    };
    var rings: [3]Ring = undefined;
    var buf: [3][]f32 = undefined;
    for (0..3) |c| {
        rings[c] = .{ .r = lv.ring[c * 5 * s ..][0 .. 5 * s], .s = s };
        buf[c] = lv.ring[(15 + c) * s ..][0..s];
    }
    const it = sp(intensity);
    const min = sp(1e-4);
    var next: usize = 0;
    for (0..h) |y| {
        while (next <= @min(h - 1, y + 2)) : (next += 1) {
            for (0..3) |c| sep5Row(rgb[c][next * s ..][0..s], rings[c].r[(next % 5) * s ..][0..s], w);
        }
        for (0..3) |c| sep5Out(rgb[c], rings[c], buf[c], y, w, h, s);
        var x: usize = 0;
        while (x < s) : (x += VL) {
            const i = y * s + x;
            const pre = absorbance(true, ld(buf[0], x) * it, ld(buf[1], x) * it, ld(buf[2], x) * it);
            var sens: [3]V = undefined;
            inline for (0..3) |c| {
                const pm = @max(pre[c], min);
                sens[c] = @max(gamma(pm) / pm, min);
            }
            var cur = absorbance(false, ld(rgb[0], i) * it, ld(rgb[1], i) * it, ld(rgb[2], i) * it);
            inline for (0..3) |c| cur[c] = cur[c] * sens[c];
            const c0 = @max(cur[0], sp(1.7557483643287353));
            const c1 = @max(cur[1], sp(1.7557483643287353));
            const c2 = @max(cur[2], sp(12.226454707163354));
            st(out[0], i, c0 - c1);
            st(out[1], i, c0 + c1);
            st(out[2], i, c2);
        }
    }
}

// ---------------------------------------------------------------- frequency bands

inline fn removeRange(comptime kw: f32, x: V) V {
    const w = sp(kw);
    return @select(f32, x > w, x - w, @select(f32, x < -w, x + w, sp(0)));
}

inline fn amplifyRange(comptime kw: f32, x: V) V {
    const w = sp(kw);
    return @select(f32, x > w, x + w, @select(f32, x < -w, x - w, x + x));
}

inline fn maximumClamp(comptime kmax: f32, v: V) V {
    const mul = sp(0.724216145665);
    const maxval = sp(kmax);
    const if_pos = madd(v - maxval, mul, maxval);
    const if_neg = madd(v + maxval, mul, -maxval);
    const pos_or_v = @select(f32, v >= maxval, if_pos, v);
    return @select(f32, v < -maxval, if_neg, pos_or_v);
}

/// SeparateMFAndHF for image k (mf = xyb[k], blurred in place). libjxl
/// copies mf to hf, blurs mf, then takes hf = copy - blurred and the range
/// step in a pass of its own, and SuppressXByY in another. The blur's output
/// rows do all of it here (the unblurred row y is still in `mf` when row y is
/// put), with the same operations.
fn separateMfHf(lv: *const Level, k: usize) void {
    const mf = lv.xyb[k];
    const hx = lv.hf(k, 0);
    const hy = lv.hf(k, 1);
    blurEpi(sigma_hf, lv, mf[0], struct {
        mf: []f32,
        hf: []f32,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.hf, i, ld(e.mf, i) - v);
            st(e.mf, i, removeRange(0.29, v));
        }
    }{ .mf = mf[0], .hf = hx });
    blurEpi(sigma_hf, lv, mf[1], struct {
        mf: []f32,
        hx: []f32,
        hy: []f32,
        inline fn put(e: @This(), i: usize, v: V) void {
            const vy = ld(e.mf, i) - v;
            st(e.hy, i, vy);
            st(e.mf, i, amplifyRange(0.1, v));
            // SuppressXByY
            const s: f32 = 0.653020556257;
            const yw = sp(46.0);
            const scaler = madd(yw / madd(vy, vy, yw), sp(1.0 - s), sp(s));
            st(e.hx, i, scaler * ld(e.hx, i));
        }
    }{ .mf = mf[1], .hx = hx, .hy = hy });
    blur(sigma_hf, lv, mf[2], mf[2]);
}

/// SeparateHFAndUHF for image k, fused into the blur the same way.
fn separateHfUhf(lv: *const Level, k: usize) void {
    blurEpi(sigma_uhf, lv, lv.hf(k, 0), struct {
        hf: []f32,
        uhf: []f32,
        inline fn put(e: @This(), i: usize, v: V) void {
            const uhfv = ld(e.hf, i) - v;
            st(e.hf, i, removeRange(1.5, v));
            st(e.uhf, i, removeRange(0.04, uhfv));
        }
    }{ .hf = lv.hf(k, 0), .uhf = lv.uhf(k, 0) });
    blurEpi(sigma_uhf, lv, lv.hf(k, 1), struct {
        hf: []f32,
        uhf: []f32,
        inline fn put(e: @This(), i: usize, v: V) void {
            const hfv = maximumClamp(28.4691806922, v);
            const uhfv = maximumClamp(5.19175294647, ld(e.hf, i) - hfv);
            st(e.uhf, i, uhfv * sp(2.69313763794));
            st(e.hf, i, amplifyRange(0.132, hfv * sp(2.155)));
        }
    }{ .hf = lv.hf(k, 1), .uhf = lv.uhf(k, 1) });
}

// ---------------------------------------------------------------- Malta

const Tap = [2]i8; // dy, dx

// MaltaUnit(MaltaTagLF): 16 five-tap lines, in libjxl's order.
const lf_lines = [16][5]Tap{
    .{ .{ 0, -4 }, .{ 0, -2 }, .{ 0, 0 }, .{ 0, 2 }, .{ 0, 4 } },
    .{ .{ -4, 0 }, .{ -2, 0 }, .{ 0, 0 }, .{ 2, 0 }, .{ 4, 0 } },
    .{ .{ -3, -3 }, .{ -2, -2 }, .{ 0, 0 }, .{ 2, 2 }, .{ 3, 3 } },
    .{ .{ -3, 3 }, .{ -2, 2 }, .{ 0, 0 }, .{ 2, -2 }, .{ 3, -3 } },
    .{ .{ -4, 1 }, .{ -2, 1 }, .{ 0, 0 }, .{ 2, -1 }, .{ 4, -1 } },
    .{ .{ -4, -1 }, .{ -2, -1 }, .{ 0, 0 }, .{ 2, 1 }, .{ 4, 1 } },
    .{ .{ -1, -4 }, .{ -1, -2 }, .{ 0, 0 }, .{ 1, 2 }, .{ 1, 4 } },
    .{ .{ 1, -4 }, .{ 1, -2 }, .{ 0, 0 }, .{ -1, 2 }, .{ -1, 4 } },
    .{ .{ -3, -2 }, .{ -2, -1 }, .{ 0, 0 }, .{ 2, 1 }, .{ 3, 2 } },
    .{ .{ -3, 2 }, .{ -2, 1 }, .{ 0, 0 }, .{ 2, -1 }, .{ 3, -2 } },
    .{ .{ -2, -3 }, .{ -1, -2 }, .{ 0, 0 }, .{ 1, 2 }, .{ 2, 3 } },
    .{ .{ -2, 3 }, .{ -1, 2 }, .{ 0, 0 }, .{ 1, -2 }, .{ 2, -3 } },
    .{ .{ 2, -4 }, .{ 1, -2 }, .{ 0, 0 }, .{ -1, 2 }, .{ -2, 4 } },
    .{ .{ -2, -4 }, .{ -1, -2 }, .{ 0, 0 }, .{ 1, 2 }, .{ 2, 4 } },
    .{ .{ -4, -2 }, .{ -2, -1 }, .{ 0, 0 }, .{ 2, 1 }, .{ 4, 2 } },
    .{ .{ -4, 2 }, .{ -2, 1 }, .{ 0, 0 }, .{ 2, -1 }, .{ 4, -2 } },
};

// MaltaUnit(MaltaTag): 16 lines of 9 or 7 taps, in libjxl's order (it lists
// some lines twice; kept).
const hf_l9 = [_][9]Tap{
    .{ .{ 0, -4 }, .{ 0, -3 }, .{ 0, -2 }, .{ 0, -1 }, .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 0, 4 } },
    .{ .{ -4, 0 }, .{ -3, 0 }, .{ -2, 0 }, .{ -1, 0 }, .{ 0, 0 }, .{ 1, 0 }, .{ 2, 0 }, .{ 3, 0 }, .{ 4, 0 } },
    .{ .{ -4, 1 }, .{ -3, 1 }, .{ -2, 1 }, .{ -1, 0 }, .{ 0, 0 }, .{ 1, 0 }, .{ 2, -1 }, .{ 3, -1 }, .{ 4, -1 } },
    .{ .{ -4, -1 }, .{ -3, -1 }, .{ -2, -1 }, .{ -1, 0 }, .{ 0, 0 }, .{ 1, 0 }, .{ 2, 1 }, .{ 3, 1 }, .{ 4, 1 } },
    .{ .{ -1, -4 }, .{ -1, -3 }, .{ -1, -2 }, .{ 0, -1 }, .{ 0, 0 }, .{ 0, 1 }, .{ 1, 2 }, .{ 1, 3 }, .{ 1, 4 } },
    .{ .{ 1, -4 }, .{ 1, -3 }, .{ 1, -2 }, .{ 0, -1 }, .{ 0, 0 }, .{ 0, 1 }, .{ -1, 2 }, .{ -1, 3 }, .{ -1, 4 } },
};
const hf_l7 = [_][7]Tap{
    .{ .{ -3, -3 }, .{ -2, -2 }, .{ -1, -1 }, .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 } },
    .{ .{ -3, 3 }, .{ -2, 2 }, .{ -1, 1 }, .{ 0, 0 }, .{ 1, -1 }, .{ 2, -2 }, .{ 3, -3 } },
    .{ .{ -3, -2 }, .{ -2, -1 }, .{ -1, -1 }, .{ 0, 0 }, .{ 1, 1 }, .{ 2, 1 }, .{ 3, 2 } },
    .{ .{ -3, 2 }, .{ -2, 1 }, .{ -1, 1 }, .{ 0, 0 }, .{ 1, -1 }, .{ 2, -1 }, .{ 3, -2 } },
    .{ .{ -2, -3 }, .{ -1, -2 }, .{ -1, -1 }, .{ 0, 0 }, .{ 1, 1 }, .{ 1, 2 }, .{ 2, 3 } },
    .{ .{ -2, 3 }, .{ -1, 2 }, .{ -1, 1 }, .{ 0, 0 }, .{ 1, -1 }, .{ 1, -2 }, .{ 2, -3 } },
};
/// libjxl's line order for MaltaTag: (is9, index)
const hf_order = [16]struct { bool, u8 }{
    .{ true, 0 },  .{ true, 1 },  .{ false, 0 }, .{ false, 1 }, .{ true, 2 },  .{ true, 3 },
    .{ true, 4 },  .{ true, 5 },  .{ false, 2 }, .{ false, 3 }, .{ false, 4 }, .{ false, 5 },
    .{ true, 5 },  .{ true, 4 },  .{ true, 3 },  .{ true, 2 },
};

/// Sum5/Sum7/Sum9 of libjxl's MaltaUnit, same association. `rows[4 + dy]`
/// is the zero-padded diff row at dy; column x + 4 is pixel x.
inline fn lineSum(comptime taps: anytype, rows: *const [9][]const f32, x: usize) V {
    const n = taps.len;
    var v: [n]V = undefined;
    inline for (taps, 0..) |t, i| v[i] = ld(rows[@intCast(@as(i32, t[0]) + 4)], x + @as(usize, @intCast(@as(i32, t[1]) + 4)));
    return switch (n) {
        5 => (v[0] + v[1]) + (v[2] + (v[3] + v[4])),
        7 => (v[0] + v[1]) + (v[2] + ((v[3] + v[4]) + (v[5] + v[6]))),
        9 => (((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))) + v[8],
        else => @compileError("lineSum"),
    };
}

inline fn maltaUnit(comptime lf: bool, rows: *const [9][]const f32, x: usize) V {
    var ret: V = undefined;
    if (lf) {
        inline for (lf_lines, 0..) |line, i| {
            const s = lineSum(line, rows, x);
            ret = if (i == 0) s * s else madd(s, s, ret);
        }
    } else {
        inline for (hf_order, 0..) |o, i| {
            const s = if (o[0]) lineSum(hf_l9[o[1]], rows, x) else lineSum(hf_l7[o[1]], rows, x);
            ret = if (i == 0) s * s else madd(s, s, ret);
        }
    }
    return ret;
}

/// How a Malta pass meets block_diff_ac: the first one of a plane stores
/// (libjxl adds to a zeroed plane; 0 + x == x for these sums of squares),
/// the last one also applies the plane's L2DiffAsymmetric (hf) and L2Diff
/// (mf) terms, in libjxl's order.
const MaltaMode = enum { set, add, add_l2 };

const L2Tail = struct {
    hf0: []const f32 = &.{},
    hf1: []const f32 = &.{},
    /// L2DiffAsymmetric's w_0gt1 * 0.8 and w_0lt1 * 0.8
    gt: V = @splat(0),
    lt: V = @splat(0),
    /// L2Diff weight of the mf plane (the Malta pass's own lum0/lum1)
    wl: V = @splat(0),
};

/// MaltaDiffMapT: the asymmetric per-pixel diffs go through a ring of 9
/// zero-padded rows (lv.pad; slot 9 stays all zero for rows outside the
/// plane), and each output row's Malta line energies go to `ac` per `mode`.
fn maltaDiffMap(comptime lf: bool, comptime mode: MaltaMode, lv: *const Level, lum0: []const f32, lum1: []const f32, w_0gt1: f64, w_0lt1: f64, norm1: f64, ac: []f32, tail: L2Tail) void {
    const len: f64 = 3.75;
    const mulli: f64 = if (lf) 0.611612573796 else 0.39905817637;
    const kWeight0: f32 = 0.5;
    const kWeight1: f32 = 0.33;
    const w_pre0gt1 = mulli * @sqrt(@as(f64, kWeight0) * w_0gt1) / (len * 2 + 1);
    const w_pre0lt1 = mulli * @sqrt(@as(f64, kWeight1) * w_0lt1) / (len * 2 + 1);
    const n2gt: f32 = @floatCast(w_pre0gt1 * norm1);
    const n2lt: f32 = @floatCast(w_pre0lt1 * norm1);
    const nf: f32 = @floatCast(norm1);

    const w: usize = lv.w;
    const h: usize = lv.h;
    const s: usize = lv.stride;
    const ps: usize = lv.pstride;
    const zero = lv.pad[9 * ps ..][0..ps];
    var next: usize = 0;
    for (0..h) |y| {
        while (next <= @min(h - 1, y + 4)) : (next += 1) {
            const r0 = lum0[next * s ..];
            const r1 = lum1[next * s ..];
            const out = lv.pad[(next % 9) * ps + 4 ..];
            var x: usize = 0;
            while (x + VL <= w) : (x += VL) st(out, x, diffV(ld(r0, x), ld(r1, x), n2gt, n2lt, nf));
            while (x < w) : (x += 1) {
                const one: @Vector(1, f32) = .{r0[x]};
                const two: @Vector(1, f32) = .{r1[x]};
                out[x] = diffAny(1, one, two, n2gt, n2lt, nf)[0];
            }
        }
        var rows: [9][]const f32 = undefined;
        inline for (0..9) |k| {
            const r = @as(isize, @intCast(y)) + @as(isize, k) - 4;
            rows[k] = if (r < 0 or r >= @as(isize, @intCast(h))) zero else lv.pad[(@as(usize, @intCast(r)) % 9) * ps ..][0..ps];
        }
        var x: usize = 0;
        while (x < w) : (x += VL) {
            const m = maltaUnit(lf, &rows, x);
            const i = y * s + x;
            switch (mode) {
                .set => st(ac, i, m),
                .add => st(ac, i, ld(ac, i) + m),
                .add_l2 => {
                    var total = l2Asym(ld(tail.hf0, i), ld(tail.hf1, i), tail.gt, tail.lt, ld(ac, i) + m);
                    const d = ld(lum0, i) - ld(lum1, i);
                    total = madd(d * d, tail.wl, total);
                    st(ac, i, total);
                },
            }
        }
    }
}

inline fn diffV(v0: V, v1: V, n2gt: f32, n2lt: f32, nf: f32) V {
    return diffAny(VL, v0, v1, n2gt, n2lt, nf);
}

inline fn diffAny(comptime n: usize, r0: @Vector(n, f32), r1: @Vector(n, f32), n2gt: f32, n2lt: f32, nf: f32) @Vector(n, f32) {
    const F = @Vector(n, f32);
    const D = @Vector(n, f64);
    const absval = @as(F, @splat(0.5)) * (@abs(r0) + @abs(r1));
    const diff = r0 - r1;
    const scaler = @as(F, @splat(n2gt)) / (@as(F, @splat(nf)) + absval);
    const d: F = scaler * diff;
    const scaler2: D = @floatCast(@as(F, @splat(n2lt)) / (@as(F, @splat(nf)) + absval));
    const fabs0: D = @floatCast(@abs(r0));
    const too_small = @as(D, @splat(0.55)) * fabs0;
    const too_big = @as(D, @splat(1.05)) * fabs0;
    const v1: D = @floatCast(r1);
    const dd: D = @floatCast(d);
    const zero: D = @splat(0);
    // row0 < 0
    const neg_a = v1 > -too_small;
    const neg_b = v1 < -too_big;
    const neg = @select(f64, neg_a, dd - scaler2 * (v1 + too_small), @select(f64, neg_b, dd + scaler2 * (-v1 - too_big), dd));
    // row0 >= 0
    const pos_a = v1 < too_small;
    const pos_b = v1 > too_big;
    const pos = @select(f64, pos_a, dd + scaler2 * (too_small - v1), @select(f64, pos_b, dd - scaler2 * (v1 - too_big), dd));
    const res = @select(f64, @as(D, @floatCast(r0)) < zero, neg, pos);
    // untouched lanes must keep d exactly (f32 -> f64 -> f32 is exact)
    return @floatCast(res);
}

// ---------------------------------------------------------------- L2 terms

/// L2DiffAsymmetric of one vector added to `prev` (vgt/vlt: the weights
/// times 0.8, as libjxl sets them).
inline fn l2Asym(v0: V, v1: V, vgt: V, vlt: V, prev: V) V {
    const diff = v0 - v1;
    const total = madd(diff * diff, vgt, prev);
    const fabs0 = @abs(v0);
    const too_small = sp(0.4) * fabs0;
    const too_big = fabs0;
    const if_neg = @select(f32, v1 > -too_small, v1 + too_small, @select(f32, v1 < -too_big, -v1 - too_big, sp(0)));
    const if_pos = @select(f32, v1 < too_small, too_small - v1, @select(f32, v1 > too_big, v1 - too_big, sp(0)));
    const v = @select(f32, v0 < sp(0), if_neg, if_pos);
    return madd(vlt, v * v, total);
}

/// L2DiffAsymmetric's vector weights from its float parameters.
fn asymWeights(w_0gt1: f32, w_0lt1: f32) [2]V {
    return .{ sp(@floatCast(@as(f64, w_0gt1) * 0.8)), sp(@floatCast(@as(f64, w_0lt1) * 0.8)) };
}

// ---------------------------------------------------------------- masking

/// CombineChannelsForMasking then DiffPrecompute, for both images in one
/// pass (`d0`, `d1`).
fn maskInputs(lv: *const Level, d0: []f32, d1: []f32) void {
    const mul: f32 = 6.19424080439;
    const bias: f32 = mul * @as(f32, 12.61050594197);
    const sqrt_bias = @sqrt(bias);
    const n = @as(usize, lv.stride) * lv.h;
    const out = [2][]f32{ d0, d1 };
    var i: usize = 0;
    while (i < n) : (i += VL) {
        inline for (0..2) |k| {
            const xd = (ld(lv.uhf(k, 0), i) + ld(lv.hf(k, 0), i)) * sp(2.5);
            const yd = ld(lv.uhf(k, 1), i) * sp(0.4) + ld(lv.hf(k, 1), i) * sp(0.4);
            const m = @sqrt(xd * xd + yd * yd);
            st(out[k], i, @sqrt(sp(mul) * @abs(m) + sp(bias)) - sp(sqrt_bias));
        }
    }
}

inline fn storeMin3(v: f32, m0: *f32, m1: *f32, m2: *f32) void {
    if (v < m2.*) {
        if (v < m0.*) {
            m2.* = m1.*;
            m1.* = m0.*;
            m0.* = v;
        } else if (v < m1.*) {
            m2.* = m1.*;
            m1.* = v;
        } else {
            m2.* = v;
        }
    }
}

/// Sorted three smallest after inserting `v` (min/max network). Same values
/// as StoreMin3's branchy insertion.
inline fn insertMin3(v: V, m0: *V, m1: *V, m2: *V) void {
    m2.* = @min(m2.*, @max(m1.*, v));
    m1.* = @min(m1.*, @max(m0.*, v));
    m0.* = @min(m0.*, v);
}

/// FuzzyErosion. The result is 0.45/0.3/0.25 of the three smallest of
/// {c, 2c, 2c} and the up-to-eight neighbours at distance 3, which libjxl's
/// StoreMin3 keeps in order (its seed c <= 2c holds: the input is a blur of
/// non-negative values). The interior runs as a branch-free vector network
/// that keeps the same three values, so the output is identical.
fn fuzzyErosion(lv: *const Level, from: []const f32, to: []f32) void {
    const k = 3;
    const w: usize = lv.w;
    const h: usize = lv.h;
    const s: usize = lv.stride;
    for (0..h) |y| {
        const r = from[y * s ..];
        const up = y >= k;
        const dn = y + k < h;
        const ru = if (up) from[(y - k) * s ..] else r;
        const rd = if (dn) from[(y + k) * s ..] else r;
        var x: usize = k;
        while (x + VL + k <= w) : (x += VL) {
            var m0 = ld(r, x);
            var m1 = m0 * sp(2);
            var m2 = m1;
            insertMin3(ld(r, x - k), &m0, &m1, &m2);
            insertMin3(ld(r, x + k), &m0, &m1, &m2);
            if (up) {
                insertMin3(ld(ru, x - k), &m0, &m1, &m2);
                insertMin3(ld(ru, x), &m0, &m1, &m2);
                insertMin3(ld(ru, x + k), &m0, &m1, &m2);
            }
            if (dn) {
                insertMin3(ld(rd, x - k), &m0, &m1, &m2);
                insertMin3(ld(rd, x), &m0, &m1, &m2);
                insertMin3(ld(rd, x + k), &m0, &m1, &m2);
            }
            st(to, y * s + x, sp(0.45) * m0 + sp(0.3) * m1 + sp(0.25) * m2);
        }
        const vend = x;
        fuzzyScalar(from, to, w, h, s, y, 0, @min(k, w));
        fuzzyScalar(from, to, w, h, s, y, @max(vend, @min(k, w)), w);
    }
}

fn fuzzyScalar(from: []const f32, to: []f32, w: usize, h: usize, s: usize, y: usize, x0: usize, x1: usize) void {
    const k = 3;
    for (x0..x1) |x| {
        var m0 = from[y * s + x];
        var m1 = 2 * m0;
        var m2 = m1;
        if (x >= k) {
            storeMin3(from[y * s + x - k], &m0, &m1, &m2);
            if (y >= k) storeMin3(from[(y - k) * s + x - k], &m0, &m1, &m2);
            if (y + k < h) storeMin3(from[(y + k) * s + x - k], &m0, &m1, &m2);
        }
        if (x + k < w) {
            storeMin3(from[y * s + x + k], &m0, &m1, &m2);
            if (y >= k) storeMin3(from[(y - k) * s + x + k], &m0, &m1, &m2);
            if (y + k < h) storeMin3(from[(y + k) * s + x + k], &m0, &m1, &m2);
        }
        if (y >= k) storeMin3(from[(y - k) * s + x], &m0, &m1, &m2);
        if (y + k < h) storeMin3(from[(y + k) * s + x], &m0, &m1, &m2);
        to[y * s + x] = 0.45 * m0 + 0.3 * m1 + 0.25 * m2;
    }
}

// ---------------------------------------------------------------- one level

/// Diffmap of one level into `lv.diffmap`. On entry st8[0..3] holds the
/// reference and st8[3..6] the distorted linear RGB (intensity not applied).
fn diffmapLevel(lv: *const Level, p: Params) void {
    for (0..2) |k| opsin(lv, .{ lv.st8[3 * k], lv.st8[3 * k + 1], lv.st8[3 * k + 2] }, lv.xyb[k], p.intensity_target);

    separateLf(lv);

    for (0..2) |k| separateMfHf(lv, k);
    for (0..2) |k| separateHfUhf(lv, k);

    // block_diff_ac: X in ac[0], Y in ac[1]; B's only term (the mf L2Diff)
    // is formed in the final combine
    const hfa: f64 = p.hf_asymmetry;
    // std::sqrt(float) is a float; the products with the weights are double
    const sq: f64 = @sqrt(p.hf_asymmetry);
    var tails: [2]L2Tail = undefined;
    for (0..2) |c| {
        const aw = asymWeights(@floatCast(wmul[c] * hfa), @floatCast(wmul[c] / hfa));
        tails[c] = .{ .hf0 = lv.hf(0, c), .hf1 = lv.hf(1, c), .gt = aw[0], .lt = aw[1], .wl = sp(@floatCast(wmul[3 + c])) };
    }
    maltaDiffMap(false, .set, lv, lv.uhf(0, 1), lv.uhf(1, 1), wUhfMalta * hfa, wUhfMalta / hfa, norm1Uhf, lv.ac[1], .{});
    maltaDiffMap(false, .set, lv, lv.uhf(0, 0), lv.uhf(1, 0), wUhfMaltaX * hfa, wUhfMaltaX / hfa, norm1UhfX, lv.ac[0], .{});
    maltaDiffMap(true, .add, lv, lv.hf(0, 1), lv.hf(1, 1), wHfMalta * sq, wHfMalta / sq, norm1Hf, lv.ac[1], .{});
    maltaDiffMap(true, .add, lv, lv.hf(0, 0), lv.hf(1, 0), wHfMaltaX * sq, wHfMaltaX / sq, norm1HfX, lv.ac[0], .{});
    maltaDiffMap(true, .add_l2, lv, lv.xyb[0][1], lv.xyb[1][1], wMfMalta, wMfMalta, norm1Mf, lv.ac[1], tails[1]);
    maltaDiffMap(true, .add_l2, lv, lv.xyb[0][0], lv.xyb[1][0], wMfMaltaX, wMfMaltaX, norm1MfX, lv.ac[0], tails[0]);

    // MaskPsychoImage (the MF X/Y planes are free now; the B ones stay for
    // the final combine). The second image's mask blur feeds its diff_ac
    // term and CombineChannelsToDiffmap straight from its output rows.
    const diff0 = lv.xyb[0][0];
    const diff1 = lv.xyb[0][1];
    const blurred0 = lv.xyb[1][1];
    const mask = lv.xyb[1][0];
    maskInputs(lv, diff0, diff1);
    blur(sigma_mask, lv, diff0, blurred0);
    fuzzyErosion(lv, blurred0, mask);
    blurEpi(sigma_mask, lv, diff1, struct {
        lv: *const Level,
        b0: []const f32,
        mask: []const f32,
        xm: V,
        wb: V,
        inline fn put(e: @This(), i: usize, v: V) void {
            const d = ld(e.b0, i) - v;
            const ac1 = ld(e.lv.ac[1], i) + sp(10.0) * d * d;
            // B's block_diff_ac: L2Diff of the mf B planes onto zero
            // (fma(d2, w, 0) == d2 * w)
            const db = ld(e.lv.xyb[0][2], i) - ld(e.lv.xyb[1][2], i);
            const ac2 = db * db * e.wb;
            // CombineChannelsToDiffmap; MaskY/MaskDcY in f64 lanes, as
            // libjxl's double functions
            const val: VD = @floatCast(ld(e.mask, i));
            const m: V = @floatCast(maskYV(val));
            const dm: V = @floatCast(maskDcYV(val));
            const dc0 = ld(e.lv.dc[0], i) * e.xm;
            const ac0 = ld(e.lv.ac[0], i) * e.xm;
            const dcs = dc0 * dm + ld(e.lv.dc[1], i) * dm + ld(e.lv.dc[2], i) * dm;
            const acs = ac0 * m + ac1 * m + ac2 * m;
            st(e.lv.diffmap, i, @sqrt(dcs + acs));
        }
    }{ .lv = lv, .b0 = blurred0, .mask = mask, .xm = sp(p.xmul), .wb = sp(@floatCast(wmul[5])) });
}

/// SeparateLFAndMF for both images plus XybLowFreqToVals and the dc
/// SetL2Diff, all on the LF blur's output rows: mf = xyb - lf in place, the
/// "vals" of image 0's lf kept (st8[0..3]), image 1's turned into dc right
/// away (only its raw Y, needed by its B, is stored: st8[3]). Channels run
/// X, Y, B so B's vals can take the raw Y, as libjxl's pass does.
fn separateLf(lv: *const Level) void {
    const xmul = sp(33.832837186260);
    const ymul = sp(14.458268100570);
    const bmul = sp(49.87984651440);
    const y_to_b = sp(-0.362267051518);
    const lf0 = [3][]f32{ lv.st8[0], lv.st8[1], lv.st8[2] };
    const ly1 = lv.st8[3];
    var wd: [3]V = undefined;
    for (0..3) |c| wd[c] = sp(@floatCast(wmul[6 + c]));

    const Raw = struct { // lf kept raw (Y, until B scales it)
        mf: []f32,
        lf: []f32,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.mf, i, ld(e.mf, i) - v);
            st(e.lf, i, v);
        }
    };
    blurEpi(sigma_lf, lv, lv.xyb[0][0], struct {
        mf: []f32,
        lf: []f32,
        m: V,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.mf, i, ld(e.mf, i) - v);
            st(e.lf, i, v * e.m);
        }
    }{ .mf = lv.xyb[0][0], .lf = lf0[0], .m = xmul });
    blurEpi(sigma_lf, lv, lv.xyb[0][1], Raw{ .mf = lv.xyb[0][1], .lf = lf0[1] });
    blurEpi(sigma_lf, lv, lv.xyb[0][2], struct {
        mf: []f32,
        lb: []f32,
        ly: []f32,
        k: [3]V,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.mf, i, ld(e.mf, i) - v);
            const yv = ld(e.ly, i);
            st(e.lb, i, madd(e.k[0], yv, v) * e.k[1]);
            st(e.ly, i, yv * e.k[2]);
        }
    }{ .mf = lv.xyb[0][2], .lb = lf0[2], .ly = lf0[1], .k = .{ y_to_b, bmul, ymul } });

    blurEpi(sigma_lf, lv, lv.xyb[1][0], struct {
        mf: []f32,
        lf0: []const f32,
        dc: []f32,
        m: V,
        w: V,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.mf, i, ld(e.mf, i) - v);
            const d = ld(e.lf0, i) - v * e.m;
            st(e.dc, i, d * d * e.w);
        }
    }{ .mf = lv.xyb[1][0], .lf0 = lf0[0], .dc = lv.dc[0], .m = xmul, .w = wd[0] });
    blurEpi(sigma_lf, lv, lv.xyb[1][1], Raw{ .mf = lv.xyb[1][1], .lf = ly1 });
    blurEpi(sigma_lf, lv, lv.xyb[1][2], struct {
        mf: []f32,
        ly1: []const f32,
        lf0: [3][]f32,
        dc: [3][]f32,
        k: [3]V,
        w: [3]V,
        inline fn put(e: @This(), i: usize, v: V) void {
            st(e.mf, i, ld(e.mf, i) - v);
            const yraw = ld(e.ly1, i);
            const lb = madd(e.k[0], yraw, v) * e.k[1];
            const ly = yraw * e.k[2];
            const dy = ld(e.lf0[1], i) - ly;
            const db = ld(e.lf0[2], i) - lb;
            st(e.dc[1], i, dy * dy * e.w[1]);
            st(e.dc[2], i, db * db * e.w[2]);
        }
    }{ .mf = lv.xyb[1][2], .ly1 = ly1, .lf0 = lf0, .dc = lv.dc, .k = .{ y_to_b, bmul, ymul }, .w = wd });
}

inline fn maskYV(delta: VD) VD {
    const c = @as(VD, @splat(2.5485944793)) / ((@as(VD, @splat(0.451936922203)) * delta) + @as(VD, @splat(0.829591754942)));
    const r = @as(VD, @splat(@as(f64, kGlobalScale))) * (@as(VD, @splat(1.0)) + c);
    return r * r;
}

inline fn maskDcYV(delta: VD) VD {
    const c = @as(VD, @splat(0.505054525019)) / ((@as(VD, @splat(3.87449418804)) * delta) + @as(VD, @splat(0.20025578522)));
    const r = @as(VD, @splat(@as(f64, kGlobalScale))) * (@as(VD, @splat(1.0)) + c);
    return r * r;
}

// ---------------------------------------------------------------- input

pub const Input = struct {
    pub const Kind = enum { u8, u16, f32 };
    kind: Kind,
    planes: [3][*]const u8,
    /// bytes
    stride: usize,
    /// integer kinds: sample -> linear light
    lut: []const f32 = &.{},
    /// f32: decode sRGB (false: already linear)
    srgb: bool = true,
};

/// Integer sample -> linear light: v / peak as libjxl loads 8/16-bit
/// samples, then the sRGB decode unless `is_srgb` is false.
pub fn buildLut(lut: []f32, bits: u5, is_srgb: bool) void {
    const scale: f32 = @as(f32, 1.0) / @as(f32, @floatFromInt((@as(u32, 1) << bits) - 1));
    for (lut, 0..) |*o, i| {
        const x = @as(f32, @floatFromInt(i)) * scale;
        o.* = if (is_srgb) srgb.toLinear(@as(@Vector(1, f32), @splat(x)))[0] else x;
    }
}

fn fillLinear(in: Input, dst: [3][]f32, w: u32, h: u32, stride: u32) void {
    for (0..3) |c| {
        for (0..h) |y| {
            const row = dst[c][y * stride ..][0..stride];
            const src = in.planes[c] + y * in.stride;
            switch (in.kind) {
                .u8 => for (0..w) |x| {
                    row[x] = in.lut[src[x]];
                },
                .u16 => {
                    const s: [*]const u16 = @ptrCast(@alignCast(src));
                    for (0..w) |x| row[x] = in.lut[s[x]];
                },
                .f32 => {
                    const s: [*]const f32 = @ptrCast(@alignCast(src));
                    var x: usize = 0;
                    while (x + VL <= w) : (x += VL) {
                        const v: V = s[x..][0..VL].*;
                        st(row, x, if (in.srgb) srgb.toLinear(v) else v);
                    }
                    while (x < w) : (x += 1) {
                        const v: @Vector(1, f32) = .{s[x]};
                        row[x] = (if (in.srgb) srgb.toLinear(v) else v)[0];
                    }
                },
            }
            @memset(row[w..], 0);
        }
    }
}

/// SubSample2x of the three planes `src` (level `a`) into `dst` (level `b`).
fn subSample2x(a: *const Level, src: [3][]const f32, b: *const Level, dst: [3][]f32) void {
    for (0..3) |c| {
        for (0..b.h) |oy| {
            const row = dst[c][oy * b.stride ..][0..b.stride];
            // whole 2x2 windows: libjxl's order 0 + a + b (row 2oy), + c + d
            const r0 = src[c][2 * oy * a.stride ..];
            const has1 = 2 * oy + 1 < a.h;
            const r1 = if (has1) src[c][(2 * oy + 1) * a.stride ..] else r0;
            const q = sp(0.25);
            var vx: usize = 0;
            while (vx + VL <= a.w / 2) : (vx += VL) {
                const p0 = ld(r0, 2 * vx);
                const p1 = ld(r0, 2 * vx + VL);
                var sum = sp(0) + q * simd.evenLanesF32(VL, p0, p1);
                sum = sum + q * simd.oddLanesF32(VL, p0, p1);
                if (has1) {
                    const p2 = ld(r1, 2 * vx);
                    const p3 = ld(r1, 2 * vx + VL);
                    sum = sum + q * simd.evenLanesF32(VL, p2, p3);
                    sum = sum + q * simd.oddLanesF32(VL, p2, p3);
                }
                st(row, vx, sum);
            }
            for (vx..b.w) |ox| {
                var sum: f32 = 0;
                const y0 = 2 * oy;
                const x0 = 2 * ox;
                for (y0..@min(y0 + 2, a.h)) |y| {
                    for (x0..@min(x0 + 2, a.w)) |x| sum += 0.25 * src[c][y * a.stride + x];
                }
                row[ox] = sum;
            }
            @memset(row[b.w..], 0);
        }
        // odd sizes: the lone last column / row was averaged as a pair
        if ((a.w & 1) != 0) {
            for (0..b.h) |oy| dst[c][oy * b.stride + b.w - 1] *= 2.0;
        }
        if ((a.h & 1) != 0) {
            for (0..b.w) |ox| dst[c][(b.h - 1) * b.stride + ox] *= 2.0;
        }
    }
}

// ---------------------------------------------------------------- entry

pub const Context = struct {
    full: Level,
    half: Level,

    pub fn init(scratch: []f32, w: u32, h: u32) Context {
        const fs = Level.size(w, h);
        return .{
            .full = Level.init(scratch[0..fs], w, h),
            .half = Level.init(scratch[fs..], halfOf(w), halfOf(h)),
        };
    }

    /// The final diffmap (row stride `full.stride`), valid after `run`.
    pub fn diffmap(self: *const Context) []const f32 {
        return self.full.diffmap;
    }
};

/// Butteraugli of `distorted` against `reference` (both w x h, w and h >= 16).
pub fn run(ctx: *Context, reference: Input, distorted: Input, p: Params) Result {
    const f = &ctx.full;
    const hl = &ctx.half;
    fillLinear(reference, .{ f.st8[0], f.st8[1], f.st8[2] }, f.w, f.h, f.stride);
    fillLinear(distorted, .{ f.st8[3], f.st8[4], f.st8[5] }, f.w, f.h, f.stride);
    subSample2x(f, .{ f.st8[0], f.st8[1], f.st8[2] }, hl, .{ hl.st8[0], hl.st8[1], hl.st8[2] });
    subSample2x(f, .{ f.st8[3], f.st8[4], f.st8[5] }, hl, .{ hl.st8[3], hl.st8[4], hl.st8[5] });
    diffmapLevel(hl, p);
    diffmapLevel(f, p);

    // AddSupersampled2x(sub, 0.5, full), then the score and the 3-norm
    const w: f32 = 0.5;
    const keep: f64 = 1.0 - 0.3 * @as(f64, w);
    var max: f32 = 0;
    var sums = [3]f64{ 0, 0, 0 };
    for (0..f.h) |y| {
        const row = f.diffmap[y * f.stride ..][0..f.w];
        const sub = hl.diffmap[(y / 2) * hl.stride ..];
        for (row, 0..) |*o, x| {
            var v: f32 = @floatCast(@as(f64, o.*) * keep);
            v += w * sub[x / 2];
            o.* = v;
            max = @max(max, v);
            const d1: f64 = v;
            var d2 = d1 * d1 * d1;
            sums[0] += d2;
            d2 *= d2;
            sums[1] += d2;
            d2 *= d2;
            sums[2] += d2;
        }
    }
    const inv_n = 1.0 / @as(f64, @floatFromInt(@as(u64, f.w) * f.h));
    const norm3 = (std.math.pow(f64, inv_n * sums[0], 1.0 / 3.0) +
        std.math.pow(f64, inv_n * sums[1], 1.0 / 6.0) +
        std.math.pow(f64, inv_n * sums[2], 1.0 / 12.0)) / 3.0;
    return .{ .max = max, .norm3 = norm3 };
}

// ---------------------------------------------------------------- heatmap

fn fuzzyClass(score: f64) f64 {
    const fuzzy_width_up = 4.8;
    const fuzzy_width_down = 4.8;
    const m0 = 2.0;
    const scaler = 0.7777;
    if (score < 1.0) {
        var val = m0 / (1.0 + @exp((score - 1.0) * fuzzy_width_down));
        val -= 1.0;
        val *= 2.0 - scaler;
        val += scaler;
        return val;
    }
    return m0 / (1.0 + @exp((score - 1.0) * fuzzy_width_up)) * scaler;
}

/// ButteraugliFuzzyInverse.
pub fn fuzzyInverse(seek: f64) f64 {
    var pos: f64 = 0;
    var range: f64 = 1.0;
    while (range >= 1e-10) : (range *= 0.5) {
        if (fuzzyClass(pos) < seek) pos -= range else pos += range;
    }
    return pos;
}

/// CreateHeatMapImage's ScoreToRgb (butteraugli_main's --distmap colors).
pub fn heatmapRgb(score: f64, good: f64, bad: f64) [3]f32 {
    const heat = [12][3]f64{
        .{ 0, 0, 0 },     .{ 0, 0, 1 },   .{ 0, 1, 1 },     .{ 0, 1, 0 },
        .{ 1, 1, 0 },     .{ 1, 0, 0 },   .{ 1, 0, 1 },     .{ 0.5, 0.5, 1.0 },
        .{ 1.0, 0.5, 0.5 }, .{ 1.0, 1.0, 0.5 }, .{ 1, 1, 1 }, .{ 1, 1, 1 },
    };
    var s = score;
    if (s < good) {
        s = (s / good) * 0.3;
    } else if (s < bad) {
        s = 0.3 + (s - good) / (bad - good) * 0.15;
    } else {
        s = 0.45 + (s - bad) / (bad * 12) * 0.5;
    }
    const n = heat.len;
    s = std.math.clamp(s * (n - 1), 0.0, @as(f64, n - 2));
    var ix: i64 = if (std.math.isNan(s)) 0 else @intFromFloat(s);
    ix = std.math.clamp(ix, 0, n - 2);
    const mixv = s - @as(f64, @floatFromInt(ix));
    const i: usize = @intCast(ix);
    var rgb: [3]f32 = undefined;
    for (0..3) |c| {
        const v = mixv * heat[i + 1][c] + (1 - mixv) * heat[i][c];
        rgb[c] = @floatCast(@sqrt(v));
    }
    return rgb;
}
