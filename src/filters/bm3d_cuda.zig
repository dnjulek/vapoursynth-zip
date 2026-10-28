//! BM3D, CUDA-output variant: a CPU port of vszipcu's `bm3d.cu` (itself
//! BM3DCUDA_RTC), fixed 8x8 blocks and groups of 8.
//!
//! Same algorithm as the GPU (matching, transforms, thresholding, weights,
//! aggregation, `extractor_exp`), not the GPU's exact rounding: `1.0f / x`
//! and the Wiener divide are plain IEEE operations here, where the GPU's
//! -use_fast_math makes them approximate (MUFU.RCP, ±1 ulp). The arithmetic
//! otherwise still follows the GPU's operation order:
//! - block distance: per column, an fma chain over the even rows and one over
//!   the odd rows, their sum, then a butterfly over the columns (xor 1, 2, 4);
//! - transforms: FFTW-derived 8-point kernels with the GPU's fma points;
//! - accumulation is `fma(w, v, E) - E` / `(w + E) - E`.
//! The GPU sums its block estimates with atomics, so its own output is only
//! deterministic with `extractor_exp` (pre-rounding makes the sums exact);
//! here the sums run in reference-block raster order. FTZ is not emulated
//! (denormal intermediates do not occur for [0, 1] video).

const std = @import("std");
const common = @import("../common.zig");
const bc = @import("bm3d_common.zig");
const simd = @import("simd.zig");

const allocator = common.kernel_allocator;

pub const V8 = bc.V8;
const I8 = @Vector(8, i32);
pub const Ring = bc.Ring;

pub const max_radius = 16;
const flt_max = std.math.floatMax(f32);
const flt_eps = std.math.floatEps(f32);

/// Basic / final sigma multipliers: the f32 value of BM3DCUDA's stepwise
/// `sigma * (3/255) ...` product (folding it in f64 is 1 ulp off).
pub const sigma_factor_basic: f32 = @bitCast(@as(u32, 0x3f021bb6));
pub const sigma_factor_final: f32 = @bitCast(@as(u32, 0x3e40c0c1));

pub const BmError = enum {
    ssd,
    sad,
    zssd,
    zsad,
    ssd_norm,

    pub fn parse(s: []const u8) ?BmError {
        const eq = std.ascii.eqlIgnoreCase;
        if (eq(s, "ssd")) return .ssd;
        if (eq(s, "sad")) return .sad;
        if (eq(s, "zssd")) return .zssd;
        if (eq(s, "zsad")) return .zsad;
        if (eq(s, "ssd/norm")) return .ssd_norm;
        return null;
    }
};

pub const Transform = enum {
    dct,
    haar,
    wht,
    bior1_5,

    pub fn parse(s: []const u8) ?Transform {
        const eq = std.ascii.eqlIgnoreCase;
        if (eq(s, "dct")) return .dct;
        if (eq(s, "haar")) return .haar;
        if (eq(s, "wht")) return .wht;
        if (eq(s, "bior1.5")) return .bior1_5;
        return null;
    }
};

/// One processing unit: a single plane, or the three planes of a `chroma`
/// (CBM3D) clip that share the block matching of plane 0.
pub const Params = struct {
    /// scaled sigma per entry plane
    sigma: [3]f32,
    /// planes of the entry that are filtered (sigma >= FLT_EPSILON)
    process: [3]bool,
    n_planes: u32,
    width: u32,
    height: u32,
    block_step: u32,
    bm_range: u32,
    ps_num: u32,
    ps_range: u32,
    radius: u32,
    bm_error: BmError,
    t2d: Transform,
    t1d: Transform,
    /// 2^extractor_exp, or 0
    extractor: f32,

    /// Furthest a group member can lie from its reference block, in rows.
    pub fn reach(self: *const Params) u32 {
        return self.bm_range + self.radius * self.ps_range;
    }

    /// Ring rows a stage needs. The full plane when the spatial window can
    /// hold fewer than 8 candidates: the GPU then fills the group with
    /// (0, 0) placeholders, which lie outside any band.
    pub fn ringRows(self: *const Params) u32 {
        const wy = @min(self.bm_range, self.height - 8) + 1;
        const wx = @min(self.bm_range, self.width - 8) + 1;
        if (wy * wx < 8) return self.height;
        return @min(self.height, 2 * self.reach() + 8);
    }

    pub fn slots(self: *const Params) u32 {
        return 2 * self.radius + 1;
    }
};

pub const PlaneIn = struct {
    data: [*]const f32,
    /// in floats
    stride: usize,

    inline fn at(self: PlaneIn, x: u32, y: u32) [*]const f32 {
        return self.data + @as(usize, y) * self.stride + x;
    }
};

inline fn load8(p: [*]const f32) V8 {
    return p[0..8].*;
}

// ---------------------------------------------------------------------------
// 8-point transforms (bm3d.cu). The arrays hold 8 vectors; each lane is an
// independent transform.

fn K(comptime x: comptime_float) V8 {
    return @splat(@as(f32, x));
}

inline fn fma(a: V8, b: V8, c: V8) V8 {
    return simd.madd(V8, a, b, c);
}

/// FFTW e10_8 / e01_8 (normalized, scaled DCT-II / DCT-III).
fn dct(comptime forward: bool, v: *[8]V8) void {
    if (forward) {
        const T1 = v[0];
        const T2 = v[7];
        const T3 = T1 - T2;
        const Tj = T1 + T2;
        const Tc = v[4];
        const Td = v[3];
        const Te = Tc - Td;
        const Tk = Tc + Td;
        const T4 = v[2];
        const T5 = v[5];
        const T6 = T4 - T5;
        const T7 = v[1];
        const T8 = v[6];
        const T9 = T7 - T8;
        const Ta = T6 + T9;
        const Tn = T7 + T8;
        const Tf = T6 - T9;
        const Tm = T4 + T5;
        const Tb = fma(-K(0.707106781186547524400844362104849039284835938), Ta, T3);
        const Tg = fma(-K(0.707106781186547524400844362104849039284835938), Tf, Te);
        v[3] = K(1.662939224605090474157576755235811513477121624) * fma(K(0.668178637919298919997757686523080761552472251), Tg, Tb);
        v[5] = -(K(1.662939224605090474157576755235811513477121624) * fma(-K(0.668178637919298919997757686523080761552472251), Tb, Tg));
        const Tp = Tj + Tk;
        const Tq = Tm + Tn;
        v[4] = K(1.414213562373095048801688724209698078569671875) * (Tp - Tq);
        v[0] = K(1.414213562373095048801688724209698078569671875) * (Tp + Tq);
        const Th = fma(K(0.707106781186547524400844362104849039284835938), Ta, T3);
        const Ti = fma(K(0.707106781186547524400844362104849039284835938), Tf, Te);
        v[1] = K(1.961570560806460898252364472268478073947867462) * fma(-K(0.198912367379658006911597622644676228597850501), Ti, Th);
        v[7] = K(1.961570560806460898252364472268478073947867462) * fma(K(0.198912367379658006911597622644676228597850501), Th, Ti);
        const Tl = Tj - Tk;
        const To = Tm - Tn;
        v[2] = K(1.847759065022573512256366378793576573644833252) * fma(-K(0.414213562373095048801688724209698078569671875), To, Tl);
        v[6] = K(1.847759065022573512256366378793576573644833252) * fma(K(0.414213562373095048801688724209698078569671875), Tl, To);
    } else {
        const KP1_662939224 = K(1.662939224605090474157576755235811513477121624);
        const KP668178637 = K(0.668178637919298919997757686523080761552472251);
        const KP1_961570560 = K(1.961570560806460898252364472268478073947867462);
        const KP198912367 = K(0.198912367379658006911597622644676228597850501);
        const KP1_847759065 = K(1.847759065022573512256366378793576573644833252);
        const KP707106781 = K(0.707106781186547524400844362104849039284835938);
        const KP414213562 = K(0.414213562373095048801688724209698078569671875);
        const KP1_414213562 = K(1.414213562373095048801688724209698078569671875);

        // ptxas fuses the v[0] product here and keeps v[4]'s (sm_86 SASS),
        // the reverse of the FFTW source's FMA(KP, T2, T1)
        const p4 = KP1_414213562 * v[4];
        const T3 = fma(KP1_414213562, v[0], p4);
        const Tj = fma(KP1_414213562, v[0], -p4);
        const T4 = v[2];
        const T5 = v[6];
        const T6 = fma(KP414213562, T5, T4);
        const Tk = fma(KP414213562, T4, -T5);
        const T8 = v[1];
        const Td = v[7];
        const T9 = v[5];
        const Ta = v[3];
        const Tb = T9 + Ta;
        const Te = Ta - T9;
        const Tc = fma(KP707106781, Tb, T8);
        const Tn = fma(-KP707106781, Te, Td);
        const Tf = fma(KP707106781, Te, Td);
        const Tm = fma(-KP707106781, Tb, T8);
        const T7 = fma(KP1_847759065, T6, T3);
        const Tg = fma(KP198912367, Tf, Tc);
        v[7] = fma(-KP1_961570560, Tg, T7);
        v[0] = fma(KP1_961570560, Tg, T7);
        const Tp = fma(-KP1_847759065, Tk, Tj);
        const Tq = fma(KP668178637, Tm, Tn);
        v[5] = fma(-KP1_662939224, Tq, Tp);
        v[2] = fma(KP1_662939224, Tq, Tp);
        const Th = fma(-KP1_847759065, T6, T3);
        const Ti = fma(-KP198912367, Tc, Tf);
        v[3] = fma(-KP1_961570560, Ti, Th);
        v[4] = fma(KP1_961570560, Ti, Th);
        const Tl = fma(KP1_847759065, Tk, Tj);
        const To = fma(-KP668178637, Tn, Tm);
        v[6] = fma(-KP1_662939224, To, Tl);
        v[1] = fma(KP1_662939224, To, Tl);
    }
}

fn transform(t: Transform, comptime forward: bool, v: *[8]V8) void {
    switch (t) {
        .dct => dct(forward, v),
        .haar => haar(forward, v),
        .wht => wht(v),
        .bior1_5 => bior15(forward, v),
    }
}

fn haar(comptime forward: bool, v: *[8]V8) void {
    const KP1_414213562 = K(1.414213562373095048801688724209698078569671875);
    const KP2 = K(2.0);
    const scale = KP1_414213562;
    if (forward) {
        const T1 = v[0] + v[1];
        const T2 = v[0] - v[1];
        const T3 = v[2] + v[3];
        const T4 = v[2] - v[3];
        const T5 = v[4] + v[5];
        const T6 = v[4] - v[5];
        const T7 = v[6] + v[7];
        const T8 = v[6] - v[7];
        const T9 = T1 + T3;
        const T10 = KP1_414213562 * (T1 - T3);
        const T11 = T5 + T7;
        const T12 = KP1_414213562 * (T5 - T7);
        const s2 = scale * KP2;
        v[0] = scale * (T9 + T11);
        v[1] = scale * (T9 - T11);
        v[2] = scale * T10;
        v[3] = scale * T12;
        v[4] = s2 * T2;
        v[5] = s2 * T4;
        v[6] = s2 * T6;
        v[7] = s2 * T8;
    } else {
        // ptxas fuses the v[2] product; 2 * v[4] is exact
        const T1 = v[0] + v[1];
        const T2 = v[0] - v[1];
        const d4 = v[4] + v[4];
        const T3 = fma(KP1_414213562, v[2], d4);
        const T4 = fma(KP1_414213562, v[2], -d4);
        const T5 = fma(-KP1_414213562, v[2], d4);
        const T6 = fma(-KP1_414213562, v[2], -d4);
        v[0] = scale * (T1 + T3);
        v[1] = scale * (T1 + T4);
        v[2] = scale * (T1 + T5);
        v[3] = scale * (T1 + T6);
        v[4] = scale * (T2 + T3);
        v[5] = scale * (T2 + T4);
        v[6] = scale * (T2 + T5);
        v[7] = scale * (T2 + T6);
    }
}

fn wht(v: *[8]V8) void {
    const scale = K(1.414213562373095048801688724209698078569671875);
    const T1 = v[0] + v[1];
    const T2 = v[0] - v[1];
    const T3 = v[2] + v[3];
    const T4 = v[2] - v[3];
    const T5 = v[4] + v[5];
    const T6 = v[4] - v[5];
    const T7 = v[6] + v[7];
    const T8 = v[6] - v[7];
    const T9 = T1 + T3;
    const T10 = T1 - T3;
    const T11 = T2 + T4;
    const T12 = T2 - T4;
    const T13 = T5 + T7;
    const T14 = T5 - T7;
    const T15 = T6 + T8;
    const T16 = T6 - T8;
    v[0] = scale * (T9 + T13);
    v[1] = scale * (T9 - T13);
    v[2] = scale * (T10 - T14);
    v[3] = scale * (T10 + T14);
    v[4] = scale * (T12 + T16);
    v[5] = scale * (T12 - T16);
    v[6] = scale * (T11 - T15);
    v[7] = scale * (T11 + T15);
}

fn bior15(comptime forward: bool, v: *[8]V8) void {
    if (forward) {
        const T1 = v[0] + v[3];
        const T2 = v[0] - v[3];
        const T3 = v[1] + v[2];
        const T4 = v[1] - v[2];
        const T5 = v[4] + v[7];
        const T6 = v[4] - v[7];
        const T7 = v[5] + v[6];
        const T8 = v[5] - v[6];
        const T9 = v[0] - v[1];
        const T10 = v[2] - v[3];
        const T11 = v[4] - v[5];
        const T12 = v[6] - v[7];
        v[0] = K(1.414213562373095048801688724209698078569671875) * (((T1 + T5) + T3) + T7);
        v[1] = fma(K(0.877670597010003062405456290501163901200537360), T1 - T5, K(1.797135031972863413496886690073811797696338403) * (T3 - T7));
        v[2] = fma(K(0.334024180361136429417383083658457088741315663), T8 - T6, fma(K(2.277437593371384746027611934034934695963515886), T2, K(1.609389232649111887192845766718020518480884560) * T4));
        v[3] = fma(K(0.334024180361136429417383083658457088741315663), T4 - T2, fma(K(2.277437593371384746027611934034934695963515886), T6, K(1.609389232649111887192845766718020518480884560) * T8));
        v[4] = K(2.828427124746190097603377448419396157139343751) * T9;
        v[5] = K(2.828427124746190097603377448419396157139343751) * T10;
        v[6] = K(2.828427124746190097603377448419396157139343751) * T11;
        v[7] = K(2.828427124746190097603377448419396157139343751) * T12;
    } else {
        // every scaling product is fused into the add that consumes it
        // (sm_86 SASS); only v[1], v[6] and v[7]'s 0.486 products stay
        const KP1_414 = K(1.414213562373095048801688724209698078569671875);
        const KP1_495 = K(1.495435764250674860795011090214036706658653686);
        const KP2_058 = K(2.058234225009388964222454285384072231477027482);
        const KP2_828 = K(2.828427124746190097603377448419396157139343751);
        const KP486 = K(0.486135912065751423025580498947083714508324707);
        const T2 = KP1_495 * v[1];
        const T10 = KP486 * v[6];
        const T12 = KP486 * v[7];
        const T13 = fma(KP1_414, v[0], T2);
        const T14 = fma(KP1_414, v[0], -T2);
        const T16 = fma(KP486, v[4], -T10);
        const T15 = fma(KP486, v[5], -T12);
        const A = fma(-KP2_058, v[2], T13);
        const B = fma(KP2_058, v[2], T13);
        const C = fma(-KP2_058, v[3], T14);
        const D = fma(KP2_058, v[3], T14);
        const v4 = v[4];
        const v5 = v[5];
        const v6 = v[6];
        const v7 = v[7];
        v[0] = B + fma(KP2_828, v4, -T15);
        v[1] = B - fma(KP2_828, v4, T15);
        v[2] = A + fma(KP2_828, v5, T16);
        v[3] = A + fma(-KP2_828, v5, T16);
        v[4] = D + fma(KP2_828, v6, T15);
        v[5] = D + fma(-KP2_828, v6, T15);
        v[6] = C - fma(-KP2_828, v7, T16);
        v[7] = C - fma(KP2_828, v7, T16);
    }
}

/// 8x8 transpose of 8 row vectors (unpack / shuffle / 128-bit permute).
inline fn transpose(v: *[8]V8) void {
    const lo = [8]i32{ 0, -1, 1, -2, 4, -5, 5, -6 };
    const hi = [8]i32{ 2, -3, 3, -4, 6, -7, 7, -8 };
    var t: [8]V8 = undefined;
    inline for (0..4) |i| {
        t[2 * i] = @shuffle(f32, v[2 * i], v[2 * i + 1], lo);
        t[2 * i + 1] = @shuffle(f32, v[2 * i], v[2 * i + 1], hi);
    }
    const s44 = [8]i32{ 0, 1, -1, -2, 4, 5, -5, -6 };
    const sEE = [8]i32{ 2, 3, -3, -4, 6, 7, -7, -8 };
    var u: [8]V8 = undefined;
    u[0] = @shuffle(f32, t[0], t[2], s44);
    u[1] = @shuffle(f32, t[0], t[2], sEE);
    u[2] = @shuffle(f32, t[1], t[3], s44);
    u[3] = @shuffle(f32, t[1], t[3], sEE);
    u[4] = @shuffle(f32, t[4], t[6], s44);
    u[5] = @shuffle(f32, t[4], t[6], sEE);
    u[6] = @shuffle(f32, t[5], t[7], s44);
    u[7] = @shuffle(f32, t[5], t[7], sEE);
    const p20 = [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 };
    const p31 = [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 };
    inline for (0..4) |i| {
        v[i] = @shuffle(f32, u[i], u[i + 4], p20);
        v[i + 4] = @shuffle(f32, u[i], u[i + 4], p31);
    }
}

/// Group layout: g[block * 8 + row], lane = column. The GPU keeps one column
/// per lane and transposes through shared memory; the per-element arithmetic
/// is the same.
const Group = [64]V8;

/// Forward leaves each block's coefficients transposed (y transform, then x
/// after one transpose); the inverse takes them that way (x, transpose, y)
/// and returns rows. Thresholding, Wiener and the 1D transform across blocks
/// are element-wise, and the DC stays at index 0 lane 0, so they don't care.
/// Two transposes per round trip instead of the GPU's four (it returns to rows
/// after each direction); only the order of the separable passes, i.e. the
/// rounding, differs.
fn transform3d(comptime forward: bool, t2d: Transform, t1d: Transform, g: *Group) void {
    for (0..8) |b| {
        const blk: *[8]V8 = g[b * 8 ..][0..8];
        transform(t2d, forward, blk);
        transpose(blk);
        transform(t2d, forward, blk);
    }
    for (0..8) |r| {
        var col: [8]V8 = undefined;
        inline for (0..8) |b| col[b] = g[b * 8 + r];
        transform(t1d, forward, &col);
        inline for (0..8) |b| g[b * 8 + r] = col[b];
    }
}

// ---------------------------------------------------------------------------
// Block matching

/// Butterfly sum over the 8 lanes, as the GPU's xor-shuffle reduction
/// (xor 1, then 2, then 4); every lane ends with lane 0's value.
inline fn reduce8(s: V8) f32 {
    const t1 = s + @shuffle(f32, s, undefined, [8]i32{ 1, 0, 3, 2, 5, 4, 7, 6 });
    const t2 = t1 + @shuffle(f32, t1, undefined, [8]i32{ 2, 3, 0, 1, 6, 7, 4, 5 });
    const t3 = t2 + @shuffle(f32, t2, undefined, [8]i32{ 4, 5, 6, 7, 0, 1, 2, 3 });
    return t3[0];
}

/// `reduce8` of 8 vectors at once (lane j = reduce8(v[j])), bit for bit: a
/// horizontal-add tree forms the same ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7))
/// per vector in 7 adds and their shuffles instead of 3 per vector, and
/// leaves the 8 results in one vector for a single compare.
inline fn reduce8x8(v: *const [8]V8) V8 {
    // [q0123(v0..v3), q4567(v0..v3)] and the same for v4..v7
    const a = hadd(hadd(v[0], v[1]), hadd(v[2], v[3]));
    const b = hadd(hadd(v[4], v[5]), hadd(v[6], v[7]));
    return hadd2(a, b);
}

/// [a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7]
inline fn hadd(a: V8, b: V8) V8 {
    const even = @shuffle(f32, a, b, [8]i32{ 0, 2, -1, -3, 4, 6, -5, -7 });
    const odd = @shuffle(f32, a, b, [8]i32{ 1, 3, -2, -4, 5, 7, -6, -8 });
    return even + odd;
}

/// reduce8x8's last step: lane j = low + high half of the vector-j sums.
inline fn hadd2(a: V8, b: V8) V8 {
    const lo = @shuffle(f32, a, b, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
    const hi = @shuffle(f32, a, b, [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 });
    return lo + hi;
}

/// reduce8x8 of the column sums of candidates p .. p + 7.
inline fn groupDistance(comptime e: BmError, rb: *const RefBlock, p: [*]const f32, stride: usize) V8 {
    var cs: [8]V8 = undefined;
    inline for (0..8) |j| cs[j] = columnSums(e, rb, p + j, stride);
    return reduce8x8(&cs);
}

/// groupDistance of the `k` groups at p, p + 8, ... into out[0..k]; returns
/// `lmin` lowered by them (select(d < lmin), NaN never wins).
inline fn fullGroups(comptime e: BmError, rb: *const RefBlock, p: [*]const f32, stride: usize, k: u32, out: [*]V8, lmin: V8) V8 {
    if (e == .ssd and simd.use_avx2) return fullGroupsSsdAsm(rb, p, stride, k, out, lmin);
    var lm = lmin;
    for (0..k) |i| {
        const d = groupDistance(e, rb, p + 8 * i, stride);
        out[i] = d;
        lm = @select(f32, d < lm, d, lm);
    }
    return lm;
}

/// fullGroups(.ssd) as one asm loop with the 8 reference rows pinned in
/// ymm8..ymm15.
///
/// LLVM keeps all 8 column sums of a group live before reducing them, runs
/// out of registers and reloads a reference row from the stack for every
/// subtraction: 2 loads per row, bound by the 2 FP load pipes at ~71 cycles
/// per group (llvm-mca znver3, matching the measurement). Here each pair of
/// candidates is reduced as soon as it exists, which fits ymm0..ymm6.
///
/// Same operations as groupDistance: d = ref - cand, rows r % 2 accumulated
/// (rows 0 and 1 as d * d, which is fma(d, d, +0) since d * d >= +0), then
/// reduce8x8's vhaddps tree and low + high halves; vminps(d, lmin) is
/// select(d < lmin, d, lmin).
fn fullGroupsSsdAsm(rb: *const RefBlock, p: [*]const f32, stride: usize, k: u32, out: [*]V8, lmin: V8) V8 {
    if (k == 0) return lmin;
    const sb = stride * 4;
    const rows = [8][]const u8{ "%[p]", "%[p],%[s]", "%[p],%[s],2", "%[p],%[s3]", "%[q]", "%[q],%[s]", "%[q],%[s],2", "%[q],%[s3]" };
    // column sums of candidate j into ymm`dst` (temps ymm4..ymm6)
    const cand = struct {
        fn f(comptime j: usize, comptime dst: usize) []const u8 {
            @setEvalBranchQuota(200000);
            comptime var s: []const u8 = "";
            inline for (0..8) |r| {
                s = s ++ std.fmt.comptimePrint("vsubps {d}({s}), %%ymm{d}, %%ymm4\n", .{ 4 * j, rows[r], 8 + r });
                const acc = 5 + r % 2;
                s = s ++ if (r < 2)
                    std.fmt.comptimePrint("vmulps %%ymm4, %%ymm4, %%ymm{d}\n", .{acc})
                else
                    std.fmt.comptimePrint("vfmadd231ps %%ymm4, %%ymm4, %%ymm{d}\n", .{acc});
            }
            return s ++ std.fmt.comptimePrint("vaddps %%ymm6, %%ymm5, %%ymm{d}\n", .{dst});
        }
    }.f;
    const code = comptime "1:\n" ++
        cand(0, 0) ++ cand(1, 1) ++
        "vhaddps %%ymm1, %%ymm0, %%ymm0\n" ++ // h01
        cand(2, 1) ++ cand(3, 2) ++
        "vhaddps %%ymm2, %%ymm1, %%ymm1\n" ++ // h23
        "vhaddps %%ymm1, %%ymm0, %%ymm0\n" ++ // a
        cand(4, 1) ++ cand(5, 2) ++
        "vhaddps %%ymm2, %%ymm1, %%ymm1\n" ++ // h45
        cand(6, 2) ++ cand(7, 3) ++
        "vhaddps %%ymm3, %%ymm2, %%ymm2\n" ++ // h67
        "vhaddps %%ymm2, %%ymm1, %%ymm1\n" ++ // b
        "vperm2f128 $0x20, %%ymm1, %%ymm0, %%ymm2\n" ++ // [a.lo, b.lo]
        "vperm2f128 $0x31, %%ymm1, %%ymm0, %%ymm3\n" ++ // [a.hi, b.hi]
        "vaddps %%ymm3, %%ymm2, %%ymm0\n" ++
        "vminps %[lm], %%ymm0, %[lm]\n" ++
        "vmovups %%ymm0, (%[out])\n" ++
        "add $32, %[p]\n" ++
        "add $32, %[q]\n" ++
        "add $32, %[out]\n" ++
        "dec %[k]\n" ++
        "jnz 1b\n";
    var pp = p;
    var qq = p + 4 * stride;
    var oo = out;
    var kk = k;
    var lm = lmin;
    asm volatile (code
        : [p] "=&r" (pp),
          [q] "=&r" (qq),
          [out] "=&r" (oo),
          [k] "=&r" (kk),
          // "x", not "{ymm7}": an explicit vector register on a tied output
          // fails in Debug builds (LLVM "scalar-to-vector conversion
          // failed"); with ymm0..ymm6 clobbered and ymm8..ymm15 taken, "x"
          // resolves to ymm7 anyway
          [lm] "=&x" (lm),
        : [p_in] "0" (pp),
          [q_in] "1" (qq),
          [out_in] "2" (oo),
          [k_in] "3" (kk),
          [lm_in] "4" (lm),
          [s] "r" (sb),
          [s3] "r" (3 * sb),
          [r0] "{ymm8}" (rb.rows[0]),
          [r1] "{ymm9}" (rb.rows[1]),
          [r2] "{ymm10}" (rb.rows[2]),
          [r3] "{ymm11}" (rb.rows[3]),
          [r4] "{ymm12}" (rb.rows[4]),
          [r5] "{ymm13}" (rb.rows[5]),
          [r6] "{ymm14}" (rb.rows[6]),
          [r7] "{ymm15}" (rb.rows[7]),
        : .{ .ymm0 = true, .ymm1 = true, .ymm2 = true, .ymm3 = true, .ymm4 = true, .ymm5 = true, .ymm6 = true, .memory = true, .cc = true });
    return lm;
}

/// The 8 best matches, ascending distance (strict `<` insertion: on ties the
/// earlier candidate stays ahead). Unfilled slots keep FLT_MAX at (0, 0).
///
/// Vector fields: callers that insert in a loop work on a local copy so the
/// list stays in registers. (Insertion through a pointer made LLVM emit
/// masked stores, slow on Zen 3 and fatal for store forwarding into the next
/// insert.)
const Match = struct {
    err: V8 = @splat(flt_max),
    x: I8 = @splat(0),
    y: I8 = @splat(0),
    z: I8 = @splat(0),

    const lane = std.simd.iota(i32, 8);
    /// lane i takes lane i - 1
    const shift1 = [8]i32{ 0, 0, 1, 2, 3, 4, 5, 6 };

    /// Branchless: the list is sorted, so the slot is the number of kept
    /// errors `<= e` (ties stay ahead), and the tail shifts by one lane.
    /// Only the first `keep` slots are maintained: an entry at or above
    /// err[keep - 1] is skipped. Slots from `keep` on then hold stale but
    /// sorted values >= err[keep - 1], never counted for a smaller `e`, and
    /// the first `keep` slots equal a full top-8 list's (stable insertion:
    /// an entry's slot depends only on the kept entries at or below it).
    inline fn insert(m: *Match, comptime keep: u32, e: f32, x: i32, y: i32, z: i32) void {
        if (!(e < m.err[keep - 1])) return;
        const le: u8 = @bitCast(m.err <= @as(V8, @splat(e)));
        m.place(@popCount(le), e, x, y, z);
    }

    /// Puts (e, x, y, z) at slot `pos`, dropping the last entry.
    inline fn place(m: *Match, pos: i32, e: f32, x: i32, y: i32, z: i32) void {
        const pv: I8 = @splat(pos);
        const before = lane < pv;
        const here = lane == pv;
        m.err = @select(f32, before, m.err, @select(f32, here, @as(V8, @splat(e)), @shuffle(f32, m.err, undefined, shift1)));
        inline for (.{ &m.x, &m.y, &m.z }, .{ x, y, z }) |f, v| {
            f.* = @select(i32, before, f.*, @select(i32, here, @as(I8, @splat(v)), @shuffle(i32, f.*, undefined, shift1)));
        }
    }
};

/// Reference block plus the statistics the error functions reuse.
const RefBlock = struct {
    rows: [8]V8,
    mean: f32 = 0,
    norm: f32 = 0,

    fn init(comptime e: BmError, p: [*]const f32, stride: usize) RefBlock {
        var rb: RefBlock = .{ .rows = undefined };
        inline for (0..8) |r| rb.rows[r] = load8(p + r * stride);
        switch (e) {
            .zssd, .zsad => rb.mean = blockMean(&rb.rows),
            .ssd_norm => rb.norm = blockNorm(&rb.rows),
            else => {},
        }
        return rb;
    }
};

inline fn blockMean(rows: *const [8]V8) f32 {
    const s = ((rows[0] + rows[1]) + (rows[2] + rows[3])) + ((rows[4] + rows[5]) + (rows[6] + rows[7]));
    return reduce8(s) * (1.0 / 64.0);
}

inline fn blockNorm(rows: *const [8]V8) f32 {
    var a: [2]V8 = .{ @splat(0), @splat(0) };
    inline for (0..8) |r| a[r % 2] = fma(rows[r], rows[r], a[r % 2]);
    return @sqrt(reduce8(a[0] + a[1]));
}

/// Per-column partial sums of `distance` for .ssd / .sad (before reduce8).
inline fn columnSums(comptime e: BmError, rb: *const RefBlock, p: [*]const f32, stride: usize) V8 {
    var a: [2]V8 = .{ @splat(0), @splat(0) };
    inline for (0..8) |r| {
        const d = rb.rows[r] - load8(p + r * stride);
        switch (e) {
            .ssd => a[r % 2] = fma(d, d, a[r % 2]),
            .sad => a[r % 2] += @abs(d),
            else => comptime unreachable,
        }
    }
    return a[0] + a[1];
}

inline fn distance(comptime e: BmError, rb: *const RefBlock, p: [*]const f32, stride: usize) f32 {
    var c: [8]V8 = undefined;
    inline for (0..8) |r| c[r] = load8(p + r * stride);
    var a: [2]V8 = .{ @splat(0), @splat(0) };
    switch (e) {
        .ssd => inline for (0..8) |r| {
            const d = rb.rows[r] - c[r];
            a[r % 2] = fma(d, d, a[r % 2]);
        },
        .sad => inline for (0..8) |r| {
            a[r % 2] += @abs(rb.rows[r] - c[r]);
        },
        .zssd, .zsad => {
            const dm: V8 = @splat(rb.mean - blockMean(&c));
            inline for (0..8) |r| {
                const d = (rb.rows[r] - c[r]) - dm;
                if (e == .zssd) a[r % 2] = fma(d, d, a[r % 2]) else a[r % 2] += @abs(d);
            }
        },
        .ssd_norm => {
            const nn = blockNorm(&c);
            inline for (0..8) |r| a[r % 2] = fma(rb.rows[r], c[r], a[r % 2]);
            return 2.0 - 2.0 * reduce8(a[0] + a[1]) / (rb.norm * nn + flt_eps);
        },
    }
    return reduce8(a[0] + a[1]);
}

/// Second pass of searchWindow's chunk: raster-order insertion of the
/// distances at or below the largest per-lane minimum (inf, from a lane
/// without finite candidates, keeps everything).
///
/// On real video ~36% of a default window passes the bound but only ~3%
/// enters, so lanes are also masked by the current worst error and gated
/// again per insert (the worst only decreases). Branch-free compaction plus
/// ungated inserts, which won on a synthetic image, ran every kept candidate
/// through place()'s dependency chain: 37% of r=2 time.
///
/// With `keep` < 8 (the predictive searches, whose first ps_num matches are
/// all that is used) the bound is the keep-th smallest lane minimum: those
/// are `keep` distinct candidates, so anything above it can't be in the top
/// `keep`. On real video that cut the predictive searches' inserts (10% of
/// their candidates with a top-8 list) and their bound checks.
fn insertBounded(m: *Match, comptime keep: u32, d: []const V8, gx: []const u32, gy: []const u32, lmin: V8, z: i32) void {
    // lane minima are never NaN (vminps keeps lmin), only finite or inf
    const bound: V8 = @splat(switch (keep) {
        8 => @reduce(.Max, lmin),
        1 => @reduce(.Min, lmin),
        else => sort8(lmin)[keep - 1],
    });
    for (d, gx, gy) |dv, x, y| {
        var mask = @as(u8, @bitCast(dv <= bound)) & @as(u8, @bitCast(dv < @as(V8, @splat(m.err[keep - 1]))));
        if (mask == 0) continue;
        const da: [8]f32 = dv;
        while (mask != 0) : (mask &= mask - 1) {
            const j = @ctz(mask);
            const e = da[j];
            if (!(e < m.err[keep - 1])) continue;
            const le: u8 = @bitCast(m.err <= @as(V8, @splat(e)));
            m.place(@popCount(le), e, @intCast(x + j), @intCast(y), z);
        }
    }
}

/// Ascending lanes (no NaN), branch-free: a bitonic network, 6 stages of
/// shuffle + min + max + blend.
inline fn sort8(v_in: V8) V8 {
    var v = v_in;
    comptime var k: u32 = 2;
    inline while (k <= 8) : (k *= 2) {
        comptime var j: u32 = k / 2;
        inline while (j >= 1) : (j /= 2) {
            const perm = comptime blk: {
                var a: [8]i32 = undefined;
                for (&a, 0..) |*e, i| e.* = @intCast(i ^ j);
                break :blk a;
            };
            // lane i keeps the min when its pair is ascending and i is the
            // lower index, or descending and i is the upper one
            const take_min = comptime blk: {
                var a: [8]bool = undefined;
                for (&a, 0..) |*e, i| e.* = ((i & j) == 0) == ((i & k) == 0);
                break :blk a;
            };
            const p = @shuffle(f32, v, undefined, perm);
            v = @select(f32, @as(@Vector(8, bool), take_min), @min(v, p), @max(v, p));
            if (j == 1) break;
        }
    }
    return v;
}

/// Exhaustive search of the window around (cx, cy) in one plane.
/// `keep`: how many of the best matches the caller uses (see insertBounded).
fn searchWindow(comptime e: BmError, mp: *Match, comptime keep: u32, rb: *const RefBlock, plane: PlaneIn, w: u32, h: u32, cx: i32, cy: i32, range: u32, z: i32) void {
    // local copy: see Match
    var lm = mp.*;
    defer mp.* = lm;
    const m = &lm;
    const r: i32 = @intCast(range);
    const left: u32 = @intCast(@max(cx - r, 0));
    const right: u32 = @intCast(@min(cx + r, @as(i32, @intCast(w - 8))));
    const top: u32 = @intCast(@max(cy - r, 0));
    const bottom: u32 = @intCast(@min(cy + r, @as(i32, @intCast(h - 8))));
    if (e == .ssd or e == .sad) {
        // Two passes per chunk of up to 64 groups of 8 candidates: compute
        // and keep the distances, then insert in raster order only those at
        // or below `bound`, the largest per-lane minimum. The 8 lane minima
        // are 8 distinct candidates of the chunk, so anything above the bound
        // has 8 strictly smaller ones and can't be in the (stable) top 8:
        // same result as inserting everything, with ~15 inserts per default
        // window instead of ~45. (An early reject on the rows 0..3 partial
        // was measured slower: within a top 8 of
        // similar noisy blocks the half distance rarely reaches the worst
        // kept one.) Plain locals, not a struct behind a pointer: see Match.
        //
        // Unused tail lanes are NaN: never below the lane minimum, never at
        // or below the bound, never inserted.
        const inf: V8 = @splat(std.math.inf(f32));
        var dbuf: [64]V8 = undefined;
        var gx: [64]u32 = undefined;
        var gy: [64]u32 = undefined;
        var n: u32 = 0;
        // per-lane minimum (NaN never wins: vminps(d, lmin))
        var lmin = inf;

        var y = top;
        while (y <= bottom) : (y += 1) {
            const row = plane.at(0, y);
            var x = left;
            while (x + 7 <= right) {
                const k = @min((right + 1 - x) / 8, 64 - n);
                lmin = fullGroups(e, rb, row + x, plane.stride, k, dbuf[n..].ptr, lmin);
                for (0..k) |i| {
                    gx[n + i] = x + 8 * @as(u32, @intCast(i));
                    gy[n + i] = y;
                }
                n += k;
                x += 8 * k;
                if (n == 64) {
                    insertBounded(m, keep, dbuf[0..n], gx[0..n], gy[0..n], lmin, z);
                    n = 0;
                    lmin = inf;
                }
            }
            if (x <= right) {
                // the 1..7 leftover candidates (reduce8x8 equals reduce8 per
                // lane). Up to 3 of them are reduced one by one: reduce8x8's
                // tree costs ~24 shuffles and adds whatever the count, ~6 per
                // candidate here. The 9-wide predictive windows leave 1 per
                // row, which the full tree made ~2.7x a grouped candidate
                // (r=2: ~15% of the time).
                @setEvalBranchQuota(20000);
                const rem = right - x + 1;
                const nan: V8 = @splat(std.math.nan(f32));
                const d: V8 = switch (@as(u3, @intCast(rem))) {
                    inline 1...3 => |cnt| blk: {
                        var v = nan;
                        inline for (0..cnt) |j| v[j] = reduce8(columnSums(e, rb, row + x + j, plane.stride));
                        break :blk v;
                    },
                    inline else => |cnt| blk: {
                        var cs: [8]V8 = @splat(@splat(0));
                        inline for (0..cnt) |j| cs[j] = columnSums(e, rb, row + x + j, plane.stride);
                        const used = std.simd.iota(u32, 8) < @as(@Vector(8, u32), @splat(cnt));
                        break :blk @select(f32, used, reduce8x8(&cs), nan);
                    },
                };
                lmin = @select(f32, d < lmin, d, lmin);
                dbuf[n] = d;
                gx[n] = x;
                gy[n] = y;
                n += 1;
                if (n == 64) {
                    insertBounded(m, keep, dbuf[0..n], gx[0..n], gy[0..n], lmin, z);
                    n = 0;
                    lmin = inf;
                }
            }
        }
        insertBounded(m, keep, dbuf[0..n], gx[0..n], gy[0..n], lmin, z);
        return;
    }
    var y = top;
    while (y <= bottom) : (y += 1) {
        const row = plane.at(0, y);
        var x = left;
        while (x <= right) : (x += 1) {
            m.insert(keep, distance(e, rb, row + x, plane.stride), @intCast(x), @intCast(y), z);
        }
    }
}

/// The group of block (x, y): spatial search, then (radius > 0) the
/// predictive search into the neighbor frames, then the reference block
/// itself at slot 0 when it was not matched.
fn matchBlock(comptime e: BmError, p: *const Params, mp: []const PlaneIn, x: u32, y: u32) Match {
    const r: i32 = @intCast(p.radius);
    const center = mp[p.radius];
    const rb = RefBlock.init(e, center.at(x, y), center.stride);
    var m: Match = .{ .z = @splat(r) };
    searchWindow(e, &m, 8, &rb, center, p.width, p.height, @intCast(x), @intCast(y), p.bm_range, r);

    if (p.radius > 0) {
        const cm = m;
        var dir: i32 = -1;
        while (dir <= 1) : (dir += 2) {
            var last = cm;
            var t: i32 = 1;
            while (t <= r) : (t += 1) {
                const z = r + dir * t;
                var f: Match = .{};
                const lx: [8]i32 = last.x;
                const ly: [8]i32 = last.y;
                // only f's first ps_num matches are used: keep that many
                // (rounded up to a comptime 1, 2, 4 or 8; more is still exact)
                for (0..p.ps_num) |i| switch (p.ps_num) {
                    1 => searchWindow(e, &f, 1, &rb, mp[@intCast(z)], p.width, p.height, lx[i], ly[i], p.ps_range, z),
                    2 => searchWindow(e, &f, 2, &rb, mp[@intCast(z)], p.width, p.height, lx[i], ly[i], p.ps_range, z),
                    3, 4 => searchWindow(e, &f, 4, &rb, mp[@intCast(z)], p.width, p.height, lx[i], ly[i], p.ps_range, z),
                    else => searchWindow(e, &f, 8, &rb, mp[@intCast(z)], p.width, p.height, lx[i], ly[i], p.ps_range, z),
                };
                const fe: [8]f32 = f.err;
                const fx: [8]i32 = f.x;
                const fy: [8]i32 = f.y;
                for (0..p.ps_num) |i| m.insert(8, fe[i], fx[i], fy[i], z);
                last = f;
            }
        }
    }

    const xi: i32 = @intCast(x);
    const yi: i32 = @intCast(y);
    const hx: u8 = @bitCast(m.x == @as(I8, @splat(xi)));
    const hy: u8 = @bitCast(m.y == @as(I8, @splat(yi)));
    const hz: u8 = @bitCast(m.z == @as(I8, @splat(r)));
    if (hx & hy & hz == 0) m.place(0, 0, xi, yi, r);
    return m;
}

// ---------------------------------------------------------------------------
// Collaborative filtering

fn loadGroup(g: *Group, planes: []const PlaneIn, m: *const Match) void {
    const mx: [8]i32 = m.x;
    const my: [8]i32 = m.y;
    const mz: [8]i32 = m.z;
    for (0..8) |b| {
        const pl = planes[@intCast(mz[b])];
        const p = pl.at(@intCast(mx[b]), @intCast(my[b]));
        inline for (0..8) |r| g[b * 8 + r] = load8(p + r * pl.stride);
    }
}

const lane0: @Vector(8, bool) = .{ true, false, false, false, false, false, false, false };

/// Hard thresholding; returns the aggregation weight 1/k.
fn hardThreshold(g: *Group, sigma: f32) f32 {
    const thr: V8 = @splat(sigma);
    // the DC coefficient (index 0, lane 0) is always kept
    const thr0 = @select(f32, lane0, @as(V8, @splat(0)), thr);
    var ks: [4]V8 = @splat(@splat(0));
    for (0..64) |i| {
        const v = g[i];
        const keep = @abs(v) >= (if (i == 0) thr0 else thr);
        ks[i % 4] += @select(f32, keep, @as(V8, @splat(1)), @as(V8, @splat(0)));
        g[i] = @select(f32, keep, v * @as(V8, @splat(1.0 / 4096.0)), @as(V8, @splat(0)));
    }
    const k = reduce8((ks[0] + ks[1]) + (ks[2] + ks[3]));
    return 1.0 / k;
}

/// Wiener shrinkage against the transformed basic estimate `rg`.
fn wiener(g: *Group, rg: *const Group, sigma: f32) f32 {
    const s2: V8 = @splat(sigma * sigma);
    var ks: [4]V8 = @splat(@splat(0));
    for (0..64) |i| {
        const r = rg[i];
        const num = r * r;
        var coeff = num / (num + s2);
        if (i == 0) coeff = @select(f32, lane0, @as(V8, @splat(1)), coeff);
        ks[i % 4] = fma(coeff, coeff, ks[i % 4]);
        g[i] = (g[i] * coeff) * @as(V8, @splat(1.0 / 4096.0));
    }
    const k = reduce8((ks[0] + ks[1]) + (ks[2] + ks[3]));
    return 1.0 / k;
}

fn accumulate(ring: *Ring, plane: u32, g: *const Group, m: *const Match, w: f32, extractor: f32, comptime temporal: bool) void {
    const wv: V8 = @splat(w);
    const ev: V8 = @splat(extractor);
    // with E = 0 the GPU still adds through fma(w, v, +0), which turns -0
    // into +0; the weight's (w + 0) - 0 is w
    const wt: V8 = if (extractor != 0) (wv + ev) - ev else wv;
    const mx: [8]i32 = m.x;
    const my: [8]i32 = m.y;
    const mz: [8]i32 = m.z;
    for (0..8) |b| {
        const x: u32 = @intCast(mx[b]);
        const y: u32 = @intCast(my[b]);
        // only non-finite input can push a member out of the band
        if (!ring.holds(y, 8)) continue;
        const slot: u32 = if (temporal) @intCast(mz[b]) else 0;
        var cur = ring.cursor(slot, plane, y);
        for (0..8) |r| {
            const val = g[b * 8 + r];
            const wd = if (extractor != 0) fma(wv, val, ev) - ev else fma(wv, val, @splat(0));
            const num = cur.numRow() + x;
            const den = cur.denRow() + x;
            num[0..8].* = load8(num) + wd;
            den[0..8].* = load8(den) + wt;
            cur.next();
        }
    }
}

/// One BM3D pass centered on window frame `radius`: matching on plane 0 of
/// `match_planes`, filtering of every processed plane, accumulation into
/// `ring` (one slot per window frame), rows handed to `sink` as they finish.
///
/// `srcs` / `refs`: [plane * slots + window frame]. `refs` (final estimate)
/// holds the basic estimate, which is also what the matching runs on.
pub fn stage(
    comptime final: bool,
    p: *const Params,
    srcs: []const PlaneIn,
    refs: []const PlaneIn,
    ring: *Ring,
    sink: anytype,
) void {
    switch (p.bm_error) {
        inline else => |e| stageImpl(final, e, p, srcs, refs, ring, sink),
    }
}

fn stageImpl(
    comptime final: bool,
    comptime e: BmError,
    p: *const Params,
    srcs: []const PlaneIn,
    refs: []const PlaneIn,
    ring: *Ring,
    sink: anytype,
) void {
    const tw = p.slots();
    const step = p.block_step;
    const reach = p.reach();
    const mp = if (final) refs[0..tw] else srcs[0..tw];
    ring.reset();

    var g: Group = undefined;
    var rg: Group = undefined;
    var _y: u32 = 0;
    while (_y < p.height - 8 + step) : (_y += step) {
        const y = @min(_y, p.height - 8);
        // nothing from here on reaches above y - reach
        ring.flush(y -| reach, sink);
        var _x: u32 = 0;
        while (_x < p.width - 8 + step) : (_x += step) {
            const x = @min(_x, p.width - 8);
            const m = matchBlock(e, p, mp, x, y);
            for (0..p.n_planes) |pl| {
                if (!p.process[pl]) continue;
                loadGroup(&g, srcs[pl * tw ..][0..tw], &m);
                transform3d(true, p.t2d, p.t1d, &g);
                const w = if (final) blk: {
                    loadGroup(&rg, refs[pl * tw ..][0..tw], &m);
                    transform3d(true, p.t2d, p.t1d, &rg);
                    break :blk wiener(&g, &rg, p.sigma[pl]);
                } else hardThreshold(&g, p.sigma[pl]);
                transform3d(false, p.t2d, p.t1d, &g);
                if (p.radius > 0) {
                    accumulate(ring, @intCast(pl), &g, &m, w, p.extractor, true);
                } else {
                    accumulate(ring, @intCast(pl), &g, &m, w, p.extractor, false);
                }
            }
        }
    }
    ring.flush(p.height, sink);
}

/// Spatial (radius 0) sink: each finished row is divided into the output
/// plane (correctly rounded, like the GPU's __fdiv_rn).
pub const DivideSink = struct {
    dst: [3]?[*]f32,
    /// in floats
    stride: [3]usize,

    pub fn rows(self: *const DivideSink, slot: u32, plane: u32, y: u32, num: []const f32, den: []const f32) void {
        std.debug.assert(slot == 0);
        const d = self.dst[plane] orelse return;
        const out = d + @as(usize, y) * self.stride[plane];
        divideRow(out[0..num.len], num, den);
    }
};

pub fn divideRow(out: []f32, num: []const f32, den: []const f32) void {
    var x: usize = 0;
    while (x + 8 <= out.len) : (x += 8) out[x..][0..8].* = load8(num.ptr + x) / load8(den.ptr + x);
    while (x < out.len) : (x += 1) out[x] = num[x] / den[x];
}

/// Per-thread scratch of one filter instance: a ring per entry.
pub const Scratch = struct {
    rings: [3]?Ring = .{ null, null, null },

    pub fn create() error{OutOfMemory}!*Scratch {
        const s = try allocator.create(Scratch);
        s.* = .{};
        return s;
    }

    pub fn destroy(self: *Scratch) void {
        for (&self.rings) |*r| {
            if (r.*) |*ring| ring.destroy();
        }
        allocator.destroy(self);
    }

    /// The ring of entry `idx`, allocated on first use.
    pub fn ringFor(self: *Scratch, idx: usize, p: *const Params) error{OutOfMemory}!*Ring {
        const r = &self.rings[idx];
        if (r.* == null) r.* = try Ring.create(p.slots(), p.n_planes, p.ringRows(), p.width, p.height);
        return &r.*.?;
    }
};
