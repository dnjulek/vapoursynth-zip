//! Core MPEG-2 / JPEG intra-block compression-artifact simulator.
//!
//! For each 8x8 block of an 8-bit plane we run the genuine FFmpeg pipeline:
//!   pixels -> forward DCT -> quantize -> dequantize -> inverse DCT -> pixels
//! There is no motion compensation and no bitstream; the artifacts come purely
//! from the quantization round-trip, exactly as an I-frame / JPEG image would.
//!
//! Everything is ported bit-faithfully from FFmpeg (libavcodec):
//!   - forward DCT : ff_jpeg_fdct_islow_8        (jfdctint_template.c)
//!   - quantize    : dct_quantize_c (intra path) (mpegvideo_enc.c)
//!   - qmat build  : ff_convert_matrix           (mpegvideo_enc.c)
//!   - dequantize  : dct_unquantize_mpeg2_intra_c (mpegvideo_unquantize.c)
//!   - inverse DCT : ff_simple_idct_int16_8bit   (simple_idct_template.c)
//!
//! The islow forward DCT leaves an overall factor of 8 in its output; the
//! MPEG-2 dequant (qscale<<=1 then >>4 == net /8) removes it, so the IDCT sees
//! true normalized coefficients. JPEG folds that /8 into its quant table.
//!
//! All inner arithmetic uses wrapping ops on i32 (the bit pattern is identical
//! to FFmpeg's mixed signed/`unsigned` math) and arithmetic right shifts, so
//! the result is bit-exact while never tripping Zig's overflow checks.
//!
//! The whole per-block DSP core runs lane-parallel on @Vector(8, i32): one
//! vector holds the same coefficient index of all 8 rows (or columns), so a
//! 1-D pass transforms the whole block at once, with in-register 8x8
//! transposes (simd.transposeI32x8) switching between row- and
//! column-parallel layouts. Every operation is a per-lane wrapping
//! add/sub/mul/shift with no cross-lane accumulation, so each lane computes
//! exactly the scalar reference expression — bit-exact by construction.
//! The scalar reference stores every stage back to a [64]i16 block; trunc16
//! reproduces those i16 store/reload wraps at the same points.

const std = @import("std");

const simd = @import("simd.zig");

pub const Codec = enum { mpeg2, jpeg };

// ===========================================================================
//  Quantization matrices / tables (natural raster order, as in FFmpeg source)
// ===========================================================================

/// ff_mpeg1_default_intra_matrix (mpeg12data.c) — used for luma and chroma.
pub const mpeg_intra_matrix = [64]i32{
    8,  16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83,
};

/// ff_mjpeg_std_luminance_quant_tbl (jpegquanttables.c).
pub const jpeg_luma_base = [64]i32{
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
};

/// ff_mjpeg_std_chrominance_quant_tbl (jpegquanttables.c).
pub const jpeg_chroma_base = [64]i32{
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
};

/// Per-instance precomputed tables, filled once in Create.
pub const QuantTables = struct {
    // --- MPEG-2 intra ---
    qmat: [64]i32 = undefined, // reciprocal: (2<<21)/(qscale2*matrix[i])
    deq: [64]i32 = undefined, // qscale2 * matrix[i]   (for dequant multiply)
    dc_q: i32 = 64, // encode DC divisor  = dc_scale << 3
    dc_scale: i32 = 8, // decode DC multiplier = 8 >> dc_prec

    // --- JPEG (idx 0 = luma, 1 = chroma) ---
    jqmat: [2][64]i32 = undefined, // reciprocal: (1<<21)/(8*qtab[i])
    jqtab: [2][64]i32 = undefined, // scaled quant table (for dequant multiply)

    /// MPEG-2: qscale in 1..31, dc_prec (intra_dc_precision) in 0..3.
    pub fn buildMpeg2(self: *QuantTables, qscale: i32, dc_prec: u5) void {
        const qscale2 = qscale << 1; // linear qscale path
        for (0..64) |i| {
            const den: i64 = @as(i64, qscale2) * mpeg_intra_matrix[i];
            self.qmat[i] = @intCast(@divTrunc(@as(i64, 2) << QMAT_SHIFT, den));
            self.deq[i] = qscale2 * mpeg_intra_matrix[i];
        }
        self.dc_scale = @as(i32, 8) >> dc_prec;
        self.dc_q = self.dc_scale << 3;
    }

    /// JPEG: quality in 1..100 (lower = more artifacts).
    pub fn buildJpeg(self: *QuantTables, quality: i32) void {
        const scale: i32 = if (quality < 50) @divTrunc(5000, quality) else 200 - quality * 2;
        const bases = [2]*const [64]i32{ &jpeg_luma_base, &jpeg_chroma_base };
        for (0..2) |p| {
            for (0..64) |i| {
                const q = std.math.clamp(@divTrunc(bases[p][i] * scale + 50, 100), 1, 255);
                self.jqtab[p][i] = q;
                // fold the islow FDCT's factor of 8 into the reciprocal
                self.jqmat[p][i] = @intCast(@divTrunc(@as(i64, 1) << 21, @as(i64, 8) * q));
            }
        }
    }
};

// ===========================================================================
//  Forward DCT — ff_jpeg_fdct_islow_8 (8-bit: CONST_BITS=13, PASS1_BITS=4)
// ===========================================================================

const CONST_BITS = 13;
const PASS1_BITS = 4;
const OUT_SHIFT = PASS1_BITS;
const PASS1_MUL = 1 << PASS1_BITS; // == FFmpeg's *(1<<PASS1_BITS)

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// One vector = one coefficient index across the 8 parallel rows/columns.
const V = @Vector(8, i32);
/// Quantizer products need 34 bits (i16 coeff × qmat ≤ 2^18).
const VU = @Vector(8, u64);

inline fn sp(comptime x: i32) V {
    return @splat(x);
}

/// Reproduce the scalar reference's store to the [64]i16 block between
/// stages: wrap each lane to i16 and sign-extend back to i32.
inline fn trunc16(v: V) V {
    const t: @Vector(8, i16) = @truncate(v);
    return @intCast(t);
}

inline fn descale(x: V, comptime n: comptime_int) V {
    return (x +% sp(1 << (n - 1))) >> @splat(n);
}

inline fn fdct1d(t: *[8]V, comptime out_round: comptime_int, comptime even_shift: comptime_int) void {
    const tmp0 = t[0] +% t[7];
    const tmp7 = t[0] -% t[7];
    const tmp1 = t[1] +% t[6];
    const tmp6 = t[1] -% t[6];
    const tmp2 = t[2] +% t[5];
    const tmp5 = t[2] -% t[5];
    const tmp3 = t[3] +% t[4];
    const tmp4 = t[3] -% t[4];

    // Even part
    const tmp10 = tmp0 +% tmp3;
    const tmp13 = tmp0 -% tmp3;
    const tmp11 = tmp1 +% tmp2;
    const tmp12 = tmp1 -% tmp2;

    t[0] = if (even_shift < 0) (tmp10 +% tmp11) *% sp(PASS1_MUL) else descale(tmp10 +% tmp11, even_shift);
    t[4] = if (even_shift < 0) (tmp10 -% tmp11) *% sp(PASS1_MUL) else descale(tmp10 -% tmp11, even_shift);

    var z1 = (tmp12 +% tmp13) *% sp(FIX_0_541196100);
    t[2] = descale(z1 +% tmp13 *% sp(FIX_0_765366865), out_round);
    t[6] = descale(z1 +% tmp12 *% sp(-FIX_1_847759065), out_round);

    // Odd part
    z1 = tmp4 +% tmp7;
    var z2 = tmp5 +% tmp6;
    var z3 = tmp4 +% tmp6;
    var z4 = tmp5 +% tmp7;
    const z5 = (z3 +% z4) *% sp(FIX_1_175875602);

    var o4 = tmp4 *% sp(FIX_0_298631336);
    var o5 = tmp5 *% sp(FIX_2_053119869);
    var o6 = tmp6 *% sp(FIX_3_072711026);
    var o7 = tmp7 *% sp(FIX_1_501321110);
    z1 = z1 *% sp(-FIX_0_899976223);
    z2 = z2 *% sp(-FIX_2_562915447);
    z3 = z3 *% sp(-FIX_1_961570560);
    z4 = z4 *% sp(-FIX_0_390180644);

    z3 +%= z5;
    z4 +%= z5;

    o4 +%= z1 +% z3;
    o5 +%= z2 +% z4;
    o6 +%= z2 +% z3;
    o7 +%= z1 +% z4;

    t[7] = descale(o4, out_round);
    t[5] = descale(o5, out_round);
    t[3] = descale(o6, out_round);
    t[1] = descale(o7, out_round);
}

/// Both islow passes over one block held as 8 row vectors (row-major in/out).
/// Each pass ends with the i16 wrap the scalar reference gets from storing
/// the pass back to its i16 block.
fn fdctIslow(rows: *[8]V) void {
    // Pass 1: rows (lanes = rows). Even outputs scaled by PASS1_BITS, odd
    // descaled by 9.
    var t = simd.transposeI32x8(rows.*);
    fdct1d(&t, CONST_BITS - PASS1_BITS, -1);
    for (&t) |*v| v.* = trunc16(v.*);
    // Pass 2: columns (lanes = columns). Removes PASS1_BITS, leaves overall
    // factor of 8.
    var u = simd.transposeI32x8(t);
    fdct1d(&u, CONST_BITS + OUT_SHIFT, OUT_SHIFT);
    for (&u) |*v| v.* = trunc16(v.*);
    rows.* = u;
}

// ===========================================================================
//  Quantize / dequantize
// ===========================================================================

const QMAT_SHIFT = 21;
const QUANT_BIAS_SHIFT = 8;
const INTRA_QUANT_BIAS = 3 << (QUANT_BIAS_SHIFT - 3); // == 96 for mpeg/jpeg
const MPEG_BIAS: i64 = INTRA_QUANT_BIAS * (1 << (QMAT_SHIFT - QUANT_BIAS_SHIFT)); // 96<<13
const MPEG_THRESH1: i64 = (1 << QMAT_SHIFT) - MPEG_BIAS - 1;
const MPEG_THRESH2: u64 = @as(u64, @intCast(MPEG_THRESH1)) << 1;
const JPEG_BIAS: i64 = 1 << (QMAT_SHIFT - 1); // symmetric round-to-nearest

// The scalar quantizers branch on sign and (mpeg2) deadzone; both reduce to
// a branchless magnitude form. qmat > 0, so sign(level) = sign(coeff) and
// |level| = |coeff| * qmat; the two sign arms `(BIAS + level) >> 21` and
// `-((BIAS - level) >> 21)` are both `±((BIAS + |level|) >> 21)`, and the
// mpeg2 window test `(u64)(level + THRESH1) > THRESH2` is exactly
// `|level| > THRESH1` (THRESH2 == 2*THRESH1). |level| ≥ 0 also makes the u64
// logical shift identical to the scalar's i64 arithmetic shift.

/// |coeff| * qmat as u64 lanes (the product needs 34 bits).
inline fn quantMag(b: V, qm: V) VU {
    return @as(VU, @intCast(@abs(b))) * @as(VU, @intCast(qm));
}

/// (bias + |level|) >> QMAT_SHIFT, narrowed back to i32 lanes (the result is
/// < 2^13, so the narrowing and the scalar's i16 store are both lossless).
inline fn quantLevel(mag: VU, comptime bias: u64) V {
    const shifted = (mag + @as(VU, @splat(bias))) >> @splat(QMAT_SHIFT);
    return @bitCast(@as(@Vector(8, u32), @truncate(shifted)));
}

/// MPEG-2 intra quantize (dct_quantize_c, intra path). Row-major in/out.
fn quantMpeg2(rows: *[8]V, qt: *const QuantTables) void {
    const dc: i32 = rows[0][0];
    for (0..8) |k| {
        // AC: deadzone threshold + rounding bias. Applying the threshold to
        // every coefficient in natural order is identical to FFmpeg's
        // scan-order last_non_zero search, since coeffs past the last
        // significant one all fail. (Lane 0 of k == 0 computes garbage from
        // qmat[0]; the DC patch below overwrites it.)
        const b = rows[k];
        const qm: V = qt.qmat[k * 8 ..][0..8].*;
        const mag = quantMag(b, qm);
        const keep = mag > @as(VU, @splat(MPEG_THRESH1));
        const level = quantLevel(mag, MPEG_BIAS);
        const signed = @select(i32, b < sp(0), -%level, level);
        rows[k] = @select(i32, keep, signed, sp(0));
    }
    // DC: special scale, block[0] assumed positive (FFmpeg comment).
    rows[0][0] = @as(i16, @intCast(@divTrunc(dc + (qt.dc_q >> 1), qt.dc_q)));
}

/// MPEG-2 intra dequantize (dct_unquantize_mpeg2_intra_c). qscale<<1 already
/// folded into qt.deq; the net >>4 with that doubling removes the FDCT's x8.
/// Branchless |level|*deq >> 4 with the sign restored by select; a zero lane
/// stays zero through the arithmetic, so the scalar's level == 0 skip needs
/// no separate case.
fn dequantMpeg2(rows: *[8]V, qt: *const QuantTables) void {
    const dc: i32 = rows[0][0];
    for (0..8) |k| {
        const lv = rows[k];
        const dq: V = qt.deq[k * 8 ..][0..8].*;
        const mag: V = @bitCast(@abs(lv)); // |lv| < 2^13 after quant
        const val = (mag *% dq) >> @splat(4);
        rows[k] = trunc16(@select(i32, lv < sp(0), -%val, val));
    }
    rows[0][0] = @as(i16, @truncate(dc *% qt.dc_scale));
}

/// JPEG quantize: plain round(coeff/(8*qtab)) over all 64 coefficients.
/// Same magnitude form; a zero coefficient yields (JPEG_BIAS >> 21) == 0, so
/// the scalar's explicit zero case folds into either sign arm.
fn quantJpeg(rows: *[8]V, qt: *const QuantTables, idx: usize) void {
    const m = &qt.jqmat[idx];
    for (0..8) |k| {
        const b = rows[k];
        const qm: V = m[k * 8 ..][0..8].*;
        const level = quantLevel(quantMag(b, qm), JPEG_BIAS);
        rows[k] = @select(i32, b < sp(0), -%level, level);
    }
}

/// JPEG dequantize: coeff = level * qtab (yields true DCT, the IDCT's scale).
fn dequantJpeg(rows: *[8]V, qt: *const QuantTables, idx: usize) void {
    const q = &qt.jqtab[idx];
    for (0..8) |k| {
        const dq: V = q[k * 8 ..][0..8].*;
        rows[k] = trunc16(rows[k] *% dq);
    }
}

// ===========================================================================
//  Inverse DCT — ff_simple_idct (8-bit: W1..W7, ROW_SHIFT=11, COL_SHIFT=20)
// ===========================================================================

const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19266;
const W4: i32 = 16383;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;
const ROW_SHIFT = 11;
const COL_SHIFT = 20;
const COL_DC_BIAS = (1 << (COL_SHIFT - 1)) / W4; // == 32

/// Row pass, row-major in/out; lanes = rows. The scalar (m4..m7) == 0 fast
/// path only skips adding zero terms, so the vector form always adds them —
/// identical. The DC-only fast path is a genuinely different rounding
/// (dc*8, not (W4*dc + rnd) >> 11), so it is kept, as a per-lane blend.
fn idctRows(rows: *[8]V) void {
    const t = simd.transposeI32x8(rows.*);
    const c0 = t[0];
    const c1 = t[1];
    const c2 = t[2];
    const c3 = t[3];
    const c4 = t[4];
    const c5 = t[5];
    const c6 = t[6];
    const c7 = t[7];

    const dc_only = (c1 | c2 | c3 | c4 | c5 | c6 | c7) == sp(0);
    const dc = trunc16(c0 *% sp(8));

    var a0 = c0 *% sp(W4) +% sp(1 << (ROW_SHIFT - 1));
    var a1 = a0;
    var a2 = a0;
    var a3 = a0;
    a0 +%= c2 *% sp(W2);
    a1 +%= c2 *% sp(W6);
    a2 -%= c2 *% sp(W6);
    a3 -%= c2 *% sp(W2);

    var b0 = c1 *% sp(W1) +% c3 *% sp(W3);
    var b1 = c1 *% sp(W3) -% c3 *% sp(W7);
    var b2 = c1 *% sp(W5) -% c3 *% sp(W1);
    var b3 = c1 *% sp(W7) -% c3 *% sp(W5);

    a0 +%= c4 *% sp(W4) +% c6 *% sp(W6);
    a1 +%= c4 *% sp(-W4) -% c6 *% sp(W2);
    a2 +%= c4 *% sp(-W4) +% c6 *% sp(W2);
    a3 +%= c4 *% sp(W4) -% c6 *% sp(W6);
    b0 +%= c5 *% sp(W5) +% c7 *% sp(W7);
    b1 +%= c5 *% sp(-W1) -% c7 *% sp(W5);
    b2 +%= c5 *% sp(W7) +% c7 *% sp(W3);
    b3 +%= c5 *% sp(W3) -% c7 *% sp(W1);

    var out: [8]V = undefined;
    out[0] = @select(i32, dc_only, dc, trunc16((a0 +% b0) >> @splat(ROW_SHIFT)));
    out[7] = @select(i32, dc_only, dc, trunc16((a0 -% b0) >> @splat(ROW_SHIFT)));
    out[1] = @select(i32, dc_only, dc, trunc16((a1 +% b1) >> @splat(ROW_SHIFT)));
    out[6] = @select(i32, dc_only, dc, trunc16((a1 -% b1) >> @splat(ROW_SHIFT)));
    out[2] = @select(i32, dc_only, dc, trunc16((a2 +% b2) >> @splat(ROW_SHIFT)));
    out[5] = @select(i32, dc_only, dc, trunc16((a2 -% b2) >> @splat(ROW_SHIFT)));
    out[3] = @select(i32, dc_only, dc, trunc16((a3 +% b3) >> @splat(ROW_SHIFT)));
    out[4] = @select(i32, dc_only, dc, trunc16((a3 -% b3) >> @splat(ROW_SHIFT)));
    rows.* = simd.transposeI32x8(out);
}

/// Column pass, row-major input; lanes = columns. Writes clamped pixels to
/// `out` (raster). offset: +128 for JPEG. The scalar c4..c7 != 0 guards only
/// skip adding zero terms, so the vector form always adds them — identical.
fn idctColsPut(rows: *const [8]V, out: *[64]u8, comptime offset: i32) void {
    const c0 = rows[0];
    const c1 = rows[1];
    const c2 = rows[2];
    const c3 = rows[3];
    const c4 = rows[4];
    const c5 = rows[5];
    const c6 = rows[6];
    const c7 = rows[7];

    var a0 = (c0 +% sp(COL_DC_BIAS)) *% sp(W4);
    var a1 = a0;
    var a2 = a0;
    var a3 = a0;
    a0 +%= c2 *% sp(W2);
    a1 +%= c2 *% sp(W6);
    a2 -%= c2 *% sp(W6);
    a3 -%= c2 *% sp(W2);

    var b0 = c1 *% sp(W1);
    var b1 = c1 *% sp(W3);
    var b2 = c1 *% sp(W5);
    var b3 = c1 *% sp(W7);
    b0 +%= c3 *% sp(W3);
    b1 -%= c3 *% sp(W7);
    b2 -%= c3 *% sp(W1);
    b3 -%= c3 *% sp(W5);

    a0 +%= c4 *% sp(W4);
    a1 -%= c4 *% sp(W4);
    a2 -%= c4 *% sp(W4);
    a3 +%= c4 *% sp(W4);
    b0 +%= c5 *% sp(W5);
    b1 -%= c5 *% sp(W1);
    b2 +%= c5 *% sp(W7);
    b3 +%= c5 *% sp(W3);
    a0 +%= c6 *% sp(W6);
    a1 -%= c6 *% sp(W2);
    a2 +%= c6 *% sp(W2);
    a3 -%= c6 *% sp(W6);
    b0 +%= c7 *% sp(W7);
    b1 -%= c7 *% sp(W5);
    b2 +%= c7 *% sp(W3);
    b3 -%= c7 *% sp(W1);

    // The clamp to [0, 255] lives inside packClampU8x4 (its pack saturations
    // are exactly the scalar reference's std.math.clamp(v, 0, 255)).
    out[0..32].* = simd.packClampU8x4(
        colPix(a0 +% b0, offset),
        colPix(a1 +% b1, offset),
        colPix(a2 +% b2, offset),
        colPix(a3 +% b3, offset),
    );
    out[32..64].* = simd.packClampU8x4(
        colPix(a3 -% b3, offset),
        colPix(a2 -% b2, offset),
        colPix(a1 -% b1, offset),
        colPix(a0 -% b0, offset),
    );
}

inline fn colPix(v: V, comptime offset: i32) V {
    return (v >> @splat(COL_SHIFT)) +% sp(offset);
}

// ===========================================================================
//  Per-block driver + per-plane loop
// ===========================================================================

inline fn processBlock(comptime codec: Codec, block: *[64]i16, out: *[64]u8, qt: *const QuantTables, idx: usize) void {
    var rows: [8]V = undefined;
    for (0..8) |r| {
        const t: @Vector(8, i16) = block[r * 8 ..][0..8].*;
        rows[r] = @intCast(t);
    }
    fdctIslow(&rows);
    switch (codec) {
        .mpeg2 => {
            quantMpeg2(&rows, qt);
            dequantMpeg2(&rows, qt);
        },
        .jpeg => {
            quantJpeg(&rows, qt, idx);
            dequantJpeg(&rows, qt, idx);
        },
    }
    idctRows(&rows);
    idctColsPut(&rows, out, if (codec == .jpeg) 128 else 0);
}

/// Compress one raw 8x8 pixel block (raster order). Exposed for golden tests;
/// production code (processPlane) routes its full-block path through the same
/// processBlock, so this exercises the identical DSP core.
pub fn compressBlock(comptime codec: Codec, src: *const [64]u8, out: *[64]u8, qt: *const QuantTables, idx: usize) void {
    var block: [64]i16 = undefined;
    const level: i16 = if (codec == .jpeg) 128 else 0;
    for (0..64) |i| block[i] = @as(i16, src[i]) - level;
    processBlock(codec, &block, out, qt, idx);
}

pub fn processPlane(
    comptime codec: Codec,
    srcp: []const u8,
    dstp: []u8,
    w: usize,
    h: usize,
    stride: usize,
    qt: *const QuantTables,
    is_chroma: bool,
) void {
    const idx: usize = if (is_chroma) 1 else 0;
    const level: i16 = if (codec == .jpeg) 128 else 0;

    var by: usize = 0;
    while (by < h) : (by += 8) {
        var bx: usize = 0;
        while (bx < w) : (bx += 8) {
            var block: [64]i16 = undefined;
            var out: [64]u8 = undefined;

            const full = (bx + 8 <= w) and (by + 8 <= h);
            if (full) {
                // Vector row load: the unrolled scalar form never SLP-vectorizes
                // (measured ~192 scalar instructions per block vs ~24).
                const level_v: @Vector(8, i16) = @splat(level);
                for (0..8) |yy| {
                    const row = (by + yy) * stride + bx;
                    const v: @Vector(8, u8) = srcp[row..][0..8].*;
                    block[yy * 8 ..][0..8].* = @as(@Vector(8, i16), v) - level_v;
                }
            } else {
                for (0..8) |yy| {
                    const sy = @min(by + yy, h - 1);
                    for (0..8) |xx| {
                        const sx = @min(bx + xx, w - 1);
                        block[yy * 8 + xx] = @as(i16, srcp[sy * stride + sx]) - level;
                    }
                }
            }

            processBlock(codec, &block, &out, qt, idx);

            if (full) {
                // Launder the block buffer: LLVM otherwise forwards the
                // packed 32-byte stores from idctColsPut into these row
                // loads and rebuilds each 8-byte row with per-byte
                // extract/shift/or scalar code (~220 instr/block measured).
                const outp = simd.opaquePtr(u8, &out);
                for (0..8) |yy| {
                    const row = (by + yy) * stride + bx;
                    dstp[row..][0..8].* = outp[yy * 8 ..][0..8].*;
                }
            } else {
                for (0..8) |yy| {
                    const dy = by + yy;
                    if (dy >= h) break;
                    const row = dy * stride;
                    for (0..8) |xx| {
                        const dx = bx + xx;
                        if (dx >= w) break;
                        dstp[row + dx] = out[yy * 8 + xx];
                    }
                }
            }
        }
    }
}
