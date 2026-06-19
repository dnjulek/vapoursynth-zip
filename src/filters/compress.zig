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

const std = @import("std");

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

inline fn descale(x: i32, comptime n: comptime_int) i32 {
    return (x +% (1 << (n - 1))) >> n;
}

inline fn fdct1d(t: *[8]i32, comptime out_round: comptime_int, comptime even_shift: comptime_int) void {
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

    t[0] = if (even_shift < 0) (tmp10 +% tmp11) *% PASS1_MUL else descale(tmp10 +% tmp11, even_shift);
    t[4] = if (even_shift < 0) (tmp10 -% tmp11) *% PASS1_MUL else descale(tmp10 -% tmp11, even_shift);

    var z1 = (tmp12 +% tmp13) *% FIX_0_541196100;
    t[2] = descale(z1 +% tmp13 *% FIX_0_765366865, out_round);
    t[6] = descale(z1 +% tmp12 *% (-FIX_1_847759065), out_round);

    // Odd part
    z1 = tmp4 +% tmp7;
    var z2 = tmp5 +% tmp6;
    var z3 = tmp4 +% tmp6;
    var z4 = tmp5 +% tmp7;
    const z5 = (z3 +% z4) *% FIX_1_175875602;

    var o4 = tmp4 *% FIX_0_298631336;
    var o5 = tmp5 *% FIX_2_053119869;
    var o6 = tmp6 *% FIX_3_072711026;
    var o7 = tmp7 *% FIX_1_501321110;
    z1 = z1 *% (-FIX_0_899976223);
    z2 = z2 *% (-FIX_2_562915447);
    z3 = z3 *% (-FIX_1_961570560);
    z4 = z4 *% (-FIX_0_390180644);

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

pub fn fdctIslow(block: *[64]i16) void {
    // Pass 1: rows. Even outputs scaled by PASS1_BITS, odd descaled by 9.
    for (0..8) |r| {
        var t: [8]i32 = undefined;
        for (0..8) |c| t[c] = block[r * 8 + c];
        fdct1d(&t, CONST_BITS - PASS1_BITS, -1);
        for (0..8) |c| block[r * 8 + c] = @truncate(t[c]);
    }
    // Pass 2: columns. Removes PASS1_BITS, leaves overall factor of 8.
    for (0..8) |c| {
        var t: [8]i32 = undefined;
        for (0..8) |r| t[r] = block[r * 8 + c];
        fdct1d(&t, CONST_BITS + OUT_SHIFT, OUT_SHIFT);
        for (0..8) |r| block[r * 8 + c] = @truncate(t[r]);
    }
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

/// MPEG-2 intra quantize (dct_quantize_c, intra path). block in natural order.
pub fn quantMpeg2(block: *[64]i16, qt: *const QuantTables) void {
    // DC: special scale, block[0] assumed positive (FFmpeg comment).
    block[0] = @intCast(@divTrunc(@as(i32, block[0]) + (qt.dc_q >> 1), qt.dc_q));

    // AC: deadzone threshold + rounding bias. Applying the threshold to every
    // coefficient in natural order is identical to FFmpeg's scan-order
    // last_non_zero search, since coeffs past the last significant one all fail.
    var i: usize = 1;
    while (i < 64) : (i += 1) {
        const level: i64 = @as(i64, block[i]) * qt.qmat[i];
        if (@as(u64, @bitCast(level + MPEG_THRESH1)) > MPEG_THRESH2) {
            block[i] = if (level > 0)
                @intCast((MPEG_BIAS + level) >> QMAT_SHIFT)
            else
                @intCast(-((MPEG_BIAS - level) >> QMAT_SHIFT));
        } else {
            block[i] = 0;
        }
    }
}

/// MPEG-2 intra dequantize (dct_unquantize_mpeg2_intra_c). qscale<<1 already
/// folded into qt.deq; the net >>4 with that doubling removes the FDCT's x8.
pub fn dequantMpeg2(block: *[64]i16, qt: *const QuantTables) void {
    block[0] = @truncate(@as(i32, block[0]) *% qt.dc_scale);
    var i: usize = 1;
    while (i < 64) : (i += 1) {
        var level: i32 = block[i];
        if (level != 0) {
            if (level < 0) {
                level = -level;
                level = (level *% qt.deq[i]) >> 4;
                level = -level;
            } else {
                level = (level *% qt.deq[i]) >> 4;
            }
            block[i] = @truncate(level);
        }
    }
}

/// JPEG quantize: plain round(coeff/(8*qtab)) over all 64 coefficients.
pub fn quantJpeg(block: *[64]i16, qt: *const QuantTables, idx: usize) void {
    const m = &qt.jqmat[idx];
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const level: i64 = @as(i64, block[i]) * m[i];
        block[i] = if (level > 0)
            @intCast((JPEG_BIAS + level) >> QMAT_SHIFT)
        else if (level < 0)
            @intCast(-((JPEG_BIAS - level) >> QMAT_SHIFT))
        else
            0;
    }
}

/// JPEG dequantize: coeff = level * qtab (yields true DCT, the IDCT's scale).
pub fn dequantJpeg(block: *[64]i16, qt: *const QuantTables, idx: usize) void {
    const q = &qt.jqtab[idx];
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        block[i] = @truncate(@as(i32, block[i]) *% q[i]);
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

/// Row pass, in place, with the DC-only fast path (left-shift only, no >>SHIFT).
fn idctRows(block: *[64]i16) void {
    for (0..8) |r| {
        const o = r * 8;
        const m1 = block[o + 1];
        const m2 = block[o + 2];
        const m3 = block[o + 3];
        const m4 = block[o + 4];
        const m5 = block[o + 5];
        const m6 = block[o + 6];
        const m7 = block[o + 7];

        if ((m1 | m2 | m3 | m4 | m5 | m6 | m7) == 0) {
            const dc: i16 = @truncate(@as(i32, block[o]) *% 8);
            for (0..8) |c| block[o + c] = dc;
            continue;
        }

        const c0: i32 = block[o];
        const c1: i32 = m1;
        const c2: i32 = m2;
        const c3: i32 = m3;

        var a0 = W4 *% c0 +% (1 << (ROW_SHIFT - 1));
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        a0 +%= W2 *% c2;
        a1 +%= W6 *% c2;
        a2 -%= W6 *% c2;
        a3 -%= W2 *% c2;

        var b0 = W1 *% c1 +% W3 *% c3;
        var b1 = W3 *% c1 -% W7 *% c3;
        var b2 = W5 *% c1 -% W1 *% c3;
        var b3 = W7 *% c1 -% W5 *% c3;

        if ((m4 | m5 | m6 | m7) != 0) {
            const c4: i32 = m4;
            const c5: i32 = m5;
            const c6: i32 = m6;
            const c7: i32 = m7;
            a0 +%= W4 *% c4 +% W6 *% c6;
            a1 +%= -W4 *% c4 -% W2 *% c6;
            a2 +%= -W4 *% c4 +% W2 *% c6;
            a3 +%= W4 *% c4 -% W6 *% c6;
            b0 +%= W5 *% c5 +% W7 *% c7;
            b1 +%= -W1 *% c5 -% W5 *% c7;
            b2 +%= W7 *% c5 +% W3 *% c7;
            b3 +%= W3 *% c5 -% W1 *% c7;
        }

        block[o + 0] = @truncate((a0 +% b0) >> ROW_SHIFT);
        block[o + 7] = @truncate((a0 -% b0) >> ROW_SHIFT);
        block[o + 1] = @truncate((a1 +% b1) >> ROW_SHIFT);
        block[o + 6] = @truncate((a1 -% b1) >> ROW_SHIFT);
        block[o + 2] = @truncate((a2 +% b2) >> ROW_SHIFT);
        block[o + 5] = @truncate((a2 -% b2) >> ROW_SHIFT);
        block[o + 3] = @truncate((a3 +% b3) >> ROW_SHIFT);
        block[o + 4] = @truncate((a3 -% b3) >> ROW_SHIFT);
    }
}

/// Column pass, writes clamped pixels to `out` (raster). offset: +128 for JPEG.
fn idctColsPut(block: *const [64]i16, out: *[64]u8, comptime offset: i32) void {
    for (0..8) |c| {
        const c0: i32 = block[c + 8 * 0];
        const c1: i32 = block[c + 8 * 1];
        const c2: i32 = block[c + 8 * 2];
        const c3: i32 = block[c + 8 * 3];
        const c4: i32 = block[c + 8 * 4];
        const c5: i32 = block[c + 8 * 5];
        const c6: i32 = block[c + 8 * 6];
        const c7: i32 = block[c + 8 * 7];

        var a0 = W4 *% (c0 +% COL_DC_BIAS);
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        a0 +%= W2 *% c2;
        a1 +%= W6 *% c2;
        a2 -%= W6 *% c2;
        a3 -%= W2 *% c2;

        var b0 = W1 *% c1;
        var b1 = W3 *% c1;
        var b2 = W5 *% c1;
        var b3 = W7 *% c1;
        b0 +%= W3 *% c3;
        b1 -%= W7 *% c3;
        b2 -%= W1 *% c3;
        b3 -%= W5 *% c3;

        if (c4 != 0) {
            a0 +%= W4 *% c4;
            a1 -%= W4 *% c4;
            a2 -%= W4 *% c4;
            a3 +%= W4 *% c4;
        }
        if (c5 != 0) {
            b0 +%= W5 *% c5;
            b1 -%= W1 *% c5;
            b2 +%= W7 *% c5;
            b3 +%= W3 *% c5;
        }
        if (c6 != 0) {
            a0 +%= W6 *% c6;
            a1 -%= W2 *% c6;
            a2 +%= W2 *% c6;
            a3 -%= W6 *% c6;
        }
        if (c7 != 0) {
            b0 +%= W7 *% c7;
            b1 -%= W5 *% c7;
            b2 +%= W3 *% c7;
            b3 -%= W1 *% c7;
        }

        out[c + 8 * 0] = clipU8(((a0 +% b0) >> COL_SHIFT) +% offset);
        out[c + 8 * 1] = clipU8(((a1 +% b1) >> COL_SHIFT) +% offset);
        out[c + 8 * 2] = clipU8(((a2 +% b2) >> COL_SHIFT) +% offset);
        out[c + 8 * 3] = clipU8(((a3 +% b3) >> COL_SHIFT) +% offset);
        out[c + 8 * 4] = clipU8(((a3 -% b3) >> COL_SHIFT) +% offset);
        out[c + 8 * 5] = clipU8(((a2 -% b2) >> COL_SHIFT) +% offset);
        out[c + 8 * 6] = clipU8(((a1 -% b1) >> COL_SHIFT) +% offset);
        out[c + 8 * 7] = clipU8(((a0 -% b0) >> COL_SHIFT) +% offset);
    }
}

inline fn clipU8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

// ===========================================================================
//  Per-block driver + per-plane loop
// ===========================================================================

inline fn processBlock(comptime codec: Codec, block: *[64]i16, out: *[64]u8, qt: *const QuantTables, idx: usize) void {
    fdctIslow(block);
    switch (codec) {
        .mpeg2 => {
            quantMpeg2(block, qt);
            dequantMpeg2(block, qt);
        },
        .jpeg => {
            quantJpeg(block, qt, idx);
            dequantJpeg(block, qt, idx);
        },
    }
    idctRows(block);
    idctColsPut(block, out, if (codec == .jpeg) 128 else 0);
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
                for (0..8) |yy| {
                    const row = (by + yy) * stride + bx;
                    inline for (0..8) |xx| block[yy * 8 + xx] = @as(i16, srcp[row + xx]) - level;
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
                for (0..8) |yy| {
                    const row = (by + yy) * stride + bx;
                    inline for (0..8) |xx| dstp[row + xx] = out[yy * 8 + xx];
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
