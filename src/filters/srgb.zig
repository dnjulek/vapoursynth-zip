//! sRGB transfer function (IEC 61966-2-1) decode, ported from libjxl
//! `TF_SRGB::DisplayFromEncoded` (lib/jxl/cms/transfer_functions-inl.h), the
//! path jxl's CMS takes for sRGB -> linear sRGB and therefore what the
//! reference ssimulacra2 tool feeds its XYB conversion.

const std = @import("std");

// Computed via af_cheb_rational (k=100) in libjxl; p/q in ascending degree.
const p = [5]f32{ 2.200248328e-04, 1.043637593e-02, 1.624820318e-01, 7.961564959e-01, 8.210152774e-01 };
const q = [5]f32{ 2.631846970e-01, 1.076976492e+00, 4.987528350e-01, -5.512498495e-02, 6.521209011e-03 };
const thresh: f32 = 0.04045;
// 1.0f / 12.92f in f32 arithmetic, as libjxl's constexpr kLowDivInv
const low_div_inv: f32 = @as(f32, 1.0) / @as(f32, 12.92);

/// Encoded sRGB -> linear light, lane-wise on any f32 vector. Bit-exact with
/// libjxl on FMA targets: Horner with fused multiply-adds (hwy MulAdd), one
/// true division, x / 12.92 as x * (1/12.92) below the threshold, sign
/// mirrored (odd extension).
pub inline fn toLinear(v: anytype) @TypeOf(v) {
    const V = @TypeOf(v);
    const n = @typeInfo(V).vector.len;
    const U = @Vector(n, u32);
    const sign: U = @splat(0x80000000);
    const bits: U = @bitCast(v);
    const s = bits & sign;
    const x: V = @bitCast(bits & ~sign);

    var yp: V = @splat(p[4]);
    var yq: V = @splat(q[4]);
    inline for (0..4) |i| {
        yp = @mulAdd(V, yp, x, @splat(p[3 - i]));
        yq = @mulAdd(V, yq, x, @splat(q[3 - i]));
    }
    const poly = yp / yq;
    const lin = x * @as(V, @splat(low_div_inv));
    const mag: U = @bitCast(@select(f32, x > @as(V, @splat(thresh)), poly, lin));
    return @bitCast((mag & ~sign) | s);
}
