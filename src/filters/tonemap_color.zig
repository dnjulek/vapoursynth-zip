//! Color science behind Tonemap: a scalar port of libplacebo's colorspace.c
//! (metadata inference, nominal luminance, primaries), tone_mapping.c (the
//! 1D tone curves) and gamut_mapping.c (the IPT gamut mappings), from
//! libplacebo v7.387.0 (275f7de, API 387) as used by vs-placebo 2.0.4's
//! Tonemap. Everything here runs
//! per frame (tone LUT, 256 entries) or once per input gamut (the 3D gamut
//! LUT); filters/tonemap.zig samples the results per pixel.
//!
//! Same logic as the C code, in f32 like it; std.math replaces libm, so the
//! LUTs agree with libplacebo's to float rounding, not bit for bit.

const std = @import("std");
const math = std.math;

pub const sdr_white: f32 = 203.0;
pub const sdr_contrast: f32 = 1000.0;
pub const hdr_black: f32 = 1e-6;
pub const hlg_peak: f32 = 1000.0;

pub const pq_m1: f32 = 2610.0 / 4096.0 * 1.0 / 4.0;
pub const pq_m2: f32 = 2523.0 / 4096.0 * 128.0;
pub const pq_c1: f32 = 3424.0 / 4096.0;
pub const pq_c2: f32 = 2413.0 / 4096.0 * 32.0;
pub const pq_c3: f32 = 2392.0 / 4096.0 * 32.0;

pub const hlg_a: f32 = 0.17883277;
pub const hlg_b: f32 = 0.28466892;
pub const hlg_c: f32 = 0.55991073;
pub const hlg_ref: f32 = 1000.0 / sdr_white;
const hlg_75: f32 = 3.17955;

inline fn powf(x: f32, y: f32) f32 {
    return math.pow(f32, x, y);
}

inline fn mix(a: f32, b: f32, x: f32) f32 {
    return x * b + (1 - x) * a;
}

inline fn clampf(x: f32, lo: f32, hi: f32) f32 {
    return @min(@max(x, lo), hi);
}

pub fn smoothstep(e0: f32, e1: f32, x_in: f32) f32 {
    if (e0 == e1) return if (x_in >= e0) 1 else 0;
    const x = clampf((x_in - e0) / (e1 - e0), 0, 1);
    return x * x * (3 - 2 * x);
}

// ---------------------------------------------------------------- matrices

pub const Mat3 = [3][3]f32;

pub const identity: Mat3 = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } };

/// a * b
pub fn mul(a: Mat3, b: Mat3) Mat3 {
    var r: Mat3 = undefined;
    for (0..3) |i| {
        for (0..3) |j| r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
    return r;
}

pub fn invert(m: Mat3) Mat3 {
    const m00: f64 = m[0][0];
    const m01: f64 = m[0][1];
    const m02: f64 = m[0][2];
    const m10: f64 = m[1][0];
    const m11: f64 = m[1][1];
    const m12: f64 = m[1][2];
    const m20: f64 = m[2][0];
    const m21: f64 = m[2][1];
    const m22: f64 = m[2][2];
    const a00 = m11 * m22 - m21 * m12;
    const a01 = -(m01 * m22 - m21 * m02);
    const a02 = m01 * m12 - m11 * m02;
    const a10 = -(m10 * m22 - m20 * m12);
    const a11 = m00 * m22 - m20 * m02;
    const a12 = -(m00 * m12 - m10 * m02);
    const a20 = m10 * m21 - m20 * m11;
    const a21 = -(m00 * m21 - m20 * m01);
    const a22 = m00 * m11 - m10 * m01;
    const det = 1.0 / (m00 * a00 + m10 * a01 + m20 * a02);
    return .{
        .{ @floatCast(det * a00), @floatCast(det * a01), @floatCast(det * a02) },
        .{ @floatCast(det * a10), @floatCast(det * a11), @floatCast(det * a12) },
        .{ @floatCast(det * a20), @floatCast(det * a21), @floatCast(det * a22) },
    };
}

pub fn apply(m: Mat3, v: [3]f32) [3]f32 {
    return .{
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    };
}

pub fn scale(m: Mat3, s: f32) Mat3 {
    var r = m;
    for (&r) |*row| {
        for (row) |*e| e.* *= s;
    }
    return r;
}

// --------------------------------------------------------------- primaries

pub const Xy = struct { x: f32 = 0, y: f32 = 0 };

pub const Raw = struct {
    red: Xy = .{},
    green: Xy = .{},
    blue: Xy = .{},
    white: Xy = .{},

    pub fn eql(a: Raw, b: Raw) bool {
        return xyEql(a.red, b.red) and xyEql(a.green, b.green) and xyEql(a.blue, b.blue) and xyEql(a.white, b.white);
    }

    /// Zero fields take `b`'s values (pl_raw_primaries_merge).
    pub fn merge(a: *Raw, b: Raw) void {
        const fields = .{ "red", "green", "blue", "white" };
        inline for (fields) |f| {
            if (@field(a, f).x == 0) @field(a, f).x = @field(b, f).x;
            if (@field(a, f).y == 0) @field(a, f).y = @field(b, f).y;
        }
    }
};

fn xyEql(a: Xy, b: Xy) bool {
    return a.x == b.x and a.y == b.y;
}

/// libplacebo's `enum pl_color_primaries` (vs-placebo's `dst_prim`).
pub const Prim = enum(u8) {
    unknown = 0,
    bt601_525,
    bt601_625,
    bt709,
    bt470m,
    ebu3213,
    bt2020,
    apple,
    adobe,
    pro_photo,
    cie1931,
    dci_p3,
    display_p3,
    v_gamut,
    s_gamut,
    film_c,
    aces_ap0,
    aces_ap1,

    pub const count = 18;

    pub fn isWideGamut(p: Prim) bool {
        return @intFromEnum(p) >= @intFromEnum(Prim.bt2020);
    }
};

const d50: Xy = .{ .x = 0.3457, .y = 0.3585 };
const d65: Xy = .{ .x = 0.3127, .y = 0.3290 };
const cie_c: Xy = .{ .x = 0.3100, .y = 0.3160 };
const cie_e: Xy = .{ .x = 1.0 / 3.0, .y = 1.0 / 3.0 };
const dci: Xy = .{ .x = 0.3140, .y = 0.3510 };
const aces_w: Xy = .{ .x = 0.32168, .y = 0.33767 };

fn rp(rx: f32, ry: f32, gx: f32, gy: f32, bx: f32, by: f32, w: Xy) Raw {
    return .{ .red = .{ .x = rx, .y = ry }, .green = .{ .x = gx, .y = gy }, .blue = .{ .x = bx, .y = by }, .white = w };
}

pub fn rawPrimaries(p: Prim) Raw {
    return switch (p) {
        .unknown, .bt709 => rp(0.640, 0.330, 0.300, 0.600, 0.150, 0.060, d65),
        .bt601_525 => rp(0.630, 0.340, 0.310, 0.595, 0.155, 0.070, d65),
        .bt601_625 => rp(0.640, 0.330, 0.290, 0.600, 0.150, 0.060, d65),
        .bt470m => rp(0.670, 0.330, 0.210, 0.710, 0.140, 0.080, cie_c),
        .ebu3213 => rp(0.630, 0.340, 0.295, 0.605, 0.155, 0.077, d65),
        .bt2020 => rp(0.708, 0.292, 0.170, 0.797, 0.131, 0.046, d65),
        .apple => rp(0.625, 0.340, 0.280, 0.595, 0.115, 0.070, d65),
        .adobe => rp(0.640, 0.330, 0.210, 0.710, 0.150, 0.060, d65),
        .pro_photo => rp(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001, d50),
        .cie1931 => rp(0.7347, 0.2653, 0.2738, 0.7174, 0.1666, 0.0089, cie_e),
        .dci_p3 => rp(0.680, 0.320, 0.265, 0.690, 0.150, 0.060, dci),
        .display_p3 => rp(0.680, 0.320, 0.265, 0.690, 0.150, 0.060, d65),
        .v_gamut => rp(0.730, 0.280, 0.165, 0.840, 0.100, -0.03, d65),
        .s_gamut => rp(0.730, 0.280, 0.140, 0.855, 0.100, -0.05, d65),
        .film_c => rp(0.681, 0.319, 0.243, 0.692, 0.145, 0.049, cie_c),
        .aces_ap0 => rp(0.7347, 0.2653, 0.0000, 1.0000, 0.0001, -0.0770, aces_w),
        .aces_ap1 => rp(0.713, 0.293, 0.165, 0.830, 0.128, 0.044, aces_w),
    };
}

inline fn cieX(xy: Xy) f32 {
    return xy.x / xy.y;
}

inline fn cieZ(xy: Xy) f32 {
    return (1 - xy.x - xy.y) / xy.y;
}

pub fn rgb2xyz(p: Raw) Mat3 {
    const X = [4]f32{ cieX(p.red), cieX(p.green), cieX(p.blue), cieX(p.white) };
    const Z = [4]f32{ cieZ(p.red), cieZ(p.green), cieZ(p.blue), cieZ(p.white) };
    var m: Mat3 = undefined;
    for (0..3) |i| {
        m[0][i] = X[i];
        m[1][i] = 1;
        m[2][i] = Z[i];
    }
    m = invert(m);
    var S: [3]f32 = undefined;
    for (0..3) |i| S[i] = m[i][0] * X[3] + m[i][1] + m[i][2] * Z[3];
    for (0..3) |i| {
        m[0][i] = S[i] * X[i];
        m[1][i] = S[i];
        m[2][i] = S[i] * Z[i];
    }
    return m;
}

const m_cat16: Mat3 = .{
    .{ 0.401288, 0.650173, -0.051461 },
    .{ -0.250268, 1.204414, 0.045854 },
    .{ -0.002079, 0.048952, 0.953127 },
};

/// mat := mat * XYZd<-XYZs (linear von Kries with CAT16)
fn chromaticAdaptation(src: Xy, dst: Xy, mat: *Mat3) void {
    if (@abs(src.x - dst.x) < 1e-6 and @abs(src.y - dst.y) < 1e-6) return;
    var tmp: Mat3 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    for (0..3) |i| {
        const cs = m_cat16[i][0] * cieX(src) + m_cat16[i][1] + m_cat16[i][2] * cieZ(src);
        const cd = m_cat16[i][0] * cieX(dst) + m_cat16[i][1] + m_cat16[i][2] * cieZ(dst);
        tmp[i][i] = cd / cs;
    }
    tmp = mul(tmp, m_cat16);
    mat.* = mul(mul(mat.*, invert(m_cat16)), tmp);
}

pub fn adaptationMatrix(src: Xy, dst: Xy) Mat3 {
    var csp = rawPrimaries(.bt709);
    csp.white = src;
    const to_xyz = rgb2xyz(csp);
    var m = invert(to_xyz);
    chromaticAdaptation(src, dst, &m);
    return mul(m, to_xyz);
}

pub fn iptRgb2Lms(p: Raw) Mat3 {
    const hpe: Mat3 = .{
        .{ 0.40024, 0.70760, -0.08081 },
        .{ -0.22630, 1.16532, 0.04570 },
        .{ 0.00000, 0.00000, 0.91822 },
    };
    const c: f32 = 0.04;
    var m: Mat3 = .{ .{ 1 - 2 * c, c, c }, .{ c, 1 - 2 * c, c }, .{ c, c, 1 - 2 * c } };
    m = mul(m, hpe);
    chromaticAdaptation(p.white, d65, &m);
    return mul(m, rgb2xyz(p));
}

pub fn iptLms2Rgb(p: Raw) Mat3 {
    return invert(iptRgb2Lms(p));
}

pub const lms2ipt: Mat3 = .{
    .{ 0.4000, 0.4000, 0.2000 },
    .{ 4.4550, -4.8510, 0.3960 },
    .{ 0.8056, 0.3572, -1.1628 },
};

pub const ipt2lms: Mat3 = .{
    .{ 1.0, 0.0975689, 0.205226 },
    .{ 1.0, -0.1138760, 0.133217 },
    .{ 1.0, 0.0326151, -0.676887 },
};

fn testPointLine(p: Xy, a: Xy, b: Xy) f32 {
    return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
}

fn testPointGamut(point: Xy, prim: Raw) bool {
    const d1 = testPointLine(point, prim.red, prim.green);
    const d2 = testPointLine(point, prim.green, prim.blue);
    const d3 = testPointLine(point, prim.blue, prim.red);
    const has_neg = d1 < -1e-6 or d2 < -1e-6 or d3 < -1e-6;
    const has_pos = d1 > 1e-6 or d2 > 1e-6 or d3 > 1e-6;
    return !(has_neg and has_pos);
}

/// `a` contains `b`.
pub fn primariesSuperset(a: Raw, b: Raw) bool {
    return testPointGamut(b.red, a) and testPointGamut(b.green, a) and testPointGamut(b.blue, a);
}

/// The white point strictly inside the primaries' triangle (otherwise the
/// RGB<->XYZ matrix is singular); also rejects a degenerate triangle.
pub fn primariesValid(p: Raw) bool {
    const d1 = testPointLine(p.white, p.red, p.green);
    const d2 = testPointLine(p.white, p.green, p.blue);
    const d3 = testPointLine(p.white, p.blue, p.red);
    return (d1 > 1e-6 and d2 > 1e-6 and d3 > 1e-6) or (d1 < -1e-6 and d2 < -1e-6 and d3 < -1e-6);
}

/// RGBd <- RGBs, relative colorimetric (pl_get_color_mapping_matrix).
pub fn colorMappingMatrix(src: Raw, dst: Raw) Mat3 {
    var m = invert(rgb2xyz(dst));
    chromaticAdaptation(src.white, dst.white, &m);
    return mul(m, rgb2xyz(src));
}

/// libplacebo's use_mastering_space: tone mapping (and the gamut map's
/// source extent, and peak detection) run in the mastering display's RGB
/// when its primaries are valid, differ from the container's and lie within
/// it. Returns container RGB -> mastering RGB.
pub fn masteringSpace(c: *const Csp) ?Mat3 {
    const cont = rawPrimaries(c.prim);
    const mast = c.hdr.prim;
    if (!primariesValid(mast) or mast.eql(cont)) return null;
    const m2c = colorMappingMatrix(mast, cont);
    for (m2c) |row| for (row) |v| if (v < -0.01) return null;
    return colorMappingMatrix(cont, mast);
}

fn dist2(a: Xy, b: Xy) f32 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return dx * dx + dy * dy;
}

pub fn primariesCompatible(a: Raw, b: Raw) bool {
    const RR = dist2(a.red, b.red);
    const RG = dist2(a.red, b.green);
    const RB = dist2(a.red, b.blue);
    const GR = dist2(a.green, b.red);
    const GG = dist2(a.green, b.green);
    const GB = dist2(a.green, b.blue);
    const BR = dist2(a.blue, b.red);
    const BG = dist2(a.blue, b.green);
    const BB = dist2(a.blue, b.blue);
    return RR < RG and RR < RB and GG < GR and GG < GB and BB < BR and BB < BG;
}

fn intersection(a: Xy, b: Xy, c: Xy, d: Xy) Xy {
    const det = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x);
    const t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / det;
    return .{
        .x = if (t != 0) a.x + t * (b.x - a.x) else 0,
        .y = if (t != 0) a.y + t * (b.y - a.y) else 0,
    };
}

fn clipPoint(x: Xy, y: Xy, z: Xy, a: Xy, b: Xy, c: Xy) Xy {
    const d1 = testPointLine(y, a, b);
    const d2 = testPointLine(y, b, c);
    if (d1 <= 0 and d2 <= 0) return y;
    if (d1 > 0 and d2 > 0) return b;
    if (d1 > 0) return intersection(a, b, y, z);
    return intersection(x, y, b, c);
}

/// `src` clipped to lie inside `dst` (pl_primaries_clip).
pub fn primariesClip(src: Raw, dst: Raw) Raw {
    return .{
        .red = clipPoint(src.green, src.red, src.blue, dst.green, dst.red, dst.blue),
        .green = clipPoint(src.blue, src.green, src.red, dst.blue, dst.green, dst.red),
        .blue = clipPoint(src.red, src.blue, src.green, dst.red, dst.blue, dst.green),
        .white = src.white,
    };
}

// ------------------------------------------------------- color spaces / HDR

pub const Trc = enum { bt1886, pq, hlg };

pub const Scaling = enum { norm, sqrt, nits, pq };

/// libplacebo's `enum pl_hdr_metadata_type` (vs-placebo's `metadata`).
pub const Metadata = enum(u8) { any = 0, none, hdr10, hdr10plus, cie_y };

pub const Hdr = struct {
    prim: Raw = .{},
    min_luma: f32 = 0,
    max_luma: f32 = 0,
    max_cll: f32 = 0,
    max_fall: f32 = 0,
    scene_max: [3]f32 = .{ 0, 0, 0 },
    scene_avg: f32 = 0,
    max_pq_y: f32 = 0,
    avg_pq_y: f32 = 0,

    pub fn hasHdr10Plus(h: *const Hdr) bool {
        return h.scene_avg != 0 and (h.scene_max[0] != 0 or h.scene_max[1] != 0 or h.scene_max[2] != 0);
    }

    pub fn hasCieY(h: *const Hdr) bool {
        return h.max_pq_y != 0 and h.avg_pq_y != 0;
    }
};

pub const Csp = struct {
    prim: Prim,
    trc: Trc,
    hdr: Hdr = .{},

    pub fn nominalPeak(trc: Trc) f32 {
        return switch (trc) {
            .bt1886 => 1.0,
            .pq => 10000.0 / sdr_white,
            .hlg => 12.0 / hlg_75,
        };
    }

    pub fn trcIsHdr(trc: Trc) bool {
        return nominalPeak(trc) > 1.0;
    }

    pub fn isHdr(c: *const Csp) bool {
        return c.hdr.max_luma > sdr_white or trcIsHdr(c.trc);
    }

    pub fn isBlackScaled(c: *const Csp) bool {
        return c.trc == .hlg;
    }

    pub fn eql(a: *const Csp, b: *const Csp) bool {
        const ha = a.hdr;
        const hb = b.hdr;
        return a.prim == b.prim and a.trc == b.trc and ha.prim.eql(hb.prim) and
            ha.min_luma == hb.min_luma and ha.max_luma == hb.max_luma and
            ha.max_cll == hb.max_cll and ha.max_fall == hb.max_fall and
            std.mem.eql(f32, &ha.scene_max, &hb.scene_max) and ha.scene_avg == hb.scene_avg and
            ha.max_pq_y == hb.max_pq_y and ha.avg_pq_y == hb.avg_pq_y;
    }
};

pub fn hdrRescale(from: Scaling, to: Scaling, x_in: f32) f32 {
    if (from == to) return x_in;
    if (x_in == 0) return x_in;
    var x = @max(x_in, 0);
    switch (from) {
        .pq => {
            x = powf(@min(x, 1), 1.0 / pq_m2);
            x = @max(x - pq_c1, 0) / (pq_c2 - pq_c3 * x);
            x = powf(x, 1.0 / pq_m1);
            x *= 10000.0;
            x /= sdr_white;
        },
        .nits => x /= sdr_white,
        .norm => {},
        .sqrt => x *= x,
    }
    switch (to) {
        .norm => return x,
        .sqrt => return @sqrt(x),
        .nits => return x * sdr_white,
        .pq => {
            x *= sdr_white / 10000.0;
            x = powf(x, pq_m1);
            x = (pq_c1 + pq_c2 * x) / (1.0 + pq_c3 * x);
            return powf(x, pq_m2);
        },
    }
}

/// Values at or below the HDR black sentinel count as true black.
pub fn signalBlack(min: f32, scaling: Scaling) f32 {
    return if (min <= hdrRescale(.nits, scaling, hdr_black)) 0 else min;
}

pub const Luma = struct { min: f32, max: f32, avg: f32 };

/// pl_color_space_nominal_luma_ex
pub fn nominalLuma(c: *const Csp, metadata: Metadata, scaling: Scaling) Luma {
    var min_l: f32 = 0;
    var max_l: f32 = 0;
    var avg_l: f32 = 0;
    const compat = struct {
        fn f(m: Metadata, want: Metadata) bool {
            return m == .any or m == want;
        }
    }.f;
    if (metadata != .none) {
        min_l = hdrRescale(.nits, scaling, c.hdr.min_luma);
        max_l = hdrRescale(.nits, scaling, c.hdr.max_luma);
        if (max_l == 0 and c.hdr.max_cll != 0) max_l = hdrRescale(.nits, scaling, c.hdr.max_cll);
    }
    if (compat(metadata, .hdr10plus) and c.hdr.hasHdr10Plus()) {
        const maxscl = @max(c.hdr.scene_max[0], @max(c.hdr.scene_max[1], c.hdr.scene_max[2]));
        if (maxscl != 0) {
            var prim = c.hdr.prim;
            prim.merge(rawPrimaries(c.prim));
            const m = rgb2xyz(prim);
            const ml = m[1][0] * c.hdr.scene_max[0] + m[1][1] * c.hdr.scene_max[1] + m[1][2] * c.hdr.scene_max[2];
            max_l = hdrRescale(.nits, scaling, ml);
            avg_l = hdrRescale(.nits, scaling, ml / maxscl * c.hdr.scene_avg);
        }
    }
    if (compat(metadata, .cie_y) and c.hdr.hasCieY()) {
        max_l = hdrRescale(.pq, scaling, c.hdr.max_pq_y);
        avg_l = hdrRescale(.pq, scaling, c.hdr.avg_pq_y);
    }

    const hdr_min = hdrRescale(.nits, scaling, hdr_black);
    const hdr_max = hdrRescale(.pq, scaling, 1.0);
    max_l = if (max_l != 0) clampf(max_l, hdr_min, hdr_max) else 0;
    min_l = if (min_l != 0) clampf(min_l, hdr_min, hdr_max) else 0;
    if ((max_l != 0 and min_l >= max_l) or min_l >= hdr_max) {
        min_l = 0;
        max_l = 0;
    }

    if (max_l == 0) {
        max_l = if (c.trc == .hlg) hdrRescale(.nits, scaling, hlg_peak) else hdrRescale(.norm, scaling, Csp.nominalPeak(c.trc));
    }
    if (min_l == 0) {
        if (Csp.trcIsHdr(c.trc)) {
            min_l = hdr_min;
        } else {
            const peak = hdrRescale(scaling, .nits, max_l);
            min_l = hdrRescale(.nits, scaling, peak / sdr_contrast);
        }
    }
    if (avg_l != 0) avg_l = clampf(avg_l, min_l, max_l);
    return .{ .min = min_l, .max = max_l, .avg = avg_l };
}

/// pl_color_space_infer (primaries and transfer are always set here)
pub fn infer(c: *Csp) void {
    const l = nominalLuma(c, .hdr10, .nits);
    c.hdr.max_luma = l.max;
    if (c.hdr.min_luma == 0) c.hdr.min_luma = l.min;
    if (!primariesValid(c.hdr.prim)) c.hdr.prim = rawPrimaries(c.prim);
}

/// pl_color_space_infer_map
pub fn inferMap(src: *Csp, dst: *Csp) void {
    const unknown_src_contrast = src.hdr.min_luma == 0;
    const unknown_dst_contrast = dst.hdr.min_luma == 0;
    const unknown_src_luminance = src.hdr.max_luma == 0;

    infer(src);
    infer(dst);

    const dynamic_src_contrast = src.isBlackScaled() or src.trc == .bt1886;
    if (unknown_src_contrast and dynamic_src_contrast) src.hdr.min_luma = dst.hdr.min_luma;

    const src_is_sdr = !src.isHdr();
    const dst_is_sdr = !dst.isHdr();
    if (unknown_dst_contrast and src_is_sdr and dst_is_sdr) dst.hdr.min_luma = src.hdr.min_luma;
    if (unknown_src_luminance and src_is_sdr and dst_is_sdr) src.hdr.max_luma = dst.hdr.max_luma;
    if (!unknown_src_luminance and src_is_sdr and dst_is_sdr) dst.hdr.max_luma = src.hdr.max_luma;
    if (src.trc == .hlg and dst.isHdr()) src.hdr.max_luma = dst.hdr.max_luma;
}

/// Linear-light range of a transfer, normalized to SDR white, with the black
/// sentinel snapped to 0 (what pl_shader_linearize/delinearize use).
pub fn linearRange(c: *const Csp) struct { min: f32, max: f32 } {
    const l = nominalLuma(c, .hdr10, .norm);
    return .{ .min = signalBlack(l.min, .norm), .max = l.max };
}

// ------------------------------------------------------------- tone mapping

pub const ToneConstants = struct {
    knee_adaptation: f32 = 0.4,
    knee_minimum: f32 = 0.1,
    knee_maximum: f32 = 0.8,
    knee_default: f32 = 0.4,
    knee_offset: f32 = 1.0,
    slope_tuning: f32 = 1.5,
    slope_offset: f32 = 0.2,
    spline_contrast: f32 = 0.5,
    reinhard_contrast: f32 = 0.5,
    linear_knee: f32 = 0.3,
    exposure: f32 = 1.0,

    fn fix(c: *ToneConstants) void {
        const eps: f32 = 1e-6;
        c.knee_adaptation = clampf(c.knee_adaptation, 0, 1);
        c.knee_minimum = clampf(c.knee_minimum, eps, 0.5 - eps);
        c.knee_maximum = clampf(c.knee_maximum, 0.5 + eps, 1 - eps);
        c.knee_default = clampf(c.knee_default, c.knee_minimum, c.knee_maximum);
        c.knee_offset = clampf(c.knee_offset, 0.5, 2.0);
        c.slope_tuning = clampf(c.slope_tuning, 0, 10);
        c.slope_offset = clampf(c.slope_offset, 0, 1);
        c.spline_contrast = clampf(c.spline_contrast, 0, 1.5);
        c.reinhard_contrast = clampf(c.reinhard_contrast, eps, 1 - eps);
        c.linear_knee = clampf(c.linear_knee, eps, 1 - eps);
        c.exposure = clampf(c.exposure, eps, 10);
    }
};

/// libplacebo's `pl_tone_map_functions` order (vs-placebo's
/// `tone_mapping_function`).
pub const ToneFn = enum(u8) {
    clip,
    st2094_40,
    st2094_10,
    bt2390,
    bt2446a,
    spline,
    reinhard,
    mobius,
    hable,
    gamma,
    linear,
    linear_light,

    pub const names = [_][]const u8{ "clip", "st2094-40", "st2094-10", "bt2390", "bt2446a", "spline", "reinhard", "mobius", "hable", "gamma", "linear", "linearlight" };

    pub fn fromName(name: []const u8) ?ToneFn {
        for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return @enumFromInt(i);
        }
        return null;
    }

    /// `param_def`, the default `tone_mapping_param`
    pub fn paramDef(f: ToneFn) f32 {
        return switch (f) {
            .st2094_40, .st2094_10 => 0.70,
            .bt2390 => 1.00,
            .spline => 0.50,
            .reinhard => 0.50,
            .mobius, .gamma => 0.30,
            .linear, .linear_light => 1.00,
            .clip, .bt2446a, .hable => 0,
        };
    }

    pub fn scaling(f: ToneFn) Scaling {
        return switch (f) {
            .bt2390, .spline, .linear => .pq,
            .linear_light => .norm,
            else => .nits,
        };
    }

    pub fn hasInverse(f: ToneFn) bool {
        return switch (f) {
            .clip, .bt2446a, .spline, .linear, .linear_light => true,
            else => false,
        };
    }
};

pub const ToneParams = struct {
    function: ToneFn,
    constants: ToneConstants = .{},
    param: f32 = 0,
    input_scaling: Scaling = .pq,
    output_scaling: Scaling = .pq,
    input_min: f32 = 0,
    input_max: f32 = 0,
    input_avg: f32 = 0,
    output_min: f32 = 0,
    output_max: f32 = 0,

    /// pl_tone_map_params_noop
    pub fn noop(p: *const ToneParams) bool {
        const in_min = hdrRescale(p.input_scaling, .nits, p.input_min);
        const in_max = hdrRescale(p.input_scaling, .nits, p.input_max);
        const out_min = hdrRescale(p.output_scaling, .nits, p.output_min);
        const out_max = hdrRescale(p.output_scaling, .nits, p.output_max);
        return @abs(in_min - out_min) < 1e-4 and in_max < out_max + 1e-2 and
            (out_max < in_max + 1e-2 or !p.function.hasInverse());
    }

    /// pl_tone_map_params_infer
    pub fn inferParams(p: *ToneParams) void {
        if (p.param != 0) switch (p.function) {
            .st2094_40, .st2094_10 => p.constants.knee_adaptation = p.param,
            .bt2390 => p.constants.knee_offset = p.param,
            .spline => p.constants.spline_contrast = p.param,
            .reinhard => p.constants.reinhard_contrast = p.param,
            .mobius, .gamma => p.constants.linear_knee = p.param,
            .linear, .linear_light => p.constants.exposure = p.param,
            else => {},
        };
        p.constants.fix();

        var sdr = hdrRescale(p.output_scaling, p.input_scaling, p.output_max);
        sdr = @min(sdr, hdrRescale(.nits, p.input_scaling, 100));
        p.input_max = @max(p.input_max, sdr);

        if (!p.function.hasInverse()) {
            const scaled_input_max = hdrRescale(p.input_scaling, p.output_scaling, p.input_max);
            p.output_max = @min(p.output_max, scaled_input_max);
        }
    }

    fn fixed(p: *const ToneParams) ToneParams {
        var f = p.*;
        f.inferParams();
        const s = f.function.scaling();
        f.input_scaling = s;
        f.output_scaling = s;
        f.input_min = hdrRescale(p.input_scaling, s, f.input_min);
        f.input_max = hdrRescale(p.input_scaling, s, f.input_max);
        f.input_avg = hdrRescale(p.input_scaling, s, f.input_avg);
        f.output_min = hdrRescale(p.output_scaling, s, f.output_min);
        f.output_max = hdrRescale(p.output_scaling, s, f.output_max);
        return f;
    }

    /// pl_tone_map_generate: `out.len` samples evenly spaced in
    /// `input_scaling` over [input_min, input_max], mapped into
    /// `output_scaling`.
    pub fn generate(p: *const ToneParams, out: []f32) void {
        std.debug.assert(out.len >= 2);
        const f = p.fixed();
        const s = f.function.scaling();
        for (out, 0..) |*o, i| {
            const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(out.len - 1));
            o.* = hdrRescale(p.input_scaling, s, mix(p.input_min, p.input_max, x));
        }
        mapLut(out, &f);
        for (out) |*o| o.* = hdrRescale(s, p.output_scaling, clampf(o.*, f.output_min, f.output_max));
    }

    /// pl_tone_map_sample: the curve at one input value.
    pub fn sample(p: *const ToneParams, x_in: f32) f32 {
        const f = p.fixed();
        const s = f.function.scaling();
        var x = [1]f32{hdrRescale(p.input_scaling, s, clampf(x_in, p.input_min, p.input_max))};
        mapLut(&x, &f);
        return hdrRescale(s, p.output_scaling, clampf(x[0], f.output_min, f.output_max));
    }
};

/// map_lut: the inverse curve when expanding, else the forward one.
fn mapLut(lut: []f32, f: *const ToneParams) void {
    if (f.output_max > f.input_max + 1e-4) {
        std.debug.assert(f.function.hasInverse());
        mapInverse(lut, f);
    } else {
        mapForward(lut, f);
    }
}

inline fn rescaleIn(x: f32, p: *const ToneParams) f32 {
    return (x - p.input_min) / (p.input_max - p.input_min);
}

inline fn rescale(x: f32, p: *const ToneParams) f32 {
    return (x - p.input_min) / (p.output_max - p.output_min);
}

inline fn rescaleOut(x: f32, p: *const ToneParams) f32 {
    return x * (p.output_max - p.output_min) + p.output_min;
}

inline fn bt1886Eotf(x: f32, min: f32, max: f32) f32 {
    const lb = powf(min, 1.0 / 2.4);
    const lw = powf(max, 1.0 / 2.4);
    return powf((lw - lb) * x + lb, 2.4);
}

inline fn bt1886Oetf(x: f32, min: f32, max: f32) f32 {
    const lb = powf(min, 1.0 / 2.4);
    const lw = powf(max, 1.0 / 2.4);
    return (powf(x, 1.0 / 2.4) - lb) / (lw - lb);
}

fn mapForward(lut: []f32, p: *const ToneParams) void {
    switch (p.function) {
        .clip => {},
        .st2094_40 => st2094_40(lut, p),
        .st2094_10 => st2094_10(lut, p),
        .bt2390 => bt2390(lut, p),
        .bt2446a => bt2446a(lut, p),
        .spline => spline(lut, p),
        .reinhard => reinhard(lut, p),
        .mobius => mobius(lut, p),
        .hable => hableMap(lut, p),
        .gamma => gammaMap(lut, p),
        .linear, .linear_light => linearMap(lut, p),
    }
}

fn mapInverse(lut: []f32, p: *const ToneParams) void {
    switch (p.function) {
        .clip => {},
        .bt2446a => bt2446aInv(lut, p),
        .spline => spline(lut, p),
        .linear, .linear_light => linearMap(lut, p),
        else => unreachable,
    }
}

/// Knee point from the scene average (ST 2094-10 inspired brightness
/// matching), returned in the params' input/output scaling.
fn st2094PickKnee(p: *const ToneParams) [2]f32 {
    const src_min = hdrRescale(p.input_scaling, .pq, p.input_min);
    const src_max = hdrRescale(p.input_scaling, .pq, p.input_max);
    const src_avg = hdrRescale(p.input_scaling, .pq, p.input_avg);
    const dst_min = hdrRescale(p.output_scaling, .pq, p.output_min);
    const dst_max = hdrRescale(p.output_scaling, .pq, p.output_max);

    const min_knee = p.constants.knee_minimum;
    const max_knee = p.constants.knee_maximum;
    const def_knee = p.constants.knee_default;
    const src_knee_min = mix(src_min, src_max, min_knee);
    const src_knee_max = mix(src_min, src_max, max_knee);
    const dst_knee_min = mix(dst_min, dst_max, min_knee);
    const dst_knee_max = mix(dst_min, dst_max, max_knee);

    var src_knee = if (src_avg != 0) src_avg else mix(src_min, src_max, def_knee);
    src_knee = clampf(src_knee, src_knee_min, src_knee_max);

    const target = (src_knee - src_min) / (src_max - src_min);
    const adapted = mix(dst_min, dst_max, target);

    const tuning = 1 - smoothstep(max_knee, def_knee, target) * smoothstep(min_knee, def_knee, target);
    var adaptation = mix(p.constants.knee_adaptation, 1.0, tuning);
    // Brightness matching only where the source knee is displayable: below
    // the output floor it would drag the knee into the black-point region
    adaptation = mix(1.0, adaptation, smoothstep(dst_min, dst_knee_min, src_knee));
    var dst_knee = mix(src_knee, adapted, adaptation);
    dst_knee = clampf(dst_knee, dst_knee_min, dst_knee_max);

    return .{ hdrRescale(.pq, p.input_scaling, src_knee), hdrRescale(.pq, p.output_scaling, dst_knee) };
}

const binom = blk: {
    var t: [17][17]u16 = @splat(@splat(0));
    for (0..17) |n| {
        t[n][0] = 1;
        for (1..n + 1) |k| t[n][k] = t[n - 1][k - 1] + (if (k < n) t[n - 1][k] else 0);
    }
    break :blk t;
};

fn st2094Intercept(N: u8, Kx: f32, Ky: f32) f32 {
    const n: f32 = @floatFromInt(N);
    if (Kx <= 0 or Ky >= 1) return 1.0 / n;
    const slope = Ky / Kx * (1 - Kx) / (1 - Ky);
    return @min(slope / n, 1.0);
}

fn st2094_40(lut: []f32, p: *const ToneParams) void {
    var P: [17]f32 = @splat(0);
    // No HDR10+ OOTF metadata is read, so this is always libplacebo's
    // brightness-matching fallback: its target T is the output peak D, and
    // neither of the D < T / D > T OOTF adjustments applies.
    const knee = st2094PickKnee(p);
    const Kx = knee[0] / p.input_max;
    const Ky = knee[1] / p.output_max;
    const slope0 = Ky / Kx * (1 - Kx) / (1 - Ky);
    const N: u8 = @intCast(math.clamp(@as(i32, @intFromFloat(@ceil(@min(slope0, 64.0)))), 2, 16));
    P[0] = 0;
    P[1] = st2094Intercept(N, Kx, Ky);
    for (2..@as(usize, N) + 1) |i| P[i] = 1;

    std.debug.assert(Kx >= 0 and Kx <= 1);
    std.debug.assert(Ky >= 0 and Ky <= 1);

    for (lut) |*v| {
        var x = bt1886Oetf(v.*, p.input_min, p.input_max);
        x = bt1886Eotf(x, 0, 1);
        if (x <= Kx and Kx != 0) {
            x *= Ky / Kx;
        } else {
            const t = (x - Kx) / (1 - Kx);
            var b: f32 = 0;
            for (0..@as(usize, N) + 1) |k| {
                const kf: f32 = @floatFromInt(k);
                b += @as(f32, @floatFromInt(binom[N][k])) * powf(t, kf) * powf(1 - t, @as(f32, @floatFromInt(N)) - kf) * P[k];
            }
            x = Ky + (1 - Ky) * b;
        }
        x = bt1886Oetf(x, 0, 1);
        v.* = bt1886Eotf(x, p.output_min, p.output_max);
    }
}

fn st2094_10(lut: []f32, p: *const ToneParams) void {
    const knee = st2094PickKnee(p);
    const x1 = p.input_min;
    const x3 = p.input_max;
    const x2 = knee[0];
    const y1 = p.output_min;
    const y3 = p.output_max;
    const y2 = knee[1];
    const cmat: Mat3 = .{
        .{ x2 * x3 * (y2 - y3), x1 * x3 * (y3 - y1), x1 * x2 * (y1 - y2) },
        .{ x3 * y3 - x2 * y2, x1 * y1 - x3 * y3, x2 * y2 - x1 * y1 },
        .{ x3 - x2, x1 - x3, x2 - x1 },
    };
    const coeffs = apply(cmat, .{ y1, y2, y3 });
    const k: f32 = @floatCast(1.0 / @as(f64, x3 * y3 * (x1 - x2) + x2 * y2 * (x3 - x1) + x1 * y1 * (x2 - x3)));
    const c1 = k * coeffs[0];
    const c2 = k * coeffs[1];
    const c3 = k * coeffs[2];
    for (lut) |*x| x.* = (c1 + c2 * x.*) / (1 + c3 * x.*);
}

fn bt2390(lut: []f32, p: *const ToneParams) void {
    const minLum = rescaleIn(p.output_min, p);
    const maxLum = rescaleIn(p.output_max, p);
    const offset = p.constants.knee_offset;
    const ks = (1 + offset) * maxLum - offset;
    const bp: f32 = if (minLum > 0) @min(1 / minLum, 4) else 4;
    const gain_inv = 1 + minLum / maxLum * powf(1 - maxLum, bp);
    const gain: f32 = if (maxLum < 1) 1 / gain_inv else 1;
    for (lut) |*v| {
        var x = rescaleIn(v.*, p);
        if (ks < 1) {
            const tb = (x - ks) / (1 - ks);
            const tb2 = tb * tb;
            const tb3 = tb2 * tb;
            const pb = (2 * tb3 - 3 * tb2 + 1) * ks + (tb3 - 2 * tb2 + tb) * (1 - ks) + (-2 * tb3 + 3 * tb2) * maxLum;
            x = if (x < ks) x else pb;
        }
        if (x < 1) {
            x += minLum * powf(1 - x, bp);
            x = gain * (x - minLum) + minLum;
        }
        v.* = x * (p.input_max - p.input_min) + p.input_min;
    }
}

fn bt2446a(lut: []f32, p: *const ToneParams) void {
    const phdr = 1 + 32 * powf(p.input_max / 10000, 1.0 / 2.4);
    const psdr = 1 + 32 * powf(p.output_max / 10000, 1.0 / 2.4);
    for (lut) |*v| {
        var x = powf(rescaleIn(v.*, p), 1.0 / 2.4);
        x = @log(1 + (phdr - 1) * x) / @log(phdr);
        if (x <= 0.7399) {
            x = 1.0770 * x;
        } else if (x < 0.9909) {
            x = (-1.1510 * x + 2.7811) * x - 0.6302;
        } else {
            x = 0.5 * x + 0.5;
        }
        x = (powf(psdr, x) - 1) / (psdr - 1);
        v.* = bt1886Eotf(x, p.output_min, p.output_max);
    }
}

fn bt2446aInv(lut: []f32, p: *const ToneParams) void {
    for (lut) |*v| {
        var x = bt1886Oetf(v.*, p.input_min, p.input_max);
        x *= 255.0;
        if (x > 70) {
            x = powf(x, (2.8305e-6 * x - 7.4622e-4) * x + 1.2528);
        } else {
            x = powf(x, (1.8712e-5 * x - 2.7334e-3) * x + 1.3141);
        }
        x = powf(x / 1000, 2.4);
        v.* = rescaleOut(x, p);
    }
}

fn spline(lut: []f32, p: *const ToneParams) void {
    const knee = st2094PickKnee(p);
    const src_pivot = knee[0];
    const dst_pivot = knee[1];
    var slope = (dst_pivot - p.output_min) / (src_pivot - p.input_min);
    var ratio = p.input_max / p.output_max - 1.0;
    ratio = clampf(p.constants.slope_tuning * ratio, p.constants.slope_offset, 1.0 + p.constants.slope_offset);
    slope = powf(slope, (1.0 - p.constants.spline_contrast) * ratio);

    const in_min = p.input_min - src_pivot;
    const in_max = p.input_max - src_pivot;
    const out_min = p.output_min - dst_pivot;
    const out_max = p.output_max - dst_pivot;

    // Hyperbolic toe P and shoulder Q: P(0) = 0, P'(0) = slope,
    // P(in_min) = out_min, with P(x) = slope * x / (1 + Pk * x); Q likewise
    const Pk = (slope * in_min - out_min) / (out_min * in_min);
    const Qk = (slope * in_max - out_max) / (out_max * in_max);

    for (lut) |*v| {
        const x = v.* - src_pivot;
        v.* = slope * x / (1.0 + (if (x > 0) Qk else Pk) * x) + dst_pivot;
    }
}

fn reinhard(lut: []f32, p: *const ToneParams) void {
    const peak = rescale(p.input_max, p);
    const contrast = p.constants.reinhard_contrast;
    const offset = (1.0 - contrast) / contrast;
    const s = (peak + offset) / peak;
    for (lut) |*v| {
        var x = rescale(v.*, p);
        x = x / (x + offset);
        x *= s;
        v.* = rescaleOut(x, p);
    }
}

fn mobiusCoeffs(j: f32, peak: f32) [3]f32 {
    const a = -j * j * (peak - 1.0) / (j * j - 2.0 * j + peak);
    const b = (j * j - 2.0 * j * peak + peak) / @max(1e-6, peak - 1.0);
    const s = (b * b + 2.0 * b * j + j * j) / (b - a);
    return .{ a, b, s };
}

fn mobius(lut: []f32, p: *const ToneParams) void {
    const peak = rescale(p.input_max, p);
    const j = p.constants.linear_knee;
    const c = mobiusCoeffs(j, peak);
    for (lut) |*v| {
        var x = rescale(v.*, p);
        x = if (x <= j) x else c[2] * (x + c[0]) / (x + c[1]);
        v.* = rescaleOut(x, p);
    }
}

fn hable(x: f32) f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn hableMap(lut: []f32, p: *const ToneParams) void {
    const peak = p.input_max / p.output_max;
    const s = 1.0 / hable(peak);
    for (lut) |*v| {
        var x = bt1886Oetf(v.*, p.input_min, p.input_max);
        x = bt1886Eotf(x, 0, peak);
        x = s * hable(x);
        x = bt1886Oetf(x, 0, 1);
        v.* = bt1886Eotf(x, p.output_min, p.output_max);
    }
}

fn gammaMap(lut: []f32, p: *const ToneParams) void {
    const peak = rescale(p.input_max, p);
    const cutoff = p.constants.linear_knee;
    const gamma = @log(cutoff) / @log(cutoff / peak);
    for (lut) |*v| {
        var x = rescale(v.*, p);
        x = if (x > cutoff) powf(x / peak, gamma) else x;
        v.* = rescaleOut(x, p);
    }
}

fn linearMap(lut: []f32, p: *const ToneParams) void {
    const gain = p.constants.exposure;
    for (lut) |*v| {
        var x = rescaleIn(v.*, p);
        x *= gain;
        v.* = rescaleOut(x, p);
    }
}

// ------------------------------------------------------------ gamut mapping

pub const GamutConstants = struct {
    colorimetric_gamma: f32 = 1.80,
    softclip_knee: f32 = 0.70,
    softclip_desat: f32 = 0.35,
    perceptual_deadzone: f32 = 0.30,
    perceptual_strength: f32 = 0.80,

    fn fix(c: *GamutConstants) void {
        c.perceptual_deadzone = clampf(c.perceptual_deadzone, 0, 1);
        c.perceptual_strength = clampf(c.perceptual_strength, 0, 1);
        c.colorimetric_gamma = clampf(c.colorimetric_gamma, 0, 10);
        c.softclip_knee = clampf(c.softclip_knee, 0, 1);
        c.softclip_desat = clampf(c.softclip_desat, 0, 1);
    }
};

/// libplacebo's `pl_gamut_map_functions` order (vs-placebo's
/// `gamut_mapping`).
pub const GamutFn = enum(u8) {
    clip,
    perceptual,
    softclip,
    relative,
    saturation,
    absolute,
    desaturate,
    darken,
    highlight,
    linear,

    pub const count = 10;

    pub fn bidirectional(f: GamutFn) bool {
        return f == .perceptual or f == .saturation;
    }
};

pub const GamutParams = struct {
    function: GamutFn,
    constants: GamutConstants = .{},
    input_gamut: Raw,
    output_gamut: Raw,
    min_luma: f32, // PQ
    max_luma: f32, // PQ
    size_I: u32 = 48,
    size_C: u32 = 32,
    size_h: u32 = 256,

    /// Floats per LUT entry: I, P, T and a pad lane for 16-byte loads.
    pub const stride = 4;

    pub fn lutLen(p: *const GamutParams) usize {
        return @as(usize, p.size_I) * p.size_C * p.size_h * stride;
    }

    /// pl_gamut_map_params_noop
    pub fn noop(p: *const GamutParams) bool {
        if (p.function == .clip) return true;
        const src = p.input_gamut;
        const dst = p.output_gamut;
        if (!primariesCompatible(dst, src)) return true;
        var need_map = !primariesSuperset(dst, src);
        need_map = need_map or !xyEql(src.white, dst.white);
        if (p.function.bidirectional()) need_map = need_map or !dst.eql(src);
        return !need_map;
    }

    /// pl_gamut_map_generate into `out` (lutLen floats): entry
    /// (h, C, I) at ((h * size_C + C) * size_I + I) * stride holds the mapped
    /// IPT of ICh = (mix(min, max, I), mix(0, 0.5, C), mix(-pi, pi, h)).
    /// Hue slices are split across up to `threads` threads, each with its
    /// own peak caches, as libplacebo's workers.
    pub fn generate(p: *const GamutParams, out: []f32, threads: usize) void {
        std.debug.assert(out.len == p.lutLen());
        const max_workers = 32;
        const per = (p.size_h + max_workers - 1) / max_workers;
        const n_chunks = (p.size_h + per - 1) / per;
        const slice_len = @as(usize, p.size_C) * p.size_I * stride;

        var workers: [max_workers]?std.Thread = @splat(null);
        var started: usize = 0;
        var h0: u32 = 0;
        var chunk: u32 = 0;
        while (chunk < n_chunks) : (chunk += 1) {
            const count = @min(per, p.size_h - h0);
            const part = out[h0 * slice_len ..][0 .. count * slice_len];
            if (started + 1 < threads) {
                if (std.Thread.spawn(.{}, generateChunk, .{ p, part, h0, count })) |t| {
                    workers[chunk] = t;
                    started += 1;
                } else |_| generateChunk(p, part, h0, count);
            } else generateChunk(p, part, h0, count);
            h0 += count;
        }
        for (workers) |w| if (w) |t| t.join();
    }
};

fn generateChunk(p: *const GamutParams, out: []f32, h0: u32, count: u32) void {
    var i: usize = 0;
    for (h0..h0 + count) |h| {
        for (0..p.size_C) |C| {
            for (0..p.size_I) |I| {
                const Ix = @as(f32, @floatFromInt(I)) / @as(f32, @floatFromInt(p.size_I - 1));
                const Cx = @as(f32, @floatFromInt(C)) / @as(f32, @floatFromInt(p.size_C - 1));
                const hx = @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(p.size_h - 1));
                const ipt = ich2ipt(.{
                    .I = mix(p.min_luma, p.max_luma, Ix),
                    .C = mix(0.0, 0.5, Cx),
                    .h = mix(-math.pi, math.pi, hx),
                });
                out[i..][0..GamutParams.stride].* = .{ ipt.I, ipt.P, ipt.T, 0 };
                i += GamutParams.stride;
            }
        }
    }
    var fixed = p.*;
    fixed.constants.fix();
    mapGamut(out, &fixed);
    // The range of libplacebo's 16-bit UNORM LUT texture (I in [0, 1], P and
    // T stored around code 32768); values stay f32 otherwise.
    const c_lo: f32 = -32768.0 / 65535.0;
    const c_hi: f32 = 32767.0 / 65535.0;
    var k: usize = 0;
    while (k < out.len) : (k += GamutParams.stride) {
        out[k] = clampf(out[k], 0, 1);
        out[k + 1] = clampf(out[k + 1], c_lo, c_hi);
        out[k + 2] = clampf(out[k + 2], c_lo, c_hi);
    }
}

const IPT = struct { I: f32, P: f32, T: f32 };
const ICh = struct { I: f32 = 0, C: f32 = 0, h: f32 = 0 };
const RGB = [3]f32;

fn ipt2ich(c: IPT) ICh {
    return .{ .I = c.I, .C = @sqrt(c.P * c.P + c.T * c.T), .h = math.atan2(c.T, c.P) };
}

fn ich2ipt(c: ICh) IPT {
    return .{ .I = c.I, .P = c.C * @cos(c.h), .T = c.C * @sin(c.h) };
}

const pq_lut_size = 1024;

/// PQ EOTF samples (10k-normalized), as libplacebo's pq_eotf_lut, plus a
/// pad entry for the interpolation at 1.0.
const pq_eotf_lut: [pq_lut_size + 1]f32 = blk: {
    @setEvalBranchQuota(200_000);
    var t: [pq_lut_size + 1]f32 = undefined;
    for (0..pq_lut_size) |i| {
        const x: f64 = @as(f64, @floatFromInt(i)) / (pq_lut_size - 1);
        const m2: f64 = 2523.0 / 4096.0 * 128.0;
        const m1: f64 = 2610.0 / 4096.0 / 4.0;
        const c1: f64 = 3424.0 / 4096.0;
        const c2: f64 = 2413.0 / 4096.0 * 32.0;
        const c3: f64 = 2392.0 / 4096.0 * 32.0;
        const xp = math.pow(f64, x, 1.0 / m2);
        t[i] = @floatCast(math.pow(f64, @max(xp - c1, 0) / (c2 - c3 * xp), 1.0 / m1));
    }
    t[pq_lut_size] = 1.0;
    break :blk t;
};

fn pqEotf(x: f32) f32 {
    const idxf = clampf(x, 0, 1) * (pq_lut_size - 1);
    const ipart: usize = @intFromFloat(@floor(idxf));
    const fpart = idxf - @floor(idxf);
    return mix(pq_eotf_lut[ipart], pq_eotf_lut[ipart + 1], fpart);
}

fn pqOetf(x_in: f32) f32 {
    var x = powf(@max(x_in, 0), pq_m1);
    x = (pq_c1 + pq_c2 * x) / (1.0 + pq_c3 * x);
    return powf(x, pq_m2);
}

const Gamut = struct {
    lms2rgb: Mat3,
    rgb2lms: Mat3,
    min_luma: f32,
    max_luma: f32,
    min_rgb: f32,
    max_rgb: f32,
    peak_cache: *ICh,
};

fn makeGamut(prim: Raw, p: *const GamutParams, cache: *ICh) Gamut {
    const eps: f32 = 1e-6;
    const m = iptRgb2Lms(prim);
    cache.* = .{};
    return .{
        .rgb2lms = m,
        .lms2rgb = invert(m),
        .min_luma = p.min_luma,
        .max_luma = p.max_luma,
        .min_rgb = pqEotf(p.min_luma) - eps,
        .max_rgb = pqEotf(p.max_luma) + eps,
        .peak_cache = cache,
    };
}

fn rgb2ipt(c: RGB, g: Gamut) IPT {
    const lms = apply(g.rgb2lms, c);
    const Lp = pqOetf(lms[0]);
    const Mp = pqOetf(lms[1]);
    const Sp = pqOetf(lms[2]);
    return .{
        .I = 0.4000 * Lp + 0.4000 * Mp + 0.2000 * Sp,
        .P = 4.4550 * Lp - 4.8510 * Mp + 0.3960 * Sp,
        .T = 0.8056 * Lp + 0.3572 * Mp - 1.1628 * Sp,
    };
}

fn iptLmsP(c: IPT) [3]f32 {
    return .{
        c.I + 0.0975689 * c.P + 0.205226 * c.T,
        c.I - 0.1138760 * c.P + 0.133217 * c.T,
        c.I + 0.0326151 * c.P - 0.676887 * c.T,
    };
}

fn ipt2rgb(c: IPT, g: Gamut) RGB {
    const lp = iptLmsP(c);
    return apply(g.lms2rgb, .{ pqEotf(lp[0]), pqEotf(lp[1]), pqEotf(lp[2]) });
}

fn inGamut(c: IPT, g: Gamut) bool {
    const lp = iptLmsP(c);
    for (lp) |v| {
        if (v < g.min_luma or v > g.max_luma) return false;
    }
    const rgb = apply(g.lms2rgb, .{ pqEotf(lp[0]), pqEotf(lp[1]), pqEotf(lp[2]) });
    for (rgb) |v| {
        if (!(v >= g.min_rgb and v <= g.max_rgb)) return false;
    }
    return true;
}

fn mixExp(c: ICh, x: f32, gamma: f32, base: f32) ICh {
    return .{ .I = base + (c.I - base) * powf(x, gamma), .C = c.C * x, .h = c.h };
}

fn scaleGamma(gamma: f32, ich: ICh, peak: ICh, g: Gamut) f32 {
    const Imin = g.min_luma;
    const Irel = @max((ich.I - Imin) / (peak.I - Imin), 0.0);
    return gamma * powf(Irel, 3) * @min(ich.C / peak.C, 1.0);
}

const max_delta: f32 = 5e-5;

fn desatBounded(I: f32, h: f32, Cmin_in: f32, Cmax_in: f32, g: Gamut) ICh {
    if (I <= g.min_luma) return .{ .I = g.min_luma, .C = 0, .h = h };
    if (I >= g.max_luma) return .{ .I = g.max_luma, .C = 0, .h = h };
    var Cmin = Cmin_in;
    var Cmax = Cmax_in;
    const maxDI = I * max_delta;
    var res: ICh = .{ .I = I, .C = (Cmin + Cmax) / 2, .h = h };
    while (true) {
        if (inGamut(ich2ipt(res), g)) Cmin = res.C else Cmax = res.C;
        res.C = (Cmin + Cmax) / 2;
        if (!(Cmax - Cmin > maxDI)) break;
    }
    return res;
}

/// Maximally saturated in-gamut color for a hue (golden-section search).
fn saturate(hue: f32, g: Gamut) ICh {
    if (g.peak_cache.I != 0 and @abs(g.peak_cache.h - hue) < 1e-3) return g.peak_cache.*;
    const invphi: f32 = 0.6180339887498948;
    const invphi2: f32 = 0.38196601125010515;
    var lo: ICh = .{ .I = g.min_luma, .h = hue };
    var hi: ICh = .{ .I = g.max_luma, .h = hue };
    var de = hi.I - lo.I;
    var a = desatBounded(lo.I + invphi2 * de, hue, 0.0, 0.5, g);
    var b = desatBounded(lo.I + invphi * de, hue, 0.0, 0.5, g);
    while (de > max_delta) {
        de *= invphi;
        if (a.C > b.C) {
            hi = b;
            b = a;
            a = desatBounded(lo.I + invphi2 * de, hue, lo.C - max_delta, 0.5, g);
        } else {
            lo = a;
            a = b;
            b = desatBounded(lo.I + invphi * de, hue, hi.C - max_delta, 0.5, g);
        }
    }
    const peak = if (a.C > b.C) a else b;
    g.peak_cache.* = peak;
    return peak;
}

fn clipGamma(ipt: IPT, gamma_in: f32, g: Gamut) IPT {
    if (ipt.I <= g.min_luma) return .{ .I = g.min_luma, .P = 0, .T = 0 };
    if (inGamut(ipt, g)) return ipt;
    const ich = ipt2ich(ipt);
    if (gamma_in == 0) return ich2ipt(desatBounded(ich.I, ich.h, 0.0, ich.C, g));
    const maxDI = @max(ich.I * max_delta, 1e-7);
    const peak = saturate(ich.h, g);
    const gamma = scaleGamma(gamma_in, ich, peak, g);
    var lo: f32 = 0;
    var hi: f32 = 1;
    var x: f32 = 0.5;
    while (true) {
        const t = mixExp(ich, x, gamma, peak.I);
        if (inGamut(ich2ipt(t), g)) lo = x else hi = x;
        x = (lo + hi) / 2;
        if (!(hi - lo > maxDI)) break;
    }
    return ich2ipt(mixExp(ich, x, gamma, peak.I));
}

fn softclip(value: f32, source: f32, target: f32, c: *const GamutConstants) f32 {
    if (target == 0) return 0;
    const peak = source / target;
    const x = @min(value / target, peak);
    const j = c.softclip_knee;
    if (x <= j or peak <= 1.0) return value;
    const m = mobiusCoeffs(j, peak);
    return m[2] * (x + m[0]) / (x + m[1]) * target;
}

fn wrap(h: f32) f32 {
    if (h > math.pi) return h - 2 * math.pi;
    if (h < -math.pi) return h + 2 * math.pi;
    return h;
}

const hs_S = 12;
const hs_N = hs_S + 2;

const HueShift = struct {
    dh: [hs_N]f32 = @splat(0),
    dddh: [hs_N]f32 = @splat(0),
    K: [hs_N]f32 = @splat(0),
    prev_hue: f32 = -10,
    prev_shift: f32 = 0,
    pts: [hs_N]struct { hue: f32, delta: f32 } = @splat(.{ .hue = 0, .delta = 0 }),

    fn prepare(s: *HueShift, src: Gamut, dst: Gamut) void {
        const O = pqEotf(src.min_luma);
        const X = pqEotf(src.max_luma);
        const M = (O + X) / 2;
        const refpoints = [hs_S]RGB{
            .{ X, O, O }, .{ O, X, O }, .{ O, O, X },
            .{ O, X, X }, .{ X, O, X }, .{ X, X, O },
            .{ O, X, M }, .{ X, O, M }, .{ X, M, O },
            .{ O, M, X }, .{ M, O, X }, .{ M, X, O },
        };
        s.* = .{};
        for (refpoints, 0..) |r, i| {
            const a = ipt2ich(rgb2ipt(r, src));
            const b = ipt2ich(rgb2ipt(r, dst));
            s.pts[i + 1] = .{ .hue = a.h, .delta = wrap(b.h - a.h) };
        }
        const Pt = @TypeOf(s.pts[0]);
        std.sort.insertion(Pt, s.pts[1 .. hs_S + 1], {}, struct {
            fn lt(_: void, a: Pt, b: Pt) bool {
                return a.hue < b.hue;
            }
        }.lt);
        s.pts[0] = s.pts[hs_S];
        s.pts[hs_S + 1] = s.pts[1];
        s.pts[0].hue -= 2 * math.pi;
        s.pts[hs_S + 1].hue += 2 * math.pi;

        var tmp: [hs_N][hs_N]f32 = @splat(@splat(0));
        var i: usize = hs_N - 1;
        while (i > 0) : (i -= 1) {
            s.dh[i - 1] = s.pts[i].hue - s.pts[i - 1].hue;
            s.dddh[i] = (s.pts[i].delta - s.pts[i - 1].delta) / s.dh[i - 1];
        }
        for (1..hs_N - 1) |k| {
            tmp[k][k] = 2 * (s.dh[k - 1] + s.dh[k]);
            if (k != 1) {
                tmp[k][k - 1] = s.dh[k - 1];
                tmp[k - 1][k] = s.dh[k - 1];
            }
            tmp[k][hs_N - 1] = 6 * (s.dddh[k + 1] - s.dddh[k]);
        }
        for (1..hs_N - 2) |k| {
            const q = tmp[k + 1][k] / tmp[k][k];
            for (1..hs_N) |j| tmp[k + 1][j] -= q * tmp[k][j];
        }
        i = hs_N - 2;
        while (i > 0) : (i -= 1) {
            var sum: f32 = 0;
            for (i..hs_N - 1) |j| sum += tmp[i][j] * s.K[j];
            s.K[i] = (tmp[i][hs_N - 1] - sum) / tmp[i][i];
        }
        s.prev_hue = -10;
    }

    fn apply(s: *HueShift, ich: ICh) ICh {
        if (!(@abs(ich.h - s.prev_hue) < 1e-6)) {
            for (0..hs_N - 1) |i| {
                if (s.pts[i + 1].hue > ich.h) {
                    const a = (s.K[i + 1] - s.K[i]) / (6 * s.dh[i]);
                    const b = s.K[i] / 2;
                    const c = s.dddh[i + 1] - (2 * s.dh[i] * s.K[i] + s.K[i + 1] * s.dh[i]) / 6;
                    const d = s.pts[i].delta;
                    const x = ich.h - s.pts[i].hue;
                    s.prev_shift = ich.h + (((a * x + b) * x + c) * x + d);
                    s.prev_hue = ich.h;
                    break;
                }
            }
        }
        return .{ .I = ich.I, .C = ich.C, .h = s.prev_shift };
    }
};

fn mapGamut(lut: []f32, p: *const GamutParams) void {
    const c = &p.constants;
    var cache_src: ICh = .{};
    var cache_dst: ICh = .{};
    const src = makeGamut(p.input_gamut, p, &cache_src);
    const dst = makeGamut(p.output_gamut, p, &cache_dst);
    const n = lut.len / GamutParams.stride;

    switch (p.function) {
        .clip => {},
        .perceptual => for (0..n) |e| {
            var ipt = load(lut, e);
            const ich = ipt2ich(ipt);
            const src_peak = saturate(ich.h, src);
            const dst_peak = saturate(ich.h, dst);
            const mapped = rgb2ipt(ipt2rgb(ipt, src), dst);
            const maxC = @max(src_peak.C, dst_peak.C);
            const k = smoothstep(c.perceptual_deadzone, 1.0, ich.C / maxC) * c.perceptual_strength;
            ipt.I = mix(ipt.I, mapped.I, k);
            ipt.P = mix(ipt.P, mapped.P, k);
            ipt.T = mix(ipt.T, mapped.T, k);
            // fold over-range channels within the source gamut's extent at
            // this hue
            const speak = ipt2rgb(ich2ipt(src_peak), dst);
            var rgb = ipt2rgb(ipt, dst);
            const mx = @max(rgb[0], @max(rgb[1], rgb[2]));
            const mn = @min(rgb[0], @min(rgb[1], rgb[2]));
            const sat = 1.0 - @max(mn, 0) / @max(mx, 1e-6);
            const ext = @max(@max(speak[0], @max(speak[1], speak[2])), mx);
            for (&rgb) |*v| {
                v.* = mix(v.*, softclip(v.*, ext, dst.max_rgb, c), sat);
                v.* = @min(@max(v.*, dst.min_rgb), dst.max_rgb);
            }
            store(lut, e, rgb2ipt(rgb, dst));
        },
        .softclip => {
            var cache_src2: ICh = .{};
            var cache_dst2: ICh = .{};
            const src_post = makeGamut(p.input_gamut, p, &cache_src2);
            const dst_post = makeGamut(p.output_gamut, p, &cache_dst2);
            var hs: HueShift = undefined;
            hs.prepare(src, dst);
            for (0..n) |e| {
                var ipt = load(lut, e);
                var s = src;
                var d = dst;
                if (ipt.I <= d.min_luma) {
                    ipt.P = 0;
                    ipt.T = 0;
                    store(lut, e, ipt);
                    continue;
                }
                var ich = ipt2ich(ipt);
                if (ich.C <= 1e-2) continue;

                var margin: f32 = 1.0;
                const shifted = hs.apply(ich);
                if (@abs(shifted.h - ich.h) >= 1e-3) {
                    const src_border = desatBounded(ich.I, ich.h, 0.0, 0.5, s);
                    const dst_border = desatBounded(ich.I, ich.h, 0.0, 0.5, d);
                    const k = smoothstep(dst_border.C * c.softclip_knee, src_border.C, ich.C);
                    ich.h = mix(ich.h, shifted.h, k);
                    s = src_post;
                    d = dst_post;
                    const shift_border = desatBounded(ich.I, ich.h, 0.0, 0.5, s);
                    margin *= @max(1.0, src_border.C / shift_border.C);
                }

                const source = saturate(ich.h, s);
                const target = saturate(ich.h, d);
                const border = desatBounded(ich.I, ich.h, 0.0, target.C, d);
                const chromaticity = mix(target.C, border.C, c.softclip_desat);
                ich.C = softclip(ich.C, margin * source.C, chromaticity, c);

                const peak = ipt2rgb(ich2ipt(.{ .I = ich.I, .C = chromaticity, .h = ich.h }), d);
                var rgb = ipt2rgb(ich2ipt(ich), d);
                for (&rgb, peak) |*v, pk| v.* = @max(softclip(v.*, pk, d.max_rgb, c), d.min_rgb);
                store(lut, e, rgb2ipt(rgb, d));
            }
        },
        .relative => for (0..n) |e| store(lut, e, clipGamma(load(lut, e), c.colorimetric_gamma, dst)),
        .desaturate => for (0..n) |e| store(lut, e, clipGamma(load(lut, e), 0.0, dst)),
        .saturation => for (0..n) |e| store(lut, e, rgb2ipt(ipt2rgb(load(lut, e), src), dst)),
        .absolute => {
            const m = adaptationMatrix(p.output_gamut.white, p.input_gamut.white);
            for (0..n) |e| {
                const rgb = apply(m, ipt2rgb(load(lut, e), dst));
                store(lut, e, clipGamma(rgb2ipt(rgb, dst), c.colorimetric_gamma, dst));
            }
        },
        .highlight => for (0..n) |e| {
            var ipt = load(lut, e);
            if (!inGamut(ipt, dst)) {
                ipt.I = @min(ipt.I + 0.1, 1.0);
                ipt.P = clampf(-1.2 * ipt.P, -0.5, 0.5);
                ipt.T = clampf(-1.2 * ipt.T, -0.5, 0.5);
            }
            store(lut, e, ipt);
        },
        .linear => {
            var gain: f32 = 1.0;
            var hue: f32 = -math.pi;
            while (hue < math.pi) : (hue += 0.1) gain = @min(gain, saturate(hue, dst).C / saturate(hue, src).C);
            for (0..n) |e| {
                var ich = ipt2ich(load(lut, e));
                ich.C *= gain;
                store(lut, e, ich2ipt(ich));
            }
        },
        .darken => {
            const points = [6]RGB{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 1, 1 }, .{ 1, 0, 1 }, .{ 1, 1, 0 } };
            var gain: f32 = 1.0;
            for (points) |pt| {
                const q = ipt2rgb(rgb2ipt(pt, src), dst);
                gain = @min(gain, 1.0 / @max(q[0], @max(q[1], q[2])));
            }
            for (0..n) |e| {
                var rgb = ipt2rgb(load(lut, e), dst);
                for (&rgb) |*v| v.* *= gain;
                store(lut, e, clipGamma(rgb2ipt(rgb, dst), c.colorimetric_gamma, dst));
            }
        },
    }
}

inline fn load(lut: []const f32, e: usize) IPT {
    const v = lut[e * GamutParams.stride ..][0..3];
    return .{ .I = v[0], .P = v[1], .T = v[2] };
}

inline fn store(lut: []f32, e: usize, c: IPT) void {
    lut[e * GamutParams.stride ..][0..3].* = .{ c.I, c.P, c.T };
}

// ----------------------------------------------------------- peak detection

pub const PeakParams = struct {
    smoothing_period: f32 = 20,
    scene_threshold_low: f32 = 1,
    scene_threshold_high: f32 = 3,
    percentile: f32 = 100,
    black_cutoff: f32 = 1,

    pub fn useHistogram(p: *const PeakParams) bool {
        return p.percentile > 0 and p.percentile < 100;
    }
};

/// One frame's measurement (PQ), as libplacebo's update_peak_buf derives
/// it from the detection buffer; `bias` is active / total work groups.
/// `max_pq` is the (percentile) peak of linear maxRGB, also the scene max;
/// `scene_avg_pq` the PQ of the mean linear maxRGB; `max_y_pq` the peak luma.
pub const Measure = struct { avg_pq: f32, max_pq: f32, bias: f32, scene_avg_pq: f32, max_y_pq: f32 };

/// The smoothed state of libplacebo's sh_color_map_obj.peak.
pub const PeakState = struct {
    avg_pq: f32 = 0,
    max_pq: f32 = 0,
    scene_avg_pq: f32 = 0,
    scene_max_pq: f32 = 0,
    max_y_pq: f32 = 0,

    pub fn update(s: *PeakState, m: Measure, p: *const PeakParams) void {
        var v = [5]f32{ m.avg_pq, m.max_pq, m.scene_avg_pq, m.max_pq, m.max_y_pq };
        const st = [5]*f32{ &s.avg_pq, &s.max_pq, &s.scene_avg_pq, &s.scene_max_pq, &s.max_y_pq };
        if (s.avg_pq == 0) {
            for (st, v) |d, x| d.* = x;
        } else {
            // ignore small deviations (rounding error)
            const epsilon: f32 = 1.0 / 16383.0;
            for (st, &v) |d, *x| {
                if (@abs(x.* - d.*) < epsilon) x.* = d.*;
            }
        }
        const coeff: f32 = if (p.smoothing_period == 0) 1.0 else 1.0 - @exp(-1.0 / p.smoothing_period);
        for (st, v) |d, x| d.* += coeff * (x - d.*);

        if (p.scene_threshold_low > 0 and p.scene_threshold_high > 0) {
            const log10_pq: f32 = 1e-2;
            const delta = m.bias * @abs(v[0] - s.avg_pq);
            const mix_coeff = smoothstep(p.scene_threshold_low * log10_pq, p.scene_threshold_high * log10_pq, delta);
            for (st, v) |d, x| d.* = mix(d.*, x, mix_coeff);
        }
    }

    /// pl_get_detected_hdr_metadata (null without data).
    pub fn detected(s: *const PeakState) ?Hdr {
        if (s.avg_pq == 0) return null;
        const scene_max = hdrRescale(.pq, .nits, s.scene_max_pq);
        return .{
            // max_y is not percentile-bounded; maxRGB bounds it
            .max_pq_y = @min(s.max_y_pq, s.max_pq),
            .avg_pq_y = s.avg_pq,
            .scene_avg = hdrRescale(.pq, .nits, s.scene_avg_pq),
            .scene_max = .{ scene_max, scene_max, scene_max },
        };
    }
};

test "tone LUT spline maps 1000 nits to SDR inside the output range" {
    var p: ToneParams = .{ .function = .spline, .param = 0.5 };
    const src: Csp = .{ .prim = .bt2020, .trc = .pq, .hdr = .{ .max_luma = 1000, .min_luma = 0.005 } };
    const dst: Csp = .{ .prim = .bt709, .trc = .bt1886, .hdr = .{ .max_luma = 203, .min_luma = 0.203 } };
    const li = nominalLuma(&src, .any, .pq);
    const lo = nominalLuma(&dst, .hdr10, .pq);
    p.input_min = li.min;
    p.input_max = li.max;
    p.output_min = lo.min;
    p.output_max = lo.max;
    p.inferParams();
    var lut: [256]f32 = undefined;
    p.generate(&lut);
    for (lut) |v| try std.testing.expect(v >= p.output_min - 1e-6 and v <= p.output_max + 1e-6);
    // monotonic
    for (1..lut.len) |i| try std.testing.expect(lut[i] >= lut[i - 1] - 1e-6);
}

test "primaries clip and superset" {
    const p3 = rawPrimaries(.display_p3);
    const b709 = rawPrimaries(.bt709);
    try std.testing.expect(primariesSuperset(p3, b709));
    try std.testing.expect(!primariesSuperset(b709, p3));
    try std.testing.expect(primariesClip(b709, p3).eql(b709));
}
