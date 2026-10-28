//! Tonemap: HDR -> SDR/HDR tone and gamut mapping, the pipeline vs-placebo's
//! Tonemap runs through libplacebo's renderer (pl_render_image with
//! pl_render_default_params), on the CPU and row by row:
//!
//!   decode (YCbCr BT.2020 NCL -> RGB, lanczos chroma upsampling)
//!   -> linearize (PQ or HLG) -> pl_shader_color_map_ex (IPT; 1D tone LUT,
//!   saturation hull, 3D gamut LUT) -> delinearize -> encode.
//!
//! Peak detection follows pl_shader_detect_peak (16x16 work-group PQ
//! statistics, IIR smoothing, scene-change hysteresis), but deterministic:
//! frame n replays the smoothing over a fixed window of earlier frames
//! (PeakCache) instead of depending on the order frames were rendered in.
//!
//! Not 1:1 with the GPU: pow/exp/log use vcl.exp2Bounded and a short log2
//! (about 1e-6 relative), the gamut LUT stays f32 (libplacebo rounds it to
//! 16 bit), 16-bit output is rounded instead of blue-noise dithered, and
//! PQ signals are clamped to [0, 1] before decoding.

const std = @import("std");
const math = std.math;

const common = @import("../common.zig");
const simd = @import("simd.zig");
const vcl = @import("../vcl.zig");
pub const tc = @import("tonemap_color.zig");
pub const dovi = @import("dovi.zig");

const allocator = common.kernel_allocator;
const io = common.io;

const VL = std.simd.suggestVectorLength(f32) orelse 4;
const V = @Vector(VL, f32);
const VU = @Vector(VL, u32);
const VB = @Vector(VL, bool);

inline fn sp(x: f32) V {
    return @splat(x);
}

inline fn madd(a: V, b: V, c: V) V {
    return simd.madd(V, a, b, c);
}

/// PL_MIX
fn mixf(a: f32, b: f32, x: f32) f32 {
    return x * b + (1 - x) * a;
}

/// vs-placebo's `src_csp` / `dst_csp` values.
pub const Space = enum(u8) { sdr = 0, hdr10 = 1, hlg = 2, dovi = 3 };

pub const min_size = 64;

pub const Config = struct {
    src_csp: Space = .hdr10,
    dst_csp: Space = .sdr,
    /// null: the dst_csp's own primaries
    dst_prim: ?tc.Prim = null,
    /// nits; 0 = from metadata
    src_max: f32 = 0,
    src_min: f32 = 0,
    dst_max: f32 = 0,
    dst_min: f32 = 0,
    peak_detect: bool = true,
    peak: tc.PeakParams = .{},
    gamut_fn: tc.GamutFn = .perceptual,
    tone_fn: tc.ToneFn = .spline,
    tone_param: f32 = 0.5,
    metadata: tc.Metadata = .any,
    show_clipping: bool = false,
    contrast_recovery: f32 = 0,
    contrast_smoothness: f32 = 3.5,
    /// pl_color_map_params fields vs-placebo does not expose (libplacebo's
    /// defaults): tone LUT size, mastering-peak soft clip, film response
    /// strength, expert constants, gamut 3D LUT size (I, C, h) and its
    /// tricubic sampling, gamut expansion and inverse tone mapping
    lut_size: u32 = 256,
    mastering_clip: f32 = 1.0,
    film_strength: f32 = 0.3,
    tone_constants: tc.ToneConstants = .{},
    gamut_constants: tc.GamutConstants = .{},
    lut3d_size: [3]u32 = .{ 48, 32, 256 },
    lut3d_tricubic: bool = false,
    gamut_expansion: bool = false,
    inverse_tone_mapping: bool = false,

    width: u32,
    height: u32,
    rgb: bool,
    float: bool,
    ssw: u1 = 0,
    ssh: u1 = 0,

    /// Largest smoothing_period whose window fits `max_window`.
    pub const max_smoothing_period = 60;
    pub const max_window = 5 * max_smoothing_period;

    /// Frames n - window + 1 .. n feed frame n's smoothed peak: five IIR
    /// time constants, after which the start state weighs < 0.7%. 0 when no
    /// frame can run detection.
    pub fn peakWindow(c: *const Config) u32 {
        if (!c.peakUsable()) return 0;
        std.debug.assert(c.peak.smoothing_period <= max_smoothing_period);
        const w = @ceil(5 * c.peak.smoothing_period);
        return @intFromFloat(math.clamp(w, 1, max_window));
    }

    /// Whether any frame can have Setup.peak_active. The target and a PQ
    /// source's peak (10000 nits) don't depend on frame metadata, so a
    /// target at 10000 nits rules detection out up front. An HLG source's
    /// peak is the target's when the target is HDR (inferMap), else it comes
    /// from per-frame metadata.
    pub fn peakUsable(c: *const Config) bool {
        if (!c.peak_detect or !(c.metadata == .any or c.metadata == .cie_y)) return false;
        const s: Setup = .init(c, &.{});
        return if (c.src_csp == .hlg) !s.dst.isHdr() else s.peak_active;
    }
};

/// Per-frame metadata the adapter reads from frame properties (the props
/// vs-placebo's Tonemap reads). Zero means absent.
pub const FrameMeta = struct {
    /// _Range / _ColorRange; null when untagged
    full_range: ?bool = null,
    /// VapourSynth _ChromaLocation
    chroma_loc: ?i64 = null,
    max_cll: f32 = 0,
    max_fall: f32 = 0,
    mastering_max: f32 = 0,
    mastering_min: f32 = 0,
    /// PLSceneAvg / PLSceneMax (nits); scene_max_len 1 (CIE Y) or 3 (MaxRGB)
    scene_avg: f32 = 0,
    scene_max: [3]f32 = .{ 0, 0, 0 },
    scene_max_len: u32 = 0,
    /// MasteringDisplayPrimariesX/Y (both of length 3)
    prim_x: ?[3]f32 = null,
    prim_y: ?[3]f32 = null,
    white: tc.Xy = .{},
    /// DolbyVisionRPU: its DM data (when signalled) and guessed profile,
    /// used whatever src_csp is, as vs-placebo does
    dovi_dm: ?dovi.DmData = null,
    dovi_profile: u8 = 0,
    /// the frame's resolved reshaping/matrix state (src_csp dovi)
    dovi_meta: ?*const dovi.Meta = null,
};

/// The renderer's view of one frame: inferred source and target spaces.
pub const Setup = struct {
    src: tc.Csp,
    dst: tc.Csp,
    full: bool,
    chroma_loc: i64,
    peak_active: bool,
    /// Dolby Vision decode (src_csp dovi)
    dovi: ?*const dovi.Meta = null,

    pub fn init(cfg: *const Config, m: *const FrameMeta) Setup {
        var src: tc.Csp = .{ .prim = .bt2020, .trc = if (cfg.src_csp == .hlg) .hlg else .pq };
        src.hdr.max_luma = cfg.src_max;
        src.hdr.min_luma = cfg.src_min;
        src.hdr.max_cll = m.max_cll;
        src.hdr.max_fall = m.max_fall;
        if (cfg.src_max < 1) src.hdr.max_luma = m.mastering_max;
        if (cfg.src_min <= 0) src.hdr.min_luma = m.mastering_min;
        if (m.scene_max_len == 1) {
            src.hdr.avg_pq_y = tc.hdrRescale(.nits, .pq, m.scene_avg);
            src.hdr.max_pq_y = tc.hdrRescale(.nits, .pq, m.scene_max[0]);
        } else if (m.scene_max_len == 3) {
            src.hdr.scene_avg = m.scene_avg;
            src.hdr.scene_max = m.scene_max;
        }
        if (m.prim_x != null and m.prim_y != null) {
            const x = m.prim_x.?;
            const y = m.prim_y.?;
            src.hdr.prim.red = .{ .x = x[0], .y = y[0] };
            src.hdr.prim.green = .{ .x = x[1], .y = y[1] };
            src.hdr.prim.blue = .{ .x = x[2], .y = y[2] };
            if (m.white.x != 0 and m.white.y != 0) src.hdr.prim.white = m.white;
        } else {
            // vs-placebo's default mastering display: DCI-P3 D65
            src.hdr.prim.merge(tc.rawPrimaries(.display_p3));
        }

        var dst: tc.Csp = switch (cfg.dst_csp) {
            .hdr10 => .{ .prim = .bt2020, .trc = .pq },
            .hlg => .{ .prim = .bt2020, .trc = .hlg },
            else => .{ .prim = .bt709, .trc = .bt1886 },
        };
        if (cfg.dst_prim) |p| {
            // libplacebo's infer_both_ref for unknown target primaries
            dst.prim = if (p != .unknown) p else if (src.prim.isWideGamut()) .bt709 else src.prim;
        }
        dst.hdr.max_luma = cfg.dst_max;
        dst.hdr.min_luma = cfg.dst_min;

        // vs-placebo's DolbyVisionRPU metadata, over the props
        if (m.dovi_dm) |*dm| {
            const pq4095 = struct {
                fn f(v: u16) f32 {
                    return @as(f32, @floatFromInt(v)) / 4095.0;
                }
            }.f;
            // DoVi -> PQ keeps the source black: the RPU's minimum is not
            // necessarily what the coded values hold
            if (cfg.src_csp == .dovi and cfg.dst_csp == .hdr10) {
                dst.hdr.min_luma = src.hdr.min_luma;
            } else if (!dm.compressed) {
                src.hdr.min_luma = tc.hdrRescale(.pq, .nits, pq4095(dm.source_min_pq));
            }
            // compressed DM data carries no source range (libdovi leaves it
            // zero, which vs-placebo would take as the peak): keep the props'
            if (!dm.compressed) src.hdr.max_luma = tc.hdrRescale(.pq, .nits, pq4095(dm.source_max_pq));
            if (dm.level1) |l1| {
                src.hdr.avg_pq_y = pq4095(l1.avg_pq);
                src.hdr.max_pq_y = pq4095(l1.max_pq);
            }
            if (dm.level6) |l6| if (src.hdr.max_cll == 0 or src.hdr.max_fall == 0) {
                src.hdr.max_cll = @floatFromInt(l6.max_cll);
                src.hdr.max_fall = @floatFromInt(l6.max_fall);
            };
        }

        tc.inferMap(&src, &dst);

        // hdr_update_peak's conditions: detection also runs alongside
        // dynamic metadata, unless both HDR10+ and CIE Y values are present
        const max_peak = if (src.trc == .hlg) src.hdr.max_luma else tc.Csp.nominalPeak(src.trc) * tc.sdr_white;
        const active = cfg.peak_detect and src.isHdr() and max_peak > dst.hdr.max_luma + 1e-6 and
            !(src.hdr.hasHdr10Plus() and src.hdr.hasCieY()) and (cfg.metadata == .any or cfg.metadata == .cie_y);

        const dv = if (cfg.src_csp == .dovi) m.dovi_meta else null;
        return .{
            .src = src,
            .dst = dst,
            // profile 5 output is full range
            .full = if (dv != null and m.dovi_profile == 5) true else m.full_range orelse cfg.rgb,
            .chroma_loc = m.chroma_loc orelse 0,
            .peak_active = active,
            .dovi = dv,
        };
    }
};

// ------------------------------------------------------------- vector math

/// log2 for x > 0 (see adaptive_grain_mask.log2Fast); <= 0 is clamped to
/// the smallest normal.
inline fn log2Fast(x_in: V) V {
    const VI = @Vector(VL, i32);
    const x = @max(x_in, @as(V, @bitCast(@as(VU, @splat(0x00800000)))));
    const off: VU = @splat(0x3f3504f3);
    const t = @as(VU, @bitCast(x)) -% off;
    const e: V = @floatFromInt(@as(VI, @bitCast(t)) >> @splat(23));
    const m: V = @bitCast((t & @as(VU, @splat(0x007fffff))) +% off);
    const f = m - sp(1.0);
    const s = f / (f + sp(2.0));
    const z = s * s;
    const l2e = 2.0 / std.math.ln2;
    var p = sp(l2e / 9.0);
    p = madd(p, z, sp(l2e / 7.0));
    p = madd(p, z, sp(l2e / 5.0));
    p = madd(p, z, sp(l2e / 3.0));
    p = madd(p, z, sp(l2e));
    return madd(s, p, e);
}

/// x^y for x >= 0 (0 for x <= 0).
inline fn powPos(x: V, y: f32) V {
    const r = vcl.exp2Bounded(6, sp(y) * log2Fast(x));
    return @select(f32, x > sp(0), r, sp(0));
}

inline fn expV(x: V) V {
    return vcl.exp2Bounded(6, x * sp(std.math.log2e));
}

inline fn lnV(x: V) V {
    return log2Fast(x) * sp(std.math.ln2);
}

// PQ transfer LUTs. Two pows and a divide per channel, 18 times a pixel,
// were most of the per-pixel cost; a linearly interpolated LUT of (value,
// next) pairs (simd.gatherPairsF32) is about twice as cheap per call. A
// single pow (BT.1886) stays a formula: it costs less than a lookup. Interpolation error, re-encoded to the
// PQ domain: <= 2e-7 above signal 0.02 and <= 2.3e-5 below it, where the
// curve leaves 0 (EOTF: sqrt of it in 2048 uniform steps, squared after
// interpolation, which halves the error near black), and <= 8.4e-7 for the
// OETF (128 steps per octave from 2^-64; 64 steps left 30-code errors on
// saturated colors, whose RGB is a difference of large LMS terms); 1e-5 is
// about a 16-bit code. A gather-free alternative was measured slower on
// Zen 3: a 16-segment degree-6 polynomial of sqrt(EOTF) with its
// coefficients picked by vpermps (as accurate as this LUT) cost 20.6 TSC
// cycles per 8 lanes against 14.7, and 3.1 against 2.1 with 16 threads.

const eotf_steps = 2048;

/// sqrt of the PQ EOTF at i / eotf_steps, 10000-nit normalized (pad entry
/// at the end).
const eotf_tab: [eotf_steps + 1][2]f32 = blk: {
    @setEvalBranchQuota(1_000_000);
    var t: [eotf_steps + 2]f32 = undefined;
    for (0..eotf_steps + 1) |i| t[i] = @floatCast(@sqrt(pqEotf64(@as(f64, @floatFromInt(i)) / eotf_steps)));
    t[eotf_steps + 1] = t[eotf_steps];
    break :blk pairUp(eotf_steps + 1, t);
};

/// (t[i], t[i + 1]) pairs for one-gather interpolation.
fn pairUp(comptime n: usize, t: [n + 1]f32) [n][2]f32 {
    var r: [n][2]f32 = undefined;
    for (&r, 0..) |*e, i| e.* = .{ t[i], t[i + 1] };
    return r;
}

/// f on [2^-octaves, 1], interpolated linearly inside 2^seg_bits segments
/// per octave; the float's exponent and top mantissa bits pick the segment.
fn LogLut(comptime octaves: u32, comptime seg_bits: u5, comptime f: fn (f64) f64) type {
    return struct {
        const n = octaves << seg_bits;
        const fbits = 23 - seg_bits;
        /// f(2^(e - octaves) * (1 + m / 2^seg_bits)) at e << seg_bits | m,
        /// plus a pad entry for x = 1.
        const tab: [n + 1][2]f32 = blk: {
            @setEvalBranchQuota(8_000_000);
            var t: [n + 2]f32 = undefined;
            for (0..n + 1) |j| {
                const e: f64 = @floatFromInt(j >> seg_bits);
                const m: f64 = @floatFromInt(j & ((1 << seg_bits) - 1));
                t[j] = @floatCast(f(math.pow(f64, 2, e - octaves) * (1 + m / (1 << seg_bits))));
            }
            t[n + 1] = t[n];
            break :blk pairUp(n + 1, t);
        };

        inline fn eval(x: V) V {
            return logLutEval(octaves, seg_bits, &tab, x);
        }
    };
}

/// Lookup in a `LogLut`-layout table (comptime or built at run time).
inline fn logLutEval(comptime octaves: u32, comptime seg_bits: u5, tab: [*]const [2]f32, x: V) V {
    const fbits = 23 - seg_bits;
    // 2^-octaves, built from its bits (a comptime pow here runs out of
    // branch quota in large inlined callers)
    const lo: f32 = @bitCast(@as(u32, (127 - octaves) << 23));
    const b: VU = @bitCast(simd.clampV(VL, x, sp(lo), sp(1)));
    const i = (b >> @splat(fbits)) - @as(VU, @splat((127 - octaves) << seg_bits));
    const fr: V = @floatFromInt(b & @as(VU, @splat((1 << fbits) - 1)));
    return lerpTab(tab, i, fr * sp(1.0 / @as(f32, 1 << fbits)));
}

const OetfLut = LogLut(64, 7, pqOetf64);

fn pqEotf64(x: f64) f64 {
    const m1: f64 = 2610.0 / 4096.0 / 4.0;
    const m2: f64 = 2523.0 / 4096.0 * 128.0;
    const c1: f64 = 3424.0 / 4096.0;
    const c2: f64 = 2413.0 / 4096.0 * 32.0;
    const c3: f64 = 2392.0 / 4096.0 * 32.0;
    const xp = math.pow(f64, x, 1.0 / m2);
    return math.pow(f64, @max(xp - c1, 0) / (c2 - c3 * xp), 1.0 / m1);
}

fn pqOetf64(y: f64) f64 {
    const m1: f64 = 2610.0 / 4096.0 / 4.0;
    const m2: f64 = 2523.0 / 4096.0 * 128.0;
    const c1: f64 = 3424.0 / 4096.0;
    const c2: f64 = 2413.0 / 4096.0 * 32.0;
    const c3: f64 = 2392.0 / 4096.0 * 32.0;
    const yp = math.pow(f64, y, m1);
    return math.pow(f64, (c1 + c2 * yp) / (1 + c3 * yp), m2);
}

inline fn lerpTab(tab: [*]const [2]f32, i: VU, f: V) V {
    const p = simd.gatherPairsF32(VL, tab, i);
    return madd(p[1] - p[0], f, p[0]);
}

/// PQ signal (clamped to [0, 1]) -> linear, 10000-nit normalized
inline fn pqEotf(x: V) V {
    const r = pqEotfRoot(x);
    return r * r;
}

/// sqrt of `pqEotf`
inline fn pqEotfRoot(x: V) V {
    const pos = simd.clampV(VL, x, sp(0), sp(1)) * sp(eotf_steps);
    const fl = @floor(pos);
    return lerpTab(&eotf_tab, @intFromFloat(fl), pos - fl);
}

/// linear (10000-nit normalized, clamped to [2^-64, 1]) -> PQ signal
inline fn pqOetf(x: V) V {
    return OetfLut.eval(x);
}

/// PQ signal -> linear, 10000-nit normalized (direct formula)
inline fn pqEotfMath(x_in: V) V {
    const x = powPos(simd.clampV(VL, x_in, sp(0), sp(1)), 1.0 / tc.pq_m2);
    const y = @max(x - sp(tc.pq_c1), sp(0)) / (sp(tc.pq_c2) - sp(tc.pq_c3) * x);
    return powPos(y, 1.0 / tc.pq_m1);
}

/// linear (10000-nit normalized) -> PQ signal (direct formula)
inline fn pqOetfMath(x_in: V) V {
    const x = powPos(x_in, tc.pq_m1);
    const y = madd(sp(tc.pq_c2), x, sp(tc.pq_c1)) / madd(sp(tc.pq_c3), x, sp(1));
    return powPos(y, tc.pq_m2);
}

/// atan2(y, x) in [-pi, pi]; (0, 0) gives 0.
inline fn atan2V(y: V, x: V) V {
    const ax = @abs(x);
    const ay = @abs(y);
    const mx = @max(ax, ay);
    const mn = @min(ax, ay);
    const t = mn / @max(mx, sp(1e-30));
    var a = vcl.atan(t);
    a = @select(f32, ay > ax, sp(math.pi / 2.0) - a, a);
    a = @select(f32, x < sp(0), sp(math.pi) - a, a);
    return @select(f32, y < sp(0), -a, a);
}

inline fn mat3(m: tc.Mat3, a: V, b: V, c: V) [3]V {
    var r: [3]V = undefined;
    inline for (0..3) |i| r[i] = madd(sp(m[i][0]), a, madd(sp(m[i][1]), b, sp(m[i][2]) * c));
    return r;
}

inline fn smoothstepV(e0: f32, e1: f32, x: V) V {
    const t = simd.clampV(VL, (x - sp(e0)) / sp(e1 - e0), sp(0), sp(1));
    return t * t * (sp(3) - sp(2) * t);
}

// ------------------------------------------------------- decode / encode

/// Transfer function constants for linearize (src) or delinearize (dst).
const Transfer = struct {
    trc: tc.Trc,
    /// HLG: OOTF gamma - 1 (linearize) or (1 - y) / y (delinearize)
    hlg_exp: f32 = 0,
    hlg_b: f32 = 0,
    /// HLG: nominal peak (SDR-white relative)
    hlg_max: f32 = 1,
    /// luma coefficients of the container primaries (HLG OOTF)
    coeffs: [3]f32 = .{ 0, 0, 0 },
    /// BT.1886: a^(-1/2.4) and b of pl_shader_delinearize
    gain: f32 = 1,
    b: f32 = 0,

    fn init(c: *const tc.Csp, comptime inverse: bool) Transfer {
        var t: Transfer = .{ .trc = c.trc };
        const r = tc.linearRange(c);
        switch (c.trc) {
            .bt1886 => {
                const lb = math.pow(f32, r.min, 1.0 / 2.4);
                const lw = math.pow(f32, r.max, 1.0 / 2.4);
                t.gain = 1.0 / (lw - lb);
                t.b = lb / (lw - lb);
            },
            .pq => {},
            .hlg => {
                const y = 1.2 * math.pow(f32, 1.111, @log2(r.max / tc.hlg_ref));
                t.hlg_b = @sqrt(3 * math.pow(f32, r.min / r.max, 1 / y));
                t.hlg_max = r.max;
                t.hlg_exp = if (inverse) (1 - y) / y else y - 1;
                t.coeffs = tc.rgb2xyz(tc.rawPrimaries(c.prim))[1];
            },
        }
        return t;
    }

    inline fn linearize(t: *const Transfer, comptime trc: tc.Trc, rgb: *[3]V) void {
        switch (trc) {
            .pq => inline for (rgb) |*x| {
                x.* = pqEotf(x.*) * sp(10000.0 / tc.sdr_white);
            },
            .hlg => {
                inline for (rgb) |*x| {
                    var v = madd(sp(1 - t.hlg_b), @max(x.*, sp(0)), sp(t.hlg_b));
                    v = @select(f32, v > sp(0.5), expV((v - sp(tc.hlg_c)) * sp(1.0 / tc.hlg_a)) + sp(tc.hlg_b), sp(4) * v * v);
                    x.* = v * sp(1.0 / 12.0);
                }
                const luma = madd(sp(t.coeffs[0]), rgb[0], madd(sp(t.coeffs[1]), rgb[1], sp(t.coeffs[2]) * rgb[2]));
                const k = sp(t.hlg_max) * powPos(luma, t.hlg_exp);
                inline for (rgb) |*x| x.* *= k;
            },
            .bt1886 => unreachable,
        }
    }

    inline fn delinearize(t: *const Transfer, comptime trc: tc.Trc, rgb: *[3]V) void {
        inline for (rgb) |*x| x.* = @max(x.*, sp(0));
        switch (trc) {
            // pow(x / a, 1 / 2.4) - b = a^(-1/2.4) * x^(1/2.4) - b
            .bt1886 => inline for (rgb) |*x| {
                x.* = madd(powPos(x.*, 1.0 / 2.4), sp(t.gain), sp(-t.b));
            },
            .pq => inline for (rgb) |*x| {
                x.* = pqOetf(x.* * sp(tc.sdr_white / 10000.0));
            },
            .hlg => {
                inline for (rgb) |*x| x.* *= sp(1.0 / t.hlg_max);
                const luma = madd(sp(t.coeffs[0]), rgb[0], madd(sp(t.coeffs[1]), rgb[1], sp(t.coeffs[2]) * rgb[2]));
                const k = sp(12) * powPos(@max(luma, sp(1e-6)), t.hlg_exp);
                inline for (rgb) |*x| {
                    const v = x.* * k;
                    const hi = madd(sp(tc.hlg_a), lnV(v - sp(tc.hlg_b)), sp(tc.hlg_c));
                    const lo = sp(0.5) * @sqrt(v);
                    x.* = (@select(f32, v > sp(1), hi, lo) - sp(t.hlg_b)) * sp(1 / (1 - t.hlg_b));
                }
            },
        }
    }
};

/// pl_color_repr_decode: raw samples -> nonlinear RGB, `rgb = m * v + c`.
/// YCbCr is always BT.2020 NCL, as vs-placebo tags it. Integer samples are
/// 16 bit; float YUV is full range with chroma centered on 0.
fn decodeTransform(rgb: bool, float: bool, full: bool, sys: enum { bt709, bt2020 }) struct { m: tc.Mat3, c: [3]f32 } {
    var m: tc.Mat3 = tc.identity;
    if (!rgb) {
        const lr: f32, const lg: f32, const lb: f32 = switch (sys) {
            .bt709 => .{ 0.2126, 0.7152, 0.0722 },
            .bt2020 => .{ 0.2627, 0.6780, 0.0593 },
        };
        m = .{
            .{ 1, 0, 2 * (1 - lr) },
            .{ 1, -2 * (1 - lb) * lb / lg, -2 * (1 - lr) * lr / lg },
            .{ 1, 2 * (1 - lb), 0 },
        };
    }
    var mul: [3]f64 = .{ 1, 1, 1 };
    var black: [3]f64 = .{ 0, 0, 0 };
    if (!float) {
        const s: f64 = 65536.0 / 65535.0;
        var ymax: f64 = 1;
        var ymin: f64 = 0;
        var cmax: f64 = 1;
        const cmid: f64 = 128.0 / 256.0 * s;
        if (!full) {
            ymax = 235.0 / 256.0 * s;
            ymin = 16.0 / 256.0 * s;
            cmax = 240.0 / 256.0 * s;
        }
        const ymul = 1.0 / (ymax - ymin);
        const cmul = 0.5 / (cmax - cmid);
        mul = .{ ymul, ymul, ymul };
        black = .{ ymin, ymin, ymin };
        if (!rgb) {
            mul[1] = cmul;
            mul[2] = cmul;
            black[1] = cmid;
            black[2] = cmid;
        }
        // samples arrive as raw 16-bit codes
        for (&mul) |*v| v.* /= 65535.0;
        for (&black) |*v| v.* *= 65535.0;
    }
    var c: [3]f32 = .{ 0, 0, 0 };
    for (0..3) |i| {
        var ci: f64 = 0;
        for (0..3) |j| {
            const e: f64 = @as(f64, m[i][j]) * mul[j];
            m[i][j] = @floatCast(e);
            ci -= e * black[j];
        }
        c[i] = @floatCast(ci);
    }
    return .{ .m = m, .c = c };
}

/// Lanczos (radius 3) taps for the two output phases of a 2x chroma
/// upsample: output 2k + p reads source k + off[p] + t, t = 0..5.
const Phases = struct {
    w: [2][6]f32,
    off: [2]i32,

    fn lanczos3(x: f64) f64 {
        if (x == 0) return 1;
        const a = @abs(x);
        if (a >= 3) return 0;
        const px = math.pi * x;
        return 3 * @sin(px) * @sin(px / 3) / (px * px);
    }

    /// `shift`: libplacebo plane shift in luma pixels (-0.5 left/top,
    /// 0 center, 0.5 bottom).
    fn init(shift: f64) Phases {
        var r: Phases = undefined;
        for (0..2) |p| {
            // luma 2k + p sits at chroma k + d
            const d = (@as(f64, @floatFromInt(p)) - shift) / 2 - 0.25;
            const base = @floor(d);
            const f = d - base;
            var sum: f64 = 0;
            var w: [6]f64 = undefined;
            for (0..6) |t| {
                w[t] = if (f == 0) (if (t == 2) 1 else 0) else lanczos3(f + 2 - @as(f64, @floatFromInt(t)));
                sum += w[t];
            }
            for (0..6) |t| r.w[p][t] = @floatCast(w[t] / sum);
            r.off[p] = @as(i32, @intFromFloat(base)) - 2;
        }
        return r;
    }
};

fn chromaShift(loc: i64) [2]f64 {
    // VapourSynth _ChromaLocation: 0 left, 1 center, 2 top-left, 3 top,
    // 4 bottom-left, 5 bottom
    const sx: f64 = switch (loc) {
        1, 3, 5 => 0,
        else => -0.5,
    };
    const sy: f64 = switch (loc) {
        2, 3 => -0.5,
        4, 5 => 0.5,
        else => 0,
    };
    return .{ sx, sy };
}

const Decode = struct {
    m: tc.Mat3,
    c: [3]f32,
    hx: Phases,
    hy: Phases,
    lin: Transfer,
    /// Dolby Vision: replaces m/c and the PQ linearization
    dovi: ?DoviDecode = null,

    fn init(cfg: *const Config, s: *const Setup) Decode {
        const d = decodeTransform(cfg.rgb, cfg.float, s.full, .bt2020);
        const sh = chromaShift(s.chroma_loc);
        return .{
            .m = d.m,
            .c = d.c,
            .hx = .init(sh[0]),
            .hy = .init(sh[1]),
            .lin = .init(&s.src, false),
            .dovi = if (s.dovi) |m| .init(m) else null,
        };
    }
};

/// pl_shader_decode_color's Dolby Vision path for one frame: the RPU's
/// per-component reshaping of the base-layer codes (polynomial or MMR
/// pieces between pivots), its YCbCr -> nonlinear RGB matrix, then PQ LMS
/// -> linear BT.2020 RGB. Integer input only (16-bit codes of the 10-bit
/// BL, as vs-placebo requires).
const DoviDecode = struct {
    comp: [3]Comp,
    /// reshaped -> nonlinear (PQ) LMS: m * v + c
    m: tc.Mat3,
    c: [3]f32,
    /// linear LMS -> linear BT.2020 RGB
    lms2rgb: tc.Mat3,

    /// constant, then 7 coefficients per MMR order
    const ncoef = 1 + 3 * dovi.mmr_coeffs;
    const Comp = struct {
        /// pieces; 0: no reshaping
        pieces: u32 = 0,
        /// pivot range, which also clamps the inputs MMR reads
        lo: f32 = 0,
        hi: f32 = 1,
        /// piece p > 0 starts at bounds[p - 1]
        bounds: [dovi.max_pieces - 1]f32 = @splat(0),
        mmr: bool = false,
        /// the highest MMR order (lower-order pieces have zero coefficients)
        order: u32 = 0,
        /// polynomial: x0 + x1 s + x2 s^2
        coef: [dovi.max_pieces][ncoef]f32 = @splat(@splat(0)),
    };

    /// Dolby Vision's BT.2020 HPE LMS -> RGB (pl_shader_decode_color_ex)
    const lms2rgb_bt2020: tc.Mat3 = .{
        .{ 3.06441879, -2.16597676, 0.10155818 },
        .{ -0.65612108, 1.78554118, -0.12943749 },
        .{ 0.01736321, -0.04725154, 1.03004253 },
    };

    fn init(meta: *const dovi.Meta) DoviDecode {
        var d: DoviDecode = .{ .comp = @splat(.{}), .m = meta.nonlinear, .c = undefined, .lms2rgb = undefined };
        for (&d.comp, &meta.comp) |*o, *r| {
            if (r.num_pivots == 0) continue;
            o.pieces = r.num_pivots - 1;
            o.lo = r.pivots[0];
            o.hi = r.pivots[r.num_pivots - 1];
            for (0..o.pieces - 1) |i| o.bounds[i] = r.pivots[i + 1];
            o.mmr = r.mmr;
            for (0..o.pieces) |i| {
                if (!r.mmr) {
                    o.coef[i][0..3].* = r.poly_coeffs[i];
                } else {
                    o.order = @max(o.order, r.mmr_order[i]);
                    o.coef[i][0] = r.mmr_constant[i];
                    for (0..3) |j| o.coef[i][1 + j * dovi.mmr_coeffs ..][0..dovi.mmr_coeffs].* = r.mmr_coeffs[i][j];
                }
            }
        }
        // the signalled offsets are in units of the composed signal; the
        // matrix already includes the levels (pl_color_repr_decode_ex)
        for (0..3) |i| {
            var c: f64 = 0;
            for (0..3) |j| c -= @as(f64, meta.nonlinear[i][j]) * meta.nonlinear_offset[j];
            d.c[i] = @floatCast(c);
        }
        d.lms2rgb = tc.mul(lms2rgb_bt2020, meta.linear);
        return d;
    }

    /// The reshaped component `s` (clamped to its pivots in `sig`).
    inline fn reshape(o: *const Comp, sig: [3]V, s: V) V {
        const used = if (o.mmr) 1 + o.order * dovi.mmr_coeffs else 3;
        // each lane's piece: the coefficients of the last pivot <= s
        var k: [ncoef]V = undefined;
        for (0..used) |j| k[j] = sp(o.coef[0][j]);
        for (1..o.pieces) |p| {
            const in = s >= sp(o.bounds[p - 1]);
            for (0..used) |j| k[j] = @select(f32, in, sp(o.coef[p][j]), k[j]);
        }
        if (!o.mmr) return madd(madd(k[2], s, k[1]), s, k[0]);
        // MMR: the components and their cross products, up to the cube
        const x = [7]V{ sig[0], sig[1], sig[2], sig[0] * sig[1], sig[0] * sig[2], sig[1] * sig[2], sig[0] * sig[1] * sig[2] };
        var r = k[0];
        var t = x;
        for (0..o.order) |j| {
            if (j > 0) {
                for (&t, x) |*v, b| v.* *= b;
            }
            const kj = k[1 + j * dovi.mmr_coeffs ..];
            for (0..dovi.mmr_coeffs) |i| r = madd(kj[i], t[i], r);
        }
        return r;
    }
};

pub fn Planes(comptime T: type) type {
    return struct {
        ptr: [3][*]const T,
        /// in elements
        stride: [3]usize,
    };
}

pub fn PlanesOut(comptime T: type) type {
    return struct {
        ptr: [3][*]T,
        stride: [3]usize,
    };
}

/// Row buffers of one call. Each f32 row has room for whole vectors past
/// the width (2x-upsampled vectors included).
const Rows = struct {
    buf: []f32,
    a: []f32,
    b: []f32,
    c: []f32,
    cu: []f32,
    cv: []f32,
    pad: []f32,
    feat: []f32,
    /// color map scratch (the tone stage's maxRGB)
    aux: []f32,

    fn init(w: u32) !Rows {
        const n = std.mem.alignForward(usize, w, 2 * VL) + 4 * VL;
        const buf = try allocator.alloc(f32, 8 * n);
        @memset(buf, 0);
        return .{
            .buf = buf,
            .a = buf[0..n],
            .b = buf[n..][0..n],
            .c = buf[2 * n ..][0..n],
            .cu = buf[3 * n ..][0..n],
            .cv = buf[4 * n ..][0..n],
            .pad = buf[5 * n ..][0..n],
            .feat = buf[6 * n ..][0..n],
            .aux = buf[7 * n ..][0..n],
        };
    }

    fn deinit(r: *Rows) void {
        allocator.free(r.buf);
    }
};

fn convertRow(comptime T: type, src: [*]const T, dst: []f32, n: usize) void {
    var x: usize = 0;
    while (x + VL <= n) : (x += VL) {
        const v: @Vector(VL, T) = src[x..][0..VL].*;
        const f: V = if (T == f32) v else @floatFromInt(v);
        dst[x..][0..VL].* = f;
    }
    while (x < n) : (x += 1) dst[x] = if (T == f32) src[x] else @floatFromInt(src[x]);
}

fn verticalRow(comptime T: type, rows: [6][*]const T, w: [6]f32, dst: []f32, n: usize) void {
    var x: usize = 0;
    while (x + VL <= n) : (x += VL) {
        var acc = sp(0);
        inline for (0..6) |t| {
            const v: @Vector(VL, T) = rows[t][x..][0..VL].*;
            const f: V = if (T == f32) v else @floatFromInt(v);
            acc = madd(sp(w[t]), f, acc);
        }
        dst[x..][0..VL].* = acc;
    }
    while (x < n) : (x += 1) {
        var acc: f32 = 0;
        inline for (0..6) |t| acc += w[t] * (if (T == f32) rows[t][x] else @as(f32, @floatFromInt(rows[t][x])));
        dst[x] = acc;
    }
}

/// 2x horizontal upsample of `src[0..cn]` into `dst[0..2*cn]`.
fn horizontal2x(src: []f32, pad: []f32, ph: *const Phases, dst: []f32, cn: usize) void {
    // 3 edge samples on the left (offsets reach -3), the rest on the right
    @memset(pad[0..3], src[0]);
    @memcpy(pad[3..][0..cn], src[0..cn]);
    @memset(pad[3 + cn ..], src[cn - 1]);
    const o0: usize = @intCast(ph.off[0] + 3);
    const o1: usize = @intCast(ph.off[1] + 3);
    var k: usize = 0;
    while (k < cn) : (k += VL) {
        var e = sp(0);
        var o = sp(0);
        inline for (0..6) |t| {
            e = madd(sp(ph.w[0][t]), pad[k + o0 + t ..][0..VL].*, e);
            o = madd(sp(ph.w[1][t]), pad[k + o1 + t ..][0..VL].*, o);
        }
        const io2 = simd.interleaveF32(VL, e, o);
        dst[2 * k ..][0..VL].* = io2[0];
        dst[2 * k + VL ..][0..VL].* = io2[1];
    }
}

/// Raw samples of output row y into rows.a/b/c (chroma upsampled).
fn decodeRow(comptime T: type, cfg: *const Config, dec: *const Decode, src: Planes(T), y: usize, rows: *Rows) void {
    const w = cfg.width;
    convertRow(T, src.ptr[0] + y * src.stride[0], rows.a, w);
    if (cfg.rgb or (cfg.ssw == 0 and cfg.ssh == 0)) {
        convertRow(T, src.ptr[1] + y * src.stride[1], rows.b, w);
        convertRow(T, src.ptr[2] + y * src.stride[2], rows.c, w);
        return;
    }
    const cw = w >> cfg.ssw;
    const ch = cfg.height >> cfg.ssh;
    const outs = [2][]f32{ rows.b, rows.c };
    const tmps = [2][]f32{ rows.cu, rows.cv };
    for (0..2) |pi| {
        const p = pi + 1;
        // vertical: into tmp (or straight into the output without ssw)
        const vdst = if (cfg.ssw == 1) tmps[pi] else outs[pi];
        if (cfg.ssh == 1) {
            const ph = y & 1;
            const k: i64 = @intCast(y >> 1);
            var rp: [6][*]const T = undefined;
            for (0..6) |t| {
                const r = math.clamp(k + dec.hy.off[ph] + @as(i64, @intCast(t)), 0, @as(i64, ch) - 1);
                rp[t] = src.ptr[p] + @as(usize, @intCast(r)) * src.stride[p];
            }
            verticalRow(T, rp, dec.hy.w[ph], vdst, cw);
        } else {
            convertRow(T, src.ptr[p] + y * src.stride[p], vdst, cw);
        }
        if (cfg.ssw == 1) horizontal2x(tmps[pi], rows.pad, &dec.hx, outs[pi], cw);
    }
}

/// Decode matrix and transfer: rows a/b/c become linear RGB (SDR-white
/// relative, source primaries).
fn linearizeRow(dec: *const Decode, rows: *Rows, n: usize) void {
    if (dec.dovi) |*d| return doviRow(d, rows, n);
    // one loop per transfer (comptime): a runtime switch would inline every
    // variant into the loop body
    switch (dec.lin.trc) {
        .bt1886 => unreachable,
        inline else => |trc| {
            var x: usize = 0;
            while (x < n) : (x += VL) {
                var rgb = mat3(dec.m, rows.a[x..][0..VL].*, rows.b[x..][0..VL].*, rows.c[x..][0..VL].*);
                inline for (&rgb, dec.c) |*v, c| v.* += sp(c);
                dec.lin.linearize(trc, &rgb);
                storeRow3(rows, x, rgb);
            }
        },
    }
}

/// Dolby Vision decode: raw 16-bit codes in rows a/b/c become linear
/// BT.2020 RGB (SDR-white relative).
fn doviRow(d: *const DoviDecode, rows: *Rows, n: usize) void {
    var x: usize = 0;
    while (x < n) : (x += VL) {
        const raw = loadRow3(rows, x);
        // the reshaping reads code / 2^bl_depth: 16-bit samples of the
        // 10-bit BL carry it shifted by 6 (pl_color_repr_code_scale)
        var s: [3]V = undefined;
        var sig: [3]V = undefined;
        inline for (0..3) |c| {
            s[c] = raw[c] * sp(1.0 / 65536.0);
            sig[c] = simd.clampV(VL, s[c], sp(d.comp[c].lo), sp(d.comp[c].hi));
        }
        inline for (0..3) |c| {
            if (d.comp[c].pieces > 0) s[c] = simd.clampV(VL, DoviDecode.reshape(&d.comp[c], sig, sig[c]), sp(0), sp(1));
        }
        var lms = mat3(d.m, s[0], s[1], s[2]);
        // PQ LMS -> linear; pqEotf clamps the signal to [0, 1] as upstream
        inline for (&lms, d.c) |*v, c| v.* = pqEotf(v.* + sp(c));
        var rgb = mat3(d.lms2rgb, lms[0], lms[1], lms[2]);
        // upstream re-encodes max(rgb, 0) to PQ, which the next
        // linearization undoes
        inline for (&rgb) |*v| v.* = @max(v.*, sp(0)) * sp(10000.0 / tc.sdr_white);
        storeRow3(rows, x, rgb);
    }
}

// --------------------------------------------------------------- color map

const lut1d_max = 4096;
/// `Tone.g`'s layout: 2^-32 (10000-nit relative, about 2e-6 nits) up to 1
const g_octaves = 32;
const g_seg_bits = 7;

const Tone = struct {
    kind: enum { clip, linear, lut },
    /// clip: [a, b]; linear: clamp(a*x + b, 0, 1) * c + d;
    /// lut: position (x * a + b) * (n - 1)
    a: f32 = 0,
    b: f32 = 0,
    c: f32 = 0,
    d: f32 = 0,
    n: u32 = 0,
    /// (value, next) pairs
    lut: [lut1d_max][2]f32 = undefined,
    /// PQ EOTF(map(PQ OETF(x))) for linear x (10000-nit relative), in
    /// `LogLut`'s layout: the tone stage's curve from linear light to linear
    /// light (per cone and for maxRGB), one lookup instead of an OETF, a
    /// tone and an EOTF lookup. Inside an octave the segments are uniform in
    /// x, 1/128 of it wide, so linear interpolation of the near-power curve
    /// is within about 1e-5 relative.
    /// The clip curve's hard corners would be smeared over a segment, and
    /// in linear light it is just a clamp to [g_lo, g_hi], so it skips `g`.
    g: [(g_octaves << g_seg_bits) + 1][2]f32 = undefined,
    g_lo: f32 = 0,
    g_hi: f32 = 0,

    /// Fills `g` (after the curve itself).
    fn buildG(t: *Tone) void {
        if (t.kind == .clip) {
            t.g_lo = tc.hdrRescale(.pq, .nits, t.a) / 10000;
            t.g_hi = tc.hdrRescale(.pq, .nits, t.b) / 10000;
            return;
        }
        const n = g_octaves << g_seg_bits;
        var r: [n + 1 + VL]f32 = undefined;
        var j: u32 = 0;
        while (j <= n) : (j += VL) {
            // node j = e << seg_bits | m: 2^(e - octaves) * (1 + m / 2^seg_bits)
            const jv = std.simd.iota(u32, VL) + @as(VU, @splat(j));
            const bits = (jv + @as(VU, @splat((127 - g_octaves) << g_seg_bits))) << @splat(23 - g_seg_bits);
            const x = @min(@as(V, @bitCast(bits)), sp(1));
            r[j..][0..VL].* = pqEotfMath(t.map(pqOetfMath(x)));
        }
        for (0..n + 1) |i| t.g[i] = .{ r[i], r[@min(i + 1, n)] };
    }

    /// PQ EOTF(map(PQ OETF(x))); x below 2^-32 (or negative) clamps up
    inline fn mapG(t: *const Tone, x: V) V {
        if (t.kind == .clip) return simd.clampV(VL, x, sp(t.g_lo), sp(t.g_hi));
        return logLutEval(g_octaves, g_seg_bits, &t.g, x);
    }

    inline fn map(t: *const Tone, x: V) V {
        switch (t.kind) {
            .clip => return simd.clampV(VL, x, sp(t.a), sp(t.b)),
            .linear => return madd(simd.clampV(VL, madd(x, sp(t.a), sp(t.b)), sp(0), sp(1)), sp(t.c), sp(t.d)),
            .lut => {
                const pos = simd.clampV(VL, madd(x, sp(t.a), sp(t.b)), sp(0), sp(1)) * sp(@floatFromInt(t.n - 1));
                const fl = @floor(pos);
                const i: VU = @intFromFloat(fl);
                const f = pos - fl;
                return lerpTab(&t.lut, i, f);
            },
        }
    }
};

/// One 3D gamut LUT (see tc.GamutParams.generate for the layout).
pub const Lut3d = struct {
    params: tc.GamutParams,
    data: []f32,
    /// plans holding it (under GamutCache.lock)
    refs: u32 = 0,
};

/// Gamut LUTs by parameters, built on first use. The input gamut comes from
/// per-frame mastering metadata, so a clip normally needs one, but spliced
/// or edited sources can change it on any frame: at most `max_kept` LUTs
/// (6 MB each by default) stay cached, more only while plans pin them, and
/// unused ones go least recently used first. A plan's LUT is pinned until
/// `release`.
pub const GamutCache = struct {
    lock: std.Io.Mutex = .init,
    /// least recently used first
    list: std.ArrayList(*Lut3d) = .empty,

    pub const max_kept = 4;

    pub fn deinit(g: *GamutCache) void {
        for (g.list.items) |l| {
            std.debug.assert(l.refs == 0);
            destroy(l);
        }
        g.list.deinit(allocator);
    }

    fn destroy(l: *Lut3d) void {
        allocator.free(l.data);
        allocator.destroy(l);
    }

    /// Frees unused LUTs, oldest first, until at most `keep` remain (fewer
    /// can't be reached while plans pin more).
    fn trim(g: *GamutCache, keep: usize) void {
        var i: usize = 0;
        while (g.list.items.len > keep and i < g.list.items.len) {
            const l = g.list.items[i];
            if (l.refs == 0) {
                _ = g.list.orderedRemove(i);
                destroy(l);
            } else i += 1;
        }
    }

    pub fn release(g: *GamutCache, l: *const Lut3d) void {
        g.lock.lockUncancelable(io);
        defer g.lock.unlock(io);
        const m: *Lut3d = @constCast(l);
        std.debug.assert(m.refs > 0);
        m.refs -= 1;
        // back to the bound once a burst of pinned LUTs drains (may free
        // `l`: the caller must not use it after this)
        if (m.refs == 0) g.trim(max_kept);
    }

    fn sameParams(a: *const tc.GamutParams, b: *const tc.GamutParams) bool {
        return a.function == b.function and a.input_gamut.eql(b.input_gamut) and a.output_gamut.eql(b.output_gamut) and
            a.min_luma == b.min_luma and a.max_luma == b.max_luma and
            std.meta.eql(a.constants, b.constants) and
            a.size_I == b.size_I and a.size_C == b.size_C and a.size_h == b.size_h;
    }

    /// The LUT for `p`, pinned: the caller must `release` it.
    fn get(g: *GamutCache, p: *const tc.GamutParams) !*const Lut3d {
        g.lock.lockUncancelable(io);
        defer g.lock.unlock(io);
        for (g.list.items, 0..) |l, i| {
            if (sameParams(&l.params, p)) {
                l.refs += 1;
                // most recently used last
                _ = g.list.orderedRemove(i);
                g.list.appendAssumeCapacity(l);
                return l;
            }
        }
        g.trim(max_kept - 1);
        try g.list.ensureUnusedCapacity(allocator, 1);
        const l = try allocator.create(Lut3d);
        errdefer allocator.destroy(l);
        l.* = .{ .params = p.*, .data = try allocator.alloc(f32, p.lutLen()), .refs = 1 };
        p.generate(l.data, std.Thread.getCpuCount() catch 1);
        g.list.appendAssumeCapacity(l);
        return l;
    }
};

const Clip = struct { hi_rgb: f32, lo_rgb: f32, hi_i: f32, lo_i: f32 };

pub const Plan = struct {
    dec: Decode,
    /// src == dst: decode and encode only
    passthrough: bool,
    map: enum { none, matrix, full },
    mat: tc.Mat3,
    /// full: container RGB -> the tone-mapping RGB (the mastering
    /// display's, when used); null when they are the same
    space: ?tc.Mat3,
    /// full: tone-mapping RGB (SDR-white relative) -> LMS (10000-nit
    /// relative); LMS -> target RGB
    rgb2lms: tc.Mat3,
    lms2rgb: tc.Mat3,
    /// contrast recovery's feature map: container RGB -> LMS
    feat_rgb2lms: tc.Mat3,
    /// mastering-peak soft clip: strength (0: off) and the peak (SDR-white
    /// relative)
    mclip: f32,
    mclip_norm: f32,
    mclip_inv: f32,
    need_tone: bool,
    tone: Tone,
    recovery: f32,
    out_min: f32,
    out_max: f32,
    /// per-cone tone application (10000-nit relative quantities)
    film: f32,
    flare: f32,
    knee_in: f32,
    inv_knee_gain: f32,
    exposure: f32,
    gamut: ?*const Lut3d,
    /// lut3d_tricubic
    tricubic: bool,
    g_scale: f32,
    g_off: f32,
    clip: ?Clip,
    delin: Transfer,
    enc: tc.Mat3,
    enc_c: [3]f32,
    /// integer outputs: the codes the target representation allows
    code_lo: [3]f32,
    code_hi: [3]f32,

    /// pl_shader_color_map_ex's setup for one frame; `detected` is the
    /// smoothed peak-detection state (applied when non-empty).
    pub fn init(plan: *Plan, cfg: *const Config, s: *const Setup, detected: ?tc.PeakState, cache: *GamutCache) !void {
        var src = s.src;
        var dst = s.dst;
        // authored dynamic metadata takes precedence over the measurement,
        // which only fills in what the source did not provide
        if (detected) |d| if (d.detected()) |m| {
            if (src.hdr.max_pq_y == 0) src.hdr.max_pq_y = m.max_pq_y;
            if (src.hdr.avg_pq_y == 0) src.hdr.avg_pq_y = m.avg_pq_y;
            if (src.hdr.scene_avg == 0) src.hdr.scene_avg = m.scene_avg;
            for (&src.hdr.scene_max, m.scene_max) |*v, mv| {
                if (v.* == 0) v.* = mv;
            }
        };
        tc.inferMap(&src, &dst);

        plan.dec = .init(cfg, s);
        // the Dolby Vision decode always runs (process' passthrough skips
        // linearizeRow)
        plan.passthrough = src.eql(&dst) and s.dovi == null;
        plan.map = .none;
        plan.mat = tc.identity;
        plan.space = null;
        plan.mclip = 0;
        plan.need_tone = false;
        plan.tone.kind = .clip;
        plan.recovery = 0;
        plan.exposure = 1;
        plan.gamut = null;
        plan.tricubic = cfg.lut3d_tricubic;
        plan.clip = null;
        plan.delin = .init(&dst, true);

        // encode: the inverse of the target representation's decode (RGB
        // out is full range; YUV keeps the source's range, as vs-placebo)
        const out_full = cfg.rgb or s.full;
        const e = decodeTransform(cfg.rgb, cfg.float, out_full, if (dst.trc == .bt1886) .bt709 else .bt2020);
        plan.enc = tc.invert(e.m);
        const ec = tc.apply(plan.enc, e.c);
        plan.enc_c = .{ -ec[0], -ec[1], -ec[2] };
        // the renderer clips integer outputs to the representation's range
        // (pl_color_repr_limits): limited range Y 16-235, C 16-240 (<< 8)
        plan.code_lo = @splat(0);
        plan.code_hi = @splat(65535);
        if (!out_full) {
            plan.code_lo = @splat(16 << 8);
            plan.code_hi = .{ 235 << 8, 240 << 8, 240 << 8 };
        }

        if (plan.passthrough) return;

        var tone: tc.ToneParams = .{ .function = cfg.tone_fn, .param = cfg.tone_param, .constants = cfg.tone_constants, .input_scaling = .pq, .output_scaling = .pq };
        const li = tc.nominalLuma(&src, cfg.metadata, .pq);
        tone.input_min = li.min;
        tone.input_max = li.max;
        tone.input_avg = li.avg;
        const lo = tc.nominalLuma(&dst, .hdr10, .pq);
        tone.output_min = lo.min;
        tone.output_max = lo.max;
        tone.inferParams();
        if (@abs(tone.input_max - tone.output_max) < 1e-6) tone.output_max = tone.input_max;
        if (@abs(tone.input_min - tone.output_min) < 1e-6) tone.output_min = tone.input_min;
        tone.input_min = tc.signalBlack(tone.input_min, .pq);
        tone.output_min = tc.signalBlack(tone.output_min, .pq);
        if (!cfg.inverse_tone_mapping) {
            // never exceed the source unless asked to
            tone.input_max = @max(tone.input_max, tone.output_max);
            // the curve must cover the metric's range (maxRGB carries
            // channel values above CIE Y), not just the luminance range
            const scene_max = @max(src.hdr.scene_max[0], @max(src.hdr.scene_max[1], src.hdr.scene_max[2]));
            if (scene_max > 0) tone.input_max = @max(tone.input_max, tc.hdrRescale(.nits, .pq, scene_max));
        }

        // Dynamic target: tone map into a virtual peak T between the display
        // peak P and the source peak S (a smooth maximum of P and a fixed PQ
        // distance below the scene average), and apply P/T as an exposure
        const out_min_phys = tone.output_min;
        var exposure: f32 = 1;
        var scene_avg = src.hdr.scene_avg;
        if (!(scene_avg > 0) and src.hdr.avg_pq_y > 0) scene_avg = tc.hdrRescale(.pq, .nits, src.hdr.avg_pq_y);
        if (scene_avg > 0 and !cfg.inverse_tone_mapping and tone.input_max > tone.output_max + 1e-4) {
            const dt_dim: f32 = 0.09;
            const dt_soft: f32 = 0.04;
            const p_pq = tone.output_max;
            const s_pq = tone.input_max;
            const u = p_pq;
            const v = tc.hdrRescale(.nits, .pq, scene_avg) - dt_dim;
            const t_pq = math.clamp(0.5 * (u + v) + math.hypot(0.5 * (u - v), dt_soft), p_pq, s_pq);
            if (t_pq > p_pq + 1e-4 and t_pq < s_pq - 1e-4) {
                exposure = tc.hdrRescale(.pq, .nits, p_pq) / tc.hdrRescale(.pq, .nits, t_pq);
                tone.output_max = t_pq;
                // the black floor pre-divided by the exposure, so it lands on
                // the display black after the exposure applies
                tone.output_min = tc.hdrRescale(.nits, .pq, tc.hdrRescale(.pq, .nits, out_min_phys) / exposure);
            }
        }

        // tone mapping and the gamut map's source extent share this space
        const space = tc.masteringSpace(&src);
        const src_space = if (space != null) src.hdr.prim else tc.rawPrimaries(src.prim);

        const gl = tc.nominalLuma(&dst, .hdr10, .pq);
        var gamut: tc.GamutParams = .{
            .function = cfg.gamut_fn,
            .input_gamut = src_space,
            .output_gamut = dst.hdr.prim,
            .min_luma = tc.signalBlack(gl.min, .pq),
            .max_luma = gl.max,
            .constants = cfg.gamut_constants,
            .size_I = cfg.lut3d_size[0],
            .size_C = cfg.lut3d_size[1],
            .size_h = cfg.lut3d_size[2],
        };
        // without gamut expansion, a bidirectional function never enlarges
        // the source gamut
        if (!cfg.gamut_expansion and gamut.function.bidirectional() and tc.primariesCompatible(gamut.input_gamut, gamut.output_gamut)) {
            gamut.output_gamut = tc.primariesClip(gamut.output_gamut, gamut.input_gamut);
        }

        plan.need_tone = !tone.noop() or exposure != 1;
        var need_gamut = !gamut.noop();

        var lms2rgb = tc.iptLms2Rgb(tc.rawPrimaries(dst.prim));
        if (need_gamut and gamut.function == .saturation) {
            lms2rgb = tc.mul(tc.mul(lms2rgb, tc.iptRgb2Lms(gamut.output_gamut)), tc.iptLms2Rgb(gamut.input_gamut));
            need_gamut = false;
        }

        if (!plan.need_tone and !need_gamut) {
            if (src.prim != dst.prim) {
                plan.map = .matrix;
                plan.mat = tc.mul(lms2rgb, tc.iptRgb2Lms(tc.rawPrimaries(src.prim)));
            }
            return;
        }

        plan.map = .full;
        plan.space = space;
        plan.rgb2lms = tc.scale(tc.iptRgb2Lms(src_space), tc.sdr_white / 10000.0);
        plan.feat_rgb2lms = tc.scale(tc.iptRgb2Lms(tc.rawPrimaries(src.prim)), tc.sdr_white / 10000.0);
        plan.lms2rgb = tc.scale(lms2rgb, 10000.0 / tc.sdr_white);
        plan.out_min = tone.output_min;
        plan.out_max = tone.output_max;
        plan.exposure = exposure;

        // Mastering-peak soft clip of the channels above the mastering peak,
        // trusted only while the measured (smoothed) peak luminance stays
        // below it: fades out between 1.1x and 1.4x of the peak.
        var mclip = math.clamp(cfg.mastering_clip, 0, 1);
        const master = s.src.hdr.max_luma;
        const max_y_pq = if (detected) |d| d.max_y_pq else 0;
        if (mclip > 0 and master > 0 and master < 9999 and max_y_pq > 0) {
            mclip *= 1 - tc.smoothstep(1.1, 1.4, tc.hdrRescale(.pq, .nits, max_y_pq) / master);
        } else mclip = 0;
        if (mclip > 1e-3) {
            plan.mclip = mclip;
            plan.mclip_norm = tc.hdrRescale(.nits, .norm, master);
            plan.mclip_inv = 1.0 / plan.mclip_norm;
        }

        if (cfg.show_clipping) {
            const eps: f32 = 1e-6;
            plan.clip = .{
                .hi_rgb = tc.hdrRescale(.pq, .norm, tone.input_max) + eps,
                .lo_rgb = tc.hdrRescale(.pq, .norm, tone.input_min) - eps,
                .hi_i = tone.input_max + eps,
                .lo_i = tone.input_min - eps,
            };
        }

        if (plan.need_tone) {
            switch (tone.function) {
                .clip => plan.tone = .{ .kind = .clip, .a = tone.input_min, .b = tone.input_max },
                .linear => {
                    const gain = tone.constants.exposure;
                    const range = tone.input_max - tone.input_min;
                    plan.tone = .{
                        .kind = .linear,
                        .a = gain / range,
                        .b = -gain / range * tone.input_min,
                        .c = tone.output_max - tone.output_min,
                        .d = tone.output_min,
                    };
                },
                else => {
                    const n = math.clamp(cfg.lut_size, 2, lut1d_max);
                    const range = tone.input_max - tone.input_min;
                    plan.tone = .{ .kind = .lut, .a = 1.0 / range, .b = -tone.input_min / range, .n = n };
                    var t: [lut1d_max]f32 = undefined;
                    tone.generate(t[0..n]);
                    for (0..n) |i| plan.tone.lut[i] = .{ t[i], t[@min(i + 1, n - 1)] };
                },
            }
            plan.tone.buildG();

            // The per-cone application: the film response's weight ramps up
            // as the shoulder attenuates the maxRGB gain below the knee's
            plan.flare = tc.hdrRescale(.pq, .nits, tone.output_min) / 10000;
            plan.film = math.clamp(cfg.film_strength, 0, 2);
            const tcn = &tone.constants;
            var knee = if (tone.input_avg != 0) tone.input_avg else mixf(tone.input_min, tone.input_max, tcn.knee_default);
            knee = math.clamp(knee, mixf(tone.input_min, tone.input_max, tcn.knee_minimum), mixf(tone.input_min, tone.input_max, tcn.knee_maximum));
            plan.knee_in = tc.hdrRescale(.pq, .nits, knee) / 10000;
            const knee_out = tc.hdrRescale(.pq, .nits, tone.sample(knee)) / 10000;
            plan.inv_knee_gain = @max(plan.knee_in, 1e-9) / @max(knee_out - plan.flare, 1e-9);
            // get_feature_map's conditions, then need_recovery
            const feature = cfg.contrast_recovery > 0 and cfg.contrast_smoothness > 1 and s.src.isHdr() and
                s.src.hdr.max_luma > s.dst.hdr.max_luma + 1e-6;
            if (feature and tone.input_max >= tone.output_max) plan.recovery = cfg.contrast_recovery;
        }

        if (need_gamut) {
            plan.gamut = try cache.get(&gamut);
            const range = gamut.max_luma - gamut.min_luma;
            plan.g_scale = 1.0 / range;
            plan.g_off = -gamut.min_luma / range;
        }
    }

    /// Unpins the plan's gamut LUT (after a successful `init`).
    pub fn deinit(plan: *Plan, cache: *GamutCache) void {
        if (plan.gamut) |g| cache.release(g);
        plan.gamut = null;
    }
};

/// lut3d_tricubic: libplacebo's `lut_tricubic` (sh_lut, SH_LUT_CUBIC), a
/// cubic B-spline over the LUT built from 8 trilinear lookups at offset
/// positions. The weights come from the unclamped position; the lookups
/// clamp to the edge like the GPU texture.
inline fn gamutLookupCubic(l: *const Lut3d, I: *V, P: *V, T: *V, xi: V, xc: V, xh: V) void {
    const Axis = struct {
        g0: V,
        h0: V,
        h1: V,

        inline fn init(x: V, n: u32) @This() {
            const scale: f32 = @floatFromInt(n - 1);
            const pos = x * sp(scale);
            const base = @floor(pos);
            const f = pos - base;
            const f2 = f * f;
            const inv = sp(1) - f;
            const inv2 = inv * inv;
            const w0 = sp(1.0 / 6.0) * inv2 * inv;
            const w1 = sp(2.0 / 3.0) - sp(0.5) * f2 * (sp(2) - f);
            const w2 = sp(2.0 / 3.0) - sp(0.5) * inv2 * (sp(2) - inv);
            const w3 = sp(1.0 / 6.0) * f2 * f;
            const g0 = w0 + w1;
            const g1 = w2 + w3;
            return .{
                .g0 = g0,
                .h0 = (w1 / g0 - sp(1) + base) * sp(1 / scale),
                .h1 = (w3 / g1 + sp(1) + base) * sp(1 / scale),
            };
        }
    };
    const ax = Axis.init(xi, l.params.size_I);
    const ac = Axis.init(xc, l.params.size_C);
    const ah = Axis.init(xh, l.params.size_h);
    const look = struct {
        inline fn f(lut: *const Lut3d, a: V, b: V, c: V) [3]V {
            var r: [3]V = undefined;
            gamutLookup(lut, &r[0], &r[1], &r[2], a, b, c);
            return r;
        }
    }.f;
    // mix(b, a, t) per channel: a * t + b * (1 - t)
    const mix3 = struct {
        inline fn f(b: [3]V, a: [3]V, t: V) [3]V {
            return .{ madd(a[0] - b[0], t, b[0]), madd(a[1] - b[1], t, b[1]), madd(a[2] - b[2], t, b[2]) };
        }
    }.f;
    // same order of lookups and mixes as the shader
    const c000 = mix3(look(l, ax.h1, ac.h0, ah.h0), look(l, ax.h0, ac.h0, ah.h0), ax.g0);
    const c010 = mix3(look(l, ax.h1, ac.h1, ah.h0), look(l, ax.h0, ac.h1, ah.h0), ax.g0);
    const z0 = mix3(c010, c000, ac.g0);
    const c001 = mix3(look(l, ax.h1, ac.h0, ah.h1), look(l, ax.h0, ac.h0, ah.h1), ax.g0);
    const c011 = mix3(look(l, ax.h1, ac.h1, ah.h1), look(l, ax.h0, ac.h1, ah.h1), ax.g0);
    const z1 = mix3(c011, c001, ac.g0);
    const r = mix3(z1, z0, ah.g0);
    I.* = r[0];
    P.* = r[1];
    T.* = r[2];
}

/// Trilinear lookup of IPT in the 3D LUT (clamped coordinates).
inline fn gamutLookup(l: *const Lut3d, I: *V, P: *V, T: *V, xi: V, xc: V, xh: V) void {
    const nI = l.params.size_I;
    const nC = l.params.size_C;
    const nH = l.params.size_h;
    const Lane = struct {
        inline fn split(x: V, n: u32) struct { VU, V } {
            const pos = simd.clampV(VL, x, sp(0), sp(1)) * sp(@floatFromInt(n - 1));
            const fl = @min(@floor(pos), sp(@floatFromInt(n - 2)));
            return .{ @intFromFloat(fl), pos - fl };
        }
    };
    const si = Lane.split(xi, nI);
    const sc = Lane.split(xc, nC);
    const sh = Lane.split(xh, nH);
    // Entries I and I + 1 are adjacent: one 32-byte load holds both, so the
    // C and h interpolations run 8 wide and only the last (I) one is 4 wide.
    // Each lane's (I, P, T, pad) result stays in a register; 8 of them are
    // transposed into the three outputs (scalar stores reloaded as vectors
    // would fail store-to-load forwarding).
    const V8 = @Vector(8, f32);
    const V4 = @Vector(4, f32);
    const s_c: u32 = nI * 4;
    const s_h: u32 = nC * nI * 4;
    const base: [VL]u32 = sh[0] * @as(VU, @splat(s_h)) + sc[0] * @as(VU, @splat(s_c)) + si[0] * @as(VU, @splat(4));
    const fi: [VL]f32 = si[1];
    const fc: [VL]f32 = sc[1];
    const fh: [VL]f32 = sh[1];
    const lerp8 = struct {
        inline fn f(a: V8, b: V8, t: f32) V8 {
            return simd.madd(V8, b - a, @splat(t), a);
        }
    }.f;
    const d = l.data;
    var r: [VL]V4 = undefined;
    inline for (0..VL) |k| {
        const o = base[k];
        const e0 = lerp8(d[o..][0..8].*, d[o + s_c ..][0..8].*, fc[k]);
        const e1 = lerp8(d[o + s_h ..][0..8].*, d[o + s_h + s_c ..][0..8].*, fc[k]);
        const e = lerp8(e0, e1, fh[k]);
        const lo = @shuffle(f32, e, undefined, [4]i32{ 0, 1, 2, 3 });
        const hi = @shuffle(f32, e, undefined, [4]i32{ 4, 5, 6, 7 });
        r[k] = simd.madd(V4, hi - lo, @splat(fi[k]), lo);
    }
    var oI: [VL]f32 = undefined;
    var oP: [VL]f32 = undefined;
    var oT: [VL]f32 = undefined;
    if (VL % 8 != 0) {
        inline for (0..VL) |k| {
            oI[k] = r[k][0];
            oP[k] = r[k][1];
            oT[k] = r[k][2];
        }
    } else inline for (0..VL / 8) |q| {
        const g = r[q * 8 ..][0..8];
        // rows of two lanes, k and k + 4: (I, P, T, _) in each 128-bit half
        var R: [4]V8 = undefined;
        inline for (0..4) |k| R[k] = @shuffle(f32, g[k], g[k + 4], [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
        const t0 = @shuffle(f32, R[0], R[1], [8]i32{ 0, -1, 1, -2, 4, -5, 5, -6 });
        const t1 = @shuffle(f32, R[2], R[3], [8]i32{ 0, -1, 1, -2, 4, -5, 5, -6 });
        const t2 = @shuffle(f32, R[0], R[1], [8]i32{ 2, -3, 3, -4, 6, -7, 7, -8 });
        const t3 = @shuffle(f32, R[2], R[3], [8]i32{ 2, -3, 3, -4, 6, -7, 7, -8 });
        oI[q * 8 ..][0..8].* = @shuffle(f32, t0, t1, [8]i32{ 0, 1, -1, -2, 4, 5, -5, -6 });
        oP[q * 8 ..][0..8].* = @shuffle(f32, t0, t1, [8]i32{ 2, 3, -3, -4, 6, 7, -7, -8 });
        oT[q * 8 ..][0..8].* = @shuffle(f32, t2, t3, [8]i32{ 0, 1, -1, -2, 4, 5, -5, -6 });
    }
    I.* = oI;
    P.* = oP;
    T.* = oT;
}

/// pl_shader_color_map_ex's stages on one vector (libplacebo v7.387). The
/// split passes and the fused loop both call these, so they compute
/// identical values.
const Map = struct {
    /// Container linear RGB (SDR-white relative) -> the tone-mapping RGB:
    /// the mastering display's when used, then its mastering-peak soft
    /// clip, which folds per-channel excursions above the peak back toward
    /// it (knee at 95% of the peak).
    inline fn prep(comptime space: bool, comptime clip: bool, p: *const Plan, R: V, G: V, B: V) [3]V {
        var c = [3]V{ R, G, B };
        if (space) c = mat3(p.space.?, R, G, B);
        if (clip) {
            const knee: f32 = 0.95;
            inline for (&c) |*x| {
                var n = x.* * sp(p.mclip_inv);
                const t = @max(n - sp(knee), sp(0)) * sp(1.0 / (1.0 - knee));
                n = @min(n, sp(knee)) + sp(1.0 - knee) * t / (sp(1) + t);
                x.* = madd(n * sp(p.mclip_norm) - x.*, sp(p.mclip), x.*);
            }
        }
        return c;
    }

    /// tone-mapping RGB -> LMS (10000-nit relative)
    inline fn lms(p: *const Plan, c: [3]V) [3]V {
        return mat3(p.rgb2lms, c[0], c[1], c[2]);
    }

    /// LMS -> PQ-encoded LMS
    inline fn pq(l: [3]V) [3]V {
        return .{ pqOetf(@max(l[0], sp(0))), pqOetf(@max(l[1], sp(0))), pqOetf(@max(l[2], sp(0))) };
    }

    /// PQ LMS -> IPT
    inline fn ipt(lp: [3]V) [3]V {
        return mat3(tc.lms2ipt, lp[0], lp[1], lp[2]);
    }

    /// maxRGB of the tone-mapping RGB, 10000-nit relative, clamped to [0, 1]
    inline fn maxRgb(c: [3]V) V {
        return simd.clampV(VL, @max(c[0], @max(c[1], c[2])) * sp(tc.sdr_white / 10000.0), sp(0), sp(1));
    }

    /// The tone application in the cone stage of IPT: the curve per cone
    /// (`o`), and the chromaticity-preserving endpoint, the gain of the
    /// maxRGB channel (`ga`), blended per cone by the film response weight,
    /// which follows the cone's own share and ramps to the full response as
    /// the shoulder attenuation deepens past the knee. Below the knee both
    /// endpoints coincide. `l` is the LMS, `lp` its PQ (recovery only, the
    /// plain curve goes through `Tone.g`), `lx` the maxRGB; returns the
    /// mapped LMS (10000-nit relative, exposure applied).
    inline fn tone(comptime cr: bool, p: *const Plan, l: [3]V, lp: if (cr) [3]V else void, lx: V, feat: ?*const [VL]f32) [3]V {
        var o: [3]V = undefined;
        if (cr) {
            // detail-preserving recovery: tone map at the low-frequency
            // level (the feature map), then re-add the detail
            const in = madd(sp(tc.lms2ipt[0][0]), lp[0], madd(sp(tc.lms2ipt[0][1]), lp[1], sp(tc.lms2ipt[0][2]) * lp[2]));
            const d = simd.clampV(VL, in, sp(0), sp(1)) - simd.clampV(VL, feat.?.*, sp(0), sp(1));
            inline for (0..3) |k| {
                const lo = simd.clampV(VL, lp[k] - d, sp(0), sp(1));
                const base = p.tone.map(lp[k]);
                const sharp = p.tone.map(lo) + (lp[k] - lo);
                o[k] = pqEotf(simd.clampV(VL, madd(sharp - base, sp(p.recovery), base), sp(p.out_min), sp(p.out_max)));
            }
        } else inline for (0..3) |k| {
            o[k] = p.tone.mapG(l[k]);
        }
        const fl = sp(p.flare);
        const ox = p.tone.mapG(lx);
        const ga = @max(ox - fl, sp(0)) / @max(lx, sp(1e-9));
        var u = simd.clampV(VL, sp(1) - ga * sp(p.inv_knee_gain), sp(0), sp(1));
        u = @select(f32, lx > sp(p.knee_in), u * u, sp(0));
        const inv = sp(1) / @max(@max(l[0], @max(l[1], l[2])), sp(1e-9));
        var r: [3]V = undefined;
        inline for (0..3) |k| {
            const s = simd.clampV(VL, l[k] * inv, sp(0), sp(1));
            const w = sp(p.film) * madd(sp(1) - s, u, s);
            const a = l[k] * ga;
            r[k] = sp(p.exposure) * (fl + madd((o[k] - fl) - a, w, a));
        }
        return r;
    }

    /// IPT -> normalized gamut LUT coordinates (intensity, chroma, hue).
    inline fn coords(p: *const Plan, c: [3]V) [3]V {
        return .{
            madd(c[0], sp(p.g_scale), sp(p.g_off)),
            sp(2) * @sqrt(c[1] * c[1] + c[2] * c[2]),
            madd(atan2V(c[2], c[1]), sp(0.5 / math.pi), sp(0.5)),
        };
    }

    /// IPT -> linear RGB of the target primaries
    inline fn fromIpt(p: *const Plan, c: [3]V) [3]V {
        const lp = mat3(tc.ipt2lms, c[0], c[1], c[2]);
        return mat3(p.lms2rgb, pqEotf(lp[0]), pqEotf(lp[1]), pqEotf(lp[2]));
    }

    /// LMS (10000-nit relative) -> linear RGB of the target primaries. The
    /// IPT round trip it skips clamps negative cones to 0 (its PQ OETF).
    inline fn fromLms(p: *const Plan, l: [3]V) [3]V {
        return mat3(p.lms2rgb, @max(l[0], sp(0)), @max(l[1], sp(0)), @max(l[2], sp(0)));
    }
};

fn colorMapRow(comptime kind: @FieldType(Plan, "map"), p: *const Plan, rows: *Rows, feat: ?[]const f32, n: usize) void {
    switch (kind) {
        .none => {},
        .matrix => {
            var x: usize = 0;
            while (x < n) : (x += VL) {
                const out = mat3(p.mat, rows.a[x..][0..VL].*, rows.b[x..][0..VL].*, rows.c[x..][0..VL].*);
                storeRow3(rows, x, out);
            }
        },
        .full => if (p.clip != null) colorMapFused(p, rows, feat, n) else colorMapSplit(p, rows, feat, n),
    }
}

inline fn loadRow3(rows: *const Rows, x: usize) [3]V {
    return .{ rows.a[x..][0..VL].*, rows.b[x..][0..VL].*, rows.c[x..][0..VL].* };
}

inline fn storeRow3(rows: *Rows, x: usize, v: [3]V) void {
    rows.a[x..][0..VL].* = v[0];
    rows.b[x..][0..VL].* = v[1];
    rows.c[x..][0..VL].* = v[2];
}

/// The full map as one pass per stage over the row. A fused iteration is
/// one long dependent chain (matrices, OETF/EOTF gathers, the tone curves,
/// atan2, 3D LUT), more than Zen 3's 256-entry reorder buffer, so
/// iterations could not overlap (the libplacebo v7.371 port's fused loop:
/// llvm-mca 467 cycles per 8 px against a 155-cycle floor). Short passes
/// let the core run several independent iterations at once; the row
/// buffers stay in L1/L2. Rows: a/b/c carry the pixel, aux the maxRGB and,
/// for contrast recovery, cu/cv/pad the PQ LMS while tone mapping.
fn colorMapSplit(p: *const Plan, rows: *Rows, feat: ?[]const f32, n: usize) void {
    // the tricubic lookup inlines 8 trilinear ones: above the default quota
    @setEvalBranchQuota(100_000);
    const flags = @as(u3, @intFromBool(p.space != null)) | @as(u3, @intFromBool(p.mclip > 0)) << 1 | @as(u3, @intFromBool(p.need_tone)) << 2;
    switch (flags) {
        inline else => |f| prepPass(f & 1 != 0, f & 2 != 0, f & 4 != 0, p, rows, n),
    }

    var x: usize = 0;
    if (p.need_tone) {
        if (p.recovery > 0) {
            while (x < n) : (x += VL) {
                const lp = Map.pq(loadRow3(rows, x));
                rows.cu[x..][0..VL].* = lp[0];
                rows.cv[x..][0..VL].* = lp[1];
                rows.pad[x..][0..VL].* = lp[2];
            }
            tonePass(true, p, rows, feat, n);
        } else tonePass(false, p, rows, feat, n);
        if (p.gamut == null) return;
    }
    x = 0;
    while (x < n) : (x += VL) storeRow3(rows, x, Map.ipt(Map.pq(loadRow3(rows, x))));

    const g = p.gamut.?;
    x = 0;
    while (x < n) : (x += VL) storeRow3(rows, x, Map.coords(p, loadRow3(rows, x)));
    x = 0;
    switch (p.tricubic) {
        inline else => |cubic| while (x < n) : (x += VL) {
            const c = loadRow3(rows, x);
            var o: [3]V = undefined;
            const lookup = if (cubic) gamutLookupCubic else gamutLookup;
            lookup(g, &o[0], &o[1], &o[2], c[0], c[1], c[2]);
            storeRow3(rows, x, o);
        },
    }
    x = 0;
    while (x < n) : (x += VL) storeRow3(rows, x, Map.fromIpt(p, loadRow3(rows, x)));
}

/// Container RGB -> LMS of the tone-mapping RGB, plus its maxRGB in aux.
fn prepPass(comptime space: bool, comptime clip: bool, comptime tone: bool, p: *const Plan, rows: *Rows, n: usize) void {
    var x: usize = 0;
    while (x < n) : (x += VL) {
        const v = loadRow3(rows, x);
        const c = Map.prep(space, clip, p, v[0], v[1], v[2]);
        if (tone) rows.aux[x..][0..VL].* = Map.maxRgb(c);
        storeRow3(rows, x, Map.lms(p, c));
    }
}

/// LMS (a/b/c; recovery: its PQ in cu/cv/pad) -> the tone-mapped LMS, or
/// straight to the target RGB without a gamut map (the IPT round trip in
/// between is an identity up to rounding).
fn tonePass(comptime cr: bool, p: *const Plan, rows: *Rows, feat: ?[]const f32, n: usize) void {
    var x: usize = 0;
    while (x < n) : (x += VL) {
        const lp = if (cr) [3]V{ rows.cu[x..][0..VL].*, rows.cv[x..][0..VL].*, rows.pad[x..][0..VL].* } else {};
        const l = Map.tone(cr, p, loadRow3(rows, x), lp, rows.aux[x..][0..VL].*, if (cr) feat.?[x..][0..VL] else null);
        storeRow3(rows, x, if (p.gamut != null) l else Map.fromLms(p, l));
    }
}

/// show_clipping: one pass, since the clip masks span every stage.
fn colorMapFused(p: *const Plan, rows: *Rows, feat: ?[]const f32, n: usize) void {
    // every stage inlined into one body exceeds the default comptime quota
    @setEvalBranchQuota(100_000);
    const c = p.clip.?;
    var x: usize = 0;
    while (x < n) : (x += VL) {
        const v = loadRow3(rows, x);
        var col = v;
        if (p.space != null) col = Map.prep(true, false, p, col[0], col[1], col[2]);
        if (p.mclip > 0) col = Map.prep(false, true, p, col[0], col[1], col[2]);
        const l = Map.lms(p, col);
        const lp = Map.pq(l);
        var ipt = Map.ipt(lp);
        const hi_rgb = sp(c.hi_rgb);
        const lo_rgb = sp(c.lo_rgb);
        var clip_hi = (col[0] > hi_rgb) | (col[1] > hi_rgb) | (col[2] > hi_rgb) | (ipt[0] > sp(c.hi_i));
        var clip_lo = (col[0] < lo_rgb) | (col[1] < lo_rgb) | (col[2] < lo_rgb) | (ipt[0] < sp(c.lo_i));
        var out: [3]V = undefined;
        if (p.need_tone) {
            const f = if (feat) |fr| fr[x..][0..VL] else null;
            const lt = if (p.recovery > 0) Map.tone(true, p, l, lp, Map.maxRgb(col), f) else Map.tone(false, p, l, {}, Map.maxRgb(col), f);
            if (p.gamut != null) ipt = Map.ipt(Map.pq(lt)) else out = Map.fromLms(p, lt);
        }
        if (p.gamut) |g| {
            const xc = Map.coords(p, ipt);
            // 1e-6 slack: a tone-mapped black lands exactly on the LUT's
            // lower bound, and the rescale rounds below it
            const e = 1e-6;
            clip_lo = clip_lo | (xc[0] < sp(-e)) | (xc[1] < sp(-e)) | (xc[2] < sp(-e));
            clip_hi = clip_hi | (xc[0] > sp(1 + e)) | (xc[1] > sp(1 + e)) | (xc[2] > sp(1 + e));
            if (p.tricubic) {
                gamutLookupCubic(g, &ipt[0], &ipt[1], &ipt[2], xc[0], xc[1], xc[2]);
            } else gamutLookup(g, &ipt[0], &ipt[1], &ipt[2], xc[0], xc[1], xc[2]);
            out = Map.fromIpt(p, ipt);
        }
        if (@reduce(.Or, clip_hi | clip_lo)) {
            // highlight: clipped highs inverted towards red, lows teal
            const k = (out[0] + out[1] + out[2]) * sp(2.0 / 3.0);
            var hc: [3]V = undefined;
            inline for (0..3) |i| hc[i] = simd.clampV(VL, k - out[i], sp(0), sp(1));
            const cmin = @min(hc[0], @min(hc[1], hc[2]));
            const cmax = @max(hc[0], @max(hc[1], hc[2]));
            const sm = smoothstepV(0, 0.3, cmax - cmin);
            const red = [3]f32{ 1, 0, 0 };
            const teal = [3]f32{ 0, 0.3, 0.3 };
            inline for (0..3) |i| {
                const t = simd.clampV(VL, (hc[i] - (cmin - sp(1e-6))) / (cmax - (cmin - sp(1e-6))), sp(0), sp(1));
                const sat = t * t * (sp(3) - sp(2) * t);
                const hi = madd(sat - sp(red[i]), sm, sp(red[i]));
                const lo = (out[i] + sp(teal[i])) * sp(0.5);
                out[i] = @select(f32, clip_hi, hi, @select(f32, clip_lo, lo, out[i]));
            }
        }
        storeRow3(rows, x, out);
    }
}

/// Linear RGB -> the target transfer, in place. Its own pass, ahead of
/// encodeRow: three independent pow (or LUT) chains per vector overlap
/// across iterations only in a short loop body.
fn delinearizeRow(t: *const Transfer, rows: *Rows, n: usize) void {
    switch (t.trc) {
        inline else => |trc| {
            var x: usize = 0;
            while (x < n) : (x += VL) {
                var rgb = loadRow3(rows, x);
                t.delinearize(trc, &rgb);
                storeRow3(rows, x, rgb);
            }
        },
    }
}

fn encodeRow(comptime O: type, p: *const Plan, rows: *Rows, dst: PlanesOut(O), y: usize, n: usize) void {
    var x: usize = 0;
    var tail: [3][VL]O = undefined;
    while (x < n) : (x += VL) {
        const rgb = loadRow3(rows, x);
        const e = mat3(p.enc, rgb[0], rgb[1], rgb[2]);
        inline for (0..3) |i| {
            const v = e[i] + sp(p.enc_c[i]);
            // integer targets: the encode matrix already yields 16-bit codes,
            // clipped to the representation's range (the bounds are integers)
            const ov: @Vector(VL, O) = if (O == f32) v else blk: {
                const q = @floor(simd.clampV(VL, v, sp(p.code_lo[i]), sp(p.code_hi[i])) + sp(0.5));
                break :blk @intFromFloat(q);
            };
            const row = dst.ptr[i] + y * dst.stride[i];
            if (x + VL <= n) {
                row[x..][0..VL].* = ov;
            } else {
                tail[i] = ov;
                @memcpy(row[x..n], tail[i][0 .. n - x]);
            }
        }
    }
}

// ------------------------------------------------------ contrast recovery

/// Low-resolution PQ intensity map (pl_shader_extract_features, then a
/// bicubic B-spline lowpass to 1/contrast_smoothness of the size) and its
/// B-spline upsampling taps.
const Features = struct {
    w: u32,
    h: u32,
    low: []f32,
    xt: []Tap4,
    yt: []Tap4,
    vrow: []f32,

    const Tap4 = struct { i: [4]u32, w: [4]f32 };

    fn deinit(f: *Features) void {
        allocator.free(f.low);
        allocator.free(f.xt);
        allocator.free(f.yt);
        allocator.free(f.vrow);
    }

    fn bspline(fr: f32) [4]f32 {
        const inv = 1 - fr;
        return .{
            inv * inv * inv / 6,
            2.0 / 3.0 - 0.5 * fr * fr * (2 - fr),
            2.0 / 3.0 - 0.5 * inv * inv * (2 - inv),
            fr * fr * fr / 6,
        };
    }

    fn upTaps(out: []Tap4, low_n: u32) void {
        const n: f32 = @floatFromInt(out.len);
        const ln: f32 = @floatFromInt(low_n);
        for (out, 0..) |*t, i| {
            const u = (@as(f32, @floatFromInt(i)) + 0.5) / n * ln - 0.5;
            const fl = @floor(u);
            t.w = bspline(u - fl);
            const base: i64 = @intFromFloat(fl);
            for (0..4) |k| t.i[k] = @intCast(math.clamp(base - 1 + @as(i64, @intCast(k)), 0, @as(i64, low_n) - 1));
        }
    }

    /// Bicubic (B = 1, C = 0) downsampling taps from `n` to `m` samples,
    /// widened by the ratio, mirrored at the edges.
    fn downTaps(n: u32, m: u32, idx: *std.ArrayList(u32), wts: *std.ArrayList(f32), starts: []u32) !void {
        const ratio = @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(m));
        const radius = 2 * @max(ratio, 1);
        for (0..m) |j| {
            starts[j] = @intCast(idx.items.len);
            const center = (@as(f64, @floatFromInt(j)) + 0.5) * ratio - 0.5;
            const lo: i64 = @intFromFloat(@ceil(center - radius));
            const hi: i64 = @intFromFloat(@floor(center + radius));
            var sum: f64 = 0;
            const first = wts.items.len;
            var i = lo;
            while (i <= hi) : (i += 1) {
                const d = @abs(@as(f64, @floatFromInt(i)) - center) / @max(ratio, 1);
                const wv = cubicB1(d);
                if (wv == 0) continue;
                const nn: i64 = n;
                var mi = @mod(i, 2 * nn);
                if (mi >= nn) mi = 2 * nn - 1 - mi;
                try idx.append(allocator, @intCast(mi));
                try wts.append(allocator, @floatCast(wv));
                sum += wv;
            }
            for (wts.items[first..]) |*wv| wv.* = @floatCast(wv.* / sum);
        }
        starts[m] = @intCast(idx.items.len);
    }

    fn cubicB1(x: f64) f64 {
        // libplacebo's cubic() with b = 1, c = 0
        if (x < 1) return (4 + x * x * (-6 + x * 3)) / 4;
        if (x < 2) return (8 + x * (-12 + x * (6 - x))) / 4;
        return 0;
    }
};

fn intensityRow(p: *const Plan, rows: *Rows, dst: []f32, n: usize) void {
    var x: usize = 0;
    while (x < n) : (x += VL) {
        const lms = mat3(p.feat_rgb2lms, rows.a[x..][0..VL].*, rows.b[x..][0..VL].*, rows.c[x..][0..VL].*);
        const Lp = pqOetf(@max(lms[0], sp(0)));
        const Mp = pqOetf(@max(lms[1], sp(0)));
        const Sp = pqOetf(@max(lms[2], sp(0)));
        dst[x..][0..VL].* = madd(sp(tc.lms2ipt[0][0]), Lp, madd(sp(tc.lms2ipt[0][1]), Mp, sp(tc.lms2ipt[0][2]) * Sp));
    }
}

fn buildFeatures(comptime T: type, cfg: *const Config, p: *const Plan, src: Planes(T), rows: *Rows) !Features {
    const w = cfg.width;
    const h = cfg.height;
    const cw: u32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(w)) / cfg.contrast_smoothness));
    const ch: u32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(h)) / cfg.contrast_smoothness));

    var hidx: std.ArrayList(u32) = .empty;
    defer hidx.deinit(allocator);
    var hw: std.ArrayList(f32) = .empty;
    defer hw.deinit(allocator);
    var vidx: std.ArrayList(u32) = .empty;
    defer vidx.deinit(allocator);
    var vw: std.ArrayList(f32) = .empty;
    defer vw.deinit(allocator);
    const hs = try allocator.alloc(u32, cw + 1);
    defer allocator.free(hs);
    const vs = try allocator.alloc(u32, ch + 1);
    defer allocator.free(vs);
    try Features.downTaps(w, cw, &hidx, &hw, hs);
    try Features.downTaps(h, ch, &vidx, &vw, vs);

    const hbuf = try allocator.alloc(f32, @as(usize, h) * cw);
    defer allocator.free(hbuf);
    for (0..h) |y| {
        decodeRow(T, cfg, &p.dec, src, y, rows);
        linearizeRow(&p.dec, rows, w);
        intensityRow(p, rows, rows.feat, w);
        const hr = hbuf[y * cw ..][0..cw];
        for (0..cw) |j| {
            var acc: f32 = 0;
            for (hs[j]..hs[j + 1]) |k| acc += hw.items[k] * rows.feat[hidx.items[k]];
            hr[j] = acc;
        }
    }

    var f: Features = .{ .w = cw, .h = ch, .low = &.{}, .xt = &.{}, .yt = &.{}, .vrow = &.{} };
    errdefer f.deinit();
    f.low = try allocator.alloc(f32, @as(usize, cw) * ch);
    @memset(f.low, 0);
    for (0..ch) |j| {
        const lr = f.low[j * cw ..][0..cw];
        for (vs[j]..vs[j + 1]) |k| {
            const wv = vw.items[k];
            const hr = hbuf[@as(usize, vidx.items[k]) * cw ..][0..cw];
            for (lr, hr) |*o, v| o.* += wv * v;
        }
    }
    f.xt = try allocator.alloc(Features.Tap4, w);
    f.yt = try allocator.alloc(Features.Tap4, h);
    f.vrow = try allocator.alloc(f32, cw);
    Features.upTaps(f.xt, cw);
    Features.upTaps(f.yt, ch);
    return f;
}

fn featureRow(f: *const Features, y: usize, dst: []f32, n: usize) void {
    const t = f.yt[y];
    const r0 = f.low[t.i[0] * f.w ..][0..f.w];
    const r1 = f.low[t.i[1] * f.w ..][0..f.w];
    const r2 = f.low[t.i[2] * f.w ..][0..f.w];
    const r3 = f.low[t.i[3] * f.w ..][0..f.w];
    for (f.vrow, r0, r1, r2, r3) |*o, a, b, c, d| o.* = t.w[0] * a + t.w[1] * b + t.w[2] * c + t.w[3] * d;
    for (0..n) |x| {
        const xt = f.xt[x];
        dst[x] = xt.w[0] * f.vrow[xt.i[0]] + xt.w[1] * f.vrow[xt.i[1]] + xt.w[2] * f.vrow[xt.i[2]] + xt.w[3] * f.vrow[xt.i[3]];
    }
}

// ------------------------------------------------------------ frame passes

/// Tone/gamut map one frame (input sample type T, output O: u16 or f32).
pub fn process(comptime T: type, comptime O: type, cfg: *const Config, p: *const Plan, src: Planes(T), dst: PlanesOut(O)) !void {
    var rows = try Rows.init(cfg.width);
    defer rows.deinit();
    const w = cfg.width;

    var feat: ?Features = null;
    if (p.recovery > 0) feat = try buildFeatures(T, cfg, p, src, &rows);
    defer if (feat) |*f| f.deinit();

    for (0..cfg.height) |y| {
        decodeRow(T, cfg, &p.dec, src, y, &rows);
        if (p.passthrough) {
            var x: usize = 0;
            while (x < w) : (x += VL) {
                var rgb = mat3(p.dec.m, rows.a[x..][0..VL].*, rows.b[x..][0..VL].*, rows.c[x..][0..VL].*);
                inline for (&rgb, p.dec.c) |*v, c| v.* += sp(c);
                rows.a[x..][0..VL].* = rgb[0];
                rows.b[x..][0..VL].* = rgb[1];
                rows.c[x..][0..VL].* = rgb[2];
            }
            encodeRow(O, p, &rows, dst, y, w);
            continue;
        }
        linearizeRow(&p.dec, &rows, w);
        if (feat) |*f| featureRow(f, y, rows.feat, w);
        switch (p.map) {
            inline else => |k| colorMapRow(k, p, &rows, if (feat != null) rows.feat else null, w),
        }
        delinearizeRow(&p.delin, &rows, w);
        encodeRow(O, p, &rows, dst, y, w);
    }
}

const pq_max = 16383;
const hist_bits = 7;
const hist_bias = 1 << (hist_bits - 1);
const hist_bins = (1 << hist_bits) - hist_bias;
const pq_bits = 14;
/// fixed-point scale of the linear maxRGB statistics (LIN_BITS = 16)
const lin_max = 65535;

/// pl_shader_detect_peak + update_peak_buf's per-frame reduction over 16x16
/// work groups (black-cutoff pixels excluded from each group's averages):
/// PQ luma (average, peak) and linear maxRGB (average, peak or percentile),
/// in the mastering display's RGB when tone mapping uses it.
pub fn measure(comptime T: type, cfg: *const Config, s: *const Setup, src: Planes(T)) !tc.Measure {
    const w = cfg.width;
    const h = cfg.height;
    var rows = try Rows.init(w);
    defer rows.deinit();
    const dec: Decode = .init(cfg, s);
    const space = tc.masteringSpace(&s.src);
    const luma = tc.rgb2xyz(if (space != null) s.src.hdr.prim else tc.rawPrimaries(s.src.prim))[1];
    const cutoff = @max(cfg.peak.black_cutoff, 0) * 1e-2;
    const use_hist = cfg.peak.useHistogram();

    const tiles = (w + 15) / 16;
    const acc = try allocator.alloc(Tile, tiles);
    defer allocator.free(acc);
    const wv = std.mem.alignForward(usize, w, VL);
    const codes = try allocator.alloc(u32, 3 * wv);
    defer allocator.free(codes);
    const ypq = codes[0..wv];
    const mlin = codes[wv..][0..wv];
    const bins = codes[2 * wv ..][0..wv];

    var hist: [hist_bins]u64 = @splat(0);
    var frame_max: u32 = 0;
    var frame_rgb: u32 = 0;
    var sum_pq: u64 = 0;
    var sum_lin: u64 = 0;
    var wg_count: u64 = 0;
    var wg_active: u64 = 0;
    var blacks: u64 = 0;

    var y: usize = 0;
    while (y < h) {
        const band = @min(16, h - y);
        @memset(acc, .{});
        for (0..band) |dy| {
            decodeRow(T, cfg, &dec, src, y + dy, &rows);
            linearizeRow(&dec, &rows, w);
            var x: usize = 0;
            while (x < w) : (x += VL) {
                var rgb = loadRow3(&rows, x);
                if (space) |m| rgb = mat3(m, rgb[0], rgb[1], rgb[2]);
                var l = madd(sp(luma[0]), rgb[0], madd(sp(luma[1]), rgb[1], sp(luma[2]) * rgb[2]));
                l = simd.clampV(VL, l * sp(tc.sdr_white / 10000.0), sp(0), sp(1));
                l = pqOetf(l);
                if (cutoff != 0) l *= smoothstepV(0, cutoff, l);
                const q: VU = @intFromFloat(@min(l * sp(pq_max), sp(pq_max)));
                ypq[x..][0..VL].* = q;
                const mrgb = simd.clampV(VL, @max(rgb[0], @max(rgb[1], rgb[2])) * sp(tc.sdr_white / 10000.0), sp(0), sp(1));
                const ml: VU = @intFromFloat(mrgb * sp(lin_max));
                mlin[x..][0..VL].* = ml;
                if (use_hist) {
                    // the histogram bins maxRGB; black-cutoff pixels go to 0
                    const mq: VU = @intFromFloat(pqOetf(mrgb) * sp(pq_max));
                    const shifted: @Vector(VL, i32) = @bitCast(mq >> @splat(pq_bits - hist_bits));
                    var bin = shifted - @as(@Vector(VL, i32), @splat(hist_bias));
                    bin = @min(@max(bin, @as(@Vector(VL, i32), @splat(0))), @as(@Vector(VL, i32), @splat(hist_bins - 1)));
                    if (cutoff != 0) bin = @select(i32, q == @as(VU, @splat(0)), @as(@Vector(VL, i32), @splat(0)), bin);
                    bins[x..][0..VL].* = @bitCast(bin);
                }
            }
            // the GPU's invocations past the last row repeat it (clamped
            // sampling; only their store is skipped), so a partial band
            // still fills its 16x16 work groups
            const reps: usize = if (dy + 1 == band) 17 - band else 1;
            for (0..reps) |_| accumulateRow(acc, ypq, mlin, bins, w, use_hist, &hist);
        }
        for (acc) |tl| {
            const black = if (cutoff != 0) tl.black else 0;
            blacks += black;
            const num = tl.count - black;
            wg_count += 1;
            if (num > 0) {
                wg_active += 1;
                sum_pq += tl.sum / num;
                sum_lin += tl.lin / num;
                frame_max = @max(frame_max, tl.max);
                frame_rgb = @max(frame_rgb, tl.rgb);
            }
        }
        y += band;
    }

    const black = tc.hdr_black;
    if (wg_active == 0) return .{ .avg_pq = black, .max_pq = black, .bias = 0, .scene_avg_pq = black, .max_y_pq = black };
    if (use_hist) hist[0] -|= blacks;
    const active: f32 = @floatFromInt(wg_active);
    const max_rgb = tc.hdrRescale(.nits, .pq, 10000.0 * @as(f32, @floatFromInt(frame_rgb)) / lin_max);
    return .{
        .avg_pq = @as(f32, @floatFromInt(sum_pq)) / (active * pq_max),
        .max_pq = if (use_hist) percentilePeak(&hist, max_rgb, cfg.peak.percentile) else max_rgb,
        .bias = active / @as(f32, @floatFromInt(wg_count)),
        .scene_avg_pq = tc.hdrRescale(.nits, .pq, 10000.0 * @as(f32, @floatFromInt(sum_lin)) / (active * lin_max)),
        .max_y_pq = @as(f32, @floatFromInt(frame_max)) / pq_max,
    };
}

const Tile = struct { sum: u32 = 0, max: u32 = 0, lin: u32 = 0, rgb: u32 = 0, black: u32 = 0, count: u32 = 0 };

/// One row of PQ and linear maxRGB codes (and histogram bins) into its work
/// groups; columns past the width repeat the last one, like the rows past
/// the height.
fn accumulateRow(acc: []Tile, ypq: []const u32, mlin: []const u32, bins: []const u32, w: usize, use_hist: bool, hist: *[hist_bins]u64) void {
    const U16 = @Vector(16, u32);
    for (acc, 0..) |*tl, t| {
        const x0 = t * 16;
        var v: U16 = undefined;
        var m: U16 = undefined;
        if (x0 + 16 <= w) {
            v = ypq[x0..][0..16].*;
            m = mlin[x0..][0..16].*;
        } else inline for (0..16) |k| {
            v[k] = ypq[@min(x0 + k, w - 1)];
            m[k] = mlin[@min(x0 + k, w - 1)];
        }
        tl.sum += @reduce(.Add, v);
        tl.max = @max(tl.max, @reduce(.Max, v));
        tl.lin += @reduce(.Add, m);
        tl.rgb = @max(tl.rgb, @reduce(.Max, m));
        tl.black += @popCount(@as(u16, @bitCast(v == @as(U16, @splat(0)))));
        tl.count += 16;
        if (use_hist) {
            for (0..16) |k| hist[bins[@min(x0 + k, w - 1)]] += 1;
        }
    }
}

/// measure_peak's histogram percentile.
fn percentilePeak(hist: *const [hist_bins]u64, frame_max: f32, percentile: f32) f32 {
    var total: u64 = 0;
    for (hist) |v| total += v;
    if (total == 0) return frame_max;
    const target: u64 = @intFromFloat(@ceil(percentile / 100.0 * @as(f32, @floatFromInt(total))));
    if (target >= total) return frame_max;
    var sum: u64 = 0;
    for (hist, 0..) |v, i| {
        const next = sum + v;
        if (next < target) {
            sum = next;
            continue;
        }
        const count_low = sum;
        const count_high = next + 1;
        const pq_low = @as(f32, @floatFromInt((i + hist_bias) << (pq_bits - hist_bits))) / pq_max;
        var pq_high = @as(f32, @floatFromInt((i + 1 + hist_bias) << (pq_bits - hist_bits))) / pq_max;
        if (count_high > total) pq_high = @max(frame_max, pq_low);
        const ratio = @as(f32, @floatFromInt(target - count_low)) / @as(f32, @floatFromInt(count_high - count_low));
        return ratio * pq_high + (1 - ratio) * pq_low;
    }
    return frame_max;
}

/// Per-frame peak measurements, shared by every request. Each frame index
/// is measured once: the first request that needs it claims it, later ones
/// wait for its result instead of measuring it again (at 16 threads the
/// window frames are mostly in flight at the same time). Entries live in
/// pages of `page_len` frames, allocated when a frame in them is first
/// claimed: a clip's advertised length can be huge (std.Loop: 2^31 frames,
/// 50 GB as one flat array) while only the requested frames' pages exist.
/// Pages are never evicted, so an entry stays valid once published.
pub const PeakCache = struct {
    pages: []std.atomic.Value(?*Page),

    pub const Entry = struct { m: tc.Measure, active: bool };

    pub const page_len = 4096;
    const Page = struct {
        state: [page_len]std.atomic.Value(u8),
        vals: [page_len]Entry,
    };

    const empty = 0;
    const busy = 1;
    const done = 2;

    /// `n` frames; costs one pointer per page up front.
    pub fn init(n: usize) !PeakCache {
        const pages = try allocator.alloc(std.atomic.Value(?*Page), math.divCeil(usize, n, page_len) catch unreachable);
        for (pages) |*p| p.* = .init(null);
        return .{ .pages = pages };
    }

    pub fn deinit(c: *PeakCache) void {
        for (c.pages) |*slot| if (slot.raw) |p| allocator.destroy(p);
        allocator.free(c.pages);
    }

    pub fn ready(c: *const PeakCache, i: usize) bool {
        const p = c.pages[i / page_len].load(.acquire) orelse return false;
        return p.state[i % page_len].load(.acquire) == done;
    }

    /// The page holding frame i, allocated on first use (concurrent first
    /// users race to publish theirs; the losers free their copy).
    fn page(c: *PeakCache, i: usize) !*Page {
        const slot = &c.pages[i / page_len];
        if (slot.load(.acquire)) |p| return p;
        const p = try allocator.create(Page);
        for (&p.state) |*s| s.* = .init(empty);
        if (slot.cmpxchgStrong(null, p, .acq_rel, .acquire)) |winner| {
            allocator.destroy(p);
            return winner.?;
        }
        return p;
    }

    /// The entry, or null once the caller has claimed it; a claimer must
    /// `put` the measurement or `release` the claim. Waits while another
    /// request measures it (that request already holds its frame, so it
    /// cannot be waiting on us). Fails only allocating the frame's page.
    pub fn acquire(c: *PeakCache, i: usize) !?Entry {
        const p = try c.page(i);
        const st = &p.state[i % page_len];
        while (true) {
            switch (st.load(.acquire)) {
                done => return p.vals[i % page_len],
                empty => if (st.cmpxchgWeak(empty, busy, .acquire, .monotonic) == null) return null,
                else => std.Thread.yield() catch {},
            }
        }
    }

    /// Publishes a claimed entry (its page exists: `acquire` made it).
    pub fn put(c: *PeakCache, i: usize, e: Entry) void {
        const p = c.pages[i / page_len].load(.acquire).?;
        p.vals[i % page_len] = e;
        p.state[i % page_len].store(done, .release);
    }

    pub fn release(c: *PeakCache, i: usize) void {
        const p = c.pages[i / page_len].load(.acquire).?;
        p.state[i % page_len].store(empty, .release);
    }

    /// The smoothed state after the window's frames in order (a frame
    /// without peak detection resets it, as libplacebo's renderer does).
    pub fn replay(entries: []const Entry, p: *const tc.PeakParams) tc.PeakState {
        var s: tc.PeakState = .{};
        for (entries) |e| {
            if (e.active) s.update(e.m, p) else s = .{};
        }
        return s;
    }
};

test "lanczos phases" {
    const left = Phases.init(-0.5);
    try std.testing.expectEqual(@as(f32, 1), left.w[0][2]);
    var sum: f32 = 0;
    for (left.w[1]) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    // symmetric half-sample phase
    try std.testing.expectApproxEqAbs(left.w[1][2], left.w[1][3], 1e-6);
}

test "atan2V matches std" {
    const ys = [_]f32{ 0.3, -0.7, 0, 1e-3, -2, 0.5, -0.5, 0 };
    const xs = [_]f32{ 0.1, 0.2, -1, -3, -1e-4, 0, -0.5, 0 };
    var yv: [VL]f32 = @splat(0);
    var xv: [VL]f32 = @splat(0);
    for (0..@min(VL, ys.len)) |i| {
        yv[i] = ys[i];
        xv[i] = xs[i];
    }
    const r: [VL]f32 = atan2V(yv, xv);
    for (0..@min(VL, ys.len)) |i| {
        const want = if (ys[i] == 0 and xs[i] == 0) 0 else math.atan2(ys[i], xs[i]);
        try std.testing.expectApproxEqAbs(want, r[i], 1e-5);
    }
}

test "tricubic gamut lookup reproduces an affine LUT" {
    // a cubic B-spline reproduces affine data exactly, so away from the
    // clamped edges it must agree with the trilinear lookup
    var l: Lut3d = .{
        .params = .{
            .function = .perceptual,
            .input_gamut = tc.rawPrimaries(.bt2020),
            .output_gamut = tc.rawPrimaries(.bt709),
            .min_luma = 0,
            .max_luma = 1,
            .size_I = 9,
            .size_C = 7,
            .size_h = 12,
        },
        .data = undefined,
    };
    const p = &l.params;
    l.data = try std.testing.allocator.alloc(f32, p.lutLen());
    defer std.testing.allocator.free(l.data);
    for (0..p.size_h) |h| for (0..p.size_C) |c| for (0..p.size_I) |i| {
        const o = ((h * p.size_C + c) * p.size_I + i) * 4;
        const fi: f32 = @floatFromInt(i);
        const fc: f32 = @floatFromInt(c);
        const fh: f32 = @floatFromInt(h);
        l.data[o..][0..4].* = .{ 0.1 + 0.03 * fi - 0.01 * fc + 0.002 * fh, -0.2 + 0.05 * fc, 0.3 - 0.02 * fh + 0.01 * fi, 0 };
    };
    // positions >= 1 texel inside every axis
    var xi: [VL]f32 = undefined;
    var xc: [VL]f32 = undefined;
    var xh: [VL]f32 = undefined;
    for (0..VL) |k| {
        const t = (@as(f32, @floatFromInt(k)) + 0.37) / @as(f32, @floatFromInt(VL));
        xi[k] = (1 + t * 6) / 8;
        xc[k] = (1 + (1 - t) * 4) / 6;
        xh[k] = (1 + t * 9) / 11;
    }
    var lin: [3]V = undefined;
    var cub: [3]V = undefined;
    gamutLookup(&l, &lin[0], &lin[1], &lin[2], xi, xc, xh);
    gamutLookupCubic(&l, &cub[0], &cub[1], &cub[2], xi, xc, xh);
    for (0..3) |ch| {
        const a: [VL]f32 = lin[ch];
        const b: [VL]f32 = cub[ch];
        for (a, b) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-5);
    }
}

test "Tone.g matches the exact curve composition" {
    // a linear (PQ-domain) curve, and the clip curve's exact clamp
    var t: Tone = .{ .kind = .linear, .a = 1.3, .b = -0.05, .c = 0.55, .d = 0.02 };
    t.buildG();
    var ok_max: f32 = 0;
    var k: usize = 0;
    while (k < 4096) : (k += VL) {
        var xs: [VL]f32 = undefined;
        for (&xs, 0..) |*v, i| v.* = math.pow(f32, 2, -24.0 * @as(f32, @floatFromInt(k + i)) / 4095.0);
        const got: [VL]f32 = t.mapG(xs);
        const want: [VL]f32 = pqEotfMath(t.map(pqOetfMath(xs)));
        for (got, want) |g, w| ok_max = @max(ok_max, @abs(g - w) / @max(w, 1e-6));
    }
    // the curve's corners (clamp of a*x + b to [0, 1]) cost one segment
    try std.testing.expect(ok_max < 2e-3);
    var c: Tone = .{ .kind = .clip, .a = 0.2, .b = 0.7 };
    c.buildG();
    const xs: [VL]f32 = @splat(0.5);
    const got: [VL]f32 = c.mapG(xs);
    try std.testing.expectApproxEqRel(tc.hdrRescale(.pq, .nits, 0.7) / 10000, got[0], 1e-5);
}

test "detection usability and window come from the config" {
    const base: Config = .{ .width = 64, .height = 64, .rgb = true, .float = false };
    var c = base;
    try std.testing.expect(c.peakUsable());
    try std.testing.expectEqual(@as(u32, 100), c.peakWindow());
    c.peak.smoothing_period = Config.max_smoothing_period;
    try std.testing.expectEqual(@as(u32, Config.max_window), c.peakWindow());
    c.peak.smoothing_period = 0;
    try std.testing.expectEqual(@as(u32, 1), c.peakWindow());

    const Case = struct { cfg: Config, usable: bool };
    var cases = [_]Case{
        .{ .cfg = base, .usable = false },
        .{ .cfg = base, .usable = false },
        .{ .cfg = base, .usable = true },
        .{ .cfg = base, .usable = false },
        .{ .cfg = base, .usable = true },
        .{ .cfg = base, .usable = true },
        .{ .cfg = base, .usable = false },
        .{ .cfg = base, .usable = false },
    };
    cases[0].cfg.peak_detect = false;
    // the metadata modes that never use the measurement
    cases[1].cfg.metadata = .hdr10;
    cases[2].cfg.metadata = .cie_y;
    // a PQ target at 10000 nits: max_peak is never above it
    cases[3].cfg.dst_csp = .hdr10;
    cases[4].cfg.dst_csp = .hdr10;
    cases[4].cfg.dst_max = 1000;
    // HLG to SDR: the peak comes from per-frame metadata
    cases[5].cfg.src_csp = .hlg;
    // HLG to HDR: the source peak is the target's
    cases[6].cfg.src_csp = .hlg;
    cases[6].cfg.dst_csp = .hdr10;
    cases[6].cfg.dst_max = 1000;
    cases[7].cfg.metadata = .none;
    for (cases, 0..) |cs, i| {
        std.testing.expectEqual(cs.usable, cs.cfg.peakUsable()) catch |e| {
            std.debug.print("case {d}\n", .{i});
            return e;
        };
        if (!cs.usable) try std.testing.expectEqual(@as(u32, 0), cs.cfg.peakWindow());
        // never active on a frame when ruled out up front
        const metas = [_]FrameMeta{ .{}, .{ .mastering_max = 1000 }, .{ .mastering_max = 4000, .max_cll = 3000 } };
        for (&metas) |*m| {
            if (Setup.init(&cs.cfg, m).peak_active) try std.testing.expect(cs.cfg.peakUsable());
        }
    }
}

test "PeakCache allocates pages on demand" {
    var fa = std.testing.FailingAllocator.init(std.heap.c_allocator, .{});
    common.test_hooks.backing = fa.allocator();
    defer common.test_hooks.backing = std.heap.c_allocator;
    // std.Loop's length: a pointer per page up front, not an entry a frame
    const n: usize = math.maxInt(i32);
    var c = try PeakCache.init(n);
    const dir = fa.allocated_bytes;
    try std.testing.expect(dir <= @sizeOf(usize) * (n / PeakCache.page_len + 1));
    const frames = [_]usize{ 0, PeakCache.page_len - 1, PeakCache.page_len, n - 1 };
    for (frames) |i| {
        try std.testing.expect(!c.ready(i));
        try std.testing.expectEqual(@as(?PeakCache.Entry, null), try c.acquire(i));
        c.release(i);
        try std.testing.expectEqual(@as(?PeakCache.Entry, null), try c.acquire(i));
        const v: f32 = @floatFromInt(i);
        c.put(i, .{ .m = .{ .avg_pq = v, .max_pq = 0, .bias = 0, .scene_avg_pq = 0, .max_y_pq = 0 }, .active = true });
        try std.testing.expect(c.ready(i));
        try std.testing.expectEqual(v, (try c.acquire(i)).?.m.avg_pq);
    }
    try std.testing.expect(!c.ready(1));
    // frames 0, page_len - 1 share a page
    try std.testing.expectEqual(dir + 3 * @sizeOf(PeakCache.Page), fa.allocated_bytes);
    c.deinit();
    try std.testing.expectEqual(fa.allocated_bytes, fa.freed_bytes);
}

/// GamutCache test helpers. Entries are identified by their parameters,
/// never by address: a freed LUT's address can come back for the next one.
const GamutCacheTest = struct {
    fn params(k: usize) tc.GamutParams {
        var in = tc.rawPrimaries(.bt2020);
        in.red.x -= 0.001 * @as(f32, @floatFromInt(k));
        return .{ .function = .perceptual, .input_gamut = in, .output_gamut = tc.rawPrimaries(.bt709), .min_luma = 0, .max_luma = 0.58, .size_I = 4, .size_C = 4, .size_h = 8 };
    }

    fn cached(cache: *const GamutCache, k: usize) bool {
        const p = params(k);
        for (cache.list.items) |l| {
            if (GamutCache.sameParams(&l.params, &p)) return true;
        }
        return false;
    }

    /// Hands a freed block of the same size and alignment straight back
    /// (newest first), as a valid allocator may.
    const Recycler = struct {
        held: std.ArrayList(Block) = .empty,
        reused: usize = 0,

        const Block = struct { mem: []u8, a: std.mem.Alignment };
        const backing = std.heap.c_allocator;

        fn allocator(r: *Recycler) std.mem.Allocator {
            return .{ .ptr = r, .vtable = &.{
                .alloc = alloc,
                .resize = std.mem.Allocator.noResize,
                .remap = std.mem.Allocator.noRemap,
                .free = free,
            } };
        }

        fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
            const r: *Recycler = @ptrCast(@alignCast(ctx));
            var i = r.held.items.len;
            while (i > 0) {
                i -= 1;
                const b = r.held.items[i];
                if (b.mem.len == len and b.a == a) {
                    _ = r.held.orderedRemove(i);
                    r.reused += 1;
                    return b.mem.ptr;
                }
            }
            return backing.rawAlloc(len, a, ra);
        }

        fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
            const r: *Recycler = @ptrCast(@alignCast(ctx));
            r.held.append(backing, .{ .mem = m, .a = a }) catch backing.rawFree(m, a, ra);
        }

        fn deinit(r: *Recycler) void {
            for (r.held.items) |b| backing.rawFree(b.mem, b.a, @returnAddress());
            r.held.deinit(backing);
        }
    };

    /// Runs `body` on the default allocator, then on a Recycler (and checks
    /// that it reused a block, when the body frees and allocates again).
    fn run(comptime expect_reuse: bool, comptime body: fn () anyerror!void) !void {
        try body();
        var r: Recycler = .{};
        defer r.deinit();
        common.test_hooks.backing = r.allocator();
        defer common.test_hooks.backing = std.heap.c_allocator;
        try body();
        if (expect_reuse) try std.testing.expect(r.reused > 0);
    }
};

test "GamutCache bounds unused LUTs and keeps pinned ones" {
    // evicted LUTs' addresses come back for later ones under the Recycler
    try GamutCacheTest.run(true, struct {
        fn body() anyerror!void {
            const T = GamutCacheTest;
            var cache: GamutCache = .{};
            defer cache.deinit();
            const pinned = try cache.get(&T.params(0));
            var pins: usize = 1;
            // released even when an expectation fails first
            defer for (0..pins) |_| cache.release(pinned);

            cache.release(try cache.get(&T.params(1)));
            for (2..10) |k| cache.release(try cache.get(&T.params(k)));
            try std.testing.expectEqual(@as(usize, GamutCache.max_kept), cache.list.items.len);
            try std.testing.expect(T.cached(&cache, 0));
            try std.testing.expect(!T.cached(&cache, 1));
            // a hit pins and returns the cached LUT (still pinned, so its
            // address is its identity)
            const hit = try cache.get(&T.params(0));
            cache.release(hit);
            try std.testing.expectEqual(pinned, hit);
            pins = 0;
            cache.release(pinned);
            // the most recently used survives the next insertions
            cache.release(try cache.get(&T.params(9)));
            for (10..13) |k| cache.release(try cache.get(&T.params(k)));
            try std.testing.expectEqual(@as(usize, GamutCache.max_kept), cache.list.items.len);
            try std.testing.expect(T.cached(&cache, 9));
            try std.testing.expect(!T.cached(&cache, 0));
        }
    }.body);
}

test "GamutCache returns to its bound when a burst drains" {
    try GamutCacheTest.run(false, struct {
        fn body() anyerror!void {
            const T = GamutCacheTest;
            var cache: GamutCache = .{};
            defer cache.deinit();
            // 12 frames in flight with distinct gamuts: all pinned at once
            var held: [12]?*const Lut3d = @splat(null);
            defer for (held) |h| if (h) |l| cache.release(l);
            for (&held, 0..) |*h, k| h.* = try cache.get(&T.params(k));
            try std.testing.expectEqual(@as(usize, held.len), cache.list.items.len);

            // finishing in mixed order, with no new gamut, trims back to the
            // bound; 4 and 7 stay pinned
            for ([_]usize{ 5, 0, 11, 3, 8, 1, 10, 6, 2, 9 }, 1..) |k, done| {
                cache.release(held[k].?);
                held[k] = null;
                // the pinned ones, or the bound
                try std.testing.expectEqual(@max(GamutCache.max_kept, held.len - done), cache.list.items.len);
            }
            try std.testing.expectEqual(@as(usize, GamutCache.max_kept), cache.list.items.len);
            try std.testing.expect(T.cached(&cache, 4) and T.cached(&cache, 7));
            // hits keep it there
            for ([_]usize{ 2, 9, 4 }) |k| cache.release(try cache.get(&T.params(k)));
            try std.testing.expectEqual(@as(usize, GamutCache.max_kept), cache.list.items.len);
        }
    }.body);
}

test "fused color map matches the split passes" {
    // show_clipping's fused loop with nothing clipped must compute what the
    // split passes do, for both gamut lookups
    var cache: GamutCache = .{};
    defer cache.deinit();
    inline for (.{ false, true }) |cubic| {
        const cfg: Config = .{ .width = 512, .height = 1, .rgb = true, .float = true, .lut3d_size = .{ 17, 9, 32 }, .lut3d_tricubic = cubic };
        const s: Setup = .init(&cfg, &.{ .mastering_max = 1000 });
        var plan: Plan = undefined;
        try plan.init(&cfg, &s, null, &cache);
        defer plan.deinit(&cache);
        try std.testing.expect(plan.map == .full and plan.gamut != null and plan.need_tone);

        var a = try Rows.init(cfg.width);
        defer a.deinit();
        var b = try Rows.init(cfg.width);
        defer b.deinit();
        // linear BT.2020 light (1 = SDR white) up to ~5 (1000 nits), mostly
        // inside the gamut LUT's range
        var rng = std.Random.DefaultPrng.init(7);
        const r = rng.random();
        for (0..cfg.width) |x| {
            const base = 0.01 + 5 * r.float(f32) * r.float(f32);
            a.a[x] = base * (0.3 + 0.7 * r.float(f32));
            a.b[x] = base * (0.3 + 0.7 * r.float(f32));
            a.c[x] = base * (0.3 + 0.7 * r.float(f32));
        }
        @memcpy(b.buf, a.buf);
        colorMapSplit(&plan, &a, null, cfg.width);
        const inf = math.inf(f32);
        plan.clip = .{ .hi_rgb = inf, .lo_rgb = -inf, .hi_i = inf, .lo_i = -inf };
        colorMapFused(&plan, &b, null, cfg.width);
        plan.clip = null;
        try std.testing.expectEqualSlices(f32, a.a[0..cfg.width], b.a[0..cfg.width]);
        try std.testing.expectEqualSlices(f32, a.b[0..cfg.width], b.b[0..cfg.width]);
        try std.testing.expectEqualSlices(f32, a.c[0..cfg.width], b.c[0..cfg.width]);
    }
}

test "pq LUTs match the formulas" {
    // compared in the PQ domain, where the output quantizes, against f64
    var k: usize = 0;
    while (k < 4096) : (k += VL) {
        var xs: [VL]f32 = undefined;
        for (&xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(k + i)) / 4095.0;
        const e: [VL]f32 = pqEotf(xs);
        for (e, xs) |v, x| {
            const back = pqOetf64(v);
            const want = pqOetf64(pqEotf64(x));
            try std.testing.expectApproxEqAbs(want, back, @as(f64, if (x > 0.02) 1e-6 else 3e-5));
        }
        var ys: [VL]f32 = undefined;
        for (&ys, 0..) |*v, i| v.* = math.pow(f32, 2, -30.0 * @as(f32, @floatFromInt(k + i)) / 4095.0);
        const o: [VL]f32 = pqOetf(ys);
        for (o, ys) |v, y| try std.testing.expectApproxEqAbs(pqOetf64(y), v, 4e-6);
        // the formula path (used where no LUT applies) stays close too
        const om: [VL]f32 = pqOetfMath(ys);
        for (om, ys) |v, y| try std.testing.expectApproxEqAbs(pqOetf64(y), v, 2e-5);
        _ = pqEotfMath(xs);
    }
}
