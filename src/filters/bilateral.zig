const std = @import("std");
const math = std.math;

const hz = @import("../common.zig");
const simd = @import("simd.zig");

/// Kernel configuration, derived by `Data.init` from the parsed arguments
/// (shared by the VapourSynth and AviSynth adapters).
pub const Data = struct {
    sigmaS: [3]f64 = .{ 0, 0, 0 },
    sigmaR: [3]f64 = .{ 0, 0, 0 },
    planes: [3]bool = .{ true, true, true },
    algorithm: [3]i32 = .{ 0, 0, 0 },
    PBFICnum: [3]u32 = .{ 0, 0, 0 },
    radius: [3]u32 = .{ 0, 0, 0 },
    samples: [3]u32 = .{ 0, 0, 0 },
    step: [3]u32 = .{ 0, 0, 0 },
    gr_lut: [3][]f32 = .{ &.{}, &.{}, &.{} },
    gs_lut: [3][]f32 = .{ &.{}, &.{}, &.{} },
    psize: u6 = 0,
    peak: f32 = 0,

    pub const max_sigma_s: f64 = 8192;

    /// Parsed arguments. `sigmaS` is already resolved per plane (see
    /// `sigmaSDefault`); `planes` already reflects the `planes` argument.
    pub const Args = struct {
        sigmaS: [3]f64,
        sigmaR: [3]f64,
        algorithm: [3]i32,
        PBFICnum: [3]u32,
        planes: [3]bool,
        yuv: bool,
        /// Plane dimensions of planes 0..2.
        plane_w: [3]u32,
        plane_h: [3]u32,
        /// 2^bits for integer formats, 65536 for float.
        hist_len: u32,
    };

    pub const Error = error{ PBFICnum, PlaneTooSmall, OutOfMemory };

    /// Message for an `init` error (callers add their filter-name prefix).
    pub fn message(e: Error) [:0]const u8 {
        return switch (e) {
            error.PBFICnum => "Invalid \"PBFICnum\" assigned, must be integer ranges in [0,256] except 1",
            error.PlaneTooSmall => "plane too small for the spatial radius derived from sigmaS; lower sigmaS or use a larger clip.",
            error.OutOfMemory => "out of memory",
        };
    }

    /// vszip's default for an unset sigmaS entry `i`: 3 for luma, scaled by the
    /// subsampling for 4:2:x chroma, else the previous plane's value.
    pub fn sigmaSDefault(prev: []const f64, i: usize, yuv: bool, ssw: u5, ssh: u5) f64 {
        if (i == 0) return 3;
        if ((i == 1) and yuv and (ssh != 0) and (ssw != 0)) {
            const factor: f64 = @floatFromInt((@as(u32, 1) << ssh) * (@as(u32, 1) << ssw));
            return prev[0] / @sqrt(factor);
        }
        return prev[i - 1];
    }

    /// Derive PBFIC counts, radii and algorithms and build the LUTs.
    pub fn init(a: Args) Error!Data {
        var d: Data = .{ .sigmaS = a.sigmaS, .sigmaR = a.sigmaR, .algorithm = a.algorithm, .PBFICnum = a.PBFICnum, .planes = a.planes };
        d.peak = @floatFromInt(a.hist_len - 1);

        for (0..3) |i| {
            if ((d.sigmaS[i] == 0) or (d.sigmaR[i] == 0)) d.planes[i] = false;
        }
        for (d.PBFICnum) |num| {
            if (num == 1) return error.PBFICnum;
        }

        for (0..3) |i| {
            if ((d.planes[i]) and (d.PBFICnum[i] == 0)) {
                if (d.sigmaR[i] >= 0.08) {
                    d.PBFICnum[i] = 4;
                } else if (d.sigmaR[i] >= 0.015) {
                    d.PBFICnum[i] = @intFromFloat(@min(16.0, @trunc(4 * 0.08 / d.sigmaR[i] + 0.5)));
                } else {
                    // clamp in float: a tiny positive sigmaR makes the quotient
                    // exceed u32 (same result as min-after-trunc for every value
                    // that fits)
                    d.PBFICnum[i] = @intFromFloat(@min(32.0, @trunc(16 * 0.015 / d.sigmaR[i] + 0.5)));
                }

                if ((i > 0) and a.yuv and (d.PBFICnum[i] % 2 == 0) and (d.PBFICnum[i] < 256)) {
                    d.PBFICnum[i] += 1;
                }
            }
        }

        var orad = [_]i32{ 0, 0, 0 };
        for (0..3) |i| {
            if (!d.planes[i]) continue;
            orad[i] = @max(@as(i32, @trunc(d.sigmaS[i] * 2 + 0.5)), 1);
            if (orad[i] < 4) {
                d.step[i] = 1;
            } else if (orad[i] < 8) {
                d.step[i] = 2;
            } else {
                d.step[i] = 3;
            }

            d.samples[i] = 1;
            d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];

            while (orad[i] * 2 > d.radius[i] * 3) {
                d.samples[i] += 1;
                d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];
                if ((d.radius[i] >= orad[i]) and (d.samples[i] > 2)) {
                    d.samples[i] -= 1;
                    d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];
                    break;
                }
            }
        }

        for (0..3) |i| {
            if ((d.planes[i]) and (d.algorithm[i] <= 0)) {
                d.algorithm[i] = if (d.step[i] == 1) 2 else (if ((d.sigmaR[i] < 0.08) and (d.samples[i] < 5)) 2 else (if (4 * d.samples[i] * d.samples[i] <= 15 * d.PBFICnum[i]) 2 else 1));
            }
        }

        for (0..3) |i| {
            if ((d.planes[i]) and (d.algorithm[i] == 2)) {
                if ((a.plane_w[i] <= 2 * d.radius[i]) or (a.plane_h[i] <= 2 * d.radius[i])) return error.PlaneTooSmall;
            }
        }

        errdefer d.deinit();
        for (0..3) |i| {
            if ((d.planes[i]) and (d.algorithm[i] == 2)) {
                const upper: u32 = d.radius[i] + 1;
                d.gs_lut[i] = try allocator.alloc(f32, upper * upper);
                gaussianFunctionSpatialLUTGeneration(d.gs_lut[i], upper, d.sigmaS[i]);
            }
        }
        for (0..3) |i| {
            if (d.planes[i]) {
                d.gr_lut[i] = try allocator.alloc(f32, a.hist_len);
                gaussianFunctionRangeLUTGeneration(d.gr_lut[i], d.peak, d.sigmaR[i]);
            }
        }
        return d;
    }

    /// Free the LUTs (unallocated slots are empty slices).
    pub fn deinit(d: *Data) void {
        for (&d.gr_lut) |*l| {
            allocator.free(l.*);
            l.* = &.{};
        }
        for (&d.gs_lut) |*l| {
            allocator.free(l.*);
            l.* = &.{};
        }
    }
};
const vszip = @import("../common.zig");

const allocator = hz.kernel_allocator;
const AlignedF32 = []align(vszip.vec_len) f32;

const VLEN = @max(vszip.vec_len / @sizeOf(f32), 1);
const Vf = @Vector(VLEN, f32);
const Vu = @Vector(VLEN, u32);

inline fn rangeIndex(comptime T: type, a: T, b: T) u32 {
    if (@typeInfo(T) == .float) {
        const ad: f32 = @abs(a - b);
        // Convert via i32: the value is provably in [0, 65535], so the signed
        // convert is exact and a single vcvttps2dq — a u32 destination costs
        // LLVM's 6-instruction unsigned-convert fixup. @intFromFloat already
        // truncates toward zero, so no separate @trunc is needed.
        const idx: i32 = @intFromFloat(@min(1.0, ad) * 65535 + 0.5);
        return @intCast(idx);
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
        // Same i32 detour as scalar rangeIndex: value in [0, 65535], one
        // vcvttps2dq per gather instead of the 6-instruction u32 fixup whose
        // two extra live constants caused the f32 kernel's accumulator spills.
        const idx_i: @Vector(VLEN, i32) = @intFromFloat(@min(@as(Vf, @splat(1.0)), ad) * @as(Vf, @splat(65535.0)) + @as(Vf, @splat(0.5)));
        idx = @intCast(idx_i);
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

pub fn bilateral(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, w: u32, h: u32, plane: u32, comptime join: bool, d: *Data) error{OutOfMemory}!void {
    _ = join; // refp already aliases srcp when there is no joint clip
    if (d.algorithm[plane] == 1) {
        try pbfic(T, srcp, refp, dstp, stride, w, h, plane, d);
    } else {
        truncated(T, srcp, refp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane], d.peak);
    }
}

// O(1) cross/joint bilateral (PBFIC) — "Real-Time O(1) Bilateral Filtering", Yang et al.
//
// Streams one PBFIC level at a time instead of keeping all `num` levels as
// full planes (16-17 planes, ~140 MB per 1080p luma call at the default
// sigmaR, which made alg1 bandwidth- and page-fault-bound at 16T). Each pixel
// only interpolates between levels k_i and k_i+1, so level k's result is
// folded into `acc` (lower term) or finished into dst (upper term) as soon as
// its row is final. Per level: one forward sweep in 8-row groups (build Wk/Jk,
// horizontal recursion, vertical forward recursion, all cache-resident) and
// one backward sweep (vertical backward recursion in a 4-row ring, fused with
// the divide and the interpolation). Every element sees the same operations
// in the same order whichever path (vector, scalar tail, 8-row or single-row
// horizontal) computes it.
fn pbfic(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, width: u32, height: u32, plane: u32, d: *Data) error{OutOfMemory}!void {
    const is_float = @typeInfo(T) == .float;
    const num = d.PBFICnum[plane];
    const pcount: u32 = stride * height;
    const gr_lut = d.gr_lut[plane];
    std.debug.assert(num >= 2 and num <= 256);

    const pbfick = try allocator.alloc(T, num);
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

    var c: Coefs = undefined;
    recursiveGaussianParameters(d.sigmaS[plane], &c.b, &c.b1, &c.b2, &c.b3);

    const wk: AlignedF32 = try allocator.alignedAlloc(f32, vszip.alignment, pcount);
    defer allocator.free(wk);
    const jk: AlignedF32 = try allocator.alignedAlloc(f32, vszip.alignment, pcount);
    defer allocator.free(jk);
    const acc: AlignedF32 = try allocator.alignedAlloc(f32, vszip.alignment, pcount);
    defer allocator.free(acc);
    // backward-recursion ring: 4 rows of Wk, then 4 rows of Jk
    const ring: AlignedF32 = try allocator.alignedAlloc(f32, vszip.alignment, 8 * @as(usize, stride));
    defer allocator.free(ring);
    const kidx = try allocator.alloc(u8, pcount);
    defer allocator.free(kidx);

    pbficIntervals(T, refp, kidx, pbfick, stride, width, height);

    for (0..num) |k| {
        const pk = pbfick[k];

        var j0: u32 = 0;
        while (j0 < height) : (j0 += 8) {
            const rows = @min(8, height - j0);
            for (j0..j0 + rows) |j| {
                // Wk = range weight at this PBFIC level; Jk = Wk*src.
                const lo = stride * j;
                var i = lo;
                const pkv: Vec(T) = @splat(pk);
                while (i + VLEN <= lo + width) : (i += VLEN) {
                    const wv = vRangeGather(T, gr_lut, pkv, refp[i..][0..VLEN].*);
                    wk[i..][0..VLEN].* = wv;
                    jk[i..][0..VLEN].* = wv * vToF(T, @as(Vec(T), srcp[i..][0..VLEN].*));
                }
                while (i < lo + width) : (i += 1) {
                    wk[i] = gr_lut[rangeIndex(T, pk, refp[i])];
                    jk[i] = wk[i] * valOf(T, srcp[i]);
                }
            }

            if (rows == 8) {
                horizontal8(wk, jk, stride * j0, stride, width, c);
            } else {
                for (j0..j0 + rows) |j| horizontalRowPair(wk, jk, stride * j, width, c);
            }

            for (j0..j0 + rows) |j| verticalForwardRow(wk, jk, @intCast(j), stride, width, c);
        }

        var j: u32 = height;
        while (j > 0) {
            j -= 1;
            pbficBackwardRow(T, wk, jk, ring, acc, kidx, refp, dstp, pbfick, @intCast(k), j, stride, width, height, c, d.peak);
        }
    }
}

const Coefs = struct { b: f32, b1: f32, b2: f32, b3: f32 };

/// One recursive-Gaussian tap: b*x + b1*y1 + b2*y2 + b3*y3, where y1 is the
/// previous output along the recursion. Associated so that only the last FMA
/// depends on y1: the horizontal recursion is a loop-carried chain, and this
/// cuts its per-step latency from mul+3 adds to one FMA.
inline fn rec(comptime V: type, x: V, y1: V, y2: V, y3: V, c: Coefs) V {
    const sp = struct {
        inline fn f(v: f32) V {
            return if (V == f32) v else @splat(v);
        }
    }.f;
    return simd.madd(V, sp(c.b1), y1, simd.madd(V, sp(c.b2), y2, simd.madd(V, sp(c.b3), y3, sp(c.b) * x)));
}

/// k_i per pixel: the first k in [0, num-2) with pbfick[k] <= r < pbfick[k+1],
/// else num-2 (also for r below pbfick[0], or NaN). pbfick is strictly
/// increasing, so that is min(#{j >= 1 : pbfick[j] <= r}, num-2) when
/// r >= pbfick[0]. Compared in f32, which is exact for every T.
fn pbficIntervals(comptime T: type, refp: []const T, kidx: []u8, pbfick: []const T, stride: u32, width: u32, height: u32) void {
    const num: u32 = @intCast(pbfick.len);
    const last: Vu = @splat(num - 2);
    const p0: Vf = @splat(valOf(T, pbfick[0]));
    for (0..height) |j| {
        const lo = stride * j;
        var i = lo;
        while (i + VLEN <= lo + width) : (i += VLEN) {
            const r = vToF(T, @as(Vec(T), refp[i..][0..VLEN].*));
            var cnt: Vu = @splat(0);
            for (pbfick[1..]) |p| cnt += @intFromBool(r >= @as(Vf, @splat(valOf(T, p))));
            const kv = @select(u32, r >= p0, @min(cnt, last), last);
            const kb: @Vector(VLEN, u8) = @intCast(kv);
            kidx[i..][0..VLEN].* = kb;
        }
        while (i < lo + width) : (i += 1) {
            var k: u32 = 0;
            while (k < (num - 2)) : (k += 1) {
                if ((refp[i] < pbfick[k + 1]) and (refp[i] >= pbfick[k])) break;
            }
            kidx[i] = @intCast(k);
        }
    }
}

/// Row `cur + off` (clamped to the last row) as seen by the in-place backward
/// recursion: rows past the bottom alias the current row, whose value at that
/// point is still the forward result; later rows are already in the ring.
inline fn backwardSrc(ring: []const f32, fwd: []const f32, stride: u32, height: u32, cur: u32, comptime off: u32, half: u32) []const f32 {
    const row = @min(cur + off, height - 1);
    if (row == cur) return fwd[@as(usize, stride) * cur ..];
    return ring[(half * 4 + row % 4) * @as(usize, stride) ..];
}

/// Row j of the backward vertical recursion for Wk and Jk, then
/// PBFIC_k = Jk/Wk (0 where Wk == 0) and its interpolation term: the lower
/// term goes to `acc` for pixels with k_i == k, and pixels with k_i == k-1
/// are finished into dst.
fn pbficBackwardRow(comptime T: type, wk: []const f32, jk: []const f32, ring: []f32, acc: []f32, kidx: []const u8, refp: []const T, dstp: []T, pbfick: []const T, k: u32, j: u32, stride: u32, width: u32, height: u32, c: Coefs, peak: f32) void {
    const num: u32 = @intCast(pbfick.len);
    const lower: usize = @as(usize, stride) * j;
    const w0 = wk[lower..];
    const q0 = jk[lower..];
    const w1 = backwardSrc(ring, wk, stride, height, j, 1, 0);
    const w2 = backwardSrc(ring, wk, stride, height, j, 2, 0);
    const w3 = backwardSrc(ring, wk, stride, height, j, 3, 0);
    const q1 = backwardSrc(ring, jk, stride, height, j, 1, 1);
    const q2 = backwardSrc(ring, jk, stride, height, j, 2, 1);
    const q3 = backwardSrc(ring, jk, stride, height, j, 3, 1);
    const wout = ring[(j % 4) * @as(usize, stride) ..];
    const qout = ring[(4 + j % 4) * @as(usize, stride) ..];
    const kr = kidx[lower..];
    const rr = refp[lower..];
    const ar = acc[lower..];
    const dr = dstp[lower..];

    const has_lo = k + 1 < num; // k_i == k needs pbfick[k + 1]
    const has_hi = k > 0; //       k_i == k-1 needs pbfick[k - 1]
    const lo_p1: f32 = if (has_lo) valOf(T, pbfick[k + 1]) else 0;
    const hi_p0: f32 = if (has_hi) valOf(T, pbfick[k - 1]) else 0;
    const hi_p1: f32 = valOf(T, pbfick[k]);
    const k_lo: u8 = @truncate(k);
    const k_hi: u8 = @truncate(k -% 1);

    const zero: Vf = @splat(0.0);
    const peakv: Vf = @splat(peak);
    const Vk = @Vector(VLEN, u8);

    var x: usize = 0;
    while (x + VLEN <= width) : (x += VLEN) {
        const wv = rec(Vf, w0[x..][0..VLEN].*, w1[x..][0..VLEN].*, w2[x..][0..VLEN].*, w3[x..][0..VLEN].*, c);
        const qv = rec(Vf, q0[x..][0..VLEN].*, q1[x..][0..VLEN].*, q2[x..][0..VLEN].*, q3[x..][0..VLEN].*, c);
        wout[x..][0..VLEN].* = wv;
        qout[x..][0..VLEN].* = qv;
        // Only lanes with k_i in {k-1, k} use this level. Real content is
        // smooth, so most vectors have none: skipping them saves the divide
        // and the ref/acc/dst traffic (alg1 is bandwidth-bound at 16T).
        // k_hi wraps to 255 at k == 0, where only k_i == 0 passes.
        const ki: Vk = kr[x..][0..VLEN].*;
        if (!@reduce(.Or, ki -% @as(Vk, @splat(k_hi)) < @as(Vk, @splat(2)))) continue;
        // Divide all lanes, then blend the wk==0 lanes to 0: zero lanes only
        // produce inf/NaN that the select discards (== the scalar tail).
        const p = @select(f32, wv == zero, zero, qv / wv);
        const rf = vToF(T, @as(Vec(T), rr[x..][0..VLEN].*));
        const av: Vf = ar[x..][0..VLEN].*;
        if (has_hi) {
            const m = ki == @as(Vk, @splat(k_hi));
            if (@reduce(.Or, m)) {
                const vf = (av + (rf - @as(Vf, @splat(hi_p0))) * p) / @as(Vf, @splat(hi_p1 - hi_p0));
                const old: Vec(T) = dr[x..][0..VLEN].*;
                dr[x..][0..VLEN].* = @select(T, m, vFinalize(T, vf, @as(Vf, @splat(1.0)), peakv), old);
            }
        }
        if (has_lo) {
            const m = ki == @as(Vk, @splat(k_lo));
            if (@reduce(.Or, m)) ar[x..][0..VLEN].* = @select(f32, m, (@as(Vf, @splat(lo_p1)) - rf) * p, av);
        }
    }
    while (x < width) : (x += 1) {
        const wv = rec(f32, w0[x], w1[x], w2[x], w3[x], c);
        const qv = rec(f32, q0[x], q1[x], q2[x], q3[x], c);
        wout[x] = wv;
        qout[x] = qv;
        const p: f32 = if (wv == 0) 0 else (qv / wv);
        const rf: f32 = valOf(T, rr[x]);
        if (has_hi and kr[x] == k_hi) {
            const vf: f32 = (ar[x] + (rf - hi_p0) * p) / (hi_p1 - hi_p0);
            dr[x] = finalize(T, vf, @as(f32, 1.0), peak);
        } else if (has_lo and kr[x] == k_lo) {
            ar[x] = (lo_p1 - rf) * p;
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

// Forward vertical recursion of row j for both buffers, in place. The
// y-recurrence is per COLUMN, so the x loop runs VLEN columns at a time (lanes
// = columns, same per-lane ops as the scalar tail). All row
// loads happen before the store, so rows clamped onto x0's own row at the top
// read the pre-store value, as in scalar.
fn verticalForwardRow(a: []f32, q: []f32, j: u32, stride: u32, width: u32, c: Coefs) void {
    const lower: usize = @as(usize, stride) * j;
    const upper: usize = lower + width;
    var x0: usize = lower;
    var x1: usize = if (j < 1) x0 else (x0 - stride);
    var x2: usize = if (j < 2) x1 else (x1 - stride);
    var x3: usize = if (j < 3) x2 else (x2 - stride);

    while (x0 + VLEN <= upper) : ({
        x0 += VLEN;
        x1 += VLEN;
        x2 += VLEN;
        x3 += VLEN;
    }) {
        inline for (.{ a, q }) |buf| {
            buf[x0..][0..VLEN].* = rec(Vf, buf[x0..][0..VLEN].*, buf[x1..][0..VLEN].*, buf[x2..][0..VLEN].*, buf[x3..][0..VLEN].*, c);
        }
    }
    while (x0 < upper) : ({
        x0 += 1;
        x1 += 1;
        x2 += 1;
        x3 += 1;
    }) {
        inline for (.{ a, q }) |buf| {
            buf[x0] = rec(f32, buf[x0], buf[x1], buf[x2], buf[x3], c);
        }
    }
}

const V8 = @Vector(8, f32);

/// One step of the recursion: returns y[n] and shifts the state.
inline fn recStep(x: V8, s: *[3]V8, c: Coefs) V8 {
    const y = rec(V8, x, s[0], s[1], s[2], c);
    s[2] = s[1];
    s[1] = s[0];
    s[0] = y;
    return y;
}

inline fn loadCol(buf: []const f32, base: usize, stride: u32, x: usize) V8 {
    var v: V8 = undefined;
    inline for (0..8) |r| v[r] = buf[base + r * stride + x];
    return v;
}

inline fn storeCol(buf: []f32, base: usize, stride: u32, x: usize, v: V8) void {
    inline for (0..8) |r| buf[base + r * stride + x] = v[r];
}

/// One 8x8 tile of the horizontal recursion: rows in, transposed to columns,
/// 8 steps (ascending or descending x), transposed back.
fn horizontalTile(comptime ascending: bool, buf: []f32, base: usize, stride: u32, x: usize, st: *[3]V8, c: Coefs) void {
    @setEvalBranchQuota(100_000); // two inlined asm transposes (comptimePrint per instruction)
    var rows: [8]V8 = undefined;
    inline for (0..8) |r| rows[r] = buf[base + r * stride + x ..][0..8].*;
    var cols = simd.transposeF32Reg(8, rows);
    inline for (0..8) |s| {
        const i = if (ascending) s else 7 - s;
        cols[i] = recStep(cols[i], st, c);
    }
    rows = simd.transposeF32Reg(8, cols);
    inline for (0..8) |r| buf[base + r * stride + x ..][0..8].* = rows[r];
}

/// In-place horizontal recursion (causal, then anti-causal) of 8 rows of both
/// buffers, lanes = rows. The x recurrence is loop-carried (one
/// FMA per step, see `rec`), so the per-row scalar loop is latency-bound;
/// here each step advances 8 rows, and 8x8 in-register transposes turn row
/// loads into column vectors. Same per-lane ops as `horizontalRowPair`.
fn horizontal8(a: []f32, q: []f32, base: usize, stride: u32, width: u32, c: Coefs) void {
    var sa: [3]V8 = @splat(loadCol(a, base, stride, 0));
    var sq: [3]V8 = @splat(loadCol(q, base, stride, 0));
    var x: usize = 1;
    while (x + 8 <= width) : (x += 8) {
        horizontalTile(true, a, base, stride, x, &sa, c);
        horizontalTile(true, q, base, stride, x, &sq, c);
    }
    while (x < width) : (x += 1) {
        storeCol(a, base, stride, x, recStep(loadCol(a, base, stride, x), &sa, c));
        storeCol(q, base, stride, x, recStep(loadCol(q, base, stride, x), &sq, c));
    }

    if (width < 2) return;
    sa = @splat(loadCol(a, base, stride, width - 1));
    sq = @splat(loadCol(q, base, stride, width - 1));
    var e: usize = width - 1; // columns [0, e) remain, processed from e-1 down
    while (e >= 8) : (e -= 8) {
        horizontalTile(false, a, base, stride, e - 8, &sa, c);
        horizontalTile(false, q, base, stride, e - 8, &sq, c);
    }
    while (e > 0) {
        e -= 1;
        storeCol(a, base, stride, e, recStep(loadCol(a, base, stride, e), &sa, c));
        storeCol(q, base, stride, e, recStep(loadCol(q, base, stride, e), &sq, c));
    }
}

// Scalar in-place horizontal recursion of one row of both buffers: the
// semantic reference for `horizontal8`, used for the last < 8 rows. Carrying
// both independent chains in one loop lets them overlap in the OoO core.
fn horizontalRowPair(buf_a: []f32, buf_c: []f32, lower: usize, width: u32, c: Coefs) void {
    const upper: usize = lower + width;

    var i: usize = lower;
    var a1: f32 = buf_a[i];
    var a2: f32 = a1;
    var a3: f32 = a2;
    var c1: f32 = buf_c[i];
    var c2: f32 = c1;
    var c3: f32 = c2;
    i += 1;

    while (i < upper) : (i += 1) {
        const a0 = rec(f32, buf_a[i], a1, a2, a3, c);
        const c0 = rec(f32, buf_c[i], c1, c2, c3, c);
        a3 = a2;
        a2 = a1;
        a1 = a0;
        buf_a[i] = a0;
        c3 = c2;
        c2 = c1;
        c1 = c0;
        buf_c[i] = c0;
    }

    i -= 1;
    a1 = buf_a[i];
    a2 = a1;
    a3 = a2;
    c1 = buf_c[i];
    c2 = c1;
    c3 = c2;
    if (i == lower) return;
    i -= 1;
    while (true) : (i -= 1) {
        const a0 = rec(f32, buf_a[i], a1, a2, a3, c);
        const c0 = rec(f32, buf_c[i], c1, c2, c3, c);
        a3 = a2;
        a2 = a1;
        a1 = a0;
        buf_a[i] = a0;
        c3 = c2;
        c2 = c1;
        c1 = c0;
        buf_c[i] = c0;
        if (i == lower) break;
    }
}
