const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const Data = @import("../vapoursynth/planeminmax.zig").Data;
const simd = @import("simd.zig");
const vszip = @import("../vszip.zig");
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

/// Zero the histogram with explicit vector stores: @memset lowers to a call
/// into Zig's compiler_rt.memset (a scalar word loop, ~1.4 Ir/byte — 360K Ir
/// per 256 KB plane call, measured), while this loop stays plain vector
/// stores. len is always a multiple of any suggested vector width.
fn zeroHist(buf: []align(vszip.vec_len) u32) void {
    const n_opt = comptime std.simd.suggestVectorLength(u32);
    if (comptime n_opt == null) {
        @memset(buf, 0);
        return;
    }
    const n = comptime n_opt.?;
    std.debug.assert(buf.len % n == 0);
    var i: usize = 0;
    while (i < buf.len) : (i += n) {
        // opaquePtr: with a comptime trip count LLVM loop-idiom-recognizes a
        // plain splat-store loop back into the very compiler_rt.memset call
        // this helper exists to avoid; laundering the store address blocks
        // the transform (portable twin: the same loop without the launder).
        simd.opaquePtr(u32, buf.ptr + i)[0..n].* = @as(@Vector(n, u32), @splat(0));
    }
}

/// Vector twin of the scalar float histogram index
/// `lossyCast(u16, v * 65535.0 + 0.5)` (whose branch diamond costs ~14.5
/// scalar Ir/px). Bit-exact per lane:
/// - mul and add stay separate ops (Zig does not contract to FMA), identical
///   to the scalar vmulss/vaddss;
/// - @max(t, 0) is llvm.maxnum, so NaN lanes become 0.0, matching lossyCast's
///   NaN => 0; t <= 0 => 0 and t >= 65535 => 65535 match its saturation;
/// - remaining lanes lie in (0, 65535), where vcvttps2dq truncation ==
///   @intFromFloat (no lane can exceed i32 range after the clamp);
/// - f16 lanes widen through f32 exactly (f32 is a superset of f16).
inline fn floatIdxVec(comptime T: type, comptime n: comptime_int, p: *const [n]T) @Vector(n, u32) {
    const raw: @Vector(n, T) = p.*;
    const v: @Vector(n, f32) = if (T == f16) @floatCast(raw) else raw;
    const t = v * @as(@Vector(n, f32), @splat(65535.0)) + @as(@Vector(n, f32), @splat(0.5));
    const c = @min(@max(t, @as(@Vector(n, f32), @splat(0.0))), @as(@Vector(n, f32), @splat(65535.0)));
    return @intCast(@as(@Vector(n, i32), @intFromFloat(c)));
}

inline fn minMaxImpl(comptime T: type, comptime use_ref: bool, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    const is_int = @typeInfo(T) == .int;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    // u8 samples can never index past 255, so its histogram shrinks 256x.
    // The other formats keep the full 65536: a u16 clip may carry samples
    // above 2^bits-1 (only 0..d.hist_size is ever scanned, but the scatter
    // must stay in bounds) and the float index saturates at 65535.
    const hist_len: usize = if (T == u8) 256 else 65536;
    const accum_buf: []align(vszip.vec_len) u32 = allocator.alignedAlloc(u32, vszip.alignment, hist_len) catch unreachable;
    defer allocator.free(accum_buf);

    zeroHist(accum_buf);

    if (is_int) {
        var srcp: []const T = src;
        for (0..h) |_| {
            for (srcp[0..w]) |v| {
                accum_buf[v] += 1;
            }
            srcp = srcp[stride..];
        }
        if (use_ref) {
            // Diff as a separate vector sweep instead of a per-pixel f64
            // absDiff fused into the histogram loop (a serial vaddsd chain,
            // ~9.5 Ir/px); bit-identical per absDiffSumInt's exactness proof.
            diffacc = @floatFromInt(absDiffSumInt(T, src, ref, stride, w, h));
        }
    } else {
        const n = comptime (std.simd.suggestVectorLength(f32) orelse 1);
        const wv = if (comptime n > 1) w - w % n else 0;
        var srcp: []const T = src;
        var refp: []const T = ref;
        for (0..h) |_| {
            var x: usize = 0;
            while (x < wv) : (x += n) {
                const idx = floatIdxVec(T, n, srcp[x..][0..n]);
                inline for (0..n) |l| {
                    accum_buf[idx[l]] += 1;
                    // The Diff prop keeps the original per-pixel left-to-
                    // right scalar f64 chain (and f16 subtraction stays in
                    // f16) — only the index math is vectorized.
                    if (use_ref) diffacc += @abs(srcp[x + l] - refp[x + l]);
                }
            }
            while (x < w) : (x += 1) {
                accum_buf[math.lossyCast(u16, @as(f32, srcp[x]) * 65535.0 + 0.5)] += 1;
                if (use_ref) diffacc += @abs(srcp[x] - refp[x]);
            }
            srcp = srcp[stride..];
            if (use_ref) refp = refp[stride..];
        }
    }

    const totalmin: u32 = @trunc(total * d.minthr);
    const totalmax: u32 = @trunc(total * d.maxthr);
    var count: u32 = 0;

    var u: u32 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break @intCast(u);
    } else d.peak;

    count = 0;
    var i: i32 = d.peak;
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    if (is_int) {
        props.setInt(d.prop.mi, retvalmin, .Append);
        props.setInt(d.prop.ma, retvalmax, .Append);
    } else {
        props.setFloat(d.prop.mi, @as(f32, @floatFromInt(retvalmin)) / 65535, .Append);
        props.setFloat(d.prop.ma, @as(f32, @floatFromInt(retvalmax)) / 65535, .Append);
    }

    if (use_ref) {
        const diff: f64 = if (is_int) diffacc / total / d.peakf else diffacc / total;
        props.setFloat(d.prop.d, diff, .Append);
    }
}

pub fn minMax(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    minMaxImpl(T, false, src, &.{}, stride, props, w, h, d);
}

pub fn minMaxRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    minMaxImpl(T, true, src, ref, stride, props, w, h, d);
}

/// Accumulator element type: f16 lanes are widened to f32 — exact (f32 is a
/// superset of f16) and avoids the per-element compiler_rt fminf/fmaxf
/// libcalls that scalar/vector f16 @min/@max lower to pre-AVX512-FP16
/// (measured: ~43 Ir/px incl. two calls).
fn AccT(comptime T: type) type {
    return if (T == f16) f32 else T;
}

const MinMax = struct { min: f64, max: f64 };

/// Whole-plane min/max with @Vector accumulators + one @reduce at the end.
/// min/max are selection operations — order-free, so this is bit-exact vs the
/// scalar element loop (which LLVM never auto-vectorizes here: measured 5.8
/// scalar Ir/px for u8, 7.8 for f32). Float lanes keep @min/@max
/// (llvm.minnum/maxnum) NaN semantics; @reduce(.Min/.Max) is NaN-ignoring the
/// same way.
fn minMaxPlane(comptime T: type, src: []const T, stride: u32, w: u32, h: u32) MinMax {
    const V = AccT(T);
    const init_min: V = if (@typeInfo(V) == .int) math.maxInt(V) else math.inf(V);
    const init_max: V = if (@typeInfo(V) == .int) 0 else -math.inf(V);

    const n_opt = comptime std.simd.suggestVectorLength(V);
    if (comptime n_opt == null) {
        var srcp: []const T = src;
        var min: V = init_min;
        var max: V = init_max;
        for (0..h) |_| {
            for (srcp[0..w]) |v| {
                min = @min(min, v);
                max = @max(max, v);
            }
            srcp = srcp[stride..];
        }
        return if (@typeInfo(V) == .int)
            .{ .min = @floatFromInt(min), .max = @floatFromInt(max) }
        else
            .{ .min = min, .max = max };
    }

    const n = comptime n_opt.?;
    var min_v: @Vector(n, V) = @splat(init_min);
    var max_v: @Vector(n, V) = @splat(init_max);
    var smin: V = init_min;
    var smax: V = init_max;
    const wv = w - w % n;
    var srcp: []const T = src;
    for (0..h) |_| {
        var x: usize = 0;
        while (x < wv) : (x += n) {
            const raw: @Vector(n, T) = srcp[x..][0..n].*;
            const v: @Vector(n, V) = if (T == f16) @floatCast(raw) else raw;
            min_v = @min(min_v, v);
            max_v = @max(max_v, v);
        }
        while (x < w) : (x += 1) {
            smin = @min(smin, srcp[x]);
            smax = @max(smax, srcp[x]);
        }
        srcp = srcp[stride..];
    }
    const rmin = @min(smin, @reduce(.Min, min_v));
    const rmax = @max(smax, @reduce(.Max, max_v));
    return if (@typeInfo(V) == .int)
        .{ .min = @floatFromInt(rmin), .max = @floatFromInt(rmax) }
    else
        .{ .min = rmin, .max = rmax };
}

fn setMinMaxProps(comptime T: type, props: *const ZAPI.ZMap(*vs.Map), d: *Data, mm: MinMax) void {
    if (@typeInfo(T) == .int) {
        props.setInt(d.prop.mi, @intFromFloat(mm.min), .Append);
        props.setInt(d.prop.ma, @intFromFloat(mm.max), .Append);
    } else {
        props.setFloat(d.prop.mi, mm.min, .Append);
        props.setFloat(d.prop.ma, mm.max, .Append);
    }
}

pub fn minMaxNoThr(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    setMinMaxProps(T, props, d, minMaxPlane(T, src, stride, w, h));
}

/// Integer |src-ref| plane sum. Exact: per-row u32 lane accumulators (a row
/// contributes at most 65535*2^16 < 2^32 per lane) flushed into a u64 total;
/// the old code's per-pixel f64 additions were also exact (every partial sum
/// of a u8/u16 plane stays < 2^53), so converting the u64 once at the end is
/// bit-identical while replacing ~14 scalar Ir/px with ~0.3.
fn absDiffSumInt(comptime T: type, src: []const T, ref: []const T, stride: u32, w: u32, h: u32) u64 {
    const n = comptime (std.simd.suggestVectorLength(T) orelse 1);
    var acc: u64 = 0;
    const wv = w - w % n;
    var srcp: []const T = src;
    var refp: []const T = ref;
    for (0..h) |_| {
        var x: usize = 0;
        if (comptime n > 1) {
            var row_acc: @Vector(n, u32) = @splat(0);
            while (x < wv) : (x += n) {
                const v: @Vector(n, T) = srcp[x..][0..n].*;
                const j: @Vector(n, T) = refp[x..][0..n].*;
                row_acc += @as(@Vector(n, u32), @max(v, j) - @min(v, j));
            }
            acc += @reduce(.Add, @as(@Vector(n, u64), row_acc));
        }
        while (x < w) : (x += 1) {
            acc += @max(srcp[x], refp[x]) - @min(srcp[x], refp[x]);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }
    return acc;
}

pub fn minMaxNoThrRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    if (@typeInfo(T) == .int) {
        diffacc = @floatFromInt(absDiffSumInt(T, src, ref, stride, w, h));
    } else {
        // Float diff stays a scalar left-to-right f64 accumulation: vector
        // partial sums would change the FP addition order and drift the Diff
        // prop. min/max (order-free) are still vectorized separately below.
        var srcp: []const T = src;
        var refp: []const T = ref;
        for (0..h) |_| {
            for (srcp[0..w], refp[0..w]) |v, j| {
                diffacc += @abs(v - j);
            }
            srcp = srcp[stride..];
            refp = refp[stride..];
        }
    }

    const mm = minMaxPlane(T, src, stride, w, h);

    var diff: f64 = diffacc / total;
    if (@typeInfo(T) == .int) {
        diff /= d.peakf;
    }

    setMinMaxProps(T, props, d, mm);
    props.setFloat(d.prop.d, diff, .Append);
}
