//! BoxBlur with comptime radius size

const std = @import("std");
const simd = @import("simd.zig");
const math = std.math;

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(u32) orelse 8;

/// Exact `floor(N / ksize)` for a bounded N, as a multiply-and-shift.
///
/// The integer kernels keep a *raw* window sum and divide once per output. A
/// Q16 reciprocal cannot do that job: u16 output needs more than
/// `16 + log2(ksize)` reciprocal bits, so a 32-bit lane holding `value << 16`
/// has none to spare, and any truncated reciprocal makes the running sum drift
/// off the value it is supposed to hold.
///
/// `narrow` means `N*mul` provably fits in 32 bits, so the divide is one
/// `vpmulld` + shift instead of a widening 32x32->64 on each dword half. u8
/// wins it at every radius up to 132 (`narrowMagicFor`) and is compute-bound
/// enough to care; u16's range never allows it.
pub fn Magic(comptime narrow_: bool) type {
    return struct {
        const Self = @This();
        pub const narrow = narrow_;

        mul: u32,
        shift: u6,

        pub inline fn apply(mg: Self, comptime T: type, sum: u32) T {
            if (narrow) return @intCast((sum *% mg.mul) >> @as(u5, @intCast(mg.shift)));
            return @intCast((@as(u64, sum) * mg.mul) >> mg.shift);
        }
    };
}

/// The accumulator peaks one pixel above the largest window sum: kernels seed
/// it with +radius (see blurInt) and slide it a pixel at a time.
inline fn sumBound(comptime T: type, radius: u32) ?struct { ksize: u64, nmax: u64 } {
    const ksize: u64 = 2 * @as(u64, radius) + 1;
    const maxval: u64 = math.maxInt(T);
    const nmax: u64 = ksize * maxval + radius;
    if (nmax + maxval >= (1 << 32)) return null;
    return .{ .ksize = ksize, .nmax = nmax };
}

/// Granlund-Montgomery: `mul = floor(2^shift/ksize) + 1` divides exactly for
/// every `N <= nmax` iff `nmax * (mul*ksize - 2^shift) < 2^shift`.
inline fn searchMagic(ksize: u64, nmax: u64, comptime narrow: bool, lo: u6, hi: u6) ?Magic(narrow) {
    var shift: u6 = lo;
    while (shift < hi) : (shift += 1) {
        const mul: u64 = (@as(u64, 1) << shift) / ksize + 1;
        if (mul >= (1 << 32)) return null; // mul only grows with shift
        if (narrow and mul * nmax >= (1 << 32)) return null;
        const err: u64 = mul * ksize - (@as(u64, 1) << shift);
        if (nmax * err < (@as(u64, 1) << shift)) return .{ .mul = @intCast(mul), .shift = shift };
    }
    return null;
}

/// Wide magic for a box of `radius`, or null when none exists (u16 runs out at
/// radius 23332; `boxBlurCreate` rejects that rather than letting getFrame lose
/// precision). The shift >= 32 search window is the one verified exhaustively
/// over the full N range at every radius; smaller shifts also divide exactly at
/// some radii but buy nothing (same instructions either way).
pub fn magicFor(comptime T: type, radius: u32) ?Magic(false) {
    const b = sumBound(T, radius) orelse return null;
    return searchMagic(b.ksize, b.nmax, false, 32, 63);
}

/// Narrow magic (32-bit product) for a box of `radius`, or null when the range
/// is too wide for one. Never exists for u16: an exact quotient needs
/// `2^shift > nmax*err`, and with `nmax ~ ksize*65535` that pushes `mul*nmax`
/// past 2^32 at every radius.
pub fn narrowMagicFor(comptime T: type, radius: u32) ?Magic(true) {
    const b = sumBound(T, radius) orelse return null;
    return searchMagic(b.ksize, b.nmax, true, 1, 32);
}

/// `(sum * mul) >> shift` across a vector.
///
/// The wide path is left as a plain widening multiply on purpose: written this
/// way each backend legalises `mul(zext, zext)` into its own widening
/// instruction -- 2x vpmuludq on AVX2, umull+umull2 on NEON. Hand-splitting it
/// into even/odd dwords to court vpmuludq directly (bitcast + mask + high-dword
/// blend) is *worse on both*: 37 vs 34 instructions on haswell, and on aarch64
/// it loses the zero-extend that umull matching needs and scalarises to
/// per-lane fmov/mul/insert (48 instructions, measured 2026-07-16).
pub inline fn scaleVec(comptime T: type, comptime n: usize, s: @Vector(n, u32), mg: anytype) @Vector(n, T) {
    if (@TypeOf(mg).narrow) {
        const mulv: @Vector(n, u32) = @splat(mg.mul);
        const sh: @Vector(n, u5) = @splat(@intCast(mg.shift));
        return @intCast((s *% mulv) >> sh);
    }

    const U64V = @Vector(n, u64);
    // Splat as u32 and widen *here*, so the multiply reads literally as
    // mul(zext(v32), zext(v32)). Splatting straight to u64 lets LICM hoist the
    // constant out as 64-bit lanes, and the backend then can no longer see it
    // was a zero-extend -- on NEON that costs umull and scalarises the loop.
    const mulv32: @Vector(n, u32) = @splat(mg.mul);
    const sh: @Vector(n, u6) = @splat(mg.shift);
    return @intCast((@as(U64V, @intCast(s)) * @as(U64V, @intCast(mulv32))) >> sh);
}

pub fn hvBlur(comptime T: type, comptime radius: u32, src: []const T, dst: []T, stride: u32, w: u32, h: u32) error{OutOfMemory}!void {
    const ksize: u32 = (radius << 1) + 1;
    const iradius: i32 = @bitCast(radius);
    const ih: i32 = @bitCast(h);

    const tmp = try allocator.alloc(T, w);
    defer allocator.free(tmp);

    // raw vertical column sums; max ksize * 65535 < 2^22 fits u32
    const col: []u32 = if (@typeInfo(T) == .int) try allocator.alloc(u32, w) else &.{};
    defer if (@typeInfo(T) == .int) allocator.free(col);

    var i: i32 = 0;
    while (i < h) : (i += 1) {
        const ui: u32 = @bitCast(i);
        const dstp: []T = dst[ui * stride ..];

        if (@typeInfo(T) == .int) {
            const mg = comptime (magicFor(T, radius) orelse unreachable);
            // interior rows shift the column window by one; edge rows
            // (mirrored window) recompute it from scratch
            if ((i > iradius) and (i + iradius < ih)) {
                colUpdate(T, col, src[(ui + radius) * stride ..], src[(ui - radius - 1) * stride ..], w);
            } else {
                const srcp = mirrorRows(T, ksize, src, i, stride, ih);
                colRecompute(T, ksize, col, &srcp, w);
            }
            colScale(T, col, tmp, w, radius, mg);
            hBlurInt(T, tmp, dstp, w, ksize);
        } else {
            const div: f32 = 1.0 / @as(f32, @floatFromInt(ksize));
            const srcp = mirrorRows(T, ksize, src, i, stride, ih);
            vBlurFloat(T, ksize, &srcp, tmp, w, div);
            hBlurFloat(T, tmp, dstp, @bitCast(w), ksize, div);
        }
    }
}

/// The ksize source row slices contributing to output row i, with mirrored
/// indices at the top/bottom edges.
inline fn mirrorRows(comptime T: type, comptime ksize: u32, src: []const T, i: i32, stride: u32, ih: i32) [ksize][]const T {
    const iradius: i32 = @intCast(ksize >> 1);
    const dist_from_bottom: i32 = ih - 1 - i;
    var srcp: [ksize][]const T = undefined;

    var k: i32 = 0;
    while (k < iradius) : (k += 1) {
        const row: i32 = if (i < iradius - k) @min(iradius - k - i, ih - 1) else (i - iradius + k);
        const urow: u32 = @bitCast(row);
        srcp[@intCast(k)] = src[urow * stride ..];
    }

    k = iradius;
    while (k < ksize) : (k += 1) {
        const row: i32 = if (dist_from_bottom < k - iradius) (i - @min(k - iradius - dist_from_bottom, i)) else (i - iradius + k);
        const urow: u32 = @bitCast(row);
        srcp[@intCast(k)] = src[urow * stride ..];
    }

    return srcp;
}

inline fn colUpdate(comptime T: type, col: []u32, add_row: []const T, sub_row: []const T, w: u32) void {
    const UV = @Vector(vec_len, u32);
    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        const a: @Vector(vec_len, T) = add_row[j..][0..vec_len].*;
        const s: @Vector(vec_len, T) = sub_row[j..][0..vec_len].*;
        var c: UV = col[j..][0..vec_len].*;
        c += @intCast(a);
        c -= @intCast(s);
        col[j..][0..vec_len].* = c;
    }
    while (j < w) : (j += 1) {
        col[j] += add_row[j];
        col[j] -= sub_row[j];
    }
}

inline fn colRecompute(comptime T: type, comptime ksize: u32, col: []u32, srcp: *const [ksize][]const T, w: u32) void {
    const UV = @Vector(vec_len, u32);
    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        var acc: UV = @splat(0);
        inline for (0..ksize) |k| {
            const v: @Vector(vec_len, T) = srcp[k][j..][0..vec_len].*;
            acc += @intCast(v);
        }
        col[j..][0..vec_len].* = acc;
    }
    while (j < w) : (j += 1) {
        var sum: u32 = 0;
        inline for (0..ksize) |k| {
            sum += srcp[k][j];
        }
        col[j] = sum;
    }
}

/// `col` holds raw window sums; the +radius seed that turns the magic's floor
/// into a round-to-nearest is added here rather than carried (see blurInt).
inline fn colScale(comptime T: type, col: []const u32, tmp: []T, w: u32, radius: u32, mg: anytype) void {
    const radv: @Vector(vec_len, u32) = @splat(radius);

    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        const c: @Vector(vec_len, u32) = col[j..][0..vec_len].*;
        tmp[j..][0..vec_len].* = scaleVec(T, vec_len, c + radv, mg);
    }
    while (j < w) : (j += 1) {
        tmp[j] = mg.apply(T, col[j] + radius);
    }
}

/// `scaleVec` before the final narrowing: the quotient in u32 lanes. Every lane
/// is a box average, so it is <= maxInt(T) and the callers' saturating packs
/// never actually saturate.
pub inline fn scaleVec32(comptime T: type, comptime n: usize, s: @Vector(n, u32), mg: anytype) @Vector(n, u32) {
    if (@TypeOf(mg).narrow) {
        const mulv: @Vector(n, u32) = @splat(mg.mul);
        const sh: @Vector(n, u5) = @splat(@intCast(mg.shift));
        return (s *% mulv) >> sh;
    }
    _ = T;
    if (comptime simd.use_avx2 and n % 2 == 0) {
        // x86 even/odd split (shift >= 32 always holds for the wide magic, see
        // magicFor). Each u64 lane carries two u32 sums; vpmuludq multiplies
        // the low dword of each qword by the zero-extended magic. The even
        // quotient ends up in the low dword of `pe >> shift`, the odd one in
        // the high dword of `po >> (shift - 32)` (its high dword is exactly
        // `po >> shift` because every quotient is < 2^32), so one vpblendd
        // merges them — no widen/repack shuffles, unlike the generic u64
        // form below (vpmovzxdq x2 + vshufps + vpermpd per 8 lanes, measured
        // as the u16 h-pass bottleneck). The generic form stays the portable
        // path: on NEON it maps to umull directly (see scaleVec).
        std.debug.assert(mg.shift >= 32);
        const H = n / 2;
        const Q = @Vector(H, u64);
        const lo32: Q = @splat(0xFFFF_FFFF);
        const s64: Q = @bitCast(s);
        const m64: Q = @splat(mg.mul);
        const pe = (s64 & lo32) * m64;
        const po = (s64 >> @splat(32)) * m64;
        const qe = pe >> @splat(mg.shift);
        const qo = po >> @splat(mg.shift - 32);
        return @bitCast((qe & lo32) | (qo & ~lo32));
    }
    const U64V = @Vector(n, u64);
    const mulv32: @Vector(n, u32) = @splat(mg.mul);
    const sh: @Vector(n, u6) = @splat(mg.shift);
    return @intCast((@as(U64V, @intCast(s)) * @as(U64V, @intCast(mulv32))) >> sh);
}

/// Vectors per narrowing store in the integer store paths below. On AVX2 at 8
/// u32 lanes the per-vector u32->u8/u16 narrow is a shuffle wall (vextracti128
/// + 2x vpshufb + vpunpckldq + a partial store per 8 px); grouping 4 (u8) or 2
/// (u16) quotient vectors lets one saturating pack (simd.packClampU8x4 /
/// packUsI32ToU16) do it with a single full-width store. Other widths keep the
/// per-vector store (AVX-512 narrows natively with vpmov*).
pub fn packGroup(comptime T: type, comptime n: usize) comptime_int {
    if (!(n == 8 and simd.use_avx2)) return 1;
    return if (T == u8) 4 else 2;
}

/// Store `g` = packGroup(T, n) quotient vectors at dst[0 .. g*n).
pub inline fn storePacked(comptime T: type, comptime n: usize, dst: []T, q: [packGroup(T, n)]@Vector(n, u32)) void {
    const g = comptime packGroup(T, n);
    if (comptime g == 4) {
        dst[0..32].* = simd.packClampU8x4(@bitCast(q[0]), @bitCast(q[1]), @bitCast(q[2]), @bitCast(q[3]));
    } else if (comptime g == 2) {
        dst[0..16].* = simd.packUsI32ToU16(@bitCast(q[0]), @bitCast(q[1]));
    } else {
        dst[0..n].* = @as(@Vector(n, T), @intCast(q[0]));
    }
}

/// In-register inclusive prefix sum (mod 2^32) of n u32 lanes.
inline fn scanDeltas(comptime n: usize, d0: @Vector(n, u32)) @Vector(n, u32) {
    var d = d0;
    comptime var sh: u32 = 1;
    inline while (sh < n) : (sh *= 2) {
        d = d +% std.simd.shiftElementsRight(d, sh, 0);
    }
    return d;
}

/// Vectorized center segment of the integer sliding-window sum:
///   for x in [x0, x1): sum += src[x+radius] - src[x-radius-1]; dst[x] = mg.apply(sum)
/// The scalar running sum is loop-carried, so LLVM never vectorizes it
/// (measured 8 scalar instr/px = 76% of the default int BoxBlur). Lanes here
/// compute an in-register prefix sum of the deltas instead. All arithmetic is
/// mod-2^32 exact and therefore bit-identical to the scalar loop: the true
/// running sum (raw window sum + radius; see blurInt) is non-negative and
/// < 2^32, so the u32 residue *is* the value, and wrapping preserves it --
/// negative deltas included, as their two's-complement residue.
///
/// The carry between vectors stays a vector splat and advances by the
/// broadcast *block total* (`carry +% splat(scan[n-1])`), which only depends on
/// the current block's loads. The loop-carried chain is therefore one vpaddd
/// per vector; carrying the last lane through a GPR (s[n-1] -> vmovd ->
/// vpbroadcastd) put an extract + two domain crossings on it (~12 cycles per 8
/// px, measured as the h-pass bottleneck).
pub fn slideSumInt(
    comptime T: type,
    srcp: []const T,
    dstp: []T,
    x0: u32,
    x1: u32,
    radius: u32,
    mg: anytype,
    sum0: u32,
) u32 {
    const n = comptime (std.simd.suggestVectorLength(u32) orelse 1);
    var sum: u32 = sum0;
    var x: u32 = x0;
    if (comptime n > 1) {
        const U = @Vector(n, u32);
        const last: @Vector(n, i32) = @splat(n - 1);
        const g = comptime packGroup(T, n);
        var carry: U = @splat(sum);
        while (x + g * n <= x1) : (x += g * n) {
            var q: [g]U = undefined;
            inline for (0..g) |k| {
                const xk = x + k * n;
                const a: @Vector(n, i32) = @intCast(@as(@Vector(n, T), srcp[xk + radius ..][0..n].*));
                const b: @Vector(n, i32) = @intCast(@as(@Vector(n, T), srcp[xk - radius - 1 ..][0..n].*));
                const d = scanDeltas(n, @bitCast(a - b));
                q[k] = scaleVec32(T, n, d +% carry, mg);
                carry = carry +% @shuffle(u32, d, undefined, last);
            }
            storePacked(T, n, dstp[x..], q);
        }
        // leftover whole vectors (fewer than one pack group)
        while (x + n <= x1) : (x += n) {
            const a: @Vector(n, i32) = @intCast(@as(@Vector(n, T), srcp[x + radius ..][0..n].*));
            const b: @Vector(n, i32) = @intCast(@as(@Vector(n, T), srcp[x - radius - 1 ..][0..n].*));
            const d = scanDeltas(n, @bitCast(a - b));
            dstp[x..][0..n].* = @as(@Vector(n, T), @intCast(scaleVec32(T, n, d +% carry, mg)));
            carry = carry +% @shuffle(u32, d, undefined, last);
        }
        sum = carry[0];
    }
    while (x < x1) : (x += 1) {
        sum +%= srcp[radius + x];
        sum -%= srcp[x - radius - 1];
        dstp[x] = mg.apply(T, sum);
    }
    return sum;
}

inline fn hBlurInt(comptime T: type, srcp: []T, dstp: []T, w: u32, comptime ksize: u32) void {
    const radius: u32 = ksize >> 1;
    const mg = comptime (magicFor(T, radius) orelse unreachable);
    var sum: u32 = @as(u32, srcp[radius]) + radius;

    for (0..radius) |x| {
        sum += @as(u32, srcp[x]) << 1;
    }

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        sum += srcp[radius + x];
        sum -= srcp[radius - x];
        dstp[x] = mg.apply(T, sum);
    }

    sum = slideSumInt(T, srcp, dstp, x, w - radius, radius, mg, sum);
    x = w - radius;

    while (x < w) : (x += 1) {
        sum += srcp[2 * w - radius - x - 1];
        sum -= srcp[x - radius - 1];
        dstp[x] = mg.apply(T, sum);
    }
}

fn vBlurFloat(comptime T: type, comptime ksize: u32, src: *const [ksize][]const T, dstp: []T, w: u32, comptime div: f32) void {
    // Accumulate in f32 regardless of T so f16 output matches the scalar
    // f32-accumulate-then-narrow path (bit-exact for f32).
    const fvec = std.simd.suggestVectorLength(f32) orelse 8;
    const FV = @Vector(fvec, f32);
    const dv: FV = @splat(div);

    var j: u32 = 0;
    const wv = w - (w % fvec);
    while (j < wv) : (j += fvec) {
        var acc: FV = @splat(0.0);
        inline for (0..ksize) |k| {
            const v: @Vector(fvec, T) = src[k][j..][0..fvec].*;
            const vf: FV = if (T == f32) v else @floatCast(v);
            acc = acc + dv * vf;
        }
        if (T == f32) {
            dstp[j..][0..fvec].* = acc;
        } else {
            dstp[j..][0..fvec].* = @as(@Vector(fvec, T), @floatCast(acc));
        }
    }
    while (j < w) : (j += 1) {
        var sum: f32 = 0.0;
        inline for (0..ksize) |k| {
            sum += div * src[k][j];
        }
        dstp[j] = if (T == f32) sum else @floatCast(sum);
    }
}

fn hBlurFloat(comptime T: type, srcp: []T, dstp: []T, w: i32, comptime ksize: u32, comptime div: f32) void {
    const radius: i32 = @as(i32, @bitCast(ksize)) >> 1;
    // Accumulate in f32 regardless of T (bit-exact for f32, f16 matches the
    // scalar f32-accumulate-then-narrow reference).
    const fvec = std.simd.suggestVectorLength(f32) orelse 8;
    const FV = @Vector(fvec, f32);
    const dv: FV = @splat(div);

    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }

    j = radius;
    const center_end: i32 = w - @min(w, radius);
    while (j + fvec <= center_end) : (j += fvec) {
        const uj: u32 = @intCast(j - radius);
        var acc: FV = @splat(0.0);
        inline for (0..ksize) |k| {
            // Laundered: the ksize overlapping tap loads otherwise fuse into
            // shuffle chains (measured 2x instructions at radius 4, 3x at 18).
            const v = simd.loaduOpaque(T, fvec, srcp[uj + k ..][0..fvec]);
            const vf: FV = if (T == f32) v else @floatCast(v);
            acc = acc + dv * vf;
        }
        if (T == f32) {
            dstp[@intCast(j)..][0..fvec].* = acc;
        } else {
            dstp[@intCast(j)..][0..fvec].* = @as(@Vector(fvec, T), @floatCast(acc));
        }
    }
    while (j < center_end) : (j += 1) {
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += div * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }
}
