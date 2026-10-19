const std = @import("std");
const simd = @import("simd.zig");

/// Vectorized limit filter. LLVM refuses to if-convert the branchy scalar
/// diamond (a division on one path), leaving ~16 scalar Ir/px with 2-3
/// data-dependent branches — measured 99% of the filter's kernel time. The
/// vector form mirrors the scalar math op-for-op in f32 and keeps every
/// @select payload in f32 (one vblendvps each; narrower payloads would repack
/// per select). The interpolation term is computed unconditionally — its
/// division cannot trap and out-of-window lanes are selected away before the
/// final conversion, so results are bit-identical to the branchy version.
pub fn process(
    comptime T: type,
    fltp: []const T,
    srcp: []const T,
    refp: []const T,
    dstp: []T,
    dark_thr: f32,
    bright_thr: f32,
    elast: f32,
) void {
    if (@typeInfo(T) == .int) {
        const peak: f32 = @floatFromInt(std.math.maxInt(T));
        if (interpWindowEmpty(dark_thr, elast, peak) and interpWindowEmpty(bright_thr, elast, peak)) {
            return processNoInterp(T, fltp, srcp, refp, dstp, dark_thr, bright_thr);
        }
    }

    // dark_thr == bright_thr (the default) makes thr1, thr2 and thr2 - thr1
    // lane-invariant: the per-lane sign compare, blend, multiply and subtract
    // disappear (4 of ~26 vector uops per 8 px — every vector op, int or FP,
    // competes for Zen 3's four SIMD pipes, which is what bounds this loop;
    // the divide is not the limit, measured). Same f32 values either way.
    if (dark_thr == bright_thr) {
        processVec(T, true, fltp, srcp, refp, dstp, dark_thr, bright_thr, elast);
    } else {
        processVec(T, false, fltp, srcp, refp, dstp, dark_thr, bright_thr, elast);
    }
}

fn processVec(
    comptime T: type,
    comptime same_thr: bool,
    fltp: []const T,
    srcp: []const T,
    refp: []const T,
    dstp: []T,
    dark_thr: f32,
    bright_thr: f32,
    elast: f32,
) void {
    const n_opt = comptime std.simd.suggestVectorLength(f32);
    const n = comptime (n_opt orelse 1);
    var i: usize = 0;

    if (comptime n > 1) {
        const V = @Vector(n, f32);
        const C = struct {
            dark: V,
            bright: V,
            elast: V,
            thr2: V,
            den: V,
        };
        const c: C = .{
            .dark = @splat(dark_thr),
            .bright = @splat(bright_thr),
            .elast = @splat(elast),
            .thr2 = @splat(dark_thr * elast),
            .den = @splat(dark_thr * elast - dark_thr),
        };
        const half_v: V = @splat(0.5);

        const K = struct {
            inline fn lanes(fltp_: []const T, srcp_: []const T, refp_: []const T, j: usize, k: C) V {
                const fraw: @Vector(n, T) = fltp_[j..][0..n].*;
                const sraw: @Vector(n, T) = srcp_[j..][0..n].*;
                const rraw: @Vector(n, T) = refp_[j..][0..n].*;
                const ff: V = if (@typeInfo(T) == .int) @floatFromInt(fraw) else @floatCast(fraw);
                const sf: V = if (@typeInfo(T) == .int) @floatFromInt(sraw) else @floatCast(sraw);
                const rf: V = if (@typeInfo(T) == .int) @floatFromInt(rraw) else @floatCast(rraw);

                const diff_signed = ff - rf;
                const diff_abs = @abs(diff_signed);
                const thr1 = if (same_thr) k.dark else @select(f32, diff_signed > @as(V, @splat(0.0)), k.bright, k.dark);
                const thr2 = if (same_thr) k.thr2 else thr1 * k.elast;
                const den = if (same_thr) k.den else thr2 - thr1;

                const interp = sf + (ff - sf) * (thr2 - diff_abs) / den;
                var out = @select(f32, diff_abs >= thr2, sf, interp);
                out = @select(f32, diff_abs <= thr1, ff, out);
                return out;
            }
        };

        if (@typeInfo(T) == .int) {
            // Integer output: `out` lies between two samples of T, so out + 0.5
            // is in [0.5, maxInt(T) + 0.5] and @intFromFloat's truncation is
            // exactly the scalar path's @trunc (the separate vroundps was a
            // no-op LLVM kept). Groups of 4 (u8) / 2 (u16) vectors narrow with
            // one saturating pack + full-width store instead of a per-8-lane
            // vextracti128 + vpackssdw + vpackuswb + vmovq chain; the in-range
            // values never saturate, so the pack is bit-identical to @intCast.
            const g = comptime if (n == 8 and simd.use_avx2) (if (T == u8) 4 else 2) else 1;
            while (i + g * n <= dstp.len) : (i += g * n) {
                var q: [g]@Vector(n, i32) = undefined;
                inline for (0..g) |k| {
                    q[k] = @intFromFloat(K.lanes(fltp, srcp, refp, i + k * n, c) + half_v);
                }
                if (comptime g == 4) {
                    dstp[i..][0..32].* = simd.packClampU8x4(q[0], q[1], q[2], q[3]);
                } else if (comptime g == 2) {
                    dstp[i..][0..16].* = simd.packUsI32ToU16(q[0], q[1]);
                } else {
                    dstp[i..][0..n].* = simd.intCastSignedToUnsigned(@Vector(n, T), q[0]);
                }
            }
            while (i + n <= dstp.len) : (i += n) {
                const q: @Vector(n, i32) = @intFromFloat(K.lanes(fltp, srcp, refp, i, c) + half_v);
                dstp[i..][0..n].* = simd.intCastSignedToUnsigned(@Vector(n, T), q);
            }
        } else {
            while (i + n <= dstp.len) : (i += n) {
                const out = K.lanes(fltp, srcp, refp, i, c);
                // For T == f16 the launder blocks LLVM's trunc-through-select
                // combine, which would otherwise repack both selects into
                // f16-domain blends ([narrow-select], see simd.opaqueF32V).
                // Pure value identity: the stored bits are unchanged.
                const w: @Vector(n, T) = @floatCast(if (T == f16) simd.opaqueF32V(n, out) else out);
                dstp[i..][0..n].* = w;
            }
        }
    }

    while (i < dstp.len) : (i += 1) {
        const f = fltp[i];
        const s = srcp[i];
        const r = refp[i];
        const sf: f32 = if (@typeInfo(T) == .int) @floatFromInt(s) else s;
        const ff: f32 = if (@typeInfo(T) == .int) @floatFromInt(f) else f;
        const rf: f32 = if (@typeInfo(T) == .int) @floatFromInt(r) else r;

        const diff_signed: f32 = ff - rf;
        const diff_abs: f32 = @abs(diff_signed);
        const thr1: f32 = if (diff_signed > 0) bright_thr else dark_thr;
        const thr2: f32 = thr1 * elast;

        var out: f32 = undefined;
        if (diff_abs <= thr1) {
            out = ff;
        } else if (diff_abs >= thr2) {
            out = sf;
        } else {
            out = sf + (ff - sf) * (thr2 - diff_abs) / (thr2 - thr1);
        }

        dstp[i] = if (@typeInfo(T) == .int) @trunc(out + 0.5) else @floatCast(out);
    }
}

/// True when no integer |flt - ref| can land strictly inside (thr1, thr2),
/// thr2 = thr1 * elast evaluated exactly as `process` does (one f32 multiply).
/// Integer samples make |diff| an integer-valued f32, and the smallest integer
/// above thr1 is floor(thr1) + 1 (exact in f32: thr1 <= 65535). If that one is
/// not below thr2 (or exceeds the largest possible |diff|), no larger one is
/// either, so the interpolation select can never fire. Default thresholds on
/// 8-bit (thr 1, elast 2 -> window (1, 2)) are exactly this case.
fn interpWindowEmpty(thr1: f32, elast: f32, peak: f32) bool {
    const thr2 = thr1 * elast;
    const k = @floor(thr1) + 1.0;
    return !(k < thr2) or k > peak;
}

/// Integer kernel for an empty interpolation window (see interpWindowEmpty):
/// per pixel `|f - r| <= thr1 ? f : s`, which is what `process` computes when
/// the interp lane is never selected — `diff_abs <= thr1` on an integer-valued
/// f32 equals `|f - r| <= floor(thr1)`, and the selected f/s round-trip through
/// f32 + 0.5 + truncation unchanged. thr1 is picked by the sign of f - r exactly
/// like the float path (> 0 -> bright, else dark). Pure native-width integer
/// compares/selects (32 u8 or 16 u16 lanes), no conversions or division.
fn processNoInterp(comptime T: type, fltp: []const T, srcp: []const T, refp: []const T, dstp: []T, dark_thr: f32, bright_thr: f32) void {
    const maxv: f32 = @floatFromInt(std.math.maxInt(T));
    const dark_i: T = @intFromFloat(@min(@floor(dark_thr), maxv));
    const bright_i: T = @intFromFloat(@min(@floor(bright_thr), maxv));
    var i: usize = 0;
    if (comptime std.simd.suggestVectorLength(T)) |n| {
        const V = @Vector(n, T);
        const dark_v: V = @splat(dark_i);
        const bright_v: V = @splat(bright_i);
        while (i + n <= dstp.len) : (i += n) {
            const f: V = fltp[i..][0..n].*;
            const sv: V = srcp[i..][0..n].*;
            const r: V = refp[i..][0..n].*;
            const dabs = @max(f, r) - @min(f, r);
            const t1 = @select(T, f > r, bright_v, dark_v);
            dstp[i..][0..n].* = @select(T, dabs <= t1, f, sv);
        }
    }
    while (i < dstp.len) : (i += 1) {
        const f = fltp[i];
        const r = refp[i];
        const dabs = @max(f, r) - @min(f, r);
        const t1 = if (f > r) bright_i else dark_i;
        dstp[i] = if (dabs <= t1) f else srcp[i];
    }
}
