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
    const n_opt = comptime std.simd.suggestVectorLength(f32);
    const n = comptime (n_opt orelse 1);
    var i: usize = 0;

    if (comptime n > 1) {
        const V = @Vector(n, f32);
        const dark_v: V = @splat(dark_thr);
        const bright_v: V = @splat(bright_thr);
        const elast_v: V = @splat(elast);
        const zero_v: V = @splat(0.0);
        const half_v: V = @splat(0.5);

        while (i + n <= dstp.len) : (i += n) {
            const fraw: @Vector(n, T) = fltp[i..][0..n].*;
            const sraw: @Vector(n, T) = srcp[i..][0..n].*;
            const rraw: @Vector(n, T) = refp[i..][0..n].*;
            const ff: V = if (@typeInfo(T) == .int) @floatFromInt(fraw) else @floatCast(fraw);
            const sf: V = if (@typeInfo(T) == .int) @floatFromInt(sraw) else @floatCast(sraw);
            const rf: V = if (@typeInfo(T) == .int) @floatFromInt(rraw) else @floatCast(rraw);

            const diff_signed = ff - rf;
            const diff_abs = @abs(diff_signed);
            const thr1 = @select(f32, diff_signed > zero_v, bright_v, dark_v);
            const thr2 = thr1 * elast_v;

            const interp = sf + (ff - sf) * (thr2 - diff_abs) / (thr2 - thr1);
            var out = @select(f32, diff_abs >= thr2, sf, interp);
            out = @select(f32, diff_abs <= thr1, ff, out);

            if (@typeInfo(T) == .int) {
                const rounded = @trunc(out + half_v);
                const w: @Vector(n, T) = @intFromFloat(rounded);
                dstp[i..][0..n].* = w;
            } else {
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
