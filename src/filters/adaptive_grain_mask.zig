const std = @import("std");
const common = @import("../common.zig");
const simd = @import("simd.zig");
const vcl = @import("../vcl.zig");
extern fn powf(x: f32, y: f32) f32;

const C4: f32 = 18.188;
const C3: f32 = -45.47;
const C2: f32 = 36.624;
const C1: f32 = -9.466;
const C0: f32 = 1.124;

pub fn calcLumaScaling(average: f32, luma_scaling: f32) f32 {
    const a = @min(@max(average, 0.0), 1.0);
    return a * a * luma_scaling;
}

/// Constants of `poly`, kept in a pinned table (see `PowConsts.base`).
const pow_consts = [_]f32{ C4, C3, C2, C1, C0 };

fn PowConsts(comptime VL: usize) type {
    return struct {
        const rows: [pow_consts.len][VL]f32 align(common.vec_len) = blk: {
            var t: [pow_consts.len][VL]f32 = undefined;
            for (pow_consts, 0..) |c, i| t[i] = @splat(c);
            break :blk t;
        };

        const Base = *const [pow_consts.len][VL]f32;

        inline fn base() Base {
            if (comptime !simd.use_avx) return &rows;
            return asm (""
                : [out] "=r" (-> Base),
                : [in] "0" (&rows),
            );
        }

        inline fn get(b: Base, comptime v: f32) @Vector(VL, f32) {
            const idx = comptime blk: {
                for (pow_consts, 0..) |c, i| {
                    if (c == v) break :blk i;
                }
                @compileError("PowConsts: constant missing from pow_consts");
            };
            return b[idx];
        }
    };
}

inline fn poly(comptime V: type, x: V, cb: anytype) V {
    const n = @typeInfo(V).vector.len;
    const P = PowConsts(n);
    return simd.madd(V, x, simd.madd(V, x, simd.madd(V, x, simd.madd(V, x, P.get(cb, C4), P.get(cb, C3)), P.get(cb, C2)), P.get(cb, C1)), P.get(cb, C0));
}

/// The integer LUT's bases 1 - x * poly(x), x = i / 255, with poly's steps
/// fused like the Rust reference's: the integer output is bit-exact with it.
/// Built at compile time, where @mulAdd is exact on every target. At run
/// time, simd.madd's unfused fallback (targets without FMA) moved some
/// entries by one code value, and @mulAdd would call libm's fmaf.
const lut_bases: [256]f32 = blk: {
    var t: [256]f32 = undefined;
    for (&t, 0..) |*b, i| {
        const x: f32 = @as(f32, @floatFromInt(i)) / 255.0;
        b.* = 1.0 - x * @mulAdd(f32, x, @mulAdd(f32, x, @mulAdd(f32, x, @mulAdd(f32, x, C4, C3), C2), C1), C0);
    }
    break :blk t;
};

pub fn buildLut(comptime T: type, lut: *[256]T, temp: f32, peak: f32) void {
    const type_max: f32 = @floatFromInt(std.math.maxInt(T));
    for (lut, lut_bases) |*e, base| {
        const v = powf(base, temp);
        e.* = @intFromFloat(@max(@min(v * peak, type_max), 0.0));
    }
}

pub fn processInt(comptime T: type, src: []const T, dst: []T, lut: *const [256]T, shift: u5) void {
    const sh: std.math.Log2Int(T) = @intCast(shift);
    for (dst, src) |*o, s| {
        o.* = lut[@as(usize, @intCast(s >> sh))];
    }
}

pub fn processIntU8(src: []const u8, dst: []u8, lut: *const [256]u8) void {
    const L = simd.LutU8;
    if (comptime L.lanes > 0) {
        const tbl = L.init(lut);
        var i: usize = 0;
        const n = dst.len - dst.len % L.lanes;
        while (i < n) : (i += L.lanes) {
            dst[i..][0..L.lanes].* = tbl.get(src[i..][0..L.lanes].*);
        }
        while (i < dst.len) : (i += 1) dst[i] = lut[src[i]];
        return;
    }
    for (dst, src) |*o, s| o.* = lut[s];
}

pub fn computeAverage(comptime T: type, src: []const T, stride: u32, w: u32, h: u32, peak: f32) f32 {
    const npix: u64 = @as(u64, w) * @as(u64, h);
    if (npix == 0) return 0.0;

    var row: []const T = src;

    if (@typeInfo(T) == .float) {
        const VL = std.simd.suggestVectorLength(f32) orelse 8;
        const Vd = @Vector(VL, f64);
        var accv: Vd = @splat(0.0);
        var tail: f64 = 0.0;
        for (0..h) |_| {
            var x: u32 = 0;
            while (x + VL <= w) : (x += VL) {
                const vf: @Vector(VL, f32) = row[x..][0..VL].*;
                accv += @as(Vd, @floatCast(vf));
            }
            while (x < w) : (x += 1) tail += row[x];
            row = row[stride..];
        }
        const sum = @reduce(.Add, accv) + tail;
        return @floatCast(sum / @as(f64, @floatFromInt(npix)));
    }

    var sum: u64 = 0;
    if (T == u8 and simd.sad_u8_lanes > 0) {
        const n = simd.sad_u8_lanes;
        const zero: @Vector(n, u8) = @splat(0);
        var acc: @Vector(n / 8, u64) = @splat(0);
        var tail: u64 = 0;
        for (0..h) |_| {
            var x: u32 = 0;
            while (x + n <= w) : (x += n) {
                acc += simd.sadU8(n, row[x..][0..n].*, zero);
            }
            while (x < w) : (x += 1) tail += row[x];
            row = row[stride..];
        }
        sum = @reduce(.Add, acc) + tail;
    } else if (w < (1 << 19)) {
        const VL = std.simd.suggestVectorLength(u32) orelse 8;
        const wv = w - w % VL;
        var tail: u64 = 0;
        for (0..h) |_| {
            var rowv: @Vector(VL, u32) = @splat(0);
            var x: u32 = 0;
            while (x < wv) : (x += VL) {
                const vt: @Vector(VL, T) = row[x..][0..VL].*;
                rowv += @as(@Vector(VL, u32), vt);
            }
            sum += @reduce(.Add, @as(@Vector(VL, u64), rowv));
            while (x < w) : (x += 1) tail += row[x];
            row = row[stride..];
        }
        sum += tail;
    } else {
        const VL = std.simd.suggestVectorLength(u32) orelse 8;
        const Vu = @Vector(VL, u64);
        var accv: Vu = @splat(0);
        var tail: u64 = 0;
        for (0..h) |_| {
            var x: u32 = 0;
            while (x + VL <= w) : (x += VL) {
                const vt: @Vector(VL, T) = row[x..][0..VL].*;
                accv += @as(Vu, @intCast(vt));
            }
            while (x < w) : (x += 1) tail += row[x];
            row = row[stride..];
        }
        sum = @reduce(.Add, accv) + tail;
    }

    const avg = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(npix)) / @as(f64, peak);
    return @floatCast(avg);
}

/// log2(x) for the mask's base (inputs below the smallest normal are clamped
/// to it, as the Cephes logf the vector path used before). x = m * 2^e with m in [sqrt(1/2), sqrt(2));
/// log2(m) = (2/ln2) * atanh(s), s = (m-1)/(m+1), |s| <= 0.1716, by its odd
/// series through s^9 (truncation < 1e-9). Shorter dependency chain than
/// Cephes's degree-9 Horner on the mantissa.
inline fn log2Fast(comptime VL: usize, x_in: @Vector(VL, f32)) @Vector(VL, f32) {
    const V = @Vector(VL, f32);
    const VU = @Vector(VL, u32);
    const VI = @Vector(VL, i32);
    const x = @max(x_in, @as(V, @bitCast(@as(VU, @splat(0x00800000)))));
    const off: VU = @splat(0x3f3504f3); // bits of sqrt(1/2)
    const t = @as(VU, @bitCast(x)) -% off;
    const e: V = @floatFromInt(@as(VI, @bitCast(t)) >> @splat(23));
    const m: V = @bitCast((t & @as(VU, @splat(0x007fffff))) +% off);
    const f = m - @as(V, @splat(1.0));
    const sv = f / (f + @as(V, @splat(2.0)));
    const z = sv * sv;
    const l2e = 2.0 / std.math.ln2;
    var p: V = @splat(l2e / 9.0);
    p = simd.madd(V, p, z, @splat(l2e / 7.0));
    p = simd.madd(V, p, z, @splat(l2e / 5.0));
    p = simd.madd(V, p, z, @splat(l2e / 3.0));
    p = simd.madd(V, p, z, @splat(l2e));
    return simd.madd(V, sv, p, e);
}

/// Largest exponent passed to exp2Bounded: the float just below 127.5, since
/// exp2Bounded rounds to the nearest integer and 2^128 is inf. 2^127.5 ~
/// 2.406e38 is also where 23.1.0's Cephes expf saturated (powf gives inf).
const max_exponent: f32 = std.math.nextAfter(f32, 127.5, 0.0);

/// Exponent of the mask, temp * log2(base), capped at `max_exponent`.
inline fn maskExponent(comptime VL: usize, xv: @Vector(VL, f32), temp: @Vector(VL, f32), cb: PowConsts(VL).Base) @Vector(VL, f32) {
    const V = @Vector(VL, f32);
    const x = simd.clampV(VL, xv, @as(V, @splat(0.0)), @as(V, @splat(1.0)));
    const base = @as(V, @splat(1.0)) - (x * poly(V, x, cb));
    return @min(temp * log2Fast(VL, base), @as(V, @splat(max_exponent)));
}

const FVL: usize = std.simd.suggestVectorLength(f32) orelse 8;

/// Float mask: (1 - x*poly(x))^temp per pixel, x clamped to [0, 1], as
/// 2^(temp * log2(base)) with the approximations above (exp2Bounded returns 0
/// below -126); within ~1e-6 relative of the libm powf the Rust adg
/// reference uses.
///
/// Loop fission over L1-sized blocks: the fused per-vector chain (poly, log2,
/// exp2) is latency-bound because the ROB holds only a few iterations
/// (llvm-mca: 32 cycles/vector against a 12-cycle port bound). Stage 1 writes
/// the exponent to a stack block and stage 2 takes exp2 of it, so the same
/// ROB keeps twice the work in flight. The scalar tail runs the same vector
/// code on a padded vector, so every pixel gets the same function.
pub fn processFloat(src: []const f32, dst: []f32, temp: f32) void {
    const VL = FVL;
    const V = @Vector(VL, f32);
    const tempv: V = @splat(temp);
    const cb = PowConsts(VL).base();

    const block = 64 * VL;
    var t: [block]f32 align(common.vec_len) = undefined;
    var i: usize = 0;
    while (i + block <= dst.len) : (i += block) {
        var k: usize = 0;
        while (k < block) : (k += VL) t[k..][0..VL].* = maskExponent(VL, src[i + k ..][0..VL].*, tempv, cb);
        k = 0;
        while (k < block) : (k += VL) dst[i + k ..][0..VL].* = vcl.exp2Bounded(6, @as(V, t[k..][0..VL].*));
    }
    while (i + VL <= dst.len) : (i += VL) {
        dst[i..][0..VL].* = vcl.exp2Bounded(6, maskExponent(VL, src[i..][0..VL].*, tempv, cb));
    }
    if (i < dst.len) {
        var buf: [VL]f32 = @splat(0.0);
        const r = dst.len - i;
        @memcpy(buf[0..r], src[i..]);
        const out: [VL]f32 = vcl.exp2Bounded(6, maskExponent(VL, buf, tempv, cb));
        @memcpy(dst[i..], out[0..r]);
    }
}
