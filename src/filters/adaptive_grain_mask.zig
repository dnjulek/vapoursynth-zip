const std = @import("std");
const simd = @import("simd.zig");
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

const pow_consts = [_]f32{
    7.0376836292e-2, -1.1514610310e-1, 1.1676998740e-1, -1.2420140846e-1,
    1.4249322787e-1, -1.6668057665e-1, 2.0000714765e-1, -2.4999993993e-1,
    3.3333331174e-1, 1.9875691500e-4,  1.3981999507e-3, 8.3334519073e-3,
    4.1665795894e-2, 1.6666665459e-1,  5.0000001201e-1, 1.44269504088896341,

    C4,              C3,               C2,              C1,
    C0,
};

fn PowConsts(comptime VL: usize) type {
    return struct {
        const rows: [pow_consts.len][VL]f32 align(64) = blk: {
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
    return @mulAdd(V, x, @mulAdd(V, x, @mulAdd(V, x, @mulAdd(V, x, P.get(cb, C4), P.get(cb, C3)), P.get(cb, C2)), P.get(cb, C1)), P.get(cb, C0));
}

inline fn polyScalar(x: f32) f32 {
    return @mulAdd(f32, x, @mulAdd(f32, x, @mulAdd(f32, x, @mulAdd(f32, x, C4, C3), C2), C1), C0);
}

pub fn getMaskValueScalar(x: f32, temp: f32) f32 {
    const base = 1.0 - (x * polyScalar(x));
    return powf(base, temp);
}

pub fn buildLut(comptime T: type, lut: *[256]T, temp: f32, peak: f32) void {
    const type_max: f32 = @floatFromInt(std.math.maxInt(T));
    for (lut, 0..) |*e, i| {
        const x: f32 = @as(f32, @floatFromInt(i)) / 255.0;
        const v = getMaskValueScalar(x, temp);
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
    if (comptime simd.use_avx2) {
        const chunks = simd.lutChunksU8(lut);
        var i: usize = 0;
        const n = dst.len - dst.len % 32;
        while (i < n) : (i += 32) {
            dst[i..][0..32].* = simd.lutU8x32(&chunks, src[i..][0..32].*);
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
    if (T == u8 and simd.use_avx2) {
        const zero32: @Vector(32, u8) = @splat(0);
        var acc: @Vector(4, u64) = @splat(0);
        var tail: u64 = 0;
        for (0..h) |_| {
            var x: u32 = 0;
            while (x + 32 <= w) : (x += 32) {
                acc += simd.sadU8x32(row[x..][0..32].*, zero32);
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

fn logPs(comptime VL: usize, x_in: @Vector(VL, f32), cb: PowConsts(VL).Base) @Vector(VL, f32) {
    const P = PowConsts(VL);
    const V = @Vector(VL, f32);
    const VI = @Vector(VL, i32);
    const VU = @Vector(VL, u32);
    const one: V = @splat(1.0);
    const min_norm_pos: V = @bitCast(@as(VU, @splat(0x00800000)));
    var x = @max(x_in, min_norm_pos);
    const xu: VU = @bitCast(x);

    var e: V = @floatFromInt(@as(VI, @intCast(xu >> @as(@Vector(VL, u5), @splat(23)))) - @as(VI, @splat(0x7f)));
    e += one;
    const mant: VU = (xu & @as(VU, @splat(0x807fffff))) | @as(VU, @splat(0x3f000000));
    x = @bitCast(mant);

    const sqrthf: V = @splat(0.707106781186547524);
    const mask = x < sqrthf;
    const tmp = @select(f32, mask, x, @as(V, @splat(0.0)));
    x = x - one;
    e = e - @select(f32, mask, one, @as(V, @splat(0.0)));
    x = x + tmp;

    const z = x * x;
    var y: V = P.get(cb, 7.0376836292e-2);
    y = @mulAdd(V, y, x, P.get(cb, -1.1514610310e-1));
    y = @mulAdd(V, y, x, P.get(cb, 1.1676998740e-1));
    y = @mulAdd(V, y, x, P.get(cb, -1.2420140846e-1));
    y = @mulAdd(V, y, x, P.get(cb, 1.4249322787e-1));
    y = @mulAdd(V, y, x, P.get(cb, -1.6668057665e-1));
    y = @mulAdd(V, y, x, P.get(cb, 2.0000714765e-1));
    y = @mulAdd(V, y, x, P.get(cb, -2.4999993993e-1));
    y = @mulAdd(V, y, x, P.get(cb, 3.3333331174e-1));
    y = y * x;
    y = y * z;

    y = @mulAdd(V, e, @as(V, @splat(-2.12194440e-4)), y);
    y = y - (z * @as(V, @splat(0.5)));
    x = x + y;
    x = @mulAdd(V, e, @as(V, @splat(0.693359375)), x);
    return x;
}

fn expPs(comptime VL: usize, x_in: @Vector(VL, f32), cb: PowConsts(VL).Base) @Vector(VL, f32) {
    const P = PowConsts(VL);
    const V = @Vector(VL, f32);
    const VI = @Vector(VL, i32);
    const VU = @Vector(VL, u32);

    const one: V = @splat(1.0);
    var x = @min(x_in, @as(V, @splat(88.3762626647949)));
    x = @max(x, @as(V, @splat(-88.3762626647949)));
    const fx = @floor(@mulAdd(V, x, P.get(cb, 1.44269504088896341), @as(V, @splat(0.5))));

    x = x - (fx * @as(V, @splat(0.693359375)));
    x = x - (fx * @as(V, @splat(-2.12194440e-4)));

    const z = x * x;
    var y: V = P.get(cb, 1.9875691500e-4);
    y = @mulAdd(V, y, x, P.get(cb, 1.3981999507e-3));
    y = @mulAdd(V, y, x, P.get(cb, 8.3334519073e-3));
    y = @mulAdd(V, y, x, P.get(cb, 4.1665795894e-2));
    y = @mulAdd(V, y, x, P.get(cb, 1.6666665459e-1));
    y = @mulAdd(V, y, x, P.get(cb, 5.0000001201e-1));
    y = @mulAdd(V, y, z, x) + one;

    const n: VI = @intFromFloat(fx);
    const pow2: V = @bitCast((@as(VU, @intCast(n + @as(VI, @splat(0x7f)))) << @as(@Vector(VL, u5), @splat(23))));
    return y * pow2;
}

fn powPs(comptime VL: usize, base: @Vector(VL, f32), exp: @Vector(VL, f32), cb: PowConsts(VL).Base) @Vector(VL, f32) {
    return expPs(VL, exp * logPs(VL, base, cb), cb);
}

const FVL: usize = std.simd.suggestVectorLength(f32) orelse 8;
pub fn processFloat(src: []const f32, dst: []f32, temp: f32) void {
    const VL = FVL;
    const V = @Vector(VL, f32);
    const one: V = @splat(1.0);
    const zero: V = @splat(0.0);
    const tempv: V = @splat(temp);

    const cb = PowConsts(VL).base();
    var i: usize = 0;
    while (i + VL <= dst.len) : (i += VL) {
        const x = simd.clampV(VL, src[i..][0..VL].*, zero, one);
        const base = one - (x * poly(V, x, cb));
        dst[i..][0..VL].* = powPs(VL, base, tempv, cb);
    }

    while (i < dst.len) : (i += 1) {
        const x = @min(@max(src[i], 0.0), 1.0);
        dst[i] = getMaskValueScalar(x, temp);
    }
}
