//! The FFTW 3.3.11 algorithms behind classic BM3D's DCTs, without FFTW:
//! REDFT10 (DCT-II) and REDFT01 (DCT-III) of size n, unnormalized
//! (REDFT01(REDFT10(x)) = 2n * x), computed exactly as the float build of
//! FFTW does for the plans the classic plugin gets:
//!
//! - n = 8: the `e10_8` / `e01_8` codelets;
//! - n = 1: REDFT10 is 2x, REDFT01 is the identity (FFTW drops the dimension);
//! - other n: `reodft010e-r2hc` (pre/post-processing around a real-to-
//!   halfcomplex transform of the same size) with the `r2cf_n` codelet and
//!   FFTW's twiddles (double `real_cexp` through the sqrt-n table, rounded
//!   to float).
//!
//! That is the plan FFTW_ESTIMATE and FFTW_PATIENT both pick for n <= 16.
//! Above 16 the R2HC child is `rdft-generic` for the primes 17..31 and the
//! `r2cf_20` / `r2cf_25` / `r2cf_32` codelets, which is what PATIENT picks
//! for them. The other composites (18, 21, 22, 24, 26, 27, 28, 30) get
//! Cooley-Tukey plans whose radix PATIENT chooses by timing, so even the
//! classic plugin is not reproducible there (two runs of the same script
//! differed at group_size 30); they, and sizes above 32, use `slowDct` (an
//! f64 sum rounded to float), equal to FFTW up to rounding.
//!
//! The codelets run the same scalar sequence on every lane, so any vector
//! width gives the scalar FFTW result.

const std = @import("std");
const cl = @import("bm3d_fftw_codelets.zig");

/// Sizes with an exact FFTW plan.
pub fn exact(n: u32) bool {
    return switch (n) {
        1...17, 19, 20, 23, 25, 29, 31, 32 => true,
        else => false,
    };
}

/// Sizes whose R2HC child is `rdft-generic`.
fn generic(n: u32) bool {
    return switch (n) {
        17, 19, 23, 29, 31 => true,
        else => false,
    };
}

pub const max_exact = 32;

/// Twiddle floats per size: reodft010e's 2 * (n/2 + 1), then (generic
/// sizes) the R2HC child's (n - 1)/2 rows of n - 1.
pub const tw_len = 2 * (max_exact / 2 + 1) + 15 * 30;

/// FFTW's triggen (AWAKE_SQRTN_TABLE: trigreal = double > float) for
/// exp(2 pi i m / nt), rounded to float.
fn cexp(m: i64, nt: i64) [2]f32 {
    var log2r: u6 = 0;
    var t = nt;
    while (t > 0) : (t = @divTrunc(t, 4)) log2r += 1;
    const radix: i64 = @as(i64, 1) << log2r;
    const w0 = realCexp(m & (radix - 1), nt);
    const w1 = realCexp((m >> log2r) * radix, nt);
    return .{ @floatCast(w1[0] * w0[0] - w1[1] * w0[1]), @floatCast(w1[1] * w0[0] + w1[0] * w0[1]) };
}

/// All twiddles of size n (n <= max_exact).
pub fn twiddles(n: u32, tw: *[tw_len]f32) void {
    // reodft010e: TW_COS / TW_SIN of j over 4n, j = 0 .. n/2
    for (0..n / 2 + 1) |j| {
        const c = cexp(@intCast(j), 4 * @as(i64, n));
        tw[2 * j] = c[0];
        tw[2 * j + 1] = c[1];
    }
    if (!generic(n)) return;
    // rdft-generic: TW_HALF over n, one row of (n - 1) floats per output
    // pair j + 1 = 1 .. (n - 1)/2
    const g = tw[2 * (n / 2 + 1) ..];
    var o: usize = 0;
    for (0..(n - 1) / 2) |j| {
        var i: u32 = 1;
        while (i + i < n) : (i += 1) {
            const m: i64 = @mod(@as(i64, i) * @as(i64, @intCast(j + 1)), n);
            const c = cexp(m, n);
            g[o] = c[0];
            g[o + 1] = c[1];
            o += 2;
        }
    }
}

/// rdft/generic.c apply_r2hc (odd n): Hartley pre-pass, then a dot product
/// per output pair, all in float.
fn genericR2hc(comptime n: u32, comptime T: type, tw: []const f32, x: *const [n]T, y: *[n]T) void {
    const w = tw[2 * (n / 2 + 1) ..];
    var buf: [n]T = undefined;
    var sr = x[0];
    buf[0] = x[0];
    var i: u32 = 1;
    while (i + i < n) : (i += 1) {
        const a = x[i];
        const b = x[n - i];
        buf[2 * i - 1] = a + b;
        sr += buf[2 * i - 1];
        buf[2 * i] = b - a;
    }
    y[0] = sr;
    i = 1;
    var wo: usize = 0;
    while (i + i < n) : (i += 1) {
        var rr = buf[0];
        var ri = splat(T, 0);
        var k: u32 = 1;
        while (k + k < n) : (k += 1) {
            rr += buf[2 * k - 1] * splat(T, w[wo + 2 * (k - 1)]);
            ri += buf[2 * k] * splat(T, w[wo + 2 * (k - 1) + 1]);
        }
        y[i] = rr;
        y[n - i] = ri;
        wo += n - 1;
    }
}

/// FFTW kernel/trig.c real_cexp: exp(2 pi i m / n) with octant reduction.
fn realCexp(m_in: i64, n_in: i64) [2]f64 {
    const k2pi: f64 = 6.2831853071795864769252867665590057683943388;
    var m = m_in;
    var n = n_in;
    var octant: u32 = 0;
    const quarter_n = n;
    n += n;
    n += n;
    m += m;
    m += m;
    if (m < 0) m += n;
    if (m > n - m) {
        m = n - m;
        octant |= 4;
    }
    if (m - quarter_n > 0) {
        m = m - quarter_n;
        octant |= 2;
    }
    if (m > quarter_n - m) {
        m = quarter_n - m;
        octant |= 1;
    }
    const theta = (k2pi * @as(f64, @floatFromInt(m))) / @as(f64, @floatFromInt(n));
    var c = @cos(theta);
    var s = @sin(theta);
    if (octant & 1 != 0) {
        const tt = c;
        c = s;
        s = tt;
    }
    if (octant & 2 != 0) {
        const tt = c;
        c = -s;
        s = tt;
    }
    if (octant & 4 != 0) s = -s;
    return .{ c, s };
}

fn splat(comptime T: type, v: f32) T {
    return if (T == f32) v else @splat(v);
}

fn r2hc(comptime n: u32, comptime T: type, tw: []const f32, x: *const [n]T, y: *[n]T) void {
    switch (n) {
        1 => y[0] = x[0],
        inline 2...16, 20, 25, 32 => |c| @field(cl, std.fmt.comptimePrint("r2cf_{d}", .{c}))(T, x, y),
        17, 19, 23, 29, 31 => genericR2hc(n, T, tw, x, y),
        else => @compileError("no R2HC plan"),
    }
}

/// In-place REDFT10 of x (size n, n exact), tw from `twiddles(n)`.
pub fn redft10(comptime n: u32, comptime T: type, tw: []const f32, x: *[n]T) void {
    if (n == 8) {
        const in = x.*;
        cl.e10_8(T, &in, x);
        return;
    }
    if (n == 1) {
        x[0] = splat(T, 2.0) * x[0];
        return;
    }
    // reodft010e-r2hc apply_re10
    var buf: [n]T = undefined;
    buf[0] = x[0];
    var i: u32 = 1;
    while (i < n - i) : (i += 1) {
        const k = i + i;
        buf[n - i] = x[k - 1];
        buf[i] = x[k];
    }
    if (i == n - i) buf[i] = x[n - 1];
    var h: [n]T = undefined;
    r2hc(n, T, tw, &buf, &h);
    const two = splat(T, 2.0);
    x[0] = two * h[0];
    i = 1;
    while (i < n - i) : (i += 1) {
        const a = two * h[i];
        const b = two * h[n - i];
        const wa = splat(T, tw[2 * i]);
        const wb = splat(T, tw[2 * i + 1]);
        x[i] = wa * a + wb * b;
        x[n - i] = wb * a - wa * b;
    }
    if (i == n - i) x[i] = (two * h[i]) * splat(T, tw[2 * i]);
}

/// In-place REDFT01 of x (size n, n exact).
pub fn redft01(comptime n: u32, comptime T: type, tw: []const f32, x: *[n]T) void {
    if (n == 8) {
        const in = x.*;
        cl.e01_8(T, &in, x);
        return;
    }
    if (n == 1) return;
    // reodft010e-r2hc apply_re01
    var buf: [n]T = undefined;
    buf[0] = x[0];
    var i: u32 = 1;
    while (i < n - i) : (i += 1) {
        const a = x[i];
        const b = x[n - i];
        const apb = a + b;
        const amb = a - b;
        const wa = splat(T, tw[2 * i]);
        const wb = splat(T, tw[2 * i + 1]);
        buf[i] = wa * amb + wb * apb;
        buf[n - i] = wa * apb - wb * amb;
    }
    if (i == n - i) buf[i] = (splat(T, 2.0) * x[i]) * splat(T, tw[2 * i]);
    var h: [n]T = undefined;
    r2hc(n, T, tw, &buf, &h);
    x[0] = h[0];
    i = 1;
    while (i < n - i) : (i += 1) {
        const a = h[i];
        const b = h[n - i];
        const k = i + i;
        x[k - 1] = a - b;
        x[k] = a + b;
    }
    if (i == n - i) x[n - 1] = h[i];
}

/// Runtime-size dispatch over the exact sizes; other sizes use `slowDct`.
pub fn dct(comptime forward: bool, comptime T: type, n: u32, tw: []const f32, x: []T) void {
    switch (n) {
        inline 1...17, 19, 20, 23, 25, 29, 31, 32 => |c| {
            if (forward) redft10(c, T, tw, x[0..c]) else redft01(c, T, tw, x[0..c]);
        },
        else => slowDct(forward, T, n, x),
    }
}

/// Direct O(n^2) REDFT10 / REDFT01 in f64, rounded to float: for sizes FFTW
/// plans with algorithms not ported here (agrees to rounding, not bits).
pub fn slowDct(comptime forward: bool, comptime T: type, n: u32, x: []T) void {
    const L = if (T == f32) 1 else @typeInfo(T).vector.len;
    var tmp: [256]T = undefined;
    std.debug.assert(n <= tmp.len);
    const nf: f64 = @floatFromInt(n);
    for (0..n) |k| {
        var acc: [L]f64 = @splat(0);
        for (0..n) |j| {
            const jf: f64 = @floatFromInt(j);
            const kf: f64 = @floatFromInt(k);
            const c = if (forward)
                2.0 * @cos(std.math.pi * (jf + 0.5) * kf / nf)
            else if (j == 0) 1.0 else 2.0 * @cos(std.math.pi * jf * (kf + 0.5) / nf);
            if (T == f32) {
                acc[0] += c * @as(f64, x[j]);
            } else {
                inline for (0..L) |l| acc[l] += c * @as(f64, x[j][l]);
            }
        }
        if (T == f32) {
            tmp[k] = @floatCast(acc[0]);
        } else {
            inline for (0..L) |l| tmp[k][l] = @floatCast(acc[l]);
        }
    }
    @memcpy(x[0..n], tmp[0..n]);
}
