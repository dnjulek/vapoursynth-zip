const std = @import("std");

pub fn atan(x: anytype) @TypeOf(x) {
    // Port of VCL2 vectormath_trig.h: atan_f

    const F32V = @TypeOf(x);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const BoolV = @Vector(vec_len, bool);

    const P3atanf: f32 = 8.05374449538E-2;
    const P2atanf: f32 = -1.38776856032E-1;
    const P1atanf: f32 = 1.99777106478E-1;
    const P0atanf: f32 = -3.33329491539E-1;

    const vm_pi_2: f32 = std.math.pi * 0.5;
    const vm_pi_4: f32 = std.math.pi * 0.25;
    const vm_sqrt2: f32 = @sqrt(2.0);

    const t: F32V = @abs(x);
    const notsmal: BoolV = t >= @as(F32V, @splat(vm_sqrt2 - 1.0));
    const notbig: BoolV = t <= @as(F32V, @splat(vm_sqrt2 + 1.0));

    var s: F32V = @select(f32, notbig, @as(F32V, @splat(vm_pi_4)), @as(F32V, @splat(vm_pi_2)));
    s = @select(f32, notsmal, s, @as(F32V, @splat(@as(f32, 0.0))));

    var a: F32V = @select(f32, notbig, t, @as(F32V, @splat(@as(f32, 0.0))));
    a += @select(f32, notsmal, @as(F32V, @splat(@as(f32, -1.0))), @as(F32V, @splat(@as(f32, 0.0))));

    var b: F32V = @select(f32, notbig, @as(F32V, @splat(@as(f32, 1.0))), @as(F32V, @splat(@as(f32, 0.0))));
    b += @select(f32, notsmal, t, @as(F32V, @splat(@as(f32, 0.0))));

    const z: F32V = a / b;
    const zz: F32V = z * z;

    var re: F32V = polynomial_3(zz, P0atanf, P1atanf, P2atanf, P3atanf);
    re = fma(re, zz * z, z) + s;
    return copysign_f32v(re, x);
}

pub fn cbrt(x: anytype) @TypeOf(x) {
    // Port of VCL2 vectormath_exp.h: cbrt_f

    const F32V = @TypeOf(x);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const U32V = @Vector(vec_len, u32);
    const U5V = @Vector(vec_len, u5);
    const BoolV = @Vector(vec_len, bool);
    const shift23: U5V = @splat(23);

    const iter: comptime_int = 4;
    const one_third: f32 = 1.0 / 3.0;
    const four_third: f32 = 4.0 / 3.0;

    const v_one_third: F32V = @splat(one_third);
    const v_four_third: F32V = @splat(four_third);
    const v_zero: F32V = @splat(@as(f32, 0.0));

    const q1: U32V = @splat(0x5480_0000);
    const q2: U32V = @splat(0x002A_AAAA);
    const q3: U32V = @splat(0x0080_0000);

    const xa: F32V = @abs(x);
    const xa3: F32V = v_one_third * xa;

    const m1: U32V = @bitCast(xa);
    const m2: U32V = q1 - ((m1 >> shift23) * q2);
    var a: F32V = @bitCast(m2);
    const underflow: BoolV = m1 <= q3;

    inline for (0..(iter - 1)) |_| {
        const a2: F32V = a * a;
        a = (v_four_third * a) - (xa3 * (a2 * a2));
    }

    const a2: F32V = a * a;
    a = a + (v_one_third * (a - (xa * (a2 * a2))));
    a = (a * a) * x;

    a = @select(f32, underflow, v_zero, a);
    return a;
}

pub fn pow(x0: anytype, y: anytype) @TypeOf(x0) {
    // Port of VCL2 vectormath_exp.h: pow_template_f

    const F32V = @TypeOf(x0);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const U32V = @Vector(vec_len, u32);
    const I32V = @Vector(vec_len, i32);
    const U5V = @Vector(vec_len, u5);
    const BoolV = @Vector(vec_len, bool);
    const shift23: U5V = @splat(23);

    const ln2f_hi: f32 = 0.693359375;
    const ln2f_lo: f32 = -2.12194440e-4;
    const ln2: f32 = 0.6931471805599453;
    const log2e: f32 = 1.4426950408889634;
    const sqrt2_half: f32 = 0.7071067811865476;

    const P0logf: f32 = 3.3333331174E-1;
    const P1logf: f32 = -2.4999993993E-1;
    const P2logf: f32 = 2.0000714765E-1;
    const P3logf: f32 = -1.6668057665E-1;
    const P4logf: f32 = 1.4249322787E-1;
    const P5logf: f32 = -1.2420140846E-1;
    const P6logf: f32 = 1.1676998740E-1;
    const P7logf: f32 = -1.1514610310E-1;
    const P8logf: f32 = 7.0376836292E-2;

    const p2expf: f32 = 1.0 / 2.0;
    const p3expf: f32 = 1.0 / 6.0;
    const p4expf: f32 = 1.0 / 24.0;
    const p5expf: f32 = 1.0 / 120.0;
    const p6expf: f32 = 1.0 / 720.0;
    const p7expf: f32 = 1.0 / 5040.0;

    const v_ln2f_hi: F32V = @splat(ln2f_hi);
    const v_ln2f_lo: F32V = @splat(ln2f_lo);
    const v_ln2: F32V = @splat(ln2);
    const v_log2e: F32V = @splat(log2e);
    const v_half: F32V = @splat(@as(f32, 0.5));
    const v_one: F32V = @splat(@as(f32, 1.0));
    const v_sqrt2_half: F32V = @splat(sqrt2_half);

    const x1: F32V = @abs(x0);
    var x: F32V = fraction_2(x1);
    const blend: BoolV = x > v_sqrt2_half;
    x = @select(f32, blend, x, x + x);
    x -= v_one;

    const x2: F32V = x * x;
    var lg1: F32V = polynomial_8(x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf);
    lg1 *= (x2 * x);

    var ef: F32V = exponent_f(x1);
    ef = @select(f32, blend, ef + v_one, ef);

    const e1: F32V = @round(ef * y);
    const yr: F32V = fma(ef, y, -e1);

    const lg: F32V = fma(v_half, -x2, x) + lg1;
    const x2err: F32V = fma(v_half * x, x, v_half * -x2);
    const lgerr: F32V = fma(v_half, x2, lg - x) - lg1;

    const e2: F32V = @round(lg * y * v_log2e);
    var v: F32V = fma(lg, y, -e2 * v_ln2f_hi);
    v = fma(-e2, v_ln2f_lo, v);

    const correction: F32V = fma(lgerr + x2err, y, -yr * v_ln2);
    v -= correction;

    x = v;
    const e3: F32V = @round(x * v_log2e);
    x = fma(-e3, v_ln2, x);

    const x2e: F32V = x * x;
    var z: F32V = polynomial_5(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf);
    z = z * x2e + x + v_one;

    const ee: F32V = e1 + e2 + e3;
    const ei: I32V = @intFromFloat(@round(ee));

    // const z_abs_bits: U32V = @bitCast(@abs(z));
    // const ej: I32V = ei + @as(I32V, @intCast(z_abs_bits >> shift23));
    // const overflow: BoolV = (ej >= @as(I32V, @splat(@as(i32, 0x0FF)))) | (ee > @as(F32V, @splat(@as(f32, 300.0))));
    // const underflow: BoolV = (ej <= @as(I32V, @splat(@as(i32, 0x000)))) | (ee < @as(F32V, @splat(@as(f32, -300.0))));

    const z_bits0: U32V = @bitCast(z);
    const ei_u: U32V = @as(U32V, @bitCast(ei));
    const z_bits: U32V = z_bits0 +% (ei_u << shift23);
    z = @bitCast(z_bits);

    // const v_zero: F32V = @splat(@as(f32, 0.0));
    // const v_inf: F32V = @bitCast(@as(U32V, @splat(0x7F80_0000)));
    // z = @select(f32, underflow, v_zero, z);
    // z = @select(f32, overflow, v_inf, z);
    return z;
}

fn fma(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    const F32V = @TypeOf(a);
    return @mulAdd(F32V, a, b, c);
}

fn copysign_f32v(magnitude: anytype, sign_source: @TypeOf(magnitude)) @TypeOf(magnitude) {
    const F32V = @TypeOf(magnitude);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const U32V = @Vector(vec_len, u32);

    const mag_bits: U32V = @bitCast(magnitude);
    const sign_bits: U32V = @bitCast(sign_source);
    const res_bits: U32V = (mag_bits & @as(U32V, @splat(0x7FFF_FFFF))) | (sign_bits & @as(U32V, @splat(0x8000_0000)));
    return @bitCast(res_bits);
}

fn polynomial_3(x: anytype, c0: f32, c1: f32, c2: f32, c3: f32) @TypeOf(x) {
    const F32V = @TypeOf(x);
    const x2 = x * x;
    return fma(
        fma(@as(F32V, @splat(c3)), x, @as(F32V, @splat(c2))),
        x2,
        fma(@as(F32V, @splat(c1)), x, @as(F32V, @splat(c0))),
    );
}

fn polynomial_5(x: anytype, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32) @TypeOf(x) {
    const F32V = @TypeOf(x);
    const x2 = x * x;
    const x4 = x2 * x2;
    return fma(
        fma(@as(F32V, @splat(c3)), x, @as(F32V, @splat(c2))),
        x2,
        fma(
            fma(@as(F32V, @splat(c5)), x, @as(F32V, @splat(c4))),
            x4,
            fma(@as(F32V, @splat(c1)), x, @as(F32V, @splat(c0))),
        ),
    );
}

fn polynomial_8(x: anytype, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32, c8: f32) @TypeOf(x) {
    const F32V = @TypeOf(x);
    const x2 = x * x;
    const x4 = x2 * x2;
    const x8 = x4 * x4;
    return fma(
        fma(fma(@as(F32V, @splat(c7)), x, @as(F32V, @splat(c6))), x2, fma(@as(F32V, @splat(c5)), x, @as(F32V, @splat(c4)))),
        x4,
        fma(fma(@as(F32V, @splat(c3)), x, @as(F32V, @splat(c2))), x2, fma(@as(F32V, @splat(c1)), x, @as(F32V, @splat(c0))) + (@as(F32V, @splat(c8)) * x8)),
    );
}

fn fraction_2(a: anytype) @TypeOf(a) {
    const F32V = @TypeOf(a);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const U32V = @Vector(vec_len, u32);

    const bits: U32V = @bitCast(a);
    const mant: U32V = bits & @as(U32V, @splat(0x007F_FFFF));
    const half_exp: U32V = mant | @as(U32V, @splat(0x3F00_0000));
    return @bitCast(half_exp);
}

fn exponent_f(a: anytype) @TypeOf(a) {
    const F32V = @TypeOf(a);
    const vec_len: comptime_int = @typeInfo(F32V).vector.len;
    const U32V = @Vector(vec_len, u32);
    const I32V = @Vector(vec_len, i32);
    const U5V = @Vector(vec_len, u5);
    const shift23: U5V = @splat(23);

    const bits: U32V = @bitCast(a);
    const exp_u: U32V = (bits >> shift23) & @as(U32V, @splat(0xFF));
    const exp_i: I32V = @as(I32V, @intCast(exp_u)) - @as(I32V, @splat(@as(i32, 127)));
    return @floatFromInt(exp_i);
}
