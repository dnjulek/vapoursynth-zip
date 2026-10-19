//! Shared SIMD codegen helpers.
//!
//! Layout: gates → generic asm emitters → opacity barriers (launders) →
//! width-generic f32 lane ops (even/odd/interleave, clamp, register
//! transposes) → fixed-shape integer helpers. Every asm path keeps its
//! portable twin as the `else` branch — the portable code is the semantic
//! reference and ships on baseline/aarch64/macos.

const std = @import("std");
const builtin = @import("builtin");

// ---------------------------------------------------------------------------
// Instruction-set gates
// ---------------------------------------------------------------------------

pub const use_avx = builtin.target.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx);

pub const use_avx2 = builtin.target.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx2);

/// 512-bit f32/i32 regime (vunpck/vshuff32x4/vpermt2ps/vmaxps at zmm width are
/// all AVX-512F). znver4/znver5 builds land here.
pub const use_avx512f = builtin.target.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512f);

/// 512-bit byte/word ops (vpmaddwd/vpunpck*wd/vpackssdw at zmm). znver4+, Skylake-X+.
pub const use_avx512bw = use_avx512f and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512bw);

/// vpermi2b/vpermb (byte permutes). znver4+, Ice Lake+ (not Skylake-X).
pub const use_avx512vbmi = use_avx512bw and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .avx512vbmi);

// ---------------------------------------------------------------------------
// Compiler workarounds
// ---------------------------------------------------------------------------

/// Vector `@intCast` from signed lanes to narrower unsigned lanes (i32->u8,
/// i32->u16, i16->u8, ...), routed through a same-width unsigned `@bitCast`.
/// Works around a Zig 0.17.0 miscompile where a plain `@intCast` clamps at
/// the *signed* max of the destination width (255 -> 127, 65535 -> 32767),
/// in every optimize mode. Negative lanes still trip the range check in safe
/// builds, since they bitcast to out-of-range unsigned values.
pub inline fn intCastSignedToUnsigned(comptime D: type, v: anytype) D {
    const info = @typeInfo(@TypeOf(v)).vector;
    const U = @Vector(info.len, @Int(.unsigned, @bitSizeOf(info.child)));
    return @intCast(@as(U, @bitCast(v)));
}

/// True when the f32 lane helpers below (evenLanesF32/oddLanesF32/
/// interleaveF32/transposeF32Reg) have a pinned-asm path at width `n` —
/// callers use this to decide vector-vs-scalar loop shapes.
pub inline fn f32LanesAccelerated(comptime n: comptime_int) bool {
    return (n == 8 and use_avx2) or (n == 16 and use_avx512f);
}

// ---------------------------------------------------------------------------
// Generic inline-asm emitters. AT&T operand order (src2, src1, dst); `x`
// constraints allocate xmm/ymm0-15 (SSE/AVX/AVX2 encodable), `v` constraints
// allocate the full EVEX file (xmm/ymm/zmm0-31) and are only used behind
// use_avx512f. Non-volatile, no clobbers, pure value→value: LLVM can
// CSE/hoist/schedule them but cannot rewrite them.
// ---------------------------------------------------------------------------

inline fn xop2(comptime mnem: []const u8, comptime Out: type, a: anytype, b: anytype) Out {
    return asm (mnem ++ " %[b], %[a], %[out]"
        : [out] "=x" (-> Out),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

inline fn xop2i(comptime mnem: []const u8, comptime imm: u8, comptime Out: type, a: anytype, b: anytype) Out {
    return asm (std.fmt.comptimePrint("{s} ${d}, %[b], %[a], %[out]", .{ mnem, imm })
        : [out] "=x" (-> Out),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

inline fn xop1i(comptime mnem: []const u8, comptime imm: u8, comptime Out: type, a: anytype) Out {
    return asm (std.fmt.comptimePrint("{s} ${d}, %[a], %[out]", .{ mnem, imm })
        : [out] "=x" (-> Out),
        : [a] "x" (a),
    );
}

inline fn zop2(comptime mnem: []const u8, comptime Out: type, a: anytype, b: anytype) Out {
    return asm (mnem ++ " %[b], %[a], %[out]"
        : [out] "=v" (-> Out),
        : [a] "v" (a),
          [b] "v" (b),
    );
}

inline fn zop2i(comptime mnem: []const u8, comptime imm: u8, comptime Out: type, a: anytype, b: anytype) Out {
    return asm (std.fmt.comptimePrint("{s} ${d}, %[b], %[a], %[out]", .{ mnem, imm })
        : [out] "=v" (-> Out),
        : [a] "v" (a),
          [b] "v" (b),
    );
}

/// dst = permute of concat(a, b) by comptime lane indices (0..31), via
/// vpermt2ps: `a` rides in as the tied first table operand and is replaced by
/// the result; `idx` materializes from the constant pool once per loop.
inline fn zpermt2ps(comptime n: comptime_int, comptime idx: [n]i32, a: @Vector(n, f32), b: @Vector(n, f32)) @Vector(n, f32) {
    var iv: @Vector(n, i32) = undefined;
    inline for (0..n) |i| iv[i] = idx[i];
    return asm ("vpermt2ps %[b], %[idx], %[out]"
        : [out] "=v" (-> @Vector(n, f32)),
        : [a] "0" (a),
          [idx] "v" (iv),
          [b] "v" (b),
    );
}

// ---------------------------------------------------------------------------
// Opacity barriers (zero-instruction launders)
// ---------------------------------------------------------------------------

/// Unaligned vector load the optimizer cannot fold into neighbouring loads.
/// LLVM fuses overlapping loads at compile-time-known small offsets (sliding
/// window taps) into aligned-load + shuffle chains — 2x the instructions,
/// shuffle-port-bound on Zen 3. The empty asm "launders" the pointer: zero
/// instructions, but the result is opaque to alias/address analysis, so
/// neighbouring overlapping loads can't be combined. The load itself stays an
/// ordinary load, correctly ordered against stores (the laundered pointer may
/// alias anything). Only use for loads whose relative offsets are
/// comptime-known — runtime-distance loads can't be fused anyway, and
/// laundering them just costs extra address arithmetic. Lane-count agnostic
/// (works unchanged at zmm widths — verified in the znver4 disassembly).
pub inline fn loaduOpaque(comptime T: type, comptime n: comptime_int, p: *const [n]T) @Vector(n, T) {
    if (comptime use_avx) {
        const q = asm (""
            : [out] "=r" (-> *const [n]T),
            : [in] "0" (p),
        );
        return q.*;
    }
    return p.*;
}

/// Launder a pointer so the address arithmetic that produced it cannot be
/// re-materialized elsewhere by LLVM. Use when a cheap-looking but expensive
/// computation (e.g. an integer modulo) feeds an address and LLVM re-sinks it
/// into a consumer's inner loop after inlining (measured in boxblur_runtime's
/// ring buffer: a loop-invariant `%` became an unpipelined 32-bit div per
/// vector block). Zero instructions; the empty asm is just an optimization
/// barrier on the pointer value.
pub inline fn opaquePtr(comptime T: type, p: [*]T) [*]T {
    if (comptime builtin.target.cpu.arch == .x86_64) {
        return asm (""
            : [out] "=r" (-> [*]T),
            : [in] "0" (p),
        );
    }
    return p;
}

/// Launder an f32 vector value so instcombine cannot look through it. Use
/// when LLVM's trunc-through-select combine would hoist a narrowing
/// conversion above @select chains, turning cheap f32-domain vblendvps into
/// narrow-payload blends plus per-select mask repacking ([narrow-select]
/// class; measured in limit_filter's f16 store path: 2x vblendvps became
/// 2x vextractf128 + 2x vpackssdw + 2x vpblendvb). Zero instructions; the
/// empty asm keeps the value in-register but opaque, so the conversion stays
/// where the source put it. Value identity — bit-exact by construction; the
/// portable twin is the identity return.
pub inline fn opaqueF32V(comptime n: comptime_int, v: @Vector(n, f32)) @Vector(n, f32) {
    if (comptime use_avx and n * 4 <= 32) {
        return asm (""
            : [out] "=x" (-> @Vector(n, f32)),
            : [in] "0" (v),
        );
    }
    if (comptime use_avx512f and n * 4 == 64) {
        return asm (""
            : [out] "=v" (-> @Vector(n, f32)),
            : [in] "0" (v),
        );
    }
    return v;
}

// ---------------------------------------------------------------------------
// f32 lane operations (width-generic: 8-lane AVX2 asm, 16-lane AVX-512 asm,
// portable @shuffle otherwise)
// ---------------------------------------------------------------------------

/// Even/odd lane deinterleave of 2n consecutive f32 held as two n-lane
/// vectors. Expressing this with @shuffle is correct but lets LLVM's vector
/// combiner trace the demanded lanes back into the feeding loads and shred
/// each plain load into partial loads + blends, rebuilt once per @shuffle
/// (measured in ssimulacra2.downscale at both 8 and 16 lanes) — the asm is
/// opaque to it. 8 lanes: vshufps+vpermpd pair each; 16 lanes: one vpermt2ps
/// each. Pure lane permutation: bit-exact.
pub inline fn evenLanesF32(comptime n: comptime_int, a: @Vector(n, f32), b: @Vector(n, f32)) @Vector(n, f32) {
    if (comptime n == 8 and use_avx2) {
        return xop1i("vpermpd", 0xd8, F8, xop2i("vshufps", 0x88, F8, a, b));
    }
    if (comptime n == 16 and use_avx512f) {
        return zpermt2ps(16, strideIdx(16, 0, 2), a, b);
    }
    return @shuffle(f32, a, b, evenMask(n));
}

pub inline fn oddLanesF32(comptime n: comptime_int, a: @Vector(n, f32), b: @Vector(n, f32)) @Vector(n, f32) {
    if (comptime n == 8 and use_avx2) {
        return xop1i("vpermpd", 0xd8, F8, xop2i("vshufps", 0xdd, F8, a, b));
    }
    if (comptime n == 16 and use_avx512f) {
        return zpermt2ps(16, strideIdx(16, 1, 2), a, b);
    }
    return @shuffle(f32, a, b, oddMask(n));
}

/// Interleave n even/odd f32 lanes into 2n consecutive f32 (two vectors):
/// {e0 o0 e1 o1 ...} — the store-side inverse of the deinterleave above.
/// The 8-lane @shuffle form lowers cleanly (in-lane unpck; verified in
/// mosquito_nr); the 16-lane form gets vpermt2ps so the combiner cannot trace
/// it into the surrounding loads/stores.
pub inline fn interleaveF32(comptime n: comptime_int, e: @Vector(n, f32), o: @Vector(n, f32)) [2]@Vector(n, f32) {
    if (comptime n == 16 and use_avx512f) {
        return .{
            zpermt2ps(16, zipIdx(16, 0), e, o),
            zpermt2ps(16, zipIdx(16, 8), e, o),
        };
    }
    return .{
        @shuffle(f32, e, o, zipMask(n, 0)),
        @shuffle(f32, e, o, zipMask(n, n / 2)),
    };
}

/// Clamp `v` into [min_v, max_v] as bare vmaxps + vminps. The portable
/// @min(@max(min_v, v), max_v) with runtime bounds lowers through a two-sided
/// NaN fixup — a hoisted vcmpunordps of the min splat plus a per-vector
/// vcmpunordps + 2 vblendvps — because llvm.maxnum/minnum must special-case a
/// NaN bound (measured in limiter.LimiterRT(f32/f16).getFrame: 5 FP-ALU
/// ops/vector vs 2; the comptime-bounds Limiter(...) variants prove bounds
/// non-NaN and get the bare form). An `assume(!isNan)` from std.debug.assert
/// does NOT reach the late SelectionDAG minnum/maxnum lowering (verified: the
/// blends remained), so the bare form is pinned here in asm.
/// PRECONDITION: min_v and max_v contain no NaN lane (limiterCreate rejects
/// NaN bounds; callers std.debug.assert it). Under that precondition this is
/// bit-identical to the portable twin for every input, NaN pixels included:
/// vmaxps takes v as src1, and VMAXPS returns src2 when either operand is NaN,
/// so a NaN v yields min_v — exactly matching maxnum(min_v, v). The vmaxps
/// result is therefore always non-NaN, so both vminps operands are non-NaN and
/// hardware min == mathematical min == minnum(t, max_v). The portable twin
/// below is the semantic reference.
pub inline fn clampV(comptime n: comptime_int, v: @Vector(n, f32), min_v: @Vector(n, f32), max_v: @Vector(n, f32)) @Vector(n, f32) {
    const V = @Vector(n, f32);
    if (comptime use_avx and n * 4 <= 32) {
        // Intel `vmaxps t, v, min_v`: src1 = v, src2 = min_v.
        return xop2("vminps", V, xop2("vmaxps", V, v, min_v), max_v);
    }
    if (comptime use_avx512f and n * 4 == 64) {
        return zop2("vminps", V, zop2("vmaxps", V, v, min_v), max_v);
    }
    return @min(@max(min_v, v), max_v);
}

// ---------------------------------------------------------------------------
// In-register f32 transposes
// ---------------------------------------------------------------------------

/// In-register n×n f32 transpose of n row vectors into n column vectors.
/// Pinned as asm for n=8 (AVX) and n=16 (AVX-512F) because LLVM's vector
/// combiner lowers the equivalent @shuffle networks through per-element stack
/// traffic — measured at ~4 instr/element on AVX2 and 3.98
/// instr/element with ~450 spills at 16 lanes on znver4,
/// vs ~1 (8 lanes, 24 shuffles) / ~0.25 (16 lanes, 64 shuffles) for the asm.
/// Pure lane permutation: bit-exact. The butterfly fallback is the portable
/// semantic reference for any power-of-two n.
pub inline fn transposeF32Reg(comptime n: comptime_int, r: [n]@Vector(n, f32)) [n]@Vector(n, f32) {
    if (comptime n == 8 and use_avx) return transposeF32x8(r);
    if (comptime n == 16 and use_avx512f) return transposeF32x16(r);
    if (comptime n == 4) return transposeF32x4(r);
    return transposeButterfly(f32, n, r);
}

const F8 = @Vector(8, f32);
const F16 = @Vector(16, f32);

/// Classic 3-stage 8x8 AVX transpose (unpck / shufps / perm2f128), 24 shuffle
/// instructions total. (Moved here from eedi3.zig, where it was validated.)
inline fn transposeF32x8(r: [8]F8) [8]F8 {
    const t0 = xop2("vunpcklps", F8, r[0], r[1]); // a0 b0 a1 b1 | a4 b4 a5 b5
    const t1 = xop2("vunpckhps", F8, r[0], r[1]);
    const t2 = xop2("vunpcklps", F8, r[2], r[3]);
    const t3 = xop2("vunpckhps", F8, r[2], r[3]);
    const t4 = xop2("vunpcklps", F8, r[4], r[5]);
    const t5 = xop2("vunpckhps", F8, r[4], r[5]);
    const t6 = xop2("vunpcklps", F8, r[6], r[7]);
    const t7 = xop2("vunpckhps", F8, r[6], r[7]);
    const q0 = xop2i("vshufps", 0x44, F8, t0, t2); // c0(r0..3) | c4(r0..3)
    const q1 = xop2i("vshufps", 0xEE, F8, t0, t2);
    const q2 = xop2i("vshufps", 0x44, F8, t1, t3);
    const q3 = xop2i("vshufps", 0xEE, F8, t1, t3);
    const q4 = xop2i("vshufps", 0x44, F8, t4, t6);
    const q5 = xop2i("vshufps", 0xEE, F8, t4, t6);
    const q6 = xop2i("vshufps", 0x44, F8, t5, t7);
    const q7 = xop2i("vshufps", 0xEE, F8, t5, t7);
    return .{
        xop2i("vperm2f128", 0x20, F8, q0, q4),
        xop2i("vperm2f128", 0x20, F8, q1, q5),
        xop2i("vperm2f128", 0x20, F8, q2, q6),
        xop2i("vperm2f128", 0x20, F8, q3, q7),
        xop2i("vperm2f128", 0x31, F8, q0, q4),
        xop2i("vperm2f128", 0x31, F8, q1, q5),
        xop2i("vperm2f128", 0x31, F8, q2, q6),
        xop2i("vperm2f128", 0x31, F8, q3, q7),
    };
}

/// 4-stage 16x16 AVX-512 transpose, 64 shuffle instructions (0.25/element):
/// unpcklps/unpckhps pairs → unpcklpd/unpckhpd quads (u[4g+j] = column 4l+j of
/// row group g in 128-bit chunk l) → two vshuff32x4 rounds gathering the
/// 128-bit chunks first across row-group pairs (imm 0x44/0xEE), then across
/// halves (imm 0x88/0xDD).
inline fn transposeF32x16(r: [16]F16) [16]F16 {
    var t: [16]F16 = undefined;
    comptime var k: u32 = 0;
    inline while (k < 8) : (k += 1) {
        t[2 * k] = zop2("vunpcklps", F16, r[2 * k], r[2 * k + 1]);
        t[2 * k + 1] = zop2("vunpckhps", F16, r[2 * k], r[2 * k + 1]);
    }
    var u: [16]F16 = undefined;
    comptime var g: u32 = 0;
    inline while (g < 4) : (g += 1) {
        u[4 * g + 0] = zop2("vunpcklpd", F16, t[4 * g + 0], t[4 * g + 2]);
        u[4 * g + 1] = zop2("vunpckhpd", F16, t[4 * g + 0], t[4 * g + 2]);
        u[4 * g + 2] = zop2("vunpcklpd", F16, t[4 * g + 1], t[4 * g + 3]);
        u[4 * g + 3] = zop2("vunpckhpd", F16, t[4 * g + 1], t[4 * g + 3]);
    }
    var out: [16]F16 = undefined;
    comptime var j: u32 = 0;
    inline while (j < 4) : (j += 1) {
        const va = zop2i("vshuff32x4", 0x44, F16, u[j], u[4 + j]); //  cols j, j+4  of rows 0-7
        const vb = zop2i("vshuff32x4", 0xEE, F16, u[j], u[4 + j]); //  cols j+8, j+12 of rows 0-7
        const vc = zop2i("vshuff32x4", 0x44, F16, u[8 + j], u[12 + j]); // rows 8-15
        const vd = zop2i("vshuff32x4", 0xEE, F16, u[8 + j], u[12 + j]);
        out[j] = zop2i("vshuff32x4", 0x88, F16, va, vc);
        out[j + 4] = zop2i("vshuff32x4", 0xDD, F16, va, vc);
        out[j + 8] = zop2i("vshuff32x4", 0x88, F16, vb, vd);
        out[j + 12] = zop2i("vshuff32x4", 0xDD, F16, vb, vd);
    }
    return out;
}

/// 4-lane variant: unpck + shufps only; the @shuffle form maps 1:1 to single
/// instructions at xmm width and has not been observed to scalarize.
inline fn transposeF32x4(r: [4]@Vector(4, f32)) [4]@Vector(4, f32) {
    const t0 = @shuffle(f32, r[0], r[1], [4]i32{ 0, -1, 1, -2 });
    const t1 = @shuffle(f32, r[0], r[1], [4]i32{ 2, -3, 3, -4 });
    const t2 = @shuffle(f32, r[2], r[3], [4]i32{ 0, -1, 1, -2 });
    const t3 = @shuffle(f32, r[2], r[3], [4]i32{ 2, -3, 3, -4 });
    return .{
        @shuffle(f32, t0, t2, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, t0, t2, [4]i32{ 2, 3, -3, -4 }),
        @shuffle(f32, t1, t3, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, t1, t3, [4]i32{ 2, 3, -3, -4 }),
    };
}

/// Decimation-in-time butterfly with a bit-reversed input ordering (cancels
/// the FFT-style reversal). Valid for any power-of-2 n; verified against a
/// scalar reference (eedi3). Lowers poorly on AVX2/AVX-512 (cross-lane masks
/// scalarize) — only the semantic reference / non-x86 path.
inline fn transposeButterfly(comptime T: type, comptime n: comptime_int, rows_in: [n]@Vector(n, T)) [n]@Vector(n, T) {
    const bits = comptime std.math.log2_int(u32, n);
    var v: [n]@Vector(n, T) = undefined;
    inline for (0..n) |i| v[i] = rows_in[comptime @bitReverse(@as(@Int(.unsigned, bits), i))];
    comptime var dist: u32 = n / 2;
    comptime var g: u32 = 1;
    inline while (dist >= 1) : ({
        dist /= 2;
        g *= 2;
    }) {
        var out: [n]@Vector(n, T) = v;
        const m = comptime unpackMasks(n, g);
        comptime var base: u32 = 0;
        inline while (base < n) : (base += 2 * dist) {
            comptime var k: u32 = 0;
            inline while (k < dist) : (k += 1) {
                const a = v[base + k];
                const b = v[base + dist + k];
                out[base + k] = @shuffle(T, a, b, m.lo);
                out[base + dist + k] = @shuffle(T, a, b, m.hi);
            }
        }
        v = out;
        if (dist == 1) break;
    }
    return v;
}

// ---- comptime mask/index builders -----------------------------------------

fn evenMask(comptime n: comptime_int) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n) |i| {
        m[i] = if (i < n / 2) @intCast(2 * i) else ~@as(i32, @intCast(2 * (i - n / 2)));
    }
    return m;
}

fn oddMask(comptime n: comptime_int) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n) |i| {
        m[i] = if (i < n / 2) @intCast(2 * i + 1) else ~@as(i32, @intCast(2 * (i - n / 2) + 1));
    }
    return m;
}

/// @shuffle mask zipping e/o lanes starting at element `base`:
/// {base, ~base, base+1, ~(base+1), ...}
fn zipMask(comptime n: comptime_int, comptime base: comptime_int) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n / 2) |i| {
        m[2 * i] = @intCast(base + i);
        m[2 * i + 1] = ~@as(i32, @intCast(base + i));
    }
    return m;
}

/// vpermt2 index vector selecting concat(a,b)[first + i*step].
fn strideIdx(comptime n: comptime_int, comptime first: comptime_int, comptime step: comptime_int) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n) |i| m[i] = first + @as(i32, @intCast(i)) * step;
    return m;
}

/// vpermt2 index vector zipping a[base..]/b[base..]: {base, n+base, base+1, ...}
fn zipIdx(comptime n: comptime_int, comptime base: comptime_int) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n / 2) |i| {
        m[2 * i] = @intCast(base + i);
        m[2 * i + 1] = @intCast(n + base + i);
    }
    return m;
}

/// Butterfly-stage @shuffle masks: interleave groups of `g` lanes from a and b.
fn unpackMasks(comptime n: comptime_int, comptime g: u32) struct { lo: [n]i32, hi: [n]i32 } {
    var lo: [n]i32 = undefined;
    var hi: [n]i32 = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const group = i / g;
        const off = i % g;
        const src_group = group / 2;
        const from_b = (group % 2) == 1;
        const a_idx: i32 = @intCast(src_group * g + off);
        lo[i] = if (from_b) ~a_idx else a_idx;
        const hi_base: i32 = @intCast((n / 2) + src_group * g + off);
        hi[i] = if (from_b) ~hi_base else hi_base;
    }
    return .{ .lo = lo, .hi = hi };
}

// ---------------------------------------------------------------------------
// Fixed-shape integer helpers (shapes set by the callers' algorithms — DCT
// blocks, wavelet pairs, SAD groups — so only feature gates apply; they embed
// in 512-bit surroundings without friction.
// ---------------------------------------------------------------------------

const I32x8 = @Vector(8, i32);
const I16x16 = @Vector(16, i16);

/// vpsadbw: per 8-byte group, sum of |a - b| into one u64 lane (n/8 lanes:
/// 4 per ymm, 8 per zmm on AVX512BW). LLVM never forms vpsadbw from the
/// portable widen+abs-diff+add form (measured in xpsnr.tempDiff1:
/// vpminub/vpmaxub/vpsubb + 2x vpmovzxbd + 2x vpaddd per 16 px). Exact
/// unconditionally: pure u8 integer math, group sum <= 8*255. The portable
/// twin below is the semantic reference.
pub inline fn sadU8(comptime n: comptime_int, a: @Vector(n, u8), b: @Vector(n, u8)) @Vector(n / 8, u64) {
    if (comptime n == 32 and use_avx2) return xop2("vpsadbw", @Vector(4, u64), a, b);
    if (comptime n == 64 and use_avx512bw) return zop2("vpsadbw", @Vector(8, u64), a, b);
    const ad: @Vector(n, u16) = @max(a, b) - @min(a, b);
    var r: @Vector(n / 8, u64) = undefined;
    inline for (0..n / 8) |i| {
        r[i] = @reduce(.Add, @as(@Vector(8, u64), @intCast(std.simd.extract(ad, i * 8, 8))));
    }
    return r;
}

/// Widest accelerated sadU8 shape (0: none, callers keep their portable form).
pub const sad_u8_lanes = if (use_avx512bw) 64 else if (use_avx2) 32 else 0;

/// vpmaddwd: adjacent-pair dot product of i16 lanes into i32 lanes. LLVM
/// never forms vpmaddwd from the portable widen-multiply-pairwise-add form
/// (measured in xpsnr.calcSquaredError u8: vpmullw + vextracti128 + 2x
/// vpmovzxwd instead). Exact iff both products of a pair cannot be -2^30
/// simultaneously (the hardware saturates only for -32768 * -32768 twice);
/// callers must guarantee that, e.g. |lanes| <= 255. n = 16 (ymm) or 32 (zmm,
/// AVX512BW). The portable twin below is the semantic reference.
pub inline fn maddwdI16(comptime n: comptime_int, a: @Vector(n, i16), b: @Vector(n, i16)) @Vector(n / 2, i32) {
    if (comptime n == 16 and use_avx2) return xop2("vpmaddwd", @Vector(8, i32), a, b);
    if (comptime n == 32 and use_avx512bw) return zop2("vpmaddwd", @Vector(16, i32), a, b);
    const p = @as(@Vector(n, i32), a) * @as(@Vector(n, i32), b);
    const ev = comptime blk: {
        var m: [n / 2]i32 = undefined;
        for (&m, 0..) |*e, i| e.* = 2 * i;
        break :blk m;
    };
    const od = comptime blk: {
        var m: [n / 2]i32 = undefined;
        for (&m, 0..) |*e, i| e.* = 2 * i + 1;
        break :blk m;
    };
    return @shuffle(i32, p, undefined, ev) + @shuffle(i32, p, undefined, od);
}

/// Per-lane (a*ka + b*kb + rnd) >> sh over n i16 lanes. vpunpck{l,h}wd pair
/// (a, b) inside each 128-bit lane, vpmaddwd against broadcast (ka, kb) forms
/// the i32 dot products, and the in-lane vpackssdw narrows back in pixel
/// order (all four ops work per 128-bit lane, so the trick is width-agnostic:
/// n = 16 on AVX2, n = 32 on AVX512BW). Exact iff no pair has both products
/// -2^30 (see maddwdI16) and every result fits i16 (vpackssdw saturates;
/// the twin's @intCast asserts it in safe builds). The portable twin below is
/// the semantic reference.
pub inline fn maddPairShrI16(comptime n: comptime_int, a: @Vector(n, i16), b: @Vector(n, i16), ka: i16, kb: i16, comptime rnd: i32, comptime sh: u5) @Vector(n, i16) {
    const V = @Vector(n, i16);
    const W = @Vector(n / 2, i32);
    // little-endian: ka lands in the even (a) word of each dword
    const pair: u32 = @as(u32, @as(u16, @bitCast(ka))) | (@as(u32, @as(u16, @bitCast(kb))) << 16);
    if (comptime (n == 16 and use_avx2) or (n == 32 and use_avx512bw)) {
        const op = if (n == 32) zop2 else xop2;
        const kp: V = @bitCast(@as(@Vector(n / 2, u32), @splat(pair)));
        const r: W = @splat(rnd);
        const lo = (op("vpmaddwd", W, op("vpunpcklwd", V, a, b), kp) + r) >> @splat(sh);
        const hi = (op("vpmaddwd", W, op("vpunpckhwd", V, a, b), kp) + r) >> @splat(sh);
        return op("vpackssdw", V, lo, hi);
    }
    const X = @Vector(n, i32);
    const res = (@as(X, a) * @as(X, @splat(ka)) + @as(X, b) * @as(X, @splat(kb)) + @as(X, @splat(rnd))) >> @splat(sh);
    return @intCast(res);
}

/// In-register 8x8 i32 transpose: classic 3-stage network (unpck dq / unpck
/// qdq / perm2i128), 24 shuffle instructions — the integer twin of
/// transposeF32x8. Inline asm because LLVM's vector combiner lowers an
/// equivalent @shuffle network through per-element stack traffic
/// ([scalarized-shuffle], measured in eedi3 on both butterfly and textbook
/// networks). Pure lane permutation: bit-exact. The portable loop below is
/// the semantic reference.
pub inline fn transposeI32x8(r: [8]I32x8) [8]I32x8 {
    if (comptime use_avx2) {
        const t0 = xop2("vpunpckldq", I32x8, r[0], r[1]); // r0c0 r1c0 r0c1 r1c1 | ...
        const t1 = xop2("vpunpckhdq", I32x8, r[0], r[1]);
        const t2 = xop2("vpunpckldq", I32x8, r[2], r[3]);
        const t3 = xop2("vpunpckhdq", I32x8, r[2], r[3]);
        const t4 = xop2("vpunpckldq", I32x8, r[4], r[5]);
        const t5 = xop2("vpunpckhdq", I32x8, r[4], r[5]);
        const t6 = xop2("vpunpckldq", I32x8, r[6], r[7]);
        const t7 = xop2("vpunpckhdq", I32x8, r[6], r[7]);
        const q0 = xop2("vpunpcklqdq", I32x8, t0, t2); // c0(r0..3) | c4(r0..3)
        const q1 = xop2("vpunpckhqdq", I32x8, t0, t2);
        const q2 = xop2("vpunpcklqdq", I32x8, t1, t3);
        const q3 = xop2("vpunpckhqdq", I32x8, t1, t3);
        const q4 = xop2("vpunpcklqdq", I32x8, t4, t6);
        const q5 = xop2("vpunpckhqdq", I32x8, t4, t6);
        const q6 = xop2("vpunpcklqdq", I32x8, t5, t7);
        const q7 = xop2("vpunpckhqdq", I32x8, t5, t7);
        return .{
            xop2i("vperm2i128", 0x20, I32x8, q0, q4),
            xop2i("vperm2i128", 0x20, I32x8, q1, q5),
            xop2i("vperm2i128", 0x20, I32x8, q2, q6),
            xop2i("vperm2i128", 0x20, I32x8, q3, q7),
            xop2i("vperm2i128", 0x31, I32x8, q0, q4),
            xop2i("vperm2i128", 0x31, I32x8, q1, q5),
            xop2i("vperm2i128", 0x31, I32x8, q2, q6),
            xop2i("vperm2i128", 0x31, I32x8, q3, q7),
        };
    }
    var out: [8]I32x8 = undefined;
    inline for (0..8) |i| {
        var col: [8]i32 = undefined;
        inline for (0..8) |j| col[j] = r[j][i];
        out[i] = col;
    }
    return out;
}

/// Clamp four 8-lane i32 rows to [0, 255] and pack them to 32 contiguous
/// bytes (row-major). The asm path folds the clamp into the pack saturations:
/// sat_u8(sat_i16(v)) == clamp(v, 0, 255) for every i32 v, so it is
/// bit-identical to the portable clamp+truncate twin below. Pinned as asm
/// because LLVM lowers the trunc-to-u8 store of blended i32 lanes through
/// per-element extract/shift/or scalar code ([narrow-select] class, ~220
/// scalar instructions per 8x8 block measured in compress); the vpermd
/// un-interleaves the packs' per-128-lane operation.
pub inline fn packClampU8x4(r0: I32x8, r1: I32x8, r2: I32x8, r3: I32x8) @Vector(32, u8) {
    if (comptime use_avx2) {
        // [r0c0-3 r1c0-3 | r0c4-7 r1c4-7] etc. as i16 lanes
        const p01 = xop2("vpackssdw", @Vector(16, i16), r0, r1);
        const p23 = xop2("vpackssdw", @Vector(16, i16), r2, r3);
        // dwords: r0c0-3 r1c0-3 r2c0-3 r3c0-3 | r0c4-7 r1c4-7 r2c4-7 r3c4-7
        const b = xop2("vpackuswb", @Vector(32, u8), p01, p23);
        return asm ("vpermd %[a], %[idx], %[out]"
            : [out] "=x" (-> @Vector(32, u8)),
            : [a] "x" (b),
              [idx] "x" (@as(I32x8, .{ 0, 4, 1, 5, 2, 6, 3, 7 })),
        );
    }
    var out: [32]u8 = undefined;
    inline for ([4]I32x8{ r0, r1, r2, r3 }, 0..) |r, i| {
        const zero: I32x8 = @splat(0);
        const max: I32x8 = @splat(255);
        const cl = intCastSignedToUnsigned(@Vector(8, u8), std.math.clamp(r, zero, max));
        out[i * 8 ..][0..8].* = cl;
    }
    return out;
}

/// The 16 rows of a 256-entry byte LUT, each 16-entry piece duplicated across
/// both 128-bit lanes for vpshufb's per-lane indexing. Built once per plane.
pub fn lutChunksU8(lut: *const [256]u8) [16]@Vector(32, u8) {
    var ch: [16]@Vector(32, u8) = undefined;
    for (0..16) |c| {
        const piece: @Vector(16, u8) = lut[c * 16 ..][0..16].*;
        ch[c] = std.simd.join(piece, piece);
    }
    return ch;
}

/// 256-entry byte LUT over 32 bytes: 16-chunk vpshufb network. There is no u8
/// gather on AVX2, so LLVM (correctly) leaves a `lut[src[i]]` loop scalar —
/// but that loop is store-commit-bound: 32 single-byte stores + 64 loads per
/// 32 px (measured 88% of AdaptiveGrainMask(u8).getFrame cycles, ~73% parked
/// on the byte stores). This form is ~5 vector ops per chunk pass and ONE
/// 32-byte store. Chunk selection is free via the vpshufb zeroing rule
/// (output byte = 0 when the index byte's bit 7 is set): for chunk c,
/// sel = (v ^ (c<<4)) +| 0x70 has bit 7 clear iff hi(v) == c, and its low
/// nibble is then lo(v), so OR-ing the 16 chunk results composes the full
/// lookup. Only the vpshufb itself is pinned asm (no Zig builtin maps to it);
/// xor/saturating-add/or are portable ops LLVM lowers 1:1. Bit-exact by
/// construction: same LUT bytes, pure integer selection. The portable twin is
/// the scalar loop in the caller (adaptive_grain_mask.processIntU8).
pub inline fn lutU8x32(chunks: *const [16]@Vector(32, u8), v: @Vector(32, u8)) @Vector(32, u8) {
    const V = @Vector(32, u8);
    const bias: V = @splat(0x70);
    var acc: V = @splat(0);
    inline for (0..16) |c| {
        const sel = (v ^ @as(V, @splat(@as(u8, c << 4)))) +| bias;
        acc |= asm ("vpshufb %[sel], %[tbl], %[out]"
            : [out] "=x" (-> V),
            : [tbl] "x" (chunks[c]),
              [sel] "x" (sel),
        );
    }
    return acc;
}

/// 256-entry byte LUT over 64 bytes (AVX512-VBMI). vpermi2b looks a byte up
/// in a 128-entry table pair by the index's low 7 bits, so two of them cover
/// the whole LUT and bit 7 picks the half (vpmovb2m + masked blend): 3 ops per
/// 64 px against the vpshufb network's 16 chunk passes per 32. Bit-exact by
/// construction (pure byte selection). The portable twin is the caller's
/// scalar loop.
pub inline fn lutU8x64(tables: *const [4]@Vector(64, u8), v: @Vector(64, u8)) @Vector(64, u8) {
    const V = @Vector(64, u8);
    const lo = vpermi2b(v, tables[0], tables[1]);
    const hi = vpermi2b(v, tables[2], tables[3]);
    return @select(u8, v >= @as(V, @splat(0x80)), hi, lo);
}

/// dst[i] = (idx[i] & 64 ? b : a)[idx[i] & 63]; idx is the tied destination.
inline fn vpermi2b(idx: @Vector(64, u8), a: @Vector(64, u8), b: @Vector(64, u8)) @Vector(64, u8) {
    return asm ("vpermi2b %[b], %[a], %[out]"
        : [out] "=v" (-> @Vector(64, u8)),
        : [a] "v" (a),
          [b] "v" (b),
          [idx] "0" (idx),
    );
}

/// Vector 256-entry byte LUT at the widest accelerated shape: 64 lanes via
/// vpermi2b on AVX512-VBMI, 32 via the vpshufb network on AVX2; `lanes == 0`
/// means no vector form (callers keep their scalar loop).
pub const LutU8 = if (use_avx512vbmi) struct {
    pub const lanes = 64;
    t: [4]@Vector(64, u8),
    pub fn init(lut: *const [256]u8) @This() {
        var t: [4]@Vector(64, u8) = undefined;
        inline for (0..4) |c| t[c] = lut[c * 64 ..][0..64].*;
        return .{ .t = t };
    }
    pub inline fn get(self: *const @This(), v: @Vector(64, u8)) @Vector(64, u8) {
        return lutU8x64(&self.t, v);
    }
} else if (use_avx2) struct {
    pub const lanes = 32;
    t: [16]@Vector(32, u8),
    pub fn init(lut: *const [256]u8) @This() {
        return .{ .t = lutChunksU8(lut) };
    }
    pub inline fn get(self: *const @This(), v: @Vector(32, u8)) @Vector(32, u8) {
        return lutU8x32(&self.t, v);
    }
} else struct {
    pub const lanes = 0;
};

/// vpshufb control packing the even words of each 128-bit lane into its low
/// qword and the odd words into its high qword.
const eo_bytes: @Vector(32, u8) = .{
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
    0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
};

inline fn wordPackEO(a: I16x16) I16x16 {
    return asm ("vpshufb %[m], %[a], %[out]"
        : [out] "=x" (-> I16x16),
        : [a] "x" (a),
          [m] "x" (eo_bytes),
    );
}

/// Even/odd lane deinterleave of 32 consecutive i16 held as two 16-lane
/// vectors: the integer twin of evenLanesF32/oddLanesF32 (vpshufb word-pack,
/// vpunpck{l,h}qdq merge, vpermq cross-lane fix). Pinned as asm for the same
/// [shredded-deinterleave] reason: fed by (laundered) plain loads, the
/// @shuffle twin below made LLVM rebuild each 32-byte load as a 7x vpinsrw
/// wall + vpblendw/vpblendd chain (measured here, MosquitoNR(u8).fwdH).
/// Pure lane permutation: bit-exact.
pub inline fn deinterleaveI16x16(a: I16x16, b: I16x16) [2]I16x16 {
    if (comptime use_avx2) {
        const pa = wordPackEO(a);
        const pb = wordPackEO(b);
        return .{
            xop1i("vpermq", 0xd8, I16x16, xop2("vpunpcklqdq", I16x16, pa, pb)),
            xop1i("vpermq", 0xd8, I16x16, xop2("vpunpckhqdq", I16x16, pa, pb)),
        };
    }
    return .{
        @shuffle(i16, a, b, [16]i32{ 0, 2, 4, 6, 8, 10, 12, 14, -1, -3, -5, -7, -9, -11, -13, -15 }),
        @shuffle(i16, a, b, [16]i32{ 1, 3, 5, 7, 9, 11, 13, 15, -2, -4, -6, -8, -10, -12, -14, -16 }),
    };
}

/// Even half of deinterleaveI16x16 for callers that discard the odd lanes.
pub inline fn evenLanesI16x16(a: I16x16, b: I16x16) I16x16 {
    if (comptime use_avx2) {
        return xop1i("vpermq", 0xd8, I16x16, xop2("vpunpcklqdq", I16x16, wordPackEO(a), wordPackEO(b)));
    }
    return @shuffle(i16, a, b, [16]i32{ 0, 2, 4, 6, 8, 10, 12, 14, -1, -3, -5, -7, -9, -11, -13, -15 });
}

/// i16 twin of interleaveF32: 16 even/odd lanes -> 32 consecutive i16.
pub inline fn interleaveI16x16(e: I16x16, o: I16x16) [2]I16x16 {
    return .{
        @shuffle(i16, e, o, [16]i32{ 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8 }),
        @shuffle(i16, e, o, [16]i32{ 8, -9, 9, -10, 10, -11, 11, -12, 12, -13, 13, -14, 14, -15, 15, -16 }),
    };
}

// ---------------------------------------------------------------------------
// IEEE round-to-nearest-even f32 -> i32 (fmtconv-compatible rounding)
// ---------------------------------------------------------------------------

const use_sse41 = builtin.target.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.target.cpu.features, .sse4_1);

/// 1.5 * 2^23: (x + magic) - magic rounds x to the nearest even integer for
/// |x| < 2^22 (both ops exactly representable; the add forces the fraction out
/// of the mantissa under the default rounding mode). Portable twin of
/// vcvtps2dq/roundss used by the Dither kernels, whose domain is
/// [-32769, 65536].
const round_magic: f32 = 12582912.0;

/// Vector f32 -> i32 with round-to-nearest-even (vcvtps2dq semantics; Zig has
/// no builtin for it — @intFromFloat truncates and @round rounds half away
/// from zero). Callers guarantee |v| < 2^22 so the portable magic-number twin
/// is exact and the conversion never overflows.
pub inline fn cvtRoundI32(comptime n: comptime_int, v: @Vector(n, f32)) @Vector(n, i32) {
    const V = @Vector(n, i32);
    if (comptime ((n == 4 and use_avx) or (n == 8 and use_avx) or (n == 16 and use_avx512f))) {
        return asm ("vcvtps2dq %[v], %[out]"
            : [out] "=x" (-> V),
            : [v] "x" (v),
        );
    }
    if (comptime (n == 4 and builtin.target.cpu.arch == .x86_64)) {
        // Baseline (non-AVX) wheel: SSE2 cvtps2dq, non-VEX 2-operand form.
        return asm ("cvtps2dq %[v], %[out]"
            : [out] "=x" (-> V),
            : [v] "x" (v),
        );
    }
    const magic: @Vector(n, f32) = @splat(round_magic);
    return @intFromFloat((v + magic) - magic);
}

/// Scalar twin of cvtRoundI32 kept in the f32 domain (the error-diffusion
/// quantizer needs both the rounded float and the pixel). roundss keeps the
/// loop-carried error-feedback chain 1 op; the magic-number twin is the
/// semantic reference (identical results for |v| < 2^22).
pub inline fn roundNearestEven(v: f32) f32 {
    if (comptime use_avx) {
        const arr: @Vector(4, f32) = .{ v, 0, 0, 0 };
        const r = asm ("vroundss $0, %[v], %[v], %[out]"
            : [out] "=x" (-> @Vector(4, f32)),
            : [v] "x" (arr),
        );
        return r[0];
    }
    if (comptime use_sse41) {
        // Non-VEX roundss is 2-operand; tie output to the input register.
        const arr: @Vector(4, f32) = .{ v, 0, 0, 0 };
        const r = asm ("roundss $0, %[v], %[out]"
            : [out] "=x" (-> @Vector(4, f32)),
            : [v] "0" (arr),
        );
        return r[0];
    }
    return (v + round_magic) - round_magic;
}

// ---------------------------------------------------------------------------
// Saturating narrowing packs (zimg-style store tails)
// ---------------------------------------------------------------------------

/// Two 8-lane i32 vectors -> one 16-lane u16 vector with unsigned saturation
/// (vpackusdw + vpermq). The saturation IS the [0, 65535] clamp — callers
/// drop their float-domain lower clamp and bias tricks. LLVM lowers the
/// portable min/max+trunc form through generic shuffles (vextracti128 +
/// vpshufb pairs — measured at 71-74% of the Dither ordered-kernel samples),
/// so the 2-op form is pinned as asm. The portable twin is the semantic
/// reference and bit-identical.
///
/// AVX2-only BY DESIGN — there is deliberately no AVX-512 twin (avx512.md §3.4,
/// verified 2026-07-07). This helper synthesizes a narrow that AVX2 lacks; on
/// AVX-512 builds the Dither paired path is inert (n_vec == 16) and LLVM emits
/// the narrow LLVM couldn't on AVX2 — a single `vpmovdw [mem],zmm` that folds
/// the store. A pinned pack would return a register and lose that fold, so it
/// cannot beat the native lowering. Confirmed bit-exact vs v3 under SDE.
pub inline fn packUsI32ToU16(lo: I32x8, hi: I32x8) @Vector(16, u16) {
    if (comptime use_avx2) {
        const p = asm ("vpackusdw %[hi], %[lo], %[out]"
            : [out] "=x" (-> @Vector(16, u16)),
            : [lo] "x" (lo),
              [hi] "x" (hi),
        );
        return asm ("vpermq $0xd8, %[p], %[out]"
            : [out] "=x" (-> @Vector(16, u16)),
            : [p] "x" (p),
        );
    }
    const zero: I32x8 = @splat(0);
    const maxv: I32x8 = @splat(65535);
    const lo_t = intCastSignedToUnsigned(@Vector(8, u16), @min(@max(lo, zero), maxv));
    const hi_t = intCastSignedToUnsigned(@Vector(8, u16), @min(@max(hi, zero), maxv));
    return std.simd.join(lo_t, hi_t);
}

/// 16-lane u16 -> 16-lane u8. PRECONDITION: every lane <= 255 (vpackuswb
/// reads lanes as SIGNED i16, so values >= 32768 would saturate to 0, and
/// 256..32767 to 255); the Dither callers guarantee it by clamping to an
/// 8-bit peak in float. The portable twin's @intCast asserts the
/// precondition in Debug builds.
///
/// AVX2-only BY DESIGN (see packUsI32ToU16 / avx512.md §3.4): the AVX-512
/// Dither build never reaches here — LLVM narrows i32->u8 directly with a
/// single `vpmovdb [mem],zmm` (store folded), which a pinned pack cannot beat.
pub inline fn packU16ToU8(v: @Vector(16, u16)) @Vector(16, u8) {
    if (comptime use_avx2) {
        const p = asm ("vpackuswb %[v], %[v], %[out]"
            : [out] "=x" (-> @Vector(32, u8)),
            : [v] "x" (v),
        );
        const q = asm ("vpermq $0xd8, %[p], %[out]"
            : [out] "=x" (-> @Vector(32, u8)),
            : [p] "x" (p),
        );
        return std.simd.extract(q, 0, 16);
    }
    return @intCast(v);
}

/// vminps against a non-NaN bound (NaN lanes come back as the bound).
/// The plain @min emits a NaN-fixup compare+blend on the critical chain
/// when LLVM cannot prove the value non-NaN (measured in the errdiff
/// wavefront); bit-identical to @min for every non-NaN input.
pub inline fn minPs(comptime n: comptime_int, v: @Vector(n, f32), bound: @Vector(n, f32)) @Vector(n, f32) {
    const V = @Vector(n, f32);
    if (comptime (use_avx and n * 4 <= 32)) {
        return xop2("vminps", V, v, bound);
    }
    if (comptime (use_avx512f and n * 4 == 64)) {
        return zop2("vminps", V, v, bound);
    }
    return @min(v, bound);
}
