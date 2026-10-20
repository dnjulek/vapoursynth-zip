const std = @import("std");
const simd = @import("simd.zig");

/// Clamp every element of `srcp` into [min, max]. Lane semantics match the
/// scalar `@min(@max(min, s), max)` exactly (llvm.minnum/maxnum: a NaN input
/// yields the bound), so this is bit-exact vs the previous per-element loop.
/// LLVM never auto-vectorized that loop (scalar cmov chains for ints, a
/// per-element compiler_rt fminf/fmaxf libcall pair for f16 — measured 81% of
/// the f16 path's instructions), hence the explicit vector shape.
pub fn clampSlice(comptime T: type, dstp: []T, srcp: []const T, min: T, max: T) void {
    std.debug.assert(dstp.len == srcp.len);
    if (comptime @typeInfo(T) == .float) {
        // Precondition for simd.clampV's bare-vmaxps/vminps form: the runtime
        // bounds must be non-NaN. limiterCreate rejects NaN min/max, so this
        // always holds; the assert documents and (in debug/safe builds)
        // enforces it. See simd.clampV for why the bare form is bit-identical
        // to @min(@max(min_v, v), max_v) under this precondition.
        std.debug.assert(!std.math.isNan(min) and !std.math.isNan(max));
    }
    if (comptime T == f16) {
        // No native f16 min/max before AVX512-FP16: clamp in f32 (F16C widen/
        // narrow). f16 values and bounds are exactly representable in f32, so
        // the round-trip is exact and NaN handling matches the old libcalls.
        const n = comptime (std.simd.suggestVectorLength(f32) orelse 8);
        const min_v: @Vector(n, f32) = @splat(min);
        const max_v: @Vector(n, f32) = @splat(max);
        var i: usize = 0;
        // 4x manual unroll: simd.clampV's inline asm defeats LLVM's loop
        // unroller, so a plain 1x loop leaves the per-vector loop control
        // un-amortized (measured +16% kernel Ir vs base). Unrolling by hand
        // amortizes the control and keeps the bare-vmaxps/vminps win.
        while (i + n * 4 <= srcp.len) : (i += n * 4) {
            inline for (0..4) |k| {
                const off = i + k * n;
                const v: @Vector(n, f16) = srcp[off..][0..n].*;
                const c = simd.clampV(n, @as(@Vector(n, f32), @floatCast(v)), min_v, max_v);
                const w: @Vector(n, f16) = @floatCast(c);
                dstp[off..][0..n].* = w;
            }
        }
        while (i + n <= srcp.len) : (i += n) {
            const v: @Vector(n, f16) = srcp[i..][0..n].*;
            const c = simd.clampV(n, @as(@Vector(n, f32), @floatCast(v)), min_v, max_v);
            const w: @Vector(n, f16) = @floatCast(c);
            dstp[i..][0..n].* = w;
        }
        while (i < srcp.len) : (i += 1) dstp[i] = @min(@max(min, srcp[i]), max);
        return;
    }

    const n_opt = comptime std.simd.suggestVectorLength(T);
    if (comptime n_opt == null) {
        for (srcp, dstp) |s, *d| d.* = @min(@max(min, s), max);
        return;
    }
    const n = comptime n_opt.?;
    const min_v: @Vector(n, T) = @splat(min);
    const max_v: @Vector(n, T) = @splat(max);
    var i: usize = 0;
    if (comptime T == f32) {
        // f32: bare vmaxps/vminps (bounds non-NaN), 4x manually unrolled —
        // simd.clampV's inline asm defeats LLVM's loop unroller, so a plain 1x
        // loop leaves loop control un-amortized (measured +16% kernel Ir vs
        // base). See the f16 branch above.
        while (i + n * 4 <= srcp.len) : (i += n * 4) {
            inline for (0..4) |k| {
                const off = i + k * n;
                const v: @Vector(n, f32) = srcp[off..][0..n].*;
                dstp[off..][0..n].* = simd.clampV(n, v, min_v, max_v);
            }
        }
        while (i + n <= srcp.len) : (i += n) {
            const v: @Vector(n, f32) = srcp[i..][0..n].*;
            dstp[i..][0..n].* = simd.clampV(n, v, min_v, max_v);
        }
    } else {
        // Ints have no NaN issue and already lower to bare vpmax/vpmin, which
        // LLVM auto-unrolls; keep the portable form.
        while (i + n <= srcp.len) : (i += n) {
            const v: @Vector(n, T) = srcp[i..][0..n].*;
            dstp[i..][0..n].* = @min(@max(min_v, v), max_v);
        }
    }
    while (i < srcp.len) : (i += 1) dstp[i] = @min(@max(min, srcp[i]), max);
}

// Clamp ranges [min, max] per plane for each format and range.
pub const full8 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 255, 255, 255 } };
pub const full9 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 511, 511, 511 } };
pub const full10 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 1023, 1023, 1023 } };
pub const full12 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 4095, 4095, 4095 } };
pub const full14 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 16383, 16383, 16383 } };
pub const full16 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 65535, 65535, 65535 } };
pub const full32 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 4294967295, 4294967295, 4294967295 } };

pub const yuv8 = [2][3]comptime_int{ .{ 16, 16, 16 }, .{ 235, 240, 240 } };
pub const yuv9 = [2][3]comptime_int{ .{ 32, 32, 32 }, .{ 470, 480, 480 } };
pub const yuv10 = [2][3]comptime_int{ .{ 64, 64, 64 }, .{ 940, 960, 960 } };
pub const yuv12 = [2][3]comptime_int{ .{ 256, 256, 256 }, .{ 3760, 3840, 3840 } };
pub const yuv14 = [2][3]comptime_int{ .{ 1024, 1024, 1024 }, .{ 15040, 15360, 15360 } };
pub const yuv16 = [2][3]comptime_int{ .{ 4096, 4096, 4096 }, .{ 60160, 61440, 61440 } };
pub const yuv32 = [2][3]comptime_int{ .{ 268435456, 268435456, 268435456 }, .{ 3942645760, 4026531840, 4026531840 } };

pub const rgb8 = [2][3]comptime_int{ .{ 16, 16, 16 }, .{ 235, 235, 235 } };
pub const rgb9 = [2][3]comptime_int{ .{ 32, 32, 32 }, .{ 470, 470, 470 } };
pub const rgb10 = [2][3]comptime_int{ .{ 64, 64, 64 }, .{ 940, 940, 940 } };
pub const rgb12 = [2][3]comptime_int{ .{ 256, 256, 256 }, .{ 3760, 3760, 3760 } };
pub const rgb14 = [2][3]comptime_int{ .{ 1024, 1024, 1024 }, .{ 15040, 15040, 15040 } };
pub const rgb16 = [2][3]comptime_int{ .{ 4096, 4096, 4096 }, .{ 60160, 60160, 60160 } };
pub const rgb32 = [2][3]comptime_int{ .{ 268435456, 268435456, 268435456 }, .{ 3942645760, 3942645760, 3942645760 } };

pub const yuvf = [2][3]comptime_float{ .{ 0, -0.5, -0.5 }, .{ 1, 0.5, 0.5 } };
pub const rgbf = [2][3]comptime_float{ .{ 0, 0, 0 }, .{ 1, 1, 1 } };
