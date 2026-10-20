//! Host-independent globals and helpers shared by the kernels in `filters/`.
//! Replaces the parts of vszip's `vszip.zig`/`helper.zig` that kernels used
//! (no VapourSynth or AviSynth types may appear here).

const std = @import("std");

/// Suggested u8 vector length for the compile target (not runtime CPU).
pub const vec_len = std.simd.suggestVectorLength(u8) orelse 32;
pub const alignment = std.mem.Alignment.fromByteUnits(vec_len);

/// Single-threaded `std.Io` used for the mutexes guarding scratch pools.
pub const io: std.Io = std.Io.Threaded.global_single_threaded.io();

/// Row alignment for filter-owned scratch buffers. vszip padded scratch rows
/// to VapourSynth's runtime frame alignment (32/64); a fixed 64 covers both
/// and keeps rows vector-aligned on AVX-512 too. Never index host frames with
/// a scratch stride (host pitches are independent).
pub const scratch_alignment: u32 = 64;

pub fn scratchAlignmentT(div: u32) u32 {
    return @divExact(scratch_alignment, div);
}

pub fn absDiff(x: anytype, y: anytype) @TypeOf(x) {
    return if (x > y) (x - y) else (y - x);
}

pub fn getVal(comptime T: type, ptr: anytype, dist: isize) T {
    const adr: isize = @intCast(@intFromPtr(ptr));
    const uadr: usize = @intCast(adr + dist);
    const ptr2: [*]const T = @ptrFromInt(uadr);
    return ptr2[0];
}

pub fn getVal2(comptime T: type, ptr: anytype, x: u32, y: u32) T {
    const ix: i32 = @intCast(x);
    const iy: i32 = @intCast(y);
    const adr: isize = @intCast(@intFromPtr(ptr));
    const uadr: usize = @intCast(adr + ix - iy);
    const ptr2: [*]const T = @ptrFromInt(uadr);
    return ptr2[0];
}

pub inline fn ceilN(x: u32, n: u32) u32 {
    return (x + (n - 1)) & ~(n - 1);
}

pub const ditherType = enum {
    none,
    ordered,
    random,
    error_diffusion,

    pub fn toString(self: ditherType) [:0]const u8 {
        return @tagName(self);
    }
};

/// Host-neutral view of a sample format for the range math below.
pub const Depth = struct {
    bits: u32,
    float: bool,
};

pub fn getLowestValue(d: Depth, chroma: bool, limited: bool) f32 {
    if (d.float) return if (chroma) -0.5 else 0.0;
    if (limited) return @floatFromInt(std.math.shl(i64, 16, d.bits - 8));
    return 0;
}

/// i64: a 32-bit integer peak overflows i32 (it used to come out as -1).
/// The f32 result is approximate for 32-bit (2^32 - 1 rounds to 2^32);
/// callers validating 32-bit values must use an exact peak instead.
pub fn getPeakValue(d: Depth, chroma: bool, limited: bool) f32 {
    if (d.float) return if (chroma) 0.5 else 1.0;
    if (limited) {
        const a: i64 = if (chroma) 240 else 235;
        return @floatFromInt(std.math.shl(i64, a, d.bits - 8));
    }
    return @floatFromInt(std.math.shl(i64, 1, d.bits) - 1);
}

/// vszip's scaleValue arithmetic: rescale `value` from `in` to `out` with the
/// target's range. Callers handle the equal-depth shortcut and pass full range
/// for float targets. Keep the association (`v *= (op - ol) / (ip - il)`):
/// `v * a / b` rounds differently in the last ulp.
pub fn rescale(value: f32, in: Depth, out: Depth, chroma: bool, limited: bool) f32 {
    const input_peak = getPeakValue(in, chroma, limited);
    const input_lowest = getLowestValue(in, chroma, limited);
    const output_peak = getPeakValue(out, chroma, limited);
    const output_lowest = getLowestValue(out, chroma, limited);
    var v = value * ((output_peak - output_lowest) / (input_peak - input_lowest));
    if (!out.float) v = @max(@min(@round(v), getPeakValue(out, false, false)), 0);
    return v;
}
