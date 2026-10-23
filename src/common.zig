//! Host-independent globals and helpers shared by the kernels in `filters/`.
//! Replaces the parts of vszip's `vszip.zig`/`helper.zig` that kernels used
//! (no VapourSynth or AviSynth types may appear here).

const std = @import("std");
const builtin = @import("builtin");

/// Allocator of the shared kernels: exactly `std.heap.c_allocator` in the
/// plugins. In host-free tests it forwards to `test_hooks.backing`, so the
/// tests can inject allocation failures and check that every partially
/// constructed object is released.
pub const kernel_allocator: std.mem.Allocator = if (builtin.is_test) test_hooks.forwarding else std.heap.c_allocator;

pub const test_hooks = struct {
    pub var backing: std.mem.Allocator = std.heap.c_allocator;

    const forwarding: std.mem.Allocator = .{ .ptr = undefined, .vtable = &.{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    } };

    fn alloc(_: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        return backing.rawAlloc(len, a, ra);
    }
    fn resize(_: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        return backing.rawResize(m, a, n, ra);
    }
    fn remap(_: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return backing.rawRemap(m, a, n, ra);
    }
    fn free(_: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        backing.rawFree(m, a, ra);
    }
};

/// Suggested u8 vector length for the compile target (not runtime CPU).
pub const vec_len = std.simd.suggestVectorLength(u8) orelse 32;
pub const alignment = std.mem.Alignment.fromByteUnits(vec_len);

/// Single-threaded `std.Io` used for the mutexes guarding scratch pools.
pub const io: std.Io = std.Io.Threaded.global_single_threaded.io();

/// Row padding, in elements of `div` bytes, for filter-owned scratch buffers:
/// one vector of the build target (`vec_len` bytes), so a row's last vector
/// stays inside it. Never index host frames with a scratch stride (host
/// pitches are independent).
pub fn scratchAlignmentT(div: u32) u32 {
    return @divExact(vec_len, div);
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
