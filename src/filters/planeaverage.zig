const std = @import("std");
const hz = @import("../helper.zig");
const simd = @import("simd.zig");

const allocator = std.heap.c_allocator;

/// Vectorized row sum/diff for integer planes whose exclude list cannot
/// match any pixel (Create strips values outside T's storage range, so the
/// conventional `exclude=[-1]` sentinel lands here with an empty slice).
/// Everything is integer arithmetic — sums fit the lane accumulator per row
/// and u64 overall — so lane/tail regrouping is bit-exact vs the scalar loop.
inline fn intRowSum(
    comptime T: type,
    comptime n: comptime_int,
    comptime use_ref: bool,
    row: []const T,
    ref_row: []const T,
    w: u32,
) struct { sum: u64, diff: u64 } {
    if (comptime T == u8 and simd.use_avx2) {
        // Native byte width: vpsadbw against zero for the sum, vpsadbw(v, j)
        // for the ref diff — ~6 instructions per 32 pixels vs ~7 per 8 for
        // the widened-i32 form below (its portable twin).
        const zero: @Vector(32, u8) = @splat(0);
        var acc: @Vector(4, u64) = @splat(0);
        var dacc: @Vector(4, u64) = @splat(0);
        var x: usize = 0;
        const wv = w - w % 32;
        while (x < wv) : (x += 32) {
            const v: @Vector(32, u8) = row[x..][0..32].*;
            acc += simd.sadU8x32(v, zero);
            if (use_ref) {
                const j: @Vector(32, u8) = ref_row[x..][0..32].*;
                dacc += simd.sadU8x32(v, j);
            }
        }
        var sum: u64 = @reduce(.Add, acc);
        var diff: u64 = if (use_ref) @reduce(.Add, dacc) else 0;
        while (x < w) : (x += 1) {
            sum += row[x];
            if (use_ref) diff += hz.absDiff(row[x], ref_row[x]);
        }
        return .{ .sum = sum, .diff = diff };
    }

    const VI = @Vector(n, i32);
    var acc: VI = @splat(0);
    var dacc: VI = @splat(0);
    var x: usize = 0;
    const wv = w - w % n;
    while (x < wv) : (x += n) {
        const v: VI = @as(@Vector(n, T), row[x..][0..n].*);
        acc += v;
        if (use_ref) {
            const j: VI = @as(@Vector(n, T), ref_row[x..][0..n].*);
            dacc += @max(v, j) - @min(v, j);
        }
    }
    var sum: u64 = @intCast(@reduce(.Add, acc));
    var diff: u64 = if (use_ref) @intCast(@reduce(.Add, dacc)) else 0;
    while (x < w) : (x += 1) {
        sum += row[x];
        if (use_ref) diff += hz.absDiff(row[x], ref_row[x]);
    }
    return .{ .sum = sum, .diff = diff };
}

/// Vectorized row statistics for integer planes with a potentially matching
/// exclude list (u8/u16 pixels, i32 lanes so the `pixel == exclude` compare
/// keeps the original mixed u8/i32 peer-type semantics). The match mask is
/// accumulated at the compare's own i32 width: a `@Vector(n, bool)`
/// accumulator materializes i16 lanes, and LLVM then narrows every compare
/// (vpackssdw) and re-widens the mask each iteration (vpmovzxwd+shifts).
/// All integer arithmetic — sums fit i32 per row and u64 overall — so the
/// result is bit-identical to the scalar loop, which LLVM never vectorizes.
inline fn intRowStats(
    comptime T: type,
    comptime n: comptime_int,
    comptime use_ref: bool,
    row: []const T,
    ref_row: []const T,
    exclude: []const i32,
    w: u32,
) struct { sum: u64, diff: u64, excluded: u32 } {
    const VI = @Vector(n, i32);
    const zero: VI = @splat(0);
    const neg1: VI = @splat(-1);
    var acc: VI = zero;
    var dacc: VI = zero;
    var cnt: VI = zero;
    var x: usize = 0;
    const wv = w - w % n;
    while (x < wv) : (x += n) {
        const v: VI = @as(@Vector(n, T), row[x..][0..n].*);
        var matched: VI = zero; // 0 / -1 per lane
        for (exclude) |e| {
            matched = @select(i32, v == @as(VI, @splat(e)), neg1, matched);
        }
        acc += ~matched & v;
        cnt -= matched;
        if (use_ref) {
            const j: VI = @as(@Vector(n, T), ref_row[x..][0..n].*);
            dacc += @max(v, j) - @min(v, j);
        }
    }
    var sum: u64 = @intCast(@reduce(.Add, acc));
    var diff: u64 = if (use_ref) @intCast(@reduce(.Add, dacc)) else 0;
    var excluded: u32 = @intCast(@reduce(.Add, cnt));
    while (x < w) : (x += 1) {
        const v = row[x];
        const found: bool = for (exclude) |e| {
            if (v == e) break true;
        } else false;
        if (found) excluded += 1 else sum += v;
        if (use_ref) diff += hz.absDiff(v, ref_row[x]);
    }
    return .{ .sum = sum, .diff = diff, .excluded = excluded };
}

fn intVecLanes(comptime T: type) comptime_int {
    if (@typeInfo(T) != .int or @bitSizeOf(T) > 16) return 1;
    return std.simd.suggestVectorLength(i32) orelse 1;
}

pub const Exclude = union(enum) {
    f: []f32,
    i: []i32,
};

const Stats = struct {
    avg: f64,
    diff: f64,
};

fn result(comptime T: type, acc: anytype, total: f64, peak: f32) f64 {
    if (total == 0) {
        return 0.0;
    } else if (@typeInfo(T) == .float) {
        return acc / total;
    } else {
        return @as(f64, @floatFromInt(acc)) / total / peak;
    }
}

pub fn average(comptime T: type, src: []const T, stride: u32, w: u32, h: u32, exclude_union: Exclude, peak: f32) f64 {
    var srcp: []const T = src;
    const exclude = if (@typeInfo(T) == .float) exclude_union.f else exclude_union.i;
    var total: u32 = @intCast(w * h);
    var acc: if (@typeInfo(T) == .float) f64 else u64 = 0;

    const n = comptime intVecLanes(T);
    if (comptime n > 1) {
        if (exclude.len == 0) {
            for (0..h) |_| {
                const st = intRowSum(T, n, false, srcp[0..w], srcp[0..w], w);
                acc += st.sum;
                srcp = srcp[stride..];
            }
        } else {
            for (0..h) |_| {
                const st = intRowStats(T, n, false, srcp[0..w], srcp[0..w], exclude, w);
                acc += st.sum;
                total -= st.excluded;
                srcp = srcp[stride..];
            }
        }
        return result(T, acc, @floatFromInt(total), peak);
    }

    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            const found: bool = for (exclude) |e| {
                if (v == e) break true;
            } else false;

            if (found) {
                total -= 1;
            } else {
                acc += v;
            }
        }
        srcp = srcp[stride..];
    }

    return result(T, acc, @floatFromInt(total), peak);
}

pub fn averageRef(comptime T: type, src: []const T, ref: []const T, stride: u32, w: u32, h: u32, exclude_union: Exclude, peak: f32) Stats {
    var srcp: []const T = src;
    var refp: []const T = ref;

    const exclude = if (@typeInfo(T) == .float) exclude_union.f else exclude_union.i;
    const _total: u32 = @intCast(w * h);
    var total = _total;
    const T2 = if (@typeInfo(T) == .float) f64 else u64;
    var acc: T2 = 0;
    var diffacc: T2 = 0;

    const n = comptime intVecLanes(T);
    if (comptime n > 1) {
        if (exclude.len == 0) {
            for (0..h) |_| {
                const st = intRowSum(T, n, true, srcp[0..w], refp[0..w], w);
                acc += st.sum;
                diffacc += st.diff;
                srcp = srcp[stride..];
                refp = refp[stride..];
            }
        } else {
            for (0..h) |_| {
                const st = intRowStats(T, n, true, srcp[0..w], refp[0..w], exclude, w);
                acc += st.sum;
                diffacc += st.diff;
                total -= st.excluded;
                srcp = srcp[stride..];
                refp = refp[stride..];
            }
        }
        const totalf: f64 = @floatFromInt(_total);
        return .{
            .avg = result(T, acc, @floatFromInt(total), peak),
            .diff = @as(f64, @floatFromInt(diffacc)) / totalf / peak,
        };
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            const found: bool = for (exclude) |e| {
                if (v == e) break true;
            } else false;

            if (found) {
                total -= 1;
            } else {
                acc += v;
            }

            diffacc += hz.absDiff(v, j);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }

    const _totalf: f64 = @floatFromInt(_total);
    return .{
        .avg = result(T, acc, @floatFromInt(total), peak),
        .diff = if (@typeInfo(T) == .float) (diffacc / _totalf) else @as(f64, @floatFromInt(diffacc)) / _totalf / peak,
    };
}
