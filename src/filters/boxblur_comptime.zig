//! BoxBlur with comptime radius size

const std = @import("std");
const math = std.math;

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(u32) orelse 8;

pub fn hvBlur(comptime T: type, comptime radius: u32, src: []const T, dst: []T, stride: u32, w: u32, h: u32) void {
    const ksize: u32 = (radius << 1) + 1;
    const iradius: i32 = @bitCast(radius);
    const ih: i32 = @bitCast(h);

    const tmp = allocator.alloc(T, w) catch unreachable;
    defer allocator.free(tmp);

    // raw vertical column sums; max ksize * 65535 < 2^22 fits u32
    const col: []u32 = if (@typeInfo(T) == .int) allocator.alloc(u32, w) catch unreachable else &.{};
    defer if (@typeInfo(T) == .int) allocator.free(col);

    var i: i32 = 0;
    while (i < h) : (i += 1) {
        const ui: u32 = @bitCast(i);
        const dstp: []T = dst[ui * stride ..];

        if (@typeInfo(T) == .int) {
            const inv: u64 = @divTrunc(((1 << 32) + @as(u64, radius)), ksize);
            // interior rows shift the column window by one; edge rows
            // (mirrored window) recompute it from scratch
            if ((i > iradius) and (i + iradius < ih)) {
                colUpdate(T, col, src[(ui + radius) * stride ..], src[(ui - radius - 1) * stride ..], w);
            } else {
                const srcp = mirrorRows(T, ksize, src, i, stride, ih);
                colRecompute(T, ksize, col, &srcp, w);
            }
            colScale(T, col, tmp, w, inv);
            hBlurInt(T, tmp, dstp, w, ksize, inv);
        } else {
            const div: f32 = 1.0 / @as(f32, @floatFromInt(ksize));
            const srcp = mirrorRows(T, ksize, src, i, stride, ih);
            vBlurFloat(T, ksize, &srcp, tmp, w, div);
            hBlurFloat(T, tmp, dstp, @bitCast(w), ksize, div);
        }
    }
}

/// The ksize source row slices contributing to output row i, with mirrored
/// indices at the top/bottom edges.
inline fn mirrorRows(comptime T: type, comptime ksize: u32, src: []const T, i: i32, stride: u32, ih: i32) [ksize][]const T {
    const iradius: i32 = @intCast(ksize >> 1);
    const dist_from_bottom: i32 = ih - 1 - i;
    var srcp: [ksize][]const T = undefined;

    var k: i32 = 0;
    while (k < iradius) : (k += 1) {
        const row: i32 = if (i < iradius - k) @min(iradius - k - i, ih - 1) else (i - iradius + k);
        const urow: u32 = @bitCast(row);
        srcp[@intCast(k)] = src[urow * stride ..];
    }

    k = iradius;
    while (k < ksize) : (k += 1) {
        const row: i32 = if (dist_from_bottom < k - iradius) (i - @min(k - iradius - dist_from_bottom, i)) else (i - iradius + k);
        const urow: u32 = @bitCast(row);
        srcp[@intCast(k)] = src[urow * stride ..];
    }

    return srcp;
}

inline fn colUpdate(comptime T: type, col: []u32, add_row: []const T, sub_row: []const T, w: u32) void {
    const UV = @Vector(vec_len, u32);
    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        const a: @Vector(vec_len, T) = add_row[j..][0..vec_len].*;
        const s: @Vector(vec_len, T) = sub_row[j..][0..vec_len].*;
        var c: UV = col[j..][0..vec_len].*;
        c += @intCast(a);
        c -= @intCast(s);
        col[j..][0..vec_len].* = c;
    }
    while (j < w) : (j += 1) {
        col[j] += add_row[j];
        col[j] -= sub_row[j];
    }
}

inline fn colRecompute(comptime T: type, comptime ksize: u32, col: []u32, srcp: *const [ksize][]const T, w: u32) void {
    const UV = @Vector(vec_len, u32);
    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        var acc: UV = @splat(0);
        inline for (0..ksize) |k| {
            const v: @Vector(vec_len, T) = srcp[k][j..][0..vec_len].*;
            acc += @intCast(v);
        }
        col[j..][0..vec_len].* = acc;
    }
    while (j < w) : (j += 1) {
        var sum: u32 = 0;
        inline for (0..ksize) |k| {
            sum += srcp[k][j];
        }
        col[j] = sum;
    }
}

inline fn colScale(comptime T: type, col: []const u32, tmp: []T, w: u32, comptime inv: u64) void {
    const U64V = @Vector(vec_len, u64);
    const invv: U64V = @splat(inv);
    const bias: U64V = @splat(1 << 31);
    const shift: @Vector(vec_len, u6) = @splat(32);

    var j: u32 = 0;
    const wv = w - (w % vec_len);
    while (j < wv) : (j += vec_len) {
        const c: @Vector(vec_len, u32) = col[j..][0..vec_len].*;
        const c64: U64V = @intCast(c);
        const out: @Vector(vec_len, T) = @intCast((c64 * invv + bias) >> shift);
        tmp[j..][0..vec_len].* = out;
    }
    while (j < w) : (j += 1) {
        tmp[j] = @intCast((@as(u64, col[j]) * inv + (1 << 31)) >> 32);
    }
}

inline fn hBlurInt(comptime T: type, srcp: []T, dstp: []T, w: u32, comptime ksize: u32, comptime inv: u64) void {
    const radius: u32 = ksize >> 1;
    var sum: u64 = srcp[radius];
    const inv2 = inv >> 16;

    for (0..radius) |x| {
        sum += @as(u32, srcp[x]) << 1;
    }

    sum = (sum * inv + (1 << 31)) >> 16;

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        sum += @as(u32, srcp[radius + x]) * inv2;
        sum -= @as(u32, srcp[radius - x]) * inv2;
        dstp[x] = @intCast(sum >> 16);
    }

    while (x < w - radius) : (x += 1) {
        sum += @as(u32, srcp[radius + x]) * inv2;
        sum -= @as(u32, srcp[x - radius - 1]) * inv2;
        dstp[x] = @intCast(sum >> 16);
    }

    while (x < w) : (x += 1) {
        sum += @as(u32, srcp[2 * w - radius - x - 1]) * inv2;
        sum -= @as(u32, srcp[x - radius - 1]) * inv2;
        dstp[x] = @intCast(sum >> 16);
    }
}

fn vBlurFloat(comptime T: type, comptime ksize: u32, src: *const [ksize][]const T, dstp: []T, w: u32, comptime div: f32) void {
    // Accumulate in f32 regardless of T so f16 output matches the scalar
    // f32-accumulate-then-narrow path (bit-exact for f32).
    const fvec = std.simd.suggestVectorLength(f32) orelse 8;
    const FV = @Vector(fvec, f32);
    const dv: FV = @splat(div);

    var j: u32 = 0;
    const wv = w - (w % fvec);
    while (j < wv) : (j += fvec) {
        var acc: FV = @splat(0.0);
        inline for (0..ksize) |k| {
            const v: @Vector(fvec, T) = src[k][j..][0..fvec].*;
            const vf: FV = if (T == f32) v else @floatCast(v);
            acc = acc + dv * vf;
        }
        if (T == f32) {
            dstp[j..][0..fvec].* = acc;
        } else {
            dstp[j..][0..fvec].* = @as(@Vector(fvec, T), @floatCast(acc));
        }
    }
    while (j < w) : (j += 1) {
        var sum: f32 = 0.0;
        inline for (0..ksize) |k| {
            sum += div * src[k][j];
        }
        dstp[j] = if (T == f32) sum else @floatCast(sum);
    }
}

fn hBlurFloat(comptime T: type, srcp: []T, dstp: []T, w: i32, comptime ksize: u32, comptime div: f32) void {
    const radius: i32 = @as(i32, @bitCast(ksize)) >> 1;
    // Accumulate in f32 regardless of T (bit-exact for f32, f16 matches the
    // scalar f32-accumulate-then-narrow reference).
    const fvec = std.simd.suggestVectorLength(f32) orelse 8;
    const FV = @Vector(fvec, f32);
    const dv: FV = @splat(div);

    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }

    j = radius;
    const center_end: i32 = w - @min(w, radius);
    while (j + fvec <= center_end) : (j += fvec) {
        const uj: u32 = @intCast(j - radius);
        var acc: FV = @splat(0.0);
        inline for (0..ksize) |k| {
            const v: @Vector(fvec, T) = srcp[uj + k ..][0..fvec].*;
            const vf: FV = if (T == f32) v else @floatCast(v);
            acc = acc + dv * vf;
        }
        if (T == f32) {
            dstp[@intCast(j)..][0..fvec].* = acc;
        } else {
            dstp[@intCast(j)..][0..fvec].* = @as(@Vector(fvec, T), @floatCast(acc));
        }
    }
    while (j < center_end) : (j += 1) {
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += div * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = if (T == f32) sum else @floatCast(sum);
    }
}
