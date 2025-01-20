//! BoxBlur with comptime radius size

const std = @import("std");
const vszip = @import("../vszip.zig");
const Data = @import("../vapoursynth/boxblur.zig").Data;
const math = std.math;

const allocator = std.heap.c_allocator;

pub fn hvBlur(comptime T: type, comptime radius: u32, src: []const T, dst: []T, stride: u32, w: u32, h: u32) void {
    const ksize: u32 = (radius << 1) + 1;
    const iradius: i32 = @bitCast(radius);
    const ih: i32 = @bitCast(h);

    var i: i32 = 0;
    while (i < h) : (i += 1) {
        const ui: u32 = @bitCast(i);
        var srcp: [ksize][]const T = undefined;
        const dstp: []T = dst[ui * stride ..];
        const dist_from_bottom: i32 = ih - 1 - i;

        const tmp = allocator.alloc(T, w) catch unreachable;
        defer allocator.free(tmp);

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

        if (@typeInfo(T) == .int) {
            const inv: u64 = @divTrunc(((1 << 32) + @as(u64, radius)), ksize);
            vBlurInt(T, &srcp, tmp, w, ksize, inv);
            hBlurInt(T, tmp, dstp, w, ksize, inv);
        } else {
            const div: T = 1.0 / @as(T, @floatFromInt(ksize));
            vBlurFloat(T, &srcp, tmp, w, ksize, div);
            hBlurFloat(T, tmp, dstp, @bitCast(w), ksize, div);
        }
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

fn hBlurFloat(comptime T: type, srcp: []T, dstp: []T, w: i32, comptime ksize: u32, comptime div: T) void {
    const radius: i32 = @as(i32, @bitCast(ksize)) >> 1;

    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: T = 0.0;
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

        dstp[@intCast(j)] = sum;
    }

    j = radius;
    while (j < w - @min(w, radius)) : (j += 1) {
        var sum: T = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += div * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: T = 0.0;
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

        dstp[@intCast(j)] = sum;
    }
}

fn vBlurInt(comptime T: type, src: [][]const T, dstp: []T, w: u32, comptime ksize: u32, comptime inv: u64) void {
    var j: u32 = 0;
    while (j < w) : (j += 1) {
        var sum: u64 = 0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            sum += src[k][j];
        }

        sum = (sum * inv + (1 << 31)) >> 16;
        dstp[j] = @intCast(sum >> 16);
    }
}

fn vBlurFloat(comptime T: type, src: [][]const T, dstp: []T, w: u32, comptime ksize: u32, comptime div: T) void {
    var j: u32 = 0;
    while (j < w) : (j += 1) {
        var sum: T = 0.0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            sum += div * src[k][j];
        }

        dstp[j] = sum;
    }
}
