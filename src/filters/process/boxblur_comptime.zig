//! BoxBlur with comptime radius size

const std = @import("std");
const allocator = std.heap.c_allocator;

inline fn hBlurInt(comptime T: type, srcp: [*]T, dstp: [*]T, w: u32, comptime ksize: u32, comptime inv: u32) void {
    const radius: u32 = ksize >> 1;
    var sum: u32 = srcp[radius];

    for (0..radius) |x| {
        sum += @as(u32, srcp[x]) << 1;
    }

    sum = sum * inv + (1 << 15);

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        sum += @as(u32, srcp[radius + x]) * inv;
        sum -= @as(u32, srcp[radius - x]) * inv;
        dstp[x] = @as(T, @intCast(sum >> 16));
    }

    while (x < w - radius) : (x += 1) {
        sum += @as(u32, srcp[radius + x]) * inv;
        sum -= @as(u32, srcp[x - radius - 1]) * inv;
        dstp[x] = @as(T, @intCast(sum >> 16));
    }

    while (x < w) : (x += 1) {
        sum += @as(u32, srcp[2 * w - radius - x - 1]) * inv;
        sum -= @as(u32, srcp[x - radius - 1]) * inv;
        dstp[x] = @as(T, @intCast(sum >> 16));
    }
}

fn hBlurFloat(comptime T: type, srcp: [*]T, dstp: [*]T, w: u32, comptime ksize: u32, comptime div: T) void {
    const radius: u32 = ksize >> 1;

    var j: u32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: u32 = w - 1 - j;
        var sum: T = 0.0;
        var k: u32 = 0;
        while (k < radius) : (k += 1) {
            const idx: u32 = if (j < radius - k) (@min(radius - k - j, w - 1)) else (j - radius + k);
            sum += div * srcp[idx];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: u32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[idx];
        }

        dstp[j] = sum;
    }

    j = radius;
    while (j < w - @min(w, radius)) : (j += 1) {
        var sum: T = 0.0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            sum += div * srcp[j - radius + k];
        }

        dstp[j] = sum;
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: u32 = w - 1 - j;
        var sum: T = 0.0;
        var k: u32 = 0;
        while (k < radius) : (k += 1) {
            const idx: u32 = if (j < radius - k) (@min(radius - k - j, w - 1)) else (j - radius + k);
            sum += div * srcp[idx];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: u32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += div * srcp[idx];
        }

        dstp[j] = sum;
    }
}

fn vBlurInt(comptime T: type, src: [][*]const T, dstp: [*]T, w: u32, comptime ksize: u32, comptime inv: u32) void {
    var j: u32 = 0;
    while (j < w) : (j += 1) {
        var sum: u32 = 0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            sum += src[k][j];
        }

        sum = sum * inv + (1 << 15);
        dstp[j] = @as(T, @intCast(sum >> 16));
    }
}

fn vBlurFloat(comptime T: type, src: [][*]const T, dstp: [*]T, w: u32, comptime ksize: u32, comptime div: T) void {
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

fn hvBlurCT(comptime T: type, comptime radius: u32, _src: [*]const u8, _dst: [*]u8, _stride: usize, w: u32, h: u32) void {
    const src: [*]const T = @ptrCast(@alignCast(_src));
    const dst: [*]T = @ptrCast(@alignCast(_dst));
    const stride = _stride >> (@sizeOf(T) >> 1);
    const ksize: u32 = (radius << 1) + 1;

    var i: u32 = 0;
    while (i < h) : (i += 1) {
        var srcp: [ksize][*]const T = undefined;
        const dstp: [*]T = dst + i * stride;
        const dist_from_bottom: u32 = h - 1 - i;

        const tmp_arr = allocator.alloc(T, w) catch unreachable;
        defer allocator.free(tmp_arr);
        const tmp: [*]T = tmp_arr.ptr;

        var k: u32 = 0;
        while (k < radius) : (k += 1) {
            const row: u32 = if (i < radius - k) (@min(radius - k - i, h - 1)) else (i - radius + k);
            srcp[k] = src + row * stride;
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const row: u32 = if (dist_from_bottom < k - radius) (i - @min(k - radius - dist_from_bottom, i)) else (i - radius + k);
            srcp[k] = src + row * stride;
        }

        if (@typeInfo(T) == .Int) {
            const inv: u32 = @divTrunc(((1 << 16) + radius), ksize);
            vBlurInt(T, &srcp, tmp, w, ksize, inv);
            hBlurInt(T, tmp, dstp, w, ksize, inv);
        } else {
            const div: T = 1.0 / @as(T, @floatFromInt(ksize));
            vBlurFloat(T, &srcp, tmp, w, ksize, div);
            hBlurFloat(T, tmp, dstp, w, ksize, div);
        }
    }
}

pub fn hvBlur(comptime T: type, src: [*]const u8, dst: [*]u8, stride: usize, w: u32, h: u32, radius: u32) void {
    switch (radius) {
        1 => hvBlurCT(T, 1, src, dst, stride, w, h),
        2 => hvBlurCT(T, 2, src, dst, stride, w, h),
        3 => hvBlurCT(T, 3, src, dst, stride, w, h),
        4 => hvBlurCT(T, 4, src, dst, stride, w, h),
        5 => hvBlurCT(T, 5, src, dst, stride, w, h),
        6 => hvBlurCT(T, 6, src, dst, stride, w, h),
        7 => hvBlurCT(T, 7, src, dst, stride, w, h),
        8 => hvBlurCT(T, 8, src, dst, stride, w, h),
        9 => hvBlurCT(T, 9, src, dst, stride, w, h),
        10 => hvBlurCT(T, 10, src, dst, stride, w, h),
        11 => hvBlurCT(T, 11, src, dst, stride, w, h),
        12 => hvBlurCT(T, 12, src, dst, stride, w, h),
        13 => hvBlurCT(T, 13, src, dst, stride, w, h),
        14 => hvBlurCT(T, 14, src, dst, stride, w, h),
        15 => hvBlurCT(T, 15, src, dst, stride, w, h),
        16 => hvBlurCT(T, 16, src, dst, stride, w, h),
        17 => hvBlurCT(T, 17, src, dst, stride, w, h),
        18 => hvBlurCT(T, 18, src, dst, stride, w, h),
        19 => hvBlurCT(T, 19, src, dst, stride, w, h),
        20 => hvBlurCT(T, 20, src, dst, stride, w, h),
        21 => hvBlurCT(T, 21, src, dst, stride, w, h),
        22 => hvBlurCT(T, 22, src, dst, stride, w, h),
        23 => hvBlurCT(T, 23, src, dst, stride, w, h),
        24 => hvBlurCT(T, 24, src, dst, stride, w, h),
        25 => hvBlurCT(T, 25, src, dst, stride, w, h),
        26 => hvBlurCT(T, 26, src, dst, stride, w, h),
        27 => hvBlurCT(T, 27, src, dst, stride, w, h),
        28 => hvBlurCT(T, 28, src, dst, stride, w, h),
        29 => hvBlurCT(T, 29, src, dst, stride, w, h),
        else => hvBlurCT(T, 30, src, dst, stride, w, h),
    }
}
