//! BoxBlur with runtime radius size

const std = @import("std");
const vszip = @import("../vszip.zig");
const Data = @import("../vapoursynth/boxblur.zig").Data;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;

pub fn hvBlur(comptime T: type, src: zapi.ZFrameRO, dst: zapi.ZFrameRW, d: *Data) void {
    const temp1 = allocator.alloc(T, d.tmp_size) catch unreachable;
    const temp2 = allocator.alloc(T, d.tmp_size) catch unreachable;
    defer allocator.free(temp1);
    defer allocator.free(temp2);

    var plane: u32 = 0;
    while (plane < d.vi.format.numPlanes) : (plane += 1) {
        if (!(d.planes[plane])) {
            continue;
        }

        const srcp = src.getReadSlice2(T, plane);
        const dstp = dst.getWriteSlice2(T, plane);
        const w, const h, const stride = src.getDimensions2(T, plane);

        hblur(T, srcp, dstp, stride, w, h, d.hradius, d.hpasses, temp1, temp2);
        vblur(T, dstp, dstp, stride, w, h, d.vradius, d.vpasses, temp1, temp2);
    }
}

inline fn blurInt(comptime T: type, srcp: []const T, src_step: u32, dstp: []T, dst_step: u32, len: u32, radius: u32) void {
    const ksize: u32 = (radius << 1) + 1;
    const inv: u64 = @divTrunc(((1 << 32) + @as(u64, radius)), ksize);
    var sum: u64 = srcp[radius * src_step];
    const inv2 = inv >> 16;

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        sum += @as(u32, srcp[x * src_step]) << 1;
    }

    sum = (sum * inv + (1 << 31)) >> 16;

    x = 0;
    while (x <= radius) : (x += 1) {
        sum += srcp[(radius + x) * src_step] * inv2;
        sum -= srcp[(radius - x) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }

    while (x < len - radius) : (x += 1) {
        sum += srcp[(radius + x) * src_step] * inv2;
        sum -= srcp[(x - radius - 1) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }

    while (x < len) : (x += 1) {
        sum += srcp[(2 * len - radius - x - 1) * src_step] * inv2;
        sum -= srcp[(x - radius - 1) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }
}

inline fn blurFloat(comptime T: type, srcp: []const T, src_step: u32, dstp: []T, dst_step: u32, len: u32, radius: u32) void {
    const ksize: T = @floatFromInt(radius * 2 + 1);
    const div: T = 1.0 / ksize;
    var sum: T = srcp[radius * src_step];

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        const srcv: T = srcp[x * src_step];
        sum += srcv * 2;
    }

    sum = sum * div;

    x = 0;
    while (x <= radius) : (x += 1) {
        const src1: T = srcp[(radius + x) * src_step];
        const src2: T = srcp[(radius - x) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = sum;
    }

    while (x < len - radius) : (x += 1) {
        const src1: T = srcp[(radius + x) * src_step];
        const src2: T = srcp[(x - radius - 1) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = sum;
    }

    while (x < len) : (x += 1) {
        const src1: T = srcp[(2 * len - radius - x - 1) * src_step];
        const src2: T = srcp[(x - radius - 1) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = sum;
    }
}

inline fn blur_passes(comptime T: type, srcp: []const T, dstp: []T, step: u32, len: u32, radius: u32, passes: i32, _tmp1: []T, _tmp2: []T) void {
    var tmp1 = _tmp1;
    var tmp2 = _tmp2;
    var p: i32 = passes;

    if (@typeInfo(T) == .int) {
        blurInt(T, srcp, step, tmp1, 1, len, radius);
        while (p > 2) : (p -= 1) {
            blurInt(T, tmp1, 1, tmp2, 1, len, radius);
            const tmp3 = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp3;
        }

        if (p > 1) {
            blurInt(T, tmp1, 1, dstp, step, len, radius);
        } else {
            var x: u32 = 0;
            while (x < len) : (x += 1) {
                dstp[x * step] = tmp1[x];
            }
        }
    } else {
        blurFloat(T, srcp, step, tmp1, 1, len, radius);
        while (p > 2) : (p -= 1) {
            blurFloat(T, tmp1, 1, tmp2, 1, len, radius);
            const tmp3 = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp3;
        }

        if (p > 1) {
            blurFloat(T, tmp1, 1, dstp, step, len, radius);
        } else {
            var x: u32 = 0;
            while (x < len) : (x += 1) {
                dstp[x * step] = tmp1[x];
            }
        }
    }
}

fn hblur(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, temp1: []T, temp2: []T) void {
    if ((passes > 0) and (radius > 0)) {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            blur_passes(
                T,
                srcp[y * stride ..],
                dstp[y * stride ..],
                1,
                w,
                radius,
                passes,
                temp1,
                temp2,
            );
        }
    } else {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            const srcp2 = srcp[(y * stride)..];
            const dstp2 = dstp[(y * stride)..];
            @memcpy(dstp2[0..w], srcp2[0..w]);
        }
    }
}

fn vblur(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, temp1: []T, temp2: []T) void {
    if ((passes > 0) and (radius > 0)) {
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            blur_passes(
                T,
                srcp[x..],
                dstp[x..],
                stride,
                h,
                radius,
                passes,
                temp1,
                temp2,
            );
        }
    }
}
