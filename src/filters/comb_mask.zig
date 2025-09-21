const std = @import("std");
const math = std.math;
const vszip = @import("../vszip.zig");

// const lines: comptime_int = 5;
// const radius: comptime_int = lines / 2;

const vec_len = std.simd.suggestVectorLength(i16) orelse 1;
const V = @Vector(vec_len, i16);

const u8_0: @Vector(vec_len, u8) = @splat(0);
const u8_255: @Vector(vec_len, u8) = @splat(255);
const i16_3: V = @splat(3);
const i16_4: V = @splat(4);

const allocator = std.heap.c_allocator;

pub fn process(
    src: []const u8,
    prv: anytype,
    dst: []u8,
    stride: u32,
    w: u32,
    h: u32,
    cthresh: i32,
    cth6: u16,
    mthresh: u16,
    comptime metric_1: bool,
    comptime expand: bool,
    comptime motion: bool,
) void {
    if (metric_1) {
        metric1Mask(src, dst, w, h, stride, cthresh);
    } else {
        metric0Mask(src, dst, w, h, stride, @intCast(cthresh), cth6);
    }

    if (expand and !motion) expandMask(dst, w, h, stride);

    if (motion) {
        const tmp = allocator.alignedAlloc(u8, vszip.alignment, stride * h) catch unreachable;
        defer allocator.free(tmp);

        motionMask(src, prv, tmp, w, h, stride, mthresh);

        for (dst, tmp) |*dd, tt| {
            dd.* &= tt;
        }

        if (expand) {
            expandMask(dst, w, h, stride);
        }
    }
}

fn metric0Mask(src: []const u8, dst: []u8, w: u32, h: u32, stride: u32, cthresh: i16, cth6: u16) void {
    const cth6_v: @Vector(vec_len, u16) = @splat(cth6);
    const cthresh_v: V = @splat(cthresh);
    const cthresh_v2: V = @splat(-cthresh);

    var sa = src.ptr + (stride << 1);
    var sb = src.ptr + stride;
    var sc = src.ptr;
    var sd = sb;
    var se = sa;
    var dstp = dst;

    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var x: u32 = 0;
        while (x < w) : (x += vec_len) {
            const sav: V = sa[x..][0..vec_len].*;
            const sbv: V = sb[x..][0..vec_len].*;
            const scv: V = sc[x..][0..vec_len].*;
            const sdv: V = sd[x..][0..vec_len].*;
            const sev: V = se[x..][0..vec_len].*;
            const d1: V = scv - sbv;
            const d2: V = scv - sdv;

            const pred = ((d1 > cthresh_v) & (d2 > cthresh_v)) | ((d1 < cthresh_v2) & (d2 < cthresh_v2));
            dstp[x..][0..vec_len].* = @select(
                u8,
                pred,
                @select(
                    u8,
                    @abs((sav + i16_4 * scv + sev) - (i16_3 * (sbv + sdv))) > cth6_v,
                    u8_255,
                    u8_0,
                ),
                u8_0,
            );
        }

        sa = sb;
        sb = sc;
        sc = sd;
        sd = se;
        se = if (y < h - 3) (se + stride) else (se - stride);
        dstp = dstp[stride..];
    }
}

const vec_len32 = std.simd.suggestVectorLength(i32) orelse 1;
const V32 = @Vector(vec_len32, i32);
const u832_0: @Vector(vec_len32, u8) = @splat(0);
const u832_255: @Vector(vec_len32, u8) = @splat(255);

fn metric1Mask(src: []const u8, dst: []u8, w: u32, h: u32, stride: u32, cthresh: i32) void {
    const cthresh_v: V32 = @splat(cthresh);
    var sb = src.ptr + stride;
    var sc = src.ptr;
    var sd = src.ptr + stride;

    var dstp = dst;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var x: u32 = 0;
        while (x < w) : (x += vec_len32) {
            const sbv: V32 = sb[x..][0..vec_len32].*;
            const scv: V32 = sc[x..][0..vec_len32].*;
            const sdv: V32 = sd[x..][0..vec_len32].*;
            const val: V32 = (sbv - scv) * (sdv - scv);
            dstp[x..][0..vec_len32].* = @select(u8, val > cthresh_v, u832_255, u832_0);
        }

        sb = sc;
        sc = sd;
        sd = if (y < h - 2) (sd + stride) else (sd - stride);
        dstp = dstp[stride..];
    }
}

fn motionMask(src: []const u8, prv: []const u8, tmp: []u8, w: u32, h: u32, stride: u32, mthresh: u16) void {
    const mthresh_v: @Vector(vec_len, u16) = @splat(mthresh);
    var srcp = src;
    var prvp = prv;
    var dstp = tmp;

    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var x: u32 = 0;
        while (x < w) : (x += vec_len) {
            const sv: V = srcp[x..][0..vec_len].*;
            const pv: V = prvp[x..][0..vec_len].*;
            dstp[x..][0..vec_len].* = @select(u8, @abs(sv - pv) > mthresh_v, u8_255, u8_0);
        }

        srcp = srcp[stride..];
        prvp = prvp[stride..];
        dstp = dstp[stride..];
    }

    var lines = allocator.alloc(u8, stride << 1) catch unreachable;
    defer allocator.free(lines);
    var line0 = lines[0..stride];
    var line1 = lines[stride..];
    @memset(line0, 0);

    y = 0;
    while (y < h) : (y += 1) {
        const next_line = stride * @min(y + 1, h - 1);
        @memcpy(line1, tmp[stride * y ..][0..stride]);

        var x: u32 = 0;
        while (x < w) : (x += 1) {
            tmp[stride * y + x] = line0[x] | tmp[stride * y + x] | tmp[next_line + x];
        }

        const ll = line0;
        line0 = line1;
        line1 = ll;
    }
}

fn expandMask(dst: []u8, w: u32, h: u32, stride: u32) void {
    var dstp = dst;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var old_left = dstp[0];
        var x: u32 = 0;
        while (x < (w - 1)) : (x += 1) {
            const new: u8 = old_left | dstp[x] | dstp[x + 1];
            old_left = dstp[x];
            dstp[x] = new;
        }

        dstp = dstp[stride..];
    }
}
