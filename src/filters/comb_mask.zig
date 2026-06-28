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
        motionMaskAnd(src, prv, dst, w, h, stride, mthresh);

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

const vec_u8 = std.simd.suggestVectorLength(u8) orelse 32;
const V8 = @Vector(vec_u8, u8);

inline fn diffRow(srcp: []const u8, prvp: []const u8, out: []u8, w: u32, mthresh_v: @Vector(vec_len, u16)) void {
    var x: u32 = 0;
    while (x < w) : (x += vec_len) {
        const sv: V = srcp[x..][0..vec_len].*;
        const pv: V = prvp[x..][0..vec_len].*;
        out[x..][0..vec_len].* = @select(u8, @abs(sv - pv) > mthresh_v, u8_255, u8_0);
    }
}

/// Motion mask (temporal diff), vertically dilated by one and ANDed into dst.
/// Single fused pass: each diff row is computed once into a 3-row ring buffer
/// (cache-resident, no full stride*h tmp / ~2 MB alloc), then dilated (3-row
/// vertical OR) and ANDed into dst. Bit-identical to the two-pass version.
fn motionMaskAnd(src: []const u8, prv: []const u8, dst: []u8, w: u32, h: u32, stride: u32, mthresh: u16) void {
    const mthresh_v: @Vector(vec_len, u16) = @splat(mthresh);

    const ring = allocator.alignedAlloc(u8, vszip.alignment, 3 * stride) catch unreachable;
    defer allocator.free(ring);
    const slot = struct {
        buf: []u8,
        stride: u32,
        inline fn get(self: @This(), j: u32) []u8 {
            return self.buf[(j % 3) * self.stride ..][0..self.stride];
        }
    }{ .buf = ring, .stride = stride };

    diffRow(src, prv, slot.get(0), w, mthresh_v);

    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const next = @min(y + 1, h - 1);
        if (next != y) {
            diffRow(src[next * stride ..], prv[next * stride ..], slot.get(next), w, mthresh_v);
        }
        const up: ?[]u8 = if (y > 0) slot.get(y - 1) else null;
        const cur = slot.get(y);
        const dn = slot.get(next);
        const dst_row = dst[y * stride ..];

        var x: u32 = 0;
        while (x < w) : (x += vec_u8) {
            const upv: V8 = if (up) |u| u[x..][0..vec_u8].* else @splat(0);
            const cv: V8 = cur[x..][0..vec_u8].*;
            const dn_v: V8 = dn[x..][0..vec_u8].*;
            var dv: V8 = dst_row[x..][0..vec_u8].*;
            dv &= upv | cv | dn_v;
            dst_row[x..][0..vec_u8].* = dv;
        }
    }
}

fn expandMask(dst: []u8, w: u32, h: u32, stride: u32) void {
    if (w < 2) return;

    const buf = allocator.alloc(u8, w) catch unreachable;
    defer allocator.free(buf);

    var dstp = dst;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        @memcpy(buf, dstp[0..w]);

        // 3-tap dilation of the original row; dst[w-1] is never written
        dstp[0] = buf[0] | buf[1];
        var x: u32 = 1;
        while (x + vec_u8 <= w - 1) : (x += vec_u8) {
            const l: V8 = buf[x - 1 ..][0..vec_u8].*;
            const c: V8 = buf[x..][0..vec_u8].*;
            const r: V8 = buf[x + 1 ..][0..vec_u8].*;
            dstp[x..][0..vec_u8].* = l | c | r;
        }
        while (x < (w - 1)) : (x += 1) {
            dstp[x] = buf[x - 1] | buf[x] | buf[x + 1];
        }

        dstp = dstp[stride..];
    }
}
