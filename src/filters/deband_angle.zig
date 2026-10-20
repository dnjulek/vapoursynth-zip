//! Deband mode 7's gradient-angle plane, shared by deband_int (u16) and
//! deband_float (f32).
//!
//! The angle of coordinate (Y,X) is a Sobel kernel over taps RD apart, read
//! with clamped coordinates, then an atan. Mode 7 compares the pixel's angle
//! with the angles at its 4 reference positions (offsets up to 128 outside
//! the plane), so every angle is computed once per plane into a scratch plane
//! and the per-pixel work becomes 5 lookups.
//!
//! Only the core region Y in [-RD, h-1+RD], X in [-RD, w-1+RD] is stored: past
//! it every tap clamps to the same edge row/column, so the angle equals the
//! core's edge value and a lookup clamps its coordinate instead (the result
//! is bit-identical to computing the angle at the unclamped coordinate).
//! That is ~30% fewer atans and a smaller buffer than a 128-pixel padding.
//!
//! The scratch comes from a per-instance `Pool`: a fresh multi-megabyte block
//! per plane per frame page-faults on first touch (measured ~1 ms per 1080p
//! luma plane on Windows).

const std = @import("std");
const vcl = @import("../vcl.zig");
const simd = @import("simd.zig");
const hz = @import("../common.zig");

const allocator = std.heap.c_allocator;

pub const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
pub const f32v = @Vector(vec_len, f32);
pub const i32v = @Vector(vec_len, i32);
const boolv = @Vector(vec_len, bool);

/// Sobel tap distance.
const RD: u32 = 20;

// Kernel lanes past the plane width (x + i < ceil(width, vec_len)) still look
// up their own angle unclamped: they must land inside the core.
comptime {
    std.debug.assert(vec_len <= RD + 1);
}

const vec2: f32v = @splat(2.0);
const vec05: f32v = @splat(0.5);
const vec1: f32v = @splat(1.0);
const vec_pi: f32v = @splat(std.math.pi);
const vec_scaled_eps: f32v = @splat(0.01 * 3.0);

/// Reusable scratch blocks for `Plane`, one per concurrent request:
/// exclusive checkout and return under a mutex (never keyed by thread).
pub const Pool = struct {
    blocks: std.ArrayList([]f32) = .empty,
    mutex: std.Io.Mutex = .init,

    pub fn deinit(self: *Pool) void {
        for (self.blocks.items) |b| allocator.free(b);
        self.blocks.deinit(allocator);
        self.* = .{};
    }

    fn take(self: *Pool, n: usize) error{OutOfMemory}![]f32 {
        const got = blk: {
            self.mutex.lockUncancelable(hz.io);
            defer self.mutex.unlock(hz.io);
            break :blk self.blocks.pop();
        };
        if (got) |b| {
            if (b.len >= n) return b;
            allocator.free(b); // a chroma-sized block asked for luma
        }
        return allocator.alloc(f32, n);
    }

    fn give(self: *Pool, b: []f32) void {
        self.mutex.lockUncancelable(hz.io);
        defer self.mutex.unlock(hz.io);
        self.blocks.append(allocator, b) catch allocator.free(b);
    }
};

pub const Plane = struct {
    /// the pooled block: the angles (`buf`), then the fill's source rows
    block: []f32,
    buf: []f32,
    /// elements per row; core (Y,X) lives at (Y+RD)*stride + (X+RD)
    stride: u32,
    width: u32,
    height: u32,

    /// Compute the angle of every core coordinate of `src` (`width` x
    /// `height` samples, `src_stride` elements per row). Return the scratch
    /// with `deinit(pool)`.
    pub fn init(comptime T: type, src: []const T, src_stride: u32, width: u32, height: u32, pool: *Pool) error{OutOfMemory}!Plane {
        // A multiple of every vec_len, so the last vector of a row stays in it.
        const stride = hz.ceilN(width + 2 * RD, hz.scratchAlignmentT(@sizeOf(f32)));
        const n = (height + 2 * RD) * stride;
        const block = try pool.take(n + SourceRows.len(width));
        const p: Plane = .{
            .block = block,
            .buf = block[0..n],
            .stride = stride,
            .width = width,
            .height = height,
        };
        var rows = SourceRows.init(block[n..], width);
        p.fill(T, src, src_stride, &rows);
        return p;
    }

    pub fn deinit(self: Plane, pool: *Pool) void {
        pool.give(self.block);
    }

    fn fill(self: Plane, comptime T: type, src: []const T, src_stride: u32, rows: *SourceRows) void {
        const rd: i32 = RD;
        const hi: i32 = @intCast(self.height);
        var next: u32 = 0; // next source row to convert
        var yc: i32 = -rd;
        while (yc <= hi - 1 + rd) : (yc += 1) {
            // The taps' rows, clamped: a whole output row shares them.
            const r0: u32 = @intCast(std.math.clamp(yc - rd, 0, hi - 1));
            const r1: u32 = @intCast(std.math.clamp(yc, 0, hi - 1));
            const r2: u32 = @intCast(std.math.clamp(yc + rd, 0, hi - 1));
            while (next <= r2) : (next += 1) rows.convert(T, src[next * src_stride ..][0..self.width], next);
            const p0 = rows.get(r0);
            const p1 = rows.get(r1);
            const p2 = rows.get(r2);
            const out = self.buf[@as(u32, @intCast(yc + rd)) * self.stride ..][0..self.stride];
            // Three passes per row: one vector's dependent chain (taps ->
            // gy/gx -> atan's division -> polynomial -> /pi) is ~80 cycles,
            // longer than the out-of-order window covers, so the fused loop
            // ran latency-bound at ~33 cycles/vector (Zen 3). Splitting it
            // after gy/gx and after the atan lets more vectors overlap. Same
            // operations in the same order: bit-identical.
            // Output column X reads padded columns X-RD, X, X+RD; out[0] is X = -RD.
            const n = self.width + 2 * RD;
            var o: u32 = 0;
            while (o < n) : (o += vec_len) {
                const c = o + RD; // padded index of X = o - RD
                const g = gradients(
                    p0[c - RD ..][0..vec_len].*,
                    p0[c..][0..vec_len].*,
                    p0[c + RD ..][0..vec_len].*,
                    p1[c - RD ..][0..vec_len].*,
                    p1[c + RD ..][0..vec_len].*,
                    p2[c - RD ..][0..vec_len].*,
                    p2[c..][0..vec_len].*,
                    p2[c + RD ..][0..vec_len].*,
                );
                rows.gx[o..][0..vec_len].* = g.gx;
                rows.ratio[o..][0..vec_len].* = g.gy / g.gx;
            }
            o = 0;
            while (o < n) : (o += vec_len) {
                rows.ratio[o..][0..vec_len].* = vcl.atan(@as(f32v, rows.ratio[o..][0..vec_len].*));
            }
            o = 0;
            while (o < n) : (o += vec_len) {
                const gx: f32v = rows.gx[o..][0..vec_len].*;
                const angle_normalized = @as(f32v, rows.ratio[o..][0..vec_len].*) / vec_pi + vec05;
                out[o..][0..vec_len].* = @select(f32, @abs(gx) < vec_scaled_eps, vec1, angle_normalized);
            }
        }
    }

    /// Lanes where all 4 reference angles are within `max_angle` of the
    /// pixel's own. `ref1` holds the vertical offsets times the source stride
    /// (`inv_src_stride` = 1 / that stride), `ref2` the horizontal offsets.
    pub inline fn boost(self: *const Plane, y: u32, x: u32, ref1: i32v, ref2: i32v, inv_src_stride: f32, max_angle: f32v) boolv {
        // ref1 is an exact multiple dy * stride with |dy| <= 128, so the
        // rounded f32 product is dy exactly: no per-lane integer division.
        const dy: i32v = @intFromFloat(@round(@as(f32v, @floatFromInt(ref1)) * @as(f32v, @splat(inv_src_stride))));
        const rd: i32v = @splat(RD);
        const lo: i32v = @splat(-@as(i32, RD));
        const hi_y: i32v = @splat(@as(i32, @intCast(self.height)) - 1 + RD);
        const hi_x: i32v = @splat(@as(i32, @intCast(self.width)) - 1 + RD);
        const st: i32v = @splat(@intCast(self.stride));
        const yv: i32v = @splat(@intCast(y));
        const xv: i32v = @as(i32v, @splat(@intCast(x))) + std.simd.iota(i32, vec_len);

        const org_row = (yv + rd) * st;
        const col = xv + rd;
        const h1 = (std.math.clamp(yv + dy, lo, hi_y) + rd) * st + col;
        const h2 = (std.math.clamp(yv - dy, lo, hi_y) + rd) * st + col;
        const w1 = org_row + std.math.clamp(xv + ref2, lo, hi_x) + rd;
        const w2 = org_row + std.math.clamp(xv - ref2, lo, hi_x) + rd;

        const org: f32v = self.buf[(y + RD) * self.stride + x + RD ..][0..vec_len].*;
        var a_h1: f32v = undefined;
        var a_h2: f32v = undefined;
        var a_w1: f32v = undefined;
        var a_w2: f32v = undefined;
        // The indices go through memory: 4 vector stores, then one scalar load
        // per index. Read from registers, each index costs a vpextrd (2 uops,
        // shuffle port), and the inlined kernel's m6 math competes for the
        // same port (measured -0.6 ms per 1080p luma plane, Zen 3; the
        // laundered pointer keeps LLVM from folding it back to extracts).
        var ix: [4]i32v = .{ h1, h2, w1, w2 };
        const ip = simd.opaquePtr(i32, @ptrCast(&ix));
        const bp = self.buf.ptr;
        inline for (0..vec_len) |i| {
            a_h1[i] = bp[@intCast(ip[i])];
            a_h2[i] = bp[@intCast(ip[vec_len + i])];
            a_w1[i] = bp[@intCast(ip[2 * vec_len + i])];
            a_w2[i] = bp[@intCast(ip[3 * vec_len + i])];
        }

        var max_diff = @max(@abs(a_h1 - org), @abs(a_h2 - org));
        max_diff = @max(max_diff, @max(@abs(a_w1 - org), @abs(a_w2 - org)));
        return max_diff <= max_angle;
    }
};

/// Ring of the last 2*RD+1 source rows as f32, each padded by 2*RD columns
/// of its edge sample on both sides: a padded read equals the clamped read,
/// so every tap of the angle fill is a plain vector load, and each source
/// row is converted once instead of once per tap.
const SourceRows = struct {
    buf: []f32,
    pitch: u32,
    /// one output row's gx and gy/gx, then atan(gy/gx) (between the passes
    /// of `fill`)
    gx: []f32,
    ratio: []f32,

    const slots = 2 * RD + 1; // rows r0..r2 of one output row are <= 2*RD apart

    fn pitchOf(width: u32) u32 {
        // Columns -2*RD .. width-1+2*RD, plus the last output vector's overrun.
        return hz.ceilN(width + 4 * RD + vec_len, hz.scratchAlignmentT(@sizeOf(f32)));
    }

    /// Elements `init` needs.
    fn len(width: u32) u32 {
        return (slots + 2) * pitchOf(width);
    }

    fn init(buf: []f32, width: u32) SourceRows {
        const pitch = pitchOf(width);
        return .{
            .buf = buf[0 .. slots * pitch],
            .pitch = pitch,
            .gx = buf[slots * pitch ..][0..pitch],
            .ratio = buf[(slots + 1) * pitch ..][0..pitch],
        };
    }

    /// Row `r` (column -2*RD first); valid while `r` is among the last
    /// `slots` converted rows.
    fn get(self: SourceRows, r: u32) []const f32 {
        return self.buf[(r % slots) * self.pitch ..][0..self.pitch];
    }

    fn convert(self: SourceRows, comptime T: type, src: []const T, r: u32) void {
        const row = self.buf[(r % slots) * self.pitch ..][0..self.pitch];
        const w: u32 = @intCast(src.len);
        @memset(row[0 .. 2 * RD], if (T == f32) src[0] else @floatFromInt(src[0]));
        const mid = row[2 * RD ..][0..w];
        if (T == f32) {
            @memcpy(mid, src);
        } else {
            // Explicit vectors: LLVM leaves the plain loop scalar (vcvtsi2ss
            // per sample, measured 0.65 ms per 1080p plane).
            var x: u32 = 0;
            while (x + vec_len <= w) : (x += vec_len) {
                const v: @Vector(vec_len, T) = src[x..][0..vec_len].*;
                mid[x..][0..vec_len].* = @as(f32v, @floatFromInt(v));
            }
            while (x < w) : (x += 1) mid[x] = @floatFromInt(src[x]);
        }
        @memset(row[2 * RD + w ..], mid[w - 1]);
    }
};

/// Angles of (y, x..x+vec_len-1) for any coordinates, taps clamped to the
/// plane: the reference definition the tests check the plane against.
fn clampedAngle(comptime T: type, src: []const T, stride: u32, width: u32, height: u32, y: i32, x: i32) f32v {
    const yc: i32v = @splat(y);
    const xc: i32v = @as(i32v, @splat(x)) + std.simd.iota(i32, vec_len);
    const rd: i32v = @splat(RD);
    const g = Gather(T){ .src = src, .stride = stride, .width = width, .height = height };
    return angleOf(
        g.at(yc - rd, xc - rd),
        g.at(yc - rd, xc),
        g.at(yc - rd, xc + rd),
        g.at(yc, xc - rd),
        g.at(yc, xc + rd),
        g.at(yc + rd, xc - rd),
        g.at(yc + rd, xc),
        g.at(yc + rd, xc + rd),
    );
}

fn Gather(comptime T: type) type {
    return struct {
        src: []const T,
        stride: u32,
        width: u32,
        height: u32,

        fn at(g: @This(), y: i32v, x: i32v) f32v {
            const zero: i32v = @splat(0);
            const cy = @max(zero, @min(y, @as(i32v, @splat(@intCast(g.height - 1)))));
            const cx = @max(zero, @min(x, @as(i32v, @splat(@intCast(g.width - 1)))));
            const offsets: @Vector(vec_len, u32) = @intCast(cy * @as(i32v, @splat(@intCast(g.stride))) + cx);
            var r: f32v = undefined;
            inline for (0..vec_len) |i| {
                r[i] = if (T == f32) g.src[offsets[i]] else @floatFromInt(g.src[offsets[i]]);
            }
            return r;
        }
    };
}

const Gradients = struct { gx: f32v, gy: f32v };

/// Sobel gradients from the 8 taps (p<x><y>: column x-RD/x/x+RD, row y-RD/y/y+RD).
inline fn gradients(p00: f32v, p10: f32v, p20: f32v, p01: f32v, p21: f32v, p02: f32v, p12: f32v, p22: f32v) Gradients {
    return .{
        .gx = (p20 + vec2 * p21 + p22) - (p00 + vec2 * p01 + p02),
        .gy = (p00 + vec2 * p10 + p20) - (p02 + vec2 * p12 + p22),
    };
}

/// Normalized gradient angle in [0, 1] from gx and gy/gx (1 where the
/// horizontal gradient is ~0).
inline fn angleFrom(gx: f32v, ratio: f32v) f32v {
    const gx_is_small: boolv = @abs(gx) < vec_scaled_eps;
    const angle_raw = vcl.atan(ratio);
    const angle_normalized = angle_raw / vec_pi + vec05;
    return @select(f32, gx_is_small, vec1, angle_normalized);
}

inline fn angleOf(p00: f32v, p10: f32v, p20: f32v, p01: f32v, p21: f32v, p02: f32v, p12: f32v, p22: f32v) f32v {
    const g = gradients(p00, p10, p20, p01, p21, p02, p12, p22);
    return angleFrom(g.gx, g.gy / g.gx);
}

fn testPlane(comptime T: type, w: u32, h: u32) !void {
    const stride = hz.ceilN(w, 32);
    const src = try std.testing.allocator.alloc(T, stride * h);
    defer std.testing.allocator.free(src);
    var prng = std.Random.DefaultPrng.init(w * 131 + h);
    const r = prng.random();
    for (src) |*v| v.* = if (T == f32) r.float(f32) else r.int(u16);

    var pool: Pool = .{};
    defer pool.deinit();
    const p = try Plane.init(T, src, stride, w, h, &pool);
    defer p.deinit(&pool);
    // Every coordinate a lookup can address (|offset| <= 128), stored or
    // clamped, must equal the angle computed at that exact coordinate.
    const wi: i32 = @intCast(w);
    const hi: i32 = @intCast(h);
    var yc: i32 = -128;
    while (yc <= hi - 1 + 128) : (yc += 1) {
        var xc: i32 = -128;
        while (xc <= wi - 1 + 128) : (xc += 1) {
            const want = clampedAngle(T, src, stride, w, h, yc, xc)[0];
            const cy: u32 = @intCast(std.math.clamp(yc, -@as(i32, RD), hi - 1 + RD) + RD);
            const cx: u32 = @intCast(std.math.clamp(xc, -@as(i32, RD), wi - 1 + RD) + RD);
            const got = p.buf[cy * p.stride + cx];
            try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
        }
    }
}

test "angle plane matches the per-coordinate angle (stored, direct and clamped)" {
    inline for (.{ u16, f32 }) |T| {
        try testPlane(T, 97, 71); // interior vectors plus every edge case
        try testPlane(T, 9, 5); // smaller than the taps: clamped only
        try testPlane(T, 1, 1);
    }
}

test "boost uses the angles at the clamped reference coordinates" {
    const w: u32 = 61;
    const h: u32 = 47;
    const stride: u32 = 64;
    const src = try std.testing.allocator.alloc(u16, stride * h);
    defer std.testing.allocator.free(src);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (src) |*v| v.* = r.int(u16);
    var pool: Pool = .{};
    defer pool.deinit();
    const p = try Plane.init(u16, src, stride, w, h, &pool);
    defer p.deinit(&pool);

    const inv: f32 = 1.0 / @as(f32, @floatFromInt(stride));
    for ([_]f32{ 0.0, 0.05, 0.2, 1.0 }) |ma| {
        var y: u32 = 0;
        while (y < h) : (y += 3) {
            var x: u32 = 0;
            while (x < w) : (x += vec_len) {
                var dy: i32v = undefined;
                var dx: i32v = undefined;
                inline for (0..vec_len) |i| {
                    dy[i] = r.intRangeAtMost(i32, -128, 128);
                    dx[i] = r.intRangeAtMost(i32, -128, 128);
                }
                const got = p.boost(y, x, dy * @as(i32v, @splat(stride)), dx, inv, @splat(ma));
                inline for (0..vec_len) |i| {
                    const yi: i32 = @intCast(y);
                    const xi: i32 = @intCast(x + i);
                    const a = struct {
                        fn f(s: []const u16, yy: i32, xx: i32) f32 {
                            return clampedAngle(u16, s, stride, w, h, yy, xx)[0];
                        }
                    }.f;
                    const org = a(src, yi, xi);
                    var md = @max(@abs(a(src, yi + dy[i], xi) - org), @abs(a(src, yi - dy[i], xi) - org));
                    md = @max(md, @max(@abs(a(src, yi, xi + dx[i]) - org), @abs(a(src, yi, xi - dx[i]) - org)));
                    try std.testing.expectEqual(md <= ma, got[i]);
                }
            }
        }
    }
}
