const std = @import("std");
const simd = @import("simd.zig");
const math = std.math;
const allocator = std.heap.c_allocator;

const XPSNR_GAMMA: u32 = 2;

const vec32 = std.simd.suggestVectorLength(i32) orelse 8;

// u8 pixels always fit i16 lanes; u16 input (10-bit nominally, but any value
// memory-wise) gets i32 lanes so results match the scalar i32 math for every
// possible input.
fn LaneInt(comptime T: type) type {
    return if (T == u8) i16 else i32;
}

fn laneCount(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(LaneInt(T)) orelse 8;
}

/// Even-lane keep mask: the x-decimated outputs live in the even lanes; odd
/// lanes are waste and get zeroed before accumulation.
fn evenMask(comptime U: type, comptime vl: comptime_int) @Vector(vl, U) {
    var m: [vl]U = undefined;
    for (&m, 0..) |*e, i| e.* = if (i % 2 == 0) math.maxInt(U) else 0;
    return m;
}

/// lane j <- lane j+1 (top lane duplicated). Derives a "+1 column" vector
/// from an already-loaded one so no vector load ever reads past the scalar
/// code's rightmost tap column (with stride == width that byte can sit one
/// past the plane allocation); the top lane is odd, i.e. decimation waste,
/// so its value is never consumed.
inline fn shiftDown1(comptime L: type, comptime vl: comptime_int, v: @Vector(vl, L)) @Vector(vl, L) {
    const mask = comptime blk: {
        var m: [vl]i32 = undefined;
        for (&m, 0..) |*e, i| e.* = @min(i + 1, vl - 1);
        break :blk m;
    };
    return @shuffle(L, v, undefined, mask);
}

/// Vertical pair sum of rows a + b at column i. Laundered loads: the five
/// comptime x-offsets (-2..+2) per row pair otherwise fuse into shuffle
/// chains (see simd.loaduOpaque).
inline fn vpair(comptime T: type, comptime vl: comptime_int, a: []const T, b: []const T, i: usize) @Vector(vl, LaneInt(T)) {
    const va: @Vector(vl, LaneInt(T)) = @intCast(simd.loaduOpaque(T, vl, a[i..][0..vl]));
    const vb: @Vector(vl, LaneInt(T)) = @intCast(simd.loaduOpaque(T, vl, b[i..][0..vl]));
    return va + vb;
}

inline fn px(row: anytype, i: usize) i32 {
    return @intCast(row[i]);
}

/// 24-tap spatial-activity high-pass over the 2x2-decimated grid
/// [x0, x1) x [y0, y1), step 2, in absolute picture coordinates (same
/// convention as spatialAct). The caller guarantees x0/y0 >= 2, and the
/// block/act construction with even frame dimensions keeps every tap
/// (columns x0-2..x1+1, rows y0-2..y1+1) inside the plane.
///
/// The taps group into symmetric vertical row pairs
///   R0 = row(y) + row(y+1), R1 = row(y-1) + row(y+2), R2 = row(y-2) + row(y+3)
/// and symmetric column pair sums S0 = R(x) + R(x+1), S1 = R(x-1) + R(x+2),
/// S2 = R(x-2) + R(x+3):
///   f = 12*S0(R0) - 3*(S1(R0) + S0(R1)) - 2*S1(R1)
///     - (S2(R0) + S2(R1) + S0(R2) + S1(R2))
/// All-integer math with lanes wide enough that nothing overflows
/// (|f| <= 48*maxval: 12240 for u8 in i16, 3145680 for u16 in i32), so any
/// regrouping is exact.
fn highds(comptime T: type, pic: []const T, o: usize, x0: usize, x1: usize, y0: usize, y1: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const LV = @Vector(vl, L);
    const U = std.meta.Int(.unsigned, @bitSizeOf(L));
    const even_mask = comptime evenMask(U, vl);
    const twelve: LV = @splat(12);
    const three: LV = @splat(3);
    const two: LV = @splat(2);

    std.debug.assert(x0 >= 2 and y0 >= 2);

    var saAct: u64 = 0;
    var y: usize = y0;
    while (y < y1) : (y += 2) {
        const rm2 = pic[(y - 2) * o ..];
        const rm1 = pic[(y - 1) * o ..];
        const rc0 = pic[y * o ..];
        const rc1 = pic[(y + 1) * o ..];
        const rp2 = pic[(y + 2) * o ..];
        const rp3 = pic[(y + 3) * o ..];

        var row_acc: @Vector(vl, u32) = @splat(0);
        var x: usize = x0;
        while (x + vl <= x1) : (x += vl) {
            const r0_m2 = vpair(T, vl, rc0, rc1, x - 2);
            const r0_m1 = vpair(T, vl, rc0, rc1, x - 1);
            const r0_c0 = vpair(T, vl, rc0, rc1, x);
            const r0_p1 = vpair(T, vl, rc0, rc1, x + 1);
            const r0_p2 = vpair(T, vl, rc0, rc1, x + 2);
            const r1_m2 = vpair(T, vl, rm1, rp2, x - 2);
            const r1_m1 = vpair(T, vl, rm1, rp2, x - 1);
            const r1_c0 = vpair(T, vl, rm1, rp2, x);
            const r1_p1 = vpair(T, vl, rm1, rp2, x + 1);
            const r1_p2 = vpair(T, vl, rm1, rp2, x + 2);
            const r2_m1 = vpair(T, vl, rm2, rp3, x - 1);
            const r2_c0 = vpair(T, vl, rm2, rp3, x);
            const r2_p1 = vpair(T, vl, rm2, rp3, x + 1);
            const r2_p2 = vpair(T, vl, rm2, rp3, x + 2);

            const s0_r0 = r0_c0 + r0_p1;
            const s1_r0 = r0_m1 + r0_p2;
            const s2_r0 = r0_m2 + shiftDown1(L, vl, r0_p2);
            const s0_r1 = r1_c0 + r1_p1;
            const s1_r1 = r1_m1 + r1_p2;
            const s2_r1 = r1_m2 + shiftDown1(L, vl, r1_p2);
            const s0_r2 = r2_c0 + r2_p1;
            const s1_r2 = r2_m1 + r2_p2;

            const f = twelve * s0_r0 - three * (s1_r0 + s0_r1) - two * s1_r1 - (s2_r0 + s2_r1 + s0_r2 + s1_r2);
            row_acc += @intCast(@abs(f) & even_mask);
        }
        saAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc)));

        while (x < x1) : (x += 2) {
            // zig fmt: off
            const f: i32 = 12 * (px(rc0, x) + px(rc0, x + 1)
            + px(rc1, x) + px(rc1, x + 1))
            - 3 * (px(rm1, x) + px(rm1, x + 1)
            + px(rp2, x) + px(rp2, x + 1))
            - 3 * (px(rc0, x - 1) + px(rc0, x + 2)
            + px(rc1, x - 1) + px(rc1, x + 2))
            - 2 * (px(rm1, x - 1) + px(rm1, x + 2)
            + px(rp2, x - 1) + px(rp2, x + 2))
            - (px(rm2, x - 1) + px(rm2, x)
            + px(rm2, x + 1) + px(rm2, x + 2)
            + px(rp3, x - 1) + px(rp3, x)
            + px(rp3, x + 1) + px(rp3, x + 2)
            + px(rm1, x - 2) + px(rc0, x - 2)
            + px(rc1, x - 2) + px(rp2, x - 2)
            + px(rm1, x + 3) + px(rc0, x + 3)
            + px(rc1, x + 3) + px(rp2, x + 3));
            // zig fmt: on
            saAct += @abs(f);
        }
    }
    return saAct;
}

/// Contiguous row load widened to activity lanes (no laundering: every load
/// in diff1st/diff2nd sits at a distinct row/x, so nothing can fuse).
inline fn vrow(comptime T: type, comptime vl: comptime_int, row: []const T, i: usize) @Vector(vl, LaneInt(T)) {
    return @intCast(@as(@Vector(vl, T), row[i..][0..vl].*));
}

// Temporal activity vs the previous frame, 2x2 block sums (large frames).
// has_prev=false means "previous frame is all zeros" (frame 0), matching the
// zero-initialized state buffers of the old implementation.
//
// Vector form: v = per-column vertical sums/diffs of the two rows, then
// v + shiftDown1(v) puts each 2x2 block sum in its even lane. Lanes never
// overflow (|t| <= 4*maxval: 1020 for u8 in i16, 262140 for u16 in i32), so
// the regrouping is exact.
inline fn diff1st(comptime T: type, comptime has_prev: bool, w_act: usize, h_act: usize, o_m0: []const T, o_p1: anytype, o: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const U = std.meta.Int(.unsigned, @bitSizeOf(L));
    const even_mask = comptime evenMask(U, vl);

    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        const c0 = o_m0[y * o ..];
        const c1 = o_m0[(y + 1) * o ..];
        var row_acc: @Vector(vl, u32) = @splat(0);
        var x: usize = 0;
        while (x + vl <= w_act) : (x += vl) {
            var v = vrow(T, vl, c0, x) + vrow(T, vl, c1, x);
            if (has_prev) {
                v -= vrow(T, vl, o_p1[y * o ..], x) + vrow(T, vl, o_p1[(y + 1) * o ..], x);
            }
            const t = v + shiftDown1(L, vl, v);
            row_acc += @intCast(@abs(t) & even_mask);
        }
        taAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc)));

        while (x < w_act) : (x += 2) {
            // zig fmt: off
            var t: i32 = @as(i32, o_m0[y * o + x]) + @as(i32, o_m0[y * o + x + 1])
            + @as(i32, o_m0[(y + 1) * o + x]) + @as(i32, o_m0[(y + 1) * o + x + 1]);
            if (has_prev) {
                t -= @as(i32, o_p1[y * o + x]) + @as(i32, o_p1[y * o + x + 1])
                + @as(i32, o_p1[(y + 1) * o + x]) + @as(i32, o_p1[(y + 1) * o + x + 1]);
            }
            // zig fmt: on
            taAct += @abs(t);
        }
    }
    return (taAct * XPSNR_GAMMA);
}

// Same skeleton as diff1st with a third frame and a doubled p1 term
// (|t| <= 8*maxval: 2040 for u8 in i16, 524280 for u16 in i32 -- exact).
inline fn diff2nd(comptime T: type, comptime has_p1: bool, comptime has_p2: bool, w_act: usize, h_act: usize, o_m0: []const T, o_p1: anytype, o_p2: anytype, o: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const LV = @Vector(vl, L);
    const U = std.meta.Int(.unsigned, @bitSizeOf(L));
    const even_mask = comptime evenMask(U, vl);
    const two: LV = @splat(2);

    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        const c0 = o_m0[y * o ..];
        const c1 = o_m0[(y + 1) * o ..];
        var row_acc: @Vector(vl, u32) = @splat(0);
        var x: usize = 0;
        while (x + vl <= w_act) : (x += vl) {
            var v = vrow(T, vl, c0, x) + vrow(T, vl, c1, x);
            if (has_p1) {
                v -= two * (vrow(T, vl, o_p1[y * o ..], x) + vrow(T, vl, o_p1[(y + 1) * o ..], x));
            }
            if (has_p2) {
                v += vrow(T, vl, o_p2[y * o ..], x) + vrow(T, vl, o_p2[(y + 1) * o ..], x);
            }
            const t = v + shiftDown1(L, vl, v);
            row_acc += @intCast(@abs(t) & even_mask);
        }
        taAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc)));

        while (x < w_act) : (x += 2) {
            // zig fmt: off
            var t: i32 = @as(i32, o_m0[y * o + x]) + @as(i32, o_m0[y * o + x + 1])
            + @as(i32, o_m0[(y + 1) * o + x]) + @as(i32, o_m0[(y + 1) * o + x + 1]);
            if (has_p1) {
                t -= 2 * (@as(i32, o_p1[y * o + x]) + @as(i32, o_p1[y * o + x + 1])
                + @as(i32, o_p1[(y + 1) * o + x]) + @as(i32, o_p1[(y + 1) * o + x + 1]));
            }
            if (has_p2) {
                t += @as(i32, o_p2[y * o + x]) + @as(i32, o_p2[y * o + x + 1])
                + @as(i32, o_p2[(y + 1) * o + x]) + @as(i32, o_p2[(y + 1) * o + x + 1]);
            }
            // zig fmt: on
            taAct += @abs(t);
        }
    }
    return (taAct * XPSNR_GAMMA);
}

// Temporal activity vs the previous frame, per pixel (small frames).
// u8 goes through vpsadbw (simd.sadU8x32, has_prev=false is SAD against
// zero); the sum-of-|diff| regrouping is exact integer math.
inline fn tempDiff1(comptime T: type, comptime has_prev: bool, block_width: usize, block_height: usize, o_m0: []const T, o_p1: anytype, o: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const LV = @Vector(vl, L);

    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < block_height) : (y += 1) {
        const cur = o_m0[y * o ..];
        var x: usize = 0;

        if (T == u8) {
            var acc: @Vector(4, u64) = @splat(0);
            while (x + 32 <= block_width) : (x += 32) {
                const c: @Vector(32, u8) = cur[x..][0..32].*;
                const p: @Vector(32, u8) = if (has_prev) o_p1[y * o + x ..][0..32].* else @splat(0);
                acc += simd.sadU8x32(c, p);
            }
            taAct += @reduce(.Add, acc) * XPSNR_GAMMA;
        } else {
            var row_acc: @Vector(vl, u32) = @splat(0);
            while (x + vl <= block_width) : (x += vl) {
                const c: LV = @intCast(@as(@Vector(vl, T), cur[x..][0..vl].*));
                const t: LV = if (has_prev) c - @as(LV, @intCast(@as(@Vector(vl, T), o_p1[y * o + x ..][0..vl].*))) else c;
                row_acc += @intCast(@abs(t));
            }
            taAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc))) * XPSNR_GAMMA;
        }

        while (x < block_width) : (x += 1) {
            const p: i32 = if (has_prev) @as(i32, o_p1[y * o + x]) else 0;
            const t: i32 = @as(i32, o_m0[y * o + x]) - p;
            taAct += @as(u64, XPSNR_GAMMA) * @as(u64, @abs(t));
        }
    }
    return taAct;
}

inline fn tempDiff2(comptime T: type, comptime has_p1: bool, comptime has_p2: bool, block_width: usize, block_height: usize, o_m0: []const T, o_p1: anytype, o_p2: anytype, o: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const LV = @Vector(vl, L);

    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < block_height) : (y += 1) {
        const cur = o_m0[y * o ..];
        var row_acc: @Vector(vl, u32) = @splat(0);
        var x: usize = 0;
        while (x + vl <= block_width) : (x += vl) {
            var t: LV = @intCast(@as(@Vector(vl, T), cur[x..][0..vl].*));
            if (has_p1) {
                const p1: LV = @intCast(@as(@Vector(vl, T), o_p1[y * o + x ..][0..vl].*));
                t -= @as(LV, @splat(2)) * p1;
            }
            if (has_p2) {
                t += @as(LV, @intCast(@as(@Vector(vl, T), o_p2[y * o + x ..][0..vl].*)));
            }
            row_acc += @intCast(@abs(t));
        }
        taAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc))) * XPSNR_GAMMA;

        while (x < block_width) : (x += 1) {
            var t: i32 = @as(i32, o_m0[y * o + x]);
            if (has_p1) t -= 2 * @as(i32, o_p1[y * o + x]);
            if (has_p2) t += @as(i32, o_p2[y * o + x]);
            taAct += @as(u64, XPSNR_GAMMA) * @as(u64, @abs(t));
        }
    }
    return taAct;
}

/// Spatial activity (3x3 Laplacian) over absolute picture coordinates
/// [x0, x1) x [y0, y1); the caller guarantees x0/y0 >= 1.
fn spatialAct(comptime T: type, pic: []const T, o: usize, x0: usize, x1: usize, y0: usize, y1: usize) u64 {
    const L = LaneInt(T);
    const vl = laneCount(T);
    const LV = @Vector(vl, L);
    const twelve: LV = @splat(12);
    const two: LV = @splat(2);

    var saAct: u64 = 0;
    var y: usize = y0;
    while (y < y1) : (y += 1) {
        const rm = pic[(y - 1) * o ..];
        const rc = pic[y * o ..];
        const rp = pic[(y + 1) * o ..];
        var row_acc: @Vector(vl, u32) = @splat(0);
        var x: usize = x0;
        while (x + vl <= x1) : (x += vl) {
            // Laundered taps: the +-1 overlapping loads per row otherwise
            // fuse into shuffle chains (~3x the instructions, measured).
            const c: LV = @intCast(simd.loaduOpaque(T, vl, rc[x..][0..vl]));
            const l: LV = @intCast(simd.loaduOpaque(T, vl, rc[x - 1 ..][0..vl]));
            const r: LV = @intCast(simd.loaduOpaque(T, vl, rc[x + 1 ..][0..vl]));
            const u: LV = @intCast(simd.loaduOpaque(T, vl, rm[x..][0..vl]));
            const d: LV = @intCast(simd.loaduOpaque(T, vl, rp[x..][0..vl]));
            const ul: LV = @intCast(simd.loaduOpaque(T, vl, rm[x - 1 ..][0..vl]));
            const ur: LV = @intCast(simd.loaduOpaque(T, vl, rm[x + 1 ..][0..vl]));
            const dl: LV = @intCast(simd.loaduOpaque(T, vl, rp[x - 1 ..][0..vl]));
            const dr: LV = @intCast(simd.loaduOpaque(T, vl, rp[x + 1 ..][0..vl]));
            const f = twelve * c - two * (l + r + u + d) - (ul + ur + dl + dr);
            row_acc += @intCast(@abs(f));
        }
        saAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc)));

        while (x < x1) : (x += 1) {
            const f: i32 = 12 * @as(i32, rc[x]) - 2 * (@as(i32, rc[x - 1]) + @as(i32, rc[x + 1]) +
                @as(i32, rm[x]) + @as(i32, rp[x])) - (@as(i32, rm[x - 1]) +
                @as(i32, rm[x + 1]) + @as(i32, rp[x - 1]) + @as(i32, rp[x + 1]));
            saAct += @abs(f);
        }
    }
    return saAct;
}

fn calcSquaredError(comptime T: type, blk_org: []const T, stride: usize, blk_rec: []const T, block_width: usize, block_height: usize) u64 {
    var sse: u64 = 0;
    var y: usize = 0;
    while (y < block_height) : (y += 1) {
        const org_row = blk_org[y * stride ..];
        const rec_row = blk_rec[y * stride ..];
        var x: usize = 0;

        if (T == u8) {
            // vpmaddwd square-accumulate: d0*d0 + d1*d1 per i32 lane is exact
            // for |d| <= 255 (see simd.maddwdI16x16).
            const vl = 16;
            var row_acc: @Vector(vl / 2, u32) = @splat(0);
            while (x + vl <= block_width) : (x += vl) {
                const o: @Vector(vl, u8) = org_row[x..][0..vl].*;
                const r: @Vector(vl, u8) = rec_row[x..][0..vl].*;
                const ad: @Vector(vl, i16) = @intCast(@max(o, r) - @min(o, r));
                row_acc += @as(@Vector(vl / 2, u32), @bitCast(simd.maddwdI16x16(ad, ad)));
            }
            sse += @reduce(.Add, @as(@Vector(vl / 2, u64), @intCast(row_acc)));
        } else {
            const vl = 8;
            var acc: @Vector(vl, u64) = @splat(0);
            while (x + vl <= block_width) : (x += vl) {
                const o: @Vector(vl, T) = org_row[x..][0..vl].*;
                const r: @Vector(vl, T) = rec_row[x..][0..vl].*;
                const ad: @Vector(vl, u64) = @intCast(@max(o, r) - @min(o, r));
                acc += ad * ad;
            }
            sse += @reduce(.Add, acc);
        }

        while (x < block_width) : (x += 1) {
            const err: i64 = @as(i32, org_row[x]) - @as(i32, rec_row[x]);
            sse += math.lossyCast(u64, err * err);
        }
    }

    return sse;
}

inline fn calcSquaredErrorAndWeight(
    comptime T: type,
    pic_org: []const T,
    stride: usize,
    pic_rec: []const T,
    pic_prv1: ?[]const T,
    pic_prv2: ?[]const T,
    offset_x: usize,
    offset_y: usize,
    block_width: usize,
    block_height: usize,
    depth: u6,
    frame_rate: u32,
    ms_act: *f64,
    width: [3]u32,
    height: [3]u32,
    temporal: bool,
) f64 {
    const uo: usize = stride;
    const w0: usize = width[0];
    const h0: usize = height[0];
    const o_m0 = pic_org[(offset_y * uo + offset_x)..];
    const p_m1: ?[]const T = if (pic_prv1) |p| p[(offset_y * uo + offset_x)..] else null;
    const p_m2: ?[]const T = if (pic_prv2) |p| p[(offset_y * uo + offset_x)..] else null;
    const r_m0 = pic_rec[(offset_y * uo + offset_x)..];

    const b_val: i64 = if ((w0 * h0) > (2048 * 1152)) 2 else 1;
    const bw: i64 = @intCast(block_width);
    const bh: i64 = @intCast(block_height);
    const x_act: i64 = if (offset_x > 0) 0 else b_val;
    const y_act: i64 = if (offset_y > 0) 0 else b_val;
    const w_act: i64 = if ((offset_x + block_width) < w0) bw else (bw - b_val);
    const h_act: i64 = if ((offset_y + block_height) < h0) bh else (bh - b_val);

    const sse: f64 = @floatFromInt(calcSquaredError(T, o_m0, stride, r_m0, block_width, block_height));

    var saAct: u64 = 0;
    var taAct: u64 = 0;

    if ((w_act <= x_act) or (h_act <= y_act)) {
        return sse;
    }

    const xa: usize = @intCast(x_act);
    const ya: usize = @intCast(y_act);
    const wa: usize = @intCast(w_act);
    const ha: usize = @intCast(h_act);

    if (b_val > 1) {
        if (w_act > 12) {
            saAct = highds(T, pic_org, uo, offset_x + xa, offset_x + wa, offset_y + ya, offset_y + ha);
        }
    } else {
        saAct = spatialAct(T, pic_org, uo, offset_x + xa, offset_x + wa, offset_y + ya, offset_y + ha);
    }

    ms_act.* = @as(f64, @floatFromInt(saAct)) / (@as(f64, @floatFromInt(wa - xa)) * @as(f64, @floatFromInt(ha - ya)));

    if (temporal) {
        if (b_val > 1) {
            if (frame_rate < 32) {
                taAct = if (p_m1) |pm1|
                    diff1st(T, true, block_width, block_height, o_m0, pm1, uo)
                else
                    diff1st(T, false, block_width, block_height, o_m0, {}, uo);
            } else {
                if (p_m1) |pm1| {
                    taAct = if (p_m2) |pm2|
                        diff2nd(T, true, true, block_width, block_height, o_m0, pm1, pm2, uo)
                    else
                        diff2nd(T, true, false, block_width, block_height, o_m0, pm1, {}, uo);
                } else {
                    taAct = diff2nd(T, false, false, block_width, block_height, o_m0, {}, {}, uo);
                }
            }
        } else {
            if (frame_rate < 32) {
                taAct = if (p_m1) |pm1|
                    tempDiff1(T, true, block_width, block_height, o_m0, pm1, uo)
                else
                    tempDiff1(T, false, block_width, block_height, o_m0, {}, uo);
            } else {
                if (p_m1) |pm1| {
                    taAct = if (p_m2) |pm2|
                        tempDiff2(T, true, true, block_width, block_height, o_m0, pm1, pm2, uo)
                    else
                        tempDiff2(T, true, false, block_width, block_height, o_m0, pm1, {}, uo);
                } else {
                    taAct = tempDiff2(T, false, false, block_width, block_height, o_m0, {}, {}, uo);
                }
            }
        }

        ms_act.* += @as(f64, @floatFromInt(taAct)) / (@as(f64, @floatFromInt(block_width)) * @as(f64, @floatFromInt(block_height)));
    }

    const sft: usize = @as(usize, 1) << (depth - 6);
    if (ms_act.* < @as(f64, @floatFromInt(sft))) {
        ms_act.* = @as(f64, @floatFromInt(sft));
    }

    ms_act.* *= ms_act.*;

    return sse;
}

pub fn getAvgXPSNR(sqrt_wsse_val: f64, sum_xpsnr_val: f64, width: u64, height: u64, max_error_64: u64, num_frames_64: u64) f64 {
    const num_frames_64f: f64 = @floatFromInt(num_frames_64);
    if (sqrt_wsse_val >= num_frames_64f) {
        const avg_dist: f64 = sqrt_wsse_val / num_frames_64f;
        const num64: f64 = @floatFromInt(width * height * max_error_64);
        return @as(f64, 10.0) * @log10(num64 / (avg_dist * avg_dist));
    }

    return sum_xpsnr_val / num_frames_64f;
}

pub fn getFrameXPSNR(sqrt_wsse: f64, width: u64, height: u64, max_error_64: u64) f64 {
    if (sqrt_wsse < 1) return math.inf(f64);
    const num64: f64 = @floatFromInt(width * height * max_error_64);
    return @as(f64, 10.0) * @log10(num64 / (sqrt_wsse * sqrt_wsse));
}

pub fn getWSSE(
    comptime T: type,
    orgp: [3][]const T,
    recp: [3][]const T,
    prv1: ?[]const T,
    prv2: ?[]const T,
    wsse64: []u64,
    width: [3]u32,
    height: [3]u32,
    strides: [3]u32,
    depth: u6,
    num_comps: u8,
    frame_rate: u32,
    temporal: bool,
) void {
    const w: u32 = width[0];
    const h: u32 = height[0];
    const wh: u32 = w * h;

    const r: f64 = @as(f64, @floatFromInt(wh)) / @as(f64, 3840.0 * 2160.0);
    const b: u32 = math.lossyCast(u32, (32.0 * @sqrt(r) + 0.5)) * 4;
    const w_blk: u32 = if (b >= 4) (w + b - 1) / b else 0;
    const h_blk: u32 = if (b >= 4) (h + b - 1) / b else 0;
    const sft: u32 = math.shl(u32, 1, (2 * depth - 9));
    const avg_act: f64 = @sqrt(16.0 * @as(f64, @floatFromInt(sft)) / @sqrt(@max(0.00001, r)));

    const sse_luma = allocator.alloc(f64, w_blk * h_blk) catch unreachable;
    const weights = allocator.alloc(f64, w_blk * h_blk) catch unreachable;
    defer allocator.free(sse_luma);
    defer allocator.free(weights);

    var y: usize = 0;
    var x: usize = 0;
    var idx_blk: usize = 0;

    if (b >= 4) {
        const stride: u32 = strides[0];
        var wsse_luma: f64 = 0.0;

        y = 0;
        while (y < h) : (y += b) {
            const uy: u32 = @intCast(y);
            const block_height: u32 = if (y + b > h) (h - uy) else b;

            x = 0;
            while (x < w) : ({
                x += b;
                idx_blk += 1;
            }) {
                const ux: u32 = @intCast(x);
                const block_width: u32 = if (x + b > w) (w - ux) else b;
                var ms_act: f64 = 1.0;
                var ms_act_prev: f64 = 0.0;
                sse_luma[idx_blk] = calcSquaredErrorAndWeight(
                    T,
                    orgp[0],
                    stride,
                    recp[0],
                    prv1,
                    prv2,
                    x,
                    y,
                    block_width,
                    block_height,
                    depth,
                    frame_rate,
                    &ms_act,
                    width,
                    height,
                    temporal,
                );

                weights[idx_blk] = 1.0 / @sqrt(ms_act);

                if (wh <= (640 * 480)) {
                    if (x == 0) {
                        ms_act_prev = if (idx_blk > 1) weights[idx_blk - 2] else 0;
                    } else {
                        ms_act_prev = if (x > b) @max(weights[idx_blk - 2], weights[idx_blk]) else weights[idx_blk];
                    }
                    if (idx_blk > w_blk) {
                        ms_act_prev = @max(ms_act_prev, weights[idx_blk - 1 - w_blk]);
                    }
                    if ((idx_blk > 0) and (weights[idx_blk - 1] > ms_act_prev)) {
                        weights[idx_blk - 1] = ms_act_prev;
                    }
                    if ((x + b >= w) and (y + b >= h) and (idx_blk > w_blk)) {
                        ms_act_prev = @max(weights[idx_blk - 1], weights[idx_blk - w_blk]);
                        if (weights[idx_blk] > ms_act_prev) {
                            weights[idx_blk] = ms_act_prev;
                        }
                    }
                }
            }
        }

        y = 0;
        idx_blk = 0;
        while (y < h) : (y += b) {
            x = 0;
            while (x < w) : ({
                x += b;
                idx_blk += 1;
            }) {
                wsse_luma += sse_luma[idx_blk] * weights[idx_blk];
            }
        }
        wsse64[0] = if (wsse_luma <= 0.0) 0 else @as(u64, @trunc(wsse_luma * avg_act + 0.5));
    }

    var c: usize = 0;
    while (c < num_comps) : (c += 1) {
        const stride: u32 = strides[c];
        const w_pln: u32 = width[c];
        const h_pln: u32 = height[c];

        if (b < 4) {
            wsse64[c] = calcSquaredError(T, orgp[c], stride, recp[c], w_pln, h_pln);
        } else if (c > 0) {
            const bx: u32 = (b * w_pln) / w;
            const by: u32 = (b * h_pln) / h;
            var wsse_chroma: f64 = 0.0;
            y = 0;
            idx_blk = 0;
            while (y < h_pln) : (y += by) {
                const block_height: usize = if (y + by > h_pln) (@as(usize, h_pln) - y) else by;
                x = 0;
                while (x < w_pln) : ({
                    x += bx;
                    idx_blk += 1;
                }) {
                    const block_width: usize = if (x + bx > w_pln) (@as(usize, w_pln) - x) else bx;
                    const uwsse_chroma: u64 = calcSquaredError(
                        T,
                        orgp[c][(y * stride + x)..],
                        stride,
                        recp[c][(y * stride + x)..],
                        block_width,
                        block_height,
                    );

                    wsse_chroma += @as(f64, @floatFromInt(uwsse_chroma)) * weights[idx_blk];
                }
            }

            wsse64[c] = if (wsse_chroma <= 0.0) 0 else math.lossyCast(u64, (wsse_chroma * avg_act + 0.5));
        }
    }
}
