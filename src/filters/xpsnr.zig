const std = @import("std");
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

fn get(sls: anytype, dist: i32) i32 {
    const ptr = sls.ptr;
    const negatv = dist < 0;
    const shift: u32 = if (negatv) @intCast(~(dist - 1)) else @intCast(dist);
    const ptr2 = if (negatv) ptr - shift else ptr + shift;
    return @intCast(ptr2[0]);
}

fn highds(comptime T: type, x_act: usize, y_act: usize, w_act: usize, h_act: usize, o_m0: []const T, o: usize) u64 {
    var saAct: u64 = 0;

    const oi: i32 = @intCast(o);
    var uy: usize = y_act;
    while (uy < h_act) : (uy += 2) {
        const y: i32 = @intCast(uy);
        var ux: usize = x_act;
        while (ux < w_act) : (ux += 2) {
            const x: i32 = @intCast(ux);
            const base: i32 = y * oi + x;
            // zig fmt: off
            const f: i32 = 12 * (get(o_m0, base) + get(o_m0, base + 1)
            + get(o_m0, base + oi) + get(o_m0, base + oi + 1))
            - 3 * (get(o_m0, base - oi) + get(o_m0, base - oi + 1)
            + get(o_m0, base + 2 * oi) + get(o_m0, base + 2 * oi + 1))
            - 3 * (get(o_m0, base - 1) + get(o_m0, base + 2)
            + get(o_m0, base + oi - 1) + get(o_m0, base + oi + 2))
            - 2 * (get(o_m0, base - oi - 1) + get(o_m0, base - oi + 2)
            + get(o_m0, base + 2 * oi - 1) + get(o_m0, base + 2 * oi + 2))
            - (get(o_m0, base - 2 * oi - 1) + get(o_m0, base - 2 * oi)
            + get(o_m0, base - 2 * oi + 1) + get(o_m0, base - 2 * oi + 2)
            + get(o_m0, base + 3 * oi - 1) + get(o_m0, base + 3 * oi)
            + get(o_m0, base + 3 * oi + 1) + get(o_m0, base + 3 * oi + 2)
            + get(o_m0, base - oi - 2) + get(o_m0, base - 2)
            + get(o_m0, base + oi - 2) + get(o_m0, base + 2 * oi - 2)
            + get(o_m0, base - oi + 3) + get(o_m0, base + 3)
            + get(o_m0, base + oi + 3) + get(o_m0, base + 2 * oi + 3));
            // zig fmt: on
            saAct += @abs(f);
        }
    }
    return saAct;
}

// Temporal activity vs the previous frame, 2x2 block sums (large frames).
// has_prev=false means "previous frame is all zeros" (frame 0), matching the
// zero-initialized state buffers of the old implementation.
inline fn diff1st(comptime T: type, comptime has_prev: bool, w_act: usize, h_act: usize, o_m0: []const T, o_p1: anytype, o: usize) u64 {
    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        var x: usize = 0;
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

inline fn diff2nd(comptime T: type, comptime has_p1: bool, comptime has_p2: bool, w_act: usize, h_act: usize, o_m0: []const T, o_p1: anytype, o_p2: anytype, o: usize) u64 {
    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        var x: usize = 0;
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
inline fn tempDiff1(comptime T: type, comptime has_prev: bool, block_width: usize, block_height: usize, o_m0: []const T, o_p1: anytype, o: usize) u64 {
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
            const c: LV = @intCast(@as(@Vector(vl, T), cur[x..][0..vl].*));
            const t: LV = if (has_prev) c - @as(LV, @intCast(@as(@Vector(vl, T), o_p1[y * o + x ..][0..vl].*))) else c;
            row_acc += @intCast(@abs(t));
        }
        taAct += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc))) * XPSNR_GAMMA;

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
            const c: LV = @intCast(@as(@Vector(vl, T), rc[x..][0..vl].*));
            const l: LV = @intCast(@as(@Vector(vl, T), rc[x - 1 ..][0..vl].*));
            const r: LV = @intCast(@as(@Vector(vl, T), rc[x + 1 ..][0..vl].*));
            const u: LV = @intCast(@as(@Vector(vl, T), rm[x..][0..vl].*));
            const d: LV = @intCast(@as(@Vector(vl, T), rp[x..][0..vl].*));
            const ul: LV = @intCast(@as(@Vector(vl, T), rm[x - 1 ..][0..vl].*));
            const ur: LV = @intCast(@as(@Vector(vl, T), rm[x + 1 ..][0..vl].*));
            const dl: LV = @intCast(@as(@Vector(vl, T), rp[x - 1 ..][0..vl].*));
            const dr: LV = @intCast(@as(@Vector(vl, T), rp[x + 1 ..][0..vl].*));
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
            const vl = 16;
            var row_acc: @Vector(vl, u32) = @splat(0);
            while (x + vl <= block_width) : (x += vl) {
                const o: @Vector(vl, u8) = org_row[x..][0..vl].*;
                const r: @Vector(vl, u8) = rec_row[x..][0..vl].*;
                const ad: @Vector(vl, u16) = @intCast(@max(o, r) - @min(o, r));
                row_acc += @intCast(ad * ad);
            }
            sse += @reduce(.Add, @as(@Vector(vl, u64), @intCast(row_acc)));
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
    const b_val: usize = if ((w0 * h0) > (2048 * 1152)) 2 else 1;
    const x_act: usize = if (offset_x > 0) 0 else b_val;
    const y_act: usize = if (offset_y > 0) 0 else b_val;
    const w_act: usize = if ((offset_x + block_width) < w0) block_width else (block_width - b_val);
    const h_act: usize = if ((offset_y + block_height) < h0) block_height else (block_height - b_val);

    const sse: f64 = @floatFromInt(calcSquaredError(T, o_m0, stride, r_m0, block_width, block_height));

    var saAct: u64 = 0;
    var taAct: u64 = 0;

    if ((w_act <= x_act) or (h_act <= y_act)) {
        return sse;
    }

    if (b_val > 1) {
        saAct = highds(T, x_act, y_act, w_act, h_act, o_m0, uo);
    } else {
        saAct = spatialAct(T, pic_org, uo, offset_x + x_act, offset_x + w_act, offset_y + y_act, offset_y + h_act);
    }

    ms_act.* = @as(f64, @floatFromInt(saAct)) / (@as(f64, @floatFromInt(w_act - x_act)) * @as(f64, @floatFromInt(h_act - y_act)));

    if (temporal) {
        if (b_val > 1) {
            if (frame_rate <= 32) {
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
            if (frame_rate <= 32) {
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
