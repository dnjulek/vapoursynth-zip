const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const XPSNR_GAMMA: u32 = 2;

fn get(sls: anytype, dist: i32) i32 {
    const ptr = sls.ptr;
    const negatv = dist < 0;
    const shift: u32 = if (negatv) @intCast(~(dist - 1)) else @intCast(dist);
    const ptr2 = if (negatv) ptr - shift else ptr + shift;
    return @intCast(ptr2[0]);
}

fn highds(comptime T: type, x_act: usize, y_act: usize, w_act: usize, h_act: usize, o_m0: []const T, o: usize) u64 {
    var saAct: u64 = 0;
    var y: usize = y_act;
    while (y < h_act) : (y += 2) {
        var x: usize = x_act;
        while (x < w_act) : (x += 2) {
            // zig fmt: off
            const f: i32 = 12 * (@as(i32, o_m0[y * o + x]) + @as(i32, o_m0[y * o + x + 1])
            + @as(i32, o_m0[(y + 1) * o + x]) + @as(i32, o_m0[(y + 1) * o + x + 1]))
            - 3 * (@as(i32, o_m0[(y - 1) * o + x]) + @as(i32, o_m0[(y - 1) * o + x + 1])
            + @as(i32, o_m0[(y + 2) * o + x]) + @as(i32, o_m0[(y + 2) * o + x + 1]))
            - 3 * (@as(i32, o_m0[y * o + x - 1]) + @as(i32, o_m0[y * o + x + 2])
            + @as(i32, o_m0[(y + 1) * o + x - 1]) + @as(i32, o_m0[(y + 1) * o + x + 2]))
            - 2 * (@as(i32, o_m0[(y - 1) * o + x - 1]) + @as(i32, o_m0[(y - 1) * o + x + 2])
            + @as(i32, o_m0[(y + 2) * o + x - 1]) + @as(i32, o_m0[(y + 2) * o + x + 2]))
            - (@as(i32, o_m0[(y - 2) * o + x - 1]) + @as(i32, o_m0[(y - 2) * o + x])
            + @as(i32, o_m0[(y - 2) * o + x + 1]) + @as(i32, o_m0[(y - 2) * o + x + 2]))
            + @as(i32, o_m0[(y + 3) * o + x - 1]) + @as(i32, o_m0[(y + 3) * o + x])
            + @as(i32, o_m0[(y + 3) * o + x + 1]) + @as(i32, o_m0[(y + 3) * o + x + 2])
            + @as(i32, o_m0[(y - 1) * o + x - 2]) + @as(i32, o_m0[y * o + x - 2])
            + @as(i32, o_m0[(y + 1) * o + x - 2]) + @as(i32, o_m0[(y + 2) * o + x - 2])
            + @as(i32, o_m0[(y - 1) * o + x + 3]) + @as(i32, o_m0[y * o + x + 3]) 
            + @as(i32, o_m0[(y + 1) * o + x + 3]) + @as(i32, o_m0[(y + 2) * o + x + 3]);
            // zig fmt: on
            saAct += @abs(f);
        }
    }
    return saAct;
}

inline fn diff1st(comptime T: type, w_act: usize, h_act: usize, o_m0: []const T, o_m1: []i16, o: usize) u64 {
    var taAct: u64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        var x: usize = 0;
        while (x < w_act) : (x += 2) {
            // zig fmt: off
            const t: i32 = @as(i32, o_m0[y * o + x]) + @as(i32, o_m0[y * o + x + 1]) 
            + @as(i32, o_m0[(y + 1) * o + x]) + @as(i32, o_m0[(y + 1) * o + x + 1]) 
            - (@as(i32, o_m1[y * o + x]) + @as(i32, o_m1[y * o + x + 1]) 
            + @as(i32, o_m1[(y + 1) * o + x]) + @as(i32, o_m1[(y + 1) * o + x + 1]));
            // zig fmt: on
            taAct += @abs(t);
            o_m1[y * o + x] = @intCast(o_m0[y * o + x]);
            o_m1[(y + 1) * o + x] = @intCast(o_m0[(y + 1) * o + x]);
            o_m1[y * o + x + 1] = @intCast(o_m0[y * o + x + 1]);
            o_m1[(y + 1) * o + x + 1] = @intCast(o_m0[(y + 1) * o + x + 1]);
        }
    }
    return (taAct * XPSNR_GAMMA);
}

inline fn diff2nd(comptime T: type, w_act: usize, h_act: usize, o_m0: []const T, o_m1: []i16, o_m2: []i16, o: usize) u64 {
    var taAct: u64 = 0;
    var tt: i64 = 0;
    var y: usize = 0;
    while (y < h_act) : (y += 2) {
        var x: usize = 0;
        while (x < w_act) : (x += 2) {
            const t: i32 = @as(i32, o_m0[y * o + x]) + @as(i32, o_m0[y * o + x + 1]) + @as(i32, o_m0[(y + 1) * o + x]) + @as(i32, o_m0[(y + 1) * o + x + 1]) - 2 *
                (@as(i32, o_m1[y * o + x]) + @as(i32, o_m1[y * o + x + 1]) + @as(i32, o_m1[(y + 1) * o + x]) + @as(i32, o_m1[(y + 1) * o + x + 1])) +
                @as(i32, o_m2[y * o + x]) + @as(i32, o_m2[y * o + x + 1]) + @as(i32, o_m2[(y + 1) * o + x]) + @as(i32, o_m2[(y + 1) * o + x + 1]);

            taAct += @abs(t);

            tt += @as(i32, o_m1[(y + 1) * o + x + 1]);

            o_m2[y * o + x] = o_m1[y * o + x];
            o_m2[(y + 1) * o + x] = o_m1[(y + 1) * o + x];
            o_m2[y * o + x + 1] = o_m1[y * o + x + 1];
            o_m2[(y + 1) * o + x + 1] = o_m1[(y + 1) * o + x + 1];
            o_m1[y * o + x] = @intCast(o_m0[y * o + x]);
            o_m1[(y + 1) * o + x] = @intCast(o_m0[(y + 1) * o + x]);
            o_m1[y * o + x + 1] = @intCast(o_m0[y * o + x + 1]);
            o_m1[(y + 1) * o + x + 1] = @intCast(o_m0[(y + 1) * o + x + 1]);
        }
    }

    return (taAct * XPSNR_GAMMA);
}

fn calcSquaredError(comptime T: type, blk_org: []const T, stride: usize, blk_rec: []const T, block_width: usize, block_height: usize) u64 {
    var sse: u64 = 0;
    var y: usize = 0;
    while (y < block_height) : (y += 1) {
        var x: usize = 0;
        while (x < block_width) : (x += 1) {
            const err: i64 = @as(i32, blk_org[y * stride + x]) - @as(i32, blk_rec[y * stride + x]);
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
    pic_org_m1: []i16,
    pic_org_m2: []i16,
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
    const o: i32 = @intCast(uo);
    const w0: usize = width[0];
    const h0: usize = height[0];
    const o_m0 = pic_org[(offset_y * uo + offset_x)..];
    const o_m1 = pic_org_m1[(offset_y * uo + offset_x)..];
    const o_m2 = pic_org_m2[(offset_y * uo + offset_x)..];
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
        var uy = y_act;
        while (uy < h_act) : (uy += 1) {
            const y: i32 = @intCast(uy);
            var ux = x_act;
            while (ux < w_act) : (ux += 1) {
                const x: i32 = @intCast(ux);
                const f: i32 = 12 * get(o_m0, y * o + x) - 2 * (get(o_m0, y * o + x - 1) + get(o_m0, y * o + x + 1) +
                    get(o_m0, (y - 1) * o + x) + get(o_m0, (y + 1) * o + x)) - (get(o_m0, (y - 1) * o + x - 1) +
                    get(o_m0, (y - 1) * o + x + 1) + get(o_m0, (y + 1) * o + x - 1) + get(o_m0, (y + 1) * o + x + 1));
                saAct += @abs(f);
            }
        }
    }

    ms_act.* = @as(f64, @floatFromInt(saAct)) / (@as(f64, @floatFromInt(w_act - x_act)) * @as(f64, @floatFromInt(h_act - y_act)));

    if (temporal) {
        if (b_val > 1) {
            if (frame_rate <= 32) {
                taAct = diff1st(T, block_width, block_height, o_m0, o_m1, uo);
            } else {
                taAct = diff2nd(T, block_width, block_height, o_m0, o_m1, o_m2, uo);
            }
        } else {
            if (frame_rate <= 32) {
                var y: usize = 0;
                while (y < block_height) : (y += 1) {
                    var x: usize = 0;
                    while (x < block_width) : (x += 1) {
                        const t: i32 = @as(i32, o_m0[y * uo + x]) - @as(i32, o_m1[y * uo + x]);
                        taAct += @as(u64, XPSNR_GAMMA) * @as(u64, @abs(t));
                        o_m1[y * uo + x] = @intCast(o_m0[y * uo + x]);
                    }
                }
            } else {
                var y: usize = 0;
                while (y < block_height) : (y += 1) {
                    var x: usize = 0;
                    while (x < block_width) : (x += 1) {
                        const t: i32 = @as(i32, o_m0[y * uo + x]) - 2 * @as(i32, o_m1[y * uo + x]) + @as(i32, o_m2[y * uo + x]);
                        taAct += @as(u64, XPSNR_GAMMA) * @as(u64, @abs(t));
                        o_m2[y * uo + x] = o_m1[y * uo + x];
                        o_m1[y * uo + x] = @intCast(o_m0[y * uo + x]);
                    }
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
    og_m1: []i16,
    og_m2: []i16,
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
    const w_blk: u32 = (w + b - 1) / b;
    const h_blk: u32 = (h + b - 1) / b;
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
                    og_m1,
                    og_m2,
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
        wsse64[0] = if (wsse_luma <= 0.0) 0 else @as(u64, @intFromFloat(wsse_luma * avg_act + 0.5));
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
