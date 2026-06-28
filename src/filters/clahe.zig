const std = @import("std");

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(i32) orelse 8;
const I32V = @Vector(vec_len, i32);
const U32V = @Vector(vec_len, u32);
const F32V = @Vector(vec_len, f32);

// sub-histograms per tile: pixels scatter round-robin so runs of equal values
// don't serialize on a single bin's load-add-store chain
const n_sub = 4;

pub fn applyCLAHE(
    comptime T: type,
    srcp: []const T,
    dstp: []T,
    stride: u32,
    width: u32,
    height: u32,
    limit: u32,
    tiles: []u32,
) void {
    const hist_size: u32 = @as(u32, 1) << @as(u32, @typeInfo(T).int.bits);
    const peak: f32 = @floatFromInt(hist_size - 1);
    const tile_width: u32 = width / tiles[0];
    const tile_height: u32 = height / tiles[1];
    const tile_size_total: u64 = @as(u64, tile_width) * tile_height;
    const lut_scale: f32 = peak / @as(f32, @floatFromInt(tile_size_total));
    var clip_limit: i32 = @intCast(@as(u64, limit) * tile_size_total / hist_size);
    clip_limit = @max(clip_limit, 1);

    const lut = allocator.alloc(T, tiles[0] * tiles[1] * hist_size) catch unreachable;
    defer allocator.free(lut);

    calcLut(T, srcp, stride, width, height, lut, tile_width, tile_height, tiles, clip_limit, lut_scale);
    interpolate(T, srcp, dstp, stride, width, height, lut, tile_width, tile_height, tiles);
}

fn calcLut(
    comptime T: type,
    srcp: []const T,
    stride: u32,
    width: u32,
    height: u32,
    lut: []T,
    tile_width: u32,
    tile_height: u32,
    tiles: []u32,
    clip_limit: i32,
    lut_scale: f32,
) void {
    const hist_size: u32 = @as(u32, 1) << @as(u32, @typeInfo(T).int.bits);
    const hist_sizei: i32 = @intCast(hist_size);
    const tiles_x = tiles[0];
    const tiles_y = tiles[1];

    const subs = allocator.alloc(u32, n_sub * hist_size) catch unreachable;
    defer allocator.free(subs);
    const tile_hist = allocator.alloc(i32, hist_size) catch unreachable;
    defer allocator.free(tile_hist);

    var ty: u32 = 0;
    while (ty < tiles_y) : (ty += 1) {
        var tx: u32 = 0;
        while (tx < tiles_x) : (tx += 1) {
            @memset(subs, 0);
            const h0 = subs[0 * hist_size ..][0..hist_size];
            const h1 = subs[1 * hist_size ..][0..hist_size];
            const h2 = subs[2 * hist_size ..][0..hist_size];
            const h3 = subs[3 * hist_size ..][0..hist_size];

            var y: u32 = ty * tile_height;
            while (y < @min((ty + 1) * tile_height, height)) : (y += 1) {
                const row = srcp[y * stride ..];
                var x: u32 = tx * tile_width;
                const xe = @min((tx + 1) * tile_width, width);
                while (x + 4 <= xe) : (x += 4) {
                    h0[row[x]] += 1;
                    h1[row[x + 1]] += 1;
                    h2[row[x + 2]] += 1;
                    h3[row[x + 3]] += 1;
                }
                while (x < xe) : (x += 1) {
                    h0[row[x]] += 1;
                }
            }

            // merge sub-histograms
            {
                var i: u32 = 0;
                while (i < hist_size) : (i += vec_len) {
                    const a: U32V = h0[i..][0..vec_len].*;
                    const b: U32V = h1[i..][0..vec_len].*;
                    const c: U32V = h2[i..][0..vec_len].*;
                    const d: U32V = h3[i..][0..vec_len].*;
                    const sum: I32V = @intCast(a + b + c + d);
                    tile_hist[i..][0..vec_len].* = sum;
                }
            }

            if (clip_limit > 0) {
                const limitv: I32V = @splat(clip_limit);
                const zerov: I32V = @splat(0);
                var clipped_acc: I32V = @splat(0);
                var i: u32 = 0;
                while (i < hist_size) : (i += vec_len) {
                    const b: I32V = tile_hist[i..][0..vec_len].*;
                    clipped_acc += @max(b - limitv, zerov);
                    tile_hist[i..][0..vec_len].* = @min(b, limitv);
                }
                const clipped: i32 = @reduce(.Add, clipped_acc);

                const redist_batch: i32 = @divTrunc(clipped, hist_sizei);
                var residual: i32 = clipped - redist_batch * hist_sizei;

                if (redist_batch != 0) {
                    const batchv: I32V = @splat(redist_batch);
                    i = 0;
                    while (i < hist_size) : (i += vec_len) {
                        const b: I32V = tile_hist[i..][0..vec_len].*;
                        tile_hist[i..][0..vec_len].* = b + batchv;
                    }
                }

                if (residual != 0) {
                    const residual_step = @max(@divTrunc(hist_sizei, residual), 1);
                    i = 0;
                    while ((i < hist_size) and (residual > 0)) : (i += @intCast(residual_step)) {
                        tile_hist[i] += 1;
                        residual -= 1;
                    }
                }
            }

            // cumulative sum -> scaled LUT row
            {
                const lut_row = lut[(ty * tiles_x + tx) * hist_size ..][0..hist_size];
                const scalev: F32V = @splat(lut_scale);
                const halfv: F32V = @splat(0.5);
                var run: i32 = 0;
                var i: u32 = 0;
                while (i < hist_size) : (i += vec_len) {
                    const v: I32V = tile_hist[i..][0..vec_len].*;
                    var p = prefixSum(v);
                    p += @as(I32V, @splat(run));
                    run = p[vec_len - 1];
                    const f: F32V = @floatFromInt(p);
                    const out: @Vector(vec_len, T) = @intFromFloat(@trunc(f * scalev + halfv));
                    lut_row[i..][0..vec_len].* = out;
                }
            }
        }
    }
}

/// In-register inclusive prefix sum.
inline fn prefixSum(v: I32V) I32V {
    const zero: I32V = @splat(0);
    var s = v;
    comptime var k: u32 = 1;
    inline while (k < vec_len) : (k *= 2) {
        const mask = comptime blk: {
            var m: [vec_len]i32 = undefined;
            for (0..vec_len) |i| {
                // lanes below k take zeros from the second operand
                m[i] = if (i < k) ~@as(i32, @intCast(i)) else @intCast(i - k);
            }
            break :blk m;
        };
        s += @shuffle(i32, s, zero, mask);
    }
    return s;
}

fn interpolate(
    comptime T: type,
    srcp: []const T,
    dstp: []T,
    stride: u32,
    width: u32,
    height: u32,
    lut: []const T,
    tile_width: u32,
    tile_height: u32,
    tiles: []u32,
) void {
    const hist_size: u32 = @as(u32, 1) << @as(u32, @typeInfo(T).int.bits);
    const tiles_x: i32 = @intCast(tiles[0]);
    const tiles_y: i32 = @intCast(tiles[1]);

    const inv_tw: f32 = 1.0 / @as(f32, @floatFromInt(tile_width));
    const inv_th: f32 = 1.0 / @as(f32, @floatFromInt(tile_height));

    // x-dependent terms are the same on every row; precompute them once
    const tx1h = allocator.alloc(u32, width) catch unreachable;
    defer allocator.free(tx1h);
    const tx2h = allocator.alloc(u32, width) catch unreachable;
    defer allocator.free(tx2h);
    const xa_arr = allocator.alloc(f32, width) catch unreachable;
    defer allocator.free(xa_arr);

    {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const txf: f32 = @as(f32, @floatFromInt(x)) * inv_tw - 0.5;
            const _tx1: i32 = @floor(txf);
            const _tx2: i32 = _tx1 + 1;
            xa_arr[x] = txf - @as(f32, @floatFromInt(_tx1));
            const tx1 = @min(@max(_tx1, 0), tiles_x - 1);
            const tx2 = @min(_tx2, tiles_x - 1);
            tx1h[x] = @as(u32, @intCast(tx1)) * hist_size;
            tx2h[x] = @as(u32, @intCast(tx2)) * hist_size;
        }
    }

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const tyf: f32 = @as(f32, @floatFromInt(y)) * inv_th - 0.5;
        var ty1: i32 = @floor(tyf);
        var ty2: i32 = ty1 + 1;
        const ya: f32 = tyf - @as(f32, @floatFromInt(ty1));

        ty1 = @min(@max(ty1, 0), tiles_y - 1);
        ty2 = @min(ty2, tiles_y - 1);
        const lut_p1h: u32 = @as(u32, @intCast(ty1 * tiles_x)) * hist_size;
        const lut_p2h: u32 = @as(u32, @intCast(ty2 * tiles_x)) * hist_size;

        const src_row = srcp[y * stride ..];
        const dst_row = dstp[y * stride ..];
        const onev: F32V = @splat(1.0);
        const yav: F32V = @splat(ya);
        const one_m_yav: F32V = @splat(1 - ya);
        const halfv: F32V = @splat(0.5);
        const p1v: U32V = @splat(lut_p1h);
        const p2v: U32V = @splat(lut_p2h);

        var x: u32 = 0;
        while (x + vec_len <= width) : (x += vec_len) {
            const sv: U32V = @intCast(@as(@Vector(vec_len, T), src_row[x..][0..vec_len].*));
            const t1: U32V = tx1h[x..][0..vec_len].*;
            const t2: U32V = tx2h[x..][0..vec_len].*;
            const idx0 = p1v + t1 + sv;
            const idx1 = p1v + t2 + sv;
            const idx2 = p2v + t1 + sv;
            const idx3 = p2v + t2 + sv;

            var l0: [vec_len]f32 = undefined;
            var l1: [vec_len]f32 = undefined;
            var l2: [vec_len]f32 = undefined;
            var l3: [vec_len]f32 = undefined;
            inline for (0..vec_len) |k| {
                l0[k] = @floatFromInt(lut[idx0[k]]);
                l1[k] = @floatFromInt(lut[idx1[k]]);
                l2[k] = @floatFromInt(lut[idx2[k]]);
                l3[k] = @floatFromInt(lut[idx3[k]]);
            }

            const xav: F32V = xa_arr[x..][0..vec_len].*;
            const one_m_xav = onev - xav;
            const lv0: F32V = l0;
            const lv1: F32V = l1;
            const lv2: F32V = l2;
            const lv3: F32V = l3;
            const res = (lv0 * one_m_xav + lv1 * xav) * one_m_yav + (lv2 * one_m_xav + lv3 * xav) * yav;
            const out: @Vector(vec_len, T) = @intFromFloat(@trunc(res + halfv));
            dst_row[x..][0..vec_len].* = out;
        }

        while (x < width) : (x += 1) {
            const src_val = src_row[x];
            const lut0: f32 = @floatFromInt(lut[lut_p1h + tx1h[x] + src_val]);
            const lut1: f32 = @floatFromInt(lut[lut_p1h + tx2h[x] + src_val]);
            const lut2: f32 = @floatFromInt(lut[lut_p2h + tx1h[x] + src_val]);
            const lut3: f32 = @floatFromInt(lut[lut_p2h + tx2h[x] + src_val]);
            const xa = xa_arr[x];
            const res: f32 = (lut0 * (1 - xa) + lut1 * xa) * (1 - ya) + (lut2 * (1 - xa) + lut3 * xa) * ya;

            dst_row[x] = @trunc(res + 0.5);
        }
    }
}
