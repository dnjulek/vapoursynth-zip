const std = @import("std");

const allocator = std.heap.c_allocator;

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
    const tile_size_total: u32 = tile_width * tile_height;
    const lut_scale: f32 = peak / @as(f32, @floatFromInt(tile_size_total));
    var clip_limit: i32 = @intCast(limit * tile_size_total / hist_size);
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
    width: usize,
    height: usize,
    lut: []T,
    tile_width: usize,
    tile_height: usize,
    tiles: []u32,
    clip_limit: i32,
    lut_scale: f32,
) void {
    const hist_size: u32 = @as(u32, 1) << @as(u32, @typeInfo(T).int.bits);
    const hist_sizei: i32 = @intCast(hist_size);
    var tile_hist: [hist_size]i32 = undefined;
    const tiles_x = tiles[0];
    const tiles_y = tiles[1];

    var ty: usize = 0;
    while (ty < tiles_y) : (ty += 1) {
        var tx: usize = 0;
        while (tx < tiles_x) : (tx += 1) {
            @memset(&tile_hist, 0);

            var y: usize = ty * tile_height;
            while (y < @min((ty + 1) * tile_height, height)) : (y += 1) {
                var x: usize = tx * tile_width;
                while (x < @min((tx + 1) * tile_width, width)) : (x += 1) {
                    tile_hist[srcp[y * stride + x]] += 1;
                }
            }

            if (clip_limit > 0) {
                var clipped: i32 = 0;
                for (&tile_hist) |*bin| {
                    if (bin.* > clip_limit) {
                        clipped += bin.* - clip_limit;
                        bin.* = clip_limit;
                    }
                }

                const redist_batch: i32 = @divTrunc(clipped, hist_sizei);
                var residual: i32 = clipped - redist_batch * hist_sizei;

                for (&tile_hist) |*bin| {
                    bin.* += redist_batch;
                }

                if (residual != 0) {
                    const residual_step = @max(@divTrunc(hist_sizei, residual), 1);
                    var i: usize = 0;
                    while ((i < hist_size) and (residual > 0)) : (i += residual_step) {
                        tile_hist[i] += 1;
                        residual -= 1;
                    }
                }
            }

            var sum: i32 = 0;
            var i: usize = 0;
            while (i < hist_size) : (i += 1) {
                sum += tile_hist[i];
                lut[(ty * tiles_x + tx) * hist_size + i] = @intFromFloat(@as(f32, @floatFromInt(sum)) * lut_scale + 0.5);
            }
        }
    }
}

fn interpolate(
    comptime T: type,
    srcp: []const T,
    dstp: []T,
    stride: u32,
    width: usize,
    height: usize,
    lut: []const T,
    tile_width: usize,
    tile_height: usize,
    tiles: []u32,
) void {
    const hist_size: u32 = @as(u32, 1) << @as(u32, @typeInfo(T).int.bits);
    const tiles_x: i32 = @intCast(tiles[0]);
    const tiles_y: i32 = @intCast(tiles[1]);

    const inv_tw: f32 = 1.0 / @as(f32, @floatFromInt(tile_width));
    const inv_th: f32 = 1.0 / @as(f32, @floatFromInt(tile_height));

    var y: usize = 0;
    while (y < height) : (y += 1) {
        const tyf: f32 = @as(f32, @floatFromInt(y)) * inv_th - 0.5;
        var ty1: i32 = @intFromFloat(@floor(tyf));
        var ty2: i32 = ty1 + 1;
        const ya: f32 = tyf - @as(f32, @floatFromInt(ty1));

        ty1 = @max(ty1, 0);
        ty2 = @min(ty2, tiles_y - 1);
        const lut_p1 = ty1 * tiles_x;
        const lut_p2 = ty2 * tiles_x;

        var x: usize = 0;
        while (x < width) : (x += 1) {
            const txf: f32 = @as(f32, @floatFromInt(x)) * inv_tw - 0.5;
            const _tx1: i32 = @intFromFloat(@floor(txf));
            const _tx2: i32 = _tx1 + 1;
            const xa: f32 = txf - @as(f32, @floatFromInt(_tx1));

            const tx1 = @max(_tx1, 0);
            const tx2 = @min(_tx2, tiles_x - 1);

            const src_val = srcp[y * stride + x];
            const p1_tx1: u32 = @intCast(lut_p1 + tx1);
            const p1_tx2: u32 = @intCast(lut_p1 + tx2);
            const p2_tx1: u32 = @intCast(lut_p2 + tx1);
            const p2_tx2: u32 = @intCast(lut_p2 + tx2);
            const lut0: f32 = @floatFromInt(lut[p1_tx1 * hist_size + src_val]);
            const lut1: f32 = @floatFromInt(lut[p1_tx2 * hist_size + src_val]);
            const lut2: f32 = @floatFromInt(lut[p2_tx1 * hist_size + src_val]);
            const lut3: f32 = @floatFromInt(lut[p2_tx2 * hist_size + src_val]);
            const res: f32 = (lut0 * (1 - xa) + lut1 * xa) * (1 - ya) + (lut2 * (1 - xa) + lut3 * xa) * ya;

            dstp[y * stride + x] = @intFromFloat(res + 0.5);
        }
    }
}
