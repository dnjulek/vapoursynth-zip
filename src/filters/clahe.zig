const std = @import("std");

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(i32) orelse 8;
const I32V = @Vector(vec_len, i32);
const U32V = @Vector(vec_len, u32);
const F32V = @Vector(vec_len, f32);

// sub-histograms per tile: pixels scatter round-robin so runs of equal values
// don't serialize on a single bin's load-add-store chain
const n_sub = 4;

/// Sample -> histogram bin mapping. Integer clips bin by value (`bits` of
/// the format, so 10-bit gets 1024 bins and a 10-bit peak, not the 16-bit
/// container's). Float clips are quantized to 16-bit bins on the fly in both
/// passes (no u16 plane is stored): bin = round((v + off) * 65535) clamped to
/// [0, 65535], off = 0.5 on YUV chroma ([-0.5, 0.5]) and 0 otherwise; NaN
/// lands in bin 0. The equalization then runs exactly as on a 16-bit clip and
/// the interpolated LUT value maps back as res / 65535 - off, unrounded.
pub fn Bins(comptime T: type, comptime clamp: bool) type {
    return struct {
        const Self = @This();
        pub const is_float = @typeInfo(T) == .float;
        /// LUT / bin storage type
        pub const L = if (T == u8) u8 else u16;
        bits: u5,
        off: f32 = 0, // float: chroma offset
        peak_f: f32,
        inv_peak: f32,

        pub fn init(bits: u5, chroma: bool) Self {
            const peak: u32 = (@as(u32, 1) << bits) - 1;
            const pf: f32 = @floatFromInt(peak);
            return .{ .bits = bits, .off = if (is_float and chroma) 0.5 else 0, .peak_f = pf, .inv_peak = 1.0 / pf };
        }

        inline fn histSize(self: Self) u32 {
            return @as(u32, 1) << self.bits;
        }

        pub inline fn bin(self: Self, v: T) u32 {
            if (comptime is_float) {
                const x = (@as(f32, @floatCast(v)) + self.off) * self.peak_f + 0.5;
                return @intFromFloat(@min(@max(x, 0.0), self.peak_f));
            }
            if (comptime clamp) return @min(@as(u32, v), (@as(u32, 1) << self.bits) - 1);
            return v;
        }

        pub inline fn bins(self: Self, v: @Vector(vec_len, T)) U32V {
            if (comptime is_float) {
                const f: F32V = if (T == f32) v else @floatCast(v);
                const x = (f + @as(F32V, @splat(self.off))) * @as(F32V, @splat(self.peak_f)) + @as(F32V, @splat(0.5));
                return @intFromFloat(@min(@max(x, @as(F32V, @splat(0.0))), @as(F32V, @splat(self.peak_f))));
            }
            const u: U32V = @intCast(v);
            if (comptime clamp) return @min(u, @as(U32V, @splat((@as(u32, 1) << self.bits) - 1)));
            return u;
        }

        /// interpolated LUT value -> output sample
        pub inline fn out(self: Self, res: F32V) @Vector(vec_len, T) {
            if (comptime is_float) {
                const r = res * @as(F32V, @splat(self.inv_peak)) - @as(F32V, @splat(self.off));
                return if (T == f32) r else @floatCast(r);
            }
            return @intFromFloat(@trunc(res + @as(F32V, @splat(0.5))));
        }

        pub inline fn out1(self: Self, res: f32) T {
            if (comptime is_float) return @floatCast(res * self.inv_peak - self.off);
            return @intFromFloat(@trunc(res + 0.5));
        }
    };
}

pub fn applyCLAHE(
    comptime T: type,
    comptime clamp: bool,
    srcp: []const T,
    dstp: []T,
    stride: u32,
    width: u32,
    height: u32,
    limit: u32,
    tiles: []u32,
    bits: u5,
    chroma: bool,
) error{OutOfMemory}!void {
    const B = Bins(T, clamp);
    const bn = B.init(bits, chroma);
    const hist_size: u32 = bn.histSize();
    const peak: f32 = bn.peak_f;
    const tile_width: u32 = width / tiles[0];
    const tile_height: u32 = height / tiles[1];
    const tile_size_total: u64 = @as(u64, tile_width) * tile_height;
    const lut_scale: f32 = peak / @as(f32, @floatFromInt(tile_size_total));
    var clip_limit: i32 = @intCast(@as(u64, limit) * tile_size_total / hist_size);
    clip_limit = @max(clip_limit, 1);

    const lut = try allocator.alloc(B.L, tiles[0] * tiles[1] * hist_size);
    defer allocator.free(lut);

    try calcLut(T, B, bn, srcp, stride, width, height, lut, tile_width, tile_height, tiles, clip_limit, lut_scale);
    if (T == u8) {
        try interpolateU8(srcp, dstp, stride, width, height, lut, tile_width, tile_height, tiles);
    } else {
        try interpolate(T, B, bn, srcp, dstp, stride, width, height, lut, tile_width, tile_height, tiles);
    }
}

fn calcLut(
    comptime T: type,
    comptime B: type,
    bn: B,
    srcp: []const T,
    stride: u32,
    width: u32,
    height: u32,
    lut: []B.L,
    tile_width: u32,
    tile_height: u32,
    tiles: []u32,
    clip_limit: i32,
    lut_scale: f32,
) error{OutOfMemory}!void {
    const hist_size: u32 = bn.histSize();
    const hist_sizei: i32 = @intCast(hist_size);
    const tiles_x = tiles[0];
    const tiles_y = tiles[1];

    // Sub-histograms only pay off when a tile has many pixels per bin (runs
    // of equal values); otherwise merging them is several times the pixel
    // work (tiles=16 on 16-bit: ~8000 px per tile against 4 x 256 KiB).
    const tile_px = @as(u64, tile_width) * tile_height;
    const subs_n: u32 = if (tile_px >= @as(u64, n_sub) * hist_size) n_sub else 1;
    const subs = try allocator.alloc(u32, n_sub * hist_size);
    defer allocator.free(subs);

    // zeroed once; each tile's merge loop re-zeroes the bins it drains, so no
    // per-tile full-buffer memset sweep (1 MiB/tile for u16) is needed
    @memset(subs[0 .. subs_n * hist_size], 0);

    // Bin range [lo, hi] of each tile's pixels; the last tile column/row also
    // covers the leftover pixels past tiles*tile_size (they interpolate from
    // the edge tiles but feed no histogram). Interpolation reads tile T's LUT
    // only at bins of pixels inside T's 3x3 tile neighborhood, so each LUT row
    // is built only over the union of those ranges.
    const ranges = try allocator.alloc([2]u32, tiles_x * tiles_y);
    defer allocator.free(ranges);
    for (0..tiles_y) |tyi| {
        const ty: u32 = @intCast(tyi);
        const y_end = if (ty == tiles_y - 1) height else (ty + 1) * tile_height;
        for (0..tiles_x) |txi| {
            const tx: u32 = @intCast(txi);
            const x_end = if (tx == tiles_x - 1) width else (tx + 1) * tile_width;
            ranges[ty * tiles_x + tx] = binRange(T, B, bn, srcp, stride, tx * tile_width, x_end, ty * tile_height, y_end);
        }
    }

    var ty: u32 = 0;
    while (ty < tiles_y) : (ty += 1) {
        var tx: u32 = 0;
        while (tx < tiles_x) : (tx += 1) {
            // own range (vector-aligned) and the neighborhood's
            const own = ranges[ty * tiles_x + tx];
            var nb = own;
            // interpolateU8 packs whole LUT rows into its quad tables
            if (comptime T == u8) nb = .{ 0, hist_size - 1 };
            for (ty -| 1..@min(ty + 2, tiles_y)) |ny| {
                for (tx -| 1..@min(tx + 2, tiles_x)) |nx| {
                    const r = ranges[ny * tiles_x + nx];
                    nb[0] = @min(nb[0], r[0]);
                    nb[1] = @max(nb[1], r[1]);
                }
            }
            const own_lo = own[0] - own[0] % vec_len;
            const own_hi = own[1] - own[1] % vec_len + vec_len;
            const nb_lo = nb[0] - nb[0] % vec_len;
            const nb_hi = nb[1] - nb[1] % vec_len + vec_len;
            std.debug.assert(nb_lo <= own_lo and own_hi <= nb_hi and nb_hi <= hist_size);

            const h0 = subs[0 * hist_size ..][0..hist_size];
            // with one sub-histogram every scatter goes to h0
            const h1 = subs[@as(u32, if (subs_n > 1) 1 else 0) * hist_size ..][0..hist_size];
            const h2 = subs[@as(u32, if (subs_n > 1) 2 else 0) * hist_size ..][0..hist_size];
            const h3 = subs[@as(u32, if (subs_n > 1) 3 else 0) * hist_size ..][0..hist_size];

            var y: u32 = ty * tile_height;
            while (y < @min((ty + 1) * tile_height, height)) : (y += 1) {
                const row = srcp[y * stride ..];
                var x: u32 = tx * tile_width;
                const xe = @min((tx + 1) * tile_width, width);
                if (comptime B.is_float) {
                    // quantize vec_len samples at once, then scatter
                    while (x + vec_len <= xe) : (x += vec_len) {
                        const b = bn.bins(row[x..][0..vec_len].*);
                        inline for (0..vec_len / 4) |k| {
                            h0[b[4 * k]] += 1;
                            h1[b[4 * k + 1]] += 1;
                            h2[b[4 * k + 2]] += 1;
                            h3[b[4 * k + 3]] += 1;
                        }
                    }
                }
                while (x + 4 <= xe) : (x += 4) {
                    h0[bn.bin(row[x])] += 1;
                    h1[bn.bin(row[x + 1])] += 1;
                    h2[bn.bin(row[x + 2])] += 1;
                    h3[bn.bin(row[x + 3])] += 1;
                }
                while (x < xe) : (x += 1) {
                    h0[bn.bin(row[x])] += 1;
                }
            }

            // Merge the sub-histograms into h0 (clearing the others for the
            // next tile) and total the excess over the clip limit; h0 keeps
            // the unclipped counts, clipped in the LUT pass below. All bins
            // outside [own_lo, own_hi) are zero. clip_limit >= 1 always
            // (applyCLAHE clamps it).
            std.debug.assert(clip_limit > 0);
            const limit_u: u32 = @intCast(clip_limit);
            const limitv: U32V = @splat(limit_u);
            var clipped_acc: U32V = @splat(0);
            {
                const zerov: U32V = @splat(0);
                var i: u32 = own_lo;
                while (i < own_hi) : (i += vec_len) {
                    var sum: U32V = h0[i..][0..vec_len].*;
                    if (subs_n > 1) {
                        inline for (.{ h1, h2, h3 }) |hh| {
                            sum += hh[i..][0..vec_len].*;
                            hh[i..][0..vec_len].* = zerov;
                        }
                        h0[i..][0..vec_len].* = sum;
                    }
                    clipped_acc += sum - @min(sum, limitv);
                }
            }
            const clipped: i32 = @intCast(@reduce(.Add, clipped_acc));
            const redist_batch: i32 = @divTrunc(clipped, hist_sizei);
            const residual: i32 = clipped - redist_batch * hist_sizei;
            // The residual +1s go to bins k*residual_step, k < residual, of
            // the clipped histogram (k*step < hist_size always: step =
            // hist_size/residual). Flag those inside the LUT range in bit 31
            // (counts stay < 2^31), added after clipping.
            const flag: u32 = 1 << 31;
            const n_res: u32 = @intCast(residual);
            const residual_step: u32 = if (residual != 0) @intCast(@max(@divTrunc(hist_sizei, residual), 1)) else 1;
            {
                var k: u32 = (nb_lo + residual_step - 1) / residual_step;
                while (k < n_res and k * residual_step < nb_hi) : (k += 1) h0[k * residual_step] |= flag;
            }

            // clip + residual + uniform redistribution, cumulative sum ->
            // scaled LUT row, re-zeroing h0. Only the scalar `run += last` is
            // loop-carried; the in-register prefix and the lane extract hang
            // off each block's own input (carrying `run` through broadcast +
            // add + extract made this loop latency-bound).
            {
                const lut_row = lut[(ty * tiles_x + tx) * hist_size ..][0..hist_size];
                const scalev: F32V = @splat(lut_scale);
                const halfv: F32V = @splat(0.5);
                const batchv: I32V = @splat(redist_batch);
                const zerov: U32V = @splat(0);
                const countv: U32V = @splat(flag - 1);
                // cumulative sum below nb_lo, where the histogram is zero:
                // nb_lo * batch plus the residual bins below nb_lo
                const res_below: u32 = @min(n_res, (nb_lo + residual_step - 1) / residual_step);
                var run: i32 = @as(i32, @intCast(nb_lo)) * redist_batch + @as(i32, @intCast(res_below));
                var i: u32 = nb_lo;
                while (i < nb_hi) : (i += vec_len) {
                    const h: U32V = h0[i..][0..vec_len].*;
                    h0[i..][0..vec_len].* = zerov;
                    const c = @min(h & countv, limitv) + (h >> @splat(31));
                    const p = prefixSum(@as(I32V, @intCast(c)) + batchv);
                    const last = p[vec_len - 1];
                    const f: F32V = @floatFromInt(p + @as(I32V, @splat(run)));
                    run += last;
                    const out: @Vector(vec_len, B.L) = @intFromFloat(@trunc(f * scalev + halfv));
                    lut_row[i..][0..vec_len].* = out;
                }
            }
        }
    }
}

/// Min and max histogram bin over a rectangle of the plane.
fn binRange(comptime T: type, comptime B: type, bn: B, srcp: []const T, stride: u32, x0: u32, x1: u32, y0: u32, y1: u32) [2]u32 {
    var lo_v: U32V = @splat(std.math.maxInt(u32));
    var hi_v: U32V = @splat(0);
    var lo: u32 = std.math.maxInt(u32);
    var hi: u32 = 0;
    var y = y0;
    while (y < y1) : (y += 1) {
        const row = srcp[y * stride ..];
        var x = x0;
        while (x + vec_len <= x1) : (x += vec_len) {
            const b = bn.bins(row[x..][0..vec_len].*);
            lo_v = @min(lo_v, b);
            hi_v = @max(hi_v, b);
        }
        while (x < x1) : (x += 1) {
            const b = bn.bin(row[x]);
            lo = @min(lo, b);
            hi = @max(hi, b);
        }
    }
    lo = @min(lo, @reduce(.Min, lo_v));
    hi = @max(hi, @reduce(.Max, hi_v));
    if (lo > hi) return .{ 0, 0 }; // empty rectangle
    return .{ lo, hi };
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
    comptime B: type,
    bn: B,
    srcp: []const T,
    dstp: []T,
    stride: u32,
    width: u32,
    height: u32,
    lut: []const B.L,
    tile_width: u32,
    tile_height: u32,
    tiles: []u32,
) error{OutOfMemory}!void {
    const hist_size: u32 = bn.histSize();
    const tiles_x: i32 = @intCast(tiles[0]);
    const tiles_y: i32 = @intCast(tiles[1]);

    const inv_tw: f32 = 1.0 / @as(f32, @floatFromInt(tile_width));
    const inv_th: f32 = 1.0 / @as(f32, @floatFromInt(tile_height));

    // x-dependent terms are the same on every row; precompute them once
    const tx1h = try allocator.alloc(u32, width);
    defer allocator.free(tx1h);
    const tx2h = try allocator.alloc(u32, width);
    defer allocator.free(tx2h);
    const xa_arr = try allocator.alloc(f32, width);
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
        const p1v: U32V = @splat(lut_p1h);
        const p2v: U32V = @splat(lut_p2h);

        var x: u32 = 0;
        while (x + vec_len <= width) : (x += vec_len) {
            const sv = bn.bins(src_row[x..][0..vec_len].*);
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
            dst_row[x..][0..vec_len].* = bn.out(res);
        }

        while (x < width) : (x += 1) {
            const src_val = bn.bin(src_row[x]);
            const lut0: f32 = @floatFromInt(lut[lut_p1h + tx1h[x] + src_val]);
            const lut1: f32 = @floatFromInt(lut[lut_p1h + tx2h[x] + src_val]);
            const lut2: f32 = @floatFromInt(lut[lut_p2h + tx1h[x] + src_val]);
            const lut3: f32 = @floatFromInt(lut[lut_p2h + tx2h[x] + src_val]);
            const xa = xa_arr[x];
            const res: f32 = (lut0 * (1 - xa) + lut1 * xa) * (1 - ya) + (lut2 * (1 - xa) + lut3 * xa) * ya;

            dst_row[x] = bn.out1(res);
        }
    }
}

/// u8 specialization of `interpolate`: the four per-pixel LUT bytes are packed
/// into one u32 gathered from a small per-band "quad" table, quartering the
/// extract/insert lookup chains that dominate the pixel loop. tx2 is a pure
/// function of tx1 (and ty2 of ty1), so a (tx1,tx2) column pair is one of only
/// tiles_x+1 slots, and the table depends on y only through the (ty1,ty2) band
/// — it is rebuilt just tiles_y+1 times per plane. The unpacked bytes feed the
/// exact FP expression of `interpolate`, so results stay bit-identical.
/// Not used for u16: the table would be tiles_x+1 slots x 65536 entries,
/// gathering from it is no cheaper than from the LUT rows themselves.
fn interpolateU8(
    srcp: []const u8,
    dstp: []u8,
    stride: u32,
    width: u32,
    height: u32,
    lut: []const u8,
    tile_width: u32,
    tile_height: u32,
    tiles: []u32,
) error{OutOfMemory}!void {
    const hist_size: u32 = 256;
    const tiles_x: i32 = @intCast(tiles[0]);
    const tiles_y: i32 = @intCast(tiles[1]);

    const inv_tw: f32 = 1.0 / @as(f32, @floatFromInt(tile_width));
    const inv_th: f32 = 1.0 / @as(f32, @floatFromInt(tile_height));

    // x-dependent terms are the same on every row; precompute them once
    const q_off = try allocator.alloc(u32, width);
    defer allocator.free(q_off);
    const xa_arr = try allocator.alloc(f32, width);
    defer allocator.free(xa_arr);
    const n_slots: u32 = tiles[0] + 1;
    const quad = try allocator.alloc(u32, n_slots * hist_size);
    defer allocator.free(quad);

    {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const txf: f32 = @as(f32, @floatFromInt(x)) * inv_tw - 0.5;
            const _tx1: i32 = @floor(txf);
            xa_arr[x] = txf - @as(f32, @floatFromInt(_tx1));
            // slot s encodes the clamped pair (tx1, tx2) = (max(s-1, 0), min(s, tiles_x-1));
            // txf >= -0.5 so _tx1 >= -1 and s is in [0, tiles_x]
            const slot: u32 = @intCast(@min(_tx1, tiles_x - 1) + 1);
            q_off[x] = slot * hist_size;
        }
    }

    var prev_p1h: u32 = std.math.maxInt(u32);
    var prev_p2h: u32 = std.math.maxInt(u32);

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

        if (lut_p1h != prev_p1h or lut_p2h != prev_p2h) {
            prev_p1h = lut_p1h;
            prev_p2h = lut_p2h;
            var slot: u32 = 0;
            while (slot < n_slots) : (slot += 1) {
                const c1: u32 = (slot -| 1) * hist_size;
                const c2: u32 = @min(slot, tiles[0] - 1) * hist_size;
                const r0 = lut[lut_p1h + c1 ..][0..hist_size];
                const r1 = lut[lut_p1h + c2 ..][0..hist_size];
                const r2 = lut[lut_p2h + c1 ..][0..hist_size];
                const r3 = lut[lut_p2h + c2 ..][0..hist_size];
                const qrow = quad[slot * hist_size ..][0..hist_size];
                for (qrow, r0, r1, r2, r3) |*q, a, b, c, d| {
                    q.* = @as(u32, a) | (@as(u32, b) << 8) | (@as(u32, c) << 16) | (@as(u32, d) << 24);
                }
            }
        }

        const src_row = srcp[y * stride ..];
        const dst_row = dstp[y * stride ..];
        const onev: F32V = @splat(1.0);
        const yav: F32V = @splat(ya);
        const one_m_yav: F32V = @splat(1 - ya);
        const halfv: F32V = @splat(0.5);
        const bytev: U32V = @splat(0xFF);

        var x: u32 = 0;
        while (x + vec_len <= width) : (x += vec_len) {
            const sv: U32V = @intCast(@as(@Vector(vec_len, u8), src_row[x..][0..vec_len].*));
            const qo: U32V = q_off[x..][0..vec_len].*;
            const idx = qo + sv;

            var g: [vec_len]u32 = undefined;
            inline for (0..vec_len) |k| {
                g[k] = quad[idx[k]];
            }
            const gv: U32V = g;
            const lv0: F32V = @floatFromInt(@as(I32V, @intCast(gv & bytev)));
            const lv1: F32V = @floatFromInt(@as(I32V, @intCast((gv >> @splat(8)) & bytev)));
            const lv2: F32V = @floatFromInt(@as(I32V, @intCast((gv >> @splat(16)) & bytev)));
            const lv3: F32V = @floatFromInt(@as(I32V, @intCast(gv >> @splat(24))));

            const xav: F32V = xa_arr[x..][0..vec_len].*;
            const one_m_xav = onev - xav;
            const res = (lv0 * one_m_xav + lv1 * xav) * one_m_yav + (lv2 * one_m_xav + lv3 * xav) * yav;
            const out: @Vector(vec_len, u8) = @intFromFloat(@trunc(res + halfv));
            dst_row[x..][0..vec_len].* = out;
        }

        while (x < width) : (x += 1) {
            const q = quad[q_off[x] + src_row[x]];
            const lut0: f32 = @floatFromInt(q & 0xFF);
            const lut1: f32 = @floatFromInt((q >> 8) & 0xFF);
            const lut2: f32 = @floatFromInt((q >> 16) & 0xFF);
            const lut3: f32 = @floatFromInt(q >> 24);
            const xa = xa_arr[x];
            const res: f32 = (lut0 * (1 - xa) + lut1 * xa) * (1 - ya) + (lut2 * (1 - xa) + lut3 * xa) * ya;

            dst_row[x] = @trunc(res + 0.5);
        }
    }
}
