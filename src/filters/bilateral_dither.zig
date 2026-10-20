const std = @import("std");
const subspl = @import("bilateral_dither_subspl.zig");
const simd = @import("simd.zig");
const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(f32) orelse 4;

/// reflect-with-edge-duplication border mirror (matches the SSE BilatData cache)
inline fn mirror(i: i32, n: i32) i32 {
    var v = i;
    while (v < 0 or v >= n) {
        if (v < 0) v = -1 - v;
        if (v >= n) v = 2 * n - 1 - v;
    }
    return v;
}

/// store value -> f32 cache: widen integers, copy floats.
inline fn toCache(comptime T: type, v: T) f32 {
    return if (T == f32) v else @floatFromInt(v);
}

/// Build one mirror-padded cache row. Interior columns [rh, rh+w) map to
/// mx = cx - rh (mirror is the identity there), so they convert/copy directly
/// and vectorized; the horizontal border cells then copy already-converted f32
/// from the interior of this same row — bit-identical to reconverting, since
/// int->f32 widening is exact and f32 copies preserve bits.
inline fn buildCacheRow(
    comptime T: type,
    srow: []const T,
    crow: []f32,
    w: usize,
    rh: usize,
    w_i: i32,
    rh_i: i32,
) void {
    if (T == f32) {
        @memcpy(crow[rh..][0..w], srow[0..w]);
    } else {
        var x: usize = 0;
        while (x + vec_len <= w) : (x += vec_len) {
            const iv: @Vector(vec_len, T) = srow[x..][0..vec_len].*;
            const fv: @Vector(vec_len, f32) = @floatFromInt(iv);
            crow[rh + x ..][0..vec_len].* = fv;
        }
        while (x < w) : (x += 1) crow[rh + x] = toCache(T, srow[x]);
    }
    var cx: usize = 0;
    while (cx < rh) : (cx += 1) {
        const mx: usize = @intCast(mirror(@as(i32, @intCast(cx)) - rh_i, w_i));
        crow[cx] = crow[rh + mx];
    }
    cx = rh + w;
    while (cx < crow.len) : (cx += 1) {
        const mx: usize = @intCast(mirror(@as(i32, @intCast(cx)) - rh_i, w_i));
        crow[cx] = crow[rh + mx];
    }
}

/// G 4-px cache groups at offsets o[0..G] joined into one 4*G-lane vector
/// (group gi -> lanes 4*gi .. 4*gi+3).
inline fn groups(comptime G: usize, buf: []const f32, o: [G]usize) @Vector(4 * G, f32) {
    if (G == 1) return buf[o[0]..][0..4].*;
    const half = G / 2;
    return std.simd.join(groups(half, buf, o[0..half].*), groups(half, buf, o[half..].*));
}

/// f32 accumulator vector -> `T` output vector: integers round-to-nearest and
/// clamp to [0, peak]; float passes through unchanged.
inline fn fromAccum(
    comptime T: type,
    comptime N: usize,
    p: @Vector(N, f32),
    peak_v: @Vector(N, f32),
    zero_v: @Vector(N, f32),
) @Vector(N, T) {
    return if (T == f32) p else @intFromFloat(@round(@max(@min(p, peak_v), zero_v)));
}

/// Row stride of the mirror-padded cache. The dense path needs `vec_len`
/// right-padding for its wide loads. The sub-sampled path instead must match
/// the C reference stride (w + 2*rh, no pad): its point lists can hold
/// out-of-window offsets for non-square geometry (a quirk of the original's
/// transposed VNC scan) that resolve via linear addressing into neighbouring
/// rows, so the stride has to be identical or those reads diverge.
pub fn cacheStride(width: u32, rh: u32, subspl_active: bool) usize {
    return @as(usize, width) + 2 * @as(usize, rh) + (if (subspl_active) @as(usize, 0) else vec_len);
}

/// Point lists -> linear cache offsets (y * cstride + x). Depends only on the
/// plane geometry, so Create resolves it once per plane: the tap loop then
/// does one load + add per list instead of loading the (x, y) pair,
/// sign-extending both and an imul per tap. Same address arithmetic, exact
/// in isize.
pub fn resolveOffsets(pls: []const subspl.Coord, width: u32, rh: u32, out: []isize) void {
    const cstride_i: isize = @intCast(cacheStride(width, rh, true));
    for (out, pls) |*o, pt| o.* = @as(isize, pt.y) * cstride_i + @as(isize, pt.x);
}

pub fn processPlane(
    comptime T: type,
    src: []const T,
    ref: ?[]const T,
    dst: []T,
    width: u32,
    height: u32,
    stride: u32,
    rh: u32,
    rv: u32,
    m: f32,
    wmax: f32,
    sum_w_min: f32,
    peak: f32, // ignored for T == f32
    offsets: ?[]const isize, // sub-sampled point lists, from resolveOffsets
    k: usize,
) void {
    const f32v = @Vector(vec_len, f32);
    const w_i: i32 = @intCast(width);
    const h_i: i32 = @intCast(height);
    const rh_i: i32 = @intCast(rh);
    const rv_i: i32 = @intCast(rv);

    // See cacheStride; extra tail rows keep the sub-sampled path's
    // out-of-window reads in-buffer.
    const subspl_active = offsets != null;
    const cstride = cacheStride(width, rh, subspl_active);
    const cheight: usize = @as(usize, height) + 2 * @as(usize, rv);
    const slack: usize = if (subspl_active) (2 * @as(usize, rh) + 2) * cstride + vec_len else 0;
    const cells = cstride * cheight + slack;

    // `src_cache` holds the values being averaged; `ref_cache` drives the value
    // weighting. With no ref clip the two alias one buffer (classic bilateral).
    const src_cache = allocator.alloc(f32, cells) catch {
        var yy: usize = 0; // graceful passthrough on OOM
        while (yy < height) : (yy += 1) @memcpy(dst[yy * stride ..][0..width], src[yy * stride ..][0..width]);
        return;
    };
    defer allocator.free(src_cache);
    if (subspl_active) @memset(src_cache[cstride * cheight ..], 0);

    const ref_cache = if (ref != null)
        (allocator.alloc(f32, cells) catch {
            var yy: usize = 0; // graceful passthrough on OOM (src_cache freed by defer)
            while (yy < height) : (yy += 1) @memcpy(dst[yy * stride ..][0..width], src[yy * stride ..][0..width]);
            return;
        })
    else
        src_cache;
    defer if (ref != null) allocator.free(ref_cache);
    if (subspl_active and ref != null) @memset(ref_cache[cstride * cheight ..], 0);

    // Build the mirror-padded float cache(s): interior rows convert directly
    // (vectorized in buildCacheRow), then each of the 2*rv vertical pad rows
    // duplicates interior cache row rv + mirror(cy - rv, h) — a whole-row copy
    // instead of re-mirroring/reconverting every cell.
    var cy: usize = 0;
    while (cy < height) : (cy += 1) {
        const crow_off = (cy + rv) * cstride;
        buildCacheRow(T, src[cy * stride ..], src_cache[crow_off..][0..cstride], width, rh, w_i, rh_i);
        if (ref) |refp| buildCacheRow(T, refp[cy * stride ..], ref_cache[crow_off..][0..cstride], width, rh, w_i, rh_i);
    }
    cy = 0;
    while (cy < cheight) : (cy += 1) {
        if (cy == rv) { // interior rows already built above
            cy += height - 1;
            continue;
        }
        const my: usize = @intCast(mirror(@as(i32, @intCast(cy)) - rv_i, h_i));
        @memcpy(src_cache[cy * cstride ..][0..cstride], src_cache[(rv + my) * cstride ..][0..cstride]);
        if (ref != null) @memcpy(ref_cache[cy * cstride ..][0..cstride], ref_cache[(rv + my) * cstride ..][0..cstride]);
    }

    const cstride_i: isize = @intCast(cstride);

    if (offsets) |offs| {
        // ---- sub-sampled "speed hack" (matching the SSE subspl path) ----
        const v4 = @Vector(4, f32);
        const v8 = @Vector(8, f32);
        const m8: v8 = @splat(m);
        const wmax8: v8 = @splat(wmax);
        const swmin8: v8 = @splat(sum_w_min);
        const zero8: v8 = @splat(0.0);
        const peak8: v8 = @splat(peak);
        const m4: v4 = @splat(m);
        const wmax4: v4 = @splat(wmax);
        const swmin4: v4 = @splat(sum_w_min);
        const zero4: v4 = @splat(0.0);
        const peak4: v4 = @splat(peak);
        // wide body: 4-px groups per vector, 16 lanes on AVX-512 else 8
        const BW = if (simd.use_avx512f) 16 else 8;
        const G = BW / 4;
        const VB = @Vector(BW, f32);
        const mb: VB = @splat(m);
        const wmaxb: VB = @splat(wmax);
        const swminb: VB = @splat(sum_w_min);
        const zerob: VB = @splat(0.0);
        const peakb: VB = @splat(peak);
        const NBR = subspl.NBR_POINT_LISTS;
        var y: usize = 0;
        while (y < height) : (y += 1) {
            // per-row list pick = RndGen value at absolute row index (single thread)
            const start: usize = (subspl.getRndAtStep(@intCast(y)) >> 8) % NBR;
            const center_base: usize = (y + rv) * cstride + rh;
            const drow: usize = y * stride;
            var x: usize = 0;
            // G adjacent 4-px groups per vector (G = 2 at 8 lanes, 4 at 16 on
            // AVX-512): the tap loop is bound by its 2-deep dependent-FMA
            // chain (~8-9 cy per 2 taps), so a wide body retires G groups per
            // chain step instead of one at the same latency. Lane block gi =
            // group x + 4*gi with its own list -- exactly the loads/taps the
            // 4-wide iterations would do, in the same per-lane order, so the
            // accumulation is bit-exact per pixel. Two such bodies per
            // iteration (2*G groups): two independent sum/sum_w chains overlap
            // the dependent adds of one with the other's loads/ALU work.
            while (x + 2 * BW <= width) : (x += 2 * BW) {
                var lists: [2 * G][]const isize = undefined;
                inline for (0..2 * G) |gi| {
                    lists[gi] = offs[((start + (x >> 2) + gi) % NBR) * k ..][0..k];
                }
                const base0: usize = center_base + x;
                const base1: usize = base0 + BW;
                const cen0: VB = src_cache[base0..][0..BW].*;
                const cen1: VB = src_cache[base1..][0..BW].*;
                const cr0: VB = ref_cache[base0..][0..BW].*;
                const cr1: VB = ref_cache[base1..][0..BW].*;
                var sum0: VB = zerob;
                var sw0: VB = zerob;
                var sum1: VB = zerob;
                var sw1: VB = zerob;
                const b0: isize = @intCast(base0);
                for (0..k) |t| {
                    var o: [2 * G]usize = undefined;
                    inline for (0..2 * G) |gi| o[gi] = @intCast(b0 + 4 * gi + lists[gi][t]);
                    const v0 = groups(G, src_cache, o[0..G].*);
                    const r0 = groups(G, ref_cache, o[0..G].*);
                    const v1 = groups(G, src_cache, o[G..].*);
                    const r1 = groups(G, ref_cache, o[G..].*);
                    const w0 = @max(simd.minPs(BW, mb - @abs(r0 - cr0), wmaxb), zerob);
                    const w1 = @max(simd.minPs(BW, mb - @abs(r1 - cr1), wmaxb), zerob);
                    sw0 += w0;
                    sw1 += w1;
                    sum0 = @mulAdd(VB, v0 - cen0, w0, sum0);
                    sum1 = @mulAdd(VB, v1 - cen1, w1, sum1);
                }
                dst[drow + x ..][0..BW].* = fromAccum(T, BW, cen0 + sum0 / @max(sw0, swminb), peakb, zerob);
                dst[drow + x + BW ..][0..BW].* = fromAccum(T, BW, cen1 + sum1 / @max(sw1, swminb), peakb, zerob);
            }
            while (x + 8 <= width) : (x += 8) {
                const la = (start + (x >> 2)) % NBR;
                const lb = if (la + 1 == NBR) 0 else la + 1;
                const cla = offs[la * k ..][0..k];
                const clb = offs[lb * k ..][0..k];
                const base: usize = center_base + x;
                const cen: v8 = src_cache[base..][0..8].*;
                const cen_ref: v8 = ref_cache[base..][0..8].*;
                var sum: v8 = zero8;
                var sum_w: v8 = zero8;
                for (cla, clb) |pa, pb| {
                    const offa: usize = @intCast(@as(isize, @intCast(base)) + pa);
                    const offb: usize = @intCast(@as(isize, @intCast(base + 4)) + pb);
                    const v = std.simd.join(@as(v4, src_cache[offa..][0..4].*), @as(v4, src_cache[offb..][0..4].*));
                    const vr = std.simd.join(@as(v4, ref_cache[offa..][0..4].*), @as(v4, ref_cache[offb..][0..4].*));
                    const diff = v - cen;
                    const dist = @abs(vr - cen_ref);
                    // minPs: bare vminps; wmax is never NaN (Create clamps it
                    // with @max(.., unit)), so NaN lanes resolve to wmax exactly
                    // like @min — drops the vcmpunordps + vblendvps fixup.
                    const wgt = @max(simd.minPs(8, m8 - dist, wmax8), zero8);
                    sum_w += wgt;
                    sum = @mulAdd(v8, diff, wgt, sum); // FMA
                }
                const denom = @max(sum_w, swmin8);
                const p = cen + sum / denom;
                dst[drow + x ..][0..8].* = fromAccum(T, 8, p, peak8, zero8);
            }
            // trailing <8 px: the original 4-wide groups with per-pixel tail
            while (x < width) : (x += 4) {
                const take = @min(@as(usize, 4), width - x);
                const list_idx = (start + (x >> 2)) % NBR;
                const cl = offs[list_idx * k ..][0..k];
                const base: usize = center_base + x;
                const cen: v4 = src_cache[base..][0..4].*;
                const cen_ref: v4 = ref_cache[base..][0..4].*;
                var sum: v4 = zero4;
                var sum_w: v4 = zero4;
                for (cl) |po| {
                    const off: usize = @intCast(@as(isize, @intCast(base)) + po);
                    const v: v4 = src_cache[off..][0..4].*;
                    const vr: v4 = ref_cache[off..][0..4].*;
                    const diff = v - cen;
                    const dist = @abs(vr - cen_ref);
                    const wgt = @max(simd.minPs(4, m4 - dist, wmax4), zero4);
                    sum_w += wgt;
                    sum = @mulAdd(v4, diff, wgt, sum); // FMA
                }
                const denom = @max(sum_w, swmin4);
                const p = cen + sum / denom;
                const out = fromAccum(T, 4, p, peak4, zero4);
                if (take == 4) {
                    dst[drow + x ..][0..4].* = out;
                } else {
                    const arr: [4]T = out;
                    var i: usize = 0;
                    while (i < take) : (i += 1) dst[drow + x + i] = arr[i];
                }
            }
        }
        return;
    }

    // ---- dense full window ----
    const m_v: f32v = @splat(m);
    const wmax_v: f32v = @splat(wmax);
    const swmin_v: f32v = @splat(sum_w_min);
    const zero_v: f32v = @splat(0.0);
    const peak_v: f32v = @splat(peak);

    var y: usize = 0;
    while (y < height) : (y += 1) {
        const center_base: usize = (y + rv) * cstride + rh;
        const drow: usize = y * stride;
        var x: usize = 0;
        while (x < width) : (x += vec_len) {
            const take = @min(vec_len, width - x);
            const base_i: isize = @intCast(center_base + x);
            const cen: f32v = src_cache[@intCast(base_i)..][0..vec_len].*;
            const cen_ref: f32v = ref_cache[@intCast(base_i)..][0..vec_len].*;

            var sum: f32v = zero_v;
            var sum_w: f32v = zero_v;
            var dy: i32 = 1 - rv_i;
            while (dy < rv_i) : (dy += 1) {
                const row_i: isize = base_i + @as(isize, dy) * cstride_i;
                var dx: i32 = 1 - rh_i;
                while (dx < rh_i) : (dx += 1) {
                    const off: usize = @intCast(row_i + @as(isize, dx));
                    const v: f32v = src_cache[off..][0..vec_len].*;
                    const vr: f32v = ref_cache[off..][0..vec_len].*;
                    const diff = v - cen;
                    const dist = @abs(vr - cen_ref);
                    const wgt = @max(simd.minPs(vec_len, m_v - dist, wmax_v), zero_v);
                    sum_w += wgt;
                    sum = @mulAdd(f32v, diff, wgt, sum); // FMA
                }
            }

            const denom = @max(sum_w, swmin_v);
            const p = cen + sum / denom;
            const out = fromAccum(T, vec_len, p, peak_v, zero_v);

            if (take == vec_len) {
                dst[drow + x ..][0..vec_len].* = out;
            } else {
                const arr: [vec_len]T = out;
                var i: usize = 0;
                while (i < take) : (i += 1) dst[drow + x + i] = arr[i];
            }
        }
    }
}

// --------------------------------------------------------------------------
// Configuration (shared by the VapourSynth and AviSynth adapters)
// --------------------------------------------------------------------------

/// Per-plane kernel parameters; `offsets` are owned (free with `deinit`).
pub const Config = struct {
    rh: [3]u32 = @splat(0),
    rv: [3]u32 = @splat(0),
    m: [3]f32 = @splat(0),
    wmax: [3]f32 = @splat(0),
    sum_w_min: [3]f32 = @splat(0),
    // sub-sampled point lists resolved to linear cache offsets per plane
    offsets: [3]?[]isize = .{ null, null, null },
    k: [3]usize = @splat(0),
    peak: f32 = 0,

    pub const Args = struct {
        is_int: bool,
        bits: u32,
        planes: [3]bool,
        num_planes: u32,
        plane_w: [3]i32,
        plane_h: [3]i32,
        radius: [3]i32,
        thr: [3]f32,
        flat: [3]f32,
        wmin: [3]f32,
        subspl: [3]f32,
    };

    pub const Error = error{ TooSmall, Radius, OutOfMemory };

    /// Message for an `init` error (callers add their filter-name prefix).
    pub fn message(e: Error) [:0]const u8 {
        return switch (e) {
            error.TooSmall => "input must be 16x16 min",
            error.Radius => "picture size must be greater than \"radius\"",
            error.OutOfMemory => "out of memory",
        };
    }

    /// The frame-size check, which vszip runs before the planes argument.
    pub fn checkSize(width: i32, height: i32) Error!void {
        if (width < 16 or height < 16) return error.TooSmall;
    }

    pub fn init(a: Args) Error!Config {
        const scale: f32 = if (a.is_int)
            @floatFromInt(@as(u32, 1) << @as(u5, @intCast(a.bits - 8)))
        else
            1.0 / 256.0;
        const unit: f32 = if (a.is_int) 1.0 else 1.0 / 65535.0;

        var d: Config = .{};
        d.peak = if (a.is_int) @floatFromInt((@as(u32, 1) << @as(u5, @intCast(a.bits))) - 1) else 0;
        errdefer d.deinit();

        for (0..a.num_planes) |p| {
            if (!a.planes[p]) continue;
            const pw = a.plane_w[p];
            const ph = a.plane_h[p];
            if (pw < a.radius[p] or ph < a.radius[p]) return error.Radius;
            const rp: u32 = @intCast(a.radius[p]);
            d.rh[p] = rp;
            d.rv[p] = rp;
            d.m[p] = @max(a.thr[p] * scale, unit);
            d.wmax[p] = @max(a.thr[p] * (1.0 - a.flat[p]) * scale, unit);
            const slot = buildSlot(rp, rp, a.subspl[p], a.wmin[p], d.wmax[p], unit) catch return error.OutOfMemory;
            d.sum_w_min[p] = slot.sum_w_min;
            d.k[p] = slot.k;
            if (slot.pl) |pts| {
                defer allocator.free(pts);
                const offs = try allocator.alloc(isize, pts.len);
                resolveOffsets(pts, @intCast(pw), rp, offs);
                d.offsets[p] = offs;
            }
        }
        return d;
    }

    pub fn deinit(d: *Config) void {
        for (&d.offsets) |*o| {
            if (o.*) |p| allocator.free(p);
            o.* = null;
        }
    }
};

const SlotCfg = struct { sum_w_min: f32, pl: ?[]subspl.Coord, k: usize };

fn buildSlot(rh: u32, rv: u32, subspl_arg: f32, wmin: f32, wmax: f32, unit: f32) !SlotCfg {
    const active = (subspl_arg >= 4.0) or (subspl_arg < 1e-3);
    if (active) {
        const lists = try subspl.generate(@intCast(rh), @intCast(rv), @floatCast(subspl_arg));
        const kf: f32 = @floatFromInt(lists.k);
        return .{ .sum_w_min = @max(wmin * wmax * kf, unit), .pl = lists.pts, .k = lists.k };
    } else {
        const area: f32 = @floatFromInt((2 * rh - 1) * (2 * rv - 1));
        return .{ .sum_w_min = @max(wmin * wmax * area, unit), .pl = null, .k = 0 };
    }
}
