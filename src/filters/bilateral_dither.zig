const std = @import("std");
const subspl = @import("bilateral_dither_subspl.zig");
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
    point_lists: ?[]const subspl.Coord,
    k: usize,
) void {
    const f32v = @Vector(vec_len, f32);
    const w_i: i32 = @intCast(width);
    const h_i: i32 = @intCast(height);
    const rh_i: i32 = @intCast(rh);
    const rv_i: i32 = @intCast(rv);

    // The dense path needs `vec_len` right-padding in the stride for the wide
    // loads. The sub-sampled path instead must match the C reference stride
    // (w + 2*rh, no pad): its point lists can hold out-of-window offsets for
    // non-square geometry (a quirk of the original's transposed VNC scan) that
    // resolve via linear addressing into neighbouring rows, so the stride has
    // to be identical or those reads diverge. Extra tail rows keep them in-buffer.
    const subspl_active = point_lists != null;
    const cstride: usize = @as(usize, width) + 2 * @as(usize, rh) + (if (subspl_active) @as(usize, 0) else @as(usize, vec_len));
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

    if (point_lists) |pls| {
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
        const NBR = subspl.NBR_POINT_LISTS;
        var y: usize = 0;
        while (y < height) : (y += 1) {
            // per-row list pick = RndGen value at absolute row index (single thread)
            const start: usize = (subspl.getRndAtStep(@intCast(y)) >> 8) % NBR;
            const center_base: usize = (y + rv) * cstride + rh;
            const drow: usize = y * stride;
            var x: usize = 0;
            // Two adjacent 4-px groups per iteration: the tap loop is bound by
            // its 2-deep dependent-FMA chain (~8-9 cy per 2 taps), so an 8-lane
            // body retires 8 px per chain step instead of 4 at the same latency.
            // Low lanes = group x / list la, high lanes = group x+4 / list lb —
            // exactly the loads/taps two 4-wide iterations would do, in the same
            // per-lane order, so the accumulation is bit-exact per pixel.
            while (x + 8 <= width) : (x += 8) {
                const la = (start + (x >> 2)) % NBR;
                const lb = if (la + 1 == NBR) 0 else la + 1;
                const cla = pls[la * k ..][0..k];
                const clb = pls[lb * k ..][0..k];
                const base: usize = center_base + x;
                const cen: v8 = src_cache[base..][0..8].*;
                const cen_ref: v8 = ref_cache[base..][0..8].*;
                var sum: v8 = zero8;
                var sum_w: v8 = zero8;
                for (cla, clb) |pa, pb| {
                    const offa: usize = @intCast(@as(isize, @intCast(base)) + @as(isize, pa.y) * cstride_i + @as(isize, pa.x));
                    const offb: usize = @intCast(@as(isize, @intCast(base + 4)) + @as(isize, pb.y) * cstride_i + @as(isize, pb.x));
                    const v = std.simd.join(@as(v4, src_cache[offa..][0..4].*), @as(v4, src_cache[offb..][0..4].*));
                    const vr = std.simd.join(@as(v4, ref_cache[offa..][0..4].*), @as(v4, ref_cache[offb..][0..4].*));
                    const diff = v - cen;
                    const dist = @abs(vr - cen_ref);
                    const wgt = @max(@min(m8 - dist, wmax8), zero8);
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
                const cl = pls[list_idx * k ..][0..k];
                const base: usize = center_base + x;
                const cen: v4 = src_cache[base..][0..4].*;
                const cen_ref: v4 = ref_cache[base..][0..4].*;
                var sum: v4 = zero4;
                var sum_w: v4 = zero4;
                for (cl) |pt| {
                    const off: usize = @intCast(@as(isize, @intCast(base)) + @as(isize, pt.y) * cstride_i + @as(isize, pt.x));
                    const v: v4 = src_cache[off..][0..4].*;
                    const vr: v4 = ref_cache[off..][0..4].*;
                    const diff = v - cen;
                    const dist = @abs(vr - cen_ref);
                    const wgt = @max(@min(m4 - dist, wmax4), zero4);
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
                    const wgt = @max(@min(m_v - dist, wmax_v), zero_v);
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
