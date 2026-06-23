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

    // Build the mirror-padded float cache(s).
    var cy: usize = 0;
    while (cy < cheight) : (cy += 1) {
        const my: usize = @intCast(mirror(@as(i32, @intCast(cy)) - rv_i, h_i));
        const srow = src[my * stride ..];
        const scrow = src_cache[cy * cstride ..];
        var cx: usize = 0;
        while (cx < cstride) : (cx += 1) {
            const mx: usize = @intCast(mirror(@as(i32, @intCast(cx)) - rh_i, w_i));
            scrow[cx] = toCache(T, srow[mx]);
        }
        if (ref) |refp| {
            const rrow = refp[my * stride ..];
            const rcrow = ref_cache[cy * cstride ..];
            cx = 0;
            while (cx < cstride) : (cx += 1) {
                const mx: usize = @intCast(mirror(@as(i32, @intCast(cx)) - rh_i, w_i));
                rcrow[cx] = toCache(T, rrow[mx]);
            }
        }
    }

    const cstride_i: isize = @intCast(cstride);

    if (point_lists) |pls| {
        // ---- sub-sampled "speed hack" (4-wide, matching the SSE subspl path) ----
        const v4 = @Vector(4, f32);
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
