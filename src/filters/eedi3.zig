const std = @import("std");
const vapoursynth = @import("vapoursynth");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;

pub const io: std.Io = std.Io.Threaded.global_single_threaded.io();
pub const allocator = std.heap.c_allocator;

pub const Scratch = struct {
    r3p: []align(vec_align) f32,
    r1p: []align(vec_align) f32,
    r1n: []align(vec_align) f32,
    r3n: []align(vec_align) f32,
    t_base: []align(vec_align) f32,
    t_costs: []align(vec_align) f32,
    pbackt: []align(vec_align) i8,
    fpath: []align(vec_align) i32,
    dmap: []i32,
    tline: []f32,
    bmask: []bool,
    block_active: []bool,

    hp3p: []align(vec_align) f32,
    hp1p: []align(vec_align) f32,
    hp1n: []align(vec_align) f32,
    hp3n: []align(vec_align) f32,

    // horizontal-only (fully-transposed pipeline): column-major scratch frames.
    // srcT/dstT hold the source and destination in transposed (column-major)
    // layout so all per-line gathers/scatters/vcheck become contiguous; maskT
    // and scpT mirror the mclip/sclip frames in the same layout. Sized off
    // plane 0 in allocScratch.
    srcT: []align(vec_align) f32,
    dstT: []align(vec_align) f32,
    maskT: []u8,
    scpT: []align(vec_align) f32,
};

pub const Data = struct {
    node: ?*vs.Node = null,
    sclip: ?*vs.Node = null,
    mclip: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    mdis: u8 = 0,
    nrad: u8 = 0,
    alpha: f32 = 0,
    beta: f32 = 0,
    gamma: f32 = 0,
    one_minus_ab: f32 = 0,

    hp: bool = false,
    dh: bool = false,
    horizontal: bool = false,
    field: u8 = 0,
    vcheck: u8 = 0,
    vthresh0: f32 = 0,
    vthresh1: f32 = 0,
    vthresh2: f32 = 0,
    rcpVthresh0: f32 = 0,
    rcpVthresh1: f32 = 0,
    rcpVthresh2: f32 = 0,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *Scratch) = undefined,
};

pub const mdis_max = 40;
pub const nrad_max = 3;
pub const tpitch_max = mdis_max * 2 + 1;
pub const tpitch_hp_max = mdis_max * 4 + 1;
pub const dp_block = 64;

pub const n_vec = std.simd.suggestVectorLength(f32) orelse 8;
pub const Vec = @Vector(n_vec, f32);
pub const vec_align: u32 = @max(32, n_vec * @sizeOf(f32));

pub const pad_h = std.mem.alignForward(u32, 2 * mdis_max + nrad_max + n_vec, n_vec);
pub const pad_buf_w = pad_h * 2;

pub const flt_max: f32 = std.math.floatMax(f32);
pub const flt_max_09: f32 = flt_max * 0.9;

inline fn shiftInMask(comptime s: u32) [n_vec]i32 {
    var m: [n_vec]i32 = undefined;
    for (0..n_vec) |i| m[i] = if (i >= s) @intCast(i - s) else -1;
    return m;
}

inline fn prefixSum(v: Vec) Vec {
    const zero: Vec = @splat(0.0);
    var p = v;
    comptime var s: u32 = 1;
    inline while (s < n_vec) : (s *= 2) {
        p += @shuffle(f32, p, zero, shiftInMask(s));
    }
    return p;
}

/// Mirror-reflect a line index (no edge duplication; fractional axis), matching
/// C++ copyPad. `h` is the number of real samples along the reflected axis.
pub inline fn reflectRow(y: i32, h: i32) u32 {
    if (h == 1) return 0;
    var r = y;
    while (r < 0 or r >= h) {
        if (r < 0) r = -r;
        if (r >= h) r = 2 * (h - 1) - r;
    }
    return @intCast(r);
}

/// Reflected SOURCE perpendicular index for an output-line offset. Shared by the
/// interp-stencil fill and vCheck across both axes; mirrors `reflectRow` with the
/// dh doubling (output line `2*k` maps to source line `k`). For the vertical path
/// `n_src` is the source height; for the horizontal path it is the source width.
pub inline fn srcCol(dh: bool, off: i32, n_src_i: i32) u32 {
    return if (dh) reflectRow(off, 2 * n_src_i) / 2 else reflectRow(off, n_src_i);
}

/// Mirror-pad the `pad_h` margins on both sides of the real region `[pad_h, pad_h+w)`.
pub fn mirrorPad(buf: []f32, w: u32) void {
    for (0..pad_h) |i| buf[pad_h + w + i] = buf[pad_h + w - 2 - i];
    for (0..pad_h) |i| buf[i] = buf[2 * pad_h - i];
}

/// Vertical path: copy a contiguous source row into a padded line buffer.
pub fn fillPaddedRow(buf: []f32, src: []const f32, w: u32) void {
    @memcpy(buf[pad_h..][0..w], src[0..w]);
    mirrorPad(buf, w);
}

/// Tile size for the boundary transposes (fully-transposed horizontal pipeline).
pub const tr_tile = 32;

/// Interleave (unpack-lo/unpack-hi) shuffle masks at lane granularity `g` for a
/// pair of vectors `a` (positive indices) and `b` (negative indices, -1-i).
inline fn unpackMasks(comptime g: u32) struct { lo: [n_vec]i32, hi: [n_vec]i32 } {
    var lo: [n_vec]i32 = undefined;
    var hi: [n_vec]i32 = undefined;
    var out: u32 = 0;
    var blk: u32 = 0;
    while (out < n_vec) : (blk += 1) {
        var t: u32 = 0;
        while (t < g) : (t += 1) {
            lo[out] = @intCast(blk * g + t);
            out += 1;
        }
        t = 0;
        while (t < g) : (t += 1) {
            lo[out] = -1 - @as(i32, @intCast(blk * g + t));
            out += 1;
        }
    }
    out = 0;
    blk = 0;
    const half = n_vec / 2;
    while (out < n_vec) : (blk += 1) {
        var t: u32 = 0;
        while (t < g) : (t += 1) {
            hi[out] = @intCast(half + blk * g + t);
            out += 1;
        }
        t = 0;
        while (t < g) : (t += 1) {
            hi[out] = -1 - @as(i32, @intCast(half + blk * g + t));
            out += 1;
        }
    }
    return .{ .lo = lo, .hi = hi };
}

inline fn log2int(comptime n: u32) u32 {
    var l: u32 = 0;
    var x = n;
    while (x > 1) : (x >>= 1) l += 1;
    return l;
}

inline fn bitrev(comptime x: u32, comptime bits: u32) u32 {
    var r: u32 = 0;
    var v = x;
    var i: u32 = 0;
    while (i < bits) : (i += 1) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

/// In-register transpose of an `n_vec`×`n_vec` f32 block held in `n_vec` row
/// vectors → `n_vec` column vectors. Decimation-in-time butterfly with a
/// bit-reversed input ordering (cancels the FFT-style reversal). Valid for any
/// power-of-2 `n_vec`; verified against a scalar reference.
inline fn transposeReg(rows_in: [n_vec]Vec) [n_vec]Vec {
    const bits = comptime log2int(n_vec);
    var v: [n_vec]Vec = undefined;
    inline for (0..n_vec) |i| v[i] = rows_in[comptime bitrev(i, bits)];
    comptime var dist: u32 = n_vec / 2;
    comptime var g: u32 = 1;
    inline while (dist >= 1) : ({
        dist /= 2;
        g *= 2;
    }) {
        var out: [n_vec]Vec = v;
        const m = comptime unpackMasks(g);
        comptime var base: u32 = 0;
        inline while (base < n_vec) : (base += 2 * dist) {
            comptime var k: u32 = 0;
            inline while (k < dist) : (k += 1) {
                const a = v[base + k];
                const b = v[base + dist + k];
                out[base + k] = @shuffle(f32, a, b, m.lo);
                out[base + dist + k] = @shuffle(f32, a, b, m.hi);
            }
        }
        v = out;
        if (dist == 1) break;
    }
    return v;
}

/// Vectorized f32 transpose using in-register `n_vec`×`n_vec` blocks for the
/// aligned interior, scalar fallback for the ragged right/bottom borders. Writes
/// `dst[c*dst_stride + r] = src[r*src_stride + c]`.
pub fn transposeF32(
    dst: []f32,
    dst_stride: u32,
    src: []const f32,
    src_stride: u32,
    w: u32,
    h: u32,
) void {
    const wv = w - (w % n_vec);
    const hv = h - (h % n_vec);

    var r0: u32 = 0;
    while (r0 < hv) : (r0 += n_vec) {
        var c0: u32 = 0;
        while (c0 < wv) : (c0 += n_vec) {
            var rows: [n_vec]Vec = undefined;
            inline for (0..n_vec) |i| {
                rows[i] = src[(r0 + i) * src_stride + c0 ..][0..n_vec].*;
            }
            const cols = transposeReg(rows);
            inline for (0..n_vec) |i| {
                dst[(c0 + i) * dst_stride + r0 ..][0..n_vec].* = cols[i];
            }
        }
        // ragged right columns for these rows
        var c: u32 = wv;
        while (c < w) : (c += 1) {
            inline for (0..n_vec) |i| {
                dst[c * dst_stride + r0 + i] = src[(r0 + i) * src_stride + c];
            }
        }
    }
    // ragged bottom rows (all columns)
    var r: u32 = hv;
    while (r < h) : (r += 1) {
        const src_row = src[r * src_stride ..];
        var c: u32 = 0;
        while (c < w) : (c += 1) dst[c * dst_stride + r] = src_row[c];
    }
}

/// Blocked transpose: writes `dst[c*dst_stride + r] = src[r*src_stride + c]` for
/// r in [0,h), c in [0,w). Cache-tiled so neither read nor write streams thrash
/// cache lines. Generic over element type (used for the small u8 mask; the f32
/// frame data uses the vectorized `transposeF32`). This is the only strided work
/// in the fully-transposed pipeline; everything between the two boundary
/// transposes is contiguous.
pub fn transposeBlocked(
    comptime T: type,
    dst: []T,
    dst_stride: u32,
    src: []const T,
    src_stride: u32,
    w: u32,
    h: u32,
) void {
    var r0: u32 = 0;
    while (r0 < h) : (r0 += tr_tile) {
        const r1 = @min(r0 + tr_tile, h);
        var c0: u32 = 0;
        while (c0 < w) : (c0 += tr_tile) {
            const c1 = @min(c0 + tr_tile, w);
            var r: u32 = r0;
            while (r < r1) : (r += 1) {
                const src_row = src[r * src_stride ..];
                var c: u32 = c0;
                while (c < c1) : (c += 1) {
                    dst[c * dst_stride + r] = src_row[c];
                }
            }
        }
    }
}

pub fn buildBmask(bmask: []bool, maskp: []const u8, w: u32, mdis: u32) void {
    const minmdis = @min(w, mdis);
    var last: i64 = -666999;

    var x: u32 = 0;
    while (x < minmdis) : (x += 1) {
        if (maskp[x] != 0) last = @as(i64, x) + mdis;
    }

    x = 0;
    while (x < w - minmdis) : (x += 1) {
        if (maskp[x + mdis] != 0) last = @as(i64, x) + mdis * 2;
        bmask[x] = (@as(i64, x) <= last);
    }

    x = w - minmdis;
    while (x < w) : (x += 1) {
        bmask[x] = (@as(i64, x) <= last);
    }
}

inline fn recomputeWindow(t_base_buf: []const f32, x: i32, u: i32, two_u: i32, nrad_i: i32) struct { f32, f32, f32 } {
    var sw0: f32 = 0;
    var sw1: f32 = 0;
    var sw2: f32 = 0;
    var ki: i32 = -nrad_i;
    while (ki <= nrad_i) : (ki += 1) {
        sw0 += t_base_buf[@intCast(x + u + ki + pad_h)];
        sw1 += t_base_buf[@intCast(x + ki + pad_h)];
        sw2 += t_base_buf[@intCast(x + two_u + ki + pad_h)];
    }
    return .{ sw0, sw1, sw2 };
}

inline fn costBlock(
    tcosts_ptr: []f32,
    t_base_buf: []const f32,
    r1p: []const f32,
    r1n: []const f32,
    x: u32,
    u: i32,
    two_u: i32,
    nrad_i: i32,
    sw0: *f32,
    sw1: *f32,
    sw2: *f32,
    alpha: f32,
    beta_abs_u: f32,
    one_minus_ab: f32,
) void {
    const xi: i32 = @intCast(x);
    const zero_v: Vec = @splat(0.0);
    const add0: Vec = t_base_buf[@intCast(xi + u + nrad_i + 1 + pad_h)..][0..n_vec].*;
    const sub0: Vec = t_base_buf[@intCast(xi + u - nrad_i + pad_h)..][0..n_vec].*;
    const add1: Vec = t_base_buf[@intCast(xi + nrad_i + 1 + pad_h)..][0..n_vec].*;
    const sub1: Vec = t_base_buf[@intCast(xi - nrad_i + pad_h)..][0..n_vec].*;
    const add2: Vec = t_base_buf[@intCast(xi + two_u + nrad_i + 1 + pad_h)..][0..n_vec].*;
    const sub2: Vec = t_base_buf[@intCast(xi + two_u - nrad_i + pad_h)..][0..n_vec].*;
    const d0: Vec = add0 - sub0;
    const d1: Vec = add1 - sub1;
    const d2: Vec = add2 - sub2;

    const ps0: Vec = prefixSum(d0);
    const ps1: Vec = prefixSum(d1);
    const ps2: Vec = prefixSum(d2);
    const excl0 = @shuffle(f32, ps0, zero_v, shiftInMask(1));
    const excl1 = @shuffle(f32, ps1, zero_v, shiftInMask(1));
    const excl2 = @shuffle(f32, ps2, zero_v, shiftInMask(1));
    const sw0_v = @as(Vec, @splat(sw0.*)) + excl0;
    const sw1_v = @as(Vec, @splat(sw1.*)) + excl1;
    const sw2_v = @as(Vec, @splat(sw2.*)) + excl2;

    sw0.* += ps0[n_vec - 1];
    sw1.* += ps1[n_vec - 1];
    sw2.* += ps2[n_vec - 1];
    const s1p_xu: Vec = r1p[@intCast(xi + u + pad_h)..][0..n_vec].*;
    const s1n_xmu: Vec = r1n[@intCast(xi - u + pad_h)..][0..n_vec].*;
    const ip_v: Vec = (s1p_xu + s1n_xmu) * @as(Vec, @splat(0.5));
    const s1p_x: Vec = r1p[@intCast(xi + pad_h)..][0..n_vec].*;
    const s1n_x: Vec = r1n[@intCast(xi + pad_h)..][0..n_vec].*;
    const v_v: Vec = @abs(s1p_x - ip_v) + @abs(s1n_x - ip_v);

    const cost_v =
        @as(Vec, @splat(alpha)) * (sw0_v + sw1_v + sw2_v) +
        @as(Vec, @splat(beta_abs_u)) +
        @as(Vec, @splat(one_minus_ab)) * v_v;

    tcosts_ptr[x..][0..n_vec].* = cost_v;
}

pub fn interpLine(
    r3p: []const f32,
    r1p: []const f32,
    r1n: []const f32,
    r3n: []const f32,
    dstp_row: []f32,
    pbackt: []i8,
    fpath: []i32,
    t_base_buf: []f32,
    t_costs: []f32,
    dmap_row: []i32,
    stride: u32,
    w: u32,
    mdis: u8,
    nrad: u8,
    alpha: f32,
    beta: f32,
    gamma_v: f32,
    one_minus_ab: f32,
    bmask: ?[]const bool,
    block_active: []bool,
) void {
    const mdis_i: i32 = mdis;
    const nrad_i: i32 = nrad;
    const tpitch: u32 = mdis * 2 + 1;
    if (w == 0) return;

    var any_active = false;
    if (bmask) |bm| {
        const nblocks = w / n_vec;
        var b: u32 = 0;
        while (b < nblocks) : (b += 1) {
            const mv: @Vector(n_vec, bool) = bm[b * n_vec ..][0..n_vec].*;
            const act = @reduce(.Or, mv);
            block_active[b] = act;
            any_active = any_active or act;
        }
        var xt: u32 = nblocks * n_vec;
        while (xt < w) : (xt += 1) {
            if (bm[xt]) any_active = true;
        }
    }

    if (bmask != null and !any_active) {
        for (0..w) |x| {
            const xi: i32 = @intCast(x);
            const bi: u32 = @intCast(xi + pad_h);
            dmap_row[x] = 0;
            dstp_row[x] = 0.5625 * (r1p[bi] + r1n[bi]) - 0.0625 * (r3p[bi] + r3n[bi]);
        }
        return;
    }

    var u: i32 = -mdis_i;
    while (u <= mdis) : (u += 1) {
        const two_u: i32 = u * 2;
        const u_idx: u32 = @intCast(mdis_i + u);
        const abs_u_f: f32 = @floatFromInt(@abs(u));

        const u_lo: i32 = @min(u, @min(0, two_u));
        const u_hi: i32 = @max(u, @max(0, two_u));
        const j_lo: i32 = u_lo - nrad_i;
        const j_hi: i32 = @as(i32, @intCast(w)) + u_hi + nrad_i + 1;
        const j_lo_v: i32 = @divFloor(j_lo, n_vec) * n_vec;
        const j_hi_v: i32 = @divFloor(j_hi + (n_vec - 1), n_vec) * n_vec;

        var j: i32 = j_lo_v;
        while (j < j_hi_v) : (j += n_vec) {
            const bi: u32 = @intCast(j + pad_h);
            const a: Vec = r3p[bi..][0..n_vec].*;
            const b: Vec = r1p[@intCast(j - two_u + pad_h)..][0..n_vec].*;
            const c: Vec = r1p[bi..][0..n_vec].*;
            const d: Vec = r1n[@intCast(j - two_u + pad_h)..][0..n_vec].*;
            const e: Vec = r1n[bi..][0..n_vec].*;
            const f_: Vec = r3n[@intCast(j - two_u + pad_h)..][0..n_vec].*;
            t_base_buf[bi..][0..n_vec].* = @abs(a - b) + @abs(c - d) + @abs(e - f_);
        }

        var sws = recomputeWindow(t_base_buf, 0, u, two_u, nrad_i);
        var sw0 = sws[0];
        var sw1 = sws[1];
        var sw2 = sws[2];

        const tcosts_ptr = t_costs[u_idx * stride ..];
        const beta_abs_u = beta * abs_u_f;

        var x: u32 = 0;
        if (bmask != null) {
            var stale = false;
            while (x + n_vec <= w) : (x += n_vec) {
                if (x != 0 and !block_active[x / n_vec]) {
                    stale = true;
                    continue;
                }
                if (stale) {
                    sws = recomputeWindow(t_base_buf, @intCast(x), u, two_u, nrad_i);
                    sw0 = sws[0];
                    sw1 = sws[1];
                    sw2 = sws[2];
                    stale = false;
                }
                costBlock(tcosts_ptr, t_base_buf, r1p, r1n, x, u, two_u, nrad_i, &sw0, &sw1, &sw2, alpha, beta_abs_u, one_minus_ab);
            }
            if (stale and x < w) {
                sws = recomputeWindow(t_base_buf, @intCast(x), u, two_u, nrad_i);
                sw0 = sws[0];
                sw1 = sws[1];
                sw2 = sws[2];
            }
        } else {
            while (x + n_vec <= w) : (x += n_vec) {
                costBlock(tcosts_ptr, t_base_buf, r1p, r1n, x, u, two_u, nrad_i, &sw0, &sw1, &sw2, alpha, beta_abs_u, one_minus_ab);
            }
        }

        while (x < w) : (x += 1) {
            const xi: i32 = @intCast(x);
            const ip = (r1p[@intCast(xi + u + pad_h)] + r1n[@intCast(xi - u + pad_h)]) * 0.5;
            const v = @abs(r1p[@intCast(xi + pad_h)] - ip) + @abs(r1n[@intCast(xi + pad_h)] - ip);
            tcosts_ptr[x] = alpha * (sw0 + sw1 + sw2) + beta_abs_u + one_minus_ab * v;
            sw0 += t_base_buf[@intCast(xi + u + nrad_i + 1 + pad_h)] - t_base_buf[@intCast(xi + u - nrad_i + pad_h)];
            sw1 += t_base_buf[@intCast(xi + nrad_i + 1 + pad_h)] - t_base_buf[@intCast(xi - nrad_i + pad_h)];
            sw2 += t_base_buf[@intCast(xi + two_u + nrad_i + 1 + pad_h)] - t_base_buf[@intCast(xi + two_u - nrad_i + pad_h)];
        }
    }

    var pcosts_buf: [2][tpitch_max + 2]f32 = undefined;
    for (&pcosts_buf[0]) |*v| v.* = flt_max_09;
    for (&pcosts_buf[1]) |*v| v.* = flt_max_09;
    var ping: u32 = 0;

    for (0..tpitch) |ui| {
        pcosts_buf[ping][ui + 1] = t_costs[ui * stride + 0];
    }

    var block_cost_x_major: [dp_block * tpitch_max]f32 = undefined;
    const gamma_vv: Vec = @splat(gamma_v);
    const flt_max_v: Vec = @splat(flt_max_09);
    const neg1_i8: @Vector(n_vec, i8) = @splat(-1);
    const zero_i8: @Vector(n_vec, i8) = @splat(0);
    const one_i8: @Vector(n_vec, i8) = @splat(1);

    var xs: u32 = 1;
    while (xs < w) {
        const xe = @min(xs + dp_block, w);
        const bw = xe - xs;

        for (0..tpitch) |ui| {
            const src_ptr = t_costs[ui * stride + xs ..];
            for (0..bw) |x_local| {
                block_cost_x_major[x_local * tpitch_max + ui] = src_ptr[x_local];
            }
        }

        for (0..bw) |x_local| {
            const x = xs + x_local;
            const piT = pbackt[(x - 1) * tpitch ..][0..tpitch];
            const tcost_base = block_cost_x_major[x_local * tpitch_max ..];

            if (bmask) |bm| {
                if (!bm[x]) {
                    const pong = ping ^ 1;
                    if (x == 1) {
                        for (0..tpitch) |ui| pcosts_buf[pong][ui + 1] = tcost_base[ui];
                        @memset(piT, 0);
                    } else {
                        pcosts_buf[pong] = pcosts_buf[ping];
                        @memcpy(piT, pbackt[(x - 2) * tpitch ..][0..tpitch]);
                    }
                    ping = pong;
                    continue;
                }
            }

            const pong = ping ^ 1;
            const p = &pcosts_buf[ping];
            const p_out = &pcosts_buf[pong];

            var u_idx: u32 = 0;
            while (u_idx + n_vec <= tpitch) : (u_idx += n_vec) {
                const p_left: Vec = p[u_idx..][0..n_vec].*;
                const p_cent: Vec = p[u_idx + 1 ..][0..n_vec].*;
                const p_right: Vec = p[u_idx + 2 ..][0..n_vec].*;

                const left_cc = p_left + gamma_vv;
                const right_cc = p_right + gamma_vv;

                const left_wins = left_cc < p_cent;
                const min1 = @select(f32, left_wins, left_cc, p_cent);
                const delta1 = @select(i8, left_wins, neg1_i8, zero_i8);

                const right_wins = right_cc < min1;
                const bval = @select(f32, right_wins, right_cc, min1);
                const best_delta = @select(i8, right_wins, one_i8, delta1);

                const tcost_v: Vec = tcost_base[u_idx..][0..n_vec].*;
                p_out[u_idx + 1 ..][0..n_vec].* = @min(bval + tcost_v, flt_max_v);
                piT[u_idx..][0..n_vec].* = best_delta;
            }

            while (u_idx < tpitch) : (u_idx += 1) {
                const left_cc = p[u_idx] + gamma_v;
                const cent_cc = p[u_idx + 1];
                const right_cc = p[u_idx + 2] + gamma_v;

                var bval = cent_cc;
                var best_delta: i8 = 0;
                if (left_cc < bval) {
                    bval = left_cc;
                    best_delta = -1;
                }
                if (right_cc < bval) {
                    bval = right_cc;
                    best_delta = 1;
                }

                p_out[u_idx + 1] = @min(bval + tcost_base[u_idx], flt_max_09);
                piT[u_idx] = best_delta;
            }
            ping ^= 1;
        }
        xs = xe;
    }

    fpath[w - 1] = 0;
    if (w >= 2) {
        var bx: u32 = w - 2;
        while (true) : (bx -= 1) {
            const u_idx: u32 = @intCast(mdis_i + fpath[bx + 1]);
            fpath[bx] = fpath[bx + 1] + pbackt[bx * tpitch + u_idx];
            if (bx == 0) break;
        }
    }

    if (bmask) |bm| {
        const zero_iv: @Vector(n_vec, i32) = @splat(0);
        var x: u32 = 0;
        while (x + n_vec <= w) : (x += n_vec) {
            const m: @Vector(n_vec, bool) = bm[x..][0..n_vec].*;
            const fp: @Vector(n_vec, i32) = fpath[x..][0..n_vec].*;
            fpath[x..][0..n_vec].* = @select(i32, m, fp, zero_iv);
        }
        while (x < w) : (x += 1) {
            if (!bm[x]) fpath[x] = 0;
        }
    }

    for (0..w) |x| {
        const xi: i32 = @intCast(x);
        const dir: i32 = fpath[x];
        dmap_row[x] = dir;
        const dir_i: i32 = @intCast(dir);
        const ad: u32 = @intCast(@abs(dir));
        dstp_row[x] = if (x >= ad * 3 and x + ad * 3 <= w - 1)
            0.5625 * (r1p[@intCast(xi + dir_i + pad_h)] + r1n[@intCast(xi - dir_i + pad_h)]) -
                0.0625 * (r3p[@intCast(xi + dir_i * 3 + pad_h)] + r3n[@intCast(xi - dir_i * 3 + pad_h)])
        else
            (r1p[@intCast(xi + dir_i + pad_h)] + r1n[@intCast(xi - dir_i + pad_h)]) * 0.5;
    }
}

pub inline fn pidx(a: i32) usize {
    return @intCast(a + @as(i32, @intCast(pad_h)));
}

inline fn ldv(buf: []const f32, a: i32) Vec {
    return buf[pidx(a)..][0..n_vec].*;
}

fn computeHpRow(dst: []f32, a: []const f32) void {
    const n = a.len;
    const c9: Vec = @splat(0.5625);
    const c1: Vec = @splat(0.0625);
    var j: usize = 1;
    while (j + n_vec <= n - 2) : (j += n_vec) {
        const a0: Vec = a[j - 1 ..][0..n_vec].*;
        const a1: Vec = a[j..][0..n_vec].*;
        const a2: Vec = a[j + 1 ..][0..n_vec].*;
        const a3: Vec = a[j + 2 ..][0..n_vec].*;
        dst[j..][0..n_vec].* = c9 * (a1 + a2) - c1 * (a0 + a3);
    }
    while (j < n - 2) : (j += 1) {
        dst[j] = 0.5625 * (a[j] + a[j + 1]) - 0.0625 * (a[j - 1] + a[j + 2]);
    }
}

pub fn interpLineHP(
    r3p: []const f32,
    r1p: []const f32,
    r1n: []const f32,
    r3n: []const f32,
    hp3p: []f32,
    hp1p: []f32,
    hp1n: []f32,
    hp3n: []f32,
    dstp_row: []f32,
    pbackt: []i8,
    fpath: []i32,
    t_costs: []f32,
    dmap_row: []i32,
    stride: u32,
    w: u32,
    mdis: u8,
    nrad: u8,
    alpha3: f32,
    beta255: f32,
    gamma255: f32,
    one_minus_ab: f32,
    bmask: ?[]const bool,
) void {
    if (w == 0) return;
    const nrad_i: i32 = nrad;
    const mdis_i: i32 = mdis;
    const cen_i: i32 = 2 * mdis_i;
    const tpitch: u32 = @intCast(4 * mdis_i + 1);

    computeHpRow(hp3p, r3p);
    computeHpRow(hp1p, r1p);
    computeHpRow(hp1n, r1n);
    computeHpRow(hp3n, r3n);

    if (bmask) |bm| {
        var any = false;
        for (0..w) |x| {
            if (bm[x]) {
                any = true;
                break;
            }
        }
        if (!any) {
            for (0..w) |x| {
                const bi = pidx(@intCast(x));
                dmap_row[x] = 0;
                dstp_row[x] = 0.5625 * (r1p[bi] + r1n[bi]) - 0.0625 * (r3p[bi] + r3n[bi]);
            }
            return;
        }
    }

    var u: i32 = -cen_i;
    while (u <= cen_i) : (u += 1) {
        const uh: i32 = u >> 1;
        const odd = (u & 1) != 0;
        const lo0: i32 = if (odd) -uh - 1 else -uh;
        const A0 = if (odd) hp3p else r3p;
        const B0 = if (odd) hp1p else r1p;
        const C0 = if (odd) hp1n else r1n;
        const D0 = if (odd) hp3n else r3n;
        const u_idx: u32 = @intCast(cen_i + u);
        const tc = t_costs[u_idx * stride ..];
        const beta_term: f32 = beta255 * @as(f32, @floatFromInt(@abs(u))) * 0.5;
        const alpha_v: Vec = @splat(alpha3);
        const beta_v: Vec = @splat(beta_term);
        const oneab_v: Vec = @splat(one_minus_ab);
        const half_v: Vec = @splat(0.5);

        var x: u32 = 0;
        while (x + n_vec <= w) : (x += n_vec) {
            const xi: i32 = @intCast(x);
            var s0: Vec = @splat(0);
            var s1: Vec = @splat(0);
            var s2: Vec = @splat(0);
            var k: i32 = -nrad_i;
            while (k <= nrad_i) : (k += 1) {
                s1 += @abs(ldv(r3p, xi + k) - ldv(r1p, xi - u + k)) +
                    @abs(ldv(r1p, xi + k) - ldv(r1n, xi - u + k)) +
                    @abs(ldv(r1n, xi + k) - ldv(r3n, xi - u + k));
                s2 += @abs(ldv(r3p, xi + u + k) - ldv(r1p, xi + k)) +
                    @abs(ldv(r1p, xi + u + k) - ldv(r1n, xi + k)) +
                    @abs(ldv(r1n, xi + u + k) - ldv(r3n, xi + k));
                s0 += @abs(ldv(A0, xi + uh + k) - ldv(B0, xi + lo0 + k)) +
                    @abs(ldv(B0, xi + uh + k) - ldv(C0, xi + lo0 + k)) +
                    @abs(ldv(C0, xi + uh + k) - ldv(D0, xi + lo0 + k));
            }
            const ip: Vec = (ldv(B0, xi + uh) + ldv(C0, xi + lo0)) * half_v;
            const v: Vec = @abs(ldv(r1p, xi) - ip) + @abs(ldv(r1n, xi) - ip);
            tc[x..][0..n_vec].* = alpha_v * (s0 + s1 + s2) + beta_v + oneab_v * v;
        }

        while (x < w) : (x += 1) {
            const xi: i32 = @intCast(x);
            var s0: f32 = 0;
            var s1: f32 = 0;
            var s2: f32 = 0;
            var k: i32 = -nrad_i;
            while (k <= nrad_i) : (k += 1) {
                s1 += @abs(r3p[pidx(xi + k)] - r1p[pidx(xi - u + k)]) +
                    @abs(r1p[pidx(xi + k)] - r1n[pidx(xi - u + k)]) +
                    @abs(r1n[pidx(xi + k)] - r3n[pidx(xi - u + k)]);
                s2 += @abs(r3p[pidx(xi + u + k)] - r1p[pidx(xi + k)]) +
                    @abs(r1p[pidx(xi + u + k)] - r1n[pidx(xi + k)]) +
                    @abs(r1n[pidx(xi + u + k)] - r3n[pidx(xi + k)]);
                s0 += @abs(A0[pidx(xi + uh + k)] - B0[pidx(xi + lo0 + k)]) +
                    @abs(B0[pidx(xi + uh + k)] - C0[pidx(xi + lo0 + k)]) +
                    @abs(C0[pidx(xi + uh + k)] - D0[pidx(xi + lo0 + k)]);
            }
            const ip: f32 = (B0[pidx(xi + uh)] + C0[pidx(xi + lo0)]) * 0.5;
            const v = @abs(r1p[pidx(xi)] - ip) + @abs(r1n[pidx(xi)] - ip);
            tc[x] = alpha3 * (s0 + s1 + s2) + beta_term + one_minus_ab * v;
        }
    }

    var pcosts: [2][tpitch_hp_max + 4]f32 = undefined;
    for (&pcosts[0]) |*p| p.* = flt_max_09;
    for (&pcosts[1]) |*p| p.* = flt_max_09;
    var ping: u32 = 0;
    for (0..tpitch) |ui| pcosts[ping][ui + 2] = t_costs[ui * stride + 0];

    var tcol: [tpitch_hp_max]f32 = undefined;
    var xc: u32 = 1;
    while (xc < w) : (xc += 1) {
        const pong = ping ^ 1;
        const piT = pbackt[(xc - 1) * tpitch ..][0..tpitch];
        if (bmask) |bm| {
            if (!bm[xc]) {
                if (xc == 1) {
                    for (0..tpitch) |ui| pcosts[pong][ui + 2] = t_costs[ui * stride + xc];
                    @memset(piT, 0);
                } else {
                    pcosts[pong] = pcosts[ping];
                    @memcpy(piT, pbackt[(xc - 2) * tpitch ..][0..tpitch]);
                }
                ping = pong;
                continue;
            }
        }

        for (0..tpitch) |ui| tcol[ui] = t_costs[ui * stride + xc];

        const flt_v: Vec = @splat(flt_max_09);
        const g1_v: Vec = @splat(gamma255 * 0.5);
        const g2_v: Vec = @splat(gamma255);
        const i8m2: @Vector(n_vec, i8) = @splat(-2);
        const i8m1: @Vector(n_vec, i8) = @splat(-1);
        const i8z: @Vector(n_vec, i8) = @splat(0);
        const i8p1: @Vector(n_vec, i8) = @splat(1);
        const i8p2: @Vector(n_vec, i8) = @splat(2);
        var ui: u32 = 0;
        while (ui + n_vec <= tpitch) : (ui += n_vec) {
            const p = &pcosts[ping];
            const c_m2 = @min(@as(Vec, p[ui + 0 ..][0..n_vec].*) + g2_v, flt_v);
            const c_m1 = @min(@as(Vec, p[ui + 1 ..][0..n_vec].*) + g1_v, flt_v);
            const c_0 = @min(@as(Vec, p[ui + 2 ..][0..n_vec].*), flt_v);
            const c_p1 = @min(@as(Vec, p[ui + 3 ..][0..n_vec].*) + g1_v, flt_v);
            const c_p2 = @min(@as(Vec, p[ui + 4 ..][0..n_vec].*) + g2_v, flt_v);
            var bval = c_m2;
            var bd = i8m2;
            var m = c_m1 < bval;
            bval = @select(f32, m, c_m1, bval);
            bd = @select(i8, m, i8m1, bd);
            m = c_0 < bval;
            bval = @select(f32, m, c_0, bval);
            bd = @select(i8, m, i8z, bd);
            m = c_p1 < bval;
            bval = @select(f32, m, c_p1, bval);
            bd = @select(i8, m, i8p1, bd);
            m = c_p2 < bval;
            bval = @select(f32, m, c_p2, bval);
            bd = @select(i8, m, i8p2, bd);
            const tcv: Vec = tcol[ui..][0..n_vec].*;
            pcosts[pong][ui + 2 ..][0..n_vec].* = @min(bval + tcv, flt_v);
            piT[ui..][0..n_vec].* = bd;
        }
        while (ui < tpitch) : (ui += 1) {
            var bval: f32 = flt_max_09;
            var best_delta: i8 = 0;
            var dv: i32 = -2;
            while (dv <= 2) : (dv += 1) {
                const vi: i32 = @as(i32, @intCast(ui)) + dv;
                const gv = gamma255 * @as(f32, @floatFromInt(@abs(dv))) * 0.5;
                const cc = @min(pcosts[ping][@intCast(vi + 2)] + gv, flt_max_09);
                if (cc < bval) {
                    bval = cc;
                    best_delta = @intCast(dv);
                }
            }
            pcosts[pong][ui + 2] = @min(bval + tcol[ui], flt_max_09);
            piT[ui] = best_delta;
        }
        ping = pong;
    }

    fpath[w - 1] = 0;
    if (w >= 2) {
        var bx: u32 = w - 2;
        while (true) : (bx -= 1) {
            const ui: u32 = @intCast(cen_i + fpath[bx + 1]);
            fpath[bx] = fpath[bx + 1] + pbackt[bx * tpitch + ui];
            if (bx == 0) break;
        }
    }

    for (0..w) |xx_| {
        const xx: u32 = @intCast(xx_);
        const xi: i32 = @intCast(xx);
        if (bmask) |bm| {
            if (!bm[xx]) {
                dmap_row[xx] = 0;
                dstp_row[xx] = 0.5625 * (r1p[pidx(xi)] + r1n[pidx(xi)]) - 0.0625 * (r3p[pidx(xi)] + r3n[pidx(xi)]);
                continue;
            }
        }
        const dir: i32 = fpath[xx];
        dmap_row[xx] = dir;
        if ((dir & 1) == 0) {
            const d2 = dir >> 1;
            const ad: u32 = @intCast(@abs(d2));
            if (xx >= ad * 3 and xx + ad * 3 <= w - 1) {
                dstp_row[xx] = 0.5625 * (r1p[pidx(xi + d2)] + r1n[pidx(xi - d2)]) -
                    0.0625 * (r3p[pidx(xi + d2 * 3)] + r3n[pidx(xi - d2 * 3)]);
            } else {
                dstp_row[xx] = (r1p[pidx(xi + d2)] + r1n[pidx(xi - d2)]) * 0.5;
            }
        } else {
            const d20 = dir >> 1;
            const d21 = (dir + 1) >> 1;
            const d30 = (dir * 3) >> 1;
            const d31 = (dir * 3 + 1) >> 1;
            const ad: u32 = @intCast(@max(@abs(d30), @abs(d31)));
            if (xx >= ad and xx + ad <= w - 1) {
                const c0 = r3p[pidx(xi + d30)] + r3p[pidx(xi + d31)];
                const c1 = r1p[pidx(xi + d20)] + r1p[pidx(xi + d21)];
                const c2 = r1n[pidx(xi - d20)] + r1n[pidx(xi - d21)];
                const c3 = r3n[pidx(xi - d30)] + r3n[pidx(xi - d31)];
                dstp_row[xx] = 0.28125 * (c1 + c2) - 0.03125 * (c0 + c3);
            } else {
                dstp_row[xx] = (r1p[pidx(xi + d20)] + r1p[pidx(xi + d21)] + r1n[pidx(xi - d20)] + r1n[pidx(xi - d21)]) * 0.25;
            }
        }
    }
}

/// Axis-agnostic vCheck post-pass. Operates on contiguous "lines": for the
/// vertical path a line is a dst row of length `L = w` and consecutive lines
/// (rows) are `lstride` apart; for the horizontal path a line is a dst column of
/// length `L = src_h` in the column-major `dst`/`src` scratch, again `lstride`
/// (= Lstride) apart. The body is therefore identical for both — only the
/// reflection bound (`n_src`) and the line geometry change.
///   `pd` is the perpendicular dst index (dst row for vertical, dst column for
///   horizontal); `i` is the position along the interpolated line; the warp
///   shifts `i`. All accesses are contiguous.
pub fn vcheckLine(
    src: []const f32,
    dst: []f32,
    scp: ?[]const f32,
    dmap: []const i32,
    tline: []f32,
    field: u8,
    L: u32, // interpolated-line length (w vertical / src_h horizontal)
    n_dst: u32, // number of dst lines (dst_h vertical / dst_w horizontal)
    n_src: u32, // number of src lines (reflection bound for ±3)
    lstride: u32, // distance between consecutive lines
    n_interp: u32,
    d: *const Data,
) void {
    var off: u32 = 1;
    while (off + 1 < n_interp) : (off += 1) {
        const pd: u32 = @as(u32, field) + 2 * off;
        if (pd < 2 or pd + 2 >= n_dst) continue;

        const dst_line = dst[pd * lstride ..];
        const dst1p = dst[(pd - 1) * lstride ..];
        const dst2p = dst[(pd - 2) * lstride ..];
        const dst1n = dst[(pd + 1) * lstride ..];
        const dst2n = dst[(pd + 2) * lstride ..];

        const pd_i: i32 = @intCast(pd);
        const n_src_i: i32 = @intCast(n_src);
        const c3p = srcCol(d.dh, pd_i - 3, n_src_i);
        const c3n = srcCol(d.dh, pd_i + 3, n_src_i);
        const dst3p = src[c3p * lstride ..];
        const dst3n = src[c3n * lstride ..];

        const dmap_cur = dmap[off * lstride ..];
        const dmap_prev = dmap[(off - 1) * lstride ..];
        const dmap_next = dmap[(off + 1) * lstride ..];
        const scp_line: ?[]const f32 = if (scp) |sc| sc[pd * lstride ..] else null;

        for (0..L) |i| {
            const dirc = dmap_cur[i];
            const cint = if (scp_line) |r|
                r[i]
            else
                0.5625 * (dst1p[i] + dst1n[i]) - 0.0625 * (dst3p[i] + dst3n[i]);

            if (dirc == 0) {
                tline[i] = cint;
                continue;
            }

            const dirt = dmap_prev[i];
            const dirb = dmap_next[i];

            if (@max(dirc * dirt, dirc * dirb) < 0 or (dirt == dirb and dirt == 0)) {
                tline[i] = cint;
                continue;
            }

            const ii: i32 = @intCast(i);
            const dirc_i: i32 = @intCast(dirc);
            const Li: i32 = @intCast(L);

            const maxoff: i32 = if (d.hp) blk: {
                if ((dirc & 1) == 0) break :blk @intCast(@abs(dirc_i >> 1));
                break :blk @intCast(@max(@abs(dirc_i >> 1), @abs((dirc_i + 1) >> 1)));
            } else @intCast(@abs(dirc_i));
            if (ii + maxoff >= Li or ii - maxoff < 0) {
                tline[i] = cint;
                continue;
            }

            var it: f32 = undefined;
            var ib: f32 = undefined;
            var vt: f32 = undefined;
            var vb: f32 = undefined;
            var dabs: u32 = undefined;
            if (d.hp and (dirc & 1) != 0) {
                const d20 = dirc_i >> 1;
                const d21 = (dirc_i + 1) >> 1;
                const ip0: u32 = @intCast(ii + d20);
                const ip1: u32 = @intCast(ii + d21);
                const im0: u32 = @intCast(ii - d20);
                const im1: u32 = @intCast(ii - d21);
                const s2psum = dst2p[ip0] + dst2p[ip1];
                const s1psum = dst1p[ip0] + dst1p[ip1];
                const pa0 = dst_line[ip0] + dst_line[ip1];
                const ps0 = dst_line[im0] + dst_line[im1];
                const s1nsum = dst1n[im0] + dst1n[im1];
                const s2nsum = dst2n[im0] + dst2n[im1];
                it = (s2psum + ps0) * 0.25;
                vt = (@abs(s2psum - s1psum) + @abs(pa0 - s1psum)) * 0.5;
                ib = (pa0 + s2nsum) * 0.25;
                vb = (@abs(s2nsum - s1nsum) + @abs(ps0 - s1nsum)) * 0.5;
                dabs = @intCast(@abs(dirc_i) >> 1);
            } else {
                const offh: i32 = if (d.hp) dirc_i >> 1 else dirc_i;
                const ipd: u32 = @intCast(ii + offh);
                const imd: u32 = @intCast(ii - offh);
                it = (dst2p[ipd] + dst_line[imd]) * 0.5;
                ib = (dst_line[ipd] + dst2n[imd]) * 0.5;
                vt = @abs(dst2p[ipd] - dst1p[ipd]) + @abs(dst_line[ipd] - dst1p[ipd]);
                vb = @abs(dst2n[imd] - dst1n[imd]) + @abs(dst_line[imd] - dst1n[imd]);
                dabs = if (d.hp) @intCast(@abs(dirc_i) >> 1) else @intCast(@abs(dirc_i));
            }
            const vc = @abs(dst_line[i] - dst1p[i]) + @abs(dst_line[i] - dst1n[i]);

            const d0 = @abs(it - dst1p[i]);
            const d1 = @abs(ib - dst1n[i]);
            const d2 = @abs(vt - vc);
            const d3 = @abs(vb - vc);

            const mdiff0: f32 = switch (d.vcheck) {
                1 => @min(d0, d1),
                2 => (d0 + d1) * 0.5,
                else => @max(d0, d1),
            };
            const mdiff1: f32 = switch (d.vcheck) {
                1 => @min(d2, d3),
                2 => (d2 + d3) * 0.5,
                else => @max(d2, d3),
            };

            const a0 = mdiff0 * d.rcpVthresh0;
            const a1 = mdiff1 * d.rcpVthresh1;
            const a2 = @max((d.vthresh2 - @as(f32, @floatFromInt(dabs))) * d.rcpVthresh2, 0.0);
            const a = @min(@max(a0, @max(a1, a2)), 1.0);

            tline[i] = (1.0 - a) * dst_line[i] + a * cint;
        }

        @memcpy(dst_line[0..L], tline[0..L]);
    }
}

/// `srcT_rows`/`dstT_rows` are the number of transposed rows (= src_w / dst_w)
/// for the fully-transposed horizontal pipeline; each row spans `stride`
/// elements. The vertical path passes 0 to skip those allocations.
pub fn allocScratch(w: u32, stride: u32, n_interp: u32, hp: bool, srcT_rows: u32, dstT_rows: u32) !*Scratch {
    const pad_buf_len = vsh.ceilN(w + pad_buf_w, vec_align);
    const tpitch_alloc: u32 = if (hp) tpitch_hp_max else tpitch_max;
    const s = try allocator.create(Scratch);
    s.r3p = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), pad_buf_len);
    s.r1p = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), pad_buf_len);
    s.r1n = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), pad_buf_len);
    s.r3n = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), pad_buf_len);
    s.t_base = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), pad_buf_len);
    s.t_costs = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), tpitch_alloc * stride);
    s.pbackt = try allocator.alignedAlloc(i8, .fromByteUnits(vec_align), stride * tpitch_alloc);
    s.fpath = try allocator.alignedAlloc(i32, .fromByteUnits(vec_align), stride);
    s.dmap = try allocator.alloc(i32, n_interp * stride);
    s.tline = try allocator.alloc(f32, stride);
    s.bmask = try allocator.alloc(bool, stride);
    s.block_active = try allocator.alloc(bool, stride / n_vec + 1);
    const hp_len: usize = if (hp) pad_buf_len else 0;
    s.hp3p = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), hp_len);
    s.hp1p = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), hp_len);
    s.hp1n = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), hp_len);
    s.hp3n = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), hp_len);
    s.srcT = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), srcT_rows * stride);
    s.dstT = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), dstT_rows * stride);
    s.maskT = try allocator.alloc(u8, srcT_rows * stride);
    s.scpT = try allocator.alignedAlloc(f32, .fromByteUnits(vec_align), dstT_rows * stride);
    return s;
}

pub fn freeScratch(s: *Scratch) void {
    allocator.free(s.r3p);
    allocator.free(s.r1p);
    allocator.free(s.r1n);
    allocator.free(s.r3n);
    allocator.free(s.t_base);
    allocator.free(s.t_costs);
    allocator.free(s.pbackt);
    allocator.free(s.fpath);
    allocator.free(s.dmap);
    allocator.free(s.tline);
    allocator.free(s.bmask);
    allocator.free(s.block_active);
    allocator.free(s.hp3p);
    allocator.free(s.hp1p);
    allocator.free(s.hp1n);
    allocator.free(s.hp3n);
    allocator.free(s.srcT);
    allocator.free(s.dstT);
    allocator.free(s.maskT);
    allocator.free(s.scpT);
    allocator.destroy(s);
}
