const std = @import("std");
const math = std.math;
const hz = @import("../helper.zig");
const simd = @import("simd.zig");

const vec_len = std.simd.suggestVectorLength(i16) orelse 8;
const VI16 = @Vector(vec_len, i16);
const VI32 = @Vector(vec_len, i32);
const VU8 = @Vector(vec_len, u8);

/// Widening tap load; LLVM folds the load+zext into one vpmovzx.
inline fn ld(p: [*]const u8) VI16 {
    const v: VU8 = p[0..vec_len].*;
    return @intCast(v);
}

/// Laundered widening load for the x-2/x/x+2 same-row taps, whose relative
/// offsets are comptime-known (the overlapping-load-fusion shape; see
/// simd.loaduOpaque). Portable twin inside loaduOpaque.
inline fn ldo(p: [*]const u8) VI16 {
    const v: VU8 = simd.loaduOpaque(u8, vec_len, p[0..vec_len]);
    return @intCast(v);
}

/// Original scalar pixel; kept verbatim as the semantic reference. Handles
/// the clamped x_left/x_right border columns the vector path peels off.
inline fn processPixel(
    dstp: []u8,
    srcp_p2: anytype,
    srcp_p1: []const u8,
    srcp: []const u8,
    srcp_n1: []const u8,
    srcp_n2: anytype,
    stride2: u32,
    width: u32,
    x: u32,
    thr: i32,
    tmax: i32,
    tmax_multiplier: i32,
    tthr2: i32,
    comptime use_tthr2: bool,
) void {
    if (use_tthr2 and
        (@abs(@as(i32, srcp_p1[x]) - srcp_n1[x]) < tthr2 and
            @abs(@as(i32, srcp_p2[x]) - srcp[x]) < tthr2 and
            @abs(@as(i32, srcp[x]) - srcp_n2[x]) < tthr2))
    {
        dstp[x] = @intCast((@as(u16, srcp_p1[x]) + @as(u16, srcp[x]) * 2 + @as(u16, srcp_n1[x])) >> 2);
    } else {
        const next_value: i32 = @as(i32, srcp[x]) + @as(i32, srcp_n1[x]);
        const prev_value: i32 = @as(i32, srcp[x]) + @as(i32, srcp_p1[x]);

        const x_left: u32 = if (x < 2) 0 else x - 2;
        const x_right: u32 = if (x > width - 3) width - 1 else x + 2;

        const current_column: i32 = @as(i32, hz.getVal2(u8, srcp.ptr, x, stride2)) + @as(i32, srcp[x]) * 2 + @as(i32, srcp[stride2 + x]);
        const cvl: i32 = hz.getVal2(u8, srcp.ptr, x_left, stride2);
        const cvr: i32 = hz.getVal2(u8, srcp.ptr, x_right, stride2);
        const curr_value: i32 = (-cvl - cvr + @as(i32, srcp[x_left]) * 2 + @as(i32, srcp[x_right]) * 2 - @as(i32, srcp[x_left + stride2]) -
            @as(i32, srcp[x_right + stride2]) + current_column * 2 + @as(i32, srcp[x]) * 12);

        var nc: i32 = hz.getVal2(u8, srcp_n1.ptr, x, stride2);
        var pc: i32 = hz.getVal2(u8, srcp_p1.ptr, x, stride2);
        nc = (nc + @as(i32, srcp_n1[x]) * 2 + @as(i32, srcp_n1[x + stride2])) - current_column;
        pc = (pc + @as(i32, srcp_p1[x]) * 2 + @as(i32, srcp_p1[x + stride2])) - current_column;
        nc = thr + tmax - @as(i32, @intCast(@abs(nc)));
        pc = thr + tmax - @as(i32, @intCast(@abs(pc)));

        const next_weight: i32 = @min(math.clamp(nc, 0, tmax + 1) * tmax_multiplier, 8192);
        const prev_weight: i32 = @min(math.clamp(pc, 0, tmax + 1) * tmax_multiplier, 8192);
        const curr_weight: i32 = (1 << 14) - (next_weight + prev_weight);
        const out: i32 = (curr_weight * @divTrunc(curr_value, 10) + prev_weight * prev_value + next_weight * next_value) >> 15;
        dstp[x] = math.lossyCast(u8, out);
    }
}

/// Interior pixels [x, x+vec_len), all taps unclamped. Bit-exact vs
/// processPixel by range proof: taps <= 255 so current_column <= 1020,
/// curr_value in [-1020, 6120], nc/pc in [-1019, 510] (thr+tmax <= 510),
/// clamp(..)*tmax_multiplier <= (tmax+1)*divTrunc(8192,tmax) <= 16384 and
/// weights <= 16384 -- every i16 intermediate is exact (integer math reorders
/// freely); @divTrunc by 10 is width-independent; the final 3-term dot
/// product is < 2^25 so i32 is exact; >>15 then [0,255] clamp mirror the
/// scalar shift and math.lossyCast.
inline fn processVec(
    dstp: []u8,
    srcp_p2: anytype,
    srcp_p1: []const u8,
    srcp: []const u8,
    srcp_n1: []const u8,
    srcp_n2: anytype,
    stride2: u32,
    x: u32,
    thr_tmax: VI16,
    tmax_p1: VI16,
    tmax_mult: VI16,
    tthr2v: VI16,
    comptime use_tthr2: bool,
) void {
    const s0 = srcp.ptr + x;
    const st = s0 - stride2;
    const sb = s0 + stride2;
    const p1p = srcp_p1.ptr + x;
    const n1p = srcp_n1.ptr + x;

    const c = ldo(s0);
    const cl = ldo(s0 - 2);
    const cr = ldo(s0 + 2);
    const ct = ldo(st);
    const cvl = ldo(st - 2);
    const cvr = ldo(st + 2);
    const cb = ldo(sb);
    const lb = ldo(sb - 2);
    const rb = ldo(sb + 2);

    const p1 = ld(p1p);
    const p1t = ld(p1p - stride2);
    const p1b = ld(p1p + stride2);
    const n1 = ld(n1p);
    const n1t = ld(n1p - stride2);
    const n1b = ld(n1p + stride2);

    const cc = ct + c + c + cb;
    const curr_value = (cl + cl + cr + cr - cvl - cvr - lb - rb) + (cc + cc + c * @as(VI16, @splat(12)));

    var nc = (n1t + n1 + n1 + n1b) - cc;
    var pc = (p1t + p1 + p1 + p1b) - cc;
    nc = thr_tmax - @as(VI16, @intCast(@abs(nc)));
    pc = thr_tmax - @as(VI16, @intCast(@abs(pc)));

    const zero: VI16 = @splat(0);
    const cap: VI16 = @splat(8192);
    const nw = @min(@min(@max(nc, zero), tmax_p1) * tmax_mult, cap);
    const pw = @min(@min(@max(pc, zero), tmax_p1) * tmax_mult, cap);
    const cw = @as(VI16, @splat(1 << 14)) - (nw + pw);

    const cv10 = @divTrunc(curr_value, @as(VI16, @splat(10)));
    const nv = c + n1;
    const pv = c + p1;

    const acc = @as(VI32, @intCast(cw)) * @as(VI32, @intCast(cv10)) +
        @as(VI32, @intCast(pw)) * @as(VI32, @intCast(pv)) +
        @as(VI32, @intCast(nw)) * @as(VI32, @intCast(nv));
    const out = acc >> @as(@Vector(vec_len, u5), @splat(15));
    const res: VI16 = @intCast(@min(@max(out, @as(VI32, @splat(0))), @as(VI32, @splat(255))));

    var outv = res;
    if (use_tthr2) {
        const p2 = ld(srcp_p2.ptr + x);
        const n2 = ld(srcp_n2.ptr + x);
        const d1: VI16 = @intCast(@abs(p1 - n1));
        const d2: VI16 = @intCast(@abs(p2 - c));
        const d3: VI16 = @intCast(@abs(c - n2));
        const m = (d1 < tthr2v) & (d2 < tthr2v) & (d3 < tthr2v);
        // (p1 + 2c + n1) >> 2 on non-negative i16 <= 1020: identical to the
        // scalar u16 shift (no vpavg -- that would round).
        const fast = (p1 + c + c + n1) >> @as(@Vector(vec_len, u4), @splat(2));
        outv = @select(i16, m, fast, res);
    }
    dstp[x..][0..vec_len].* = @as(VU8, @intCast(outv));
}

pub fn process(
    dstp: []u8,
    srcp_p2: anytype,
    srcp_p1: []const u8,
    srcp: []const u8,
    srcp_n1: []const u8,
    srcp_n2: anytype,
    stride: u32,
    width: u32,
    thr: i32,
    tmax: i32,
    tmax_multiplier: i32,
    tthr2: i32,
    comptime use_tthr2: bool,
) void {
    const stride2: u32 = stride << 1;

    if (width < vec_len + 4) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            processPixel(dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, stride2, width, x, thr, tmax, tmax_multiplier, tthr2, use_tthr2);
        }
        return;
    }

    // Border columns keep the scalar x_left/x_right clamps.
    for ([4]u32{ 0, 1, width - 2, width - 1 }) |bx| {
        processPixel(dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, stride2, width, bx, thr, tmax, tmax_multiplier, tthr2, use_tthr2);
    }

    const thr_tmax: VI16 = @splat(@intCast(thr + tmax));
    const tmax_p1: VI16 = @splat(@intCast(tmax + 1));
    const tmax_mult: VI16 = @splat(@intCast(tmax_multiplier));
    // |tap diff| <= 255, so capping tthr2 at 256 preserves every comparison
    // while keeping the broadcast within i16.
    const tthr2v: VI16 = @splat(@intCast(@min(tthr2, 256)));

    var x: u32 = 2;
    while (x + vec_len <= width - 2) : (x += vec_len) {
        processVec(dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, stride2, x, thr_tmax, tmax_p1, tmax_mult, tthr2v, use_tthr2);
    }
    if (x < width - 2) {
        // Overlapped final block: rewrites some interior pixels with identical
        // values (dst is never read here), avoiding a scalar tail.
        processVec(dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, stride2, width - 2 - vec_len, thr_tmax, tmax_p1, tmax_mult, tthr2v, use_tthr2);
    }
}
