const std = @import("std");
const assert = std.debug.assert;

const VEC = 8;
const Vf = @Vector(VEC, f32);

pub const Match = struct {
    err: f32,
    x: i32,
    y: i32,
    t: i32,
};

pub const Loc = struct {
    x: i32,
    y: i32,
};

fn matchLessThan(_: void, a: Match, b: Match) bool {
    return a.err < b.err;
}

pub fn selectTop(items: []Match, k: usize) usize {
    if (items.len <= k) {
        std.sort.insertion(Match, items, {}, matchLessThan);
        return items.len;
    }

    std.sort.insertion(Match, items[0..k], {}, matchLessThan);
    for (items[k..]) |e| {
        if (e.err < items[k - 1].err) {
            var pos = k - 1;
            while (pos > 0 and items[pos - 1].err > e.err) : (pos -= 1) {}
            var j = k - 1;
            while (j > pos) : (j -= 1) items[j] = items[j - 1];
            items[pos] = e;
        }
    }
    return k;
}

fn distanceScalar(cur: []const f32, refp: []const f32, stride: u32, x: u32, y: u32, bs: u32) f32 {
    var acc: f32 = 0.0;
    var p: u32 = 0;
    var py: u32 = 0;
    while (py < bs) : (py += 1) {
        const row = refp[(y + py) * stride + x ..];
        var px: u32 = 0;
        while (px < bs) : (px += 1) {
            const d = cur[p] - row[px];
            acc += d * d;
            p += 1;
        }
    }
    return acc;
}

pub fn computeBlockDistancesWindow(
    comptime BS: ?u32,
    errs: *std.ArrayList(f32),
    alloc: std.mem.Allocator,
    cur: []const f32,
    refp: []const f32,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    stride: u32,
    bs_rt: u32,
) void {
    const bs = BS orelse bs_rt;
    const cx = right - left + 1;
    const rows = bottom - top + 1;
    errs.ensureUnusedCapacity(alloc, rows * cx) catch unreachable;
    const base = errs.items.len;
    errs.items.len = base + rows * cx;
    const out = errs.items[base..][0 .. rows * cx];

    if (cx < VEC) {
        var bm_y = top;
        while (bm_y <= bottom) : (bm_y += 1) {
            const row = out[(bm_y - top) * cx ..][0..cx];
            var bm_x = left;
            while (bm_x <= right) : (bm_x += 1) {
                row[bm_x - left] = distanceScalar(cur, refp, stride, bm_x, bm_y, bs);
            }
        }
        return;
    }

    var pend_src: [4]u32 = undefined;
    var pend_dst: [4]u32 = undefined;
    var pend: usize = 0;

    var bm_y = top;
    while (bm_y <= bottom) : (bm_y += 1) {
        var bm_x = left;
        while (true) {
            const anchored = @min(bm_x, right + 1 - VEC);
            pend_src[pend] = bm_y * stride + anchored;
            pend_dst[pend] = (bm_y - top) * cx + (anchored - left);
            pend += 1;
            if (pend == 4) {
                const r4 = distanceVec4(BS, cur, refp, pend_src, stride, bs);
                inline for (0..4) |a| out[pend_dst[a]..][0..VEC].* = r4[a];
                pend = 0;
            }
            if (anchored + VEC > right) break;
            bm_x = anchored + VEC;
        }
    }
    for (0..pend) |a| {
        out[pend_dst[a]..][0..VEC].* = distanceVecAt(BS, cur, refp, pend_src[a], stride, bs);
    }
}

inline fn distanceVec4(comptime BS: ?u32, cur: []const f32, refp: []const f32, srcs: [4]u32, stride: u32, bs_rt: u32) [4][VEC]f32 {
    const bs = BS orelse bs_rt;
    var acc: [4]Vf = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var p: u32 = 0;
    var py: u32 = 0;
    while (py < bs) : (py += 1) {
        var rows: [4][]const f32 = undefined;
        inline for (0..4) |a| rows[a] = refp[srcs[a] + py * stride ..];
        var px: u32 = 0;
        while (px < bs) : (px += 1) {
            const cv: Vf = @splat(cur[p]);
            inline for (0..4) |a| {
                const d = cv - @as(Vf, rows[a][px..][0..VEC].*);
                acc[a] += d * d;
            }
            p += 1;
        }
    }
    return .{ acc[0], acc[1], acc[2], acc[3] };
}

inline fn distanceVecAt(comptime BS: ?u32, cur: []const f32, refp: []const f32, src: u32, stride: u32, bs_rt: u32) [VEC]f32 {
    const bs = BS orelse bs_rt;
    var acc: Vf = @splat(0.0);
    var p: u32 = 0;
    var py: u32 = 0;
    while (py < bs) : (py += 1) {
        const row = refp[src + py * stride ..];
        var px: u32 = 0;
        while (px < bs) : (px += 1) {
            const d = @as(Vf, @splat(cur[p])) - @as(Vf, row[px..][0..VEC].*);
            acc += d * d;
            p += 1;
        }
    }
    return acc;
}

pub fn computeBlockDistancesAt(
    comptime BS: ?u32,
    errs: *std.ArrayList(f32),
    alloc: std.mem.Allocator,
    cur: []const f32,
    refp: []const f32,
    locations: []const Loc,
    stride: u32,
    bs_rt: u32,
) void {
    const bs = BS orelse bs_rt;
    errs.ensureUnusedCapacity(alloc, locations.len) catch unreachable;
    const base = errs.items.len;
    errs.items.len = base + locations.len;
    const out = errs.items[base..][0..locations.len];

    var pend_src: [4]u32 = undefined;
    var pend_dst: [4]u32 = undefined;
    var pend: usize = 0;

    var i: usize = 0;
    while (i < locations.len) {
        var e = i + 1;
        while (e < locations.len and locations[e].y == locations[i].y and locations[e].x == locations[e - 1].x + 1) : (e += 1) {}
        const run = e - i;
        if (run >= VEC) {
            var off: usize = 0;
            while (true) {
                const a = @min(off, run - VEC);
                const loc = locations[i + a];
                pend_src[pend] = @as(u32, @intCast(loc.y)) * stride + @as(u32, @intCast(loc.x));
                pend_dst[pend] = @intCast(i + a);
                pend += 1;
                if (pend == 4) {
                    const r4 = distanceVec4(BS, cur, refp, pend_src, stride, bs);
                    inline for (0..4) |k| out[pend_dst[k]..][0..VEC].* = r4[k];
                    pend = 0;
                }
                if (a + VEC >= run) break;
                off = a + VEC;
            }
        } else {
            for (i..e) |j| {
                out[j] = distanceScalar(cur, refp, stride, @intCast(locations[j].x), @intCast(locations[j].y), bs);
            }
        }
        i = e;
    }
    for (0..pend) |k| {
        out[pend_dst[k]..][0..VEC].* = distanceVecAt(BS, cur, refp, pend_src[k], stride, bs);
    }
}

pub const IdxErr = struct { err: f32, idx: u32 };

pub fn selectTopIdx(errs: []const f32, k: usize, out: []IdxErr) usize {
    const kk = @min(k, errs.len);
    if (kk == 0) return 0;

    var count: usize = 0;
    while (count < kk) : (count += 1) {
        insertIdx(out, count, errs[count], @intCast(count));
    }

    var i: usize = kk;
    var thr = out[kk - 1].err;
    while (i + VEC <= errs.len) : (i += VEC) {
        const v: Vf = errs[i..][0..VEC].*;
        if (!@reduce(.Or, v < @as(Vf, @splat(thr)))) continue;
        inline for (0..VEC) |l| {
            if (errs[i + l] < out[kk - 1].err) {
                insertIdx(out, kk - 1, errs[i + l], @intCast(i + l));
            }
        }
        thr = out[kk - 1].err;
    }
    while (i < errs.len) : (i += 1) {
        if (errs[i] < out[kk - 1].err) {
            insertIdx(out, kk - 1, errs[i], @intCast(i));
        }
    }
    return kk;
}

inline fn insertIdx(out: []IdxErr, len: usize, e: f32, idx: u32) void {
    var pos = len;
    while (pos > 0 and out[pos - 1].err > e) : (pos -= 1) {}
    var j = len;
    while (j > pos) : (j -= 1) out[j] = out[j - 1];
    out[pos] = .{ .err = e, .idx = idx };
}

pub fn generateSearchLocations(
    centers: []const Match,
    num_centers: usize,
    bs: u32,
    width: u32,
    height: u32,
    ps_range: u32,
    out: *std.ArrayList(Loc),
    scratch: *std.ArrayList(Loc),
    alloc: std.mem.Allocator,
) void {
    out.clearRetainingCapacity();

    for (centers[0..num_centers]) |c| {
        const left: i32 = @max(c.x - @as(i32, @intCast(ps_range)), 0);
        const right: i32 = @min(c.x + @as(i32, @intCast(ps_range)), @as(i32, @intCast(width - bs)));
        const top: i32 = @max(c.y - @as(i32, @intCast(ps_range)), 0);
        const bottom: i32 = @min(c.y + @as(i32, @intCast(ps_range)), @as(i32, @intCast(height - bs)));

        scratch.clearRetainingCapacity();
        scratch.ensureUnusedCapacity(alloc, out.items.len + @as(usize, @intCast(bottom - top + 1)) * @as(usize, @intCast(right - left + 1))) catch unreachable;

        var i: usize = 0;
        var yy = top;
        var xx = left;
        while (true) {
            const have_old = i < out.items.len;
            const have_new = yy <= bottom;
            if (!have_old and !have_new) break;

            var take_old = false;
            var take_new = false;
            if (have_old and have_new) {
                const o = out.items[i];
                if (o.y < yy or (o.y == yy and o.x < xx)) {
                    take_old = true;
                } else if (o.y == yy and o.x == xx) {
                    take_old = true;
                    take_new = true;
                } else {
                    take_new = true;
                }
            } else if (have_old) {
                take_old = true;
            } else {
                take_new = true;
            }

            if (take_old) {
                scratch.appendAssumeCapacity(out.items[i]);
                i += 1;
            }
            if (take_new) {
                if (!take_old) scratch.appendAssumeCapacity(.{ .x = xx, .y = yy });
                xx += 1;
                if (xx > right) {
                    xx = left;
                    yy += 1;
                }
            }
        }

        std.mem.swap(std.ArrayList(Loc), out, scratch);
    }
}

pub fn loadPatches(
    comptime BS: ?u32,
    a: [*]f32,
    lda: u32,
    mean_patch: ?[]f32,
    srcps: []const []const f32,
    errors: []const Match,
    stride: u32,
    bs_rt: u32,
) void {
    const bs = BS orelse bs_rt;
    for (errors, 0..) |e, i| {
        const src = srcps[@intCast(e.t)];
        var col = a + i * lda;
        var py: u32 = 0;
        while (py < bs) : (py += 1) {
            const row = src[(@as(u32, @intCast(e.y)) + py) * stride + @as(u32, @intCast(e.x)) ..][0..bs];
            @memcpy(col[0..bs], row);
            col += bs;
        }
    }
    if (mean_patch) |mp| {
        const m = bs * bs;
        @memset(mp[0..m], 0.0);
        for (0..errors.len) |i| {
            const col = a + i * lda;
            for (0..m) |p| mp[p] += col[p];
        }
    }
}

pub fn subtractMean(a: [*]f32, lda: u32, mean_patch: []f32, m: u32, n: u32) void {
    const inv: f32 = @floatFromInt(n);
    for (0..m) |p| mean_patch[p] /= inv;
    for (0..n) |i| {
        const col = a + i * lda;
        for (0..m) |p| col[p] -= mean_patch[p];
    }
}

pub fn addMean(a: [*]f32, lda: u32, mean_patch: []const f32, m: u32, n: u32) void {
    for (0..n) |i| {
        const col = a + i * lda;
        for (0..m) |p| col[p] += mean_patch[p];
    }
}

pub fn col2im(
    comptime BS: ?u32,
    agg: [*]f32,
    agg_stride: u32,
    a: [*]const f32,
    lda: u32,
    errors: []const Match,
    height: u32,
    bs_rt: u32,
    weight: f32,
) void {
    const bs = BS orelse bs_rt;
    const wv: Vf = @splat(weight);
    for (errors, 0..) |e, i| {
        var patch = a + i * lda;
        const x: u32 = @intCast(e.x);
        const y: u32 = @intCast(e.y);
        const t: u32 = @intCast(e.t);
        var wdst = agg + (@as(usize, t * 2) * height + y) * agg_stride + x;
        var wgt = agg + (@as(usize, t * 2 + 1) * height + y) * agg_stride + x;
        var py: u32 = 0;
        while (py < bs) : (py += 1) {
            var px: u32 = 0;
            while (px + VEC <= bs) : (px += VEC) {
                const pv: Vf = patch[px..][0..VEC].*;
                const dv: Vf = wdst[px..][0..VEC].*;
                const gv: Vf = wgt[px..][0..VEC].*;
                wdst[px..][0..VEC].* = dv + pv * wv;
                wgt[px..][0..VEC].* = gv + wv;
            }
            while (px < bs) : (px += 1) {
                wdst[px] += patch[px] * weight;
                wgt[px] += weight;
            }
            wdst += agg_stride;
            wgt += agg_stride;
            patch += bs;
        }
    }
}

pub fn aggregation(
    dstp: []f32,
    intermediate: []const f32,
    width: u32,
    height: u32,
    stride: u32,
) void {
    const wdst = intermediate;
    const weight = intermediate[height * width ..];
    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const drow = dstp[y * stride ..];
        const nrow = wdst[y * width ..];
        const wrow = weight[y * width ..];
        var x: u32 = 0;
        while (x + VEC <= width) : (x += VEC) {
            const nv: Vf = nrow[x..][0..VEC].*;
            const wv: Vf = wrow[x..][0..VEC].*;
            drow[x..][0..VEC].* = nv / wv;
        }
        while (x < width) : (x += 1) {
            drow[x] = nrow[x] / wrow[x];
        }
    }
}

pub const max_q = 64;
pub const max_sweeps_default = 30;

pub fn defaultTol(comptime F: type) F {
    return if (F == f64) 1e-6 else std.math.floatEps(F);
}

pub const Result = struct { k: u32, sweeps: u32 };

pub fn Workspace(comptime F: type) type {
    return struct {
        g: []F,
        v: []F,
        eig: []F,
        r: []f32,
        scratch: []f32,

        wide: []F,
    };
}

pub fn patchEstimate(
    comptime F: type,
    a: []const f32,
    dst: []f32,
    m: usize,
    n: usize,
    lda: usize,
    c: f32,
    k0: u32,
    ws: Workspace(F),
) Result {
    assert(m >= 1 and n >= 1);
    assert(lda >= m);
    assert(k0 <= 1);
    assert(c >= 0);
    const q = @min(m, n);
    assert(q <= max_q);
    assert(ws.g.len >= q * q and ws.v.len >= q * q);
    assert(ws.eig.len >= q and ws.r.len >= q);
    assert(ws.scratch.len >= m * n + q * q);
    assert(a.len >= lda * (n - 1) + m and dst.len >= lda * (n - 1) + m);

    const g = ws.g[0 .. q * q];
    const v = ws.v[0 .. q * q];
    const eig = ws.eig[0..q];

    if (n <= m) gramAtA(F, a, m, n, lda, g, ws.wide) else gramAAt(F, a, m, n, lda, g);
    const sweeps = jacobiEigSym(F, q, g, v, max_sweeps_default, defaultTol(F));
    extractSortedEig(F, q, g, v, eig);
    const k = shrinkFactors(F, q, eig, c, k0, ws.r);
    if (n <= m)
        reconTall(F, a, dst, m, n, lda, k, ws.r, v, ws.scratch)
    else
        reconWide(F, a, dst, m, n, lda, k, ws.r, v, ws.scratch);
    return .{ .k = k, .sweeps = sweeps };
}

inline fn widenVec(comptime F: type, x: Vf) @Vector(VEC, F) {
    if (F == f32) return x;
    return @floatCast(x);
}

fn gramAtA(comptime F: type, a: []const f32, m: usize, n: usize, lda: usize, g: []F, wide: []F) void {
    const VF = @Vector(VEC, F);
    assert(wide.len >= m * n);
    for (0..n) |i| {
        const ci = a[i * lda ..][0..m];
        const wi = wide[i * m ..][0..m];
        var p: usize = 0;
        while (p + VEC <= m) : (p += VEC) {
            const x: Vf = ci[p..][0..VEC].*;
            wi[p..][0..VEC].* = widenVec(F, x);
        }
        while (p < m) : (p += 1) wi[p] = ci[p];
    }
    for (0..n) |i| {
        const ci = wide[i * m ..][0..m];
        for (i..n) |j| {
            const cj = wide[j * m ..][0..m];
            var acc: VF = @splat(0);
            var p: usize = 0;
            while (p + VEC <= m) : (p += VEC) {
                const x: VF = ci[p..][0..VEC].*;
                const y: VF = cj[p..][0..VEC].*;
                acc = @mulAdd(VF, x, y, acc);
            }
            var s: F = @reduce(.Add, acc);
            while (p < m) : (p += 1) s = @mulAdd(F, ci[p], cj[p], s);
            g[i * n + j] = s;
            g[j * n + i] = s;
        }
    }
}

fn gramAAt(comptime F: type, a: []const f32, m: usize, n: usize, lda: usize, g: []F) void {
    assert(m <= max_q);
    @memset(g[0 .. m * m], 0);
    var col: [max_q]F = undefined;
    for (0..n) |kk| {
        const ck = a[kk * lda ..][0..m];
        for (0..m) |p| col[p] = ck[p];
        for (0..m) |i| {
            const ai = col[i];
            for (i..m) |j| g[i * m + j] = @mulAdd(F, ai, col[j], g[i * m + j]);
        }
    }
    for (1..m) |i| {
        for (0..i) |j| g[i * m + j] = g[j * m + i];
    }
}

inline fn rotateRows(comptime F: type, x: []F, y: []F, cs: F, sn: F) void {
    const VF = @Vector(VEC, F);
    const cv: VF = @splat(cs);
    const sv: VF = @splat(sn);
    const q = x.len;
    var i: usize = 0;
    while (i + VEC <= q) : (i += VEC) {
        const xp: VF = x[i..][0..VEC].*;
        const yp: VF = y[i..][0..VEC].*;
        x[i..][0..VEC].* = cv * xp - sv * yp;
        y[i..][0..VEC].* = sv * xp + cv * yp;
    }
    while (i < q) : (i += 1) {
        const xp = x[i];
        const yp = y[i];
        x[i] = cs * xp - sn * yp;
        y[i] = sn * xp + cs * yp;
    }
}

fn jacobiEigSym(comptime F: type, q: usize, g: []F, vt: []F, max_sweeps: u32, tol: F) u32 {
    @memset(vt[0 .. q * q], 0);
    for (0..q) |i| vt[i * q + i] = 1;
    if (q == 1) return 0;

    const theta_big = @sqrt(std.math.floatMax(F)) * 0.25;

    var sweep: u32 = 0;
    while (sweep < max_sweeps) : (sweep += 1) {
        var gmax: F = 0;
        for (0..q) |i| gmax = @max(gmax, g[i * q + i]);
        const floor_abs = std.math.floatEps(F) * gmax;

        var rotated = false;
        for (0..q - 1) |p| {
            const gp = g[p * q ..][0..q];
            const vp = vt[p * q ..][0..q];
            for (p + 1..q) |j| {
                const apq = gp[j];
                const app = gp[p];
                const ajj = g[j * q + j];
                const thresh = @max(tol * @sqrt(@max(app, 0) * @max(ajj, 0)), floor_abs);
                if (@abs(apq) <= thresh) continue;
                rotated = true;

                const theta = (ajj - app) / (2 * apq);
                const t = if (@abs(theta) > theta_big)
                    1 / (2 * theta)
                else blk: {
                    const at = @abs(theta) + @sqrt(theta * theta + 1);
                    break :blk if (theta >= 0) 1 / at else -1 / at;
                };
                const cs = 1 / @sqrt(t * t + 1);
                const sn = t * cs;
                const h = t * apq;

                const gj = g[j * q ..][0..q];

                rotateRows(F, gp, gj, cs, sn);
                gp[p] = app - h;
                gj[j] = ajj + h;
                gp[j] = 0;
                gj[p] = 0;
                var i: usize = 0;
                while (i < q) : (i += 1) {
                    g[i * q + p] = gp[i];
                    g[i * q + j] = gj[i];
                }

                rotateRows(F, vp, vt[j * q ..][0..q], cs, sn);
            }
        }
        if (!rotated) return sweep;
    }
    return max_sweeps;
}

pub const batch_n: usize = @max(4, std.simd.suggestVectorLength(f64) orelse 4);
const Vb = @Vector(batch_n, f64);

fn jacobiEigSymBatch(q: usize, g: []f64, vt: []f64, max_sweeps: u32, tol: f64) u32 {
    const L = batch_n;
    @memset(vt[0 .. q * q * L], 0);
    for (0..q) |i| {
        for (0..L) |l| vt[(i * q + i) * L + l] = 1;
    }
    if (q == 1) return 0;

    const theta_big = @sqrt(std.math.floatMax(f64)) * 0.25;
    const zero: Vb = @splat(0.0);
    const one: Vb = @splat(1.0);

    var sweep: u32 = 0;
    while (sweep < max_sweeps) : (sweep += 1) {
        var gmax: Vb = zero;
        for (0..q) |i| gmax = @max(gmax, @as(Vb, g[(i * q + i) * L ..][0..L].*));
        const floor_abs = @as(Vb, @splat(std.math.floatEps(f64))) * gmax;

        var rotated = false;
        for (0..q - 1) |p| {
            const gp = g[p * q * L ..][0 .. q * L];
            const vp = vt[p * q * L ..][0 .. q * L];
            for (p + 1..q) |j| {
                const apq: Vb = gp[j * L ..][0..L].*;
                const app: Vb = gp[p * L ..][0..L].*;
                const ajj: Vb = g[(j * q + j) * L ..][0..L].*;
                const thresh = @max(@as(Vb, @splat(tol)) * @sqrt(@max(app, zero) * @max(ajj, zero)), floor_abs);
                const active = @abs(apq) > thresh;
                if (!@reduce(.Or, active)) continue;
                rotated = true;

                const theta = (ajj - app) / (@as(Vb, @splat(2.0)) * apq);
                const at = @abs(theta) + @sqrt(theta * theta + one);
                const t_signed = @select(f64, theta >= zero, one / at, -(one / at));
                const t_guarded = @select(f64, @abs(theta) > @as(Vb, @splat(theta_big)), one / (@as(Vb, @splat(2.0)) * theta), t_signed);
                const t = @select(f64, active, t_guarded, zero);
                const cs = one / @sqrt(t * t + one);
                const sn = t * cs;
                const h = t * apq;

                const gj = g[j * q * L ..][0 .. q * L];
                rotateRowsBatch(gp, gj, cs, sn);
                gp[p * L ..][0..L].* = app - h;
                gj[j * L ..][0..L].* = ajj + h;
                gp[j * L ..][0..L].* = @select(f64, active, zero, apq);
                gj[p * L ..][0..L].* = @select(f64, active, zero, apq);
                var i: usize = 0;
                while (i < q) : (i += 1) {
                    g[(i * q + p) * L ..][0..L].* = @as(Vb, gp[i * L ..][0..L].*);
                    g[(i * q + j) * L ..][0..L].* = @as(Vb, gj[i * L ..][0..L].*);
                }
                rotateRowsBatch(vp, vt[j * q * L ..][0 .. q * L], cs, sn);
            }
        }
        if (!rotated) return sweep;
    }
    return max_sweeps;
}

inline fn rotateRowsBatch(x: []f64, y: []f64, cs: Vb, sn: Vb) void {
    const L = batch_n;
    const q = x.len / L;
    var i: usize = 0;
    while (i < q) : (i += 1) {
        const xp: Vb = x[i * L ..][0..L].*;
        const yp: Vb = y[i * L ..][0..L].*;
        x[i * L ..][0..L].* = cs * xp - sn * yp;
        y[i * L ..][0..L].* = sn * xp + cs * yp;
    }
}

pub fn patchEstimateBatch(
    as: [batch_n][]const f32,
    dsts: [batch_n][]f32,
    count: usize,
    m: usize,
    n: usize,
    lda: usize,
    c: f32,
    k0: u32,
    ws: Workspace(f64),
    gb: []f64,
    vb: []f64,
) [batch_n]u32 {
    const L = batch_n;
    assert(count >= 1 and count <= L);
    const q = @min(m, n);
    assert(q <= max_q);
    assert(gb.len >= q * q * L and vb.len >= q * q * L);

    for (0..count) |l| {
        const g = ws.g[0 .. q * q];
        if (n <= m) gramAtA(f64, as[l], m, n, lda, g, ws.wide) else gramAAt(f64, as[l], m, n, lda, g);
        for (0..q * q) |e| gb[e * L + l] = g[e];
    }
    for (count..L) |l| {
        for (0..q * q) |e| gb[e * L + l] = gb[e * L];
    }

    _ = jacobiEigSymBatch(q, gb, vb, max_sweeps_default, defaultTol(f64));

    var ks: [batch_n]u32 = undefined;
    for (0..count) |l| {
        const g = ws.g[0 .. q * q];
        const v = ws.v[0 .. q * q];
        for (0..q) |i| {
            g[i * q + i] = gb[(i * q + i) * L + l];
            for (0..q) |j| v[i * q + j] = vb[(i * q + j) * L + l];
        }
        const eig = ws.eig[0..q];
        extractSortedEig(f64, q, g, v, eig);
        const k = shrinkFactors(f64, q, eig, c, k0, ws.r);
        if (n <= m)
            reconTall(f64, as[l], dsts[l], m, n, lda, k, ws.r, v, ws.scratch)
        else
            reconWide(f64, as[l], dsts[l], m, n, lda, k, ws.r, v, ws.scratch);
        ks[l] = k;
    }
    return ks;
}

fn extractSortedEig(comptime F: type, q: usize, g: []const F, vt: []F, eig: []F) void {
    for (0..q) |i| eig[i] = @max(g[i * q + i], 0);
    for (0..q) |i| {
        var best = i;
        for (i + 1..q) |j| {
            if (eig[j] > eig[best]) best = j;
        }
        if (best != i) {
            std.mem.swap(F, &eig[i], &eig[best]);
            for (0..q) |rr| std.mem.swap(F, &vt[i * q + rr], &vt[best * q + rr]);
        }
    }
}

fn shrinkFactors(comptime F: type, q: usize, eig: []const F, c: f32, k0: u32, r: []f32) u32 {
    assert(c >= 0);
    const cf: F = c;
    var i: usize = @min(@as(usize, k0), q);
    for (0..i) |ii| r[ii] = 1;
    while (i < q) : (i += 1) {
        const lam = eig[i];
        const tmp = lam - cf;
        if (!(tmp > 0)) break;
        const s = @sqrt(lam);
        const sp = (s + @sqrt(tmp)) * 0.5;
        r[i] = @floatCast(sp / s);
    }
    const k: u32 = @intCast(i);
    for (i..q) |ii| r[ii] = 0;
    return k;
}

fn reconTall(
    comptime F: type,
    a: []const f32,
    dst: []f32,
    m: usize,
    n: usize,
    lda: usize,
    k: u32,
    r: []const f32,
    vt: []const F,
    scratch: []f32,
) void {
    const q = n;
    const ku: usize = k;
    if (ku == 0) {
        for (0..n) |j| @memset(dst[j * lda ..][0..m], 0);
        return;
    }
    const b = scratch;

    for (0..ku) |i| {
        var vc: [max_q]f32 = undefined;
        for (0..n) |kk| vc[kk] = @floatCast(vt[i * q + kk]);
        const bi = b[i * m ..][0..m];
        var p: usize = 0;
        while (p + VEC <= m) : (p += VEC) {
            var acc: Vf = @splat(0);
            for (0..n) |kk| {
                const av: Vf = a[kk * lda + p ..][0..VEC].*;
                acc = @mulAdd(Vf, av, @splat(vc[kk]), acc);
            }
            bi[p..][0..VEC].* = acc;
        }
        while (p < m) : (p += 1) {
            var s: f32 = 0;
            for (0..n) |kk| s = @mulAdd(f32, a[kk * lda + p], vc[kk], s);
            bi[p] = s;
        }
    }

    for (0..n) |j| {
        var w: [max_q]f32 = undefined;
        for (0..ku) |i| w[i] = r[i] * @as(f32, @floatCast(vt[i * q + j]));
        const dj = dst[j * lda ..][0..m];
        var p: usize = 0;
        while (p + VEC <= m) : (p += VEC) {
            var acc: Vf = @splat(0);
            for (0..ku) |i| {
                const bv: Vf = b[i * m + p ..][0..VEC].*;
                acc = @mulAdd(Vf, bv, @splat(w[i]), acc);
            }
            dj[p..][0..VEC].* = acc;
        }
        while (p < m) : (p += 1) {
            var s: f32 = 0;
            for (0..ku) |i| s = @mulAdd(f32, b[i * m + p], w[i], s);
            dj[p] = s;
        }
    }
}

fn reconWide(
    comptime F: type,
    a: []const f32,
    dst: []f32,
    m: usize,
    n: usize,
    lda: usize,
    k: u32,
    r: []const f32,
    ut: []const F,
    scratch: []f32,
) void {
    const q = m;
    const ku: usize = k;
    if (ku == 0) {
        for (0..n) |j| @memset(dst[j * lda ..][0..m], 0);
        return;
    }
    const uf = scratch[0 .. m * ku];
    const cmat = scratch[q * q ..][0 .. ku * n];
    for (0..ku) |i| {
        for (0..m) |p| uf[i * m + p] = @floatCast(ut[i * q + p]);
    }

    for (0..ku) |i| {
        const ui = uf[i * m ..][0..m];
        for (0..n) |j| {
            const aj = a[j * lda ..][0..m];
            var acc: Vf = @splat(0);
            var p: usize = 0;
            while (p + VEC <= m) : (p += VEC) {
                const x: Vf = ui[p..][0..VEC].*;
                const y: Vf = aj[p..][0..VEC].*;
                acc = @mulAdd(Vf, x, y, acc);
            }
            var s: f32 = @reduce(.Add, acc);
            while (p < m) : (p += 1) s = @mulAdd(f32, ui[p], aj[p], s);
            cmat[i * n + j] = r[i] * s;
        }
    }

    for (0..n) |j| {
        const dj = dst[j * lda ..][0..m];
        var p: usize = 0;
        while (p + VEC <= m) : (p += VEC) {
            var acc: Vf = @splat(0);
            for (0..ku) |i| {
                const uv: Vf = uf[i * m + p ..][0..VEC].*;
                acc = @mulAdd(Vf, uv, @splat(cmat[i * n + j]), acc);
            }
            dj[p..][0..VEC].* = acc;
        }
        while (p < m) : (p += 1) {
            var s: f32 = 0;
            for (0..ku) |i| s = @mulAdd(f32, uf[i * m + p], cmat[i * n + j], s);
            dj[p] = s;
        }
    }
}
