const std = @import("std");
const allocator = std.heap.c_allocator;

pub const Coord = extern struct { x: i16, y: i16 };

pub const NBR_POINT_LISTS: usize = 23;
const MAX_SUBSPL_POINTS: i32 = 4096;
const SPIRAL_THRESHOLD: i32 = 32;
const FSTB_PI: f64 = 3.1415926535897932384626433832795;
const MW_MARGIN: i32 = 1024;
const VNC_KS: i32 = 9;

// ---- libm (matches the C port's glibc math exactly) ----
extern fn cos(x: f64) f64;
extern fn sin(x: f64) f64;
extern fn pow(x: f64, y: f64) f64;
extern fn exp(x: f64) f64;
extern fn floor(x: f64) f64;
extern fn nearbyintf(x: f32) f32;

// fstb::round_int — round-to-nearest-even (default FP mode), like the C port.
fn roundInt(x: f32) i32 {
    return @intFromFloat(nearbyintf(x));
}
fn roundIntF64(x: f64) i32 {
    return roundInt(@floatCast(x));
}
fn limitInt(x: i32, lo: i32, hi: i32) i32 {
    return if (x < lo) lo else if (x > hi) hi else x;
}

// ---- RndGen LCG (per-row list pick + spiral random completion) ----
inline fn rndNextVal(v: u32) u32 {
    return v *% 1664525 +% 1013904223;
}

/// RndGen value after `step + 1` advances from seed 1 (single-threaded row n).
pub fn getRndAtStep(step: i32) u32 {
    var v: u32 = 1;
    var i: i32 = 0;
    while (i <= step) : (i += 1) v = rndNextVal(v);
    return v;
}

// ---- minstd_rand0 + uniform_int_distribution (libstdc++ default engine) ----
fn minstdSeed(seed: u32) u32 {
    const s = seed % 2147483647;
    return if (s == 0) 1 else s;
}
fn minstdNext(state: *u32) u32 {
    state.* = @intCast((@as(u64, state.*) * 16807) % 2147483647);
    return state.*;
}
fn minstdDist(state: *u32, n: i32) i32 {
    const urng_range: u32 = 2147483645;
    const un: u32 = @intCast(n);
    const scaling: u32 = urng_range / un;
    const past: u32 = un * scaling;
    var ret: u32 = undefined;
    while (true) {
        ret = minstdNext(state) - 1;
        if (ret < past) break;
    }
    return @intCast(ret / scaling);
}

// ---- MatrixWrap with toroidal wrap-around (VoidAndCluster scratch) ----
fn MW(comptime T: type) type {
    return struct {
        const Self = @This();
        w: i32,
        h: i32,
        data: []T,

        fn init(w: i32, h: i32) !Self {
            const n: usize = @intCast(w * h);
            const d = try allocator.alloc(T, n);
            @memset(d, 0);
            return .{ .w = w, .h = h, .data = d };
        }
        fn deinit(self: *Self) void {
            allocator.free(self.data);
        }
        inline fn idx(self: Self, x: i32, y: i32) usize {
            const xx = @mod(x, self.w);
            const yy = @mod(y, self.h);
            return @intCast(yy * self.w + xx);
        }
        inline fn get(self: Self, x: i32, y: i32) T {
            return self.data[self.idx(x, y)];
        }
        inline fn set(self: *Self, x: i32, y: i32, v: T) void {
            self.data[self.idx(x, y)] = v;
        }
        inline fn add(self: *Self, x: i32, y: i32, v: T) void {
            self.data[self.idx(x, y)] += v;
        }
    };
}
const MWu16 = MW(u16);
const MWf64 = MW(f64);

fn vncCreateGaussKernel() !MWf64 {
    var ker = try MWf64.init(VNC_KS, VNC_KS);
    const kh = @divTrunc(VNC_KS - 1, 2);
    const inv2s2: f64 = 1.0 / (2.0 * 1.5 * 1.5);
    var j: i32 = 0;
    while (j <= kh) : (j += 1) {
        var i: i32 = 0;
        while (i <= kh) : (i += 1) {
            const c = exp(-@as(f64, @floatFromInt(i * i + j * j)) * inv2s2);
            ker.set(i, j, c);
            ker.set(-i, j, c);
            ker.set(i, -j, c);
            ker.set(-i, -j, c);
        }
    }
    return ker;
}

fn vncGenerateInitialMat(m: *MWu16) !void {
    const thr: f64 = 0.1;
    const w = m.w;
    const h = m.h;
    var err = try MWf64.init(w, h);
    defer err.deinit();
    var dir: i32 = 1;
    var pass: i32 = 0;
    while (pass < 2) : (pass += 1) {
        var y: i32 = 0;
        while (y < h) : (y += 1) {
            const x_beg: i32 = if (dir < 0) w - 1 else 0;
            const x_end: i32 = if (dir < 0) -1 else w;
            var x: i32 = x_beg;
            while (x != x_end) : (x += dir) {
                const e0 = err.get(x, y);
                err.set(x, y, 0.0);
                const val = thr + e0;
                const qnt = roundInt(@floatCast(val));
                const qntc: i32 = if (qnt < 0) 0 else if (qnt > 1) 1 else qnt;
                m.set(x, y, @intCast(qntc));
                const e = val - @as(f64, @floatFromInt(qntc));
                const e2 = e * 0.5;
                const e4 = e * 0.25;
                err.add(x + dir, y, e2);
                err.add(x - dir, y + 1, e4);
                err.add(x, y + 1, e4);
            }
            dir = -dir;
        }
    }
}

/// First position (scan order) whose `color`-masked Gaussian cluster sum is the
/// strict maximum — equivalent to the C `arr[0]` after find_cluster_kernel.
fn vncFindCluster(m: MWu16, kern: MWf64, color: u16) struct { x: i32, y: i32 } {
    var best_v: f64 = -1.0;
    var bx: i32 = 0;
    var by: i32 = 0;
    const kw2 = @divTrunc(VNC_KS - 1, 2);
    const kh2 = @divTrunc(VNC_KS - 1, 2);
    var y: i32 = 0;
    while (y < m.h) : (y += 1) {
        var x: i32 = 0;
        while (x < m.w) : (x += 1) {
            if (m.get(x, y) != color) continue;
            var sum: f64 = 0.0;
            var j: i32 = -kh2;
            while (j <= kh2) : (j += 1) {
                var i: i32 = -kw2;
                while (i <= kw2) : (i += 1) {
                    if (m.get(x + i, y + j) == color) sum += kern.get(i, j);
                }
            }
            if (sum > best_v) {
                best_v = sum;
                bx = x;
                by = y;
            }
        }
    }
    return .{ .x = bx, .y = by };
}

fn vncHomogenizeInitialMat(m: *MWu16, kern: MWf64) void {
    while (true) {
        const c = vncFindCluster(m.*, kern, 1);
        m.set(c.x, c.y, 0);
        const v = vncFindCluster(m.*, kern, 0);
        m.set(v.x, v.y, 1);
        if (c.x == v.x and c.y == v.y) break;
    }
}

fn vncCountElt(m: MWu16, val: u16) i32 {
    var total: i32 = 0;
    for (m.data) |d| {
        if (d == val) total += 1;
    }
    return total;
}

fn createVncMatrix(vnc: *MWu16, vnc_size: i32) !void {
    var kern = try vncCreateGaussKernel();
    defer kern.deinit();
    var mat_base = try MWu16.init(vnc_size, vnc_size);
    defer mat_base.deinit();
    try vncGenerateInitialMat(&mat_base);
    vncHomogenizeInitialMat(&mat_base, kern);

    @memset(vnc.data, 0);
    const area: usize = @intCast(vnc_size * vnc_size);

    {
        var rank = vncCountElt(mat_base, 1);
        var mat = try MWu16.init(vnc_size, vnc_size);
        defer mat.deinit();
        @memcpy(mat.data, mat_base.data);
        while (rank > 0) {
            rank -= 1;
            const c = vncFindCluster(mat, kern, 1);
            mat.data[@intCast(c.y * vnc_size + c.x)] = 0;
            vnc.data[@intCast(c.y * vnc_size + c.x)] = @intCast(rank);
        }
    }
    {
        var rank = vncCountElt(mat_base, 1);
        var mat = try MWu16.init(vnc_size, vnc_size);
        defer mat.deinit();
        @memcpy(mat.data, mat_base.data);
        while (rank < vnc_size * vnc_size) {
            const v = vncFindCluster(mat, kern, 0);
            mat.data[@intCast(v.y * vnc_size + v.x)] = 1;
            vnc.data[@intCast(v.y * vnc_size + v.x)] = @intCast(rank);
            rank += 1;
        }
    }
    _ = area;
}

pub const PointLists = struct { pts: []Coord, k: usize };

/// Build all NBR_POINT_LISTS point lists for one (radius_h, radius_v, subspl)
/// geometry. Caller owns `pts` (free with the c_allocator).
pub fn generate(r_h: i32, r_v: i32, subspl: f64) !PointLists {
    const base_area = (r_h * 2 - 1) * (r_v * 2 - 1);
    var actual_subspl = subspl;
    if (subspl < 1e-3) actual_subspl = @floatFromInt(r_h + r_v);
    const k_i = limitInt(roundInt(@floatCast(@as(f64, @floatFromInt(base_area)) / actual_subspl)), 3, MAX_SUBSPL_POINTS);
    const K: usize = @intCast(k_i);

    const pts = try allocator.alloc(Coord, K * NBR_POINT_LISTS);
    errdefer allocator.free(pts);

    const max_h = r_h * 2 - 1;
    const max_v = r_v * 2 - 1;
    const vnc_size = limitInt(@divTrunc(@max(max_h, max_v) * 3, 2), 16, 32);
    const vnc_area = vnc_size * vnc_size;

    var vnc_mat: ?MWu16 = null;
    defer if (vnc_mat) |*vm| vm.deinit();
    if (k_i >= SPIRAL_THRESHOLD) {
        vnc_mat = try MWu16.init(vnc_size, vnc_size);
        try createVncMatrix(&vnc_mat.?, vnc_size);
    }

    // Three independent libstdc++ default_random_engine copies (seed 1).
    var ms_a = minstdSeed(1);
    var ms_x = minstdSeed(1);
    var ms_y = minstdSeed(1);
    // RndGen state, shared across all lists (spiral random completion).
    var rnd_val: u32 = 1;

    // The VNC scan below indexes done[] as `x + y*max_h` with x in [0,max_v) and
    // y in [0,max_h); for non-square geometry with max_h > max_v (e.g. YUV440)
    // that exceeds max_h*max_v. The original/C reference indexes out of bounds
    // here (silent UB in a vector<bool>/calloc buffer) — those geometries have no
    // well-defined reference, so we just size the buffer to the largest reachable
    // index to stay safe and deterministic. For the common formats (square, or
    // max_v >= max_h like YUV422) the bound is exactly max_h*max_v, so behaviour
    // is unchanged and bit-identical to the C port.
    const done_size: i32 = @max(max_h * max_v, (max_h - 1) * max_h + max_v);
    const done = try allocator.alloc(bool, @intCast(done_size));
    defer allocator.free(done);

    var list_cnt: usize = 0;
    while (list_cnt < NBR_POINT_LISTS) : (list_cnt += 1) {
        const cur = pts[list_cnt * K ..][0..K];
        @memset(done, false);
        cur[0] = .{ .x = 0, .y = 0 };
        done[@intCast((r_h - 1) + (r_v - 1) * max_h)] = true;
        var point_cnt: usize = 1;

        if (k_i < SPIRAL_THRESHOLD) {
            const angle_base = @as(f64, @floatFromInt(minstdDist(&ms_a, @intCast(NBR_POINT_LISTS)))) * (FSTB_PI * 0.5 / @as(f64, @floatFromInt(NBR_POINT_LISTS)));
            const arm_dir: i32 = 1 - (@as(i32, @intCast(list_cnt)) & 2);
            const narm: i32 = 4;
            const npa: i32 = @divTrunc(k_i - 1, narm);
            const amul = 2.0 * FSTB_PI / @as(f64, @floatFromInt(narm)) * @as(f64, @floatFromInt(arm_dir));
            var p: i32 = 0;
            while (p < npa) : (p += 1) {
                var posd = @as(f64, @floatFromInt(p)) / @as(f64, @floatFromInt(npa));
                posd = pow(posd, 3.0 / 5.0);
                var a: i32 = 0;
                while (a < narm) : (a += 1) {
                    const ang = angle_base + (posd * 2.0 + @as(f64, @floatFromInt(a))) * amul;
                    const x = roundIntF64(cos(ang) * posd * @as(f64, @floatFromInt(r_h - 1)));
                    const y = roundIntF64(sin(ang) * posd * @as(f64, @floatFromInt(r_v - 1)));
                    const da = (x + r_h - 1) + (y + r_v - 1) * max_h;
                    if (da >= 0 and da < max_h * max_v and !done[@intCast(da)]) {
                        cur[point_cnt] = .{ .x = @intCast(x), .y = @intCast(y) };
                        done[@intCast(da)] = true;
                        point_cnt += 1;
                    }
                }
            }
            while (point_cnt < K) {
                rnd_val = rndNextVal(rnd_val);
                const x = @as(i32, @intCast((rnd_val >> 8) % @as(u32, @intCast(max_h)))) - (r_h - 1);
                rnd_val = rndNextVal(rnd_val);
                const y = @as(i32, @intCast((rnd_val >> 8) % @as(u32, @intCast(max_v)))) - (r_v - 1);
                const da = (x + r_h - 1) + (y + r_v - 1) * max_h;
                if (!done[@intCast(da)]) {
                    cur[point_cnt] = .{ .x = @intCast(x), .y = @intCast(y) };
                    done[@intCast(da)] = true;
                    point_cnt += 1;
                }
            }
        } else {
            const win_w = max_h;
            const win_h = max_v;
            const ofs_x = minstdDist(&ms_x, max_h);
            const ofs_y = minstdDist(&ms_y, max_v);
            var cur_lvl: i32 = 0;
            var trg_lvl: i32 = @intFromFloat(floor(@as(f64, @floatFromInt(vnc_area)) / actual_subspl));
            while (point_cnt < K) {
                var y: i32 = 0;
                while (y < win_w and point_cnt < K) : (y += 1) {
                    var x: i32 = 0;
                    while (x < win_h and point_cnt < K) : (x += 1) {
                        const v: i32 = @intCast(vnc_mat.?.get(x + ofs_x, y + ofs_y));
                        if (v >= cur_lvl and v < trg_lvl) {
                            const px = x - (r_h - 1);
                            const py = y - (r_v - 1);
                            const da = (px + r_h - 1) + (py + r_v - 1) * max_h;
                            if (!done[@intCast(da)]) {
                                cur[point_cnt] = .{ .x = @intCast(px), .y = @intCast(py) };
                                done[@intCast(da)] = true;
                                point_cnt += 1;
                            }
                        }
                    }
                }
                cur_lvl = trg_lvl;
                trg_lvl += 1;
            }
        }
    }

    return .{ .pts = pts, .k = K };
}
