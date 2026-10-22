//! BM3D, classic variant: a port of VapourSynth-BM3D r12 (mawen1250,
//! `bm3d.Basic/Final/VBasic/VFinal`), output-identical to its x86-64 build.
//!
//! Numerics follow the C++ step by step:
//! - block distance: float SSD in the SSE2 order (4 accumulators for column
//!   mod 4 over 8-wide chunks, ((s0+s1)+s2)+s3, then the B%8 residue columns
//!   added one by one); key = float(double(ssd) * distMul); a candidate
//!   matches when ssd <= float(thSSE);
//! - selection: reference block first, then the best by (key, y, x)
//!   (V-BM3D: (key, z, y, x)), which is what the plugin's partial_sort /
//!   stable_sort produce;
//! - 3D transform: FFTW REDFT10/REDFT01 along x, then y, then z
//!   (`bm3d_fftw.zig`, bit-exact with FFTW's float build);
//! - hard threshold |c| > thr (strict), Wiener r^2 / (r^2 + s), L2 norm in
//!   the SSE2 4-lane order; weights numW = float(double(denW) / finalAMP);
//! - accumulation num += float(v * numW), den += denW in reference-block
//!   raster order, z order, row-major; dst = num / den.
//! No step contracts into an FMA: the plugin's x86-64 build has none.
//!
//! The speedups do not touch the arithmetic: distances for 8 consecutive
//! candidates run on 8 lanes (each lane repeats the scalar order), the
//! selection is a bounded insertion instead of a sort, and accumulation goes
//! through `bm3d_common.Ring` (only the rows still in reach are kept).

const std = @import("std");
const common = @import("../common.zig");
const bc = @import("bm3d_common.zig");
const fftw = @import("bm3d_fftw.zig");

const allocator = common.kernel_allocator;

pub const V8 = bc.V8;
pub const Ring = bc.Ring;

pub const max_radius = 16;
pub const max_group = 256;
pub const max_block = 64;

// ---------------------------------------------------------------------------
// Parameters

pub const Profile = enum {
    fast,
    lc,
    np,
    high,
    vn,

    pub fn parse(s: []const u8) ?Profile {
        return std.meta.stringToEnum(Profile, s);
    }
};

/// Profile defaults (BM3D_Para / VBM3D_Para).
pub const Defaults = struct {
    block_size: i64,
    block_step: i64,
    group_size: i64,
    bm_range: i64,
    bm_step: i64 = 1,
    lambda: f64 = 0,
    radius: i64 = 0,
    ps_num: i64 = 2,
    ps_range: i64 = 0,
    ps_step: i64 = 1,

    pub fn init(profile: Profile, wiener: bool, temporal: bool) Defaults {
        var d: Defaults = .{
            .block_size = 8,
            .bm_range = 16,
            .block_step = if (wiener) 3 else 4,
            .group_size = if (wiener) 32 else 16,
            .lambda = if (wiener) 0 else 2.7,
        };
        switch (profile) {
            .fast => {
                d.bm_range = 9;
                d.group_size = 8;
                d.block_step = if (wiener) 7 else 8;
            },
            .lc => {
                d.bm_range = 9;
                d.block_step = if (wiener) 5 else 6;
                if (wiener) d.group_size = 16;
            },
            .high => d.block_step = if (wiener) 2 else 3,
            .vn => if (wiener) {
                d.block_size = 11;
                d.block_step = 6;
            } else {
                d.block_step = 4;
                d.group_size = 32;
                d.lambda = 2.8;
            },
            .np => {},
        }
        if (temporal) {
            d.radius = 3;
            d.group_size = 8;
            d.bm_range = 12;
            d.ps_range = if (wiener) 6 else 5;
            switch (profile) {
                .fast => {
                    d.radius = 1;
                    d.bm_range = 7;
                    d.ps_range = if (wiener) 5 else 4;
                },
                .lc => {
                    d.radius = 2;
                    d.bm_range = 9;
                    d.ps_range = if (wiener) 5 else 4;
                },
                .high => {
                    d.radius = 4;
                    d.bm_range = 16;
                    d.ps_range = if (wiener) 8 else 7;
                },
                .vn => {
                    d.radius = 4;
                    d.group_size = 16;
                },
                .np => {},
            }
        }
        return d;
    }

    /// thMSE_Default, from the (already parsed) sigma of plane 0.
    pub fn thMse(profile: Profile, wiener: bool, sigma0: f64) f64 {
        if (!wiener) return if (profile == .vn) 1000 + sigma0 * 150 else 400 + sigma0 * 80;
        return if (profile == .vn) 400 + sigma0 * 40 else 200 + sigma0 * 10;
    }
};

pub const Matrix = enum(i64) {
    gbr = 0,
    bt709 = 1,
    unspecified = 2,
    fcc = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    bt2020nc = 9,
    bt2020c = 10,
    opp = 100,

    pub fn fromInt(v: i64) ?Matrix {
        return std.enums.fromInt(Matrix, v);
    }

    /// ColorMatrix_Default
    pub fn default(w: i64, h: i64) Matrix {
        if (w > 2048 or h > 1536) return .bt2020nc;
        if (w > 1024 or h > 576) return .bt709;
        return .smpte170m;
    }

    fn kParams(m: Matrix) [3]f80 {
        // the plugin's double literals, widened to long double (80-bit on
        // its mingw build)
        const k: [3]f64 = switch (m) {
            .gbr => .{ 0, 1, 0 },
            .fcc => .{ 0.30, 0.59, 0.11 },
            .bt470bg, .smpte170m => .{ 0.299, 0.587, 0.114 },
            .smpte240m => .{ 0.212, 0.701, 0.087 },
            .bt2020nc, .bt2020c => .{ 0.2627, 0.6780, 0.0593 },
            .opp => .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },
            else => .{ 0.2126, 0.7152, 0.0722 },
        };
        return .{ k[0], k[1], k[2] };
    }

    /// ColorMatrix_RGB2YUV_Parameter<double>: rows Y, U, V.
    pub fn rgb2yuv(m: Matrix) [3][3]f64 {
        switch (m) {
            .gbr => return .{ .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 1, 0, 0 } },
            .opp => {
                const third: f80 = 1.0 / @as(f80, 3.0);
                const t: f64 = @floatCast(third);
                return .{ .{ t, t, t }, .{ 0.5, 0, -0.5 }, .{ 0.25, -0.5, 0.25 } };
            },
            else => {
                const k = m.kParams();
                const yr: f64 = @floatCast(k[0]);
                const yg: f64 = @floatCast(k[1]);
                const yb: f64 = @floatCast(k[2]);
                // computed in long double from the double-rounded Y row
                const Yr: f80 = yr;
                const Yg: f80 = yg;
                const Yb: f80 = yb;
                const half: f80 = 0.5;
                return .{
                    .{ yr, yg, yb },
                    .{ @floatCast(-Yr * half / (1.0 - Yb)), @floatCast(-Yg * half / (1.0 - Yb)), 0.5 },
                    .{ 0.5, @floatCast(-Yg * half / (1.0 - Yr)), @floatCast(-Yb * half / (1.0 - Yr)) },
                };
            },
        }
    }

    /// Row norms (sigma and th_mse scaling).
    pub fn norms(m: Matrix) [3]f64 {
        const r = m.rgb2yuv();
        var n: [3]f64 = undefined;
        for (0..3) |i| n[i] = @sqrt(r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2]);
        return n;
    }
};

/// Thresholds (Basic) or Wiener sigma^2 (Final) per group size and number
/// of zero-frequency axes (BM3D_FilterData).
pub const FilterData = struct {
    final_amp: [max_group]f64 = undefined,
    coef: [max_group][4]f32 = undefined,

    pub fn init(wiener: bool, sigma: f64, group: u32, block: u32, lambda: f64) FilterData {
        var f: FilterData = .{};
        for (1..group + 1) |i| {
            const amp_i: i64 = 2 * @as(i64, @intCast(i)) * 2 * @as(i64, block) * 2 * @as(i64, block);
            const amp: f64 = @floatFromInt(amp_i);
            f.final_amp[i - 1] = amp;
            const fwd = @sqrt(amp);
            var c: [4]f64 = undefined;
            if (wiener) {
                const base = @max(sigma * fwd * sigma * fwd, @as(f64, std.math.floatMin(f32)));
                for (0..4) |flag| c[flag] = base * @as(f64, @floatFromInt(@as(u32, 1) << @intCast(flag)));
            } else {
                const t = sigma * lambda * fwd;
                c = .{ t, t * @sqrt(@as(f64, 2)), t * @as(f64, 2), t * @sqrt(@as(f64, 8)) };
            }
            for (0..4) |flag| f.coef[i - 1][flag] = @floatCast(c[flag]);
        }
        return f;
    }
};

/// One BM3D pass over the planes of a clip (one entry: plane 0 is the
/// matching plane; `process` selects the filtered planes).
pub const Params = struct {
    wiener: bool,
    n_planes: u32,
    process: [3]bool,
    /// plane sigma (scaled) is 0: Final passes the source through (RGB only)
    zero_sigma: [3]bool,
    width: u32,
    height: u32,
    block: u32,
    step: u32,
    group: u32,
    bm_range: u32,
    bm_step: u32,
    /// already multiplied by normY
    th_mse: f64,
    radius: u32,
    ps_num: u32,
    ps_range: u32,
    ps_step: u32,
    f: [3]FilterData,
    /// twiddles of every transform size up to max(group, block)
    tw: *const Twiddles,

    pub fn reach(self: *const Params) u32 {
        return self.bm_range + self.radius * self.ps_range;
    }

    pub fn ringRows(self: *const Params) u32 {
        return @min(self.height, 2 * self.reach() + self.block);
    }

    pub fn slots(self: *const Params) u32 {
        return 2 * self.radius + 1;
    }
};

pub const Twiddles = struct {
    /// FFTW twiddles of every size with an exact plan (`bm3d_fftw.twiddles`)
    data: [fftw.max_exact + 1][fftw.tw_len]f32 = undefined,

    pub fn init(self: *Twiddles, upto: u32) void {
        for (1..@min(upto, fftw.max_exact) + 1) |n| fftw.twiddles(@intCast(n), &self.data[n]);
    }

    /// Twiddles of size n (empty above max_exact: `slowDct` needs none).
    pub fn get(self: *const Twiddles, n: u32) []const f32 {
        return if (n <= fftw.max_exact) &self.data[n] else &.{};
    }
};

pub const PlaneIn = struct {
    data: [*]const f32,
    stride: usize,

    inline fn at(self: PlaneIn, x: u32, y: u32) [*]const f32 {
        return self.data + @as(usize, y) * self.stride + x;
    }
};

// ---------------------------------------------------------------------------
// Block matching

const Cand = struct {
    key: f32,
    z: i32,
    y: i32,
    x: i32,

    fn lessKYX(a: Cand, b: Cand) bool {
        if (a.key != b.key) return a.key < b.key;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    }

    fn lessKZYX(a: Cand, b: Cand) bool {
        if (a.key != b.key) return a.key < b.key;
        if (a.z != b.z) return a.z < b.z;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    }
};

/// The best `cap` candidates of one frame, kept sorted by (key, y, x), in any
/// arrival order.
const Top = struct {
    items: []Cand,
    len: u32 = 0,
    /// `Matcher.rejectBound` of the current worst key, kept across rows
    reject: ?f32 = null,
    reject_worst: ?f32 = null,

    fn insert(t: *Top, c: Cand) void {
        const cap: u32 = @intCast(t.items.len);
        if (cap == 0) return;
        if (t.len == cap and !Cand.lessKYX(c, t.items[cap - 1])) return;
        var pos = if (t.len < cap) t.len else cap - 1;
        while (pos > 0 and Cand.lessKYX(c, t.items[pos - 1])) : (pos -= 1) {
            t.items[pos] = t.items[pos - 1];
        }
        t.items[pos] = c;
        if (t.len < cap) t.len += 1;
    }

    fn slice(t: *const Top) []const Cand {
        return t.items[0..t.len];
    }
};

/// The reference block, row-major B*B, and the distance constants.
const Matcher = struct {
    ref: [max_block * max_block]f32,
    block: u32,
    th_sse: f32,
    dist_mul: f64,

    fn init(m: *Matcher, p: *const Params, plane: PlaneIn, x: u32, y: u32) void {
        const b = p.block;
        m.block = b;
        for (0..b) |r| @memcpy(m.ref[r * b ..][0..b], plane.at(x, y + @as(u32, @intCast(r)))[0..b]);
        const mse2sse = @as(f64, @floatFromInt(b * b)) * 1.0 * 1.0 / @as(f64, 255 * 255);
        m.dist_mul = 1.0 / mse2sse;
        m.th_sse = @floatCast(p.th_mse * mse2sse);
    }

    /// SSD of one candidate in the plugin's SSE2 order.
    fn ssd1(m: *const Matcher, p: [*]const f32, stride: usize) f32 {
        const b = m.block;
        const sw = b - b % 8;
        var dist: f32 = 0;
        if (sw > 0) {
            var s: [4]f32 = @splat(0);
            for (0..b) |r| {
                const rr = m.ref[r * b ..];
                const pp = p + r * stride;
                var c: usize = 0;
                while (c < sw) : (c += 8) {
                    inline for (0..4) |l| {
                        const d = rr[c + l] - pp[c + l];
                        s[l] += d * d;
                    }
                    inline for (0..4) |l| {
                        const d = rr[c + 4 + l] - pp[c + 4 + l];
                        s[l] += d * d;
                    }
                }
            }
            dist += ((s[0] + s[1]) + s[2]) + s[3];
        }
        if (sw < b) {
            for (0..b) |r| {
                const rr = m.ref[r * b ..];
                const pp = p + r * stride;
                for (sw..b) |c| {
                    const d = rr[c] - pp[c];
                    dist += d * d;
                }
            }
        }
        return dist;
    }

    /// SSDs of 8 candidates at x, x+1, ..., x+7 (lane i = candidate i), each
    /// lane in the same order as `ssd1`.
    ///
    /// Null when no lane can match. A partial sum is a lower bound of the
    /// final distance (every term is >= 0, and adding a non-negative float
    /// never lowers a sum), so once every lane's partial sum reaches `reject`
    /// (`rejectBound`) the rest cannot change the outcome. It is checked once,
    /// halfway: the branch is data dependent and mispredicts often, and on
    /// real 1080p content (B = 8) one check at row 4 beat checks at rows
    /// 2/4/6 by 10% and every row by 20%.
    fn ssd8(m: *const Matcher, p: [*]const f32, stride: usize, reject: f32) ?V8 {
        const b = m.block;
        const sw = b - b % 8;
        const rv: V8 = @splat(reject);
        const half = b / 2;
        var dist: V8 = @splat(0);
        if (sw > 0) {
            var s: [4]V8 = @splat(@splat(0));
            for (0..b) |r| {
                if (r == half and @reduce(.And, ((s[0] + s[1]) + s[2]) + s[3] >= rv)) return null;
                const rr = m.ref[r * b ..];
                const pp = p + r * stride;
                var c: usize = 0;
                while (c < sw) : (c += 8) {
                    inline for (0..4) |l| {
                        const d = @as(V8, @splat(rr[c + l])) - @as(V8, pp[c + l ..][0..8].*);
                        s[l] += d * d;
                    }
                    inline for (0..4) |l| {
                        const d = @as(V8, @splat(rr[c + 4 + l])) - @as(V8, pp[c + 4 + l ..][0..8].*);
                        s[l] += d * d;
                    }
                }
            }
            dist += ((s[0] + s[1]) + s[2]) + s[3];
        }
        if (sw < b) {
            for (0..b) |r| {
                if (sw == 0 and r == half and @reduce(.And, dist >= rv)) return null;
                const rr = m.ref[r * b ..];
                const pp = p + r * stride;
                for (sw..b) |c| {
                    const d = @as(V8, @splat(rr[c])) - @as(V8, pp[c..][0..8].*);
                    dist += d * d;
                }
            }
        }
        return dist;
    }

    /// A distance from which no candidate can enter: above th_sse, or (top
    /// full) a key not below the worst kept key. The second is the f32 one
    /// ulp above the nearest f32 of w / mul, so it exceeds w / mul exactly
    /// (the f64 quotient is off by far less than half an f32 ulp) and every
    /// d at or above it has d * mul >= w, hence key(d) >= w. It is at most
    /// two ulps above the tightest bound, which only costs a rare reject.
    fn rejectBound(m: *const Matcher, worst: ?f32) f32 {
        const inf = std.math.inf(f32);
        const lim = std.math.nextAfter(f32, m.th_sse, inf);
        const w = worst orelse return lim;
        const q: f32 = @floatCast(@as(f64, w) / m.dist_mul);
        return @min(lim, std.math.nextAfter(f32, q, inf));
    }

    inline fn key(m: *const Matcher, dist: f32) f32 {
        return @floatCast(@as(f64, dist) * m.dist_mul);
    }

    /// Tests row y, columns x0, x0 + step, ... x1 (inclusive); `skip_x` is
    /// left out (the reference position in its own frame).
    fn row(m: *const Matcher, plane: PlaneIn, width: u32, y: u32, x0: u32, x1: u32, step: u32, skip_x: ?u32, z: i32, top: *Top) void {
        const base = plane.at(0, y);
        var x = x0;
        // 8 consecutive candidates per vector while the reads stay in the
        // row (x + 7 + block <= width)
        if (step == 1 and x + 7 <= x1 and x1 + m.block <= width) {
            // x1 + block <= width holds for every window, so a vector ending
            // at x1 stays in the row: the last partial vector is shifted back
            // to x1 - 7 and its lanes below `x` (already tested) are masked
            const lane: @Vector(8, u32) = std.simd.iota(u32, 8);
            while (x <= x1) {
                const xs = @min(x, x1 - 7);
                const w: ?f32 = if (top.len == top.items.len and top.len > 0) top.items[top.len - 1].key else null;
                if (top.reject == null or w != top.reject_worst) {
                    top.reject_worst = w;
                    top.reject = m.rejectBound(w);
                }
                const live = @as(@Vector(8, u32), @splat(xs)) + lane >= @as(@Vector(8, u32), @splat(x));
                x = xs + 8;
                const reject = top.reject.?;
                const d = m.ssd8(base + xs, plane.stride, reject) orelse continue;
                // d < reject covers d <= th_sse (reject <= next float above it)
                const hit = @select(bool, live, d < @as(V8, @splat(reject)), @as(@Vector(8, bool), @splat(false)));
                if (!@reduce(.Or, hit)) continue;
                const keys: V8 = @floatCast(@as(@Vector(8, f64), d) * @as(@Vector(8, f64), @splat(m.dist_mul)));
                inline for (0..8) |l| {
                    const xl: u32 = xs + @as(u32, l);
                    if (hit[l] and xl != skip_x) top.insert(.{ .key = keys[l], .z = z, .y = @intCast(y), .x = @intCast(xl) });
                }
            }
            return;
        }
        while (x <= x1) : (x += step) {
            if (x == skip_x) continue;
            const d = m.ssd1(base + x, plane.stride);
            if (d <= m.th_sse) top.insert(.{ .key = m.key(d), .z = z, .y = @intCast(y), .x = @intCast(x) });
        }
    }
};

/// _SearchBoundary
fn bound(pos: i64, edge: i64, range: i64, step: i64) i64 {
    const r = @divTrunc(@min(range, @as(i64, @intCast(@abs(pos - edge)))), step) * step;
    return if (pos > edge) pos - r else pos + r;
}

/// The window of `range` around (x, y), aligned to (x, y) by `step` and
/// clipped to the plane.
const Window = struct {
    l: u32,
    r: u32,
    t: u32,
    b: u32,

    fn init(x: u32, y: u32, w: u32, h: u32, blk: u32, range_in: u32, step: u32) Window {
        const range: i64 = @divTrunc(@as(i64, range_in), step) * step;
        return .{
            .l = @intCast(bound(x, 0, range, step)),
            .r = @intCast(bound(x, w - blk, range, step)),
            .t = @intCast(bound(y, 0, range, step)),
            .b = @intCast(bound(y, h - blk, range, step)),
        };
    }
};

/// Spatial group of the reference block at (x, y) of window frame `cur`:
/// the reference first, then up to group - 1 matches by (key, y, x).
/// Returns the count written to `out`.
fn matchFrame(p: *const Params, m: *const Matcher, plane: PlaneIn, x: u32, y: u32, z: i32, out: []Cand, scratch: []Cand) u32 {
    out[0] = .{ .key = 0, .z = z, .y = @intCast(y), .x = @intCast(x) };
    if (p.group == 1 or p.th_mse <= 0) return 1;
    var top: Top = .{ .items = scratch[0 .. p.group - 1] };
    const win = Window.init(x, y, p.width, p.height, p.block, p.bm_range, p.bm_step);
    // the reference row first, then rows alternately above and below: near
    // blocks tend to match best, which tightens `Top.reject` early (the
    // result does not depend on the order)
    m.row(plane, p.width, y, win.l, win.r, p.bm_step, x, z, &top);
    var off = p.bm_step;
    while (off <= y - win.t or off <= win.b - y) : (off += p.bm_step) {
        if (off <= y - win.t) m.row(plane, p.width, y - off, win.l, win.r, p.bm_step, null, z, &top);
        if (off <= win.b - y) m.row(plane, p.width, y + off, win.l, win.r, p.bm_step, null, z, &top);
    }
    @memcpy(out[1..][0..top.len], top.slice());
    return 1 + top.len;
}

/// Sorted union of the windows of `range` around each seed (GenSearchPos),
/// as runs (y, x0..x1) in (y, x) order.
const Run = struct { y: u32, x0: u32, x1: u32 };

fn searchRuns(p: *const Params, seeds: []const Cand, runs: *std.ArrayList(Run), mask: *std.ArrayList(bool)) error{OutOfMemory}!void {
    runs.clearRetainingCapacity();
    if (seeds.len == 0) return;
    // bounding box of all windows, then a mask
    var bx0: u32 = std.math.maxInt(u32);
    var by0: u32 = std.math.maxInt(u32);
    var bx1: u32 = 0;
    var by1: u32 = 0;
    var wins: [max_group]Window = undefined;
    for (seeds, 0..) |s, i| {
        const w = Window.init(@intCast(s.x), @intCast(s.y), p.width, p.height, p.block, p.ps_range, p.ps_step);
        wins[i] = w;
        bx0 = @min(bx0, w.l);
        by0 = @min(by0, w.t);
        bx1 = @max(bx1, w.r);
        by1 = @max(by1, w.b);
    }
    const bw = bx1 - bx0 + 1;
    const bh = by1 - by0 + 1;
    try mask.resize(allocator, @as(usize, bw) * bh);
    @memset(mask.items, false);
    for (wins[0..seeds.len]) |w| {
        var yy = w.t;
        while (yy <= w.b) : (yy += p.ps_step) {
            var xx = w.l;
            while (xx <= w.r) : (xx += p.ps_step) mask.items[(yy - by0) * bw + (xx - bx0)] = true;
        }
    }
    for (0..bh) |r| {
        var c: u32 = 0;
        while (c < bw) {
            if (!mask.items[r * bw + c]) {
                c += 1;
                continue;
            }
            const start = c;
            while (c < bw and mask.items[r * bw + c]) c += 1;
            try runs.append(allocator, .{ .y = by0 + @as(u32, @intCast(r)), .x0 = bx0 + start, .x1 = bx0 + c - 1 });
        }
    }
}

/// BlockMatchingMulti over search positions: the best `group` by (key, y, x).
fn matchRuns(p: *const Params, m: *const Matcher, plane: PlaneIn, runs: []const Run, z: i32, out: []Cand) u32 {
    var top: Top = .{ .items = out[0..p.group] };
    for (runs) |r| {
        if (p.ps_step == 1) {
            m.row(plane, p.width, r.y, r.x0, r.x1, 1, null, z, &top);
        } else {
            // windows of step > 1 interleave: test the marked positions only
            var x = r.x0;
            while (x <= r.x1) : (x += 1) m.row(plane, p.width, r.y, x, x, 1, null, z, &top);
        }
    }
    return top.len;
}

/// Per-thread matching scratch.
pub const MatchScratch = struct {
    runs: std.ArrayList(Run) = .empty,
    mask: std.ArrayList(bool) = .empty,
    /// window-frame lists, then the merged group
    lists: []Cand = &.{},
    tmp: []Cand = &.{},

    fn ensure(s: *MatchScratch, n: usize) error{OutOfMemory}!void {
        if (s.lists.len >= n) return;
        allocator.free(s.lists);
        allocator.free(s.tmp);
        s.lists = &.{};
        s.tmp = &.{};
        s.lists = try allocator.alloc(Cand, n);
        s.tmp = try allocator.alloc(Cand, n);
    }

    pub fn deinit(s: *MatchScratch) void {
        s.runs.deinit(allocator);
        s.mask.deinit(allocator);
        allocator.free(s.lists);
        allocator.free(s.tmp);
    }
};

/// V-BM3D group (VBM3D_Process_Base::BlockMatching): current frame, then
/// predictive search backward then forward, merged by (key, z, y, x) with
/// the reference first. `mp` holds plane 0 of the window frames, `cur` the
/// reference frame's index in it.
fn matchTemporal(p: *const Params, m: *const Matcher, mp: []const PlaneIn, cur: u32, x: u32, y: u32, s: *MatchScratch, out: []Cand) error{OutOfMemory}!u32 {
    const g = p.group;
    const frames: u32 = @intCast(mp.len);
    try s.ensure(@as(usize, g) * frames + 1);
    const curz: i32 = @intCast(cur);
    if (g == 1 or p.th_mse <= 0) {
        out[0] = .{ .key = 0, .z = curz, .y = @intCast(y), .x = @intCast(x) };
        return 1;
    }
    var all = s.lists;
    var n = matchFrame(p, m, mp[cur], x, y, curz, all, s.tmp);
    const cur_n = n;
    const n_seed = @min(p.ps_num, cur_n);

    var dir: i32 = -1;
    while (dir <= 1) : (dir += 2) {
        // seeds: the current frame's top ps_num, then each frame's own top
        // (kept when a frame has no match)
        var seeds: [max_group]Cand = undefined;
        @memcpy(seeds[0..n_seed], all[0..n_seed]);
        var n_seeds = n_seed;
        var f: i32 = curz + dir;
        while (f >= 0 and f < frames) : (f += dir) {
            try searchRuns(p, seeds[0..n_seeds], &s.runs, &s.mask);
            const fn_ = matchRuns(p, m, mp[@intCast(f)], s.runs.items, f, all[n..]);
            if (fn_ > 0) {
                n_seeds = @min(p.ps_num, fn_);
                @memcpy(seeds[0..n_seeds], all[n..][0..n_seeds]);
            }
            n += fn_;
        }
    }

    // reference first, then the best g - 1 of the rest by (key, z, y, x)
    out[0] = all[0];
    var len: u32 = 0;
    const cap = g - 1;
    for (all[1..n]) |c| {
        if (len == cap and !Cand.lessKZYX(c, out[cap])) continue;
        var pos = if (len < cap) len + 1 else cap;
        while (pos > 1 and Cand.lessKZYX(c, out[pos - 1])) : (pos -= 1) out[pos] = out[pos - 1];
        out[pos] = c;
        if (len < cap) len += 1;
    }
    return 1 + len;
}

// ---------------------------------------------------------------------------
// 3D transform and collaborative filtering

/// FFTW r2r 3D on [g][b][b] (row-major): x, then y, then z.
fn transform3d(comptime forward: bool, tw: *const Twiddles, data: []f32, g: u32, b: u32) void {
    if (b == 8) return transform3d8(forward, tw, data, g);
    var line: [max_group]f32 = undefined;
    const bb = b * b;
    // x
    for (0..g * b) |r| fftw.dct(forward, f32, b, tw.get(b), data[r * b ..][0..b]);
    // y
    for (0..g) |z| {
        for (0..b) |x| {
            for (0..b) |y| line[y] = data[z * bb + y * b + x];
            fftw.dct(forward, f32, b, tw.get(b), line[0..b]);
            for (0..b) |y| data[z * bb + y * b + x] = line[y];
        }
    }
    // z
    for (0..bb) |i| {
        for (0..g) |z| line[z] = data[z * bb + i];
        fftw.dct(forward, f32, g, tw.get(g), line[0..g]);
        for (0..g) |z| data[z * bb + i] = line[z];
    }
}

/// 8x8 transpose of row vectors.
inline fn transpose8(v: *[8]V8) void {
    const lo = [8]i32{ 0, -1, 1, -2, 4, -5, 5, -6 };
    const hi = [8]i32{ 2, -3, 3, -4, 6, -7, 7, -8 };
    var t: [8]V8 = undefined;
    inline for (0..4) |i| {
        t[2 * i] = @shuffle(f32, v[2 * i], v[2 * i + 1], lo);
        t[2 * i + 1] = @shuffle(f32, v[2 * i], v[2 * i + 1], hi);
    }
    const s44 = [8]i32{ 0, 1, -1, -2, 4, 5, -5, -6 };
    const sEE = [8]i32{ 2, 3, -3, -4, 6, 7, -7, -8 };
    var u: [8]V8 = undefined;
    u[0] = @shuffle(f32, t[0], t[2], s44);
    u[1] = @shuffle(f32, t[0], t[2], sEE);
    u[2] = @shuffle(f32, t[1], t[3], s44);
    u[3] = @shuffle(f32, t[1], t[3], sEE);
    u[4] = @shuffle(f32, t[4], t[6], s44);
    u[5] = @shuffle(f32, t[4], t[6], sEE);
    u[6] = @shuffle(f32, t[5], t[7], s44);
    u[7] = @shuffle(f32, t[5], t[7], sEE);
    const p20 = [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 };
    const p31 = [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 };
    inline for (0..4) |i| {
        v[i] = @shuffle(f32, u[i], u[i + 4], p20);
        v[i + 4] = @shuffle(f32, u[i], u[i + 4], p31);
    }
}

/// B = 8: each block as 8 row vectors; x through a transpose, y across the
/// rows, z across the blocks (lane = column throughout).
fn transform3d8(comptime forward: bool, tw: *const Twiddles, data: []f32, g: u32) void {
    const rows: [*][8]f32 = @ptrCast(data.ptr);
    for (0..g) |z| {
        var v: [8]V8 = undefined;
        inline for (0..8) |r| v[r] = rows[z * 8 + r];
        transpose8(&v);
        if (forward) fftw.redft10(8, V8, tw.get(8), &v) else fftw.redft01(8, V8, tw.get(8), &v);
        transpose8(&v);
        if (forward) fftw.redft10(8, V8, tw.get(8), &v) else fftw.redft01(8, V8, tw.get(8), &v);
        inline for (0..8) |r| rows[z * 8 + r] = v[r];
    }
    var col: [max_group]V8 = undefined;
    for (0..8) |r| {
        for (0..g) |z| col[z] = rows[z * 8 + r];
        fftw.dct(forward, V8, g, tw.get(g), col[0..g]);
        for (0..g) |z| rows[z * 8 + r] = col[z];
    }
}

/// Coefficient table entry for (z, y, x) of a group of size g.
inline fn coefAt(c: *const [4]f32, z: usize, y: usize, x: usize) f32 {
    const flag = @as(u32, @intFromBool(x == 0)) + @intFromBool(y == 0) + @intFromBool(z == 0);
    return c[flag];
}

/// Hard thresholding: |c| > thr kept, else 0. Returns the retained count.
/// (The plugin's SSE loop and scalar tail decide alike: |c| > thr.)
fn hardThreshold(data: []f32, g: u32, b: u32, coef: *const [4]f32) u32 {
    if (b == 8) return hardThreshold8(data, g, coef);
    var kept: u32 = 0;
    var i: usize = 0;
    for (0..g) |z| {
        for (0..b) |y| {
            for (0..b) |x| {
                const t = coefAt(coef, z, y, x);
                if (@abs(data[i]) > t) kept += 1 else data[i] = 0;
                i += 1;
            }
        }
    }
    return kept;
}

const lane0: @Vector(8, bool) = .{ true, false, false, false, false, false, false, false };

/// Threshold row for flag = (y == 0) + (z == 0), plus 1 at x == 0.
inline fn rowCoef(coef: *const [4]f32, flag_zy: u32) V8 {
    return @select(f32, lane0, @as(V8, @splat(coef[flag_zy + 1])), @as(V8, @splat(coef[flag_zy])));
}

fn hardThreshold8(data: []f32, g: u32, coef: *const [4]f32) u32 {
    const rows: [*][8]f32 = @ptrCast(data.ptr);
    var kept: u32 = 0;
    for (0..g) |z| {
        for (0..8) |y| {
            const t = rowCoef(coef, @as(u32, @intFromBool(y == 0)) + @intFromBool(z == 0));
            const v: V8 = rows[z * 8 + y];
            const keep = @abs(v) > t;
            kept += @popCount(@as(u8, @bitCast(keep)));
            rows[z * 8 + y] = @select(f32, keep, v, @as(V8, @splat(0)));
        }
    }
    return kept;
}

/// Wiener shrinkage of `data` against the transformed basic estimate `ref`;
/// returns the L2 norm of the coefficients in the plugin's order (4 SSE
/// lanes over index mod 4, ((l0+l1)+l2)+l3, then the size % 4 tail one by
/// one).
fn wienerShrink(data: []f32, ref: []const f32, g: u32, b: u32, coef: *const [4]f32) f32 {
    if (b == 8) return wiener8(data, ref, g, coef);
    const total = @as(usize, g) * b * b;
    const sw = total - total % 4;
    var lane: [4]f32 = @splat(0);
    var tail: [3]f32 = undefined;
    var i: usize = 0;
    for (0..g) |z| {
        for (0..b) |y| {
            for (0..b) |x| {
                const r = ref[i];
                const r2 = r * r;
                const w = r2 / (r2 + coefAt(coef, z, y, x));
                data[i] = data[i] * w;
                if (i < sw) lane[i % 4] += w * w else tail[i - sw] = w * w;
                i += 1;
            }
        }
    }
    var l2: f32 = 0;
    if (sw > 0) l2 += ((lane[0] + lane[1]) + lane[2]) + lane[3];
    for (tail[0 .. total - sw]) |t| l2 += t;
    return l2;
}

/// B = 8: a row is 8 consecutive group elements, so the SSE lanes (index
/// mod 4) are x mod 4: each row adds its low half, then its high half.
fn wiener8(data: []f32, ref: []const f32, g: u32, coef: *const [4]f32) f32 {
    const V4 = @Vector(4, f32);
    const rows: [*][8]f32 = @ptrCast(data.ptr);
    const rrows: [*]const [8]f32 = @ptrCast(ref.ptr);
    var acc: V4 = @splat(0);
    for (0..g) |z| {
        for (0..8) |y| {
            const s = rowCoef(coef, @as(u32, @intFromBool(y == 0)) + @intFromBool(z == 0));
            const r: V8 = rrows[z * 8 + y];
            const r2 = r * r;
            const w = r2 / (r2 + s);
            rows[z * 8 + y] = @as(V8, rows[z * 8 + y]) * w;
            const w2 = w * w;
            acc += @shuffle(f32, w2, undefined, [4]i32{ 0, 1, 2, 3 });
            acc += @shuffle(f32, w2, undefined, [4]i32{ 4, 5, 6, 7 });
        }
    }
    return 0 + (((acc[0] + acc[1]) + acc[2]) + acc[3]);
}

/// Per-thread scratch of one classic filter instance.
pub const Scratch = struct {
    ring: ?Ring = null,
    group: []f32 = &.{},
    ref_group: []f32 = &.{},
    match: MatchScratch = .{},
    cands: []Cand = &.{},

    pub fn create() error{OutOfMemory}!*Scratch {
        const s = try allocator.create(Scratch);
        s.* = .{};
        return s;
    }

    pub fn destroy(self: *Scratch) void {
        if (self.ring) |*r| r.destroy();
        allocator.free(self.group);
        allocator.free(self.ref_group);
        allocator.free(self.cands);
        self.match.deinit();
        allocator.destroy(self);
    }

    pub fn prepare(self: *Scratch, p: *const Params) error{OutOfMemory}!*Ring {
        const n = @as(usize, p.group) * p.block * p.block;
        if (self.group.len < n) {
            allocator.free(self.group);
            allocator.free(self.ref_group);
            self.group = &.{};
            self.ref_group = &.{};
            self.group = try allocator.alloc(f32, n);
            self.ref_group = try allocator.alloc(f32, n);
        }
        if (self.cands.len < p.group + 1) {
            allocator.free(self.cands);
            self.cands = &.{};
            self.cands = try allocator.alloc(Cand, p.group + 1);
        }
        if (self.ring == null) self.ring = try Ring.create(p.slots(), p.n_planes, p.ringRows(), p.width, p.height);
        return &self.ring.?;
    }
};

fn loadGroup(dst: []f32, planes: []const PlaneIn, code: []const Cand, b: u32) void {
    var i: usize = 0;
    for (code) |c| {
        const pl = planes[@intCast(c.z)];
        for (0..b) |r| {
            const src = pl.at(@intCast(c.x), @intCast(c.y + @as(i32, @intCast(r))));
            @memcpy(dst[i..][0..b], src[0..b]);
            i += b;
        }
    }
}

/// AddTo(numW) / CountTo(denW) into the ring (slot = window frame):
/// num += float(v * numW), den += denW, element by element.
fn accumulate(ring: *Ring, plane: u32, data: []const f32, code: []const Cand, b: u32, slot_base: i32, num_w: f32, den_w: f32, scaled: bool) void {
    var i: usize = 0;
    const nw: V8 = @splat(num_w);
    const dw: V8 = @splat(den_w);
    for (code) |c| {
        const y: u32 = @intCast(c.y);
        const x: u32 = @intCast(c.x);
        const slot: u32 = @intCast(slot_base + c.z);
        var cur = ring.cursor(slot, plane, y);
        for (0..b) |_| {
            defer cur.next();
            const num = cur.numRow() + x;
            const den = cur.denRow() + x;
            var k: usize = 0;
            while (k + 8 <= b) : (k += 8) {
                const v: V8 = data[i + k ..][0..8].*;
                num[k..][0..8].* = @as(V8, num[k..][0..8].*) + (if (scaled) v * nw else v);
                den[k..][0..8].* = @as(V8, den[k..][0..8].*) + dw;
            }
            while (k < b) : (k += 1) {
                num[k] += if (scaled) data[i + k] * num_w else data[i + k];
                den[k] += den_w;
            }
            i += b;
        }
    }
}

/// One BM3D pass centered on window frame `cur` of the stage's window:
/// matching on plane 0 of `mp`, filtering of every processed plane,
/// accumulation into `ring` (slot = slot_base + window frame), rows handed to
/// `sink.rows(slot, plane, y, num, den)` as they finish.
///
/// `srcs` / `refs` (Final: the basic estimate): [plane * frames + frame].
pub fn stage(
    p: *const Params,
    mp: []const PlaneIn,
    cur: u32,
    srcs: []const PlaneIn,
    refs: []const PlaneIn,
    slot_base: i32,
    sc: *Scratch,
    sink: anytype,
) error{OutOfMemory}!void {
    const ring = try sc.prepare(p);
    try sc.match.ensure(@as(usize, p.group) * mp.len + 1);
    ring.reset();
    const frames: u32 = @intCast(mp.len);
    const b = p.block;
    const reach = p.reach();
    const bottom = p.height - b;
    const right = p.width - b;
    var m: Matcher = undefined;
    const code_buf = sc.cands;

    var j: u32 = 0;
    while (true) : (j += p.step) {
        if (j >= bottom + p.step) break;
        if (j > bottom) j = bottom;
        // nothing from here on reaches above j - reach
        ring.flush(j -| reach, sink);
        var i: u32 = 0;
        while (true) : (i += p.step) {
            if (i >= right + p.step) break;
            if (i > right) i = right;
            m.init(p, mp[cur], i, j);
            const n = if (frames > 1)
                try matchTemporal(p, &m, mp, cur, i, j, &sc.match, code_buf)
            else
                matchFrame(p, &m, mp[0], i, j, 0, code_buf, sc.match.tmp);
            const code = code_buf[0..n];
            const g: u32 = n;
            const len = @as(usize, g) * b * b;
            for (0..p.n_planes) |pl| {
                if (!p.process[pl]) continue;
                const plane: u32 = @intCast(pl);
                const f = &p.f[pl];
                const group = sc.group[0..len];
                loadGroup(group, srcs[pl * frames ..][0..frames], code, b);
                if (p.wiener and p.zero_sigma[pl]) {
                    accumulate(ring, plane, group, code, b, slot_base, 1, 1, false);
                    continue;
                }
                transform3d(true, p.tw, group, g, b);
                var den_w: f32 = undefined;
                if (p.wiener) {
                    const rg = sc.ref_group[0..len];
                    loadGroup(rg, refs[pl * frames ..][0..frames], code, b);
                    transform3d(true, p.tw, rg, g, b);
                    const l2 = @max(std.math.floatEps(f32), wienerShrink(group, rg, g, b, &f.coef[g - 1]));
                    den_w = 1.0 / l2;
                } else {
                    const kept = hardThreshold(group, g, b, &f.coef[g - 1]);
                    den_w = if (kept < 1) 1 else 1.0 / @as(f32, @floatFromInt(kept));
                }
                transform3d(false, p.tw, group, g, b);
                const num_w: f32 = @floatCast(@as(f64, den_w) / f.final_amp[g - 1]);
                accumulate(ring, plane, group, code, b, slot_base, num_w, den_w, true);
            }
        }
    }
    ring.flush(p.height, sink);
}

// ---------------------------------------------------------------------------
// Sample conversions (Conversion.hpp), per plane

/// Quantization of an integer plane (GetQuanPara).
const Quant = struct {
    floor: i32,
    neutral: i32,
    ceil: i32,

    fn init(bits: u32, full: bool, chroma: bool) Quant {
        const b: u5 = @intCast(bits);
        if (full) {
            return .{ .floor = 0, .neutral = if (chroma) @as(i32, 1) << (b - 1) else 0, .ceil = (@as(i32, 1) << b) - 1 };
        }
        const fl: i32 = @as(i32, 16) << (b - 8);
        return .{
            .floor = fl,
            .neutral = if (chroma) @as(i32, 1) << (b - 1) else fl,
            .ceil = (if (chroma) @as(i32, 240) else 235) << (b - 8),
        };
    }
};

/// Int2Float: x * gain + offset in float; luma to [0, 1], chroma to
/// [-0.5, 0.5].
pub fn intToFloat(comptime T: type, dst: [*]f32, dst_stride: usize, src: [*]const T, src_stride: usize, w: u32, h: u32, bits: u32, chroma: bool, full: bool) void {
    const q = Quant.init(bits, full, chroma);
    const gain: f32 = 1.0 / @as(f32, @floatFromInt(q.ceil - q.floor));
    const offset: f32 = -@as(f32, @floatFromInt(q.neutral)) * gain + 0.0;
    for (0..h) |y| {
        const s = src + y * src_stride;
        const d = dst + y * dst_stride;
        for (0..w) |x| d[x] = @as(f32, @floatFromInt(s[x])) * gain + offset;
    }
}

/// Float2Int: in double, +0.5 (0.499999 for full-range chroma), clipped to
/// the whole code range, truncated.
pub fn floatToInt(comptime T: type, dst: [*]T, dst_stride: usize, src: [*]const f32, src_stride: usize, w: u32, h: u32, bits: u32, chroma: bool, full: bool) void {
    const q = Quant.init(bits, full, chroma);
    const gain: f64 = @floatFromInt(q.ceil - q.floor);
    const neutral: f64 = if (chroma) @floatFromInt(q.neutral) else @floatFromInt(q.floor);
    const pc_chroma = chroma and q.floor < q.neutral and @mod(q.floor + q.ceil, 2) == 1;
    const offset: f64 = (-0.0 * gain + neutral) + (if (pc_chroma) @as(f64, 0.499999) else 0.5);
    const hi: f64 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    for (0..h) |y| {
        const s = src + y * src_stride;
        const d = dst + y * dst_stride;
        for (0..w) |x| {
            const v = @as(f64, s[x]) * gain + offset;
            const c = if (v <= 0) 0 else if (v >= hi) hi else v;
            d[x] = @intFromFloat(c);
        }
    }
}

fn rgbRange(comptime T: type, bits: u32, full: bool) struct { f32, f32 } {
    if (T == f32) return .{ 0, 1 };
    const q = Quant.init(bits, full, false);
    return .{ @floatFromInt(q.floor), @floatFromInt(q.ceil) };
}

inline fn toF(comptime T: type, v: T) f32 {
    return if (T == f32) v else @floatFromInt(v);
}

/// RGB to OPP (MatrixConvert_RGB2YUV, float output); `u`/`v` null: Y only
/// (ConvertToY, same Y formula).
pub fn rgbToOpp(comptime T: type, y: [*]f32, u: ?[*]f32, v: ?[*]f32, dst_stride: usize, r: [*]const T, g: [*]const T, b: [*]const T, src_stride: usize, w: u32, h: u32, bits: u32, full: bool) void {
    const sf, const sc = rgbRange(T, bits, full);
    const srange = sc - sf;
    const gain_y: f32 = 1.0 / (srange * 3.0);
    const off_y: f32 = -sf * 3.0 * gain_y + 0.0;
    const gain_u: f32 = 1.0 / (srange * 2.0);
    const gain_v: f32 = 1.0 / (srange * 4.0);
    for (0..h) |row| {
        const so = row * src_stride;
        const do_ = row * dst_stride;
        for (0..w) |x| {
            const R = toF(T, r[so + x]);
            const G = toF(T, g[so + x]);
            const B = toF(T, b[so + x]);
            y[do_ + x] = ((R + G) + B) * gain_y + off_y;
            if (u) |up| up[do_ + x] = (R - B) * gain_u + 0.0;
            if (v) |vp| vp[do_ + x] = ((R - G * 2.0) + B) * gain_v + 0.0;
        }
    }
}

/// OPP to RGB (MatrixConvert_YUV2RGB, float math), clipped and rounded for
/// integer output. Float OPP input.
pub fn oppToRgb(comptime T: type, r: [*]T, g: [*]T, b: [*]T, dst_stride: usize, y: [*]const f32, u: [*]const f32, v: [*]const f32, src_stride: usize, w: u32, h: u32, bits: u32, full: bool) void {
    oppToRgbQ(f32, T, r, g, b, dst_stride, y, u, v, src_stride, w, h, 32, bits, full);
}

/// MatrixConvert_YUV2RGB (OPP) from S samples (full-range integer or float
/// OPP) to T (RGB, `full` range for integer output).
pub fn oppToRgbQ(comptime S: type, comptime T: type, r: [*]T, g: [*]T, b: [*]T, dst_stride: usize, y: [*]const S, u: [*]const S, v: [*]const S, src_stride: usize, w: u32, h: u32, src_bits: u32, bits: u32, full: bool) void {
    const df, const dc = rgbRange(T, bits, full);
    var sfy: f32 = 0;
    var scy: f32 = 1;
    var sfc: f32 = -0.5;
    var snc: f32 = 0;
    var scc: f32 = 0.5;
    if (S != f32) {
        const qy = Quant.init(src_bits, true, false);
        const qc = Quant.init(src_bits, true, true);
        sfy = @floatFromInt(qy.floor);
        scy = @floatFromInt(qy.ceil);
        sfc = @floatFromInt(qc.floor);
        snc = @floatFromInt(qc.neutral);
        scc = @floatFromInt(qc.ceil);
    }
    const gain_y: f32 = (dc - df) / (scy - sfy);
    const gain_c: f32 = (dc - df) / (scc - sfc);
    const two3: f32 = @floatCast(2.0 / @as(f80, 3.0));
    const four3: f32 = @floatCast(-4.0 / @as(f80, 3.0));
    const ry = 1.0 * gain_y;
    const ru = 1.0 * gain_c;
    const rv = two3 * gain_c;
    const gy = 1.0 * gain_y;
    const gu = 0.0 * gain_c;
    const gv = four3 * gain_c;
    const by = 1.0 * gain_y;
    const bu = -1.0 * gain_c;
    const bv = two3 * gain_c;
    const round: f32 = if (T == f32) 0 else 0.5;
    const off_r = ((-sfy * ry) - snc * (ru + rv)) + df + round;
    const off_g = ((-sfy * gy) - snc * (gu + gv)) + df + round;
    const off_b = ((-sfy * by) - snc * (bu + bv)) + df + round;
    const hi: f32 = if (T == f32) 0 else @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    const L = struct {
        inline fn load(p: [*]const S) V8 {
            const s: @Vector(8, S) = p[0..8].*;
            return if (S == f32) s else @floatFromInt(s);
        }
        /// `clipF` then the integer conversion on 8 lanes (min/max equal the
        /// branches for finite values; i32 + truncate avoids the vector
        /// `@intCast` miscompile)
        inline fn store(p: [*]T, t: V8, top: V8) void {
            if (T == f32) {
                p[0..8].* = t;
            } else {
                const i: @Vector(8, i32) = @intFromFloat(@min(@max(t, @as(V8, @splat(0))), top));
                p[0..8].* = @as(@Vector(8, T), @truncate(@as(@Vector(8, u32), @bitCast(i))));
            }
        }
    };
    const top: V8 = @splat(hi);
    const ry8: V8 = @splat(ry);
    const ru8: V8 = @splat(ru);
    const rv8: V8 = @splat(rv);
    const gy8: V8 = @splat(gy);
    const gv8: V8 = @splat(gv);
    const by8: V8 = @splat(by);
    const bu8: V8 = @splat(bu);
    const bv8: V8 = @splat(bv);
    const or8: V8 = @splat(off_r);
    const og8: V8 = @splat(off_g);
    const ob8: V8 = @splat(off_b);
    for (0..h) |row| {
        const so = row * src_stride;
        const do_ = row * dst_stride;
        // 8 pixels per step, the same operations per lane as the scalar tail
        // (the scalar loop ran at half r12's speed for integer output)
        var x: usize = 0;
        while (x + 8 <= w) : (x += 8) {
            const Y = L.load(y + so + x);
            const U = L.load(u + so + x);
            const V = L.load(v + so + x);
            L.store(r + do_ + x, ((ry8 * Y + ru8 * U) + rv8 * V) + or8, top);
            L.store(g + do_ + x, (gy8 * Y + gv8 * V) + og8, top);
            L.store(b + do_ + x, ((by8 * Y + bu8 * U) + bv8 * V) + ob8, top);
        }
        while (x < w) : (x += 1) {
            const Y = toF(S, y[so + x]);
            const U = toF(S, u[so + x]);
            const V = toF(S, v[so + x]);
            const tr = ((ry * Y + ru * U) + rv * V) + off_r;
            const tg = (gy * Y + gv * V) + off_g;
            const tb = ((by * Y + bu * U) + bv * V) + off_b;
            if (T == f32) {
                r[do_ + x] = tr;
                g[do_ + x] = tg;
                b[do_ + x] = tb;
            } else {
                r[do_ + x] = @intFromFloat(clipF(tr, hi));
                g[do_ + x] = @intFromFloat(clipF(tg, hi));
                b[do_ + x] = @intFromFloat(clipF(tb, hi));
            }
        }
    }
}

/// RGB2OPP to 16-bit (MatrixConvert_RGB2YUV OPP with calcType double, full
/// range output, PC chroma rounding 0.499999).
pub fn rgbToOpp16(comptime S: type, y: [*]u16, u: [*]u16, v: [*]u16, dst_stride: usize, r: [*]const S, g: [*]const S, b: [*]const S, src_stride: usize, w: u32, h: u32, src_bits: u32, full: bool) void {
    var sf: f64 = 0;
    var sc: f64 = 1;
    if (S != f32) {
        const q = Quant.init(src_bits, full, false);
        sf = @floatFromInt(q.floor);
        sc = @floatFromInt(q.ceil);
    }
    const srange = sc - sf;
    const gain_y: f64 = 65535.0 / (srange * 3.0);
    const off_y: f64 = (-sf * 3.0 * gain_y + 0.0) + 0.5;
    const gain_u: f64 = 65535.0 / (srange * 2.0);
    const gain_v: f64 = 65535.0 / (srange * 4.0);
    const off_c: f64 = 32768.0 + 0.499999;
    // 8 pixels per step in f64 lanes, the same operations per lane as the
    // scalar tail (the scalar loop ran ~20% behind r12 at 1T)
    const F8 = @Vector(8, f64);
    const gy: F8 = @splat(gain_y);
    const oy: F8 = @splat(off_y);
    const gu: F8 = @splat(gain_u);
    const gv: F8 = @splat(gain_v);
    const oc: F8 = @splat(off_c);
    for (0..h) |row| {
        const so = row * src_stride;
        const do_ = row * dst_stride;
        var x: usize = 0;
        while (x + 8 <= w) : (x += 8) {
            const R = toF64x8(S, r[so + x ..][0..8].*);
            const G = toF64x8(S, g[so + x ..][0..8].*);
            const B = toF64x8(S, b[so + x ..][0..8].*);
            y[do_ + x ..][0..8].* = clip16x8(((R + G) + B) * gy + oy);
            u[do_ + x ..][0..8].* = clip16x8((R - B) * gu + oc);
            v[do_ + x ..][0..8].* = clip16x8(((R - G * @as(F8, @splat(2.0))) + B) * gv + oc);
        }
        while (x < w) : (x += 1) {
            const R: f64 = toF(S, r[so + x]);
            const G: f64 = toF(S, g[so + x]);
            const B: f64 = toF(S, b[so + x]);
            y[do_ + x] = clip16(((R + G) + B) * gain_y + off_y);
            u[do_ + x] = clip16((R - B) * gain_u + off_c);
            v[do_ + x] = clip16(((R - G * 2.0) + B) * gain_v + off_c);
        }
    }
}

inline fn clip16(v: f64) u16 {
    return @intFromFloat(if (v <= 0) 0 else if (v >= 65535) 65535 else v);
}

inline fn toF64x8(comptime T: type, v: @Vector(8, T)) @Vector(8, f64) {
    return if (T == f32) @floatCast(v) else @floatFromInt(v);
}

/// `clip16` on 8 lanes (min/max equal the branches for finite values). The
/// conversion goes through i32 and a truncate: a direct vector cast to u16
/// hits the Zig 0.17.0 signed→narrower-unsigned `@intCast` miscompile.
inline fn clip16x8(v: @Vector(8, f64)) @Vector(8, u16) {
    const c = @min(@max(v, @as(@Vector(8, f64), @splat(0))), @as(@Vector(8, f64), @splat(65535)));
    const i: @Vector(8, i32) = @intFromFloat(c);
    return @truncate(@as(@Vector(8, u32), @bitCast(i)));
}

inline fn clipF(v: f32, hi: f32) f32 {
    return if (v <= 0) 0 else if (v >= hi) hi else v;
}

/// Final division of the aggregated planes.
pub fn divideRow(out: []f32, num: []const f32, den: []const f32) void {
    for (out, num, den) |*o, n, d| o.* = n / d;
}


