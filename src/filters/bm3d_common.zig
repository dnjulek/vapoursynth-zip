//! Parts shared by the two BM3D kernels (`bm3d_cuda.zig`, `bm3d_classic.zig`):
//!
//! - `Ring`: per-thread numerator/denominator accumulators that hold only the
//!   rows a stage can still touch. A stage walks its reference blocks top to
//!   bottom and every group member lies within `reach` rows of its reference
//!   block, so rows above `y_ref - reach` are final and are handed to a sink
//!   (division into the output frame, or a fold into the temporal cache).
//! - `rcp`: bit-exact emulation of the GPU's approximate reciprocal.
//! - `TemporalCache`: the fused temporal aggregation (one BM3D pass per center
//!   frame, its slices summed per output frame in VAggregate's order).

const std = @import("std");
const common = @import("../common.zig");
const simd = @import("simd.zig");

const allocator = common.kernel_allocator;
const io = common.io;

pub const V8 = @Vector(8, f32);

// ---------------------------------------------------------------------------
// Ring accumulators

/// Accumulators for `slots` window frames x `planes` planes, each holding
/// `rows` rows of numerator and denominator (row y lives at y % rows).
pub const Ring = struct {
    slots: u32,
    planes: u32,
    rows: u32,
    width: u32,
    height: u32,
    /// floats per ring row (multiple of 16, at least width + 8)
    stride: u32,
    /// first row not yet flushed
    lo: u32 = 0,
    buf: []align(64) f32,

    pub fn create(slots: u32, planes: u32, rows: u32, width: u32, height: u32) error{OutOfMemory}!Ring {
        std.debug.assert(rows >= 8 and rows <= height);
        const stride = std.mem.alignForward(u32, width + 8, 16);
        const n = @as(usize, slots) * planes * 2 * rows * stride;
        const buf = try allocator.alignedAlloc(f32, .@"64", n);
        @memset(buf, 0);
        return .{ .slots = slots, .planes = planes, .rows = rows, .width = width, .height = height, .stride = stride, .buf = buf };
    }

    pub fn destroy(self: *Ring) void {
        allocator.free(self.buf);
    }

    /// Same geometry: the scratch can be reused as is.
    pub fn matches(self: *const Ring, slots: u32, planes: u32, rows: u32, width: u32, height: u32) bool {
        return self.slots == slots and self.planes == planes and self.rows == rows and self.width == width and self.height == height;
    }

    pub fn reset(self: *Ring) void {
        self.lo = 0;
    }

    /// Row y of (slot, plane); `den` selects the denominator.
    pub inline fn row(self: *const Ring, slot: u32, plane: u32, den: bool, y: u32) [*]f32 {
        const r = y % self.rows;
        const base = ((@as(usize, slot) * self.planes + plane) * 2 + @intFromBool(den)) * self.rows;
        return self.buf.ptr + (base + r) * self.stride;
    }

    /// Walks rows y, y+1, ... of (slot, plane) with one division (for the
    /// first ring index) instead of one per row: accumulation calls `row`
    /// for every row of every group member, and `y % rows` dominated it.
    pub const Cursor = struct {
        num: [*]f32,
        den: [*]f32,
        stride: usize,
        r: u32,
        rows: u32,

        pub inline fn numRow(c: *const Cursor) [*]f32 {
            return c.num + c.r * c.stride;
        }

        pub inline fn denRow(c: *const Cursor) [*]f32 {
            return c.den + c.r * c.stride;
        }

        pub inline fn next(c: *Cursor) void {
            c.r += 1;
            if (c.r == c.rows) c.r = 0;
        }
    };

    pub inline fn cursor(self: *const Ring, slot: u32, plane: u32, y: u32) Cursor {
        return .{
            .num = self.row(slot, plane, false, 0),
            .den = self.row(slot, plane, true, 0),
            .stride = self.stride,
            .r = y % self.rows,
            .rows = self.rows,
        };
    }

    /// Whether a block spanning rows y..y+bs-1 is still held.
    pub inline fn holds(self: *const Ring, y: u32, bs: u32) bool {
        return y >= self.lo and y + bs <= self.lo + self.rows;
    }

    /// Hands rows lo..upto-1 (clamped to the plane) to `sink.rows(slot,
    /// plane, y, num, den)` for every slot and plane, then clears them.
    pub fn flush(self: *Ring, upto: u32, sink: anytype) void {
        const end = @min(upto, self.height);
        while (self.lo < end) : (self.lo += 1) {
            const y = self.lo;
            for (0..self.slots) |s| {
                for (0..self.planes) |p| {
                    const num = self.row(@intCast(s), @intCast(p), false, y)[0..self.width];
                    const den = self.row(@intCast(s), @intCast(p), true, y)[0..self.width];
                    sink.rows(@intCast(s), @intCast(p), y, num, den);
                    @memset(num, 0);
                    @memset(den, 0);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// GPU reciprocal

/// `rcp.approx.ftz.f32` (MUFU.RCP, the reciprocal `1.0f / x` compiles to under
/// -use_fast_math) as measured on sm_86: it depends only on the mantissa
/// (verified over exponents -60..100) and is the correctly rounded 1/x plus
/// -1, 0 or +1 ulp. `bm3d_rcp.zlib` holds that correction for all 2^23
/// mantissas, 2 bits each (code = delta + 1, four per byte, low bits first).
const rcp_zlib = @embedFile("bm3d_rcp.zlib");
const rcp_table_len = (1 << 23) / 4;

var rcp_once_lock: std.Io.Mutex = .init;
var rcp_table: std.atomic.Value(?[*]const u8) = .init(null);

/// The decoded correction table (2 MiB), decompressed on first use and kept
/// for the life of the process.
pub fn rcpTable() error{OutOfMemory}![*]const u8 {
    if (rcp_table.load(.acquire)) |t| return t;
    rcp_once_lock.lockUncancelable(io);
    defer rcp_once_lock.unlock(io);
    if (rcp_table.load(.acquire)) |t| return t;
    // never freed: shared by every filter instance of the process; 3 bytes
    // of padding for rcp's 4-byte gathers at the last entries
    const buf = try std.heap.c_allocator.alloc(u8, rcp_table_len + 3);
    @memset(buf[rcp_table_len..], 0);
    var in: std.Io.Reader = .fixed(rcp_zlib);
    var dec: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
    var out: std.Io.Writer = .fixed(buf[0..rcp_table_len]);
    const n = dec.reader.streamRemaining(&out) catch unreachable; // embedded, checked by the tests
    std.debug.assert(n == rcp_table_len);
    rcp_table.store(buf.ptr, .release);
    return buf.ptr;
}

/// GPU `1.0f / x` for positive normal x whose reciprocal is normal.
pub inline fn rcpScalar(table: [*]const u8, x: f32) f32 {
    const m: u32 = @as(u32, @bitCast(x)) & 0x7f_ffff;
    const code: i32 = (table[m >> 2] >> @intCast((m & 3) * 2)) & 3;
    const r: i32 = @bitCast(1.0 / x);
    return @bitCast(r + code - 1);
}

pub inline fn rcp(comptime N: comptime_int, table: [*]const u8, x: @Vector(N, f32)) @Vector(N, f32) {
    const U = @Vector(N, u32);
    const r: @Vector(N, f32) = @as(@Vector(N, f32), @splat(1.0)) / x;
    const m: U = @as(U, @bitCast(x)) & @as(U, @splat(0x7f_ffff));
    var delta: @Vector(N, i32) = undefined;
    if (N == 8 and simd.use_avx2) {
        // the table bytes of all 8 lanes in one gather (the per-lane
        // extract / load / insert was ~18% of the Wiener stage's time)
        const word = gather8(table, @bitCast(m >> @splat(2)));
        const sh: @Vector(N, u5) = @intCast((m & @as(U, @splat(3))) * @as(U, @splat(2)));
        delta = @as(@Vector(N, i32), @bitCast((word >> sh) & @as(U, @splat(3)))) - @as(@Vector(N, i32), @splat(1));
    } else {
        inline for (0..N) |i| {
            const code: i32 = (table[m[i] >> 2] >> @intCast((m[i] & 3) * 2)) & 3;
            delta[i] = code - 1;
        }
    }
    return @bitCast(@as(@Vector(N, i32), @bitCast(r)) + delta);
}

/// vpgatherdd: lane i = the 4 bytes at base + idx[i] (little endian).
inline fn gather8(base: [*]const u8, idx: @Vector(8, i32)) @Vector(8, u32) {
    var out: @Vector(8, u32) = undefined;
    var mask: @Vector(8, i32) = @splat(-1);
    asm ("vpgatherdd %[mask], (%[base],%[idx],1), %[out]"
        : [out] "=&x" (out),
          [mask] "=&x" (mask),
        : [base] "r" (base),
          [idx] "x" (idx),
          [mask_in] "1" (mask),
    );
    return out;
}

// ---------------------------------------------------------------------------
// Fused temporal aggregation

/// Which BM3D pass and stack slot feeds contribution `i` of output frame n.
pub const Contrib = struct { m: i32, slot: u32 };

/// The temporal window conventions of the two plugins.
pub const Window = enum {
    /// vszipcu / BM3DCUDA BM3Dv2: stage m covers frames clamp(m - r + z),
    /// z = 0..2r; output n sums stage clamp(n - r + i), slot aggZ(i), for
    /// i = 0..2r (edge stages count twice, like VAggregate).
    cuda,
    /// VapourSynth-BM3D VBasic/VFinal + VAggregate: stage m covers frames
    /// m + k, k = -min(m, r)..min(N-1-m, r), in slot r + k; output n sums
    /// the stages n + k over the same clamped range, oldest first.
    classic,

    /// Contributions of output frame n, in summation order.
    pub fn contribs(w: Window, n: i32, nf: i32, r: i32, out: []Contrib) []Contrib {
        switch (w) {
            .cuda => {
                const k: usize = @intCast(2 * r + 1);
                for (0..k) |i_| {
                    const i: i32 = @intCast(i_);
                    const z = @min(@max(2 * r - i, n - nf + 1 + r), n + r);
                    out[i_] = .{ .m = std.math.clamp(n - r + i, 0, nf - 1), .slot = @intCast(z) };
                }
                return out[0..k];
            },
            .classic => {
                const b = -@min(n, r);
                const f = @min(nf - 1 - n, r);
                var c: usize = 0;
                var k = b;
                while (k <= f) : (k += 1) {
                    out[c] = .{ .m = n + k, .slot = @intCast(r - k) };
                    c += 1;
                }
                return out[0..c];
            },
        }
    }
};

pub const max_window = 33;

/// Output frames being aggregated, keyed by frame number. Each holds full
/// numerator/denominator planes; a stage's rows are folded in as soon as
/// they are final, but only in contribution order (row by row): rows that
/// arrive early are parked until the earlier ones are in. The result is the
/// same float sum, in the same order, as the stacked clip plus VAggregate.
///
/// Locking: `lock` guards the bookkeeping (targets, pins, claims) and each
/// target's own `lock` its rows and counters; the order is cache then
/// target, and `fold` never holds both. The row additions run under the
/// target lock only: with one lock for both (and targets allocated and
/// zeroed under it), 16 threads at 1080p r = 2 spent ~10% of their time
/// waiting for it. A stage is claimed by one thread at a time; threads that
/// need a claimed stage wait. Stage work never waits, so there is no
/// deadlock: the claiming thread already holds every source frame the stage
/// reads. (Computing stages ahead of the window instead of waiting was
/// measured: no gain at 16 threads, so requests stay at n - 2r .. n + 2r.)
pub const TemporalCache = struct {
    window: Window,
    radius: i32,
    num_frames: i32,
    /// processed planes and their geometry (VapourSynth plane index)
    process: [3]bool,
    width: [3]u32,
    height: [3]u32,
    total_rows: u32,
    /// target count above which unpinned targets are evicted
    cap: u32,

    lock: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    targets: std.AutoHashMapUnmanaged(i32, *Target) = .empty,
    /// stages being computed
    computing: std.AutoHashMapUnmanaged(i32, void) = .empty,
    /// recently returned frames: late rows for them are dropped instead of
    /// recreating the target
    done: [64]i32 = @splat(-1),
    done_pos: usize = 0,
    /// retired targets kept for reuse: a fresh one costs page faults on two
    /// full planes per processed plane, under the cache lock
    spare: [4]?*Target = @splat(null),

    pub const Target = struct {
        n: i32,
        /// cache lock: requests and folds in progress, and whether the
        /// request returned (the last unpin then drops it)
        pins: u32 = 0,
        returned: bool = false,
        /// guards everything below
        lock: std.Io.Mutex = .init,
        k: u32,
        contribs: [max_window]Contrib,
        /// a row is undefined until its first contribution (`next` 0)
        num: [3][]f32 = .{ &.{}, &.{}, &.{} },
        den: [3][]f32 = .{ &.{}, &.{}, &.{} },
        /// per plane and row: index of the next contribution to fold
        next: [3][]u8 = .{ &.{}, &.{}, &.{} },
        /// per plane, contribution (up to max_window) and row: a row (num
        /// then den) that arrived before its turn; all null when retired
        parked: [3][]?[*]f32 = .{ &.{}, &.{}, &.{} },
        /// rows (all planes) received per contribution
        received: [max_window]u32 = @splat(0),
        remaining: u32,

        fn freeParked(t: *Target, c: *const TemporalCache) void {
            for (0..3) |p| {
                if (!c.process[p]) continue;
                for (t.parked[p]) |*pk| {
                    if (pk.*) |row| allocator.free(row[0 .. 2 * c.width[p]]);
                    pk.* = null;
                }
            }
        }

        fn destroy(t: *Target, c: *const TemporalCache) void {
            t.freeParked(c);
            for (0..3) |p| {
                if (!c.process[p]) continue;
                allocator.free(t.parked[p]);
                allocator.free(t.next[p]);
                allocator.free(t.num[p]);
                allocator.free(t.den[p]);
            }
            allocator.destroy(t);
        }

        fn has(t: *const Target, c: *const TemporalCache, p: usize, i: usize, y: usize) bool {
            return t.next[p][y] > i or t.parked[p][i * c.height[p] + y] != null;
        }
    };

    pub fn init(window: Window, radius: u32, num_frames: i32, process: [3]bool, width: [3]u32, height: [3]u32, threads: u32) TemporalCache {
        var rows: u32 = 0;
        for (0..3) |p| {
            if (process[p]) rows += height[p];
        }
        return .{
            .window = window,
            .radius = @intCast(radius),
            .num_frames = num_frames,
            .process = process,
            .width = width,
            .height = height,
            .total_rows = rows,
            // requested frames, plus the 2r ahead that the stages computed
            // for them already feed
            .cap = threads + 4 * radius + 2,
        };
    }

    pub fn deinit(self: *TemporalCache) void {
        var it = self.targets.valueIterator();
        while (it.next()) |t| t.*.destroy(self);
        for (self.spare) |sp| {
            if (sp) |t| t.destroy(self);
        }
        self.targets.deinit(allocator);
        self.computing.deinit(allocator);
    }

    fn isDone(self: *const TemporalCache, n: i32) bool {
        for (self.done) |d| {
            if (d == n) return true;
        }
        return false;
    }

    fn create(self: *TemporalCache, n: i32) error{OutOfMemory}!*Target {
        const t = for (&self.spare) |*sp| {
            if (sp.*) |t| {
                sp.* = null;
                break t;
            }
        } else try self.alloc();
        errdefer self.retire(t);
        // buffers (and the all-null `parked`) are kept
        t.* = .{
            .n = n,
            .k = 0,
            .contribs = undefined,
            .remaining = self.total_rows,
            .num = t.num,
            .den = t.den,
            .next = t.next,
            .parked = t.parked,
        };
        t.k = @intCast(self.window.contribs(n, self.num_frames, self.radius, &t.contribs).len);
        for (0..3) |p| {
            if (self.process[p]) @memset(t.next[p], 0);
        }
        try self.targets.put(allocator, n, t);
        return t;
    }

    fn alloc(self: *const TemporalCache) error{OutOfMemory}!*Target {
        const t = try allocator.create(Target);
        t.* = .{ .n = -1, .k = 0, .contribs = undefined, .remaining = 0 };
        errdefer t.destroy(self);
        for (0..3) |p| {
            if (!self.process[p]) continue;
            const plane = @as(usize, self.width[p]) * self.height[p];
            t.num[p] = try allocator.alloc(f32, plane);
            t.den[p] = try allocator.alloc(f32, plane);
            t.next[p] = try allocator.alloc(u8, self.height[p]);
            t.parked[p] = try allocator.alloc(?[*]f32, max_window * @as(usize, self.height[p]));
            @memset(t.parked[p], null);
        }
        return t;
    }

    /// Drops a target already out of `targets`: kept as a spare if there is
    /// room. Cache lock held.
    fn retire(self: *TemporalCache, t: *Target) void {
        t.freeParked(self);
        for (&self.spare) |*sp| {
            if (sp.* == null) {
                sp.* = t;
                return;
            }
        }
        t.destroy(self);
    }

    /// Evicts unpinned targets, farthest from `near` first, until below cap.
    fn makeRoom(self: *TemporalCache, near: i32) void {
        while (self.targets.count() >= self.cap) {
            var victim: ?*Target = null;
            var it = self.targets.valueIterator();
            while (it.next()) |tp| {
                const t = tp.*;
                if (t.pins > 0) continue;
                if (victim == null or @abs(t.n - near) > @abs(victim.?.n - near)) victim = t;
            }
            const v = victim orelse return;
            _ = self.targets.remove(v.n);
            self.retire(v);
        }
    }

    fn feeds(self: *const TemporalCache, n: i32, m: i32, slot: u32) bool {
        var buf: [max_window]Contrib = undefined;
        for (self.window.contribs(n, self.num_frames, self.radius, &buf)) |c| {
            if (c.m == m and c.slot == slot) return true;
        }
        return false;
    }

    /// Folds row y of plane p from (stage m, slot) into every target it
    /// feeds. Called by the stage thread's sink.
    pub fn fold(self: *TemporalCache, m: i32, slot: u32, p: usize, y: u32, num: []const f32, den: []const f32) void {
        if (!self.process[p]) return;
        const span = 2 * self.radius;
        var n = @max(m - span, 0);
        while (n <= @min(m + span, self.num_frames - 1)) : (n += 1) {
            if (!self.feeds(n, m, slot)) continue;
            const t = self.pinFor(n) orelse continue;
            t.lock.lockUncancelable(io);
            self.foldInto(t, m, slot, p, y, num, den);
            t.lock.unlock(io);
            self.lock.lockUncancelable(io);
            self.unpin(t);
            self.lock.unlock(io);
        }
    }

    /// Target n, pinned; created when missing if there is room (the request
    /// usually follows), never for returned frames.
    fn pinFor(self: *TemporalCache, n: i32) ?*Target {
        self.lock.lockUncancelable(io);
        defer self.lock.unlock(io);
        const t = self.targets.get(n) orelse blk: {
            if (self.isDone(n)) return null;
            self.makeRoom(n);
            if (self.targets.count() >= self.cap) return null;
            break :blk self.create(n) catch return null;
        };
        t.pins += 1;
        return t;
    }

    /// Cache lock held.
    fn unpin(self: *TemporalCache, t: *Target) void {
        t.pins -= 1;
        if (t.pins > 0 or !t.returned) return;
        _ = self.targets.remove(t.n);
        self.done[self.done_pos] = t.n;
        self.done_pos = (self.done_pos + 1) % self.done.len;
        self.retire(t);
    }

    fn foldInto(self: *TemporalCache, t: *Target, m: i32, slot: u32, p: usize, y: u32, num: []const f32, den: []const f32) void {
        const w = self.width[p];
        const h = self.height[p];
        for (t.contribs[0..t.k], 0..) |c, i| {
            if (c.m != m or c.slot != slot) continue;
            if (t.has(self, p, i, y)) continue;
            if (t.next[p][y] == i) {
                addRow(t, p, w, y, num, den, i == 0);
                t.next[p][y] += 1;
                t.received[i] += 1;
                // earlier arrivals that are now next in line
                while (t.next[p][y] < t.k) {
                    const j = t.next[p][y];
                    const idx = @as(usize, j) * h + y;
                    const pk = t.parked[p][idx] orelse break;
                    addRow(t, p, w, y, pk[0..w], pk[w .. 2 * w], false);
                    allocator.free(pk[0 .. 2 * w]);
                    t.parked[p][idx] = null;
                    t.next[p][y] += 1;
                }
                if (t.next[p][y] == t.k) t.remaining -= 1;
            } else {
                // without memory the row is simply not received: the frame's
                // own request recomputes the stage
                const row = allocator.alloc(f32, 2 * w) catch continue;
                @memcpy(row[0..w], num);
                @memcpy(row[w..], den);
                t.parked[p][i * h + y] = row.ptr;
                t.received[i] += 1;
            }
        }
    }

    /// Adds a row; the first one is stored as 0 + v (-0 becomes +0, as when
    /// accumulating into zeroed planes), without reading the target.
    fn addRow(t: *Target, p: usize, w: u32, y: u32, num: []const f32, den: []const f32, first: bool) void {
        const o = @as(usize, y) * w;
        const tn = t.num[p][o..][0..w];
        const td = t.den[p][o..][0..w];
        if (first) {
            for (tn, td, num, den) |*a, *b, n, d| {
                a.* = 0 + n;
                b.* = 0 + d;
            }
            return;
        }
        var x: usize = 0;
        while (x + 8 <= w) : (x += 8) {
            tn[x..][0..8].* = @as(V8, tn[x..][0..8].*) + @as(V8, num[x..][0..8].*);
            td[x..][0..8].* = @as(V8, td[x..][0..8].*) + @as(V8, den[x..][0..8].*);
        }
        while (x < w) : (x += 1) {
            tn[x] += num[x];
            td[x] += den[x];
        }
    }

    /// Completes output frame n: computes (or waits for) every stage it
    /// needs through `stages.compute(m)`, hands its planes to
    /// `out.plane(p, num, den)`, then drops the target.
    pub fn produce(self: *TemporalCache, n: i32, stages: anytype, out: anytype) error{OutOfMemory}!void {
        self.lock.lockUncancelable(io);
        defer self.lock.unlock(io);

        for (&self.done) |*d| {
            if (d.* == n) d.* = -1;
        }
        const t = self.targets.get(n) orelse blk: {
            self.makeRoom(n);
            break :blk try self.create(n);
        };
        t.pins += 1;
        errdefer t.pins -= 1;

        while (true) {
            // a missing contribution whose stage nobody is computing
            var claim: ?i32 = null;
            t.lock.lockUncancelable(io);
            const remaining = t.remaining;
            if (remaining > 0) for (t.contribs[0..t.k], 0..) |c, i| {
                if (t.received[i] < self.total_rows and !self.computing.contains(c.m)) {
                    claim = c.m;
                    break;
                }
            };
            t.lock.unlock(io);
            if (remaining == 0) break;
            if (claim) |m| {
                try self.computing.put(allocator, m, {});
                self.lock.unlock(io);
                const r = stages.compute(m);
                self.lock.lockUncancelable(io);
                _ = self.computing.remove(m);
                self.cond.broadcast(io);
                try r;
            } else {
                self.cond.waitUncancelable(io, &self.lock);
            }
        }

        // complete: folds skip it now, so it can be read unlocked
        self.lock.unlock(io);
        for (0..3) |p| {
            if (self.process[p]) out.plane(p, t.num[p], t.den[p]);
        }
        self.lock.lockUncancelable(io);
        t.returned = true;
        self.unpin(t);
    }
};

/// Stack slot of window entry i for output frame n (vszipcu's VAggregate):
/// the slot of stage clamp(n - r + i) that holds frame n.
pub fn aggZ(i: i32, n: i32, nf: i32, r: i32) i32 {
    return @min(@max(2 * r - i, n - nf + 1 + r), n + r);
}

// ---------------------------------------------------------------------------
// Shared converted frames

/// f32 planes of source frames, shared by every stage that reads them: a
/// frame is converted once however many windows it is in. Entries are
/// pinned while a stage uses them; unpinned ones are evicted (oldest
/// first) above `cap`. A frame being converted by one thread is waited for
/// by the others.
pub const PlaneCache = struct {
    floats: usize,
    cap: u32,
    lock: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    entries: std.AutoHashMapUnmanaged(u64, *Entry) = .empty,
    tick: u64 = 0,

    const Entry = struct {
        data: []f32,
        ready: bool = false,
        pins: u32 = 0,
        last: u64 = 0,
    };

    pub fn init(floats: usize, cap: u32) PlaneCache {
        return .{ .floats = floats, .cap = cap };
    }

    pub fn deinit(self: *PlaneCache) void {
        var it = self.entries.valueIterator();
        while (it.next()) |e| {
            allocator.free(e.*.data);
            allocator.destroy(e.*);
        }
        self.entries.deinit(allocator);
    }

    /// The planes of `key`, pinned; `fill.fill(data)` converts them when
    /// absent. Pair with `release`.
    pub fn acquire(self: *PlaneCache, key: u64, fill: anytype) ![]const f32 {
        self.lock.lockUncancelable(io);
        defer self.lock.unlock(io);
        while (true) {
            if (self.entries.get(key)) |e| {
                if (!e.ready) {
                    self.cond.waitUncancelable(io, &self.lock);
                    continue;
                }
                e.pins += 1;
                self.tick += 1;
                e.last = self.tick;
                return e.data;
            }
            self.evict();
            const e = try allocator.create(Entry);
            e.* = .{ .data = allocator.alloc(f32, self.floats) catch |err| {
                allocator.destroy(e);
                return err;
            } };
            e.pins = 1;
            self.entries.put(allocator, key, e) catch |err| {
                allocator.free(e.data);
                allocator.destroy(e);
                return err;
            };
            // convert unlocked; others wait on `ready`
            self.lock.unlock(io);
            const r = fill.fill(e.data);
            self.lock.lockUncancelable(io);
            r catch |err| {
                _ = self.entries.remove(key);
                allocator.free(e.data);
                allocator.destroy(e);
                self.cond.broadcast(io);
                return err;
            };
            e.ready = true;
            self.tick += 1;
            e.last = self.tick;
            self.cond.broadcast(io);
            return e.data;
        }
    }

    pub fn release(self: *PlaneCache, key: u64) void {
        self.lock.lockUncancelable(io);
        defer self.lock.unlock(io);
        if (self.entries.get(key)) |e| e.pins -= 1;
    }

    fn evict(self: *PlaneCache) void {
        while (self.entries.count() >= self.cap) {
            var victim: ?u64 = null;
            var oldest: u64 = std.math.maxInt(u64);
            var it = self.entries.iterator();
            while (it.next()) |kv| {
                const e = kv.value_ptr.*;
                if (e.pins > 0 or !e.ready) continue;
                if (e.last < oldest) {
                    oldest = e.last;
                    victim = kv.key_ptr.*;
                }
            }
            const k = victim orelse return;
            const e = self.entries.fetchRemove(k).?.value;
            allocator.free(e.data);
            allocator.destroy(e);
        }
    }
};
