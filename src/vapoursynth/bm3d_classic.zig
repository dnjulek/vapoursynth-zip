//! BM3DBasic / BM3DFinal / BM3DVBasic / BM3DVFinal: VapourSynth-BM3D r12
//! (`bm3d.*`) on the CPU, same arguments, defaults and output
//! (`filters/bm3d_classic.zig`). The V* functions aggregate internally:
//! they return what VBasic/VFinal + VAggregate return, in the input format.

const std = @import("std");
const math = std.math;

const filter = @import("../filters/bm3d_classic.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

pub const Basic = Entry(false, false);
pub const Final = Entry(true, false);
pub const VBasic = Entry(false, true);
pub const VFinal = Entry(true, true);

fn Entry(comptime wiener: bool, comptime temporal: bool) type {
    return struct {
        pub const filter_name = (if (temporal) "BM3DV" else "BM3D") ++ (if (wiener) "Final" else "Basic");
        pub const params = "input:vnode;ref:vnode" ++ (if (wiener) ";" else ":opt;") ++
            "profile:data:opt;sigma:float[]:opt;" ++ (if (temporal) "radius:int:opt;" else "") ++
            "block_size:int:opt;block_step:int:opt;group_size:int:opt;bm_range:int:opt;bm_step:int:opt;" ++
            (if (temporal) "ps_num:int:opt;ps_range:int:opt;ps_step:int:opt;" else "") ++
            "th_mse:float:opt;" ++ (if (wiener) "" else "hard_thr:float:opt;") ++ "matrix:int:opt;";

        pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
            createImpl(wiener, temporal, filter_name, in, out, core, vsapi);
        }
    };
}

const Kind = enum { gray, yuv, rgb };

pub const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    kind: Kind = .gray,
    /// float input: the planes are used in place (Gray/YUV)
    float: bool = false,
    bits: u32 = 8,
    matrix: filter.Matrix = .bt709,
    /// all planes skipped: frames pass through
    skip: bool = false,
    p: filter.Params = undefined,
    tw: *filter.Twiddles = undefined,
    cache: bc.TemporalCache = undefined,
    planes: bc.PlaneCache = undefined,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *Scratch) = undefined,
};

/// Per-thread scratch: the kernel's plus the f32 working planes.
const Scratch = struct {
    k: *filter.Scratch,
    planes: std.ArrayList([]f32) = .empty,

    fn destroy(self: *Scratch) void {
        for (self.planes.items) |p| allocator.free(p);
        self.planes.deinit(allocator);
        self.k.destroy();
        allocator.destroy(self);
    }

    /// Working plane `i` (w * h floats).
    fn plane(self: *Scratch, i: usize, size: usize) ![]f32 {
        while (self.planes.items.len <= i) try self.planes.append(allocator, &.{});
        if (self.planes.items[i].len < size) {
            allocator.free(self.planes.items[i]);
            self.planes.items[i] = &.{};
            self.planes.items[i] = try allocator.alloc(f32, size);
        }
        return self.planes.items[i][0..size];
    }
};

fn acquireScratch(d: *Data) ?*Scratch {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |sc| return sc;
    const k = filter.Scratch.create() catch return null;
    const sc = allocator.create(Scratch) catch {
        k.destroy();
        return null;
    };
    sc.* = .{ .k = k };
    d.pool.put(tid, sc) catch {
        sc.destroy();
        return null;
    };
    return sc;
}

const ZF = ZAPI.ZFrame(*const vs.Frame);

/// Range of a frame: untagged RGB is full, untagged Gray/YUV limited; OPP
/// (BM3D_OPP=1) Gray/YUV is full. `_Range` (VapourSynth) is what the
/// plugin reads.
fn fullRange(f: ZF, rgb: bool) bool {
    const props = f.getPropertiesRO();
    const range = props.getInt(i64, "_Range");
    if (rgb) return range == null or range.? != 0;
    const opp = props.getInt(i64, "BM3D_OPP");
    if (opp != null and opp.? == 1) return true;
    return range != null and range.? != 0;
}

/// Converts the wanted planes of `f` into `bufs` (plane-sized each).
fn convert(d: *const Data, f: ZF, full: bool, want: [3]bool, y_only: bool, bufs: [3][]f32) void {
    const w: u32 = @intCast(d.vi.width);
    const h: u32 = @intCast(d.vi.height);
    switch (d.kind) {
        .gray, .yuv => {
            for (0..3) |p| {
                if (!want[p]) continue;
                const chroma = p > 0;
                if (d.bits == 8) {
                    filter.intToFloat(u8, bufs[p].ptr, w, f.getReadSlice2(u8, p).ptr, f.getStride2(u8, p), w, h, 8, chroma, full);
                } else {
                    filter.intToFloat(u16, bufs[p].ptr, w, f.getReadSlice2(u16, p).ptr, f.getStride2(u16, p), w, h, d.bits, chroma, full);
                }
            }
        },
        .rgb => {
            const u = if (y_only) null else bufs[1].ptr;
            const v = if (y_only) null else bufs[2].ptr;
            switch (d.bits) {
                inline 8, 16, 32 => |b| {
                    const T = if (b == 8) u8 else if (b == 16) u16 else f32;
                    filter.rgbToOpp(T, bufs[0].ptr, u, v, w, f.getReadSlice2(T, 0).ptr, f.getReadSlice2(T, 1).ptr, f.getReadSlice2(T, 2).ptr, f.getStride2(T, 0), w, h, b, full);
                },
                else => {
                    filter.rgbToOpp(u16, bufs[0].ptr, u, v, w, f.getReadSlice2(u16, 0).ptr, f.getReadSlice2(u16, 1).ptr, f.getReadSlice2(u16, 2).ptr, f.getStride2(u16, 0), w, h, d.bits, full);
                },
            }
        },
    }
}

/// Planes read in place: float Gray/YUV needs no conversion.
fn inPlace(d: *const Data) bool {
    return d.float and d.kind != .rgb;
}

fn viewOf(f: ZF, want: [3]bool, out: *[3]filter.PlaneIn) void {
    for (0..3) |p| {
        if (want[p]) out[p] = .{ .data = f.getReadSlice2(f32, p).ptr, .stride = f.getStride2(f32, p) };
    }
}

/// Converted (or in-place) f32 planes of one frame, [plane], in the
/// thread's scratch.
fn loadPlanes(
    d: *const Data,
    sc: *Scratch,
    slot: usize,
    f: ZF,
    full: bool,
    want: [3]bool,
    y_only: bool,
    out: *[3]filter.PlaneIn,
) !void {
    if (inPlace(d)) return viewOf(f, want, out);
    const w: u32 = @intCast(d.vi.width);
    const size = @as(usize, w) * @as(u32, @intCast(d.vi.height));
    var bufs: [3][]f32 = undefined;
    for (0..3) |p| {
        const need = if (d.kind == .rgb) (p == 0 or !y_only) else want[p];
        bufs[p] = if (need) try sc.plane(slot * 3 + p, size) else &.{};
        if (need) out[p] = .{ .data = bufs[p].ptr, .stride = w };
    }
    convert(d, f, full, want, y_only, bufs);
}

/// Writes finished rows into the output frame: num / den, then the plugin's
/// float -> output conversion. RGB rows wait for all three OPP planes.
const OutSink = struct {
    d: *const Data,
    dst: ZAPI.ZFrame(*vs.Frame),
    full: bool,
    rgb_rows: *[3][]f32,

    fn emit(self: *const OutSink, p: usize, y: u32, row: []const f32) void {
        const d = self.d;
        const w: u32 = @intCast(row.len);
        switch (d.kind) {
            .gray, .yuv => {
                if (d.float) {
                    const o = self.dst.getWriteSlice2(f32, p);
                    @memcpy(o[y * self.dst.getStride2(f32, p) ..][0..w], row);
                } else if (d.bits == 8) {
                    const o = self.dst.getWriteSlice2(u8, p);
                    filter.floatToInt(u8, o[y * self.dst.getStride2(u8, p) ..].ptr, 0, row.ptr, 0, w, 1, 8, p > 0, self.full);
                } else {
                    const o = self.dst.getWriteSlice2(u16, p);
                    filter.floatToInt(u16, o[y * self.dst.getStride2(u16, p) ..].ptr, 0, row.ptr, 0, w, 1, d.bits, p > 0, self.full);
                }
            },
            .rgb => {
                @memcpy(self.rgb_rows[p][0..w], row);
                if (p != 2) return;
                const r = self.rgb_rows;
                switch (d.bits) {
                    inline 8, 16, 32 => |b| {
                        const T = if (b == 8) u8 else if (b == 16) u16 else f32;
                        const s = self.dst.getStride2(T, 0);
                        filter.oppToRgb(T, self.dst.getWriteSlice2(T, 0)[y * s ..].ptr, self.dst.getWriteSlice2(T, 1)[y * s ..].ptr, self.dst.getWriteSlice2(T, 2)[y * s ..].ptr, 0, r[0].ptr, r[1].ptr, r[2].ptr, 0, w, 1, b, self.full);
                    },
                    else => {
                        const s = self.dst.getStride2(u16, 0);
                        filter.oppToRgb(u16, self.dst.getWriteSlice2(u16, 0)[y * s ..].ptr, self.dst.getWriteSlice2(u16, 1)[y * s ..].ptr, self.dst.getWriteSlice2(u16, 2)[y * s ..].ptr, 0, r[0].ptr, r[1].ptr, r[2].ptr, 0, w, 1, d.bits, self.full);
                    },
                }
            },
        }
    }
};

/// Spatial sink: divide, then emit.
const DivideSink = struct {
    out: *const OutSink,
    process: [3]bool,
    tmp: []f32,

    pub fn rows(self: *const DivideSink, slot: u32, plane: u32, y: u32, num: []const f32, den: []const f32) void {
        std.debug.assert(slot == 0);
        if (!self.process[plane]) return;
        const row = self.tmp[0..num.len];
        filter.divideRow(row, num, den);
        self.out.emit(plane, y, row);
    }
};

fn Spatial(comptime wiener: bool, comptime name: []const u8) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const rdef = d.ref_node != null;
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (rdef) zapi.requestFrameFilter(n, d.ref_node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            if (rejectOpp(&zapi, src, d, name)) return null;
            if (d.skip) return src.addFrameRef().frame;
            const ref = if (rdef) zapi.initZFrame(d.ref_node, n) else src;
            defer if (rdef) ref.deinit();

            const rgb = d.kind == .rgb;
            const full = fullRange(src, rgb);
            const ref_full = if (rdef) fullRange(ref, rgb) else full;
            const dst = src.newVideoFrame2(if (rgb) .{ true, true, true } else d.p.process);

            const sc = acquireScratch(d) orelse return hz.frameOom(&zapi, name, dst);
            run(wiener, d, sc, src, ref, rdef, full, ref_full, dst) catch return hz.frameOom(&zapi, name, dst);
            return dst.frame;
        }
    };
}

fn rejectOpp(zapi: *const ZAPI, src: ZF, d: *const Data, comptime name: []const u8) bool {
    if (d.kind == .rgb or d.matrix == .opp) return false;
    const opp = src.getPropertiesRO().getInt(i64, "BM3D_OPP");
    if (opp == null or opp.? != 1) return false;
    zapi.setFilterError(name ++ ": the input frame property \"BM3D_OPP=1\" indicates opponent color space input, \"matrix=100\" must be specified");
    return true;
}

fn run(comptime wiener: bool, d: *Data, sc: *Scratch, src: ZF, ref: ZF, rdef: bool, full: bool, ref_full: bool, dst: ZAPI.ZFrame(*vs.Frame)) !void {
    const p = &d.p;
    const proc = p.process;
    var sp: [3]filter.PlaneIn = undefined;
    var rp: [3]filter.PlaneIn = undefined;
    // src: plane 0 when filtered or used for matching; ref: plane 0 for
    // matching, chroma only for Wiener
    var want_src = proc;
    if (!rdef) want_src[0] = true;
    try loadPlanes(d, sc, 0, src, full, want_src, false, &sp);
    if (rdef) {
        var want_ref: [3]bool = .{ true, false, false };
        if (wiener) want_ref = .{ true, proc[1], proc[2] };
        try loadPlanes(d, sc, 1, ref, ref_full, want_ref, d.kind == .rgb and !wiener, &rp);
    } else rp = sp;

    const w: usize = @intCast(d.vi.width);
    const tmp = try sc.plane(6, w);
    var rgb_rows: [3][]f32 = .{ try sc.plane(7, w), try sc.plane(8, w), try sc.plane(9, w) };
    const out: OutSink = .{ .d = d, .dst = dst, .full = full, .rgb_rows = &rgb_rows };
    const sink: DivideSink = .{ .out = &out, .process = proc, .tmp = tmp };
    const np = p.n_planes;
    try filter.stage(p, rp[0..1], 0, sp[0..np], rp[0..np], 0, sc.k, &sink);
}


fn Temporal(comptime wiener: bool, comptime name: []const u8) type {
    return struct {
        const max_win = 2 * filter.max_radius + 1;
        const max_span = 4 * filter.max_radius + 1;

        /// Source frames n - 2r .. n + 2r (clamped): every stage n - r ..
        /// n + r reads within them.
        const Frames = struct {
            lo: i32,
            src: [max_span]ZF = undefined,
            ref: [max_span]ZF = undefined,
            count: usize = 0,
        };

        /// Planes of the reference clip that a stage reads.
        fn refWant(d: *const Data) [3]bool {
            if (!wiener) return .{ true, false, false };
            return .{ true, d.p.process[1], d.p.process[2] };
        }

        fn srcWant(d: *const Data) [3]bool {
            var w = d.p.process;
            if (d.ref_node == null) w[0] = true;
            return w;
        }

        const Fill = struct {
            d: *const Data,
            f: ZF,
            full: bool,
            want: [3]bool,
            y_only: bool,

            pub fn fill(self: *const Fill, data: []f32) !void {
                const size = @as(usize, @intCast(self.d.vi.width)) * @as(usize, @intCast(self.d.vi.height));
                convert(self.d, self.f, self.full, self.want, self.y_only, .{ data[0..size], data[size .. 2 * size], data[2 * size .. 3 * size] });
            }
        };

        /// Window frame planes: in place, or pinned in the shared cache
        /// (returns the key to release).
        fn planesOf(d: *Data, clip: u64, fnum: i32, f: ZF, want: [3]bool, y_only: bool, out: *[3]filter.PlaneIn) !?u64 {
            if (inPlace(d)) {
                viewOf(f, want, out);
                return null;
            }
            const key = (clip << 32) | @as(u64, @intCast(fnum));
            const fill: Fill = .{ .d = d, .f = f, .full = fullRange(f, d.kind == .rgb), .want = want, .y_only = y_only };
            const data = try d.planes.acquire(key, &fill);
            const w: usize = @intCast(d.vi.width);
            const size = w * @as(usize, @intCast(d.vi.height));
            for (0..3) |p| out[p] = .{ .data = data[p * size ..].ptr, .stride = w };
            return key;
        }

        const Stages = struct {
            d: *Data,
            fr: *const Frames,
            sc: *Scratch,

            /// Stage m: VBasic/VFinal's window m + k, k = -min(m, r) ..
            /// min(N-1-m, r), written to slots r + k.
            pub fn compute(self: *const Stages, m: i32) error{OutOfMemory}!void {
                const d = self.d;
                const r: i32 = @intCast(d.p.radius);
                const nf = d.vi.numFrames;
                const b = -@min(m, r);
                const fwd = @min(nf - 1 - m, r);
                const frames: usize = @intCast(fwd - b + 1);
                const rdef = d.ref_node != null;
                var keys: [2 * max_win]u64 = undefined;
                var nkeys: usize = 0;
                defer for (keys[0..nkeys]) |k| d.planes.release(k);

                var sp: [max_win][3]filter.PlaneIn = undefined;
                var rp: [max_win][3]filter.PlaneIn = undefined;
                for (0..frames) |i| {
                    const fnum = m + b + @as(i32, @intCast(i));
                    const k: usize = @intCast(fnum - self.fr.lo);
                    if (try planesOf(d, 0, fnum, self.fr.src[k], srcWant(d), false, &sp[i])) |key| {
                        keys[nkeys] = key;
                        nkeys += 1;
                    }
                    if (rdef) {
                        if (try planesOf(d, 1, fnum, self.fr.ref[k], refWant(d), d.kind == .rgb and !wiener, &rp[i])) |key| {
                            keys[nkeys] = key;
                            nkeys += 1;
                        }
                    } else rp[i] = sp[i];
                }
                var mp: [max_win]filter.PlaneIn = undefined;
                var srcs: [3 * max_win]filter.PlaneIn = undefined;
                var refs: [3 * max_win]filter.PlaneIn = undefined;
                const np = d.p.n_planes;
                for (0..frames) |i| {
                    mp[i] = rp[i][0];
                    for (0..np) |p| {
                        srcs[p * frames + i] = sp[i][p];
                        refs[p * frames + i] = rp[i][p];
                    }
                }
                const sink: FoldSink = .{ .cache = &d.cache, .m = m };
                try filter.stage(&d.p, mp[0..frames], @intCast(-b), srcs[0 .. np * frames], refs[0 .. np * frames], r + b, self.sc.k, &sink);
            }
        };

        const FoldSink = struct {
            cache: *bc.TemporalCache,
            m: i32,

            pub fn rows(self: *const FoldSink, slot: u32, plane: u32, y: u32, num: []const f32, den: []const f32) void {
                self.cache.fold(self.m, slot, plane, y, num, den);
            }
        };

        /// Divides the aggregated planes and converts them to the output
        /// format (RGB waits for all three OPP planes).
        const Out = struct {
            out: *const OutSink,
            sc: *Scratch,
            w: usize,
            h: usize,

            pub fn plane(self: *const Out, p: usize, num: []const f32, den: []const f32) void {
                const w = self.w;
                const rgb = self.out.d.kind == .rgb;
                if (rgb and p < 2) {
                    filter.divideRow(self.sc.planes.items[10 + p][0 .. w * self.h], num, den);
                    return;
                }
                const tmp = self.sc.planes.items[6][0..w];
                for (0..self.h) |y| {
                    filter.divideRow(tmp, num[y * w ..][0..w], den[y * w ..][0..w]);
                    if (rgb) {
                        self.out.emit(0, @intCast(y), self.sc.planes.items[10][y * w ..][0..w]);
                        self.out.emit(1, @intCast(y), self.sc.planes.items[11][y * w ..][0..w]);
                    }
                    self.out.emit(p, @intCast(y), tmp);
                }
            }
        };

        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const rdef = d.ref_node != null;
            const span: i32 = 2 * @as(i32, @intCast(d.p.radius));
            const lo = @max(n - span, 0);
            const hi = @min(n + span, d.vi.numFrames - 1);
            if (activation_reason == .Initial) {
                var i = lo;
                while (i <= hi) : (i += 1) {
                    zapi.requestFrameFilter(i, d.node);
                    if (rdef) zapi.requestFrameFilter(i, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            var fr: Frames = .{ .lo = lo };
            defer for (0..fr.count) |i| {
                fr.src[i].deinit();
                if (rdef) fr.ref[i].deinit();
            };
            var i = lo;
            while (i <= hi) : (i += 1) {
                fr.src[fr.count] = zapi.initZFrame(d.node, i);
                if (rdef) fr.ref[fr.count] = zapi.initZFrame(d.ref_node, i);
                fr.count += 1;
            }
            const src = fr.src[@intCast(n - lo)];
            for (fr.src[0..fr.count]) |f| {
                if (rejectOpp(&zapi, f, d, name)) return null;
            }

            const rgb = d.kind == .rgb;
            const dst = src.newVideoFrame2(if (rgb) .{ true, true, true } else d.p.process);
            const sc = acquireScratch(d) orelse return hz.frameOom(&zapi, name, dst);
            const w: usize = @intCast(d.vi.width);
            const h: usize = @intCast(d.vi.height);
            // scratch: 6 = a row, 7..9 = RGB rows, 10..11 = held RGB planes
            _ = sc.plane(6, w) catch return hz.frameOom(&zapi, name, dst);
            var rgb_rows: [3][]f32 = undefined;
            for (0..3) |k| rgb_rows[k] = sc.plane(7 + k, w) catch return hz.frameOom(&zapi, name, dst);
            if (rgb) {
                _ = sc.plane(10, w * h) catch return hz.frameOom(&zapi, name, dst);
                _ = sc.plane(11, w * h) catch return hz.frameOom(&zapi, name, dst);
            }
            const out_sink: OutSink = .{ .d = d, .dst = dst, .full = fullRange(src, rgb), .rgb_rows = &rgb_rows };
            const out: Out = .{ .out = &out_sink, .sc = sc, .w = w, .h = h };
            const stages: Stages = .{ .d = d, .fr = &fr, .sc = sc };
            d.cache.produce(n, &stages, &out) catch return hz.frameOom(&zapi, name, dst);
            return dst.frame;
        }
    };
}

fn classicFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    var it = d.pool.valueIterator();
    while (it.next()) |sc| sc.*.destroy();
    d.pool.deinit();
    if (d.p.radius > 0) {
        d.cache.deinit();
        d.planes.deinit();
    }
    allocator.destroy(d.tw);
    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

fn createImpl(comptime wiener: bool, comptime temporal: bool, comptime name: [:0]const u8, in: ?*const vs.Map, out: ?*vs.Map, core: ?*vs.Core, vsapi: ?*const vs.API) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("input").?;
    d.ref_node = map_in.getNode("ref");
    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.ref_node);
    };
    const fail = struct {
        fn f(mo: anytype, comptime msg: []const u8) void {
            mo.setError(name ++ ": " ++ msg);
        }
    }.f;

    const vi = d.vi;
    const fmt = vi.format;
    if (!vapoursynth.vshelper.isConstantVideoFormat(vi)) return fail(map_out, "Invalid input clip, only constant format input supported");
    if ((fmt.sampleType == .Integer and fmt.bitsPerSample > 16) or (fmt.sampleType == .Float and fmt.bitsPerSample != 32)) {
        return fail(map_out, "Invalid input clip, only 8-16 bit integer or 32 bit float formats supported");
    }
    d.float = fmt.sampleType == .Float;
    d.bits = @intCast(fmt.bitsPerSample);
    d.kind = switch (fmt.colorFamily) {
        .RGB => .rgb,
        .YUV => .yuv,
        else => .gray,
    };
    const rdef = d.ref_node != null;
    if (rdef) {
        const rvi = zapi.getVideoInfo(d.ref_node);
        if (!vapoursynth.vshelper.isConstantVideoFormat(rvi)) return fail(map_out, "Invalid clip \"ref\", only constant format input supported");
        if (!vapoursynth.vshelper.isSameVideoFormat(&rvi.format, &fmt)) return fail(map_out, "input clip and clip \"ref\" must be of the same format");
        if (rvi.width != vi.width or rvi.height != vi.height) return fail(map_out, "input clip and clip \"ref\" must be of the same width and height");
        if (rvi.numFrames != vi.numFrames) return fail(map_out, "input clip and clip \"ref\" must have the same number of frames");
    }

    const prof_s = map_in.getData("profile", 0) orelse "fast";
    const profile = filter.Profile.parse(prof_s) orelse return fail(map_out, "Unrecognized \"profile\" specified, should be \"fast\", \"lc\", \"np\", \"high\" or \"vn\"");
    const def = filter.Defaults.init(profile, wiener, temporal);

    var sigma: [3]f64 = @splat(10.0);
    // the C++ plugin ignores values past the third; vszip rejects them like
    // every other array argument
    hz.checkArrayLen(map_in, "sigma", 3, name, map_out) catch return;
    const ns = map_in.numElements("sigma") orelse 0;
    for (0..ns) |i| {
        const sv = map_in.getValue2(f64, "sigma", i).?;
        if (!math.isFinite(sv) or sv < 0) return fail(map_out, "Invalid \"sigma\" assigned, must be a non-negative floating point number");
        sigma[i] = sv;
    }
    if (ns > 0) for (ns..3) |i| {
        sigma[i] = sigma[i - 1];
    };

    // V-BM3D: radius before block_size (VBM3D_Data_Base order)
    var radius: i64 = 0;
    if (temporal) {
        radius = map_in.getValue(i64, "radius") orelse def.radius;
        if (radius < 1 or radius > 16) return fail(map_out, "Invalid \"radius\" assigned, must be an integer in [1, 16]");
    }

    const block = map_in.getValue(i64, "block_size") orelse def.block_size;
    if (map_in.getValue(i64, "block_size") != null and (block < 1 or block > 64)) return fail(map_out, "Invalid \"block_size\" assigned, must be an integer in [1, 64]");
    if (block > vi.width or block > vi.height) return fail(map_out, "Invalid \"block_size\", must not exceed width or height of the frame");
    const step = map_in.getValue(i64, "block_step") orelse @min(def.block_step, block);
    if (step < 1 or step > block) return fail(map_out, "Invalid \"block_step\" assigned, must be an integer in [1, block_size]");
    const group = map_in.getValue(i64, "group_size") orelse def.group_size;
    if (group < 1 or group > 256) return fail(map_out, "Invalid \"group_size\" assigned, must be an integer in [1, 256]");
    const bm_range = map_in.getValue(i64, "bm_range") orelse def.bm_range;
    if (bm_range < 1) return fail(map_out, "Invalid \"bm_range\" assigned, must be a positive integer");
    const bm_step = map_in.getValue(i64, "bm_step") orelse def.bm_step;
    if (bm_step < 1 or bm_step > bm_range) return fail(map_out, "Invalid \"bm_step\" assigned, must be an integer in [1, bm_range]");

    var ps_num: i64 = 0;
    var ps_range: i64 = 0;
    var ps_step: i64 = 1;
    if (temporal) {
        ps_num = map_in.getValue(i64, "ps_num") orelse def.ps_num;
        if (ps_num < 1 or ps_num > group) return fail(map_out, "Invalid \"ps_num\" assigned, must be an integer in [1, group_size]");
        ps_range = map_in.getValue(i64, "ps_range") orelse def.ps_range;
        if (ps_range < 1) return fail(map_out, "Invalid \"ps_range\" assigned, must be a positive integer");
        ps_step = map_in.getValue(i64, "ps_step") orelse def.ps_step;
        if (ps_step < 1 or ps_step > ps_range) return fail(map_out, "Invalid \"ps_step\" assigned, must be an integer in [1, ps_range]");
    }

    var th_mse = filter.Defaults.thMse(profile, wiener, sigma[0]);
    if (map_in.getValue(f64, "th_mse")) |t| {
        if (!math.isFinite(t) or t < 0) return fail(map_out, "Invalid \"th_mse\" assigned, must be a non-negative floating point number");
        th_mse = t;
    }

    var lambda = def.lambda;
    if (!wiener) {
        if (map_in.getValue(f64, "hard_thr")) |t| {
            if (!math.isFinite(t) or t <= 0) return fail(map_out, "Invalid \"hard_thr\" assigned, must be a positive floating point number");
            lambda = t;
        }
    }

    // matrix
    const mval = map_in.getValue(i64, "matrix");
    if (d.kind == .rgb) {
        d.matrix = .opp;
    } else if (mval == null or mval.? == 2) {
        d.matrix = filter.Matrix.default(vi.width, vi.height);
    } else {
        d.matrix = filter.Matrix.fromInt(mval.?) orelse return fail(map_out, "Unsupported \"matrix\" specified");
        if (d.matrix == .unspecified) d.matrix = filter.Matrix.default(vi.width, vi.height);
    }

    // processed planes
    var process: [3]bool = @splat(true);
    const np: usize = @intCast(fmt.numPlanes);
    for (0..3) |i| {
        if (i >= np or (d.kind != .rgb and sigma[i] == 0)) process[i] = false;
    }
    if (process[1] or process[2]) {
        if (fmt.subSamplingH != 0 or fmt.subSamplingW != 0) {
            return fail(map_out, "input clip: sub-sampled format is not supported when chroma is processed, convert it to YUV444 or RGB first. For the best quality, RGB colorspace is recommended as input.");
        }
    }
    d.skip = !(process[0] or process[1] or process[2]);
    if (temporal and d.skip) return fail(map_out, "Invalid \"sigma\" assigned, no plane of the input clip would be processed");
    if (temporal and @as(i64, vi.height) * (radius * 2 + 1) * 2 > math.maxInt(i32)) {
        return fail(map_out, "Invalid input clip, its height multiplied by (radius * 2 + 1) * 2 must not exceed 2147483647");
    }

    // filter data
    const norms = d.matrix.norms();
    th_mse *= norms[0];
    const tw = allocator.create(filter.Twiddles) catch return fail(map_out, "failed to allocate memory");
    tw.init(@intCast(@max(group, block)));
    d.tw = tw;
    // chroma planes are filtered jointly with plane 0's matching
    const joint = d.kind == .rgb or process[1] or process[2];
    d.p = .{
        .wiener = wiener,
        .n_planes = if (joint) 3 else 1,
        .process = process,
        .zero_sigma = .{ sigma[0] == 0, sigma[1] == 0, sigma[2] == 0 },
        .width = @intCast(vi.width),
        .height = @intCast(vi.height),
        .block = @intCast(block),
        .step = @intCast(step),
        .group = @intCast(group),
        .bm_range = @intCast(@min(bm_range, 1 << 20)),
        .bm_step = @intCast(bm_step),
        .th_mse = th_mse,
        .radius = @intCast(radius),
        .ps_num = @intCast(ps_num),
        .ps_range = @intCast(@min(ps_range, 1 << 20)),
        .ps_step = @intCast(ps_step),
        .f = undefined,
        .tw = tw,
    };
    for (0..3) |i| {
        if (process[i]) d.p.f[i] = filter.FilterData.init(wiener, sigma[i] / @as(f64, 255) * norms[i], @intCast(group), @intCast(block), lambda);
    }

    if (temporal) {
        var info: vs.CoreInfo = undefined;
        zapi.getCoreInfo(core, &info);
        const threads: u32 = @intCast(@max(info.numThreads, 1));
        const wpl: u32 = @intCast(vi.width);
        const hpl: u32 = @intCast(vi.height);
        var pw: [3]u32 = @splat(0);
        var ph: [3]u32 = @splat(0);
        for (0..3) |i| {
            if (process[i]) {
                pw[i] = wpl;
                ph[i] = hpl;
            }
        }
        d.cache = bc.TemporalCache.init(.classic, @intCast(radius), vi.numFrames, process, pw, ph, threads);
        // both clips: the frames in flight plus one window around them
        d.planes = bc.PlaneCache.init(3 * @as(usize, wpl) * hpl, 2 * (threads + 2 * @as(u32, @intCast(radius)) + 2));
    }

    const data = allocator.create(Data) catch {
        allocator.destroy(tw);
        return fail(map_out, "failed to allocate memory");
    };
    data.* = d;
    data.pool_lock = .init;
    data.pool = std.AutoHashMap(std.Thread.Id, *Scratch).init(allocator);
    keep = true;

    const gf: vs.FilterGetFrame = if (temporal)
        (if (wiener) &Temporal(true, name).getFrame else &Temporal(false, name).getFrame)
    else
        (if (wiener) &Spatial(true, name).getFrame else &Spatial(false, name).getFrame);
    const rp: vs.RequestPattern = if (temporal) .General else .StrictSpatial;
    var deps: [2]vs.FilterDependency = .{
        .{ .source = d.node, .requestPattern = rp },
        .{ .source = d.ref_node, .requestPattern = rp },
    };
    zapi.createVideoFilter(out, name, d.vi, gf, classicFree, .Parallel, deps[0..if (rdef) 2 else 1], data);
}

// ---------------------------------------------------------------------------
// BM3DVAggregate / RGB2OPP / OPP2RGB

/// Aggregates a stacked clip made by `bm3d.VBasic` / `bm3d.VFinal`
/// (VapourSynth-BM3D's VAggregate: same checks, same summation order).
pub const VAggregate = struct {
    pub const filter_name = "BM3DVAggregate";
    pub const params = "input:vnode;radius:int:opt;sample:int:opt;";

    const AggData = struct {
        node: ?*vs.Node = null,
        vi: *const vs.VideoInfo = undefined,
        vi_out: vs.VideoInfo = undefined,
        radius: i32 = 1,
        float_out: bool = false,
    };

    fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
        const d: *AggData = @ptrCast(@alignCast(instance_data));
        const zapi = ZAPI.init(vsapi, core, frame_ctx);
        const r = d.radius;
        const nf = d.vi.numFrames;
        const b = -@min(n, r);
        const f = @min(nf - 1 - n, r);
        if (activation_reason == .Initial) {
            var i = n + b;
            while (i <= n + f) : (i += 1) zapi.requestFrameFilter(i, d.node);
            return null;
        }
        if (activation_reason != .AllFramesReady) return null;

        const frames: usize = @intCast(f - b + 1);
        var src: [2 * filter.max_radius + 1]ZF = undefined;
        for (0..frames) |i| src[i] = zapi.initZFrame(d.node, n + b + @as(i32, @intCast(i)));
        defer for (src[0..frames]) |s| s.deinit();
        const cur = src[@intCast(-b)];
        const props = cur.getPropertiesRO();

        if (props.getInt(i64, "BM3D_V_radius")) |pr| {
            if (pr != r) {
                var buf: [256]u8 = undefined;
                const msg = std.mem.printSentinel(&buf, "BM3DVAggregate: argument \"radius={d}\" mismatches the input frame property \"BM3D_V_radius={d}\" which indicates the radius used in bm3d.VBasic or bm3d.VFinal", .{ r, pr }, 0) catch "BM3DVAggregate: radius mismatch";
                zapi.setFilterError(msg);
                return null;
            }
        }
        var process: [3]bool = @splat(true);
        if (props.getIntArray("BM3D_V_process")) |arr| {
            if (arr.len == 3) {
                for (0..3) |i| process[i] = arr[i] != 0;
            }
        }
        const opp = props.getInt(i64, "BM3D_OPP");
        const range = props.getInt(i64, "_Range");
        const full = (opp != null and opp.? == 1) or (range != null and range.? != 0);

        const dst = zapi.initZFrameFromVi(&d.vi_out, cur.frame);
        const np: usize = @intCast(d.vi.format.numPlanes);
        for (0..np) |p| {
            const w, const h_in, const sstride = cur.getDimensions2(f32, p);
            const h = h_in / @as(u32, @intCast((2 * r + 1) * 2));
            if (!process[p]) {
                // the plugin leaves these uninitialized; zero is deterministic
                @memset(dst.getWriteSlice(p), 0);
                continue;
            }
            var nums: [2 * filter.max_radius + 1][*]const f32 = undefined;
            var dens: [2 * filter.max_radius + 1][*]const f32 = undefined;
            var o: i32 = r - b;
            for (0..frames) |i| {
                const sp = src[i].getReadSlice2(f32, p).ptr;
                const po: usize = @as(usize, h) * sstride;
                nums[i] = sp + po * @as(usize, @intCast(o * 2));
                dens[i] = sp + po * @as(usize, @intCast(o * 2 + 1));
                o -= 1;
            }
            var row: [16384]f32 = undefined;
            for (0..h) |y| {
                const out_row = row[0..w];
                for (0..w) |x| {
                    var num: f32 = 0;
                    var den: f32 = 0;
                    for (0..frames) |i| {
                        num += nums[i][y * sstride + x];
                        den += dens[i][y * sstride + x];
                    }
                    out_row[x] = num / den;
                }
                if (d.float_out) {
                    @memcpy(dst.getWriteSlice2(f32, p)[y * dst.getStride2(f32, p) ..][0..w], out_row);
                } else {
                    filter.floatToInt(u16, dst.getWriteSlice2(u16, p)[y * dst.getStride2(u16, p) ..].ptr, 0, out_row.ptr, 0, w, 1, 16, p > 0, full);
                }
            }
        }
        const dprops = dst.getPropertiesRW();
        dprops.deleteKey("BM3D_V_radius");
        dprops.deleteKey("BM3D_V_process");
        return dst.frame;
    }

    fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
        const d: *AggData = @ptrCast(@alignCast(instance_data));
        ZAPI.init(vsapi, core, null).freeNode(d.node);
        allocator.destroy(d);
    }

    pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
        const zapi = ZAPI.init(vsapi, core, null);
        const map_in = zapi.initZMap(in);
        const map_out = zapi.initZMap(out);
        var d: AggData = .{};
        d.node, d.vi = map_in.getNodeVi("input").?;
        var keep = false;
        defer if (!keep) zapi.freeNode(d.node);
        const name = filter_name;
        const vi = d.vi;
        if (!vapoursynth.vshelper.isConstantVideoFormat(vi)) return map_out.setError(name ++ ": Invalid input clip, only constant format input supported");
        if (vi.format.sampleType != .Float or vi.format.bitsPerSample != 32) return map_out.setError(name ++ ": Invalid input clip, only accept 32 bit float format clip from bm3d.VBasic or bm3d.VFinal");
        if (vi.format.colorFamily == .RGB) return map_out.setError(name ++ ": Invalid input clip, must be of Gray or YUV color family");
        const radius_arg = map_in.getValue(i64, "radius");
        if (radius_arg) |ra| {
            if (ra < 1 or ra > 16) return map_out.setError(name ++ ": Invalid \"radius\" assigned, must be an integer in [1, 16]");
        }
        // the stacked height cannot tell a wrong radius apart: check the
        // one VBasic/VFinal attached
        var err_buf: [1024]u8 = undefined;
        const f0 = zapi.getFrame(0, d.node, &err_buf) orelse return map_out.setError(name ++ ": failed to get the first frame of input clip");
        const pr = zapi.initZMap(zapi.getFramePropertiesRO(f0)).getInt(i64, "BM3D_V_radius");
        zapi.freeFrame(f0);
        var radius: i64 = 1;
        if (radius_arg) |ra| {
            if (pr != null and pr.? != ra) {
                var buf: [256]u8 = undefined;
                const msg = std.mem.printSentinel(&buf, name ++ ": argument \"radius={d}\" mismatches the input frame property \"BM3D_V_radius={d}\" which indicates the radius used in bm3d.VBasic or bm3d.VFinal", .{ ra, pr.? }, 0) catch name ++ ": radius mismatch";
                return map_out.setError(msg);
            }
            radius = ra;
        } else if (pr) |p| {
            if (p < 1 or p > 16) {
                var buf: [256]u8 = undefined;
                const msg = std.mem.printSentinel(&buf, name ++ ": Invalid input frame property \"BM3D_V_radius={d}\", must be an integer in [1, 16]", .{p}, 0) catch name ++ ": invalid BM3D_V_radius";
                return map_out.setError(msg);
            }
            radius = p;
        }
        const slots: i64 = (radius * 2 + 1) * 2;
        if (@rem(@as(i64, vi.height), slots << @intCast(vi.format.subSamplingH)) != 0) {
            return map_out.setError(name ++ ": Invalid input clip, its height is not a multiple of (radius * 2 + 1) * 2 times the vertical sub-sampling. Make sure \"radius\" matches the one used in bm3d.VBasic or bm3d.VFinal");
        }
        const sample = map_in.getValue(i64, "sample") orelse 0;
        if (sample != 0 and sample != 1) return map_out.setError(name ++ ": Invalid 'sample' assigned, must be 0 (integer sample type) or 1 (float sample type)");
        d.radius = @intCast(radius);
        d.float_out = sample == 1;
        d.vi_out = vi.*;
        d.vi_out.height = @intCast(@divExact(vi.height, slots));
        _ = zapi.queryVideoFormat(&d.vi_out.format, vi.format.colorFamily, if (d.float_out) .Float else .Integer, if (d.float_out) 32 else 16, vi.format.subSamplingW, vi.format.subSamplingH);

        const data = allocator.create(AggData) catch return map_out.setError(name ++ ": failed to allocate memory");
        data.* = d;
        keep = true;
        var deps = [_]vs.FilterDependency{.{ .source = d.node, .requestPattern = .General }};
        zapi.createVideoFilter(out, name, &data.vi_out, getFrame, free, .Parallel, &deps, data);
    }
};

/// RGB to opponent color space (bm3d.RGB2OPP): YUV444P16 or YUV444PS,
/// full range, BM3D_OPP=1.
pub const RGB2OPP = Opp(true);
/// Opponent color space back to RGB (bm3d.OPP2RGB): RGB48 or RGBS.
pub const OPP2RGB = Opp(false);

fn Opp(comptime forward: bool) type {
    return struct {
        pub const filter_name = if (forward) "RGB2OPP" else "OPP2RGB";
        pub const params = "input:vnode;sample:int:opt;";

        const OppData = struct {
            node: ?*vs.Node = null,
            vi: *const vs.VideoInfo = undefined,
            vi_out: vs.VideoInfo = undefined,
            float_out: bool = false,
        };

        fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            defer if (activation_reason == .AllFramesReady) { // TEMP-PROF
                _ = prof_n.fetchAdd(1, .monotonic); // TEMP-PROF
            }; // TEMP-PROF
            const d: *OppData = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;
            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            const props = src.getPropertiesRO();
            if (!forward) {
                const opp = props.getInt(i64, "BM3D_OPP");
                if (opp == null or opp.? != 1) {
                    zapi.setFilterError(filter_name ++ ": the input frame property \"BM3D_OPP=1\" is missing, only opponent color space input is supported");
                    return null;
                }
            }
            const range = props.getInt(i64, "_Range");
            // untagged RGB is full range; OPP input is always full range
            const full = if (forward) (range == null or range.? != 0) else true;
            const dst = zapi.initZFrameFromVi(&d.vi_out, src.frame);
            const w: u32 = @intCast(d.vi.width);
            const h: u32 = @intCast(d.vi.height);
            const bits: u32 = @intCast(d.vi.format.bitsPerSample);
            const float_in = d.vi.format.sampleType == .Float;
            var tk0 = rdtscP(); // TEMP-PROF
            for (0..2) |pass| { // TEMP-PROF
            switch (bits) {
                inline 8, 16, 32, 9, 10, 12, 14 => |bb| {
                    const S = if (bb == 32) f32 else if (bb == 8) u8 else u16;
                    const ss = src.getStride2(S, 0);
                    const sp = .{ src.getReadSlice2(S, 0).ptr, src.getReadSlice2(S, 1).ptr, src.getReadSlice2(S, 2).ptr };
                    if (bb == 32 and !float_in) unreachable;
                    if (forward) {
                        if (d.float_out) {
                            const ds = dst.getStride2(f32, 0);
                            filter.rgbToOpp(S, dst.getWriteSlice2(f32, 0).ptr, dst.getWriteSlice2(f32, 1).ptr, dst.getWriteSlice2(f32, 2).ptr, ds, sp[0], sp[1], sp[2], ss, w, h, bb, full);
                        } else {
                            filter.rgbToOpp16(S, dst.getWriteSlice2(u16, 0).ptr, dst.getWriteSlice2(u16, 1).ptr, dst.getWriteSlice2(u16, 2).ptr, dst.getStride2(u16, 0), sp[0], sp[1], sp[2], ss, w, h, bb, full);
                        }
                    } else {
                        if (d.float_out) {
                            filter.oppToRgbQ(S, f32, dst.getWriteSlice2(f32, 0).ptr, dst.getWriteSlice2(f32, 1).ptr, dst.getWriteSlice2(f32, 2).ptr, dst.getStride2(f32, 0), sp[0], sp[1], sp[2], ss, w, h, bb, 32, true);
                        } else {
                            filter.oppToRgbQ(S, u16, dst.getWriteSlice2(u16, 0).ptr, dst.getWriteSlice2(u16, 1).ptr, dst.getWriteSlice2(u16, 2).ptr, dst.getStride2(u16, 0), sp[0], sp[1], sp[2], ss, w, h, bb, 16, true);
                        }
                    }
                },
                else => unreachable,
            }
            const tk1 = rdtscP(); // TEMP-PROF
            _ = (if (pass == 0) prof_k else prof_a).fetchAdd(tk1 - tk0, .monotonic); // TEMP-PROF
            tk0 = tk1; // TEMP-PROF
            } // TEMP-PROF
            const dp = dst.getPropertiesRW();
            if (forward) {
                dp.setInt("_Matrix", 2, .Replace);
                dp.setInt("_Range", 1, .Replace);
                dp.setInt("BM3D_OPP", 1, .Replace);
            } else {
                dp.setInt("_Matrix", 0, .Replace);
                dp.setInt("_Range", 1, .Replace);
                dp.deleteKey("BM3D_OPP");
            }
            return dst.frame;
        }

        fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
            const d: *OppData = @ptrCast(@alignCast(instance_data));
            const pn = @max(prof_n.load(.monotonic), 1); // TEMP-PROF
            std.debug.print("TEMP-PROF frames {d} first pass {d} kcyc/frame, second (warm) pass {d} kcyc/frame\n", .{ pn, prof_k.load(.monotonic) / pn / 1000, prof_a.load(.monotonic) / pn / 1000 }); // TEMP-PROF
            ZAPI.init(vsapi, core, null).freeNode(d.node);
            allocator.destroy(d);
        }

        pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
            const zapi = ZAPI.init(vsapi, core, null);
            const map_in = zapi.initZMap(in);
            const map_out = zapi.initZMap(out);
            var d: OppData = .{};
            d.node, d.vi = map_in.getNodeVi("input").?;
            var keep = false;
            defer if (!keep) zapi.freeNode(d.node);
            const vi = d.vi;
            const fmt = vi.format;
            if (!vapoursynth.vshelper.isConstantVideoFormat(vi)) return map_out.setError(filter_name ++ ": Invalid input clip, only constant format input supported");
            if ((fmt.sampleType == .Integer and fmt.bitsPerSample > 16) or (fmt.sampleType == .Float and fmt.bitsPerSample != 32)) {
                return map_out.setError(filter_name ++ ": Invalid input clip, only 8-16 bit int or 32 bit float formats supported");
            }
            if (forward) {
                if (fmt.colorFamily != .RGB) return map_out.setError(filter_name ++ ": Invalid input clip, must be of RGB color family");
            } else {
                if (fmt.colorFamily != .YUV) return map_out.setError(filter_name ++ ": Invalid input clip, must be of YUV color family");
                if (fmt.subSamplingW != 0 or fmt.subSamplingH != 0) return map_out.setError(filter_name ++ ": Invalid input clip, sub-sampled format is not supported");
            }
            const sample = map_in.getValue(i64, "sample") orelse 0;
            if (sample != 0 and sample != 1) return map_out.setError(filter_name ++ ": Invalid 'sample' assigned, must be 0 (integer sample type) or 1 (float sample type)");
            d.float_out = sample == 1;
            d.vi_out = vi.*;
            _ = zapi.queryVideoFormat(&d.vi_out.format, if (forward) .YUV else .RGB, if (d.float_out) .Float else .Integer, if (d.float_out) 32 else 16, 0, 0);
            const data = allocator.create(OppData) catch return map_out.setError(filter_name ++ ": failed to allocate memory");
            data.* = d;
            keep = true;
            var deps = [_]vs.FilterDependency{.{ .source = d.node, .requestPattern = .StrictSpatial }};
            zapi.createVideoFilter(out, filter_name, &data.vi_out, getFrame, free, .Parallel, &deps, data);
        }
    };
}

fn rdtscP() u64 { // TEMP-PROF
    var lo: u32 = undefined; // TEMP-PROF
    var hi: u32 = undefined; // TEMP-PROF
    asm volatile ("rdtsc" : [lo] "={eax}" (lo), [hi] "={edx}" (hi)); // TEMP-PROF
    return (@as(u64, hi) << 32) | lo; // TEMP-PROF
} // TEMP-PROF
var prof_k = std.atomic.Value(u64).init(0); // TEMP-PROF
var prof_a = std.atomic.Value(u64).init(0); // TEMP-PROF
var prof_n = std.atomic.Value(u64).init(0); // TEMP-PROF
