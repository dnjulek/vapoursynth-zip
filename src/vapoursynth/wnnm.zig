const std = @import("std");
const math = std.math;

const filter = @import("../filters/wnnm.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "WNNM";

const max_radius = 15;
const max_window = 2 * max_radius + 1;

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    vi_raw: vs.VideoInfo = undefined,

    sigma: [3]f32 = undefined,
    process: [3]bool = undefined,
    bs: u32 = 8,
    step: u32 = 8,
    gs: u32 = 8,
    bm_range: u32 = 7,
    radius: u32 = 0,
    ps_num: u32 = 2,
    ps_range: u32 = 4,
    residual: bool = false,
    adaptive: bool = true,

    lda: u32 = 64,
    q: u32 = 8,
    max_iw: u32 = 0,
    max_ih: u32 = 0,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *Bufs) = undefined,
};

const Bufs = struct {
    arena: std.heap.ArenaAllocator,
    cur: []f32,
    amat: [filter.batch_n][]f32,
    aout: [filter.batch_n][]f32,
    mean: [filter.batch_n][]f32,
    sel: []filter.Match,
    selidx: []filter.IdxErr,
    gbatch: []f64,
    vbatch: []f64,
    intermediate: []f32,
    est: filter.Workspace(f64),
    errsf: std.ArrayList(f32),
    errors: std.ArrayList(filter.Match),
    center: std.ArrayList(filter.Match),
    temporal: std.ArrayList(filter.Match),
    locs: std.ArrayList(filter.Loc),
    locs_scratch: std.ArrayList(filter.Loc),
    alloc: std.mem.Allocator,

    fn create(d: *const Data) *Bufs {
        const b = allocator.create(Bufs) catch unreachable;
        b.arena = std.heap.ArenaAllocator.init(allocator);
        const aa = b.arena.allocator();
        const al = comptime std.mem.Alignment.fromByteUnits(64);
        const m = d.bs * d.bs;
        const q = d.q;
        const nb = filter.batch_n;
        for (0..nb) |l| {
            b.amat[l] = aa.alignedAlloc(f32, al, d.lda * d.gs) catch unreachable;
            b.aout[l] = aa.alignedAlloc(f32, al, d.lda * d.gs) catch unreachable;
            b.mean[l] = if (d.residual) aa.alignedAlloc(f32, al, m) catch unreachable else &.{};
        }
        b.cur = aa.alignedAlloc(f32, al, m) catch unreachable;
        b.sel = aa.alloc(filter.Match, d.gs * nb) catch unreachable;
        b.selidx = aa.alloc(filter.IdxErr, @max(d.gs, d.ps_num)) catch unreachable;
        b.gbatch = aa.alloc(f64, q * q * nb) catch unreachable;
        b.vbatch = aa.alloc(f64, q * q * nb) catch unreachable;
        b.intermediate = if (d.radius == 0) aa.alignedAlloc(f32, al, 2 * d.max_iw * d.max_ih) catch unreachable else &.{};
        b.est = .{
            .g = aa.alloc(f64, q * q) catch unreachable,
            .v = aa.alloc(f64, q * q) catch unreachable,
            .eig = aa.alloc(f64, q) catch unreachable,
            .r = aa.alloc(f32, q) catch unreachable,
            .scratch = aa.alignedAlloc(f32, al, m * d.gs + q * q) catch unreachable,
            .wide = aa.alignedAlloc(f64, al, m * d.gs) catch unreachable,
        };
        b.errsf = .empty;
        b.errors = .empty;
        b.center = .empty;
        b.temporal = .empty;
        b.locs = .empty;
        b.locs_scratch = .empty;
        b.alloc = aa;
        return b;
    }

    fn destroy(b: *Bufs) void {
        b.arena.deinit();
        allocator.destroy(b);
    }
};

fn acquireBufs(d: *Data) *Bufs {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |b| return b;
    const b = Bufs.create(d);
    d.pool.put(tid, b) catch unreachable;
    return b;
}

fn processPlane(
    comptime BS: ?u32,
    d: *const Data,
    b: *Bufs,
    srcps: []const []const f32,
    refps: []const []const f32,
    agg: [*]f32,
    agg_stride: u32,
    w: u32,
    h: u32,
    stride: u32,
    sigma: f32,
) void {
    const bs = BS orelse d.bs;
    const m = bs * bs;
    const radius: i32 = @intCast(d.radius);
    const center: usize = d.radius;
    const k0: u32 = if (d.residual) 0 else 1;

    const temp_r = h - bs;
    const temp_c = w - bs;

    const gsn: f32 = @floatFromInt(d.gs);
    const c_full: f32 = 8.0 * @sqrt(2.0 * gsn) * sigma * sigma;
    var slot: usize = 0;

    var _y: u32 = 0;
    while (_y < temp_r + d.step) : (_y += d.step) {
        const y = @min(_y, temp_r);
        var _x: u32 = 0;
        while (_x < temp_c + d.step) : (_x += d.step) {
            const x = @min(_x, temp_c);

            b.errors.clearRetainingCapacity();

            {
                const refp = refps[center];
                var py: u32 = 0;
                while (py < bs) : (py += 1) {
                    @memcpy(b.cur[py * bs ..][0..bs], refp[(y + py) * stride + x ..][0..bs]);
                }
            }

            const top = y -| d.bm_range;
            const bottom = @min(y + d.bm_range, temp_r);
            const left = x -| d.bm_range;
            const right = @min(x + d.bm_range, temp_c);
            const cx = right - left + 1;

            if (d.radius == 0) {
                b.errsf.clearRetainingCapacity();
                filter.computeBlockDistancesWindow(BS, &b.errsf, b.alloc, b.cur, refps[center], top, bottom, left, right, stride, bs);
                const cnt = filter.selectTopIdx(b.errsf.items, d.gs, b.selidx);
                b.errors.ensureUnusedCapacity(b.alloc, cnt) catch unreachable;
                for (b.selidx[0..cnt]) |ie| {
                    b.errors.appendAssumeCapacity(.{
                        .err = ie.err,
                        .x = @intCast(left + ie.idx % cx),
                        .y = @intCast(top + ie.idx / cx),
                        .t = 0,
                    });
                }
            } else {
                b.errsf.clearRetainingCapacity();
                filter.computeBlockDistancesWindow(BS, &b.errsf, b.alloc, b.cur, refps[center], top, bottom, left, right, stride, bs);
                const active_ps_num = @min(d.ps_num, b.errsf.items.len);
                const cnt_c = filter.selectTopIdx(b.errsf.items, @max(d.gs, d.ps_num), b.selidx);
                b.center.clearRetainingCapacity();
                b.center.ensureUnusedCapacity(b.alloc, cnt_c) catch unreachable;
                for (b.selidx[0..cnt_c]) |ie| {
                    b.center.appendAssumeCapacity(.{
                        .err = ie.err,
                        .x = @intCast(left + ie.idx % cx),
                        .y = @intCast(top + ie.idx / cx),
                        .t = radius,
                    });
                }
                b.errors.appendSlice(b.alloc, b.center.items) catch unreachable;

                var direction: i32 = -1;
                while (direction <= 1) : (direction += 2) {
                    b.temporal.clearRetainingCapacity();
                    b.temporal.appendSlice(b.alloc, b.center.items) catch unreachable;

                    var i: u32 = 1;
                    while (i <= d.radius) : (i += 1) {
                        const t_idx = radius + direction * @as(i32, @intCast(i));

                        const num_centers = @min(active_ps_num, b.temporal.items.len);
                        filter.generateSearchLocations(b.temporal.items, num_centers, bs, w, h, d.ps_range, &b.locs, &b.locs_scratch, b.alloc);

                        b.errsf.clearRetainingCapacity();
                        filter.computeBlockDistancesAt(BS, &b.errsf, b.alloc, b.cur, refps[@intCast(t_idx)], b.locs.items, stride, bs);
                        const cnt_t = filter.selectTopIdx(b.errsf.items, @max(d.gs, d.ps_num), b.selidx);
                        b.temporal.clearRetainingCapacity();
                        b.temporal.ensureUnusedCapacity(b.alloc, cnt_t) catch unreachable;
                        for (b.selidx[0..cnt_t]) |ie| {
                            b.temporal.appendAssumeCapacity(.{
                                .err = ie.err,
                                .x = b.locs.items[ie.idx].x,
                                .y = b.locs.items[ie.idx].y,
                                .t = t_idx,
                            });
                        }
                        b.errors.appendSlice(b.alloc, b.temporal.items) catch unreachable;
                    }
                }
            }

            const n: u32 = @intCast(filter.selectTop(b.errors.items, d.gs));
            b.errors.items.len = n;

            {
                var has_center = false;
                for (b.errors.items) |e| {
                    if (e.x == @as(i32, @intCast(x)) and e.y == @as(i32, @intCast(y)) and e.t == radius) {
                        has_center = true;
                        break;
                    }
                }
                if (!has_center) {
                    b.errors.items[0] = .{ .err = 0.0, .x = @intCast(x), .y = @intCast(y), .t = radius };
                }
            }

            if (n == d.gs) {
                filter.loadPatches(BS, b.amat[slot].ptr, d.lda, if (d.residual) b.mean[slot] else null, srcps, b.errors.items, stride, bs);
                if (d.residual) filter.subtractMean(b.amat[slot].ptr, d.lda, b.mean[slot], m, n);
                @memcpy(b.sel[slot * d.gs ..][0..n], b.errors.items);
                slot += 1;
                if (slot == filter.batch_n) {
                    flushBatch(BS, d, b, agg, agg_stride, h, bs, m, c_full, k0, slot);
                    slot = 0;
                }
            } else {
                flushBatch(BS, d, b, agg, agg_stride, h, bs, m, c_full, k0, slot);
                slot = 0;

                filter.loadPatches(BS, b.amat[0].ptr, d.lda, if (d.residual) b.mean[0] else null, srcps, b.errors.items, stride, bs);
                if (d.residual) filter.subtractMean(b.amat[0].ptr, d.lda, b.mean[0], m, n);

                const nf: f32 = @floatFromInt(n);
                const c: f32 = 8.0 * @sqrt(2.0 * nf) * sigma * sigma;
                const res = filter.patchEstimate(f64, b.amat[0], b.aout[0], m, n, d.lda, c, k0, b.est);
                const weight: f32 = if (d.adaptive and res.k > 0) 1.0 / @as(f32, @floatFromInt(res.k)) else 1.0;

                if (d.residual) filter.addMean(b.aout[0].ptr, d.lda, b.mean[0], m, n);

                filter.col2im(BS, agg, agg_stride, b.aout[0].ptr, d.lda, b.errors.items, h, bs, weight);
            }
        }
    }
    flushBatch(BS, d, b, agg, agg_stride, h, bs, m, c_full, k0, slot);
}

fn flushBatch(
    comptime BS: ?u32,
    d: *const Data,
    b: *Bufs,
    agg: [*]f32,
    agg_stride: u32,
    h: u32,
    bs: u32,
    m: u32,
    c: f32,
    k0: u32,
    count: usize,
) void {
    if (count == 0) return;
    const n = d.gs;

    var as: [filter.batch_n][]const f32 = undefined;
    var dsts: [filter.batch_n][]f32 = undefined;
    for (0..filter.batch_n) |l| {
        as[l] = b.amat[l];
        dsts[l] = b.aout[l];
    }

    const ks: [filter.batch_n]u32 = if (count == 1) blk: {
        const res = filter.patchEstimate(f64, b.amat[0], b.aout[0], m, n, d.lda, c, k0, b.est);
        var arr: [filter.batch_n]u32 = @splat(0);
        arr[0] = res.k;
        break :blk arr;
    } else filter.patchEstimateBatch(as, dsts, count, m, n, d.lda, c, k0, b.est, b.gbatch, b.vbatch);

    for (0..count) |l| {
        const weight: f32 = if (d.adaptive and ks[l] > 0) 1.0 / @as(f32, @floatFromInt(ks[l])) else 1.0;
        if (d.residual) filter.addMean(b.aout[l].ptr, d.lda, b.mean[l], m, n);
        filter.col2im(BS, agg, agg_stride, b.aout[l].ptr, d.lda, b.sel[l * d.gs ..][0..n], h, bs, weight);
    }
}

fn Wnnm(comptime BS: ?u32) type {
    return struct {
        pub fn getFrameSpatial(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (d.ref_node != null) zapi.requestFrameFilter(n, d.ref_node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            const ref = if (d.ref_node != null) zapi.initZFrame(d.ref_node, n) else src.addFrameRef();
            defer ref.deinit();
            const dst = src.newVideoFrame2(d.process);

            const bufs = acquireBufs(d);

            var plane: u32 = 0;
            while (plane < d.vi.format.numPlanes) : (plane += 1) {
                if (!d.process[plane]) continue;

                const w, const h, const stride = src.getDimensions2(f32, plane);
                const srcp = src.getReadSlice2(f32, plane);
                const refp = ref.getReadSlice2(f32, plane);
                const dstp = dst.getWriteSlice2(f32, plane);

                const intermediate = bufs.intermediate[0 .. 2 * h * w];
                @memset(intermediate, 0.0);

                processPlane(BS, d, bufs, &.{srcp}, &.{refp}, intermediate.ptr, w, w, h, stride, d.sigma[plane]);
                filter.aggregation(dstp, intermediate, w, h, stride);
            }

            return dst.frame;
        }

        pub fn getFrameRaw(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const radius: i32 = @intCast(d.radius);
            const num_frames = 2 * d.radius + 1;

            if (activation_reason == .Initial) {
                var i: i32 = -radius;
                while (i <= radius) : (i += 1) {
                    const fid = math.clamp(n + i, 0, d.vi.numFrames - 1);
                    zapi.requestFrameFilter(fid, d.node);
                    if (d.ref_node != null) zapi.requestFrameFilter(fid, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            comptime std.debug.assert(max_window >= 2 * max_radius + 1);
            var srcs: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
            var refs: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
            for (0..num_frames) |i| {
                const fid = math.clamp(n - radius + @as(i32, @intCast(i)), 0, d.vi.numFrames - 1);
                srcs[i] = zapi.initZFrame(d.node, fid);
                refs[i] = if (d.ref_node != null) zapi.initZFrame(d.ref_node, fid) else srcs[i].addFrameRef();
            }
            defer for (0..num_frames) |i| {
                srcs[i].deinit();
                refs[i].deinit();
            };

            const dst = zapi.initZFrameFromVi(&d.vi_raw, srcs[d.radius].frame);

            const bufs = acquireBufs(d);

            var srcps: [max_window][]const f32 = undefined;
            var refps: [max_window][]const f32 = undefined;

            var plane: u32 = 0;
            while (plane < d.vi.format.numPlanes) : (plane += 1) {
                if (!d.process[plane]) continue;

                const w, const h, const stride = srcs[d.radius].getDimensions2(f32, plane);
                for (0..num_frames) |i| {
                    srcps[i] = srcs[i].getReadSlice2(f32, plane);
                    refps[i] = refs[i].getReadSlice2(f32, plane);
                }
                const dstp = dst.getWriteSlice2(f32, plane);
                @memset(dstp[0 .. 2 * num_frames * h * stride], 0.0);

                processPlane(BS, d, bufs, srcps[0..num_frames], refps[0..num_frames], dstp.ptr, stride, w, h, stride, d.sigma[plane]);
            }

            return dst.frame;
        }
    };
}

fn wnnmFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    var it = d.pool.valueIterator();
    while (it.next()) |b| Bufs.destroy(b.*);
    d.pool.deinit();

    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

const VagData = struct {
    node: ?*vs.Node = null,
    src_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    radius: u32 = 0,
    process: [3]bool = undefined,
    max_w: u32 = 0,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, []f32) = undefined,
};

fn vagAcquireBuffer(d: *VagData) []f32 {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |b| return b;
    const b = allocator.alignedAlloc(f32, comptime std.mem.Alignment.fromByteUnits(64), 2 * d.max_w) catch unreachable;
    d.pool.put(tid, b) catch unreachable;
    return b;
}

fn vagGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *VagData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);
    const radius: i32 = @intCast(d.radius);
    const num_frames = 2 * d.radius + 1;

    if (activation_reason == .Initial) {
        var i: i32 = -radius;
        while (i <= radius) : (i += 1) {
            zapi.requestFrameFilter(math.clamp(n + i, 0, d.vi.numFrames - 1), d.node);
        }
        zapi.requestFrameFilter(n, d.src_node);
        return null;
    }
    if (activation_reason != .AllFramesReady) return null;

    const src = zapi.initZFrame(d.src_node, n);
    defer src.deinit();

    var raws: [max_window]@TypeOf(zapi.initZFrame(d.node, n)) = undefined;
    for (0..num_frames) |i| {
        const fid = math.clamp(n - radius + @as(i32, @intCast(i)), 0, d.vi.numFrames - 1);
        raws[i] = zapi.initZFrame(d.node, fid);
    }
    defer for (0..num_frames) |i| raws[i].deinit();

    const dst = src.newVideoFrame2(d.process);

    const buffer = vagAcquireBuffer(d);

    var plane: u32 = 0;
    while (plane < d.vi.format.numPlanes) : (plane += 1) {
        if (!d.process[plane]) continue;

        const w, const h, const stride = src.getDimensions2(f32, plane);
        const dstp = dst.getWriteSlice2(f32, plane);
        const num = buffer[0..w];
        const den = buffer[d.max_w..][0..w];

        var y: u32 = 0;
        while (y < h) : (y += 1) {
            @memset(num, 0.0);
            @memset(den, 0.0);
            for (0..num_frames) |i| {
                const raw = raws[i].getReadSlice2(f32, plane);

                const slab = math.clamp(
                    2 * radius - @as(i32, @intCast(i)),
                    n - d.vi.numFrames + 1 + radius,
                    n + radius,
                );
                const base = (@as(usize, @intCast(slab)) * 2 * h + y) * stride;
                const nrow = raw[base..][0..w];
                const drow = raw[base + h * stride ..][0..w];
                vagAccumulate(num, nrow);
                vagAccumulate(den, drow);
            }
            vagDivide(dstp[y * stride ..][0..w], num, den);
        }
    }

    return dst.frame;
}

const Vg = @Vector(8, f32);

fn vagAccumulate(acc: []f32, row: []const f32) void {
    var x: usize = 0;
    while (x + 8 <= acc.len) : (x += 8) {
        acc[x..][0..8].* = @as(Vg, acc[x..][0..8].*) + @as(Vg, row[x..][0..8].*);
    }
    while (x < acc.len) : (x += 1) acc[x] += row[x];
}

fn vagDivide(dst: []f32, num: []const f32, den: []const f32) void {
    var x: usize = 0;
    while (x + 8 <= dst.len) : (x += 8) {
        dst[x..][0..8].* = @as(Vg, num[x..][0..8].*) / @as(Vg, den[x..][0..8].*);
    }
    while (x < dst.len) : (x += 1) dst[x] = num[x] / den[x];
}

fn vagFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *VagData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    var it = d.pool.valueIterator();
    while (it.next()) |b| allocator.free(b.*);
    d.pool.deinit();

    zapi.freeNode(d.node);
    zapi.freeNode(d.src_node);
    allocator.destroy(d);
}

fn selectGetFrame(bs: u32, radius: u32) vs.FilterGetFrame {
    return switch (bs) {
        4 => if (radius == 0) &Wnnm(4).getFrameSpatial else &Wnnm(4).getFrameRaw,
        6 => if (radius == 0) &Wnnm(6).getFrameSpatial else &Wnnm(6).getFrameRaw,
        8 => if (radius == 0) &Wnnm(8).getFrameSpatial else &Wnnm(8).getFrameRaw,
        16 => if (radius == 0) &Wnnm(16).getFrameSpatial else &Wnnm(16).getFrameRaw,
        else => if (radius == 0) &Wnnm(null).getFrameSpatial else &Wnnm(null).getFrameRaw,
    };
}

pub fn wnnmCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;

    if (!vapoursynth.vshelper.isConstantVideoFormat(d.vi) or d.vi.format.sampleType != .Float or d.vi.format.bitsPerSample != 32) {
        map_out.setError(filter_name ++ ": only constant format 32 bit float input supported.");
        zapi.freeNode(d.node);
        return;
    }

    var raw_sigma: [3]f32 = undefined;
    for (0..3) |i| {
        raw_sigma[i] = map_in.getFloat2(f32, "sigma", i) orelse
            (if (i == 0) 3.0 else raw_sigma[i - 1]);
        if (raw_sigma[i] < 0.0) {
            map_out.setError(filter_name ++ ": \"sigma\" must be non-negative.");
            zapi.freeNode(d.node);
            return;
        }
    }
    for (0..3) |i| {
        d.process[i] = raw_sigma[i] >= math.floatEps(f32);
        d.sigma[i] = raw_sigma[i] / 255.0;
    }

    var skip = true;
    for (0..@intCast(d.vi.format.numPlanes)) |i| skip = skip and !d.process[i];
    if (skip) {
        _ = map_out.setNode("clip", d.node, .Replace);
        zapi.freeNode(d.node);
        return;
    }

    const err = struct {
        fn set(mo: anytype, za: *const ZAPI, dd: *Data, comptime msg: [:0]const u8) void {
            mo.setError(filter_name ++ ": " ++ msg);
            za.freeNode(dd.node);
            za.freeNode(dd.ref_node);
        }
    }.set;

    const bs_i = map_in.getValue(i32, "block_size") orelse 8;
    if (bs_i < 1 or bs_i > 64) return err(map_out, &zapi, &d, "\"block_size\" must be in [1, 64].");
    d.bs = @intCast(bs_i);

    const step_i = map_in.getValue(i32, "block_step") orelse bs_i;
    if (step_i < 1 or step_i > bs_i) return err(map_out, &zapi, &d, "\"block_step\" must be positive and no larger than \"block_size\".");
    d.step = @intCast(step_i);

    const gs_i = map_in.getValue(i32, "group_size") orelse 8;
    if (gs_i < 1 or gs_i > 256) return err(map_out, &zapi, &d, "\"group_size\" must be in [1, 256].");
    d.gs = @intCast(gs_i);
    if (@min(@as(u32, @intCast(bs_i * bs_i)), d.gs) > filter.max_q) {
        return err(map_out, &zapi, &d, "min(block_size^2, group_size) must be <= 64.");
    }

    const bm_i = map_in.getValue(i32, "bm_range") orelse 7;
    if (bm_i < 0) return err(map_out, &zapi, &d, "\"bm_range\" must be non-negative.");
    d.bm_range = @intCast(bm_i);

    const radius_i = map_in.getValue(i32, "radius") orelse 0;
    if (radius_i < 0 or radius_i > max_radius) return err(map_out, &zapi, &d, "\"radius\" must be in [0, 15].");
    d.radius = @intCast(radius_i);

    const ps_num_i = map_in.getValue(i32, "ps_num") orelse 2;
    if (ps_num_i < 1 or ps_num_i > 256) return err(map_out, &zapi, &d, "\"ps_num\" must be in [1, 256].");
    d.ps_num = @intCast(ps_num_i);

    const ps_range_i = map_in.getValue(i32, "ps_range") orelse 4;
    if (ps_range_i < 0) return err(map_out, &zapi, &d, "\"ps_range\" must be non-negative.");
    d.ps_range = @intCast(ps_range_i);

    d.residual = map_in.getBool("residual") orelse false;
    d.adaptive = map_in.getBool("adaptive_aggregation") orelse true;

    {
        const ssw: u5 = @intCast(d.vi.format.subSamplingW);
        const ssh: u5 = @intCast(d.vi.format.subSamplingH);
        for (0..@intCast(d.vi.format.numPlanes)) |i| {
            if (!d.process[i]) continue;
            const pw: u32 = if (i == 0) @intCast(d.vi.width) else @as(u32, @intCast(d.vi.width)) >> ssw;
            const ph: u32 = if (i == 0) @intCast(d.vi.height) else @as(u32, @intCast(d.vi.height)) >> ssh;
            if (d.bs > pw or d.bs > ph) {
                return err(map_out, &zapi, &d, "\"block_size\" must not exceed the dimensions of any processed plane.");
            }
            d.max_iw = @max(d.max_iw, pw);
            d.max_ih = @max(d.max_ih, ph);
        }
    }

    d.ref_node = map_in.getNode("rclip");
    if (d.ref_node != null) {
        const ref_vi = zapi.getVideoInfo(d.ref_node);
        if (!vapoursynth.vshelper.isSameVideoInfo(d.vi, ref_vi) or ref_vi.numFrames != d.vi.numFrames) {
            return err(map_out, &zapi, &d, "\"rclip\" must be of the same format, dimensions and number of frames as \"clip\".");
        }
    }

    const m = d.bs * d.bs;
    d.lda = hz.ceilN(m, 16);
    d.q = @min(m, d.gs);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    data.pool_lock = .init;
    data.pool = std.AutoHashMap(std.Thread.Id, *Bufs).init(allocator);

    if (d.radius == 0) {
        var deps_buf: [2]vs.FilterDependency = undefined;
        var n_deps: usize = 1;
        deps_buf[0] = .{ .source = d.node, .requestPattern = .StrictSpatial };
        if (d.ref_node != null) {
            deps_buf[1] = .{ .source = d.ref_node, .requestPattern = .StrictSpatial };
            n_deps = 2;
        }
        zapi.createVideoFilter(out, filter_name, d.vi, selectGetFrame(d.bs, 0), wnnmFree, .Parallel, deps_buf[0..n_deps], data);
        if (map_out.getError() != null) wnnmFree(data, core, vsapi);
        return;
    }

    data.vi_raw = d.vi.*;
    data.vi_raw.height *= @intCast(2 * (2 * d.radius + 1));

    var deps_buf: [2]vs.FilterDependency = undefined;
    var n_deps: usize = 1;
    deps_buf[0] = .{ .source = d.node, .requestPattern = .General };
    if (d.ref_node != null) {
        deps_buf[1] = .{ .source = d.ref_node, .requestPattern = .General };
        n_deps = 2;
    }

    const src_node2 = zapi.addNodeRef(d.node);

    const raw_node = zapi.createVideoFilter2("WNNMRaw", &data.vi_raw, selectGetFrame(d.bs, d.radius), wnnmFree, .Parallel, deps_buf[0..n_deps], data);
    if (raw_node == null) {
        map_out.setError(filter_name ++ ": failed to create internal WNNMRaw filter.");
        zapi.freeNode(src_node2);
        wnnmFree(data, core, vsapi);
        return;
    }

    const vag: *VagData = allocator.create(VagData) catch unreachable;
    vag.* = .{
        .node = raw_node,
        .src_node = src_node2,
        .vi = data.vi,
        .radius = d.radius,
        .process = d.process,
        .max_w = d.max_iw,
    };
    vag.pool_lock = .init;
    vag.pool = std.AutoHashMap(std.Thread.Id, []f32).init(allocator);

    const vag_deps = [_]vs.FilterDependency{
        .{ .source = raw_node, .requestPattern = .General },
        .{ .source = src_node2, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "WNNMVAggregate", data.vi, &vagGetFrame, vagFree, .Parallel, &vag_deps, vag);
    if (map_out.getError() != null) vagFree(vag, core, vsapi);
}
