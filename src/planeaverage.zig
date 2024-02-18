const std = @import("std");
const vszip = @import("vszip.zig");
const helper = @import("helper.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const math = std.math;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const pe = vs.MapPropertyError;
const ma = vs.MapAppendMode;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneAverage";

const PlaneAverageData = struct {
    node: *vs.Node,
    node2: ?*vs.Node,
    exclude: Exclude,
    dt: helper.DataType,
    peak: f32,
    process: [3]bool,
};

const Exclude = union(enum) {
    FLOAT: []const f32,
    INT: []const i32,
};

const Stats = struct {
    avg: f64,
    diff: f64,
};

fn absDiff(comptime T: type, x: anytype, y: anytype) T {
    if (@typeInfo(T) == .Float) {
        return @abs(x - y);
    } else {
        return if (x > y) (x - y) else (y - x);
    }
}

fn average(comptime T: type, src: [*]const u8, _stride: usize, w: usize, h: usize, exclude_union: Exclude, peak: f32) f64 {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const exclude = if (@typeInfo(T) == .Float) exclude_union.FLOAT else exclude_union.INT;
    var total: i64 = @intCast(w * h);
    var acc: if (@typeInfo(T) == .Float) f64 else u64 = 0;

    for (0..h) |_| {
        for (srcp[0..w]) |x| {
            const found: bool = for (exclude) |e| {
                if (x == e) break true;
            } else false;

            if (found) {
                total -= 1;
            } else {
                acc += x;
            }
        }
        srcp += stride;
    }

    return result(T, acc, @floatFromInt(total), peak);
}

fn result(comptime T: type, acc: anytype, total: f64, peak: f32) f64 {
    if (total == 0) {
        return 0.0;
    } else if (@typeInfo(T) == .Float) {
        return acc / total;
    } else {
        return @as(f64, @floatFromInt(acc)) / total / peak;
    }
}

fn average_ref(comptime T: type, src: [*]const u8, ref: [*]const u8, _stride: usize, w: usize, h: usize, exclude_union: Exclude, peak: f32) Stats {
    var srcp: [*]const T = @ptrCast(@alignCast(src));
    var refp: [*]const T = @ptrCast(@alignCast(ref));
    const stride: usize = @divTrunc(_stride, @sizeOf(T));
    const exclude = if (@typeInfo(T) == .Float) exclude_union.FLOAT else exclude_union.INT;
    const _total: i64 = @intCast(w * h);
    var total = _total;
    const T2 = if (@typeInfo(T) == .Float) f64 else u64;
    var acc: T2 = 0;
    var diffacc: T2 = 0;

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            const found: bool = for (exclude) |e| {
                if (v == e) break true;
            } else false;

            if (found) {
                total -= 1;
            } else {
                acc += v;
            }

            diffacc += absDiff(T, v, j);
        }
        srcp += stride;
        refp += stride;
    }

    const _totalf: f64 = @floatFromInt(_total);
    return .{
        .avg = result(T, acc, @floatFromInt(total), peak),
        .diff = if (@typeInfo(T) == .Float) (diffacc / _totalf) else @as(f64, @floatFromInt(diffacc)) / _totalf / peak,
    };
}

export fn planeAverageGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *PlaneAverageData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
        if (d.node2) |node| {
            vsapi.?.requestFrameFilter.?(n, node, frame_ctx);
        }
    } else if (activation_reason == ar.AllFramesReady) {
        const src = vsapi.?.getFrameFilter.?(n, d.node, frame_ctx);
        defer vsapi.?.freeFrame.?(src);
        var ref: ?*const vs.Frame = null;
        if (d.node2) |node| {
            ref = vsapi.?.getFrameFilter.?(n, node, frame_ctx);
            defer vsapi.?.freeFrame.?(ref);
        }

        const fi = vsapi.?.getVideoFrameFormat.?(src);
        const dst = vsapi.?.copyFrame.?(src, core).?;
        const props = vsapi.?.getFramePropertiesRW.?(dst);

        var plane: c_int = 0;
        while (plane < fi.numPlanes) : (plane += 1) {
            if (!(d.process[@intCast(plane)])) {
                continue;
            }

            const srcp: [*]const u8 = vsapi.?.getReadPtr.?(src, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: usize = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: usize = @intCast(vsapi.?.getFrameWidth.?(src, plane));
            var avg: f64 = undefined;

            if (ref == null) {
                avg = switch (d.dt) {
                    .U8 => average(u8, srcp, stride, w, h, d.exclude, d.peak),
                    .U16 => average(u16, srcp, stride, w, h, d.exclude, d.peak),
                    .F32 => average(f32, srcp, stride, w, h, d.exclude, d.peak),
                };
            } else {
                const refp: [*]const u8 = vsapi.?.getReadPtr.?(ref, plane);
                const stats = switch (d.dt) {
                    .U8 => average_ref(u8, srcp, refp, stride, w, h, d.exclude, d.peak),
                    .U16 => average_ref(u16, srcp, refp, stride, w, h, d.exclude, d.peak),
                    .F32 => average_ref(f32, srcp, refp, stride, w, h, d.exclude, d.peak),
                };
                _ = vsapi.?.mapSetFloat.?(props, "psmDiff", stats.diff, ma.Append);
                avg = stats.avg;
            }
            _ = vsapi.?.mapSetFloat.?(props, "psmAvg", avg, ma.Append);
        }

        return dst;
    }

    return null;
}

export fn planeAverageFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *PlaneAverageData = @ptrCast(@alignCast(instance_data));
    switch (d.exclude) {
        .INT => allocator.free(d.exclude.INT),
        .FLOAT => allocator.free(d.exclude.FLOAT),
    }

    if (d.node2) |node| {
        vsapi.?.freeNode.?(node);
    }

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn planeAverageCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: PlaneAverageData = undefined;
    var err: pe = undefined;

    d.node = vsapi.?.mapGetNode.?(in, "clipa", 0, &err).?;
    d.node2 = vsapi.?.mapGetNode.?(in, "clipb", 0, &err);
    helper.compareNodes(out, d.node, d.node2, filter_name, vsapi) catch return;

    const vi = vsapi.?.getVideoInfo.?(d.node);
    d.dt = @enumFromInt(vi.format.bytesPerSample);
    d.peak = @floatFromInt(math.shl(i32, 1, vi.format.bitsPerSample) - 1);
    var nodes = [_]?*vs.Node{ d.node, d.node2 };
    var process = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &process, vi.format.numPlanes, filter_name, vsapi) catch return;
    d.process = process;

    const ne: usize = @intCast(vsapi.?.mapNumElements.?(in, "exclude"));
    const exclude_in = vsapi.?.mapGetIntArray.?(in, "exclude", &err);

    if (d.dt == .F32) {
        const buff = allocator.alloc(f32, ne) catch unreachable;
        for (0..ne) |i| {
            buff[i] = @floatFromInt(exclude_in[i]);
        }

        d.exclude = Exclude{ .FLOAT = buff };
    } else {
        const buff = allocator.alloc(i32, ne) catch unreachable;
        for (0..ne) |i| {
            buff[i] = math.lossyCast(i32, exclude_in[i]);
        }

        d.exclude = Exclude{ .INT = buff };
    }

    const data: *PlaneAverageData = allocator.create(PlaneAverageData) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = rp.StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, vi, planeAverageGetFrame, planeAverageFree, fm.Parallel, &deps, deps.len, data, core);
}
