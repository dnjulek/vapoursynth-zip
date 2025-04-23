const std = @import("std");
const math = std.math;

const filter = @import("../filters/limiter.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const BPSType = hz.BPSType;

const allocator = std.heap.c_allocator;
pub const filter_name = "Limiter";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    max: [3]u32 = .{ 0, 0, 0 },
    min: [3]u32 = .{ 0, 0, 0 },
    maxf: [3]f32 = .{ 0, 0, 0 },
    minf: [3]f32 = .{ 0, 0, 0 },
};

pub fn LimiterRT(comptime T: type, np: comptime_int, idx: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n, frame_ctx, core);
                defer src.deinit();
                const dst = src.newVideoFrame();

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
                    if (!(comptime_planes[idx][plane])) {
                        continue;
                    }

                    const max: T = if (@typeInfo(T) == .int) @intCast(d.max[plane]) else @floatCast(d.maxf[plane]);
                    const min: T = if (@typeInfo(T) == .int) @intCast(d.min[plane]) else @floatCast(d.minf[plane]);

                    for (
                        src.getReadSlice2(T, plane),
                        dst.getWriteSlice2(T, plane),
                    ) |*srcp, *dstp| {
                        dstp.* = @min(@max(min, srcp.*), max);
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

pub fn Limiter(comptime T: type, rng: anytype, np: comptime_int, idx: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n, frame_ctx, core);
                defer src.deinit();
                const dst = src.newVideoFrame();

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
                    if (!(comptime_planes[idx][plane])) {
                        continue;
                    }

                    for (
                        src.getReadSlice2(T, plane),
                        dst.getWriteSlice2(T, plane),
                    ) |*srcp, *dstp| {
                        dstp.* = @min(@max(rng[0][plane], srcp.*), rng[1][plane]);
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

export fn limiterFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub export fn limiterCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip");

    const min_in = map_in.getFloatArray("min");
    const max_in = map_in.getFloatArray("max");

    const num_planes = d.vi.format.numPlanes;
    const peak = hz.getPeak(d.vi);

    var planes: [3]bool = .{ true, true, true };
    const nodes = [_]?*vs.Node{d.node};
    hz.mapGetPlanes(map_in, map_out, &nodes, &planes, num_planes, filter_name, &zapi) catch return;

    var has_min = false;
    if (min_in) |arr| {
        has_min = true;
        if (arr.len != num_planes) {
            map_out.setError(filter_name ++ ": min array must have the same number of elements as planes.");
            zapi.freeNode(d.node);
            return;
        }

        for (0..arr.len) |i| {
            if (d.vi.format.sampleType == .Integer) {
                const val: i64 = @intFromFloat(arr[i]);

                if (val < 0) {
                    map_out.setError(filter_name ++ ": min value must be greater than or equal to 0.");
                    zapi.freeNode(d.node);
                    return;
                }

                d.min[i] = @intCast(val);
            } else {
                d.minf[i] = @floatCast(arr[i]);
            }
        }
    }

    var has_max = false;
    if (max_in) |arr| {
        has_max = true;
        if (arr.len != num_planes) {
            map_out.setError(filter_name ++ ": max array must have the same number of elements as planes.");
            zapi.freeNode(d.node);
            return;
        }

        for (0..arr.len) |i| {
            if (d.vi.format.sampleType == .Integer) {
                const val: i64 = @intFromFloat(arr[i]);

                if (val > peak) {
                    map_out.setError(filter_name ++ ": max value must be less than or equal to peak value.");
                    zapi.freeNode(d.node);
                    return;
                }

                d.max[i] = @intCast(val);
            } else {
                d.maxf[i] = @floatCast(arr[i]);
            }
        }
    }

    if (has_min and !has_max) {
        map_out.setError(filter_name ++ ": min array is set but max array is not.");
        zapi.freeNode(d.node);
        return;
    }

    if (!has_min and has_max) {
        map_out.setError(filter_name ++ ": max array is set but min array is not.");
        zapi.freeNode(d.node);
        return;
    }

    const bps = BPSType.select(map_out, d.node, d.vi, filter_name) catch return;

    var i: u32 = 0;
    const idx: u32 = while (i < comptime_planes.len) : (i += 1) {
        if (std.mem.eql(bool, &comptime_planes[i], &planes)) break i;
    } else 0;

    const get_frame: vs.FilterGetFrame = filter.getFrame(
        (has_max or has_min),
        map_in.getBool("tv_range") orelse false,
        d.vi.format.colorFamily == .YUV,
        num_planes,
        bps,
        idx,
    );

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, d.vi, get_frame, limiterFree, .Parallel, &deps, data, core);
}

pub const comptime_planes: [8][3]bool = .{
    .{ true, false, false },
    .{ false, true, false },
    .{ false, false, true },

    .{ false, true, true },
    .{ true, false, true },
    .{ true, true, false },

    .{ true, true, true },
    .{ false, false, false },
};
