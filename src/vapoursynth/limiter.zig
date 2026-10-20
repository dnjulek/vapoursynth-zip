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
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame2(comptime_planes[idx]);

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
                    if (!(comptime_planes[idx][plane])) continue;

                    const max: T = if (@typeInfo(T) == .int) @intCast(d.max[plane]) else @floatCast(d.maxf[plane]);
                    const min: T = if (@typeInfo(T) == .int) @intCast(d.min[plane]) else @floatCast(d.minf[plane]);

                    filter.clampSlice(T, dst.getWriteSlice2(T, plane), src.getReadSlice2(T, plane), min, max);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

pub fn Limiter(comptime T: type, rng: anytype, np: comptime_int, idx: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame2(comptime_planes[idx]);

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
                    if (!(comptime_planes[idx][plane])) continue;

                    filter.clampSlice(T, dst.getWriteSlice2(T, plane), src.getReadSlice2(T, plane), rng[0][plane], rng[1][plane]);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn selectGetFrame(use_rt: bool, tv_range: bool, yuv: bool, num_planes: i32, bps: hz.BPSType, idx: u32) vs.FilterGetFrame {
    var get_frame: vs.FilterGetFrame = undefined;
    if (use_rt) {
        get_frame = switch (num_planes) {
            inline 1...3 => |np| switch (idx) {
                inline 0...(comptime_planes.len - 1) => |i| switch (bps) {
                    .U8 => &LimiterRT(u8, np, i).getFrame,
                    .U9, .U10, .U12, .U14, .U16 => &LimiterRT(u16, np, i).getFrame,
                    .U32 => &LimiterRT(u32, np, i).getFrame,
                    .F16 => &LimiterRT(f16, np, i).getFrame,
                    .F32 => &LimiterRT(f32, np, i).getFrame,
                },
                else => unreachable,
            },
            else => unreachable,
        };
    } else {
        if (tv_range) {
            get_frame = switch (num_planes) {
                inline 1...3 => |np| switch (idx) {
                    inline 0...(comptime_planes.len - 1) => |i| switch (bps) {
                        .U8 => if (yuv) &Limiter(u8, filter.yuv8, np, i).getFrame else &Limiter(u8, filter.rgb8, np, i).getFrame,
                        .U9 => if (yuv) &Limiter(u16, filter.yuv9, np, i).getFrame else &Limiter(u16, filter.rgb9, np, i).getFrame,
                        .U10 => if (yuv) &Limiter(u16, filter.yuv10, np, i).getFrame else &Limiter(u16, filter.rgb10, np, i).getFrame,
                        .U12 => if (yuv) &Limiter(u16, filter.yuv12, np, i).getFrame else &Limiter(u16, filter.rgb12, np, i).getFrame,
                        .U14 => if (yuv) &Limiter(u16, filter.yuv14, np, i).getFrame else &Limiter(u16, filter.rgb14, np, i).getFrame,
                        .U16 => if (yuv) &Limiter(u16, filter.yuv16, np, i).getFrame else &Limiter(u16, filter.rgb16, np, i).getFrame,
                        .U32 => if (yuv) &Limiter(u32, filter.yuv32, np, i).getFrame else &Limiter(u32, filter.rgb32, np, i).getFrame,
                        .F16 => if (yuv) &Limiter(f16, filter.yuvf, np, i).getFrame else &Limiter(f16, filter.rgbf, np, i).getFrame,
                        .F32 => if (yuv) &Limiter(f32, filter.yuvf, np, i).getFrame else &Limiter(f32, filter.rgbf, np, i).getFrame,
                    },
                    else => unreachable,
                },
                else => unreachable,
            };
        } else {
            get_frame = switch (num_planes) {
                inline 1...3 => |np| switch (idx) {
                    inline 0...(comptime_planes.len - 1) => |i| switch (bps) {
                        .U8 => &Limiter(u8, filter.full8, np, i).getFrame,
                        .U9 => &Limiter(u16, filter.full9, np, i).getFrame,
                        .U10 => &Limiter(u16, filter.full10, np, i).getFrame,
                        .U12 => &Limiter(u16, filter.full12, np, i).getFrame,
                        .U14 => &Limiter(u16, filter.full14, np, i).getFrame,
                        .U16 => &Limiter(u16, filter.full16, np, i).getFrame,
                        .U32 => &Limiter(u32, filter.full32, np, i).getFrame,
                        .F16 => if (yuv) &Limiter(f16, filter.yuvf, np, i).getFrame else &Limiter(f16, filter.rgbf, np, i).getFrame,
                        .F32 => if (yuv) &Limiter(f32, filter.yuvf, np, i).getFrame else &Limiter(f32, filter.rgbf, np, i).getFrame,
                    },
                    else => unreachable,
                },
                else => unreachable,
            };
        }
    }

    return get_frame;
}

fn limiterFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

/// One bound array (`min` or `max`): one value per plane, every value finite.
/// Integer clips truncate toward zero and must land in [0, peak]; float clips
/// must lie in [-peak, peak] (peak = largest finite sample value) so narrowing
/// cannot produce +-inf. Every check runs in f64 *before* any conversion.
fn parseBounds(
    comptime which: []const u8,
    arr: []const f64,
    num_planes: c_int,
    int_clip: bool,
    peak: f64,
    out_int: *[3]u32,
    out_float: *[3]f32,
    map_out: anytype,
) error{invalidArgument}!void {
    if (arr.len != num_planes) {
        map_out.setError(filter_name ++ ": " ++ which ++ " array must have the same number of elements as planes.");
        return error.invalidArgument;
    }

    for (arr, 0..) |v, i| {
        try hz.checkFinite(v, filter_name, which, i, map_out);
        if (int_clip) {
            const t = @trunc(v);
            if (t < 0) {
                map_out.setError(filter_name ++ ": " ++ which ++ " value must be greater than or equal to 0.");
                return error.invalidArgument;
            }
            if (v > peak) {
                map_out.setError(filter_name ++ ": " ++ which ++ " value must be less than or equal to peak value.");
                return error.invalidArgument;
            }
            out_int[i] = @intFromFloat(t);
        } else {
            try hz.checkArg(f64, v, -peak, peak, filter_name, which, map_out);
            out_float[i] = @floatCast(v);
        }
    }
}

pub fn limiterCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;

    const min_in = map_in.getFloatArray("min");
    const max_in = map_in.getFloatArray("max");

    const num_planes = d.vi.format.numPlanes;
    // Exact integer peak (2^32 - 1 is not representable in getPeakValue's f32);
    // float clips use the largest finite value of their sample type as the
    // bound limit, so the narrowed bound (f32, or f16 in LimiterRT) stays finite.
    const bits: u6 = @intCast(d.vi.format.bitsPerSample);
    const peak: f64 = if (d.vi.format.sampleType == .Integer)
        @floatFromInt((@as(u64, 1) << bits) - 1)
    else if (bits == 16)
        math.floatMax(f16)
    else
        math.floatMax(f32);

    var planes: [3]bool = .{ true, true, true };
    const nodes = [_]?*vs.Node{d.node};
    hz.mapGetPlanes(map_in, map_out, &nodes, &planes, num_planes, filter_name, &zapi) catch return;

    const int_clip = d.vi.format.sampleType == .Integer;
    const has_min = min_in != null;
    if (min_in) |arr| {
        parseBounds("min", arr, num_planes, int_clip, peak, &d.min, &d.minf, map_out) catch {
            zapi.freeNode(d.node);
            return;
        };
    }

    const has_max = max_in != null;
    if (max_in) |arr| {
        parseBounds("max", arr, num_planes, int_clip, peak, &d.max, &d.maxf, map_out) catch {
            zapi.freeNode(d.node);
            return;
        };
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

    if (has_min and has_max) {
        for (0..@as(usize, @intCast(num_planes))) |p| {
            const bad = if (d.vi.format.sampleType == .Integer)
                d.min[p] > d.max[p]
            else
                d.minf[p] > d.maxf[p];
            if (bad) {
                map_out.setError(filter_name ++ ": min value must be less than or equal to max value.");
                zapi.freeNode(d.node);
                return;
            }
        }
    }

    const bps = BPSType.select(map_out, d.node, d.vi, filter_name) catch return;

    // Default args on an 8/16/32-bit integer clip clamp to the full range of
    // the carrier type — the identity. Pass the clip through instead of
    // instantiating a filter that byte-copies every plane (measured: the full8
    // variant is 100% of kernel work for the default 8-bit invocation).
    // 9..14-bit formats are excluded: their u16 carrier can hold values above
    // the nominal peak, which the full-range table really does clamp.
    if (!has_min and !(map_in.getBool("tv_range") orelse false) and
        (bps == .U8 or bps == .U16 or bps == .U32))
    {
        _ = map_out.consumeNode("clip", d.node, .Replace);
        return;
    }

    var i: u32 = 0;
    const idx: u32 = while (i < comptime_planes.len) : (i += 1) {
        if (std.mem.eql(bool, &comptime_planes[i], &planes)) break i;
    } else 0;

    const tv_range = map_in.getBool("tv_range") orelse false;
    const mask = map_in.getBool("mask") orelse false;
    const yuv = (d.vi.format.colorFamily == .YUV) and !mask;
    const get_frame: vs.FilterGetFrame = selectGetFrame(has_min, tv_range, yuv, num_planes, bps, idx);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, d.vi, get_frame, limiterFree, .Parallel, &deps, data);
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
