const std = @import("std");
const math = std.math;

const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const zapi = vapoursynth.zigapi;

const BPSType = @import("../helper.zig").BPSType;

const allocator = std.heap.c_allocator;
pub const filter_name = "Limiter";

const LimiterData = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    max: [3]u32,
    min: [3]u32,
    maxf: [3]f32,
    minf: [3]f32,
};

pub fn LimiterRT(comptime T: type, np: comptime_int) type {
    return struct {
        fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *LimiterData = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.ZFrame.init(d.node, n, frame_ctx, core, vsapi);
                defer src.deinit();
                const dst = src.newVideoFrame();

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
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

pub fn Limiter(comptime T: type, rng: anytype, np: comptime_int) type {
    return struct {
        fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *LimiterData = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.ZFrame.init(d.node, n, frame_ctx, core, vsapi);
                defer src.deinit();
                const dst = src.newVideoFrame();

                comptime var plane = 0;
                inline while (plane < np) : (plane += 1) {
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
    const d: *LimiterData = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn limiterCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: LimiterData = undefined;
    const map_in = zapi.ZMapRO.init(in, vsapi);
    const map_out = zapi.ZMapRW.init(out, vsapi);

    d.node, d.vi = map_in.getNodeVi("clip");

    const min_in = map_in.getFloatArray("min");
    const max_in = map_in.getFloatArray("max");
    const tv_range = map_in.getBool("tv_range") orelse false;

    const num_planes = d.vi.format.numPlanes;

    var has_min = false;
    if (min_in) |arr| {
        has_min = true;
        if (arr.len != num_planes) {
            map_out.setError(filter_name ++ ": min array must have the same number of elements as planes.");
            vsapi.?.freeNode.?(d.node);
            return;
        }

        for (0..arr.len) |i| {
            if (d.vi.format.sampleType == .Integer) {
                d.min[i] = @intFromFloat(arr[i]);
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
            vsapi.?.freeNode.?(d.node);
            return;
        }

        for (0..arr.len) |i| {
            if (d.vi.format.sampleType == .Integer) {
                d.max[i] = @intFromFloat(arr[i]);
            } else {
                d.maxf[i] = @floatCast(arr[i]);
            }
        }
    }

    if (has_min and !has_max) {
        map_out.setError(filter_name ++ ": min array is set but max array is not.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    if (!has_min and has_max) {
        map_out.setError(filter_name ++ ": max array is set but min array is not.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    const bps = BPSType.select(map_out, d.node, d.vi, filter_name) catch return;

    const full8 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 255, 255, 255 } };
    const full9 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 511, 511, 511 } };
    const full10 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 1023, 1023, 1023 } };
    const full12 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 4095, 4095, 4095 } };
    const full14 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 16383, 16383, 16383 } };
    const full16 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 65535, 65535, 65535 } };
    const full32 = [2][3]comptime_int{ .{ 0, 0, 0 }, .{ 4294967295, 4294967295, 4294967295 } };

    const yuv8 = [2][3]comptime_int{ .{ 16, 16, 16 }, .{ 235, 240, 240 } };
    const yuv9 = [2][3]comptime_int{ .{ 32, 32, 32 }, .{ 470, 480, 480 } };
    const yuv10 = [2][3]comptime_int{ .{ 64, 64, 64 }, .{ 940, 960, 960 } };
    const yuv12 = [2][3]comptime_int{ .{ 256, 256, 256 }, .{ 3760, 3840, 3840 } };
    const yuv14 = [2][3]comptime_int{ .{ 1024, 1024, 1024 }, .{ 15040, 15360, 15360 } };
    const yuv16 = [2][3]comptime_int{ .{ 4096, 4096, 4096 }, .{ 60160, 61440, 61440 } };
    const yuv32 = [2][3]comptime_int{ .{ 268435456, 268435456, 268435456 }, .{ 3942645760, 4026531840, 4026531840 } };

    const rgb8 = [2][3]comptime_int{ .{ 16, 16, 16 }, .{ 235, 235, 235 } };
    const rgb9 = [2][3]comptime_int{ .{ 32, 32, 32 }, .{ 470, 470, 470 } };
    const rgb10 = [2][3]comptime_int{ .{ 64, 64, 64 }, .{ 940, 940, 940 } };
    const rgb12 = [2][3]comptime_int{ .{ 256, 256, 256 }, .{ 3760, 3760, 3760 } };
    const rgb14 = [2][3]comptime_int{ .{ 1024, 1024, 1024 }, .{ 15040, 15040, 15040 } };
    const rgb16 = [2][3]comptime_int{ .{ 4096, 4096, 4096 }, .{ 60160, 60160, 60160 } };
    const rgb32 = [2][3]comptime_int{ .{ 268435456, 268435456, 268435456 }, .{ 3942645760, 3942645760, 3942645760 } };

    const yuvf = [2][3]comptime_float{ .{ 0, -0.5, -0.5 }, .{ 1, 0.5, 0.5 } };
    const rgbf = [2][3]comptime_float{ .{ 0, 0, 0 }, .{ 1, 1, 1 } };

    const use_rt = (has_max) or (has_min);
    const yuv = d.vi.format.colorFamily == .YUV;
    var get_frame: vs.FilterGetFrame = undefined;

    if (use_rt) {
        get_frame = switch (num_planes) {
            inline 1...3 => |np| switch (bps) {
                .U8 => &LimiterRT(u8, np).getFrame,
                .U9, .U10, .U12, .U14, .U16 => &LimiterRT(u16, np).getFrame,
                .U32 => &LimiterRT(u32, np).getFrame,
                .F16 => &LimiterRT(f16, np).getFrame,
                .F32 => &LimiterRT(f32, np).getFrame,
            },
            else => unreachable,
        };
    } else {
        if (tv_range) {
            get_frame = switch (num_planes) {
                inline 1...3 => |np| switch (bps) {
                    .U8 => if (yuv) &Limiter(u8, yuv8, np).getFrame else &Limiter(u8, rgb8, np).getFrame,
                    .U9 => if (yuv) &Limiter(u16, yuv9, np).getFrame else &Limiter(u16, rgb9, np).getFrame,
                    .U10 => if (yuv) &Limiter(u16, yuv10, np).getFrame else &Limiter(u16, rgb10, np).getFrame,
                    .U12 => if (yuv) &Limiter(u16, yuv12, np).getFrame else &Limiter(u16, rgb12, np).getFrame,
                    .U14 => if (yuv) &Limiter(u16, yuv14, np).getFrame else &Limiter(u16, rgb14, np).getFrame,
                    .U16 => if (yuv) &Limiter(u16, yuv16, np).getFrame else &Limiter(u16, rgb16, np).getFrame,
                    .U32 => if (yuv) &Limiter(u32, yuv32, np).getFrame else &Limiter(u32, rgb32, np).getFrame,
                    .F16 => if (yuv) &Limiter(f16, yuvf, np).getFrame else &Limiter(f16, rgbf, np).getFrame,
                    .F32 => if (yuv) &Limiter(f32, yuvf, np).getFrame else &Limiter(f32, rgbf, np).getFrame,
                },
                else => unreachable,
            };
        } else {
            get_frame = switch (num_planes) {
                inline 1...3 => |np| switch (bps) {
                    .U8 => &Limiter(u8, full8, np).getFrame,
                    .U9 => &Limiter(u16, full9, np).getFrame,
                    .U10 => &Limiter(u16, full10, np).getFrame,
                    .U12 => &Limiter(u16, full12, np).getFrame,
                    .U14 => &Limiter(u16, full14, np).getFrame,
                    .U16 => &Limiter(u16, full16, np).getFrame,
                    .U32 => &Limiter(u32, full32, np).getFrame,
                    .F16 => if (yuv) &Limiter(f16, yuvf, np).getFrame else &Limiter(f16, rgbf, np).getFrame,
                    .F32 => if (yuv) &Limiter(f32, yuvf, np).getFrame else &Limiter(f32, rgbf, np).getFrame,
                },
                else => unreachable,
            };
        }
    }

    const data: *LimiterData = allocator.create(LimiterData) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, get_frame, limiterFree, .Parallel, &deps, deps.len, data, core);
}
