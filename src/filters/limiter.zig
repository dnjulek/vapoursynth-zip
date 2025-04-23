const std = @import("std");
const vs = @import("../vszip.zig").vapoursynth.vapoursynth4;
const plugin = @import("../vapoursynth/limiter.zig");
const hz = @import("../helper.zig");
const comptime_planes = plugin.comptime_planes;

pub fn getFrame(use_rt: bool, tv_range: bool, yuv: bool, num_planes: i32, bps: hz.BPSType, idx: u32) vs.FilterGetFrame {
    var get_frame: vs.FilterGetFrame = undefined;
    if (use_rt) {
        get_frame = switch (num_planes) {
            inline 1...3 => |np| switch (idx) {
                inline 0...(comptime_planes.len - 1) => |i| switch (bps) {
                    .U8 => &plugin.LimiterRT(u8, np, i).getFrame,
                    .U9, .U10, .U12, .U14, .U16 => &plugin.LimiterRT(u16, np, i).getFrame,
                    .U32 => &plugin.LimiterRT(u32, np, i).getFrame,
                    .F16 => &plugin.LimiterRT(f16, np, i).getFrame,
                    .F32 => &plugin.LimiterRT(f32, np, i).getFrame,
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
                        .U8 => if (yuv) &plugin.Limiter(u8, yuv8, np, i).getFrame else &plugin.Limiter(u8, rgb8, np, i).getFrame,
                        .U9 => if (yuv) &plugin.Limiter(u16, yuv9, np, i).getFrame else &plugin.Limiter(u16, rgb9, np, i).getFrame,
                        .U10 => if (yuv) &plugin.Limiter(u16, yuv10, np, i).getFrame else &plugin.Limiter(u16, rgb10, np, i).getFrame,
                        .U12 => if (yuv) &plugin.Limiter(u16, yuv12, np, i).getFrame else &plugin.Limiter(u16, rgb12, np, i).getFrame,
                        .U14 => if (yuv) &plugin.Limiter(u16, yuv14, np, i).getFrame else &plugin.Limiter(u16, rgb14, np, i).getFrame,
                        .U16 => if (yuv) &plugin.Limiter(u16, yuv16, np, i).getFrame else &plugin.Limiter(u16, rgb16, np, i).getFrame,
                        .U32 => if (yuv) &plugin.Limiter(u32, yuv32, np, i).getFrame else &plugin.Limiter(u32, rgb32, np, i).getFrame,
                        .F16 => if (yuv) &plugin.Limiter(f16, yuvf, np, i).getFrame else &plugin.Limiter(f16, rgbf, np, i).getFrame,
                        .F32 => if (yuv) &plugin.Limiter(f32, yuvf, np, i).getFrame else &plugin.Limiter(f32, rgbf, np, i).getFrame,
                    },
                    else => unreachable,
                },
                else => unreachable,
            };
        } else {
            get_frame = switch (num_planes) {
                inline 1...3 => |np| switch (idx) {
                    inline 0...(comptime_planes.len - 1) => |i| switch (bps) {
                        .U8 => &plugin.Limiter(u8, full8, np, i).getFrame,
                        .U9 => &plugin.Limiter(u16, full9, np, i).getFrame,
                        .U10 => &plugin.Limiter(u16, full10, np, i).getFrame,
                        .U12 => &plugin.Limiter(u16, full12, np, i).getFrame,
                        .U14 => &plugin.Limiter(u16, full14, np, i).getFrame,
                        .U16 => &plugin.Limiter(u16, full16, np, i).getFrame,
                        .U32 => &plugin.Limiter(u32, full32, np, i).getFrame,
                        .F16 => if (yuv) &plugin.Limiter(f16, yuvf, np, i).getFrame else &plugin.Limiter(f16, rgbf, np, i).getFrame,
                        .F32 => if (yuv) &plugin.Limiter(f32, yuvf, np, i).getFrame else &plugin.Limiter(f32, rgbf, np, i).getFrame,
                    },
                    else => unreachable,
                },
                else => unreachable,
            };
        }
    }

    return get_frame;
}

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
