//! Host-free tests: every kernel in `filters/` is compiled here without the
//! VapourSynth or AviSynth binding (this test root imports neither), which
//! proves the shared kernels carry no host dependency. Generic kernels are
//! additionally instantiated for the sample types the plugins use, so lazy
//! analysis cannot hide a host import or an invalid body.

const std = @import("std");

const kernels = .{
    @import("filters/adaptive_grain_mask.zig"),
    @import("filters/bilateral.zig"),
    @import("filters/bilateral_dither.zig"),
    @import("filters/bilateral_dither_subspl.zig"),
    @import("filters/boxblur_comptime.zig"),
    @import("filters/boxblur_runtime.zig"),
    @import("filters/checkmate.zig"),
    @import("filters/clahe.zig"),
    @import("filters/color_map.zig"),
    @import("filters/comb_mask.zig"),
    @import("filters/comb_mask_mt.zig"),
    @import("filters/compress.zig"),
    @import("filters/deband_float.zig"),
    @import("filters/deband_int.zig"),
    @import("filters/deband_setup.zig"),
    @import("filters/dither.zig"),
    @import("filters/dither_errdiff.zig"),
    @import("filters/dither_pattern.zig"),
    @import("filters/eedi3.zig"),
    @import("filters/image_read.zig"),
    @import("filters/limit_filter.zig"),
    @import("filters/limiter.zig"),
    @import("filters/mosquito_nr.zig"),
    @import("filters/mosquito_nr_float.zig"),
    @import("filters/planeaverage.zig"),
    @import("filters/planeminmax.zig"),
    @import("filters/simd.zig"),
    @import("filters/srgb.zig"),
    @import("filters/ssimulacra2.zig"),
    @import("filters/wnnm.zig"),
    @import("filters/xpsnr.zig"),
    @import("vcl.zig"),
    @import("common.zig"),
};

test "kernels compile without a host binding" {
    inline for (kernels) |k| std.testing.refAllDecls(k);
}

const limiter = @import("filters/limiter.zig");
const planeaverage = @import("filters/planeaverage.zig");
const boxblur_rt = @import("filters/boxblur_runtime.zig");
const limit_filter = @import("filters/limit_filter.zig");

test "limiter clampSlice matches scalar clamp at SIMD boundaries" {
    var src: [131]u16 = undefined;
    var dst: [131]u16 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast((i * 997) % 65536);
    inline for (.{ 1, 15, 16, 17, 63, 64, 65, 131 }) |n| {
        limiter.clampSlice(u16, dst[0..n], src[0..n], 1000, 50000);
        for (dst[0..n], src[0..n]) |d, s| try std.testing.expectEqual(@min(@max(1000, s), 50000), d);
    }
    var fs: [37]f32 = undefined;
    var fd: [37]f32 = undefined;
    for (&fs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 18.0 - 0.5;
    limiter.clampSlice(f32, &fd, &fs, -0.25, 0.75);
    for (fd, fs) |d, s| try std.testing.expectEqual(@min(@max(-0.25, s), 0.75), d);
}

test "planeaverage matches a scalar oracle with stride padding" {
    const w = 37;
    const h = 5;
    const stride = 48;
    var buf: [stride * h]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @intCast((i * 31 + 7) % 256);
    var sum: u64 = 0;
    for (0..h) |y| for (0..w) |x| {
        sum += buf[y * stride + x];
    };
    const want = @as(f64, @floatFromInt(sum)) / (w * h) / 255.0;
    const ex = planeaverage.Exclude{ .i = &.{} };
    try std.testing.expectApproxEqRel(want, planeaverage.average(u8, &buf, stride, w, h, ex, 255), 1e-12);
}

test "generic kernels instantiate for every plugin sample type" {
    inline for (.{ u8, u16, f32 }) |T| {
        _ = &boxblur_rt.hblur;
        _ = &limit_filter.process;
        const s: []const T = &.{};
        const d: []T = &.{};
        limit_filter.process(T, s, s, s, d, 1, 1, 2);
        limiter.clampSlice(T, d, s, 0, 1);
    }
}
