//! vszip_Limiter(clip, float[] "min", float[] "max", bool "tv_range", bool "mask", int[] "planes")

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/limiter.zig");
const hz = @import("helper.zig");
const BPSType = hz.BPSType;

pub const filter_name = "vszip_Limiter";
pub const params = "c[min]f*[max]f*[tv_range]b[mask]b[planes]i*";

const Data = struct {
    format: avs.Format,
    max: [3]u32 = .{ 0, 0, 0 },
    min: [3]u32 = .{ 0, 0, 0 },
    maxf: [3]f32 = .{ 0, 0, 0 },
    minf: [3]f32 = .{ 0, 0, 0 },
};

const F = avs.Filter(Data, filter_name);

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

fn clampKernel(comptime T: type) fn ([]T, []const T, [2]T) void {
    return struct {
        fn k(dst: []T, src: []const T, b: [2]T) void {
            filter.clampSlice(T, dst, src, b[0], b[1]);
        }
    }.k;
}

/// `rng` is a comptime [2][3] table (min row, max row) or `null` for the
/// runtime bounds stored in Data.
fn Limiter(comptime T: type, comptime rng: anytype, comptime np: comptime_int, comptime idx: comptime_int) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            const env = ctx.env();
            const process = comptime_planes[idx];
            hz.copyUnprocessed(env, dst, src, d.format, &process);

            inline for (0..np) |plane| {
                if (comptime !process[plane]) continue;
                const id = d.format.planeId(plane);
                const b: [2]T = if (comptime @TypeOf(rng) == @TypeOf(null))
                    (if (@typeInfo(T) == .int)
                        .{ @intCast(d.min[plane]), @intCast(d.max[plane]) }
                    else
                        .{ @floatCast(d.minf[plane]), @floatCast(d.maxf[plane]) })
                else
                    .{ rng[0][plane], rng[1][plane] };
                hz.elementwise(T, dst.planeMut(T, id), src.plane(T, id), b, clampKernel(T));
            }
            return dst;
        }
    };
}

fn select(comptime T: type, comptime rng: anytype, np: u8, idx: u32) avs.filter.RawGetFrame {
    return switch (np) {
        inline 1...3 => |cnp| switch (idx) {
            inline 0...(comptime_planes.len - 1) => |i| F.getFrame(Limiter(T, rng, cnp, i).getFrame),
            else => unreachable,
        },
        else => unreachable,
    };
}

fn getFrameFn(use_rt: bool, tv_range: bool, yuv: bool, np: u8, bps: BPSType, idx: u32) avs.filter.RawGetFrame {
    if (use_rt) return switch (bps) {
        .U8 => select(u8, null, np, idx),
        .U10, .U12, .U14, .U16 => select(u16, null, np, idx),
        .F32 => select(f32, null, np, idx),
    };
    if (tv_range) return switch (bps) {
        .U8 => if (yuv) select(u8, filter.yuv8, np, idx) else select(u8, filter.rgb8, np, idx),
        .U10 => if (yuv) select(u16, filter.yuv10, np, idx) else select(u16, filter.rgb10, np, idx),
        .U12 => if (yuv) select(u16, filter.yuv12, np, idx) else select(u16, filter.rgb12, np, idx),
        .U14 => if (yuv) select(u16, filter.yuv14, np, idx) else select(u16, filter.rgb14, np, idx),
        .U16 => if (yuv) select(u16, filter.yuv16, np, idx) else select(u16, filter.rgb16, np, idx),
        .F32 => if (yuv) select(f32, filter.yuvf, np, idx) else select(f32, filter.rgbf, np, idx),
    };
    return switch (bps) {
        .U8 => select(u8, filter.full8, np, idx),
        .U10 => select(u16, filter.full10, np, idx),
        .U12 => select(u16, filter.full12, np, idx),
        .U14 => select(u16, filter.full14, np, idx),
        .U16 => select(u16, filter.full16, np, idx),
        .F32 => if (yuv) select(f32, filter.yuvf, np, idx) else select(f32, filter.rgbf, np, idx),
    };
}

/// One bound array (`min` or `max`): one value per color plane, every value
/// finite. Integer clips truncate toward zero and must land in [0, peak];
/// float clips must lie in [-floatMax, floatMax]. Checks run in f64 before
/// any conversion.
fn parseBounds(ctx: *avs.Create, comptime which: []const u8, i: usize, num_planes: u8, int_clip: bool, peak: f64, out_int: *[3]u32, out_float: *[3]f32) !bool {
    var buf: [3]f64 = undefined;
    const arr = (try ctx.floatArray(f64, i, which, &buf, -math.floatMax(f64), math.floatMax(f64))) orelse return false;
    if (arr.len != num_planes)
        return ctx.fail(which ++ " array must have the same number of elements as planes.", .{});
    for (arr, 0..) |v, p| {
        if (int_clip) {
            const t = @trunc(v);
            if (t < 0) return ctx.fail(which ++ " value must be greater than or equal to 0.", .{});
            if (v > peak) return ctx.fail(which ++ " value must be less than or equal to peak value.", .{});
            out_int[p] = @intFromFloat(t);
        } else {
            if (v < -peak or v > peak) return ctx.fail(which ++ "[{d}] value {d} is out of range [{d}, {d}].", .{ p, v, -peak, peak });
            out_float[p] = @floatCast(v);
        }
    }
    return true;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    var d: Data = .{ .format = format };

    const num_planes = format.color_planes;
    const int_clip = !format.isFloat();
    const peak: f64 = if (int_clip) format.peak() else math.floatMax(f32);

    const process4 = try ctx.planes(5, num_planes, true);
    const planes: [3]bool = .{ process4[0], process4[1], process4[2] };

    const has_min = try parseBounds(ctx, "min", 1, num_planes, int_clip, peak, &d.min, &d.minf);
    const has_max = try parseBounds(ctx, "max", 2, num_planes, int_clip, peak, &d.max, &d.maxf);
    if (has_min and !has_max) return ctx.fail("min array is set but max array is not.", .{});
    if (!has_min and has_max) return ctx.fail("max array is set but min array is not.", .{});
    if (has_min) {
        for (0..num_planes) |p| {
            const bad = if (int_clip) d.min[p] > d.max[p] else d.minf[p] > d.maxf[p];
            if (bad) return ctx.fail("min value must be less than or equal to max value.", .{});
        }
    }

    const tv_range = try ctx.boolean(3, "tv_range", false);
    const mask = try ctx.boolean(4, "mask", false);
    const bps = BPSType.fromFormat(format);

    // Default args on an 8/16-bit clip clamp to the carrier's full range: the
    // identity. Pass the clip through instead of copying every plane. 10..14
    // bit are excluded: their u16 carrier can hold values above the nominal
    // peak, which the full-range table really does clamp.
    if (!has_min and !tv_range and (bps == .U8 or bps == .U16)) {
        return avs.Create.returnClip(clip.retain());
    }

    const idx = hz.planeIndex(comptime_planes, planes, 0);
    const yuv = format.family == .yuv and !mask;
    const get_frame = getFrameFn(has_min, tv_range, yuv, num_planes, bps, idx);
    const out = try F.create(ctx.env, clip, d, get_frame, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
