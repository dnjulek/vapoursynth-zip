//! avszip_AdaptiveGrainMask(clip, float "luma_scaling")
//!
//! Output: a Gray clip of the input's depth, computed from logical plane 0
//! (Y, or R for planar RGB, as in vszip; AviSynth's default plane of planar
//! RGB is G).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/adaptive_grain_mask.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_AdaptiveGrainMask";
pub const params = "c[luma_scaling]f";

const Data = struct {
    luma_scaling: f32,
    peak: f32,
    shift: u5,
    /// Input plane read: logical plane 0 of the input format.
    src_plane: c_int,
};

const F = avs.Filter(Data, filter_name);

fn AdaptiveGrainMask(comptime T: type) type {
    return struct {
        const Lut = struct { lut: *const [256]T, shift: u5 };

        fn intRow(dst: []T, src: []const T, l: Lut) void {
            if (comptime T == u8) {
                filter.processIntU8(src, dst, l.lut);
            } else {
                filter.processInt(T, src, dst, l.lut, l.shift);
            }
        }

        fn floatRow(dst: []f32, src: []const f32, temp: f32) void {
            filter.processFloat(src, dst, temp);
        }

        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            const dp = dst.planeMut(T, c.AVS_PLANAR_Y);
            const raw = src.plane(T, d.src_plane);
            const sp = try hz.canonical(T, raw, dp.pitch, &scratch);
            const avg = filter.computeAverage(T, sp.fullSlice(), @intCast(sp.stride()), sp.width, sp.height, d.peak);
            const temp = filter.calcLumaScaling(avg, d.luma_scaling);

            // Per row: the LUT is indexed by sample value, so uninitialized
            // inter-row padding must never reach it.
            if (comptime @typeInfo(T) == .float) {
                hz.rowwise(f32, f32, dp, raw, temp, floatRow);
            } else {
                var lut: [256]T = undefined;
                filter.buildLut(T, &lut, temp, d.peak);
                hz.rowwise(T, T, dp, raw, Lut{ .lut = &lut, .shift = d.shift }, intRow);
            }
            return dst;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    var d: Data = undefined;
    d.src_plane = format.planeId(0);
    if (format.isFloat()) {
        d.peak = 1.0;
        d.shift = 0;
    } else {
        d.peak = @floatCast(format.peak());
        d.shift = @intCast(if (format.bits > 8) format.bits - 8 else 0);
    }
    d.luma_scaling = try ctx.float(f32, 1, "luma_scaling", 10.0, -math.floatMax(f32), math.floatMax(f32));

    const gf = switch (format.sample) {
        inline else => |s| F.getFrame(AdaptiveGrainMask(s.Type()).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice, .vi = hz.withFormat(vi.*, hz.grayOf(format)) });
    return avs.Create.returnClip(out);
}
