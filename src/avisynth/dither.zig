//! avszip_Dither(clip, int bitdepth, int "dither_type", int "sample_type", bool "fulls", bool "fulld")
//!
//! Bit-depth conversion with dithering (fmtconv `bitdepth` + zimg modes).
//! AviSynth has no half-float format, so float output is 32-bit only. When
//! fulls/fulld are given, the output's `_ColorRange` is stamped (0 = full).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const hz = @import("helper.zig");
const filter = @import("../filters/dither.zig");
const errdiff = @import("../filters/dither_errdiff.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_Dither";
pub const params = "c[bitdepth]i[dither_type]i[sample_type]i[fulls]b[fulld]b";

pub const Mode = filter.Mode;

const Data = struct {
    format_in: avs.Format,
    num_planes: u32,
    cfg: filter.Config,
    /// Alpha conversion (full range, rounded, never dithered) when the clip
    /// has an alpha plane.
    alpha: ?filter.Config,
    stamp_range: ?hz.Range,
};

const F = avs.Filter(Data, filter_name);

/// Shared frame loop: new output frame (props from src), alpha plane copied
/// through unconverted when bit depths match, else converted like a luma
/// plane is not part of vszip; alpha is rejected at create time instead.
fn Frames(comptime T: type, comptime U: type) type {
    return struct {
        const Planes = struct { src: []const T, dst: []U, w: u32, h: u32, ss: u32, ds: u32 };

        fn begin(ctx: *avs.FrameCtx, n: c_int) !struct { src: avs.Frame, dst: avs.Frame } {
            const src = try ctx.childFrame(n);
            errdefer src.release();
            const dst = try ctx.newFrame(src);
            return .{ .src = src, .dst = dst };
        }

        fn plane(d: *const Data, src: avs.Frame, dst: avs.Frame, p: usize, scratch: *hz.Scratch) !Planes {
            const id = d.format_in.planeId(p);
            const raw = src.plane(T, id);
            const sp = try hz.canonical(T, raw, raw.pitch, scratch);
            const dp = dst.planeMut(U, id);
            return .{ .src = sp.fullSlice(), .dst = dp.fullSlice(), .w = dp.width, .h = dp.height, .ss = @intCast(sp.stride()), .ds = @intCast(dp.stride()) };
        }

        /// Convert the alpha plane (if any) and stamp the range.
        fn finish(d: *const Data, ctx: *avs.FrameCtx, src: avs.Frame, dst: avs.Frame, scratch: *hz.Scratch) !void {
            if (d.alpha) |ac| {
                if (ac.identity) {
                    avs.copyPlane(ctx.env(), dst, src, c.AVS_PLANAR_A);
                } else {
                    const pl = try plane(d, src, dst, 3, scratch);
                    if (comptime @typeInfo(U) == .float) {
                        filter.Convert(T, U).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, ac.sc[0]);
                    } else {
                        filter.Ordered(T, U, .round).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, ac.sc[0], ac.peak_out, 3, ac.bits_in, ac.bits_out, ac.int_kernel[0]);
                    }
                }
            }
            if (d.stamp_range) |r| hz.setRange(ctx.env(), dst, r);
        }
    };
}

fn OrderedFilter(comptime T: type, comptime U: type, comptime mode: filter.OrderedMode) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const Fr = Frames(T, U);
            const f = try Fr.begin(ctx, n);
            defer f.src.release();
            errdefer f.dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            for (0..d.num_planes) |p| {
                const pl = try Fr.plane(d, f.src, f.dst, p, &scratch);
                filter.Ordered(T, U, mode).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, d.cfg.sc[p], d.cfg.peak_out, @intCast(p), d.cfg.bits_in, d.cfg.bits_out, d.cfg.int_kernel[p]);
            }
            try Fr.finish(d, ctx, f.src, f.dst, &scratch);
            return f.dst;
        }
    };
}

fn ErrdiffFilter(comptime T: type, comptime U: type, comptime mode: errdiff.FmtcMode) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const Fr = Frames(T, U);
            const f = try Fr.begin(ctx, n);
            defer f.src.release();
            errdefer f.dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            const err_buf = try errdiff.ErrorBuffer.init(d.cfg.err_buf_len);
            defer err_buf.deinit();
            for (0..d.num_planes) |p| {
                const pl = try Fr.plane(d, f.src, f.dst, p, &scratch);
                err_buf.clear();
                if (@typeInfo(T) == .int and d.cfg.int_kernel[p]) {
                    errdiff.Fmtc(T, U, mode).processPlaneInt(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, errdiff.IntQuant.init(d.cfg.bits_in, d.cfg.bits_out), &err_buf);
                } else {
                    const dif_bits = @as(i32, d.cfg.bits_in) - @as(i32, d.cfg.bits_out);
                    errdiff.Fmtc(T, U, mode).processPlaneFloat(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, d.cfg.sc[p].scale, d.cfg.sc[p].offset, d.cfg.peak_out, dif_bits, &err_buf);
                }
            }
            try Fr.finish(d, ctx, f.src, f.dst, &scratch);
            return f.dst;
        }
    };
}

fn ZimgErrdiffFilter(comptime T: type, comptime U: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const Fr = Frames(T, U);
            const f = try Fr.begin(ctx, n);
            defer f.src.release();
            errdefer f.dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            const error_top = try allocator.alloc(f32, d.cfg.err_buf_len);
            defer allocator.free(error_top);
            const error_cur = try allocator.alloc(f32, d.cfg.err_buf_len);
            defer allocator.free(error_cur);
            for (0..d.num_planes) |p| {
                const pl = try Fr.plane(d, f.src, f.dst, p, &scratch);
                errdiff.Zimg(T, U).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, d.cfg.sc[p].scale, d.cfg.sc[p].offset, d.cfg.peak_out, error_top, error_cur);
            }
            try Fr.finish(d, ctx, f.src, f.dst, &scratch);
            return f.dst;
        }
    };
}

fn MulAddUpFilter(comptime T: type, comptime U: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const Fr = Frames(T, U);
            const f = try Fr.begin(ctx, n);
            defer f.src.release();
            errdefer f.dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            for (0..d.num_planes) |p| {
                const pl = try Fr.plane(d, f.src, f.dst, p, &scratch);
                filter.MulAddUp(T, U).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, d.cfg.gain_up, d.cfg.add_up[p]);
            }
            try Fr.finish(d, ctx, f.src, f.dst, &scratch);
            return f.dst;
        }
    };
}

fn ConvertFilter(comptime T: type, comptime U: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const Fr = Frames(T, U);
            const f = try Fr.begin(ctx, n);
            defer f.src.release();
            errdefer f.dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            for (0..d.num_planes) |p| {
                const pl = try Fr.plane(d, f.src, f.dst, p, &scratch);
                filter.Convert(T, U).processPlane(pl.src, pl.dst, pl.w, pl.h, pl.ss, pl.ds, d.cfg.sc[p]);
            }
            try Fr.finish(d, ctx, f.src, f.dst, &scratch);
            return f.dst;
        }
    };
}

fn passGetFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    var src = try ctx.childFrame(n);
    if (d.stamp_range == null) return src;
    errdefer src.release();
    try ctx.makePropertyWritable(&src);
    hz.setRange(ctx.env(), src, d.stamp_range.?);
    return src;
}

fn selectModeGetFrame(comptime T: type, comptime U: type, mode: Mode) avs.filter.RawGetFrame {
    switch (mode) {
        .zimg_error_diffusion => return F.getFrame(ZimgErrdiffFilter(T, U).getFrame),
        inline .fmtc_filter_lite, .fmtc_stucki, .fmtc_atkinson, .fmtc_floyd, .fmtc_ostro => |m| {
            return F.getFrame(ErrdiffFilter(T, U, comptime m.fmtcErrdiff().?).getFrame);
        },
        inline else => |m| return F.getFrame(OrderedFilter(T, U, comptime m.ordered().?).getFrame),
    }
}

fn selectGetFrame(in: avs.SampleType, out: avs.SampleType, mode: Mode, upconv: bool) avs.filter.RawGetFrame {
    switch (in) {
        inline else => |ti| {
            const T = comptime ti.Type();
            switch (out) {
                .f32 => {
                    if (T == f32) return F.getFrame(passGetFrame);
                    return F.getFrame(ConvertFilter(T, f32).getFrame);
                },
                inline .u8, .u16 => |to| {
                    const U = comptime to.Type();
                    if (comptime @typeInfo(T) == .int) {
                        if (upconv) {
                            if (comptime @bitSizeOf(U) >= @bitSizeOf(T)) return F.getFrame(MulAddUpFilter(T, U).getFrame);
                            unreachable;
                        }
                    }
                    return selectModeGetFrame(T, U, mode);
                },
            }
        },
    }
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    const bitdepth = try arg.int(i64, "bitdepth", null, math.minInt(i64), math.maxInt(i64));
    const dst_float_default = bitdepth == 32;
    const st = try arg.int(i64, "sample_type", if (dst_float_default) 1 else 0, math.minInt(i64), math.maxInt(i64));
    const dst_float = st != 0;

    const bits_ok = if (dst_float) bitdepth == 32 else (bitdepth >= 8 and bitdepth <= 16);
    if (!bits_ok) {
        if (dst_float and bitdepth == 16) return ctx.fail("16-bit float output is not available in AviSynth+ (use bitdepth=32).", .{});
        return ctx.fail("unsupported bitdepth (int: 8-16, float: 32).", .{});
    }
    if (!dst_float and bitdepth != 8 and bitdepth != 10 and bitdepth != 12 and bitdepth != 14 and bitdepth != 16)
        return ctx.fail("AviSynth+ integer formats are 8, 10, 12, 14 or 16 bit.", .{});

    const imode = try arg.int(i64, "dither_type", @backingInt(Mode.zimg_random), math.minInt(i64), math.maxInt(i64));
    if (imode < 0 or imode > @backingInt(Mode.fmtc_quasirnd)) return ctx.fail("dither_type must be 0-11.", .{});
    const mode: Mode = @fromBackingInt(@intCast(@as(i32, @intCast(imode))));

    var out_fmt = format;
    out_fmt.bits = @intCast(bitdepth);
    out_fmt.sample = if (dst_float) .f32 else if (bitdepth == 8) .u8 else .u16;

    const src_float = format.isFloat();
    var d: Data = undefined;
    d.format_in = format;
    d.num_planes = format.color_planes;

    const fulls_opt: ?bool = if (arg.defined("fulls")) try arg.boolean("fulls", false) else null;
    const fulld_opt: ?bool = if (arg.defined("fulld")) try arg.boolean("fulld", false) else null;
    const range_default = format.family == .rgb;
    const fulls = fulls_opt orelse range_default;
    const fulld = fulld_opt orelse fulls;
    d.stamp_range = if (fulls_opt != null or fulld_opt != null) (if (fulld) hz.Range.FULL else .LIMITED) else null;

    // Alpha is data, not color: full range at both ends, plainly rounded
    // (up-conversion then down-conversion returns the original values).
    d.alpha = if (format.has_alpha) filter.Config.init(.{
        .bits_in = format.bits,
        .src_float = src_float,
        .bits_out = @intCast(bitdepth),
        .dst_float = dst_float,
        .fulls = true,
        .fulld = true,
        .mode = .round,
        .is_yuv = false,
        .width = @intCast(vi.width),
    }) else null;

    d.cfg = filter.Config.init(.{
        .bits_in = format.bits,
        .src_float = src_float,
        .bits_out = @intCast(bitdepth),
        .dst_float = dst_float,
        .fulls = fulls,
        .fulld = fulld,
        .mode = mode,
        .is_yuv = format.family == .yuv,
        .width = @intCast(vi.width),
    });

    const gf = if (d.cfg.identity) F.getFrame(passGetFrame) else selectGetFrame(format.sample, out_fmt.sample, mode, d.cfg.upconv);
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice, .vi = hz.withFormat(vi.*, out_fmt) });
    return avs.Create.returnClip(out);
}
