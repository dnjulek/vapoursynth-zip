const std = @import("std");

const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const filter = @import("../filters/dither.zig");
const errdiff = @import("../filters/dither_errdiff.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Dither";

pub const Mode = enum(i32) {
    round = 0,
    zimg_ordered = 1,
    zimg_random = 2,
    zimg_error_diffusion = 3,
    fmtc_ordered_bayer = 4,
    fmtc_filter_lite = 5,
    fmtc_stucki = 6,
    fmtc_atkinson = 7,
    fmtc_floyd = 8,
    fmtc_ostro = 9,
    fmtc_void_cluster = 10,
    fmtc_quasirnd = 11,

    fn ordered(self: Mode) ?filter.OrderedMode {
        return switch (self) {
            .round => .round,
            .zimg_ordered => .zimg_ordered,
            .zimg_random => .zimg_random,
            .fmtc_ordered_bayer => .fmtc_bayer,
            .fmtc_void_cluster => .fmtc_void,
            .fmtc_quasirnd => .fmtc_quasi,
            else => null,
        };
    }

    fn fmtcErrdiff(self: Mode) ?errdiff.FmtcMode {
        return switch (self) {
            .fmtc_filter_lite => .filter_lite,
            .fmtc_stucki => .stucki,
            .fmtc_atkinson => .atkinson,
            .fmtc_floyd => .floyd,
            .fmtc_ostro => .ostro,
            else => null,
        };
    }
};

const Data = struct {
    node: *vs.Node,
    vi_out: vs.VideoInfo,
    num_planes: u32,
    sc: [3]filter.ScaleOffset,
    int_kernel: [3]bool,
    bits_in: u6,
    bits_out: u6,
    peak_out: f32,
    gain_up: u16,
    add_up: [3]u16,
    err_buf_len: u32,
    stamp_range: ?i64,
};

const PlaneFmt = struct {
    bits: u6,
    is_float: bool,
    full: bool,
    chroma: bool,

    fn span(self: PlaneFmt) f64 {
        if (self.is_float) return 1.0;
        const bits: u5 = @intCast(self.bits);
        if (self.full) return @floatFromInt((@as(u32, 1) << bits) - 1);
        const base: u32 = if (self.chroma) 224 else 219;
        return @floatFromInt(base << @intCast(bits - 8));
    }

    fn min(self: PlaneFmt) f64 {
        if (self.is_float) return if (self.chroma) -0.5 else 0.0;
        if (self.full) return if (self.chroma) 0.5 else 0.0;
        return @floatFromInt(@as(u32, 16) << @intCast(self.bits - 8));
    }
};

const GainAdd = struct { gain: f64, add: f64 };

fn gainAdd(src: PlaneFmt, dst: PlaneFmt) GainAdd {
    const gain = dst.span() / src.span();
    return .{ .gain = gain, .add = dst.min() - src.min() * gain };
}

fn stampRange(dst: anytype, d: *const Data) void {
    if (d.stamp_range) |cr| {
        const props = dst.getPropertiesRW();
        props.setInt("_ColorRange", cr, .Replace);
    }
}

fn OrderedFilter(comptime T: type, comptime U: type, comptime mode: filter.OrderedMode) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi_out.format });

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);

                    filter.Ordered(T, U, mode).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.sc[plane], d.peak_out, plane, d.bits_in, d.bits_out, d.int_kernel[plane]);
                }

                stampRange(dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

fn ErrdiffFilter(comptime T: type, comptime U: type, comptime mode: errdiff.FmtcMode) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi_out.format });

                const err_buf = errdiff.ErrorBuffer.init(d.err_buf_len);
                defer err_buf.deinit();

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);
                    err_buf.clear();

                    if (@typeInfo(T) == .int and d.int_kernel[plane]) {
                        errdiff.Fmtc(T, U, mode).processPlaneInt(srcp, dstp, w, h, src_stride, dst_stride, errdiff.IntQuant.init(d.bits_in, d.bits_out), &err_buf);
                    } else {
                        const dif_bits = @as(i32, d.bits_in) - @as(i32, d.bits_out);
                        errdiff.Fmtc(T, U, mode).processPlaneFloat(srcp, dstp, w, h, src_stride, dst_stride, d.sc[plane].scale, d.sc[plane].offset, d.peak_out, dif_bits, &err_buf);
                    }
                }

                stampRange(dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

fn ZimgErrdiffFilter(comptime T: type, comptime U: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi_out.format });

                const error_top = allocator.alloc(f32, d.err_buf_len) catch unreachable;
                defer allocator.free(error_top);
                const error_cur = allocator.alloc(f32, d.err_buf_len) catch unreachable;
                defer allocator.free(error_cur);

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);

                    errdiff.Zimg(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.sc[plane].scale, d.sc[plane].offset, d.peak_out, error_top, error_cur);
                }

                stampRange(dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

fn MulAddUpFilter(comptime T: type, comptime U: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi_out.format });

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);

                    filter.MulAddUp(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.gain_up, d.add_up[plane]);
                }

                stampRange(dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

fn ConvertFilter(comptime T: type, comptime U: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi_out.format });

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);

                    filter.Convert(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.sc[plane]);
                }

                stampRange(dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

fn passGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        if (d.stamp_range == null) return src.frame;
        defer src.deinit();
        const dst = src.copyFrame();
        stampRange(dst, d);
        return dst.frame;
    }

    return null;
}

fn selectModeGetFrame(comptime T: type, comptime U: type, mode: Mode) vs.FilterGetFrame {
    switch (mode) {
        .zimg_error_diffusion => return &ZimgErrdiffFilter(T, U).getFrame,
        inline .fmtc_filter_lite, .fmtc_stucki, .fmtc_atkinson, .fmtc_floyd, .fmtc_ostro => |m| {
            return &ErrdiffFilter(T, U, comptime m.fmtcErrdiff().?).getFrame;
        },
        inline else => |m| {
            return &OrderedFilter(T, U, comptime m.ordered().?).getFrame;
        },
    }
}

fn selectGetFrame(dt_in: hz.DataType, dt_out: hz.DataType, mode: Mode, upconv: bool) vs.FilterGetFrame {
    switch (dt_in) {
        inline .U8, .U16, .F16, .F32 => |ti| {
            const T = comptime ti.ToType();
            switch (dt_out) {
                inline .F16, .F32 => |to| {
                    const U = comptime to.ToType();
                    if (T == U) return &passGetFrame;
                    return &ConvertFilter(T, U).getFrame;
                },
                inline .U8, .U16 => |to| {
                    const U = comptime to.ToType();
                    if (comptime @typeInfo(T) == .int) {
                        if (upconv) {
                            if (comptime @bitSizeOf(U) >= @bitSizeOf(T)) {
                                return &MulAddUpFilter(T, U).getFrame;
                            }
                            unreachable;
                        }
                    }
                    return selectModeGetFrame(T, U, mode);
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

fn ditherFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn ditherCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi = map_in.getNodeVi("clip").?;

    if (!vsh.isConstantVideoFormat(vi)) {
        map_out.setError(filter_name ++ ": clip must have a constant format and dimensions.");
        zapi.freeNode(d.node);
        return;
    }

    const dt_in = hz.DataType.select(map_out, d.node, vi, filter_name, false) catch return;

    const bitdepth = map_in.getInt(i32, "bitdepth").?;
    const dst_float_default = bitdepth == 32;
    const dst_float = map_in.getBool("sample_type") orelse dst_float_default;

    const bits_ok = if (dst_float) (bitdepth == 16 or bitdepth == 32) else (bitdepth >= 8 and bitdepth <= 16);
    if (!bits_ok) {
        map_out.setError(filter_name ++ ": unsupported bitdepth (int: 8-16, float: 16 or 32).");
        zapi.freeNode(d.node);
        return;
    }

    const imode = map_in.getInt(i32, "dither_type") orelse @intFromEnum(Mode.zimg_random);
    if (imode < 0 or imode > @intFromEnum(Mode.fmtc_quasirnd)) {
        map_out.setError(filter_name ++ ": dither_type must be 0-11.");
        zapi.freeNode(d.node);
        return;
    }
    const mode: Mode = @enumFromInt(imode);

    const src_float = vi.format.sampleType == .Float;
    d.bits_in = @intCast(vi.format.bitsPerSample);
    d.bits_out = @intCast(bitdepth);
    d.num_planes = @intCast(vi.format.numPlanes);
    d.peak_out = if (dst_float) 0.0 else @floatFromInt((@as(u32, 1) << @intCast(d.bits_out)) - 1);

    d.vi_out = vi.*;
    _ = zapi.queryVideoFormat(
        &d.vi_out.format,
        vi.format.colorFamily,
        if (dst_float) .Float else .Integer,
        bitdepth,
        vi.format.subSamplingW,
        vi.format.subSamplingH,
    );
    if (d.vi_out.format.colorFamily == .Undefined) {
        map_out.setError(filter_name ++ ": invalid output format.");
        zapi.freeNode(d.node);
        return;
    }

    const dt_out = hz.DataType.select(map_out, d.node, &d.vi_out, filter_name, false) catch return;
    const fulls_opt = map_in.getBool("fulls");
    const fulld_opt = map_in.getBool("fulld");
    const range_default = vi.format.colorFamily == .RGB;
    const fulls = fulls_opt orelse range_default;
    const fulld = fulld_opt orelse fulls;
    d.stamp_range = if (fulls_opt != null or fulld_opt != null)
        @as(i64, if (fulld) 0 else 1)
    else
        null;

    const int_up = !src_float and !dst_float and d.bits_out > d.bits_in;
    const both_limited = !fulls and !fulld;
    const full_upshift = fulls and fulld and d.bits_in == 8 and d.bits_out == 16 and mode == .round;
    const upconv = int_up and (both_limited or full_upshift);
    const is_yuv = vi.format.colorFamily == .YUV;
    for (0..3) |plane| {
        const chroma = is_yuv and plane > 0;
        const pf_in: PlaneFmt = .{ .bits = d.bits_in, .is_float = src_float, .full = fulls, .chroma = chroma };
        const pf_out: PlaneFmt = .{ .bits = d.bits_out, .is_float = dst_float, .full = fulld, .chroma = chroma };
        const ga = gainAdd(pf_in, pf_out);
        d.sc[plane] = .{ .scale = @floatCast(ga.gain), .offset = @floatCast(ga.add) };
        d.int_kernel[plane] = !src_float and !dst_float and d.bits_in > d.bits_out and
            @abs(ga.gain * @as(f64, @floatFromInt(@as(u32, 1) << @intCast(d.bits_in - d.bits_out))) - 1.0) < 1e-6 and
            @abs(ga.add) < 1e-6;
    }

    d.gain_up = 1;
    d.add_up = .{ 0, 0, 0 };
    if (upconv) {
        d.gain_up = @intFromFloat(@round(d.sc[0].scale));
        for (0..3) |plane| d.add_up[plane] = @intFromFloat(@round(-d.sc[plane].offset));
        const cmax: f64 = @floatFromInt((@as(u32, 1) << @intCast(d.bits_in)) - 1);
        std.debug.assert(cmax * @as(f64, @floatFromInt(d.gain_up)) <= @as(f64, d.peak_out));
    }

    d.err_buf_len = @as(u32, @intCast(vi.width)) + 4;

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const identity = d.bits_in == d.bits_out and src_float == dst_float and fulls == fulld;
    const gf = if (identity) &passGetFrame else selectGetFrame(dt_in, dt_out, mode, upconv);

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, &data.vi_out, gf, ditherFree, .Parallel, &deps, data);
}
