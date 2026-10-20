const std = @import("std");

const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const filter = @import("../filters/dither.zig");
const errdiff = @import("../filters/dither_errdiff.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsc = vapoursynth.vsconstants;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Dither";

pub const Mode = filter.Mode;

const Data = struct {
    node: *vs.Node,
    vi_out: vs.VideoInfo,
    num_planes: u32,
    cfg: filter.Config,
    stamp_range: ?vsc.Range,
};

fn stampRange(dst: anytype, d: *const Data) void {
    if (d.stamp_range) |r| {
        const props = dst.getPropertiesRW();
        props.setRange(r);
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

                    filter.Ordered(T, U, mode).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.cfg.sc[plane], d.cfg.peak_out, plane, d.cfg.bits_in, d.cfg.bits_out, d.cfg.int_kernel[plane]);
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

                const err_buf = errdiff.ErrorBuffer.init(d.cfg.err_buf_len) catch return hz.frameOom(&zapi, filter_name, dst);
                defer err_buf.deinit();

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);
                    err_buf.clear();

                    if (@typeInfo(T) == .int and d.cfg.int_kernel[plane]) {
                        errdiff.Fmtc(T, U, mode).processPlaneInt(srcp, dstp, w, h, src_stride, dst_stride, errdiff.IntQuant.init(d.cfg.bits_in, d.cfg.bits_out), &err_buf);
                    } else {
                        const dif_bits = @as(i32, d.cfg.bits_in) - @as(i32, d.cfg.bits_out);
                        errdiff.Fmtc(T, U, mode).processPlaneFloat(srcp, dstp, w, h, src_stride, dst_stride, d.cfg.sc[plane].scale, d.cfg.sc[plane].offset, d.cfg.peak_out, dif_bits, &err_buf);
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

                const error_top = allocator.alloc(f32, d.cfg.err_buf_len) catch unreachable;
                defer allocator.free(error_top);
                const error_cur = allocator.alloc(f32, d.cfg.err_buf_len) catch unreachable;
                defer allocator.free(error_cur);

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(U, plane);
                    const w, const h, const src_stride = src.getDimensions2(T, plane);
                    _, _, const dst_stride = dst.getDimensions2(U, plane);

                    errdiff.Zimg(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.cfg.sc[plane].scale, d.cfg.sc[plane].offset, d.cfg.peak_out, error_top, error_cur);
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

                    filter.MulAddUp(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.cfg.gain_up, d.cfg.add_up[plane]);
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

                    filter.Convert(T, U).processPlane(srcp, dstp, w, h, src_stride, dst_stride, d.cfg.sc[plane]);
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

    const imode = map_in.getInt(i32, "dither_type") orelse @backingInt(Mode.zimg_random);
    if (imode < 0 or imode > @backingInt(Mode.fmtc_quasirnd)) {
        map_out.setError(filter_name ++ ": dither_type must be 0-11.");
        zapi.freeNode(d.node);
        return;
    }
    const mode: Mode = @fromBackingInt(@intCast(imode));

    const src_float = vi.format.sampleType == .Float;
    d.num_planes = @intCast(vi.format.numPlanes);

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
        @as(vsc.Range, if (fulld) .FULL else .LIMITED)
    else
        null;

    d.cfg = filter.Config.init(.{
        .bits_in = @intCast(vi.format.bitsPerSample),
        .src_float = src_float,
        .bits_out = @intCast(bitdepth),
        .dst_float = dst_float,
        .fulls = fulls,
        .fulld = fulld,
        .mode = mode,
        .is_yuv = vi.format.colorFamily == .YUV,
        .width = @intCast(vi.width),
    });

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const gf = if (d.cfg.identity) &passGetFrame else selectGetFrame(dt_in, dt_out, mode, d.cfg.upconv);

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, &data.vi_out, gf, ditherFree, .Parallel, &deps, data);
}
