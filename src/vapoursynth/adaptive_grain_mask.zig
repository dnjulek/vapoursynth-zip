const std = @import("std");

const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const filter = @import("../filters/adaptive_grain_mask.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "AdaptiveGrainMask";

const Data = struct {
    node: *vs.Node,
    vi: vs.VideoInfo,
    luma_scaling: f32,
    peak: f32,
    shift: u5,
};

fn AdaptiveGrainMask(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame3(.{ .format = &d.vi.format });

                const srcp = src.getReadSlice2(T, 0);
                const dstp = dst.getWriteSlice2(T, 0);
                const w, const h, const stride = src.getDimensions2(T, 0);

                const avg = filter.computeAverage(T, srcp, stride, w, h, d.peak);
                const temp = filter.calcLumaScaling(avg, d.luma_scaling);

                if (comptime @typeInfo(T) == .float) {
                    filter.processFloat(srcp, dstp, temp);
                } else {
                    var lut: [256]T = undefined;
                    filter.buildLut(T, &lut, temp, d.peak);
                    if (comptime T == u8) {
                        std.debug.assert(d.shift == 0);
                        filter.processIntU8(srcp, dstp, &lut);
                    } else {
                        filter.processInt(T, srcp, dstp, &lut, d.shift);
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn adaptiveGrainMaskFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn adaptiveGrainMaskCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const in_vi = map_in.getNodeVi("clip").?;

    if (!vsh.isConstantVideoFormat(in_vi)) {
        map_out.setError(filter_name ++ ": clip must have a constant format and dimensions.");
        zapi.freeNode(d.node);
        return;
    }

    const dt = hz.DataType.select(map_out, d.node, in_vi, filter_name, false) catch return;
    if (dt == .F16) {
        map_out.setError(filter_name ++ ": half precision float input is not supported.");
        zapi.freeNode(d.node);
        return;
    }

    const bits: u6 = @intCast(in_vi.format.bitsPerSample);
    if (dt == .F32) {
        d.peak = 1.0;
        d.shift = 0;
    } else {
        d.peak = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
        d.shift = @intCast(if (bits > 8) bits - 8 else 0);
    }

    d.luma_scaling = @floatCast(map_in.getValue(f64, "luma_scaling") orelse 10.0);

    d.vi = in_vi.*;
    _ = zapi.queryVideoFormat(&d.vi.format, .Gray, in_vi.format.sampleType, @intCast(bits), 0, 0);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const getFrameFn = switch (dt) {
        .U8 => &AdaptiveGrainMask(u8).getFrame,
        .U16 => &AdaptiveGrainMask(u16).getFrame,
        .F32 => &AdaptiveGrainMask(f32).getFrame,
        else => unreachable,
    };

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, &data.vi, getFrameFn, adaptiveGrainMaskFree, .Parallel, &deps, data);
}
