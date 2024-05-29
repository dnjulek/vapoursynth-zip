const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process = @import("../filters/bilateral.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "Bilateral";

pub const BilateralData = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    dt: helper.DataType,
    sigmaS: [3]f64,
    sigmaR: [3]f64,
    process: [3]bool,
    algorithm: [3]i32,
    PBFICnum: [3]u32,
    radius: [3]u32,
    samples: [3]u32,
    step: [3]u32,
    gr_lut: [3][]f32,
    gs_lut: [3][]f32,
    psize: u6,
    peak: f32,
    join: bool,
};

export fn bilateralGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BilateralData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
        if (d.join) {
            vsapi.?.requestFrameFilter.?(n, d.node2, frame_ctx);
        }
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node1, n, frame_ctx, core, vsapi);
        defer src.deinit();

        var ref = src;
        if (d.join) {
            ref = zapi.Frame.init(d.node2, n, frame_ctx, core, vsapi);
            defer ref.deinit();
        }

        const dst = src.newVideoFrame2(d.process);
        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            if (!(d.process[plane])) {
                continue;
            }

            const srcp = src.getReadSlice(plane);
            const refp = ref.getReadSlice(plane);
            const dstp = dst.getWriteSlice(plane);
            const w, const h, const stride = src.getDimensions(plane);

            switch (d.dt) {
                .U8 => bilateral2D(u8, srcp, refp, dstp, stride, w, h, plane, d),
                .U16 => bilateral2D(u16, srcp, refp, dstp, stride, w, h, plane, d),
                .F16 => bilateral2DFloat(f16, srcp, refp, dstp, stride, w, h, plane, d),
                .F32 => bilateral2DFloat(f32, srcp, refp, dstp, stride, w, h, plane, d),
            }
        }

        return dst.frame;
    }

    return null;
}

export fn bilateralFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *BilateralData = @ptrCast(@alignCast(instance_data));

    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        if (d.process[i]) {
            allocator.free(d.gr_lut[i]);

            if (d.algorithm[i] == 2) {
                allocator.free(d.gs_lut[i]);
            }
        }
    }

    vsapi.?.freeNode.?(d.node1);
    vsapi.?.freeNode.?(d.node2);
    allocator.destroy(d);
}

pub export fn bilateralCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: BilateralData = undefined;
    var err: vs.MapPropertyError = undefined;

    const map = zapi.Map.init(in, out, vsapi);
    d.node1, d.vi = map.getNodeVi("clip");
    d.dt = helper.DataType.select(map, d.node1, d.vi, filter_name) catch return;

    const yuv: bool = (d.vi.format.colorFamily == vs.ColorFamily.YUV);
    const peak: u32 = helper.getPeak(d.vi);
    d.peak = @floatFromInt(peak);

    var i: u32 = 0;
    var m: i32 = vsapi.?.mapNumElements.?(in, "sigmaS");
    while (i < 3) : (i += 1) {
        const ssw: i32 = d.vi.format.subSamplingW;
        const ssh: i32 = d.vi.format.subSamplingH;
        if (i < m) {
            d.sigmaS[i] = vsapi.?.mapGetFloat.?(in, "sigmaS", @as(c_int, @intCast(i)), &err);
        } else if (i == 0) {
            d.sigmaS[0] = 3.0;
        } else if ((i == 1) and (yuv) and (ssh == 1) and (ssw == 1)) {
            const j: f64 = @floatFromInt((ssh + 1) * (ssw + 1));
            d.sigmaS[1] = d.sigmaS[0] / @sqrt(j);
        } else {
            d.sigmaS[i] = d.sigmaS[i - 1];
        }

        if (d.sigmaS[i] < 0.0) {
            vsapi.?.mapSetError.?(out, "Bilateral: Invalid \"sigmaS\" assigned, must be non-negative float number");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    m = vsapi.?.mapNumElements.?(in, "sigmaR");
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.sigmaR[i] = vsapi.?.mapGetFloat.?(in, "sigmaR", @as(c_int, @intCast(i)), &err);
        } else if (i == 0) {
            d.sigmaR[i] = 0.02;
        } else {
            d.sigmaR[i] = d.sigmaR[i - 1];
        }

        if (d.sigmaR[i] < 0) {
            vsapi.?.mapSetError.?(out, "Bilateral: Invalid \"sigmaR\" assigned, must be non-negative float number");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    const n: i32 = d.vi.format.numPlanes;
    m = vsapi.?.mapNumElements.?(in, "planes");
    while (i < 3) : (i += 1) {
        if ((i > 0) and (yuv)) {
            d.process[i] = false;
        } else {
            d.process[i] = m <= 0;
        }
    }

    i = 0;
    while (i < m) : (i += 1) {
        const o: u32 = @intCast(vsapi.?.mapGetInt.?(in, "planes", @as(c_int, @intCast(i)), &err));
        if ((o < 0) or (o >= n)) {
            vsapi.?.mapSetError.?(out, "Bilateral: plane index out of range");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        if (d.process[o]) {
            vsapi.?.mapSetError.?(out, "Bilateral: plane specified twice");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        d.process[o] = true;
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.sigmaS[i] == 0.0) or (d.sigmaR[i] == 0.0)) {
            d.process[i] = false;
        }
    }

    i = 0;
    m = vsapi.?.mapNumElements.?(in, "algorithm");
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.algorithm[i] = vsh.mapGetN(i32, in, "algorithm", @intCast(i), vsapi).?;
        } else if (i == 0) {
            d.algorithm[i] = 0;
        } else {
            d.algorithm[i] = d.algorithm[i - 1];
        }

        if ((d.algorithm[i] < 0) or (d.algorithm[i] > 2)) {
            vsapi.?.mapSetError.?(out, "Bilateral: Invalid \"algorithm\" assigned, must be integer ranges in [0,2]");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    m = vsapi.?.mapNumElements.?(in, "PBFICnum");
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.PBFICnum[i] = vsh.mapGetN(u32, in, "PBFICnum", @intCast(i), vsapi).?;
        } else if (i == 0) {
            d.PBFICnum[i] = 0;
        } else {
            d.PBFICnum[i] = d.PBFICnum[i - 1];
        }

        if ((d.PBFICnum[i] < 0) or (d.PBFICnum[i] == 1) or (d.PBFICnum[i] > 256)) {
            vsapi.?.mapSetError.?(out, "Bilateral: Invalid \"PBFICnum\" assigned, must be integer ranges in [0,256] except 1");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    d.join = false;
    d.node2 = vsapi.?.mapGetNode.?(in, "ref", 0, &err);
    if (d.node2 != null) {
        d.join = true;
        const rvi: *const vs.VideoInfo = vsapi.?.getVideoInfo.?(d.node2);
        if ((d.vi.width != rvi.width) or (d.vi.height != rvi.height)) {
            vsapi.?.mapSetError.?(out, "Bilateral: input clip and clip \"ref\" must be of the same size");
            vsapi.?.freeNode.?(d.node1);
            vsapi.?.freeNode.?(d.node2);
            return;
        }
        if (d.vi.format.colorFamily != rvi.format.colorFamily) {
            vsapi.?.mapSetError.?(out, "Bilateral: input clip and clip \"ref\" must be of the same color family");
            vsapi.?.freeNode.?(d.node1);
            vsapi.?.freeNode.?(d.node2);
            return;
        }
        if ((d.vi.format.subSamplingH != rvi.format.subSamplingH) or (d.vi.format.subSamplingW != rvi.format.subSamplingW)) {
            vsapi.?.mapSetError.?(out, "Bilateral: input clip and clip \"ref\" must be of the same subsampling");
            vsapi.?.freeNode.?(d.node1);
            vsapi.?.freeNode.?(d.node2);
            return;
        }
        if (d.vi.format.bitsPerSample != rvi.format.bitsPerSample) {
            vsapi.?.mapSetError.?(out, "Bilateral: input clip and clip \"ref\" must be of the same bit depth");
            vsapi.?.freeNode.?(d.node1);
            vsapi.?.freeNode.?(d.node2);
            return;
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.process[i]) and (d.PBFICnum[i] == 0)) {
            if (d.sigmaR[i] >= 0.08) {
                d.PBFICnum[i] = 4;
            } else if (d.sigmaR[i] >= 0.015) {
                d.PBFICnum[i] = @min(16, @as(u32, @intFromFloat(4.0 * 0.08 / d.sigmaR[i] + 0.5)));
            } else {
                d.PBFICnum[i] = @min(32, @as(u32, @intFromFloat(16.0 * 0.015 / d.sigmaR[i] + 0.5)));
            }

            if ((i > 0) and yuv and (d.PBFICnum[i] % 2 == 0) and (d.PBFICnum[i] < 256)) {
                d.PBFICnum[i] += 1;
            }
        }
    }

    i = 0;
    var orad = [_]i32{ 0, 0, 0 };
    while (i < 3) : (i += 1) {
        if (d.process[i]) {
            orad[i] = @max(@as(i32, @intFromFloat(d.sigmaS[i] * 2.0 + 0.5)), 1);
            if (orad[i] < 4) {
                d.step[i] = 1;
            } else if (orad[i] < 8) {
                d.step[i] = 2;
            } else {
                d.step[i] = 3;
            }

            d.samples[i] = 1;
            d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];

            while (orad[i] * 2 > d.radius[i] * 3) {
                d.samples[i] += 1;
                d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];
                if ((d.radius[i] >= orad[i]) and (d.samples[i] > 2)) {
                    d.samples[i] -= 1;
                    d.radius[i] = 1 + (d.samples[i] - 1) * d.step[i];
                    break;
                }
            }
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if (d.process[i]) {
            if (d.algorithm[i] <= 0) {
                d.algorithm[i] = if (d.step[i] == 1) 2 else (if ((d.sigmaR[i] < 0.08) and (d.samples[i] < 5)) 2 else (if (4 * d.samples[i] * d.samples[i] <= 15 * d.PBFICnum[i]) 2 else 1));
            }
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.process[i]) and (d.algorithm[i] == 2)) {
            const upper: u32 = d.radius[i] + 1;
            d.gs_lut[i] = allocator.alloc(f32, upper * upper) catch unreachable;
            process.gaussianFunctionSpatialLUTGeneration(d.gs_lut[i], upper, d.sigmaS[i]);
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if (d.process[i]) {
            d.gr_lut[i] = allocator.alloc(f32, peak + 1) catch unreachable;
            process.gaussianFunctionRangeLUTGeneration(d.gr_lut[i], peak, d.sigmaR[i]);
        }
    }

    const data: *BilateralData = allocator.create(BilateralData) catch unreachable;
    data.* = d;

    var deps1 = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node1,
            .requestPattern = .StrictSpatial,
        },
    };

    var deps_len: c_int = deps1.len;
    var deps: [*]const vs.FilterDependency = &deps1;
    if (d.node2 != null) {
        var deps2 = [_]vs.FilterDependency{
            deps1[0],
            vs.FilterDependency{
                .source = d.node2,
                .requestPattern = if (d.vi.numFrames <= vsapi.?.getVideoInfo.?(d.node2).numFrames) .StrictSpatial else .General,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, bilateralGetFrame, bilateralFree, .Parallel, deps, deps_len, data, core);
}

fn bilateral2D(comptime T: type, src: []const u8, ref: []const u8, dst: []u8, _stride: u32, width: u32, height: u32, plane: u32, d: *BilateralData) void {
    const srcp: []const T = @as([*]const T, @ptrCast(@alignCast(src)))[0..src.len];
    const refp: []const T = @as([*]const T, @ptrCast(@alignCast(ref)))[0..ref.len];
    const dstp: []T = @as([*]T, @ptrCast(@alignCast(dst)))[0..dst.len];
    const stride: u32 = _stride >> (@sizeOf(T) >> 1);

    if (d.algorithm[plane] == 1) {
        process.bilateralAlg1(
            T,
            srcp,
            dstp,
            refp,
            stride,
            width,
            height,
            plane,
            d,
        );
    } else {
        if (d.join) {
            process.bilateralAlg2Ref(
                T,
                dstp,
                srcp,
                refp,
                d.gs_lut[plane],
                d.gr_lut[plane],
                stride,
                width,
                height,
                d.radius[plane],
                d.step[plane],
                d.peak,
            );
        } else {
            process.bilateralAlg2(
                T,
                dstp,
                srcp,
                d.gs_lut[plane],
                d.gr_lut[plane],
                stride,
                width,
                height,
                d.radius[plane],
                d.step[plane],
                d.peak,
            );
        }
    }
}

fn bilateral2DFloat(comptime T: type, src: []const u8, ref: []const u8, dst: []u8, _stride: u32, width: u32, height: u32, plane: u32, d: *BilateralData) void {
    const srcp: []const T = @as([*]const T, @ptrCast(@alignCast(src)))[0..src.len];
    const refp: []const T = @as([*]const T, @ptrCast(@alignCast(ref)))[0..ref.len];
    const dstp: []T = @as([*]T, @ptrCast(@alignCast(dst)))[0..dst.len];
    const stride: u32 = _stride >> (@sizeOf(T) >> 1);

    if (d.algorithm[plane] == 1) {
        process.bilateralAlg1Float(
            T,
            srcp,
            dstp,
            refp,
            stride,
            width,
            height,
            plane,
            d,
        );
    } else {
        if (d.join) {
            process.bilateralAlg2RefFloat(
                T,
                dstp,
                srcp,
                refp,
                d.gs_lut[plane],
                d.gr_lut[plane],
                stride,
                width,
                height,
                d.radius[plane],
                d.step[plane],
            );
        } else {
            process.bilateralAlg2Float(
                T,
                dstp,
                srcp,
                d.gs_lut[plane],
                d.gr_lut[plane],
                stride,
                width,
                height,
                d.radius[plane],
                d.step[plane],
            );
        }
    }
}
