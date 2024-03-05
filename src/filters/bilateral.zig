const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process = @import("process/bilateral.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const math = std.math;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const pe = vs.MapPropertyError;
const ma = vs.MapAppendMode;
const st = vs.SampleType;

const allocator = std.heap.c_allocator;
pub const filter_name = "Bilateral";

pub const BilateralData = struct {
    node1: *vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    dt: helper.DataType,
    sigmaS: [3]f64,
    sigmaR: [3]f64,
    planes: [3]bool,
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

fn bilateral2D(comptime T: type, src: [*]const u8, ref: [*]const u8, dst: [*]u8, _stride: usize, width: u32, height: u32, plane: u32, d: *BilateralData) void {
    const srcp: [*]const T = @as([*]const T, @ptrCast(@alignCast(src)));
    const refp: [*]const T = @as([*]const T, @ptrCast(@alignCast(ref)));
    const dstp: [*]T = @as([*]T, @ptrCast(@alignCast(dst)));
    const stride: usize = _stride >> (@sizeOf(T) >> 1);

    if (d.algorithm[plane] == 1) {
        process.Bilateral2D_1(
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
            process.Bilateral2D_2ref(
                T,
                dstp,
                srcp,
                refp,
                d.gs_lut[plane].ptr,
                d.gr_lut[plane].ptr,
                stride,
                width,
                height,
                d.radius[plane],
                d.step[plane],
                d.peak,
            );
        } else {
            process.Bilateral2D_2(
                T,
                dstp,
                srcp,
                d.gs_lut[plane].ptr,
                d.gr_lut[plane].ptr,
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

export fn bilateralGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BilateralData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
        if (d.join) {
            vsapi.?.requestFrameFilter.?(n, d.node2, frame_ctx);
        }
    } else if (activation_reason == ar.AllFramesReady) {
        const src = vsapi.?.getFrameFilter.?(n, d.node1, frame_ctx);
        defer vsapi.?.freeFrame.?(src);
        var ref = src;
        if (d.join) {
            ref = vsapi.?.getFrameFilter.?(n, d.node2, frame_ctx);
            defer vsapi.?.freeFrame.?(ref);
        }

        const dst = helper.newVideoFrame2(src, &d.planes, core, vsapi);
        var plane: c_int = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const uplane: u32 = @intCast(plane);
            if (!(d.planes[uplane])) {
                continue;
            }

            const srcp: [*]const u8 = vsapi.?.getReadPtr.?(src, plane);
            const refp: [*]const u8 = vsapi.?.getReadPtr.?(ref, plane);
            const dstp: [*]u8 = vsapi.?.getWritePtr.?(dst, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: u32 = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: u32 = @intCast(vsapi.?.getFrameWidth.?(src, plane));
            switch (d.dt) {
                .U8 => bilateral2D(u8, srcp, refp, dstp, stride, w, h, uplane, d),
                .U16 => bilateral2D(u16, srcp, refp, dstp, stride, w, h, uplane, d),
                .F32 => bilateral2D(u16, srcp, refp, dstp, stride, w, h, uplane, d),
            }
        }

        return dst;
    }

    return null;
}

export fn bilateralFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *BilateralData = @ptrCast(@alignCast(instance_data));

    vsapi.?.freeNode.?(d.node1);
    vsapi.?.freeNode.?(d.node2);
    allocator.destroy(d);
}

pub export fn bilateralCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: BilateralData = undefined;
    var err: pe = undefined;

    d.node1 = vsapi.?.mapGetNode.?(in, "clip", 0, &err).?;
    d.vi = vsapi.?.getVideoInfo.?(d.node1);
    d.dt = @enumFromInt(d.vi.format.bytesPerSample);

    const yuv: bool = (d.vi.format.colorFamily == vs.ColorFamily.YUV);
    const bps = d.vi.format.bitsPerSample;
    const peak: u32 = math.shl(u32, 1, bps);
    d.peak = @floatFromInt(peak);

    if ((d.vi.format.sampleType != st.Integer) or ((d.vi.format.bytesPerSample != 1) and (d.vi.format.bytesPerSample != 2))) {
        vsapi.?.mapSetError.?(out, "Bilateral: Invalid input clip, Only 8-16 bit int formats supported");
        vsapi.?.freeNode.?(d.node1);
        return;
    }

    var i: usize = 0;
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
            d.planes[i] = false;
        } else {
            d.planes[i] = m <= 0;
        }
    }

    i = 0;
    while (i < m) : (i += 1) {
        const o: usize = @intCast(vsapi.?.mapGetInt.?(in, "planes", @as(c_int, @intCast(i)), &err));
        if ((o < 0) or (o >= n)) {
            vsapi.?.mapSetError.?(out, "Bilateral: plane index out of range");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        if (d.planes[o]) {
            vsapi.?.mapSetError.?(out, "Bilateral: plane specified twice");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        d.planes[o] = true;
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.sigmaS[i] == 0.0) or (d.sigmaR[i] == 0.0)) {
            d.planes[i] = false;
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
        if ((d.planes[i]) and (d.PBFICnum[i] == 0)) {
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
        if (d.planes[i]) {
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
        if (d.algorithm[i] <= 0) {
            d.algorithm[i] = if (d.step[i] == 1) 2 else (if ((d.sigmaR[i] < 0.08) and (d.samples[i] < 5)) 2 else (if (4 * d.samples[i] * d.samples[i] <= 15 * d.PBFICnum[i]) 2 else 1));
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.planes[i]) and (d.algorithm[i] == 2)) {
            const upper: usize = d.radius[i] + 1;
            d.gs_lut[i] = allocator.alloc(f32, upper * upper) catch unreachable;
            process.gaussianFunctionSpatialLUTGeneration(d.gs_lut[i].ptr, upper, d.sigmaS[i]);
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if (d.planes[i]) {
            d.gr_lut[i] = allocator.alloc(f32, peak + 1) catch unreachable;
            process.gaussianFunctionRangeLUTGeneration(d.gr_lut[i].ptr, peak, d.sigmaR[i]);
        }
    }

    const data: *BilateralData = allocator.create(BilateralData) catch unreachable;
    data.* = d;

    var deps1 = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node1,
            .requestPattern = rp.StrictSpatial,
        },
    };

    var deps_len: c_int = deps1.len;
    var deps: [*]const vs.FilterDependency = &deps1;
    if (d.node2 != null) {
        var deps2 = [_]vs.FilterDependency{
            deps1[0],
            vs.FilterDependency{
                .source = d.node2,
                .requestPattern = if (d.vi.numFrames <= vsapi.?.getVideoInfo.?(d.node2).numFrames) rp.StrictSpatial else rp.General,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, bilateralGetFrame, bilateralFree, fm.Parallel, deps, deps_len, data, core);
}
