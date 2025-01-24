const std = @import("std");
const math = std.math;

const filter = @import("../filters/bilateral.zig");
const helper = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;

const allocator = std.heap.c_allocator;
pub const filter_name = "Bilateral";

pub const Data = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
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

fn Bilateral(comptime T: type, comptime join: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
                if (join) {
                    vsapi.?.requestFrameFilter.?(n, d.node2, frame_ctx);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.ZFrame.init(d.node1, n, frame_ctx, core, vsapi);
                defer src.deinit();

                var ref = src;
                if (join) {
                    ref = zapi.ZFrame.init(d.node2, n, frame_ctx, core, vsapi);
                    defer ref.deinit();
                }

                const dst = src.newVideoFrame2(d.process);
                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.process[plane])) {
                        continue;
                    }

                    const srcp = src.getReadSlice2(T, plane);
                    const refp = if (join) ref.getReadSlice2(T, plane) else srcp;
                    const dstp = dst.getWriteSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    filter.bilateral(T, srcp, refp, dstp, stride, w, h, plane, join, d);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

export fn bilateralFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));

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
    var d: Data = undefined;

    const map_in = zapi.ZMap.init(in, vsapi);
    const map_out = zapi.ZMap.init(out, vsapi);
    d.node1, d.vi = map_in.getNodeVi("clip");
    const dt = helper.DataType.select(map_out, d.node1, d.vi, filter_name) catch return;

    const yuv: bool = (d.vi.format.colorFamily == vs.ColorFamily.YUV);
    const peak: u32 = helper.getPeak(d.vi);
    d.peak = @floatFromInt(peak);

    var i: u32 = 0;
    var m = map_in.numElements("sigmaS") orelse 0;
    while (i < 3) : (i += 1) {
        const ssw: i32 = d.vi.format.subSamplingW;
        const ssh: i32 = d.vi.format.subSamplingH;
        if (i < m) {
            d.sigmaS[i] = map_in.getFloat2(f64, "sigmaS", i).?;
        } else if (i == 0) {
            d.sigmaS[0] = 3;
        } else if ((i == 1) and (yuv) and (ssh == 1) and (ssw == 1)) {
            const j: f64 = @floatFromInt((ssh + 1) * (ssw + 1));
            d.sigmaS[1] = d.sigmaS[0] / @sqrt(j);
        } else {
            d.sigmaS[i] = d.sigmaS[i - 1];
        }

        if (d.sigmaS[i] < 0) {
            map_out.setError("Bilateral: Invalid \"sigmaS\" assigned, must be non-negative float number");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    m = map_in.numElements("sigmaR") orelse 0;
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.sigmaR[i] = map_in.getFloat2(f64, "sigmaR", i).?;
        } else if (i == 0) {
            d.sigmaR[i] = 0.02;
        } else {
            d.sigmaR[i] = d.sigmaR[i - 1];
        }

        if (d.sigmaR[i] < 0) {
            map_out.setError("Bilateral: Invalid \"sigmaR\" assigned, must be non-negative float number");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    const n: i32 = d.vi.format.numPlanes;
    m = map_in.numElements("planes") orelse 0;
    while (i < 3) : (i += 1) {
        if ((i > 0) and (yuv)) {
            d.process[i] = false;
        } else {
            d.process[i] = m <= 0;
        }
    }

    i = 0;
    while (i < m) : (i += 1) {
        const o = map_in.getInt2(u32, "planes", i).?;
        if ((o < 0) or (o >= n)) {
            map_out.setError("Bilateral: plane index out of range");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        if (d.process[o]) {
            map_out.setError("Bilateral: plane specified twice");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
        d.process[o] = true;
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.sigmaS[i] == 0) or (d.sigmaR[i] == 0)) {
            d.process[i] = false;
        }
    }

    i = 0;
    m = map_in.numElements("algorithm") orelse 0;
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.algorithm[i] = map_in.getInt2(i32, "algorithm", i).?;
        } else if (i == 0) {
            d.algorithm[i] = 0;
        } else {
            d.algorithm[i] = d.algorithm[i - 1];
        }

        if ((d.algorithm[i] < 0) or (d.algorithm[i] > 2)) {
            map_out.setError("Bilateral: Invalid \"algorithm\" assigned, must be integer ranges in [0,2]");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    i = 0;
    m = map_in.numElements("PBFICnum") orelse 0;
    while (i < 3) : (i += 1) {
        if (i < m) {
            d.PBFICnum[i] = map_in.getInt2(u32, "PBFICnum", i).?;
        } else if (i == 0) {
            d.PBFICnum[i] = 0;
        } else {
            d.PBFICnum[i] = d.PBFICnum[i - 1];
        }

        if ((d.PBFICnum[i] < 0) or (d.PBFICnum[i] == 1) or (d.PBFICnum[i] > 256)) {
            map_out.setError("Bilateral: Invalid \"PBFICnum\" assigned, must be integer ranges in [0,256] except 1");
            vsapi.?.freeNode.?(d.node1);
            return;
        }
    }

    d.node2 = map_in.getNode("ref");
    d.join = d.node2 != null;
    if (d.join) {
        const nodes = [_]?*vs.Node{ d.node1, d.node2 };
        helper.compareNodes(map_out, &nodes, .BIGGER_THAN, filter_name, vsapi) catch return;
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if ((d.process[i]) and (d.PBFICnum[i] == 0)) {
            if (d.sigmaR[i] >= 0.08) {
                d.PBFICnum[i] = 4;
            } else if (d.sigmaR[i] >= 0.015) {
                d.PBFICnum[i] = @min(16, @as(u32, @intFromFloat(4 * 0.08 / d.sigmaR[i] + 0.5)));
            } else {
                d.PBFICnum[i] = @min(32, @as(u32, @intFromFloat(16 * 0.015 / d.sigmaR[i] + 0.5)));
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
            orad[i] = @max(@as(i32, @intFromFloat(d.sigmaS[i] * 2 + 0.5)), 1);
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
            filter.gaussianFunctionSpatialLUTGeneration(d.gs_lut[i], upper, d.sigmaS[i]);
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if (d.process[i]) {
            d.gr_lut[i] = allocator.alloc(f32, peak + 1) catch unreachable;
            filter.gaussianFunctionRangeLUTGeneration(d.gr_lut[i], peak, d.sigmaR[i]);
        }
    }

    const data: *Data = allocator.create(Data) catch unreachable;
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
                .requestPattern = .StrictSpatial,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    const getFrame = switch (dt) {
        .U8 => if (d.join) &Bilateral(u8, true).getFrame else &Bilateral(u8, false).getFrame,
        .U16 => if (d.join) &Bilateral(u16, true).getFrame else &Bilateral(u16, false).getFrame,
        .F16 => if (d.join) &Bilateral(f16, true).getFrame else &Bilateral(f16, false).getFrame,
        .F32 => if (d.join) &Bilateral(f32, true).getFrame else &Bilateral(f32, false).getFrame,
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, bilateralFree, .Parallel, deps, deps_len, data, core);
}
