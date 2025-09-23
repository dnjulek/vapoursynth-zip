const std = @import("std");
const math = std.math;

// const filter = @import("../filters/adg.zig");
const vszip = @import("../vszip.zig");
const hz = @import("../helper.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "ADGMask";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    shift: u4 = 0,
    peak: f32 = 0,
    scaling: f32 = 0,
};

const ranges: [256]f32 = .{ 0, -0.004257431, -0.008254769, -0.012001722, -0.015508121, -0.018783767, -0.02183859, -0.024682568, -0.027325694, -0.02977789, -0.032049183, -0.03414936, -0.036088336, -0.03787587, -0.03952162, -0.041035064, -0.042425737, -0.043702893, -0.0448757, -0.04595321, -0.04694417, -0.04785733, -0.04870123, -0.049484093, -0.050214168, -0.050899304, -0.051547382, -0.05216592, -0.052762352, -0.0533438, -0.053917404, -0.054489832, -0.055067748, -0.05565766, -0.05626576, -0.05689808, -0.0575606, -0.058258943, -0.058998678, -0.059785217, -0.060623668, -0.06151911, -0.06247639, -0.063500255, -0.0645953, -0.06576589, -0.06701631, -0.06835075, -0.06977318, -0.07128751, -0.07289747, -0.07460675, -0.07641886, -0.07833714, -0.08036503, -0.08250559, -0.08476207, -0.08713738, -0.089634515, -0.09225627, -0.09500545, -0.09788466, -0.1008966, -0.10404375, -0.10732864, -0.11075358, -0.114320934, -0.11803318, -0.12189222, -0.12590054, -0.13006012, -0.13437316, -0.13884155, -0.14346747, -0.14825279, -0.15319955, -0.15830952, -0.16358478, -0.16902699, -0.17463808, -0.18041988, -0.18637426, -0.19250283, -0.1988074, -0.20528978, -0.21195163, -0.21879464, -0.22582062, -0.23303142, -0.24042855, -0.24801347, -0.25578836, -0.2637546, -0.27191436, -0.2802685, -0.28881967, -0.2975689, -0.3065182, -0.31566918, -0.32502395, -0.33458376, -0.34435064, -0.35432598, -0.36451226, -0.37491056, -0.38552338, -0.396352, -0.40739894, -0.41866496, -0.4301528, -0.44186383, -0.4538007, -0.4659645, -0.4783577, -0.4909822, -0.50384, -0.5169327, -0.5302632, -0.5438324, -0.5576428, -0.5716973, -0.5859968, -0.60054374, -0.6153412, -0.63039017, -0.6456929, -0.6612515, -0.6770693, -0.69314754, -0.70948756, -0.72609353, -0.7429665, -0.76010865, -0.777522, -0.7952088, -0.81317216, -0.83141375, -0.8499343, -0.86873764, -0.88782495, -0.9071999, -0.92686224, -0.9468147, -0.9670592, -0.98759806, -1.0084318, -1.0295634, -1.0509933, -1.0727249, -1.0947567, -1.1170919, -1.1397297, -1.1626734, -1.1859213, -1.2094755, -1.233334, -1.2575015, -1.2819697, -1.3067466, -1.3318251, -1.357207, -1.3828892, -1.4088709, -1.435149, -1.4617155, -1.4885731, -1.5157181, -1.5431428, -1.5708361, -1.598799, -1.6270252, -1.6555051, -1.6842229, -1.7131745, -1.742353, -1.7717404, -1.8013232, -1.8310845, -1.8610176, -1.8911027, -1.9213127, -1.9516343, -1.9820459, -2.0125256, -2.043041, -2.073575, -2.1040785, -2.134558, -2.1649425, -2.1952279, -2.2253582, -2.2553155, -2.2850382, -2.3145134, -2.3436801, -2.3725085, -2.4009445, -2.4289598, -2.4564931, -2.4835224, -2.509986, -2.5358584, -2.5610871, -2.585641, -2.6094983, -2.6325924, -2.6549327, -2.67646, -2.697193, -2.7170916, -2.7361383, -2.7543614, -2.771728, -2.7882915, -2.8040392, -2.8190002, -2.8332214, -2.8467426, -2.859582, -2.8718457, -2.8835657, -2.8948002, -2.905658, -2.91623, -2.926573, -2.9368489, -2.9471292, -2.9575531, -2.9682052, -2.979286, -2.990905, -3.0032525, -3.016443, -3.0307305, -3.0462928, -3.0633898, -3.0821753, -3.1030161, -3.126214, -3.152117, -3.1811666, -3.2137668, -3.2505846, -3.2923207, -3.3397686, -3.3940463, -3.45649, -3.5288575, -3.6135817, -3.713947, -3.8349154, -3.9840882, -4.174093, -4.428615, -4.80098, -5.4611893 };

pub fn process(comptime T: type, src: []const T, dst: []T, shift: u4, scaling: f32, peak: f32) void {
    const LT: type = if (@typeInfo(T) == .int) T else u16;
    var lut: [256]LT = undefined;

    for (&lut, ranges) |*lx, rx| {
        const x: f32 = @min(@max(@mulAdd(f32, @exp(scaling * rx), peak, 0.5), 0), peak);
        lx.* = @intFromFloat(x);
    }

    if (@typeInfo(T) == .int) {
        for (src, dst) |sx, *dx| {
            dx.* = lut[@as(usize, sx) >> shift];
        }
    } else {
        for (src, dst) |sx, *dx| {
            const idx = @as(usize, @intFromFloat(sx * peak)) >> shift;
            dx.* = @as(T, @floatFromInt(lut[idx])) / peak;
        }
    }
}

fn ADG(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame();

                const avg = src.getPropertiesRO().getValue(f32, "PlaneStatsAverage").?;
                const scaling = avg * avg * d.scaling;

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    process(T, srcp, dstp, d.shift, scaling, d.peak);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    _ = map_out; // autofix
    d.node, d.vi = map_in.getNodeVi("clip").?;

    d.scaling = map_in.getValue(f32, "luma_scaling") orelse 8;
    d.shift = @intCast(@min(d.vi.format.bitsPerSample, 16) - 8);
    d.peak = hz.getPeakValue(&d.vi.format, false, .FULL);
    if (d.vi.format.sampleType == .Float) {
        d.peak = std.math.maxInt(u16);
    }

    const args = zapi.createZMap();
    _ = args.consumeNode("clipa", d.node, .Replace);
    var ret = args.invoke(zapi.getPluginByID2(.Std), "PlaneStats");
    d.node = ret.getNode("clip");
    ret.free();
    args.free();

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = switch (d.vi.format.bytesPerSample) {
        1 => &ADG(u8).getFrame,
        2 => &ADG(u16).getFrame,
        4 => &ADG(f32).getFrame,
        else => unreachable,
    };

    zapi.createVideoFilter(out, filter_name, d.vi, gf, free, .Parallel, &deps, data);
}
