const std = @import("std");
const math = std.math;

const filter = @import("../filters/adg.zig");
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

                const props = src.getPropertiesRO();
                const avg = props.getValue(f32, "PlaneStatsAverage").?;
                const scaling = avg * avg * d.scaling;

                const srcp = src.getReadSlice2(T, 0);
                const dstp = dst.getWriteSlice2(T, 0);

                if (@typeInfo(T) == .float) {
                    const min = props.getValue(f32, "PlaneStatsMin").?;
                    const max = props.getValue(f32, "PlaneStatsMax").?;

                    if (max > 1.0 or min < 0.0) {
                        filter.processFloatClamp(T, srcp, dstp, d.shift, scaling, d.peak);
                    } else {
                        filter.processFloat(T, srcp, dstp, d.shift, scaling, d.peak);
                    }
                } else {
                    filter.processInt(T, srcp, dstp, d.shift, scaling, d.peak);
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
