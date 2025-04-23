const std = @import("std");
const math = std.math;

const filter = @import("../filters/planeaverage.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneAverage";

const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    exclude: filter.Exclude = undefined,
    peak: f32 = 0,
    planes: [3]bool = .{ true, false, false },
    prop: StringProp = undefined,
};

const StringProp = struct {
    d: [:0]u8,
    a: [:0]u8,
};

fn PlaneAverage(comptime T: type, comptime refb: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node1, frame_ctx);
                if (refb) {
                    zapi.requestFrameFilter(n, d.node2, frame_ctx);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node1, n, frame_ctx, core);
                const ref = if (refb) zapi.initZFrame(d.node2, n, frame_ctx, core);
                const dst = src.copyFrame();
                const props = dst.getPropertiesRW();
                props.deleteKey(d.prop.d);
                props.deleteKey(d.prop.a);

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) {
                        continue;
                    }

                    const srcp = src.getReadSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    var avg: f64 = undefined;

                    if (refb) {
                        const refp = ref.getReadSlice2(T, plane);
                        const stats = filter.averageRef(T, srcp, refp, stride, w, h, d.exclude, d.peak);
                        props.setFloat(d.prop.d, stats.diff, .Append);
                        avg = stats.avg;
                    } else {
                        avg = filter.average(T, srcp, stride, w, h, d.exclude, d.peak);
                    }
                    props.setFloat(d.prop.a, avg, .Append);
                }

                src.deinit();
                if (refb) ref.deinit();

                return dst.frame;
            }

            return null;
        }
    };
}

export fn planeAverageFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi);

    switch (d.exclude) {
        .i => allocator.free(d.exclude.i),
        .f => allocator.free(d.exclude.f),
    }

    allocator.free(d.prop.d);
    allocator.free(d.prop.a);

    if (d.node2) |node| zapi.freeNode(node);
    zapi.freeNode(d.node1);
    allocator.destroy(d);
}

pub export fn planeAverageCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node1, d.vi = map_in.getNodeVi("clipa");
    const dt = hz.DataType.select(map_out, d.node1, d.vi, filter_name) catch return;

    d.node2 = map_in.getNode("clipb");
    const refb = d.node2 != null;
    const nodes = [_]?*vs.Node{ d.node1, d.node2 };
    if (refb) {
        hz.compareNodes(map_out, &nodes, .BIGGER_THAN, filter_name, &zapi) catch return;
    }

    hz.mapGetPlanes(map_in, map_out, &nodes, &d.planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;
    d.peak = @floatFromInt(hz.getPeak(d.vi));

    const prop_in = map_in.getData("prop", 0) orelse "psm";
    d.prop = .{
        .d = std.fmt.allocPrintZ(allocator, "{s}Diff", .{prop_in}) catch unreachable,
        .a = std.fmt.allocPrintZ(allocator, "{s}Avg", .{prop_in}) catch unreachable,
    };

    const exclude_in = map_in.getIntArray("exclude");
    if (exclude_in) |ein| {
        if (d.vi.format.sampleType == .Float) {
            d.exclude = filter.Exclude{ .f = allocator.alloc(f32, ein.len) catch unreachable };
            for (d.exclude.f, ein) |*df, *ei| {
                df.* = @floatFromInt(ei.*);
            }
        } else {
            d.exclude = filter.Exclude{ .i = allocator.alloc(i32, ein.len) catch unreachable };
            for (d.exclude.i, ein) |*di, *ei| {
                di.* = math.lossyCast(i32, ei.*);
            }
        }
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const rp2: vs.RequestPattern = if (refb and (d.vi.numFrames <= zapi.getVideoInfo(d.node2).numFrames)) .StrictSpatial else .FrameReuseLastOnly;
    const deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = rp2 },
    };

    const getFrame = switch (dt) {
        .U8 => if (refb) &PlaneAverage(u8, true).getFrame else &PlaneAverage(u8, false).getFrame,
        .U16 => if (refb) &PlaneAverage(u16, true).getFrame else &PlaneAverage(u16, false).getFrame,
        .F16 => if (refb) &PlaneAverage(f16, true).getFrame else &PlaneAverage(f16, false).getFrame,
        .F32 => if (refb) &PlaneAverage(f32, true).getFrame else &PlaneAverage(f32, false).getFrame,
    };

    const ndeps: usize = if (refb) 2 else 1;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, planeAverageFree, .Parallel, deps[0..ndeps], data, core);
}
