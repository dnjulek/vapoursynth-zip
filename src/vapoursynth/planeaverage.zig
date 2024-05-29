const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const filter = @import("../filters/planeaverage.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneAverage";

const Data = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    exclude: filter.Exclude,
    peak: f32,
    planes: [3]bool,
    prop_buff: ?[]u8,
    prop: ?StringProp,
};

const StringProp = struct {
    d: []u8,
    a: []u8,
};

fn PlaneAverage(comptime T: type, comptime refb: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node1, frame_ctx);
                if (refb) {
                    vsapi.?.requestFrameFilter.?(n, d.node2.?, frame_ctx);
                }
            } else if (activation_reason == .AllFramesReady) {
                var src = zapi.Frame.init(d.node1, n, frame_ctx, core, vsapi);
                defer src.deinit();

                var ref: ?zapi.Frame = null;
                if (refb) {
                    ref = zapi.Frame.init(d.node2.?, n, frame_ctx, core, vsapi);
                    defer ref.?.deinit();
                }

                const dst = src.copyFrame();
                const props = dst.getPropertiesRW();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) {
                        continue;
                    }

                    const srcp = src.getReadSlice(plane);
                    const w, const h, const stride = src.getDimensions(plane);
                    var avg: f64 = undefined;

                    if (refb) {
                        const refp = ref.?.getReadSlice(plane);
                        const stats = filter.averageRef(T, srcp, refp, stride, w, h, d.exclude, d.peak);
                        _ = vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.d.ptr else "psmDiff", stats.diff, .Append);
                        avg = stats.avg;
                    } else {
                        avg = filter.average(T, srcp, stride, w, h, d.exclude, d.peak);
                    }
                    _ = vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.a.ptr else "psmAvg", avg, .Append);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

export fn planeAverageFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    switch (d.exclude) {
        .i => allocator.free(d.exclude.i),
        .f => allocator.free(d.exclude.f),
    }

    if (d.node2) |node| vsapi.?.freeNode.?(node);
    if (d.prop_buff) |buff| allocator.free(buff);
    vsapi.?.freeNode.?(d.node1);
    allocator.destroy(d);
}

pub export fn planeAverageCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;

    var map = zapi.Map.init(in, out, vsapi);
    d.node1, d.vi = map.getNodeVi("clipa");
    const dt = helper.DataType.select(map, d.node1, d.vi, filter_name) catch return;

    d.node2, const vi2 = map.getNodeVi("clipb");
    helper.compareNodes(out, d.node1, d.node2, d.vi, vi2, filter_name, vsapi) catch return;

    d.peak = @floatFromInt(helper.getPeak(d.vi));
    var nodes = [_]?*vs.Node{ d.node1, d.node2 };
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.prop = getPropData(map, "prop", allocator, &d.prop_buff);
    const exclude_in = map.getIntArray("exclude");
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
                .requestPattern = if (d.vi.numFrames <= vi2.numFrames) .StrictSpatial else .General,
            },
        };

        deps_len = deps2.len;
        deps = &deps2;
    }

    const refb = d.node2 != null;
    const getFrame = switch (dt) {
        .U8 => if (refb) &PlaneAverage(u8, true).getFrame else &PlaneAverage(u8, false).getFrame,
        .U16 => if (refb) &PlaneAverage(u16, true).getFrame else &PlaneAverage(u16, false).getFrame,
        .F16 => if (refb) &PlaneAverage(f16, true).getFrame else &PlaneAverage(f16, false).getFrame,
        .F32 => if (refb) &PlaneAverage(f32, true).getFrame else &PlaneAverage(f32, false).getFrame,
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, planeAverageFree, .Parallel, deps, deps_len, data, core);
}

pub fn getPropData(map: zapi.Map, comptime key: []const u8, data_allocator: std.mem.Allocator, data_buff: *?[]u8) ?StringProp {
    var err: vs.MapPropertyError = undefined;
    data_buff.* = null;
    const data_ptr = map.vsapi.?.mapGetData.?(map.in, key.ptr, 0, &err);
    if (err != .Success) {
        return null;
    }

    const data_len = map.vsapi.?.mapGetDataSize.?(map.in, key.ptr, 0, &err);
    if ((err != .Success) or (data_len < 1)) {
        return null;
    }

    const udata_len: u32 = @bitCast(data_len);
    const diff_len = udata_len + 5;
    const avg_len = udata_len + 4;
    const data = data_ptr[0..udata_len];
    data_buff.* = data_allocator.alloc(u8, diff_len + avg_len) catch unreachable;
    return .{
        .d = std.fmt.bufPrint(data_buff.*.?[0..diff_len], "{s}Diff\x00", .{data}) catch unreachable,
        .a = std.fmt.bufPrint(data_buff.*.?[diff_len..], "{s}Avg\x00", .{data}) catch unreachable,
    };
}
