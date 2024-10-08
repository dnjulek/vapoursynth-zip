const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const filter = @import("../filters/planeminmax.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneMinMax";

pub const Data = struct {
    node1: ?*vs.Node,
    node2: ?*vs.Node,
    vi: *const vs.VideoInfo,
    peak: u16,
    minthr: f32,
    maxthr: f32,
    hist_size: u32,
    planes: [3]bool,
    prop_buff: ?[]u8,
    prop: ?StringProp,
};

const StringProp = struct {
    d: []u8,
    ma: []u8,
    mi: []u8,
};

fn PlaneMinMax(comptime T: type, comptime refb: bool) type {
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
                    var stats: filter.Stats = undefined;
                    if (refb) {
                        const refp = ref.?.getReadSlice(plane);
                        stats = if (@typeInfo(T) == .int) filter.minMaxIntRef(T, srcp, refp, stride, w, h, d) else filter.minMaxFloatRef(T, srcp, refp, stride, w, h, d);

                        _ = switch (stats) {
                            .i => vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.d.ptr else "psmDiff", stats.i.diff, .Append),
                            .f => vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.d.ptr else "psmDiff", stats.f.diff, .Append),
                        };
                    } else {
                        stats = if (@typeInfo(T) == .int) filter.minMaxInt(T, srcp, stride, w, h, d) else filter.minMaxFloat(T, srcp, stride, w, h, d);
                    }

                    switch (stats) {
                        .i => {
                            _ = vsapi.?.mapSetInt.?(props, if (d.prop != null) d.prop.?.ma.ptr else "psmMax", stats.i.max, .Append);
                            _ = vsapi.?.mapSetInt.?(props, if (d.prop != null) d.prop.?.mi.ptr else "psmMin", stats.i.min, .Append);
                        },
                        .f => {
                            _ = vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.ma.ptr else "psmMax", stats.f.max, .Append);
                            _ = vsapi.?.mapSetFloat.?(props, if (d.prop != null) d.prop.?.mi.ptr else "psmMin", stats.f.min, .Append);
                        },
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

export fn planeMinMaxFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    if (d.node2) |node| vsapi.?.freeNode.?(node);
    if (d.prop_buff) |buff| allocator.free(buff);
    vsapi.?.freeNode.?(d.node1);
    allocator.destroy(d);
}

pub export fn planeMinMaxCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;

    var map = zapi.Map.init(in, out, vsapi);
    d.node1, d.vi = map.getNodeVi("clipa");
    const dt = helper.DataType.select(map, d.node1, d.vi, filter_name) catch return;

    d.node2, const vi2 = map.getNodeVi("clipb");
    helper.compareNodes(out, d.node1, d.node2, d.vi, vi2, filter_name, vsapi) catch return;

    var nodes = [_]?*vs.Node{ d.node1, d.node2 };
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hist_size = if (d.vi.format.sampleType == .Float) 65536 else math.shl(u32, 1, d.vi.format.bitsPerSample);
    d.peak = @intCast(d.hist_size - 1);
    d.maxthr = getThr(in, out, &nodes, "maxthr", vsapi) catch return;
    d.minthr = getThr(in, out, &nodes, "minthr", vsapi) catch return;
    d.prop = getPropData(map, "prop", allocator, &d.prop_buff);

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
        .U8 => if (refb) &PlaneMinMax(u8, true).getFrame else &PlaneMinMax(u8, false).getFrame,
        .U16 => if (refb) &PlaneMinMax(u16, true).getFrame else &PlaneMinMax(u16, false).getFrame,
        .F16 => if (refb) &PlaneMinMax(f16, true).getFrame else &PlaneMinMax(f16, false).getFrame,
        .F32 => if (refb) &PlaneMinMax(f32, true).getFrame else &PlaneMinMax(f32, false).getFrame,
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, planeMinMaxFree, .Parallel, deps, deps_len, data, core);
}

pub fn getThr(in: ?*const vs.Map, out: ?*vs.Map, nodes: []?*vs.Node, comptime key: []const u8, vsapi: ?*const vs.API) !f32 {
    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        for (nodes) |node| vsapi.?.freeNode.?(node);
    }

    const thr = vsh.mapGetN(f32, in, key.ptr, 0, vsapi) orelse 0;
    if (thr < 0 or thr > 1) {
        err_msg = filter_name ++ ": " ++ key ++ " should be a float between 0.0 and 1.0";
        return error.ValidationError;
    }

    return thr;
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
    const max_len = udata_len + 4;
    const min_len = udata_len + 4;
    const data = data_ptr[0..udata_len];
    data_buff.* = data_allocator.alloc(u8, diff_len + max_len + min_len) catch unreachable;
    return .{
        .d = std.fmt.bufPrint(data_buff.*.?[0..diff_len], "{s}Diff\x00", .{data}) catch unreachable,
        .ma = std.fmt.bufPrint(data_buff.*.?[diff_len..], "{s}Max\x00", .{data}) catch unreachable,
        .mi = std.fmt.bufPrint(data_buff.*.?[(diff_len + max_len)..], "{s}Min\x00", .{data}) catch unreachable,
    };
}
