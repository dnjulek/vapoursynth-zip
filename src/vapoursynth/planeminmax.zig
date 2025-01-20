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
                const src = zapi.ZFrame.init(d.node1, n, frame_ctx, core, vsapi);
                defer src.deinit();

                var ref: ?zapi.ZFrameRO = null;
                if (refb) {
                    ref = zapi.ZFrame.init(d.node2.?, n, frame_ctx, core, vsapi);
                    defer ref.?.deinit();
                }

                const dst = src.copyFrame();
                const props = dst.getProperties();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) {
                        continue;
                    }

                    const srcp = src.getReadSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    var stats: filter.Stats = undefined;
                    if (refb) {
                        const refp = ref.?.getReadSlice2(T, plane);
                        stats = if (@typeInfo(T) == .int) filter.minMaxIntRef(T, srcp, refp, stride, w, h, d) else filter.minMaxFloatRef(T, srcp, refp, stride, w, h, d);

                        _ = switch (stats) {
                            .i => props.setFloat(if (d.prop != null) d.prop.?.d else "psmDiff", stats.i.diff, .Append),
                            .f => props.setFloat(if (d.prop != null) d.prop.?.d else "psmDiff", stats.f.diff, .Append),
                        };
                    } else {
                        stats = if (@typeInfo(T) == .int) filter.minMaxInt(T, srcp, stride, w, h, d) else filter.minMaxFloat(T, srcp, stride, w, h, d);
                    }

                    switch (stats) {
                        .i => {
                            props.setInt(if (d.prop != null) d.prop.?.ma else "psmMax", stats.i.max, .Append);
                            props.setInt(if (d.prop != null) d.prop.?.mi else "psmMin", stats.i.min, .Append);
                        },
                        .f => {
                            props.setFloat(if (d.prop != null) d.prop.?.ma else "psmMax", stats.f.max, .Append);
                            props.setFloat(if (d.prop != null) d.prop.?.mi else "psmMin", stats.f.min, .Append);
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

    const map_in = zapi.ZMap.init(in, vsapi);
    const map_out = zapi.ZMap.init(out, vsapi);
    d.node1, d.vi = map_in.getNodeVi("clipa");
    const dt = helper.DataType.select(map_out, d.node1, d.vi, filter_name) catch return;

    d.node2, const vi2 = map_in.getNodeVi("clipb");
    helper.compareNodes(map_out, d.node1, d.node2, d.vi, vi2, filter_name, vsapi) catch return;

    var nodes = [_]?*vs.Node{ d.node1, d.node2 };
    var planes = [3]bool{ true, false, false };
    helper.mapGetPlanes(map_in, map_out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hist_size = if (d.vi.format.sampleType == .Float) 65536 else math.shl(u32, 1, d.vi.format.bitsPerSample);
    d.peak = @intCast(d.hist_size - 1);
    d.maxthr = getThr(map_in, map_out, &nodes, "maxthr", vsapi) catch return;
    d.minthr = getThr(map_in, map_out, &nodes, "minthr", vsapi) catch return;
    d.prop = getString(map_in, "prop", allocator, &d.prop_buff);

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

pub fn getThr(in: zapi.ZMapRO, out: zapi.ZMapRW, nodes: []?*vs.Node, comptime key: []const u8, vsapi: ?*const vs.API) !f32 {
    var err_msg: ?[]const u8 = null;
    errdefer {
        out.setError(err_msg.?);
        for (nodes) |node| vsapi.?.freeNode.?(node);
    }

    const thr = in.getFloat(f32, key) orelse 0;
    if (thr < 0 or thr > 1) {
        err_msg = filter_name ++ ": " ++ key ++ " should be a float between 0.0 and 1.0";
        return error.ValidationError;
    }

    return thr;
}

pub fn getString(map: zapi.ZMapRO, comptime key: []const u8, data_allocator: std.mem.Allocator, data_buff: *?[]u8) ?StringProp {
    data_buff.* = null;
    const data = map.getData(key, 0) orelse return null;

    const diff_len = data.len + 5;
    const max_len = data.len + 4;
    const min_len = data.len + 4;
    data_buff.* = data_allocator.alloc(u8, diff_len + max_len + min_len) catch unreachable;
    return .{
        .d = std.fmt.bufPrint(data_buff.*.?[0..diff_len], "{s}Diff\x00", .{data}) catch unreachable,
        .ma = std.fmt.bufPrint(data_buff.*.?[diff_len..], "{s}Max\x00", .{data}) catch unreachable,
        .mi = std.fmt.bufPrint(data_buff.*.?[(diff_len + max_len)..], "{s}Min\x00", .{data}) catch unreachable,
    };
}
