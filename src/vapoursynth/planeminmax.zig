const std = @import("std");
const math = std.math;

const filter = @import("../filters/planeminmax.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "PlaneMinMax";

pub const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    peak: u16 = 0,
    peakf: f32 = 0,
    minthr: f32 = 0,
    maxthr: f32 = 0,
    hist_size: u32 = 0,
    planes: [3]bool = .{ true, false, false },
    prop: StringProp = undefined,
};

const StringProp = struct {
    d: [:0]u8,
    ma: [:0]u8,
    mi: [:0]u8,
};

fn PlaneMinMax(comptime T: type, comptime refb: bool, comptime no_thr: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node1);
                if (refb) zapi.requestFrameFilter(n, d.node2);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node1, n);
                const ref = if (refb) zapi.initZFrame(d.node2, n);
                const dst = src.copyFrame();
                const props = dst.getPropertiesRW();
                props.deleteKey(d.prop.d);
                props.deleteKey(d.prop.ma);
                props.deleteKey(d.prop.mi);

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) continue;

                    const srcp = src.getReadSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);

                    if (refb) {
                        const refp = ref.getReadSlice2(T, plane);
                        if (no_thr) {
                            filter.minMaxNoThrRef(T, srcp, refp, stride, &props, w, h, d);
                        } else {
                            if (@typeInfo(T) == .int)
                                filter.minMaxIntRef(T, srcp, refp, stride, &props, w, h, d)
                            else
                                filter.minMaxFloatRef(T, srcp, refp, stride, &props, w, h, d);
                        }
                    } else {
                        if (no_thr) {
                            filter.minMaxNoThr(T, srcp, stride, &props, w, h, d);
                        } else {
                            if (@typeInfo(T) == .int)
                                filter.minMaxInt(T, srcp, stride, &props, w, h, d)
                            else
                                filter.minMaxFloat(T, srcp, stride, &props, w, h, d);
                        }
                    }
                }

                src.deinit();
                if (refb) ref.deinit();

                return dst.frame;
            }

            return null;
        }
    };
}

fn planeMinMaxFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    allocator.free(d.prop.d);
    allocator.free(d.prop.ma);
    allocator.free(d.prop.mi);

    if (d.node2) |node| zapi.freeNode(node);
    zapi.freeNode(d.node1);
    allocator.destroy(d);
}

pub fn planeMinMaxCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node1, d.vi = map_in.getNodeVi("clipa").?;
    const dt = hz.DataType.select(map_out, d.node1, d.vi, filter_name, false) catch return;

    d.node2 = map_in.getNode("clipb");
    const refb = d.node2 != null;
    const nodes = [_]?*vs.Node{ d.node1, d.node2 };
    if (refb) {
        hz.compareNodes(map_out, &nodes, .BIGGER_THAN, filter_name, &zapi) catch return;
    }

    hz.mapGetPlanes(map_in, map_out, &nodes, &d.planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;
    d.hist_size = if (d.vi.format.sampleType == .Float) 65536 else math.shl(u32, 1, d.vi.format.bitsPerSample);
    d.peak = @intCast(d.hist_size - 1);
    d.peakf = @floatFromInt(d.peak);
    d.maxthr = getThr(map_in, map_out, &nodes, "maxthr", &zapi) catch return;
    d.minthr = getThr(map_in, map_out, &nodes, "minthr", &zapi) catch return;

    const prop_in = map_in.getData("prop", 0) orelse "psm";
    d.prop = .{
        .d = std.fmt.allocPrintSentinel(allocator, "{s}Diff", .{prop_in}, 0) catch unreachable,
        .ma = std.fmt.allocPrintSentinel(allocator, "{s}Max", .{prop_in}, 0) catch unreachable,
        .mi = std.fmt.allocPrintSentinel(allocator, "{s}Min", .{prop_in}, 0) catch unreachable,
    };

    const no_thr = d.maxthr == 0 and d.minthr == 0;
    const do_chroma = d.planes[1] or d.planes[2];

    if (do_chroma and !no_thr and
        (d.vi.format.colorFamily == .YUV) and
        (d.vi.format.sampleType == .Float))
    {
        map_out.setError(filter_name ++ ": you can't use maxthr/minthr with float chroma, use planes=[0] or maxthr/minthr=0");
        zapi.freeNode(d.node1);
        if (refb) zapi.freeNode(d.node2);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const rp2: vs.RequestPattern = if (refb and (d.vi.numFrames <= zapi.getVideoInfo(d.node2).numFrames)) .StrictSpatial else .FrameReuseLastOnly;
    const deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = rp2 },
    };

    var getFrame: vs.FilterGetFrame = undefined;
    if (no_thr) {
        getFrame = switch (dt) {
            .U8 => if (refb) &PlaneMinMax(u8, true, true).getFrame else &PlaneMinMax(u8, false, true).getFrame,
            .U16 => if (refb) &PlaneMinMax(u16, true, true).getFrame else &PlaneMinMax(u16, false, true).getFrame,
            .F16 => if (refb) &PlaneMinMax(f16, true, true).getFrame else &PlaneMinMax(f16, false, true).getFrame,
            .F32 => if (refb) &PlaneMinMax(f32, true, true).getFrame else &PlaneMinMax(f32, false, true).getFrame,
            .U32 => unreachable,
        };
    } else {
        getFrame = switch (dt) {
            .U8 => if (refb) &PlaneMinMax(u8, true, false).getFrame else &PlaneMinMax(u8, false, false).getFrame,
            .U16 => if (refb) &PlaneMinMax(u16, true, false).getFrame else &PlaneMinMax(u16, false, false).getFrame,
            .F16 => if (refb) &PlaneMinMax(f16, true, false).getFrame else &PlaneMinMax(f16, false, false).getFrame,
            .F32 => if (refb) &PlaneMinMax(f32, true, false).getFrame else &PlaneMinMax(f32, false, false).getFrame,
            .U32 => unreachable,
        };
    }

    const ndeps: usize = if (refb) 2 else 1;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, planeMinMaxFree, .Parallel, deps[0..ndeps], data);
}

pub fn getThr(in: ZAPI.ZMap(?*const vs.Map), out: ZAPI.ZMap(?*vs.Map), nodes: []const ?*vs.Node, comptime key: [:0]const u8, zapi: *const ZAPI) !f32 {
    var err_msg: ?[:0]const u8 = null;
    errdefer {
        out.setError(err_msg.?);
        for (nodes) |node| {
            if (node) |n| {
                zapi.freeNode(n);
            }
        }
    }

    const thr = in.getValue(f32, key) orelse 0;
    if (thr < 0 or thr > 1) {
        err_msg = filter_name ++ ": " ++ key ++ " should be a float between 0.0 and 1.0";
        return error.ValidationError;
    }

    return thr;
}
