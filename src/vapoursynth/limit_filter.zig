const std = @import("std");
const math = std.math;

const filter = @import("../filters/limit_filter.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "LimitFilter";

const Data = struct {
    flt: *vs.Node = undefined,
    src: *vs.Node = undefined,
    ref: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    dark_thr: [3]f32 = @splat(1),
    bright_thr: [3]f32 = @splat(1),
    elast: [3]f32 = @splat(1),
    planes: [3]bool = .{ true, true, true },
};

fn LimitFilter(comptime T: type, comptime refb: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.flt);
                zapi.requestFrameFilter(n, d.src);
                if (refb) zapi.requestFrameFilter(n, d.ref);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.src, n);
                const flt = zapi.initZFrame(d.flt, n);
                const ref = if (refb) zapi.initZFrame(d.ref, n);
                const dst = src.newVideoFrame2(d.planes);

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) continue;

                    const fltp = flt.getReadSlice2(T, plane);
                    const srcp = src.getReadSlice2(T, plane);
                    const refp = if (refb) ref.getReadSlice2(T, plane) else srcp;
                    const dstp = dst.getWriteSlice2(T, plane);

                    filter.process(
                        T,
                        fltp,
                        srcp,
                        refp,
                        dstp,
                        d.dark_thr[plane],
                        d.bright_thr[plane],
                        d.elast[plane],
                    );
                }

                flt.deinit();
                src.deinit();
                if (refb) ref.deinit();
                return dst.frame;
            }

            return null;
        }
    };
}

fn limitFilterFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.flt);
    zapi.freeNode(d.src);
    if (d.ref) |nd| zapi.freeNode(nd);
    allocator.destroy(d);
}

pub fn limitFilterCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.flt, d.vi = map_in.getNodeVi("flt").?;
    const dt = hz.DataType.select(map_out, d.flt, d.vi, filter_name, false) catch return;

    d.src = map_in.getNode("src").?;
    d.ref = map_in.getNode("ref");
    const refb = d.ref != null;

    const nodes = [_]?*vs.Node{ d.flt, d.src, d.ref };
    hz.compareNodes(map_out, &nodes, .SAME_LEN, filter_name, &zapi) catch return;
    hz.mapGetPlanes(map_in, map_out, &nodes, &d.planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;
    d.dark_thr = hz.getArray(f32, 1, 0, 255, "dark_thr", filter_name, map_in, map_out, &nodes, &zapi) catch return;
    d.bright_thr = hz.getArray(f32, 1, 0, 255, "bright_thr", filter_name, map_in, map_out, &nodes, &zapi) catch return;
    d.elast = hz.getArray(f32, 2, 0, math.maxInt(u16), "elast", filter_name, map_in, map_out, &nodes, &zapi) catch return;

    for (0..3) |i| {
        d.dark_thr[i] = hz.scaleValue(d.dark_thr[i], d.flt, &zapi, .{});
        d.bright_thr[i] = hz.scaleValue(d.bright_thr[i], d.flt, &zapi, .{});
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.flt, .requestPattern = .StrictSpatial },
        .{ .source = d.src, .requestPattern = .StrictSpatial },
        .{ .source = d.ref, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = switch (dt) {
        .U8 => if (refb) &LimitFilter(u8, true).getFrame else &LimitFilter(u8, false).getFrame,
        .U16 => if (refb) &LimitFilter(u16, true).getFrame else &LimitFilter(u16, false).getFrame,
        .F16 => if (refb) &LimitFilter(f16, true).getFrame else &LimitFilter(f16, false).getFrame,
        .F32 => if (refb) &LimitFilter(f32, true).getFrame else &LimitFilter(f32, false).getFrame,
        .U32 => unreachable,
    };

    const deps_len: usize = if (refb) deps.len else (deps.len - 1);
    zapi.createVideoFilter(out, filter_name, d.vi, gf, limitFilterFree, .Parallel, deps[0..deps_len], data);
}
