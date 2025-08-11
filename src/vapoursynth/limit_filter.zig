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

    dark_thr: f32 = 1,
    bright_thr: f32 = 1,
    elast: f32 = 2,
};

fn LimitFilter(comptime T: type, comptime refb: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.flt, frame_ctx);
                zapi.requestFrameFilter(n, d.src, frame_ctx);
                if (refb) zapi.requestFrameFilter(n, d.ref, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.src, n, frame_ctx);
                const flt = zapi.initZFrame(d.flt, n, frame_ctx);
                const ref = if (refb) zapi.initZFrame(d.ref, n, frame_ctx);
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const fltp = flt.getReadSlice2(T, plane);
                    const srcp = src.getReadSlice2(T, plane);
                    const refp = if (refb) ref.getReadSlice2(T, plane) else srcp;
                    const dstp = dst.getWriteSlice2(T, plane);

                    filter.process(T, fltp, srcp, refp, dstp, d.dark_thr, d.bright_thr, d.elast);
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

fn limitFilterFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core);

    zapi.freeNode(d.flt);
    zapi.freeNode(d.src);
    if (d.ref) |nd| zapi.freeNode(nd);
    allocator.destroy(d);
}

pub fn limitFilterCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.flt, d.vi = map_in.getNodeVi("flt").?;
    const dt = hz.DataType.select(map_out, d.flt, d.vi, filter_name, false) catch return;

    d.src = map_in.getNode("src").?;
    d.ref = map_in.getNode("ref");
    const refb = d.ref != null;

    const nodes = [_]?*vs.Node{ d.flt, d.src, d.ref };
    hz.compareNodes(map_out, &nodes, .SAME_LEN, filter_name, &zapi) catch return;

    d.dark_thr = map_in.getFloat(f32, "dark_thr") orelse 1;
    d.bright_thr = map_in.getFloat(f32, "bright_thr") orelse 1;
    d.elast = map_in.getFloat(f32, "elast") orelse 2;

    d.dark_thr = hz.scaleValue(d.dark_thr, d.flt, &zapi, .{});
    d.bright_thr = hz.scaleValue(d.bright_thr, d.flt, &zapi, .{});

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
