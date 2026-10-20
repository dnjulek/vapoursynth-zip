const std = @import("std");
const math = std.math;

const filter = @import("../filters/checkmate.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Checkmate";
pub const params = "clip:vnode;thr:int:opt;tmax:int:opt;tthr2:int:opt;";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    p: filter.Params = undefined,
};

fn Checkmate(comptime use_tthr2: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(@max(0, n - 1), d.node);
                zapi.requestFrameFilter(n, d.node);
                zapi.requestFrameFilter(@min(n + 1, d.vi.numFrames - 1), d.node);

                if (use_tthr2) {
                    zapi.requestFrameFilter(@max(0, n - 2), d.node);
                    zapi.requestFrameFilter(@min(n + 2, d.vi.numFrames - 1), d.node);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src_p1 = zapi.initZFrame(d.node, @max(0, n - 1));
                const src = zapi.initZFrame(d.node, n);
                const src_n1 = zapi.initZFrame(d.node, @min(n + 1, d.vi.numFrames - 1));
                const src_p2 = if (use_tthr2) zapi.initZFrame(d.node, @max(0, n - 2));
                const src_n2 = if (use_tthr2) zapi.initZFrame(d.node, @min(n + 2, d.vi.numFrames - 1));
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp_p2 = if (use_tthr2) src_p2.getReadSlice(plane);
                    const srcp_n2 = if (use_tthr2) src_n2.getReadSlice(plane);
                    const w, const h, const stride = src.getDimensions(plane);
                    filter.processPlane(use_tthr2, dst.getWriteSlice(plane), srcp_p2, src_p1.getReadSlice(plane), src.getReadSlice(plane), src_n1.getReadSlice(plane), srcp_n2, stride, w, h, d.p);
                }

                src_p1.deinit();
                src.deinit();
                src_n1.deinit();
                if (use_tthr2) {
                    src_p2.deinit();
                    src_n2.deinit();
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn checkmateFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
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
    d.node, d.vi = map_in.getNodeVi("clip").?;

    if ((d.vi.format.sampleType != .Integer) or (d.vi.format.bitsPerSample != 8)) {
        map_out.setError(filter_name ++ ": only 8 bit int format supported.");
        zapi.freeNode(d.node);
        return;
    }

    const ssw: u5 = @intCast(d.vi.format.subSamplingW);
    const ssh: u5 = @intCast(d.vi.format.subSamplingH);
    d.p = filter.Params.init(
        map_in.getValue(i32, "thr") orelse 12,
        map_in.getValue(i32, "tmax") orelse 12,
        map_in.getValue(i32, "tthr2") orelse 0,
        @as(u32, @intCast(d.vi.width)) >> ssw,
        @as(u32, @intCast(d.vi.height)) >> ssh,
    ) catch |e| {
        hz.setArgError(map_out, "{s}: {s}", .{ filter_name, filter.Params.message(e) });
        zapi.freeNode(d.node);
        return;
    };

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .General },
    };

    const getFrame = if (d.p.tthr2 > 0) &Checkmate(true).getFrame else &Checkmate(false).getFrame;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, checkmateFree, .Parallel, &deps, data);
}
