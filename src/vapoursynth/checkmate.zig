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

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,

    thr: i32 = 0,
    tmax: i32 = 0,
    tthr2: i32 = 0,
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
                    var srcp_p1 = src_p1.getReadSlice(plane);
                    var srcp = src.getReadSlice(plane);
                    var srcp_n1 = src_n1.getReadSlice(plane);
                    var dstp = dst.getWriteSlice(plane);

                    const srcp_p2 = if (use_tthr2) src_p2.getReadSlice(plane);
                    const srcp_n2 = if (use_tthr2) src_n2.getReadSlice(plane);

                    const w, const h, const stride = src.getDimensions(plane);
                    const stride2 = stride << 1;

                    @memcpy(dstp[0..stride2], srcp[0..stride2]);

                    srcp_p1 = srcp_p1[stride2..];
                    srcp = srcp[stride2..];
                    srcp_n1 = srcp_n1[stride2..];
                    dstp = dstp[stride2..];

                    var y: u32 = 2;
                    while (y < h - 2) : (y += 1) {
                        filter.process(dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, stride, w, d.thr, d.tmax, d.tthr2, use_tthr2);
                        srcp_p1 = srcp_p1[stride..];
                        srcp = srcp[stride..];
                        srcp_n1 = srcp_n1[stride..];
                        dstp = dstp[stride..];
                    }

                    @memcpy(dstp[0..stride2], srcp[0..stride2]);
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

pub fn checkmateCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
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

    d.thr = map_in.getValue(i32, "thr") orelse 12;
    d.tmax = map_in.getValue(i32, "tmax") orelse 12;
    d.tthr2 = map_in.getValue(i32, "tthr2") orelse 0;

    if ((d.tmax < 1) or (d.tmax > 255)) {
        map_out.setError(filter_name ++ ": tmax value should be in range [1;255].");
        zapi.freeNode(d.node);
        return;
    }

    if (d.tthr2 < 0) {
        map_out.setError(filter_name ++ ": tthr2 should be non-negative.");
        zapi.freeNode(d.node);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .General },
    };

    const getFrame = if (d.tthr2 > 0) &Checkmate(true).getFrame else &Checkmate(false).getFrame;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, checkmateFree, .Parallel, &deps, data);
}
