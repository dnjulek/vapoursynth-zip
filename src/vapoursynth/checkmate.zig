const std = @import("std");
const math = std.math;

const filter = @import("../filters/checkmate.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");
const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;

const allocator = std.heap.c_allocator;
pub const filter_name = "Checkmate";

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    thr: i32,
    tmax: i32,
    tthr2: i32,
};

fn Checkmate(comptime use_tthr2: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(@max(0, n - 1), d.node, frame_ctx);
                vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
                vsapi.?.requestFrameFilter.?(@min(n + 1, d.vi.numFrames - 1), d.node, frame_ctx);

                if (use_tthr2) {
                    vsapi.?.requestFrameFilter.?(@max(0, n - 2), d.node, frame_ctx);
                    vsapi.?.requestFrameFilter.?(@min(n + 2, d.vi.numFrames - 1), d.node, frame_ctx);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src_p1 = zapi.ZFrame.init(d.node, @max(0, n - 1), frame_ctx, core, vsapi);
                const src = zapi.ZFrame.init(d.node, n, frame_ctx, core, vsapi);
                const src_n1 = zapi.ZFrame.init(d.node, @min(n + 1, d.vi.numFrames - 1), frame_ctx, core, vsapi);

                var src_p2: ?zapi.ZFrameRO = null;
                var src_n2: ?zapi.ZFrameRO = null;
                if (use_tthr2) {
                    src_p2 = zapi.ZFrame.init(d.node, @max(0, n - 2), frame_ctx, core, vsapi);
                    src_n2 = zapi.ZFrame.init(d.node, @min(n + 2, d.vi.numFrames - 1), frame_ctx, core, vsapi);
                }

                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    var srcp_p1 = src_p1.getReadSlice(plane);
                    var srcp = src.getReadSlice(plane);
                    var srcp_n1 = src_n1.getReadSlice(plane);
                    var dstp = dst.getWriteSlice(plane);

                    var srcp_p2: ?[]const u8 = null;
                    var srcp_n2: ?[]const u8 = null;

                    if (use_tthr2) {
                        srcp_p2 = src_p2.?.getReadSlice(plane);
                        srcp_n2 = src_n2.?.getReadSlice(plane);
                    }

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
                if (d.tthr2 > 0) {
                    src_p2.?.deinit();
                    src_n2.?.deinit();
                }

                return dst.frame;
            }

            return null;
        }
    };
}

export fn checkmateFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn checkmateCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;
    const map_in = zapi.ZMap.init(in, vsapi);
    const map_out = zapi.ZMap.init(out, vsapi);

    d.node, d.vi = map_in.getNodeVi("clip");

    if ((d.vi.format.sampleType != .Integer) or (d.vi.format.bitsPerSample != 8)) {
        map_out.setError(filter_name ++ ": only 8 bit int format supported.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    d.thr = map_in.getInt(i32, "thr") orelse 12;
    d.tmax = map_in.getInt(i32, "tmax") orelse 12;
    d.tthr2 = map_in.getInt(i32, "tthr2") orelse 0;

    if ((d.tmax < 1) or (d.tmax > 255)) {
        map_out.setError(filter_name ++ ": tmax value should be in range [1;255].");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    if (d.tthr2 < 0) {
        map_out.setError(filter_name ++ ": tthr2 should be non-negative.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .General,
        },
    };

    const getFrame = if (d.tthr2 > 0) &Checkmate(true).getFrame else &Checkmate(false).getFrame;
    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, checkmateFree, .Parallel, &deps, deps.len, data, core);
}
