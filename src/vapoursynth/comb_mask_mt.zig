const std = @import("std");
const math = std.math;

const filter = @import("../filters/comb_mask_mt.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "CombMaskMT";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    thy1: u8 = 0,
    thy2: u8 = 0,
    thr_diff: u8 = 0,
};

fn CombMaskMT(comptime same_thr: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);

                defer src.deinit();

                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice(plane);
                    const dstp = dst.getWriteSlice(plane);
                    const w, const h, const stride = src.getDimensions(plane);
                    filter.process(
                        srcp,
                        dstp,
                        stride,
                        w,
                        h,
                        d.thy1,
                        d.thy2,
                        d.thr_diff,
                        same_thr,
                    );
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn combMaskMTFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn combMaskMTCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
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

    const thy1 = map_in.getValue(i16, "thY1") orelse 30;
    const thy2 = map_in.getValue(i16, "thY2") orelse 30;

    if (thy1 > 255 or thy1 < 0) {
        map_out.setError(filter_name ++ ": thY1 value should be in range [0;255]");
        zapi.freeNode(d.node);
        return;
    }

    if (thy2 > 255 or thy2 < 0) {
        map_out.setError(filter_name ++ ": thY2 value should be in range [0;255]");
        zapi.freeNode(d.node);
        return;
    }

    if (thy1 > thy2) {
        map_out.setError(filter_name ++ ": thY1 can't be greater than thY2");
        zapi.freeNode(d.node);
        return;
    }

    d.thy1 = @intCast(thy1);
    d.thy2 = @intCast(thy2);
    d.thr_diff = d.thy2 - d.thy1;

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = if (d.thy1 == d.thy2) &CombMaskMT(true).getFrame else &CombMaskMT(false).getFrame;
    zapi.createVideoFilter(out, filter_name, d.vi, gf, combMaskMTFree, .Parallel, &deps, data);
}
