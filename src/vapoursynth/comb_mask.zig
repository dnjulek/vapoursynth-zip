const std = @import("std");
const math = std.math;

const filter = @import("../filters/comb_mask.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "CombMask";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    cthresh: i32 = 0,
    cth6: u16 = 0,
    mthresh: u16 = 0,
};

fn CombMask(comptime metric_1: bool, comptime expand: bool, comptime motion: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (motion) zapi.requestFrameFilter(@max(0, n - 1), d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                const prv = if (motion) zapi.initZFrame(d.node, @max(0, n - 1));
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice(plane);
                    const prvp = if (motion) prv.getReadSlice(plane);
                    const dstp = dst.getWriteSlice(plane);
                    const w, const h, const stride = src.getDimensions(plane);
                    filter.process(
                        srcp,
                        prvp,
                        dstp,
                        stride,
                        w,
                        h,
                        d.cthresh,
                        d.cth6,
                        d.mthresh,
                        metric_1,
                        expand,
                        motion,
                    );
                }

                if (motion) prv.deinit();
                src.deinit();
                return dst.frame;
            }

            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
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

    const cthresh = map_in.getValue(i32, "cthresh") orelse 6;
    const mthresh = map_in.getValue(i16, "mthresh") orelse 9;
    const expand = map_in.getBool("expand") orelse true;
    const metric_1 = map_in.getBool("metric") orelse false;
    const cth_max: i32 = if (metric_1) 65025 else 255;

    if ((cthresh > cth_max) or (cthresh < 0)) {
        const msg = std.fmt.allocPrintSentinel(
            allocator,
            filter_name ++
                ": cthresh must be between 0 and {} when metric = {any}.",
            .{ cth_max, metric_1 },
            0,
        ) catch unreachable;
        map_out.setError(msg);
        zapi.freeNode(d.node);
        allocator.free(msg);
        return;
    }

    if ((mthresh > 255) or (mthresh < 0)) {
        map_out.setError(filter_name ++ ": mthresh must be between 0 and 255.");
        zapi.freeNode(d.node);
        return;
    }

    d.cthresh = cthresh;
    d.cth6 = @intCast(cthresh * 6);
    d.mthresh = @intCast(mthresh);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    var gf: vs.FilterGetFrame = undefined;
    if (expand) {
        if (mthresh > 0) {
            gf = if (metric_1) &CombMask(true, true, true).getFrame else &CombMask(false, true, true).getFrame;
        } else {
            gf = if (metric_1) &CombMask(true, true, false).getFrame else &CombMask(false, true, false).getFrame;
        }
    } else {
        if (mthresh > 0) {
            gf = if (metric_1) &CombMask(true, false, true).getFrame else &CombMask(false, false, true).getFrame;
        } else {
            gf = if (metric_1) &CombMask(true, false, false).getFrame else &CombMask(false, false, false).getFrame;
        }
    }

    zapi.createVideoFilter(out, filter_name, d.vi, gf, free, .Parallel, &deps, data);
}
