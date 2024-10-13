const std = @import("std");
const vszip = @import("../vszip.zig");
const hz = @import("../helper.zig");
// const filter = @import("../filters/comb_mask.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "CombMask";

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    metric: i16,
    cthresh: i16,
    mthresh: i16,
    exapnd: bool,
    need_buff: bool,
};

fn metric1(srcp: []const u8, dstp: []u8, stride: u32, width: u32, height: u32, cthresh: i16) void {
    var sc = srcp;
    var dt = dstp;
    var st1 = sc[stride..];
    var st2 = st1[stride..];
    var sb1 = sc[stride..];
    var sb2 = sb1[stride..];

    const cth6 = cthresh * 6;

    var y: u32 = 0;
    while (y < height - 2) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            dt[x] = 0;
            const d1: i16 = @as(i16, sc[x]) - st1[x];
            const d2: i16 = @as(i16, sc[x]) - sb1[x];
            if ((d1 > cthresh and d2 > cthresh) or (d1 < -cthresh and d2 < -cthresh)) {
                const f0: u16 = @as(u16, st2[x]) + 4 * @as(u16, sc[x]) + @as(u16, sb2[x]);
                const f1: u16 = 3 * (@as(u16, st1[x]) + @as(u16, sb1[x]));

                if (hz.absDiff(f0, f1) > cth6) {
                    dt[x] = 255;
                }
            }
        }

        st2 = st1;
        st1 = sc;
        sc = sb1;
        sb1 = sb2;
        sb2 = sb2[stride..];
        dt = dt[stride..];
    }
}

fn combMaskGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node, n, frame_ctx, core, vsapi);

        defer src.deinit();

        const dst = src.newVideoFrame();

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const srcp = src.getReadSlice(plane);
            const dstp = dst.getWriteSlice(plane);
            const w, const h, const stride = src.getDimensions(plane);
            // filter.process(srcp, dstp, stride, w, h, d.thy1, d.thy2);

            metric1(srcp, dstp, stride, w, h, d.cthresh);
        }

        return dst.frame;
    }

    return null;
}

export fn combMaskFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn combMaskCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;
    var map = zapi.Map.init(in, out, vsapi);

    d.node, d.vi = map.getNodeVi("clip");
    if ((d.vi.format.sampleType != .Integer) or (d.vi.format.bitsPerSample != 8)) {
        map.setError(filter_name ++ ": only 8 bit int format supported.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    d.metric = map.getInt(i16, "metric") orelse 0;
    d.cthresh = map.getInt(i16, "cthresh") orelse (if (d.metric == 0) 6 else 10);
    d.mthresh = map.getInt(i16, "mthresh") orelse 9;
    d.exapnd = map.getBool("exapnd") orelse true;

    if (d.metric != 0 and d.metric != 1) {
        map.setError(filter_name ++ ": metric must be set to 0 or 1.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    if (d.metric == 0) {
        if (d.cthresh < 0 or d.cthresh > 255) {
            map.setError(filter_name ++ ": cthresh must be between 0 and 255 on metric 0.");
            vsapi.?.freeNode.?(d.node);
            return;
        }
    } else {
        if (d.cthresh < 0 or d.cthresh > 65025) {
            map.setError(filter_name ++ ": cthresh must be between 0 and 65025 on metric 1.");
            vsapi.?.freeNode.?(d.node);
            return;
        }
    }

    if (d.mthresh < 0 or d.mthresh > 255) {
        map.setError(filter_name ++ ": mthresh must be between 0 and 255.");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, combMaskGetFrame, combMaskFree, .Parallel, &deps, deps.len, data, core);
}
