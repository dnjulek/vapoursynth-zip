const std = @import("std");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const filter = @import("../filters/bilateral_dither.zig");
const subspl = @import("../filters/bilateral_dither_subspl.zig");
const hz = @import("../helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "BilateralDither";

pub const Data = struct {
    node: *vs.Node,
    ref: ?*vs.Node,
    vi: *const vs.VideoInfo,
    num_planes: u32,
    planes: [3]bool,
    rh: [3]u32,
    rv: [3]u32,
    m: [3]f32,
    wmax: [3]f32,
    sum_w_min: [3]f32,
    point_lists: [3]?[]subspl.Coord,
    k: [3]usize,
    peak: f32,
};

fn freePointLists(d: *Data) void {
    for (d.point_lists) |pl| {
        if (pl) |p| allocator.free(p);
    }
}

const SlotCfg = struct { sum_w_min: f32, pl: ?[]subspl.Coord, k: usize };

fn buildSlot(rh: u32, rv: u32, subspl_arg: f32, wmin: f32, wmax: f32, unit: f32) !SlotCfg {
    const active = (subspl_arg >= 4.0) or (subspl_arg < 1e-3);
    if (active) {
        const lists = try subspl.generate(@intCast(rh), @intCast(rv), @floatCast(subspl_arg));
        const kf: f32 = @floatFromInt(lists.k);
        return .{ .sum_w_min = @max(wmin * wmax * kf, unit), .pl = lists.pts, .k = lists.k };
    } else {
        const area: f32 = @floatFromInt((2 * rh - 1) * (2 * rv - 1));
        return .{ .sum_w_min = @max(wmin * wmax * area, unit), .pl = null, .k = 0 };
    }
}

fn GetFrame(comptime T: type) type {
    return struct {
        fn gf(
            n: c_int,
            ar: vs.ActivationReason,
            instance: ?*anyopaque,
            _: ?*?*anyopaque,
            frame_ctx: ?*vs.FrameContext,
            core: ?*vs.Core,
            vsapi: ?*const vs.API,
        ) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (ar == .Initial) {
                zapi.requestFrameFilter(n, d.node);
                if (d.ref) |refn| zapi.requestFrameFilter(n, refn);
            } else if (ar == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const ref = if (d.ref) |refn| zapi.initZFrame(refn, n) else null;
                defer if (ref) |r| r.deinit();
                const dst = src.newVideoFrame2(d.planes);

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    if (!d.planes[plane]) continue;
                    const w = src.getWidth(plane);
                    const h = src.getHeight(plane);
                    const stride = src.getStride2(T, plane);
                    const srcp = src.getReadSlice2(T, plane);
                    const refp: ?[]const T = if (ref) |r| r.getReadSlice2(T, plane) else null;
                    const dstp = dst.getWriteSlice2(T, plane);

                    filter.processPlane(
                        T,
                        srcp,
                        refp,
                        dstp,
                        w,
                        h,
                        stride,
                        d.rh[plane],
                        d.rv[plane],
                        d.m[plane],
                        d.wmax[plane],
                        d.sum_w_min[plane],
                        d.peak,
                        d.point_lists[plane],
                        d.k[plane],
                    );
                }

                return dst.frame;
            }
            return null;
        }
    };
}

fn free(instance: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance));
    vsapi.?.freeNode.?(d.node);
    if (d.ref) |refn| vsapi.?.freeNode.?(refn);
    freePointLists(d);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core, null);
    const zin = zapi.initZMap(in);
    const zout = zapi.initZMap(out);

    d.node, d.vi = zin.getNodeVi("clip").?;
    d.ref = null;
    const fmt = d.vi.format;

    if (!vsh.isConstantVideoFormat(d.vi)) {
        zout.setError(filter_name ++ ": only constant format input supported");
        zapi.freeNode(d.node);
        return;
    }

    const is_int = fmt.sampleType == .Integer;
    if (is_int) {
        if (fmt.bitsPerSample < 8 or fmt.bitsPerSample > 16) {
            zout.setError(filter_name ++ ": integer input must be 8..16 bit");
            zapi.freeNode(d.node);
            return;
        }
    } else if (fmt.bitsPerSample != 32) {
        zout.setError(filter_name ++ ": float input must be 32 bit");
        zapi.freeNode(d.node);
        return;
    }

    const nodes = [_]?*vs.Node{d.node};
    const radius = hz.getArray(i32, 16, 2, 16384, "radius", filter_name, zin, zout, &nodes, &zapi) catch return;
    const thr = hz.getArray(f32, 2.5, 0, 65535, "thr", filter_name, zin, zout, &nodes, &zapi) catch return;
    const flat = hz.getArray(f32, 0.4, 0, 1, "flat", filter_name, zin, zout, &nodes, &zapi) catch return;
    const wmin = hz.getArray(f32, 0, 0, 65535, "wmin", filter_name, zin, zout, &nodes, &zapi) catch return;
    const subspl_arg = hz.getArray(f32, 0, 0, 4096, "subspl", filter_name, zin, zout, &nodes, &zapi) catch return;

    if (d.vi.width < 16 or d.vi.height < 16) {
        zout.setError(filter_name ++ ": input must be 16x16 min");
        zapi.freeNode(d.node);
        return;
    }

    const scale: f32 = if (is_int)
        @floatFromInt(@as(u32, 1) << @as(u5, @intCast(fmt.bitsPerSample - 8)))
    else
        1.0 / 256.0;

    const unit: f32 = if (is_int) 1.0 else 1.0 / 65535.0;
    d.peak = if (is_int)
        @floatFromInt((@as(u32, 1) << @as(u5, @intCast(fmt.bitsPerSample))) - 1)
    else
        0;

    const np: u32 = @intCast(fmt.numPlanes);
    d.num_planes = np;
    d.point_lists = .{ null, null, null };

    d.planes = .{ true, true, true };
    hz.mapGetPlanes(zin, zout, &nodes, &d.planes, fmt.numPlanes, filter_name, &zapi) catch return;

    const ssw: u5 = @intCast(fmt.subSamplingW);
    const ssh: u5 = @intCast(fmt.subSamplingH);

    var p: u32 = 0;
    while (p < np) : (p += 1) {
        if (!d.planes[p]) continue;
        const pw: i32 = if (p > 0) d.vi.width >> ssw else d.vi.width;
        const ph: i32 = if (p > 0) d.vi.height >> ssh else d.vi.height;
        if (pw < radius[p] or ph < radius[p]) {
            zout.setError(filter_name ++ ": picture size must be greater than \"radius\"");
            freePointLists(&d);
            zapi.freeNode(d.node);
            return;
        }
        const rp: u32 = @intCast(radius[p]);
        d.rh[p] = rp;
        d.rv[p] = rp;
        d.m[p] = @max(thr[p] * scale, unit);
        d.wmax[p] = @max(thr[p] * (1.0 - flat[p]) * scale, unit);
        const slot = buildSlot(rp, rp, subspl_arg[p], wmin[p], d.wmax[p], unit) catch {
            zout.setError(filter_name ++ ": out of memory");
            freePointLists(&d);
            zapi.freeNode(d.node);
            return;
        };
        d.sum_w_min[p] = slot.sum_w_min;
        d.point_lists[p] = slot.pl;
        d.k[p] = slot.k;
    }

    d.ref = zin.getNode("ref");
    if (d.ref) |refn| {
        if (!vsh.isSameVideoInfo(zapi.getVideoInfo(refn), d.vi)) {
            zout.setError(filter_name ++ ": \"ref\" must have the same format and dimensions as \"clip\"");
            freePointLists(&d);
            zapi.freeNode(d.node);
            zapi.freeNode(refn);
            return;
        }
    }

    const data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
        .{ .source = d.ref, .requestPattern = .StrictSpatial },
    };
    const num_deps: usize = if (d.ref != null) 2 else 1;

    const gf: vs.FilterGetFrame = if (!is_int)
        GetFrame(f32).gf
    else if (fmt.bytesPerSample == 1)
        GetFrame(u8).gf
    else
        GetFrame(u16).gf;

    zapi.createVideoFilter(out, filter_name, d.vi, gf, free, .Parallel, deps[0..num_deps], data);
}
