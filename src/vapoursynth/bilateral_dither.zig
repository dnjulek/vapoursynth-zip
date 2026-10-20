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
pub const params = "clip:vnode;ref:vnode:opt;radius:int[]:opt;thr:float[]:opt;flat:float[]:opt;wmin:float[]:opt;subspl:float[]:opt;planes:int[]:opt;";

pub const Data = struct {
    node: *vs.Node,
    ref: ?*vs.Node,
    vi: *const vs.VideoInfo,
    num_planes: u32,
    planes: [3]bool,
    cfg: filter.Config,
};

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
                        d.cfg.rh[plane],
                        d.cfg.rv[plane],
                        d.cfg.m[plane],
                        d.cfg.wmax[plane],
                        d.cfg.sum_w_min[plane],
                        d.cfg.peak,
                        d.cfg.offsets[plane],
                        d.cfg.k[plane],
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
    d.cfg.deinit();
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

    filter.Config.checkSize(d.vi.width, d.vi.height) catch |e| {
        hz.setArgError(zout, "{s}: {s}", .{ filter_name, filter.Config.message(e) });
        zapi.freeNode(d.node);
        return;
    };

    const np: u32 = @intCast(fmt.numPlanes);
    d.num_planes = np;

    d.planes = .{ true, true, true };
    hz.mapGetPlanes(zin, zout, &nodes, &d.planes, fmt.numPlanes, filter_name, &zapi) catch return;

    const ssw: u5 = @intCast(fmt.subSamplingW);
    const ssh: u5 = @intCast(fmt.subSamplingH);
    var plane_w: [3]i32 = @splat(0);
    var plane_h: [3]i32 = @splat(0);
    for (0..np) |p| {
        plane_w[p] = if (p > 0) d.vi.width >> ssw else d.vi.width;
        plane_h[p] = if (p > 0) d.vi.height >> ssh else d.vi.height;
    }

    d.cfg = filter.Config.init(.{
        .is_int = is_int,
        .bits = @intCast(fmt.bitsPerSample),
        .planes = d.planes,
        .num_planes = np,
        .plane_w = plane_w,
        .plane_h = plane_h,
        .radius = radius,
        .thr = thr,
        .flat = flat,
        .wmin = wmin,
        .subspl = subspl_arg,
    }) catch |e| {
        hz.setArgError(zout, "{s}: {s}", .{ filter_name, filter.Config.message(e) });
        zapi.freeNode(d.node);
        return;
    };

    d.ref = zin.getNode("ref");
    if (d.ref) |refn| {
        if (!vsh.isSameVideoInfo(zapi.getVideoInfo(refn), d.vi)) {
            zout.setError(filter_name ++ ": \"ref\" must have the same format and dimensions as \"clip\"");
            d.cfg.deinit();
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
