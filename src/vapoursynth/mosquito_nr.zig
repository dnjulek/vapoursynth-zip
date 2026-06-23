const std = @import("std");

const core = @import("../filters/mosquito_nr.zig");
const core_float = @import("../filters/mosquito_nr_float.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "MosquitoNR";

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    strength: [3]i32,
    restore: [3]i32,
    radius: [3]u32,
    bits: u6,
    num_planes: u32,
    planes: [3]bool,
};

fn Filter(comptime T: type) type {
    return struct {
        pub fn getFrame(
            n: c_int,
            activation_reason: vs.ActivationReason,
            instance_data: ?*anyopaque,
            _: ?*?*anyopaque,
            frame_ctx: ?*vs.FrameContext,
            c: ?*vs.Core,
            vsapi: ?*const vs.API,
        ) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, c, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();

                const dst = src.newVideoFrame2(d.planes);
                const proc = if (T == f32) core_float.MosquitoNRFloat.process else core.MosquitoNR(T).process;

                var plane: u32 = 0;
                while (plane < d.num_planes) : (plane += 1) {
                    if (!d.planes[plane]) continue;

                    const w, const h, const stride = src.getDimensions2(T, plane);
                    const srcp = src.getReadSlice2(T, plane).ptr;
                    const dstp = dst.getWriteSlice2(T, plane).ptr;

                    proc(
                        dstp,
                        stride,
                        srcp,
                        w,
                        h,
                        d.strength[plane],
                        d.restore[plane],
                        d.radius[plane],
                        d.bits,
                        plane > 0,
                        allocator,
                    ) catch {
                        dst.deinit();
                        zapi.setFilterError(filter_name ++ ": out of memory");
                        return null;
                    };
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, c: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, c, null);
    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, c: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;
    const zapi = ZAPI.init(vsapi, c, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;
    const fmt = d.vi.format;
    const ok_int = fmt.sampleType == .Integer and fmt.bitsPerSample >= 8 and fmt.bitsPerSample <= 16;
    const ok_float = fmt.sampleType == .Float and fmt.bitsPerSample == 32;
    if (!vsh.isConstantVideoFormat(d.vi) or !(ok_int or ok_float)) {
        map_out.setError(filter_name ++ ": only constant-format 8..16 bit integer or 32 bit float input is supported.");
        zapi.freeNode(d.node);
        return;
    }
    if (fmt.colorFamily != .YUV and fmt.colorFamily != .Gray) {
        map_out.setError(filter_name ++ ": input must be YUV or Gray.");
        zapi.freeNode(d.node);
        return;
    }

    d.planes = .{ true, false, false };
    hz.mapGetPlanes(map_in, map_out, &[_]?*vs.Node{d.node}, &d.planes, fmt.numPlanes, filter_name, &zapi) catch return;

    const np: u32 = @intCast(fmt.numPlanes);
    var p: u32 = 0;
    while (p < np) : (p += 1) {
        if (!d.planes[p]) continue;
        const ssw: u5 = if (p > 0) @intCast(fmt.subSamplingW) else 0;
        const ssh: u5 = if (p > 0) @intCast(fmt.subSamplingH) else 0;
        if ((d.vi.width >> ssw) < 4 or (d.vi.height >> ssh) < 4) {
            map_out.setError(filter_name ++ ": input is too small (need at least 4x4 per processed plane).");
            zapi.freeNode(d.node);
            return;
        }
    }

    const nodes = [_]?*vs.Node{d.node};
    d.strength = hz.getArray(i32, 16, 0, 32, "strength", filter_name, map_in, map_out, &nodes, &zapi) catch return;
    d.restore = hz.getArray(i32, 128, 0, 128, "restore", filter_name, map_in, map_out, &nodes, &zapi) catch return;
    const radius = hz.getArray(i32, 2, 1, 2, "radius", filter_name, map_in, map_out, &nodes, &zapi) catch return;
    for (0..3) |i| d.radius[i] = @intCast(radius[i]);

    d.bits = @intCast(fmt.bitsPerSample);
    d.num_planes = np;

    const data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = if (fmt.sampleType == .Float)
        &Filter(f32).getFrame
    else if (fmt.bytesPerSample == 1)
        &Filter(u8).getFrame
    else
        &Filter(u16).getFrame;

    zapi.createVideoFilter(out, filter_name, d.vi, gf, free, .Parallel, &deps, data);
}
