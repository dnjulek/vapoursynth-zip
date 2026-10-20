const std = @import("std");
const math = std.math;

const filter = @import("../filters/bilateral.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Bilateral";
const max_sigma_s = filter.Data.max_sigma_s;

pub const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    k: filter.Data = .{},
};

fn Bilateral(comptime T: type, comptime join: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node1);
                if (join) {
                    zapi.requestFrameFilter(n, d.node2);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node1, n);
                defer src.deinit();

                const ref = if (join) zapi.initZFrame(d.node2, n) else src;
                defer if (join) ref.deinit();

                const dst = src.newVideoFrame2(d.k.planes);
                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.k.planes[plane])) continue;

                    const srcp = src.getReadSlice2(T, plane);
                    const refp = if (join) ref.getReadSlice2(T, plane) else srcp;
                    const dstp = dst.getWriteSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    filter.bilateral(T, srcp, refp, dstp, stride, w, h, plane, join, &d.k) catch {
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

fn bilateralFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    d.k.deinit();
    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.destroy(d);
}

pub fn bilateralCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node1, d.vi = map_in.getNodeVi("clip").?;
    const dt = hz.DataType.select(map_out, d.node1, d.vi, filter_name, false) catch return;

    const yuv: bool = (d.vi.format.colorFamily == vs.ColorFamily.YUV);
    const ssw: u5 = @intCast(d.vi.format.subSamplingW);
    const ssh: u5 = @intCast(d.vi.format.subSamplingH);

    var sigma_s: [3]f64 = undefined;
    var i: u32 = 0;
    // Hand-parsed for the derived chroma default, so the shared length check
    // has to be explicit: indices >= 3 would otherwise be ignored unvalidated.
    hz.checkArrayLen(map_in, "sigmaS", 3, filter_name, map_out) catch {
        zapi.freeNode(d.node1);
        return;
    };
    const m = map_in.numElements("sigmaS") orelse 0;
    while (i < 3) : (i += 1) {
        if (i < m) {
            sigma_s[i] = map_in.getFloat2(f64, "sigmaS", i).?;
            hz.checkFinite(sigma_s[i], filter_name, "sigmaS", i, map_out) catch {
                zapi.freeNode(d.node1);
                return;
            };
        } else {
            sigma_s[i] = filter.Data.sigmaSDefault(&sigma_s, i, yuv, ssw, ssh);
        }

        if (sigma_s[i] < 0) {
            map_out.setError("Bilateral: Invalid \"sigmaS\" assigned, must be non-negative float number");
            zapi.freeNode(d.node1);
            return;
        }

        // The spatial radius derivation runs in i32/u32 (orad = 2*sigma,
        // samples^2 * 4); bound sigma so none of it can overflow. 8192 means a
        // ~16k px window: far past any plane the truncated kernel accepts.
        if (sigma_s[i] > max_sigma_s) {
            hz.setArgError(map_out, "Bilateral: Invalid \"sigmaS\" assigned, must be at most {d}", .{max_sigma_s});
            zapi.freeNode(d.node1);
            return;
        }
    }

    const sigma_r = hz.getArray(f64, 0.02, 0, math.floatMax(f64), "sigmaR", filter_name, map_in, map_out, &.{d.node1}, &zapi) catch return;
    const algorithm = hz.getArray(i32, 0, 0, 2, "algorithm", filter_name, map_in, map_out, &.{d.node1}, &zapi) catch return;
    const pbfic_num = hz.getArray(u32, 0, 0, 256, "PBFICnum", filter_name, map_in, map_out, &.{d.node1}, &zapi) catch return;
    var planes: [3]bool = .{ true, true, true };
    hz.mapGetPlanes(map_in, map_out, &.{d.node1}, &planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;

    var plane_w: [3]u32 = undefined;
    var plane_h: [3]u32 = undefined;
    for (0..3) |p| {
        const sw: u5 = if (p == 0) 0 else ssw;
        const sh: u5 = if (p == 0) 0 else ssh;
        plane_w[p] = @as(u32, @intCast(d.vi.width)) >> sw;
        plane_h[p] = @as(u32, @intCast(d.vi.height)) >> sh;
    }

    d.k = filter.Data.init(.{
        .sigmaS = sigma_s,
        .sigmaR = sigma_r,
        .algorithm = algorithm,
        .PBFICnum = pbfic_num,
        .planes = planes,
        .yuv = yuv,
        .plane_w = plane_w,
        .plane_h = plane_h,
        .hist_len = hz.getHistLen(d.vi),
    }) catch |e| {
        hz.setArgError(map_out, "{s}: {s}", .{ filter_name, filter.Data.message(e) });
        zapi.freeNode(d.node1);
        return;
    };

    d.node2 = map_in.getNode("ref");
    const refb = d.node2 != null;
    const nodes = [_]?*vs.Node{ d.node1, d.node2 };
    if (refb) {
        hz.compareNodes(map_out, &nodes, .BIGGER_THAN, filter_name, &zapi) catch {
            d.k.deinit();
            return;
        };
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const rp2: vs.RequestPattern = if (refb and (d.vi.numFrames <= zapi.getVideoInfo(d.node2).numFrames)) .StrictSpatial else .FrameReuseLastOnly;
    const deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = rp2 },
    };

    const getFrame = hz.selectRefFilter(Bilateral, dt, refb, false);

    const ndeps: usize = if (refb) 2 else 1;
    zapi.createVideoFilter(out, filter_name, d.vi, getFrame, bilateralFree, .Parallel, deps[0..ndeps], data);
}
