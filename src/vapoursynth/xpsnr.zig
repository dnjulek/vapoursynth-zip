const std = @import("std");
const math = std.math;

const filter = @import("../filters/xpsnr.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;
const Mutex = std.Io.Mutex;

const allocator = std.heap.c_allocator;
pub const filter_name = "XPSNR";

pub const Data = struct {
    node1: *vs.Node = undefined,
    node2: *vs.Node = undefined,
    vi: *const vs.VideoInfo = undefined,
    mutex: Mutex = .init,

    depth: u6 = 0,
    num_comps: u8 = 0,
    frame_rate: u32 = 0,
    max_error_64: u64 = 0,
    num_frames_64: u64 = 0,
    sum_wdist: [3]f64 = .{ 0, 0, 0 },
    sum_xpsnr: [3]f64 = .{ 0, 0, 0 },
    width: [3]u32 = .{ 0, 0, 0 },
    height: [3]u32 = .{ 0, 0, 0 },
    temporal: bool = true,
    verbose: bool = true,
};

fn XPSNR(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node1);
                zapi.requestFrameFilter(n, d.node2);
                if (d.temporal) {
                    if (n > 0) zapi.requestFrameFilter(n - 1, d.node1);
                    if ((d.frame_rate >= 32) and (n > 1)) zapi.requestFrameFilter(n - 2, d.node1);
                }
            } else if (activation_reason == .AllFramesReady) {
                const src1 = zapi.initZFrame(d.node1, n);
                const src2 = zapi.initZFrame(d.node2, n);
                defer src1.deinit();
                defer src2.deinit();
                const dst = src2.copyFrame();

                // Temporal activity is computed against the actual previous
                // frame(s); a missing previous frame (n == 0/1) means "all
                // zeros", matching the zero-initialized state of the old
                // serial implementation.
                var src_p1: ?@TypeOf(src1) = null;
                var src_p2: ?@TypeOf(src1) = null;
                if (d.temporal and (n > 0)) src_p1 = zapi.initZFrame(d.node1, n - 1);
                if (d.temporal and (d.frame_rate >= 32) and (n > 1)) src_p2 = zapi.initZFrame(d.node1, n - 2);
                defer if (src_p1) |f| f.deinit();
                defer if (src_p2) |f| f.deinit();

                const orgp = src1.getReadSlices2(T);
                const recp = src2.getReadSlices2(T);
                const prv1: ?[]const T = if (src_p1) |f| f.getReadSlice2(T, 0) else null;
                const prv2: ?[]const T = if (src_p2) |f| f.getReadSlice2(T, 0) else null;
                var wsse64 = [3]u64{ 0, 0, 0 };
                var cur_xpsnr = [3]f64{ math.inf(f64), math.inf(f64), math.inf(f64) };

                var strides: [3]u32 = .{ 0, 0, 0 };
                var c: u32 = 0;
                while (c < d.num_comps) : (c += 1) {
                    strides[c] = src1.getStride2(T, c);
                }

                filter.getWSSE(T, orgp, recp, prv1, prv2, &wsse64, d.width, d.height, strides, d.depth, d.num_comps, d.frame_rate, d.temporal);

                var i: u32 = 0;
                while (i < d.num_comps) : (i += 1) {
                    const sqrt_wsse: f64 = @sqrt(@floatFromInt(wsse64[i]));
                    cur_xpsnr[i] = filter.getFrameXPSNR(sqrt_wsse, d.width[i], d.height[i], d.max_error_64);
                }

                d.mutex.lockUncancelable(vszip.io);
                d.num_frames_64 += 1;
                i = 0;
                while (i < d.num_comps) : (i += 1) {
                    d.sum_wdist[i] += @sqrt(@as(f64, @floatFromInt(wsse64[i])));
                    d.sum_xpsnr[i] += cur_xpsnr[i];
                }
                d.mutex.unlock(vszip.io);

                const dst_props = dst.getPropertiesRW();
                dst_props.setFloat("XPSNR_Y", cur_xpsnr[0], .Replace);
                dst_props.setFloat("XPSNR_U", cur_xpsnr[1], .Replace);
                dst_props.setFloat("XPSNR_V", cur_xpsnr[2], .Replace);

                return dst.frame;
            }
            return null;
        }
    };
}

fn xpsnrFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    if (d.verbose) {
        var stdout_buffer: [1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writerStreaming(vszip.io, &stdout_buffer);
        const stdout = &stdout_writer.interface;
        stdout.print("XPSNR average, {} frames  ", .{d.num_frames_64}) catch unreachable;
        const char = [_]u8{ 'y', 'u', 'v' };

        for (0..d.num_comps) |i| {
            const xpsnr = filter.getAvgXPSNR(d.sum_wdist[i], d.sum_xpsnr[i], d.width[i], d.height[i], d.max_error_64, d.num_frames_64);
            stdout.print("{c}: {d:.04}  ", .{ char[i], xpsnr }) catch unreachable;
        }

        stdout.print("\n", .{}) catch unreachable;
        stdout.flush() catch unreachable;
    }

    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.destroy(d);
}

pub fn xpsnrCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    _ = user_data;
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node1, const vi1 = map_in.getNodeVi("reference").?;

    if (vi1.format.colorFamily != .YUV) {
        map_out.setError(filter_name ++ " : only supports YUV format clips");
        zapi.freeNode(d.node1);
        return;
    }

    if ((vi1.format.bitsPerSample != 8) and (vi1.format.bitsPerSample != 10)) {
        map_out.setError(filter_name ++ " : only supports 8 or 10 bit clips");
        zapi.freeNode(d.node1);
        return;
    }

    if (((vi1.width & 1) != 0) or ((vi1.height & 1) != 0)) {
        map_out.setError(filter_name ++ " : only supports even width and height");
        zapi.freeNode(d.node1);
        return;
    }

    d.node2, const vi2 = map_in.getNodeVi("distorted").?;
    const bps1: u32 = @intCast(vi1.format.bitsPerSample);
    const bps2: u32 = @intCast(vi2.format.bitsPerSample);
    if (bps1 < bps2) {
        d.node1 = hz.bitDepth(bps2, d.node1, .none, &zapi);
    } else if (bps1 > bps2) {
        d.node2 = hz.bitDepth(bps1, d.node2, .none, &zapi);
    }

    d.vi = zapi.getVideoInfo(d.node1);
    hz.compareNodes(map_out, &.{ d.node1, d.node2 }, .SAME_LEN, filter_name, &zapi) catch return;

    d.temporal = map_in.getBool("temporal") orelse true;

    d.verbose = map_in.getBool("verbose") orelse true;

    d.depth = @intCast(d.vi.format.bitsPerSample);
    d.max_error_64 = math.shl(u64, 1, d.depth) - 1;
    d.max_error_64 *= d.max_error_64;

    d.frame_rate = if (vi2.fpsDen != 0)
        @intCast(@divTrunc(vi2.fpsNum, vi2.fpsDen))
    else if (d.vi.fpsDen != 0)
        @intCast(@divTrunc(d.vi.fpsNum, d.vi.fpsDen))
    else
        0;
    d.num_comps = @intCast(d.vi.format.numPlanes);

    const whv = whFromVi(d.vi);
    d.width = whv.w;
    d.height = whv.h;

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .General },
        .{ .source = d.node2, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = if (d.vi.format.bytesPerSample == 1) &XPSNR(u8).getFrame else &XPSNR(u16).getFrame;
    zapi.createVideoFilter(out, filter_name, d.vi, gf, xpsnrFree, .Parallel, &deps, data);
}

fn whFromVi(vi: *const vs.VideoInfo) struct { w: [3]u32, h: [3]u32 } {
    const w: u32 = @intCast(vi.width);
    const h: u32 = @intCast(vi.height);

    if (vi.format.numPlanes == 1) {
        return .{ .w = [3]u32{ w, 0, 0 }, .h = [3]u32{ h, 0, 0 } };
    }

    const w_chroma: u32 = w >> @as(u5, @intCast(vi.format.subSamplingW));
    const h_chroma: u32 = h >> @as(u5, @intCast(vi.format.subSamplingH));
    return .{ .w = [3]u32{ w, w_chroma, w_chroma }, .h = [3]u32{ h, h_chroma, h_chroma } };
}
