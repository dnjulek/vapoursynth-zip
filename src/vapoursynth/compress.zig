const std = @import("std");

const filter = @import("../filters/compress.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const Codec = filter.Codec;

const allocator = std.heap.c_allocator;
pub const filter_name = "Compress";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    qt: filter.QuantTables = .{},
    process: [3]bool = .{ true, true, true },
};

fn Compress(comptime codec: Codec) type {
    return struct {
        pub fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (ar == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (ar == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame2(d.process);

                var plane: usize = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!d.process[plane]) continue;

                    const srcp = src.getReadSlice(plane);
                    const dstp = dst.getWriteSlice(plane);
                    const w, const h, const stride = src.getDimensions(plane);
                    filter.processPlane(codec, srcp, dstp, w, h, stride, &d.qt, plane != 0);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn compressFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn compressCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;

    const fmt = d.vi.format;
    if ((fmt.sampleType != .Integer) or (fmt.bitsPerSample != 8) or
        ((fmt.colorFamily != .Gray) and (fmt.colorFamily != .YUV)))
    {
        map_out.setError(filter_name ++ ": only 8-bit integer Gray or YUV formats are supported.");
        zapi.freeNode(d.node);
        return;
    }

    const codec_i = map_in.getValue(i32, "codec") orelse 0;
    if (codec_i < 0 or codec_i > 1) {
        map_out.setError(filter_name ++ ": codec must be 0 (mpeg2) or 1 (jpeg).");
        zapi.freeNode(d.node);
        return;
    }
    const codec: Codec = @enumFromInt(codec_i);

    switch (codec) {
        .mpeg2 => {
            const qscale = map_in.getValue(i32, "qscale") orelse 8;
            if (qscale < 1 or qscale > 31) {
                map_out.setError(filter_name ++ ": qscale must be between 1 and 31.");
                zapi.freeNode(d.node);
                return;
            }
            const dc_prec = map_in.getValue(i32, "dc_prec") orelse 0;
            if (dc_prec < 0 or dc_prec > 3) {
                map_out.setError(filter_name ++ ": dc_prec must be between 0 and 3.");
                zapi.freeNode(d.node);
                return;
            }
            d.qt.buildMpeg2(qscale, @intCast(dc_prec));
        },
        .jpeg => {
            const quality = map_in.getValue(i32, "quality") orelse 50;
            if (quality < 1 or quality > 100) {
                map_out.setError(filter_name ++ ": quality must be between 1 and 100.");
                zapi.freeNode(d.node);
                return;
            }
            d.qt.buildJpeg(quality);
        },
    }

    const chroma = map_in.getBool("chroma") orelse true;
    d.process = .{ true, chroma, chroma };

    const data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = switch (codec) {
        .mpeg2 => &Compress(.mpeg2).getFrame,
        .jpeg => &Compress(.jpeg).getFrame,
    };

    zapi.createVideoFilter(out, filter_name, d.vi, gf, compressFree, .Parallel, &deps, data);
}
