const std = @import("std");
const math = std.math;

const filter_ssim = @import("../filters/ssimulacra2.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "SSIMULACRA2";
const Scratch = []align(vszip.vec_len) f32;

/// How one clip's frames are read: native RGB (int via LUT, or f32), or the
/// RGBS that toRGBS made from YUV/GRAY.
const Source = struct {
    kind: filter_ssim.Input.Kind = .f32,
    srgb: bool = true,
    lut: []f32 = &.{},

    fn input(self: *const Source, f: anytype) filter_ssim.Input {
        return .{
            .kind = self.kind,
            .planes = .{ f.getReadSlice(0).ptr, f.getReadSlice(1).ptr, f.getReadSlice(2).ptr },
            .stride = f.getStride(0),
            .lut = self.lut,
            .srgb = self.srgb,
        };
    }
};

const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
    src1: Source = .{},
    src2: Source = .{},
    // Reusable scratch buffers; getFrame runs concurrently so each call pops
    // its own buffer (or allocates one) and pushes it back when done.
    pool: std.ArrayList(Scratch) = .empty,
    pool_mutex: std.Io.Mutex = .init,
};

fn acquireScratch(d: *Data, size: usize) Scratch {
    d.pool_mutex.lockUncancelable(vszip.io);
    defer d.pool_mutex.unlock(vszip.io);
    if (d.pool.pop()) |buf| return buf;
    return allocator.alignedAlloc(f32, vszip.alignment, size) catch unreachable;
}

fn releaseScratch(d: *Data, buf: Scratch) void {
    d.pool_mutex.lockUncancelable(vszip.io);
    defer d.pool_mutex.unlock(vszip.io);
    d.pool.append(allocator, buf) catch unreachable;
}

fn ssimulacra2GetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node1);
        zapi.requestFrameFilter(n, d.node2);
    } else if (activation_reason == .AllFramesReady) {
        const src1 = zapi.initZFrame(d.node1, n);
        const src2 = zapi.initZFrame(d.node2, n);
        defer src1.deinit();
        defer src2.deinit();

        // only the frame prop matters: pass the (refcounted) input frame on
        const dst = src1.copyFrame();
        const w = src1.getWidth(0);
        const h = src1.getHeight(0);

        const scratch = acquireScratch(d, filter_ssim.scratchSize(w, h));
        defer releaseScratch(d, scratch);
        const val = filter_ssim.process(d.src1.input(&src1), d.src2.input(&src2), w, h, scratch);
        const dst_prop = dst.getPropertiesRW();
        dst_prop.setFloat("SSIMULACRA2", val, .Replace);
        return dst.frame;
    }
    return null;
}

fn ssimulacraFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    for (d.pool.items) |buf| {
        allocator.free(buf);
    }
    d.pool.deinit(allocator);

    allocator.free(d.src1.lut);
    allocator.free(d.src2.lut);
    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.destroy(d);
}

pub fn ssimulacraCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node1, const vi1 = map_in.getNodeVi("reference").?;
    d.node2, const vi2 = map_in.getNodeVi("distorted").?;

    if ((vi1.width != vi2.width) or (vi1.height != vi2.height)) {
        map_out.setError(filter_name ++ " : clips must have the same dimensions.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    if (vi1.numFrames != vi2.numFrames) {
        map_out.setError(filter_name ++ " : clips must have the same length.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    if (((vi1.format.sampleType == .Float) and (vi1.format.bitsPerSample == 16)) or
        ((vi2.format.sampleType == .Float) and (vi2.format.bitsPerSample == 16)))
    {
        map_out.setError(filter_name ++ " : half-float (f16) format is not supported.");
        zapi.freeNode(d.node1);
        zapi.freeNode(d.node2);
        return;
    }

    prepare(&d.node1, &d.src1, &zapi);
    prepare(&d.node2, &d.src2, &zapi);

    const vi_out = zapi.getVideoInfo(d.node1);
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, filter_name, vi_out, ssimulacra2GetFrame, ssimulacraFree, .Parallel, &deps, data);
}

/// RGB clips are read natively (8-16 bit int through a sample -> linear LUT,
/// f32 decoded per row); YUV and GRAY go through toRGBS first. The sRGB
/// decode runs inside the filter (libjxl's TF_SRGB, see filters/srgb.zig)
/// unless frame 0 says the clip is already linear.
fn prepare(node: *?*vs.Node, src: *Source, zapi: *const ZAPI) void {
    const fmt = zapi.getVideoInfo(node.*).format;
    const native = fmt.colorFamily == .RGB and
        ((fmt.sampleType == .Integer and fmt.bitsPerSample <= 16) or (fmt.sampleType == .Float and fmt.bitsPerSample == 32));
    if (!native) node.* = hz.toRGBS(node.*, zapi);

    const frame = zapi.getFrame(0, node.*, null);
    defer zapi.freeFrame(frame);
    src.srgb = zapi.initZMap(zapi.getFramePropertiesRO(frame)).getTransfer() != .LINEAR;

    const f = zapi.getVideoInfo(node.*).format;
    if (f.sampleType == .Integer) {
        src.kind = if (f.bytesPerSample == 1) .u8 else .u16;
        src.lut = allocator.alloc(f32, if (f.bytesPerSample == 1) 256 else 65536) catch unreachable;
        filter_ssim.buildLut(src.lut, @intCast(f.bitsPerSample), src.srgb);
    } else {
        src.kind = .f32;
    }
}
