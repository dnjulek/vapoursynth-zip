const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process_ct = @import("process/boxblur_comptime.zig");
const process_rt = @import("process/boxblur_runtime.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const math = std.math;
const ar = vs.ActivationReason;
const rp = vs.RequestPattern;
const fm = vs.FilterMode;
const pe = vs.MapPropertyError;
const ma = vs.MapAppendMode;

const allocator = std.heap.c_allocator;
pub const filter_name = "BoxBlur";

pub const BoxblurData = struct {
    node: *vs.Node,
    vi: *const vs.VideoInfo,
    hradius: u32,
    vradius: u32,
    hpasses: i32,
    vpasses: i32,
    tmp_size: u32,
    planes: [3]bool,
    dt: helper.DataType,
};

pub fn hvBlurRT(comptime T: type, src: ?*const vs.Frame, dst: ?*vs.Frame, d: *BoxblurData, vsapi: ?*const vs.API) void {
    const tmp1 = allocator.alloc(T, d.tmp_size) catch unreachable;
    const tmp2 = allocator.alloc(T, d.tmp_size) catch unreachable;
    defer allocator.free(tmp1);
    defer allocator.free(tmp2);

    var plane: c_int = 0;
    while (plane < d.vi.format.numPlanes) : (plane += 1) {
        if (!(d.planes[@intCast(plane)])) {
            continue;
        }

        const srcp: [*]const T = @ptrCast(@alignCast(vsapi.?.getReadPtr.?(src, plane)));
        const dstp: [*]T = @ptrCast(@alignCast(vsapi.?.getWritePtr.?(dst, plane)));
        var stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
        stride >>= (@sizeOf(T) >> 1);

        const h: u32 = @intCast(vsapi.?.getFrameHeight.?(src, plane));
        const w: u32 = @intCast(vsapi.?.getFrameWidth.?(src, plane));

        process_rt.hblur(
            T,
            srcp,
            dstp,
            stride,
            w,
            h,
            d.hradius,
            d.hpasses,
            tmp1.ptr,
            tmp2.ptr,
        );

        process_rt.vblur(
            T,
            dstp,
            dstp,
            stride,
            w,
            h,
            d.vradius,
            d.vpasses,
            tmp1.ptr,
            tmp2.ptr,
        );
    }
}

export fn boxBlurRTGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BoxblurData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == ar.AllFramesReady) {
        const src = vsapi.?.getFrameFilter.?(n, d.node, frame_ctx);
        defer vsapi.?.freeFrame.?(src);
        const dst = helper.newVideoFrame2(src, &d.planes, core, vsapi);

        switch (d.dt) {
            .U8 => hvBlurRT(u8, src, dst, d, vsapi),
            .U16 => hvBlurRT(u16, src, dst, d, vsapi),
            .F32 => hvBlurRT(f32, src, dst, d, vsapi),
        }

        return dst;
    }

    return null;
}

export fn boxBlurCTGetFrame(n: c_int, activation_reason: ar, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BoxblurData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == ar.Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == ar.AllFramesReady) {
        const src = vsapi.?.getFrameFilter.?(n, d.node, frame_ctx);
        defer vsapi.?.freeFrame.?(src);
        const dst = helper.newVideoFrame2(src, &d.planes, core, vsapi);

        var plane: c_int = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            if (!(d.planes[@intCast(plane)])) {
                continue;
            }

            const srcp = vsapi.?.getReadPtr.?(src, plane);
            const dstp = vsapi.?.getWritePtr.?(dst, plane);
            const stride: usize = @intCast(vsapi.?.getStride.?(src, plane));
            const h: u32 = @intCast(vsapi.?.getFrameHeight.?(src, plane));
            const w: u32 = @intCast(vsapi.?.getFrameWidth.?(src, plane));

            switch (d.dt) {
                .U8 => process_ct.hvBlur(u8, srcp, dstp, stride, w, h, d.vradius),
                .U16 => process_ct.hvBlur(u16, srcp, dstp, stride, w, h, d.vradius),
                .F32 => process_ct.hvBlur(f32, srcp, dstp, stride, w, h, d.vradius),
            }
        }

        return dst;
    }

    return null;
}

export fn boxBlurFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *BoxblurData = @ptrCast(@alignCast(instance_data));

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn boxBlurCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: BoxblurData = undefined;
    var err: pe = undefined;

    d.node = vsapi.?.mapGetNode.?(in, "clip", 0, &err).?;
    d.vi = vsapi.?.getVideoInfo.?(d.node);
    d.dt = @enumFromInt(d.vi.format.bytesPerSample);
    d.tmp_size = @intCast(@max(d.vi.width, d.vi.height));

    var nodes = [_]?*vs.Node{d.node};
    var planes = [3]bool{ true, true, true };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hradius = vsh.mapGetN(u32, in, "hradius", 0, vsapi) orelse 1;
    d.vradius = vsh.mapGetN(u32, in, "vradius", 0, vsapi) orelse 1;
    d.hpasses = vsh.mapGetN(i32, in, "hpasses", 0, vsapi) orelse 1;
    d.vpasses = vsh.mapGetN(i32, in, "vpasses", 0, vsapi) orelse 1;
    const use_rt: bool = (d.hradius != d.vradius) or (d.hradius > 30) or (d.hpasses > 1) or (d.vpasses > 1);

    const vblur = (d.vradius > 0) and (d.vpasses > 0);
    const hblur = (d.hradius > 0) and (d.hpasses > 0);
    if (!vblur and !hblur) {
        vsapi.?.mapSetError.?(out, filter_name ++ ": nothing to be performed");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    const data: *BoxblurData = allocator.create(BoxblurData) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = rp.StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, if (use_rt) boxBlurRTGetFrame else boxBlurCTGetFrame, boxBlurFree, fm.Parallel, &deps, deps.len, data, core);
}
