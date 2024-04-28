const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const process_ct = @import("process/boxblur_comptime.zig");
const process_rt = @import("process/boxblur_runtime.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "BoxBlur";

pub const BoxblurData = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    hradius: u32,
    vradius: u32,
    hpasses: i32,
    vpasses: i32,
    tmp_size: u32,
    planes: [3]bool,
    dt: helper.DataType,
};

export fn boxBlurRTGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BoxblurData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node, n, frame_ctx, core, vsapi);
        defer src.deinit();
        const dst = src.newVideoFrame2(d.planes);

        switch (d.dt) {
            .U8 => hvBlurRT(u8, src, dst, d),
            .U16 => hvBlurRT(u16, src, dst, d),
            .F16 => hvBlurRT(f16, src, dst, d),
            .F32 => hvBlurRT(f32, src, dst, d),
        }

        return dst.frame;
    }

    return null;
}

export fn boxBlurCTGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
    _ = frame_data;
    const d: *BoxblurData = @ptrCast(@alignCast(instance_data));

    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.Frame.init(d.node, n, frame_ctx, core, vsapi);
        defer src.deinit();
        const dst = src.newVideoFrame2(d.planes);

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            if (!(d.planes[plane])) {
                continue;
            }

            const srcp = src.getReadSlice(plane);
            const dstp = dst.getWriteSlice(plane);
            const w, const h, const stride = src.getDimensions(plane);

            switch (d.dt) {
                .U8 => process_ct.hvBlur(u8, srcp, dstp, stride, w, h, d.vradius),
                .U16 => process_ct.hvBlur(u16, srcp, dstp, stride, w, h, d.vradius),
                .F16 => process_ct.hvBlur(f16, srcp, dstp, stride, w, h, d.vradius),
                .F32 => process_ct.hvBlur(f32, srcp, dstp, stride, w, h, d.vradius),
            }
        }

        return dst.frame;
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

    var map = zapi.Map.init(in, out, vsapi);
    d.node, d.vi = map.getNodeVi("clip");

    d.dt = helper.DataType.select(map, d.node, d.vi, filter_name) catch return;
    d.tmp_size = @intCast(@max(d.vi.width, d.vi.height));

    var nodes = [_]?*vs.Node{d.node};
    var planes = [3]bool{ true, true, true };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hradius = map.getInt(u32, "hradius") orelse 1;
    d.vradius = map.getInt(u32, "vradius") orelse 1;
    d.hpasses = map.getInt(i32, "hpasses") orelse 1;
    d.vpasses = map.getInt(i32, "vpasses") orelse 1;

    const use_rt: bool = (d.hradius != d.vradius) or (d.hradius > 22) or (d.hpasses > 1) or (d.vpasses > 1);

    const vblur = (d.vradius > 0) and (d.vpasses > 0);
    const hblur = (d.hradius > 0) and (d.hpasses > 0);
    if (!vblur and !hblur) {
        map.setError(filter_name ++ ": nothing to be performed");
        vsapi.?.freeNode.?(d.node);
        return;
    }

    const data: *BoxblurData = allocator.create(BoxblurData) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .StrictSpatial,
        },
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, if (use_rt) boxBlurRTGetFrame else boxBlurCTGetFrame, boxBlurFree, .Parallel, &deps, deps.len, data, core);
}

fn hvBlurRT(comptime T: type, src: zapi.Frame, dst: zapi.Frame, d: *BoxblurData) void {
    const temp1 = allocator.alloc(T, d.tmp_size) catch unreachable;
    const temp2 = allocator.alloc(T, d.tmp_size) catch unreachable;
    defer allocator.free(temp1);
    defer allocator.free(temp2);

    var plane: u32 = 0;
    while (plane < d.vi.format.numPlanes) : (plane += 1) {
        if (!(d.planes[plane])) {
            continue;
        }

        const src8 = src.getReadSlice(plane);
        const dst8 = dst.getWriteSlice(plane);
        const srcp: []const T = @as([*]const T, @ptrCast(@alignCast(src8)))[0..src8.len];
        const dstp: []T = @as([*]T, @ptrCast(@alignCast(dst8)))[0..dst8.len];
        const w, const h, var stride = src.getDimensions(plane);
        stride >>= (@sizeOf(T) >> 1);

        process_rt.hblur(T, srcp, dstp, stride, w, h, d.hradius, d.hpasses, temp1, temp2);
        process_rt.vblur(T, dstp, dstp, stride, w, h, d.vradius, d.vpasses, temp1, temp2);
    }
}
