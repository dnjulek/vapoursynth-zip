const std = @import("std");
const vszip = @import("../vszip.zig");
const helper = @import("../helper.zig");
const boxblur_ct = @import("../filters/boxblur_comptime.zig");
const boxblur_rt = @import("../filters/boxblur_runtime.zig");

const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;
const math = std.math;

const allocator = std.heap.c_allocator;
pub const filter_name = "BoxBlur";

pub const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    hradius: u32,
    vradius: u32,
    hpasses: i32,
    vpasses: i32,
    tmp_size: u32,
    planes: [3]bool,
};

pub fn BoxBlurCT(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

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
                    boxblur_ct.hvBlur(T, srcp, dstp, stride, w, h, d.hradius);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn BoxBlurRT(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));

            if (activation_reason == .Initial) {
                vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
            } else if (activation_reason == .AllFramesReady) {
                var src = zapi.Frame.init(d.node, n, frame_ctx, core, vsapi);
                defer src.deinit();
                const dst = src.newVideoFrame2(d.planes);

                boxblur_rt.hvBlur(T, src, dst, d);
                return dst.frame;
            }

            return null;
        }
    };
}

export fn boxBlurFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn boxBlurCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = undefined;

    var map = zapi.Map.init(in, out, vsapi);
    d.node, d.vi = map.getNodeVi("clip");
    const dt = helper.DataType.select(map, d.node, d.vi, filter_name) catch return;

    d.tmp_size = @intCast(@max(d.vi.width, d.vi.height));

    var nodes = [_]?*vs.Node{d.node};
    var planes = [3]bool{ true, true, true };
    helper.mapGetPlanes(in, out, &nodes, &planes, d.vi.format.numPlanes, filter_name, vsapi) catch return;
    d.planes = planes;

    d.hradius = map.getInt(u32, "hradius") orelse 1;
    d.vradius = map.getInt(u32, "vradius") orelse 1;
    d.hpasses = map.getInt(i32, "hpasses") orelse 1;
    d.vpasses = map.getInt(i32, "vpasses") orelse 1;

    const vblur = (d.vradius > 0) and (d.vpasses > 0);
    const hblur = (d.hradius > 0) and (d.hpasses > 0);
    if (!vblur and !hblur) {
        map.setError(filter_name ++ ": nothing to be performed");
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

    const use_rt: bool = (d.hradius != d.vradius) or (d.hradius > 22) or (d.hpasses > 1) or (d.vpasses > 1);
    const getFrame = switch (dt) {
        .U8 => if (use_rt) &BoxBlurRT(u8).getFrame else &BoxBlurCT(u8).getFrame,
        .U16 => if (use_rt) &BoxBlurRT(u16).getFrame else &BoxBlurCT(u16).getFrame,
        .F16 => if (use_rt) &BoxBlurRT(f16).getFrame else &BoxBlurCT(f16).getFrame,
        .F32 => if (use_rt) &BoxBlurRT(f32).getFrame else &BoxBlurCT(f32).getFrame,
    };

    vsapi.?.createVideoFilter.?(out, filter_name, d.vi, getFrame, boxBlurFree, .Parallel, &deps, deps.len, data, core);
}
