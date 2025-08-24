const std = @import("std");
const math = std.math;

const boxblur_ct = @import("../filters/boxblur_comptime.zig");
const boxblur_rt = @import("../filters/boxblur_runtime.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "BoxBlur";

pub const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    hradius: u32 = 0,
    vradius: u32 = 0,
    hpasses: i32 = 0,
    vpasses: i32 = 0,
    tmp_size: u32 = 0,
    planes: [3]bool = .{ true, true, true },
};

pub fn BoxBlurCT(comptime T: type, radius: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame2(d.planes);

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) continue;

                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    boxblur_ct.hvBlur(T, radius, srcp, dstp, stride, w, h);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn BoxBlurRT(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                const dst = src.newVideoFrame2(d.planes);
                defer src.deinit();

                const temp1 = allocator.alloc(T, d.tmp_size) catch unreachable;
                const temp2 = allocator.alloc(T, d.tmp_size) catch unreachable;
                defer allocator.free(temp1);
                defer allocator.free(temp2);

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) continue;

                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);

                    boxblur_rt.hblur(T, srcp, dstp, stride, w, h, d.hradius, d.hpasses, temp1, temp2);
                    boxblur_rt.vblur(T, dstp, dstp, stride, w, h, d.vradius, d.vpasses, temp1, temp2);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn boxBlurFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn boxBlurCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;
    const dt = hz.DataType.select(map_out, d.node, d.vi, filter_name, false) catch return;

    d.tmp_size = @intCast(@max(d.vi.width, d.vi.height));

    var nodes = [_]?*vs.Node{d.node};
    hz.mapGetPlanes(map_in, map_out, &nodes, &d.planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;

    d.hradius = map_in.getValue(u32, "hradius") orelse 1;
    d.vradius = map_in.getValue(u32, "vradius") orelse 1;
    d.hpasses = map_in.getValue(i32, "hpasses") orelse 1;
    d.vpasses = map_in.getValue(i32, "vpasses") orelse 1;

    const vblur = (d.vradius > 0) and (d.vpasses > 0);
    const hblur = (d.hradius > 0) and (d.hpasses > 0);
    if (!vblur and !hblur) {
        map_out.setError(filter_name ++ ": nothing to be performed");
        zapi.freeNode(d.node);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const use_rt: bool = (d.hradius != d.vradius) or (d.hradius > 22) or (d.hpasses > 1) or (d.vpasses > 1);
    var get_frame: vs.FilterGetFrame = undefined;
    if (use_rt) {
        get_frame = switch (dt) {
            .U8 => &BoxBlurRT(u8).getFrame,
            .U16 => &BoxBlurRT(u16).getFrame,
            .F16 => &BoxBlurRT(f16).getFrame,
            .F32 => &BoxBlurRT(f32).getFrame,
            .U32 => unreachable,
        };
    } else {
        get_frame = switch (d.hradius) {
            inline 1...22 => |r| switch (dt) {
                .U8 => &BoxBlurCT(u8, r).getFrame,
                .U16 => &BoxBlurCT(u16, r).getFrame,
                .F16 => &BoxBlurCT(f16, r).getFrame,
                .F32 => &BoxBlurCT(f32, r).getFrame,
                .U32 => unreachable,
            },
            else => unreachable,
        };
    }

    zapi.createVideoFilter(out, filter_name, d.vi, get_frame, boxBlurFree, .Parallel, &deps, data);
}
