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
pub const params = "clip:vnode;planes:int[]:opt;hradius:int:opt;hpasses:int:opt;vradius:int:opt;vpasses:int:opt";

pub const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    p: boxblur_rt.Params = undefined,
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
                    boxblur_ct.hvBlur(T, radius, srcp, dstp, stride, w, h) catch return hz.frameOom(&zapi, filter_name, dst);
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

                // one scratch block per frame, shared by every plane (plane 0 is the widest)
                const sc = boxblur_rt.Scratch(T).init(@intCast(d.vi.width), d.tmp_size, d.p.vradius, @intCast(@max(d.p.vpasses, 1))) catch {
                    dst.deinit();
                    zapi.setFilterError(filter_name ++ ": out of memory");
                    return null;
                };
                defer sc.deinit();

                // multi-pass vblur ping-pongs between full planes; a VS frame
                // serves as the scratch plane (the core pools frame memory).
                // Created lazily: the common single-pass cases don't need it.
                var scratch: ?@TypeOf(dst) = null;
                defer if (scratch) |s| s.deinit();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    if (!(d.planes[plane])) continue;

                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const w, const h, const stride = src.getDimensions2(T, plane);
                    const tmp = struct {
                        scratch: *?@TypeOf(dst),
                        src: @TypeOf(src),
                        plane: u32,

                        pub fn get(self: @This()) error{}![]T {
                            if (self.scratch.* == null) self.scratch.* = self.src.newVideoFrame();
                            return self.scratch.*.?.getWriteSlice2(T, self.plane);
                        }
                    }{ .scratch = &scratch, .src = src, .plane = plane };
                    boxblur_rt.blurPlane(T, d.p, srcp, dstp, stride, w, h, &sc, tmp) catch return hz.frameOom(&zapi, filter_name, dst);
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

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;
    const dt = hz.DataType.select(map_out, d.node, d.vi, filter_name, false) catch return;

    d.tmp_size = @intCast(@max(d.vi.width, d.vi.height));

    var nodes = [_]?*vs.Node{d.node};
    hz.mapGetPlanes(map_in, map_out, &nodes, &d.planes, d.vi.format.numPlanes, filter_name, &zapi) catch return;

    d.p = .{
        .hradius = map_in.getValue(u32, "hradius") orelse 1,
        .vradius = map_in.getValue(u32, "vradius") orelse 1,
        .hpasses = map_in.getValue(i32, "hpasses") orelse 1,
        .vpasses = map_in.getValue(i32, "vpasses") orelse 1,
    };

    {
        const num_planes: u32 = @intCast(d.vi.format.numPlanes);
        const ssw: u5 = @intCast(d.vi.format.subSamplingW);
        const ssh: u5 = @intCast(d.vi.format.subSamplingH);
        var pw: [3]u32 = undefined;
        var ph: [3]u32 = undefined;
        for (0..num_planes) |p| {
            pw[p] = @as(u32, @intCast(d.vi.width)) >> if (p == 0) 0 else ssw;
            ph[p] = @as(u32, @intCast(d.vi.height)) >> if (p == 0) 0 else ssh;
        }
        const int_bytes: u8 = switch (dt) {
            .U8 => 1,
            .U16 => 2,
            else => 0,
        };
        d.p.check(int_bytes, d.planes[0..num_planes], pw[0..num_planes], ph[0..num_planes]) catch |e| {
            hz.setArgError(map_out, "{s}: {s}", .{ filter_name, boxblur_rt.Params.message(e) });
            zapi.freeNode(d.node);
            return;
        };
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    // CT vs RT: see boxblur_rt.Params.useRT.
    const is_float = (dt == .F16) or (dt == .F32);
    const use_rt = d.p.useRT(is_float);
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
        get_frame = switch (d.p.hradius) {
            inline 1...8 => |r| switch (dt) {
                .F16 => &BoxBlurCT(f16, r).getFrame,
                .F32 => &BoxBlurCT(f32, r).getFrame,
                .U8, .U16, .U32 => unreachable,
            },
            else => unreachable,
        };
    }

    zapi.createVideoFilter(out, filter_name, d.vi, get_frame, boxBlurFree, .Parallel, &deps, data);
}
