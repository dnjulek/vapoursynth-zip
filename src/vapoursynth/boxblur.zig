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

                const hb = (d.hradius > 0) and (d.hpasses > 0);
                const vb = (d.vradius > 0) and (d.vpasses > 0);

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

                    if (vb) {
                        if ((d.vpasses == 1) and (!hb)) {
                            boxblur_rt.vblur(T, srcp, dstp, dstp, stride, w, h, d.vradius, 1);
                        } else if ((d.vpasses == 1) and (h >= 2 * d.vradius + 2)) {
                            boxblur_rt.hvBlurFused(T, srcp, dstp, stride, w, h, d.hradius, d.hpasses, d.vradius, temp1, temp2);
                        } else {
                            if (scratch == null) scratch = src.newVideoFrame();
                            const tmpp = scratch.?.getWriteSlice2(T, plane);
                            if (hb) {
                                // land the hblur result so the vblur sweeps end in dstp
                                const h_out: []T = if (@mod(d.vpasses, 2) == 1) tmpp else dstp;
                                boxblur_rt.hblur(T, srcp, h_out, stride, w, h, d.hradius, d.hpasses, temp1, temp2);
                                boxblur_rt.vblur(T, h_out, tmpp, dstp, stride, w, h, d.vradius, d.vpasses);
                            } else {
                                boxblur_rt.vblur(T, srcp, tmpp, dstp, stride, w, h, d.vradius, d.vpasses);
                            }
                        }
                    } else {
                        boxblur_rt.hblur(T, srcp, dstp, stride, w, h, d.hradius, d.hpasses, temp1, temp2);
                    }
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

    {
        const ssw: u5 = @intCast(d.vi.format.subSamplingW);
        const ssh: u5 = @intCast(d.vi.format.subSamplingH);
        var p: u32 = 0;
        while (p < d.vi.format.numPlanes) : (p += 1) {
            if (!d.planes[p]) continue;
            const sw: u5 = if (p == 0) 0 else ssw;
            const sh: u5 = if (p == 0) 0 else ssh;
            const pw: u32 = @as(u32, @intCast(d.vi.width)) >> sw;
            const ph: u32 = @as(u32, @intCast(d.vi.height)) >> sh;
            if (hblur and (@as(u64, d.hradius) * 2 >= pw)) {
                map_out.setError(filter_name ++ ": hradius too large; 2*hradius must be < the (smallest processed) plane width.");
                zapi.freeNode(d.node);
                return;
            }
            if (vblur and (@as(u64, d.vradius) * 2 >= ph)) {
                map_out.setError(filter_name ++ ": vradius too large; 2*vradius must be < the (smallest processed) plane height.");
                zapi.freeNode(d.node);
                return;
            }
        }
    }

    // The integer kernels divide the raw window sum by ksize with an exact
    // multiply-and-shift; past radius 23331 (u16) no exact magic exists. The
    // plane-size checks above already rule this out at any sane resolution, but
    // reject it here rather than let getFrame silently lose precision.
    {
        const magic_ok = switch (dt) {
            inline .U8, .U16 => |t| blk: {
                const T = if (t == .U8) u8 else u16;
                const h_ok = !hblur or (boxblur_ct.magicFor(T, d.hradius) != null);
                const v_ok = !vblur or (boxblur_ct.magicFor(T, d.vradius) != null);
                break :blk h_ok and v_ok;
            },
            else => true,
        };
        if (!magic_ok) {
            map_out.setError(filter_name ++ ": radius too large to divide exactly at this bit depth.");
            zapi.freeNode(d.node);
            return;
        }
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    // CT (comptime-radius tap kernels) vs RT (running-sum) selection, measured
    // 2026-07-03 on Zen 3 (callgrind Ir at every radius + 3-rep wall-clock):
    //  - integer: RT wins at every radius (Ir -7..-19%, wall >= CT incl. the
    //    default r=1), so int always takes RT;
    //  - float: CT's O(ksize) tap kernels only beat the (scalar-h) running sum
    //    below ~r9 on this box; from r>=9 RT is 1.5-2.7x. CT therefore remains
    //    for float radius 1..8 only.
    //  - !hblur/!vblur must take RT: CT's hvBlur applies both directions
    //    unconditionally (an hpasses=0 request used to get h-blurred anyway).
    // Boundary moves change outputs slightly (CT and RT differ in pass order,
    // fixed-point rounding and edge-mirror convention: interior <=2 LSB int /
    // ~1e-5 f32, edge bands up to ~0.1% of range) — the same difference that
    // always existed across the old r22 boundary.
    const is_float = (dt == .F16) or (dt == .F32);
    const use_rt: bool = !hblur or !vblur or !is_float or
        (d.hradius != d.vradius) or (d.hradius > 8) or
        (d.hpasses > 1) or (d.vpasses > 1);
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
