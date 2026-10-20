const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

const filter_int = @import("../filters/deband_int.zig");
const filter_float = @import("../filters/deband_float.zig");
const setup = @import("../filters/deband_setup.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;
const Maps = hz.Maps;

const allocator = std.heap.c_allocator;
pub const filter_name = "Deband";

const Mode = setup.Mode;

const Data = struct {
    node: *vs.Node = undefined,
    vi: *const vs.VideoInfo = undefined,
    k: setup.Data = .{},
};

/// Parse the arguments with vszip's messages (validation order matters).
fn parseInputs(m: *const Maps) !setup.Data.Inputs {
    const thr_in: [3]f64 = try m.getArray("thr", 3, .{ 0.99, 0.99, 0.99 }, 0, 255);
    const thr1_in: [3]f64 = try m.getArray("thr1", 3, thr_in, 0, 255);
    const thr2_in: [3]f64 = try m.getArray("thr2", 3, thr_in, 0, 255);
    const grain_in: [3]f64 = try m.getArray("grain", 2, .{ 0, 0, 0 }, 0, 127);
    const sample_mode = try m.getValue(i32, "sample_mode", 2, 1, 7);
    const range = try m.getValue(i32, "range", 15, 0, 255);
    const seed = try m.getValue(i32, "seed", 0, math.minInt(i32), math.maxInt(i32));
    const blur_first = m.in.getBool("blur_first") orelse true;
    const dynamic_grain = m.in.getBool("dynamic_grain") orelse false;
    const keep_tv_range = m.in.getBool("keep_tv_range") orelse false;
    const angle_boost = try m.getValue(f32, "angle_boost", 1.5, 0, math.maxInt(u16));
    const max_angle = try m.getValue(f32, "max_angle", 0.15, 0, 1);
    const random_param_ref = try m.getValue(f64, "random_param_ref", 1.0, 0, 255);
    const random_param_grain = try m.getValue(f64, "random_param_grain", 1.0, 0, 255);
    const random_algo_ref = try m.getValue(i32, "random_algo_ref", 1, 0, 2);
    const random_algo_grain = try m.getValue(i32, "random_algo_grain", 1, 0, 2);
    return .{
        .thr = thr_in,
        .thr1 = thr1_in,
        .thr2 = thr2_in,
        .grain = grain_in,
        .sample_mode = sample_mode,
        .range = range,
        .seed = seed,
        .blur_first = blur_first,
        .dynamic_grain = dynamic_grain,
        .keep_tv_range = keep_tv_range,
        .angle_boost = angle_boost,
        .max_angle = max_angle,
        .random_param_ref = random_param_ref,
        .random_param_grain = random_param_grain,
        .random_algo_ref = random_algo_ref,
        .random_algo_grain = random_algo_grain,
    };
}

fn F3KDBInt(comptime mode: Mode, comptime blur_first: bool, comptime add_grain: [3]bool, np: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const k = &d.k;
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame();

                inline for (0..np) |plane| {
                    const src_slice = src.getReadSlice2(u16, plane);
                    const dst_slice = dst.getWriteSlice2(u16, plane);
                    const stride = src.getStride(plane) >> 1;
                    const width = src.getWidth(plane);
                    const height = src.getHeight(plane);
                    var grain = if (add_grain[plane]) k.tb.grain_int[plane];
                    if (add_grain[plane] and k.dynamic_grain) {
                        const offset = k.tb.grain_offsets[@intCast(n)];
                        grain = grain[offset..];
                    }

                    filter_int.processPlane(
                        src_slice,
                        dst_slice,
                        k.tb.ref1[plane],
                        k.tb.ref2[plane],
                        grain,
                        stride,
                        width,
                        height,
                        k.thr.u[plane],
                        k.thr1.u[plane],
                        k.thr2.u[plane],
                        k.pixel_min[plane],
                        k.pixel_max[plane],
                        k.angle_boost,
                        k.max_angle,
                        &k.tb.angles,
                        mode,
                        blur_first,
                        add_grain[plane],
                    );
                }

                return dst.frame;
            }
            return null;
        }
    };
}

fn F3KDBFloat(comptime mode: Mode, comptime blur_first: bool, comptime add_grain: [3]bool, np: comptime_int) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const k = &d.k;
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame();

                inline for (0..np) |plane| {
                    const src_slice = src.getReadSlice2(f32, plane);
                    const dst_slice = dst.getWriteSlice2(f32, plane);
                    const stride = src.getStride(plane) >> 2;
                    const width = src.getWidth(plane);
                    const height = src.getHeight(plane);
                    var grain = if (add_grain[plane]) k.tb.grain_float[plane];
                    if (add_grain[plane] and k.dynamic_grain) {
                        const offset = k.tb.grain_offsets[@intCast(n)];
                        grain = grain[offset..];
                    }

                    filter_float.processPlane(
                        src_slice,
                        dst_slice,
                        k.tb.ref1[plane],
                        k.tb.ref2[plane],
                        grain,
                        stride,
                        width,
                        height,
                        k.thr.f[plane],
                        k.thr1.f[plane],
                        k.thr2.f[plane],
                        k.angle_boost,
                        k.max_angle,
                        &k.tb.angles,
                        k.pixel_minf[plane],
                        k.pixel_maxf[plane],
                        mode,
                        blur_first,
                        add_grain[plane],
                    );
                }

                return dst.frame;
            }
            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.k.tb.deinit(&d.k);
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    var print_buf: [512]u8 = @splat(0);

    const zapi = ZAPI.init(vsapi, core, null);
    const zin = zapi.initZMap(in);
    const zout = zapi.initZMap2(out, &print_buf);
    const maps = Maps.init(&zin, &zout, filter_name);

    d.node, const vi_in = zin.getNodeVi("clip").?;
    if (!vsh.isConstantVideoFormat(vi_in)) {
        zout.setError2("{s}: clip must have constant format [{}x{}-{}]", .{ filter_name, vi_in.width, vi_in.height, vi_in.format.colorFamily });
        zapi.freeNode(d.node);
        return;
    }

    if (vi_in.format.sampleType == .Float and vi_in.format.bitsPerSample != 32) {
        zout.setError2("{s}: only 32-bit format is supported when float clip", .{filter_name});
        zapi.freeNode(d.node);
        return;
    }

    if (vi_in.format.bitsPerSample < 16) {
        d.node = hz.bitDepth(16, d.node, .none, &zapi);
    }

    d.vi = zapi.getVideoInfo(d.node);
    const inputs = parseInputs(&maps) catch {
        zapi.freeNode(d.node);
        return;
    };

    // The ref tables address frames with VapourSynth's (CPU-dependent) frame
    // strides, so they are built for exactly those.
    d.k = .{
        .width = d.vi.width,
        .height = d.vi.height,
        .num_frames = d.vi.numFrames,
        .is_int = d.vi.format.sampleType == .Integer,
        .is_yuv = d.vi.format.colorFamily == .YUV,
        .num_planes = @intCast(d.vi.format.numPlanes),
        .strides = hz.strideFromVi(d.vi),
        .ssw = @intCast(d.vi.format.subSamplingW),
        .ssh = @intCast(d.vi.format.subSamplingH),
    };
    d.k.configure(inputs);

    d.k.tb = setup.TempBuff.init(&d.k) catch {
        zout.setError2("{s}: failed to initialize temp buffer (out of memory)", .{filter_name});
        zapi.freeNode(d.node);
        return;
    };

    const data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = comptimeGetFrame(
        d.k.sample_mode,
        d.k.blur_first,
        d.k.add_grain,
        d.vi.format.numPlanes,
        d.vi.format.sampleType,
    );

    var out_node = zapi.createVideoFilter2(filter_name, d.vi, gf, free, .Parallel, &deps, data).?;
    if (vi_in.format.bitsPerSample < 16) {
        const bits: u32 = @intCast(vi_in.format.bitsPerSample);
        out_node = hz.bitDepth(bits, out_node, .error_diffusion, &zapi);
    }

    _ = zout.consumeNode("clip", out_node, .Replace);
}

fn comptimeGetFrame(mode: Mode, blur_first: bool, add_grain: [3]bool, num_planes: i32, st: vs.SampleType) vs.FilterGetFrame {
    const comptime_grain: [4][3]bool = .{
        .{ true, false, false },
        .{ false, true, true },
        .{ true, true, true },
        .{ false, false, false },
    };

    var i: u32 = 0;
    const g_idx: u32 = while (i < comptime_grain.len) : (i += 1) {
        if (std.mem.eql(bool, &comptime_grain[i], &add_grain)) break i;
    } else 0;

    const get_frame: vs.FilterGetFrame = switch (num_planes) {
        inline 1...3 => |np| switch (mode) {
            inline else => |m| switch (g_idx) {
                inline 0...3 => |gi| switch (blur_first) {
                    inline false, true => |bf| switch (st) {
                        .Integer => F3KDBInt(m, bf, comptime_grain[gi], np).getFrame,
                        .Float => F3KDBFloat(m, bf, comptime_grain[gi], np).getFrame,
                    },
                },
                else => unreachable,
            },
        },
        else => unreachable,
    };

    return get_frame;
}
