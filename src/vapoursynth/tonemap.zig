const std = @import("std");
const math = std.math;

const filter = @import("../filters/tonemap.zig");
const tc = filter.tc;
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Tonemap";
/// vs-placebo's Tonemap signature, minus the Dolby Vision clip (`use_dovi`
/// is accepted, but only as 0) and `visualize_lut`; plus every other
/// pl_color_map_params / pl_peak_detect_params field that changes the
/// output (libplacebo's names and defaults): `black_cutoff`,
/// `contrast_smoothness`, `lut_size`, `mastering_clip`, `film_strength`, the
/// tone and gamut constants, `lut3d_size`, `lut3d_tricubic`,
/// `gamut_expansion` and `inverse_tone_mapping`. `log_level` is accepted
/// and ignored.
pub const params = "clip:vnode;src_csp:int:opt;dst_csp:int:opt;dst_prim:int:opt;" ++
    "src_max:float:opt;src_min:float:opt;dst_max:float:opt;dst_min:float:opt;" ++
    "dynamic_peak_detection:int:opt;smoothing_period:float:opt;" ++
    "scene_threshold_low:float:opt;scene_threshold_high:float:opt;percentile:float:opt;black_cutoff:float:opt;" ++
    "gamut_mapping:int:opt;tone_mapping_function:int:opt;tone_mapping_function_s:data:opt;tone_mapping_param:float:opt;" ++
    "metadata:int:opt;use_dovi:int:opt;show_clipping:int:opt;contrast_recovery:float:opt;contrast_smoothness:float:opt;" ++
    "log_level:int:opt;" ++
    "lut_size:int:opt;mastering_clip:float:opt;film_strength:float:opt;inverse_tone_mapping:int:opt;" ++
    "knee_adaptation:float:opt;knee_minimum:float:opt;knee_maximum:float:opt;knee_default:float:opt;knee_offset:float:opt;" ++
    "slope_tuning:float:opt;slope_offset:float:opt;spline_contrast:float:opt;reinhard_contrast:float:opt;" ++
    "linear_knee:float:opt;exposure:float:opt;" ++
    "gamut_expansion:int:opt;lut3d_size:int[]:opt;lut3d_tricubic:int:opt;" ++
    "perceptual_deadzone:float:opt;perceptual_strength:float:opt;colorimetric_gamma:float:opt;" ++
    "softclip_knee:float:opt;softclip_desat:float:opt;";

/// Whether the tone constant that `tone_mapping_param` sets for `f`
/// (pl_tone_map_params_infer) was passed explicitly.
fn paramConstantGiven(map_in: anytype, f: tc.ToneFn) bool {
    return switch (f) {
        .st2094_40, .st2094_10 => map_in.numElements("knee_adaptation") != null,
        .bt2390 => map_in.numElements("knee_offset") != null,
        .spline => map_in.numElements("spline_contrast") != null,
        .reinhard => map_in.numElements("reinhard_contrast") != null,
        .mobius, .gamma => map_in.numElements("linear_knee") != null,
        .linear, .linear_light => map_in.numElements("exposure") != null,
        .clip, .bt2446a, .hable => false,
    };
}

const max_window = 300;

const Data = struct {
    node: ?*vs.Node = null,
    vi: vs.VideoInfo = .{},
    cfg: filter.Config = undefined,
    window: u32 = 0,
    peaks: ?filter.PeakCache = null,
    gamut: filter.GamutCache = .{},
};

/// A frame property as f32, stored as float or int (sources differ).
fn propNum(props: anytype, comptime key: [:0]const u8) f32 {
    if (props.getFloat(f64, key)) |v| return if (math.isFinite(v)) @floatCast(v) else 0;
    if (props.getInt(i64, key)) |v| return @floatFromInt(v);
    return 0;
}

fn propTriple(props: anytype, comptime key: [:0]const u8) ?[3]f32 {
    const arr = props.getFloatArray(key) orelse return null;
    if (arr.len != 3) return null;
    var r: [3]f32 = undefined;
    for (&r, arr) |*o, v| o.* = if (math.isFinite(v)) @floatCast(v) else 0;
    return r;
}

fn readMeta(frame: anytype) filter.FrameMeta {
    const props = frame.getPropertiesRO();
    var m: filter.FrameMeta = .{};
    m.full_range = if (props.getRange()) |r| r == .FULL else if (props.getColorRange()) |r| r == .FULL else null;
    m.chroma_loc = props.getInt(i64, "_ChromaLocation");
    m.max_cll = propNum(props, "ContentLightLevelMax");
    m.max_fall = propNum(props, "ContentLightLevelAverage");
    m.mastering_max = propNum(props, "MasteringDisplayMaxLuminance");
    m.mastering_min = propNum(props, "MasteringDisplayMinLuminance");
    m.scene_avg = propNum(props, "PLSceneAvg");
    if (props.getFloatArray("PLSceneMax")) |arr| {
        if (arr.len == 1 or arr.len == 3) {
            m.scene_max_len = @intCast(arr.len);
            for (arr, 0..) |v, i| m.scene_max[i] = if (math.isFinite(v)) @floatCast(v) else 0;
        }
    }
    m.prim_x = propTriple(props, "MasteringDisplayPrimariesX");
    m.prim_y = propTriple(props, "MasteringDisplayPrimariesY");
    m.white = .{ .x = propNum(props, "MasteringDisplayWhitePointX"), .y = propNum(props, "MasteringDisplayWhitePointY") };
    return m;
}

fn planesIn(comptime T: type, f: anytype) filter.Planes(T) {
    return .{
        .ptr = .{ f.getReadSlice2(T, 0).ptr, f.getReadSlice2(T, 1).ptr, f.getReadSlice2(T, 2).ptr },
        .stride = .{ f.getStride2(T, 0), f.getStride2(T, 1), f.getStride2(T, 2) },
    };
}

fn vsPrimaries(p: tc.Prim) i64 {
    return switch (p) {
        .bt601_525 => 6,
        .bt601_625 => 5,
        .unknown, .bt709 => 1,
        .bt470m => 4,
        .ebu3213 => 22,
        .bt2020 => 9,
        .cie1931 => 10,
        .dci_p3 => 11,
        .display_p3 => 12,
        .film_c => 8,
        else => 2,
    };
}

fn Tonemap(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            const lo: c_int = if (d.window > 1) @max(0, n - @as(c_int, @intCast(d.window)) + 1) else n;

            if (activation_reason == .Initial) {
                // Earlier window frames are fetched only until measured; a
                // frame not ready now was not ready then, so it was requested.
                var i = lo;
                while (i < n) : (i += 1) {
                    if (!d.peaks.?.ready(@intCast(i))) zapi.requestFrameFilter(i, d.node);
                }
                zapi.requestFrameFilter(n, d.node);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            const meta = readMeta(src);
            const setup: filter.Setup = .init(&d.cfg, &meta);

            var detected: ?tc.PeakState = null;
            if (setup.peak_active) {
                var entries: [max_window]filter.PeakCache.Entry = undefined;
                const count: usize = @intCast(n - lo + 1);
                // own frame first: earlier ones are likely measured already
                var k = count;
                while (k > 0) {
                    k -= 1;
                    const i: c_int = lo + @as(c_int, @intCast(k));
                    const ui: usize = @intCast(i);
                    entries[k] = d.peaks.?.acquire(ui) orelse blk: {
                        const e = (if (i == n) measure(d, &setup, src) else measureOther(d, &zapi, i)) catch {
                            d.peaks.?.release(ui);
                            zapi.setFilterError(filter_name ++ ": out of memory");
                            return null;
                        };
                        d.peaks.?.put(ui, e);
                        break :blk e;
                    };
                }
                detected = filter.PeakCache.replay(entries[0..count], &d.cfg.peak);
            }

            const plan = allocator.create(filter.Plan) catch {
                zapi.setFilterError(filter_name ++ ": out of memory");
                return null;
            };
            defer allocator.destroy(plan);
            plan.init(&d.cfg, &setup, detected, &d.gamut) catch {
                zapi.setFilterError(filter_name ++ ": out of memory");
                return null;
            };

            const dst = src.newVideoFrame3(.{ .format = &d.vi.format });
            const out: filter.PlanesOut(T) = .{
                .ptr = .{ dst.getWriteSlice2(T, 0).ptr, dst.getWriteSlice2(T, 1).ptr, dst.getWriteSlice2(T, 2).ptr },
                .stride = .{ dst.getStride2(T, 0), dst.getStride2(T, 1), dst.getStride2(T, 2) },
            };
            filter.process(T, T, &d.cfg, plan, planesIn(T, src), out) catch return hz.frameOom(&zapi, filter_name, dst);

            const props = dst.getPropertiesRW();
            const dst_trc = setup.dst.trc;
            props.setInt("_Matrix", if (d.cfg.rgb) 0 else if (dst_trc == .bt1886) 1 else 9, .Replace);
            props.setInt("_Transfer", switch (dst_trc) {
                .bt1886 => 1,
                .pq => 16,
                .hlg => 18,
            }, .Replace);
            props.setInt("_Primaries", vsPrimaries(setup.dst.prim), .Replace);
            props.setRange(if (d.cfg.rgb or d.cfg.float or setup.full) .FULL else .LIMITED);
            props.deleteKey("_ChromaLocation");
            return dst.frame;
        }

        fn measure(d: *Data, s: *const filter.Setup, f: anytype) !filter.PeakCache.Entry {
            const m = try filter.measure(T, &d.cfg, s, planesIn(T, f));
            return .{ .m = m, .active = s.peak_active };
        }

        fn measureOther(d: *Data, zapi: *const ZAPI, i: c_int) !filter.PeakCache.Entry {
            const f = zapi.initZFrame(d.node, i);
            defer f.deinit();
            const meta = readMeta(f);
            const s: filter.Setup = .init(&d.cfg, &meta);
            return measure(d, &s, f);
        }
    };
}

fn tonemapFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    if (d.peaks) |*p| p.deinit();
    d.gamut.deinit();
    zapi.freeNode(d.node);
    allocator.destroy(d);
}

fn fail(map_out: anytype, zapi: *const ZAPI, node: ?*vs.Node, msg: [:0]const u8) void {
    map_out.setError(msg);
    zapi.freeNode(node);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    var print_buf: [512]u8 = @splat(0);
    const map_out = zapi.initZMap2(out, &print_buf);
    d.node, const vi = map_in.getNodeVi("clip").?;
    const fmt = vi.format;

    if (!vapoursynth.vshelper.isConstantVideoFormat(vi)) return fail(map_out, &zapi, d.node, filter_name ++ ": only constant format input supported.");
    const int16 = fmt.sampleType == .Integer and fmt.bitsPerSample == 16;
    const f32in = fmt.sampleType == .Float and fmt.bitsPerSample == 32;
    const family_ok = fmt.colorFamily == .RGB or (fmt.colorFamily == .YUV and fmt.subSamplingW <= 1 and fmt.subSamplingH <= 1);
    if (!(int16 or f32in) or !family_ok) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": input must be 16-bit integer or 32-bit float RGB or YUV (4:4:4, 4:2:2, 4:2:0, 4:4:0).");
    }
    if (vi.width < filter.min_size or vi.height < filter.min_size) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": input must be at least 64x64.");
    }

    const mp = hz.Maps.init(&map_in, &map_out, filter_name);
    const src_csp = mp.getValue(i64, "src_csp", 1, 0, 3) catch return zapi.freeNode(d.node);
    const dst_csp = mp.getValue(i64, "dst_csp", 0, 0, 3) catch return zapi.freeNode(d.node);
    if (src_csp == 3 or dst_csp == 3 or (map_in.getValue(i64, "use_dovi") orelse 0) != 0) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": Dolby Vision is not supported.");
    }
    if (src_csp == 0) return fail(map_out, &zapi, d.node, filter_name ++ ": src_csp must be 1 (HDR10) or 2 (HLG).");

    const tone_index = mp.getValue(i64, "tone_mapping_function", @intFromEnum(tc.ToneFn.spline), 0, tc.ToneFn.names.len - 1) catch return zapi.freeNode(d.node);
    var tone_fn: tc.ToneFn = @enumFromInt(tone_index);
    if (map_in.getData("tone_mapping_function_s", 0)) |name| {
        tone_fn = tc.ToneFn.fromName(name) orelse {
            hz.setArgError(map_out, "{s}: unknown tone_mapping_function_s '{s}'.", .{ filter_name, name });
            return zapi.freeNode(d.node);
        };
    }

    const fmax = 10000.0;
    const dst_prim: ?tc.Prim = if (map_in.getValue(i64, "dst_prim")) |_| blk: {
        const v = mp.getValue(i64, "dst_prim", 0, 0, tc.Prim.count - 1) catch return zapi.freeNode(d.node);
        break :blk @enumFromInt(v);
    } else null;

    // vs-placebo always passes tone_mapping_param (param_def when absent),
    // which overrides the function's tone constant; keep that default, but
    // not over an explicitly passed constant
    const param_def: f64 = if (paramConstantGiven(map_in, tone_fn)) 0 else tone_fn.paramDef();

    // lut3d_size: (I, C, h); 0 keeps that axis' default, like PL_DEF
    var lut3d: [3]u32 = .{ 48, 32, 256 };
    if (map_in.numElements("lut3d_size")) |len| {
        const a = map_in.getIntArray("lut3d_size") orelse &[_]i64{};
        if (len != 3 or a.len != 3) return fail(map_out, &zapi, d.node, filter_name ++ ": lut3d_size must have 3 values (I, C, h).");
        for (a, &lut3d) |v, *o| {
            if (v == 0) continue;
            if (v < 2 or v > 1024) return fail(map_out, &zapi, d.node, filter_name ++ ": lut3d_size values must be 0 (default) or 2..1024.");
            o.* = @intCast(v);
        }
        // 16 bytes per entry: cap the LUT at 256 MB
        if (@as(u64, lut3d[0]) * lut3d[1] * lut3d[2] > 1 << 24) {
            return fail(map_out, &zapi, d.node, filter_name ++ ": lut3d_size is too large (I * C * h must be at most 16777216).");
        }
    }

    d.cfg = .{
        .src_csp = @enumFromInt(src_csp),
        .dst_csp = @enumFromInt(dst_csp),
        .dst_prim = dst_prim,
        .src_max = @floatCast(mp.getValue(f64, "src_max", 0, 0, fmax) catch return zapi.freeNode(d.node)),
        .src_min = @floatCast(mp.getValue(f64, "src_min", 0, 0, fmax) catch return zapi.freeNode(d.node)),
        .dst_max = @floatCast(mp.getValue(f64, "dst_max", 0, 0, fmax) catch return zapi.freeNode(d.node)),
        .dst_min = @floatCast(mp.getValue(f64, "dst_min", 0, 0, fmax) catch return zapi.freeNode(d.node)),
        .peak_detect = (mp.getValue(i64, "dynamic_peak_detection", 1, 0, 1) catch return zapi.freeNode(d.node)) != 0,
        .peak = .{
            .smoothing_period = @floatCast(mp.getValue(f64, "smoothing_period", 20, 0, 1000) catch return zapi.freeNode(d.node)),
            .scene_threshold_low = @floatCast(mp.getValue(f64, "scene_threshold_low", 1, 0, 100) catch return zapi.freeNode(d.node)),
            .scene_threshold_high = @floatCast(mp.getValue(f64, "scene_threshold_high", 3, 0, 100) catch return zapi.freeNode(d.node)),
            .percentile = @floatCast(mp.getValue(f64, "percentile", 100, 0, 100) catch return zapi.freeNode(d.node)),
            .black_cutoff = @floatCast(mp.getValue(f64, "black_cutoff", 1, 0, 100) catch return zapi.freeNode(d.node)),
        },
        .gamut_fn = @enumFromInt(mp.getValue(i64, "gamut_mapping", @intFromEnum(tc.GamutFn.perceptual), 0, tc.GamutFn.count - 1) catch return zapi.freeNode(d.node)),
        .tone_fn = tone_fn,
        .tone_param = @floatCast(mp.getValue(f64, "tone_mapping_param", param_def, 0, 10) catch return zapi.freeNode(d.node)),
        .metadata = @enumFromInt(mp.getValue(i64, "metadata", 0, 0, 4) catch return zapi.freeNode(d.node)),
        .show_clipping = (mp.getValue(i64, "show_clipping", 0, 0, 1) catch return zapi.freeNode(d.node)) != 0,
        .contrast_recovery = @floatCast(mp.getValue(f64, "contrast_recovery", 0, 0, 10) catch return zapi.freeNode(d.node)),
        .contrast_smoothness = @floatCast(mp.getValue(f64, "contrast_smoothness", 3.5, 1, 32) catch return zapi.freeNode(d.node)),
        .lut_size = @intCast(mp.getValue(i64, "lut_size", 256, 2, 4096) catch return zapi.freeNode(d.node)),
        .mastering_clip = @floatCast(mp.getValue(f64, "mastering_clip", 1, 0, 1) catch return zapi.freeNode(d.node)),
        .film_strength = @floatCast(mp.getValue(f64, "film_strength", 0.3, 0, 2) catch return zapi.freeNode(d.node)),
        .inverse_tone_mapping = (mp.getValue(i64, "inverse_tone_mapping", 0, 0, 1) catch return zapi.freeNode(d.node)) != 0,
        // ranges as documented on pl_tone_map_constants / pl_gamut_map_constants
        .tone_constants = .{
            .knee_adaptation = @floatCast(mp.getValue(f64, "knee_adaptation", 0.4, 0, 1) catch return zapi.freeNode(d.node)),
            .knee_minimum = @floatCast(mp.getValue(f64, "knee_minimum", 0.1, 0, 0.5) catch return zapi.freeNode(d.node)),
            .knee_maximum = @floatCast(mp.getValue(f64, "knee_maximum", 0.8, 0.5, 1) catch return zapi.freeNode(d.node)),
            .knee_default = @floatCast(mp.getValue(f64, "knee_default", 0.4, 0, 1) catch return zapi.freeNode(d.node)),
            .knee_offset = @floatCast(mp.getValue(f64, "knee_offset", 1, 0.5, 2) catch return zapi.freeNode(d.node)),
            .slope_tuning = @floatCast(mp.getValue(f64, "slope_tuning", 1.5, 0, 10) catch return zapi.freeNode(d.node)),
            .slope_offset = @floatCast(mp.getValue(f64, "slope_offset", 0.2, 0, 1) catch return zapi.freeNode(d.node)),
            .spline_contrast = @floatCast(mp.getValue(f64, "spline_contrast", 0.5, 0, 1.5) catch return zapi.freeNode(d.node)),
            .reinhard_contrast = @floatCast(mp.getValue(f64, "reinhard_contrast", 0.5, 0, 1) catch return zapi.freeNode(d.node)),
            .linear_knee = @floatCast(mp.getValue(f64, "linear_knee", 0.3, 0, 1) catch return zapi.freeNode(d.node)),
            .exposure = @floatCast(mp.getValue(f64, "exposure", 1, 0, 10) catch return zapi.freeNode(d.node)),
        },
        .gamut_expansion = (mp.getValue(i64, "gamut_expansion", 0, 0, 1) catch return zapi.freeNode(d.node)) != 0,
        .lut3d_size = lut3d,
        .lut3d_tricubic = (mp.getValue(i64, "lut3d_tricubic", 0, 0, 1) catch return zapi.freeNode(d.node)) != 0,
        .gamut_constants = .{
            .perceptual_deadzone = @floatCast(mp.getValue(f64, "perceptual_deadzone", 0.3, 0, 1) catch return zapi.freeNode(d.node)),
            .perceptual_strength = @floatCast(mp.getValue(f64, "perceptual_strength", 0.8, 0, 1) catch return zapi.freeNode(d.node)),
            .colorimetric_gamma = @floatCast(mp.getValue(f64, "colorimetric_gamma", 1.8, 0, 10) catch return zapi.freeNode(d.node)),
            .softclip_knee = @floatCast(mp.getValue(f64, "softclip_knee", 0.7, 0, 1) catch return zapi.freeNode(d.node)),
            .softclip_desat = @floatCast(mp.getValue(f64, "softclip_desat", 0.35, 0, 1) catch return zapi.freeNode(d.node)),
        },
        .width = @intCast(vi.width),
        .height = @intCast(vi.height),
        .rgb = fmt.colorFamily == .RGB,
        .float = f32in,
        .ssw = @intCast(fmt.subSamplingW),
        .ssh = @intCast(fmt.subSamplingH),
    };
    d.window = @min(d.cfg.peakWindow(), max_window);

    d.vi = vi.*;
    _ = zapi.getVideoFormatByID(&d.vi.format, switch (fmt.colorFamily) {
        .RGB => if (f32in) .RGBS else .RGB48,
        else => if (f32in) .YUV444PS else .YUV444P16,
    });

    if (d.cfg.peak_detect) {
        d.peaks = filter.PeakCache.init(@intCast(vi.numFrames)) catch return fail(map_out, &zapi, d.node, filter_name ++ ": out of memory");
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = if (d.window > 1) .General else .StrictSpatial },
    };
    const gf: vs.FilterGetFrame = if (f32in) &Tonemap(f32).getFrame else &Tonemap(u16).getFrame;
    zapi.createVideoFilter(out, filter_name, &d.vi, gf, tonemapFree, .Parallel, &deps, data);
}
