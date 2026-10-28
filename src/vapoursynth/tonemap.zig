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
/// vs-placebo's Tonemap signature, minus `visualize_lut` and `log_level`
/// (libplacebo's debug overlay and logging); plus every other
/// pl_color_map_params / pl_peak_detect_params field that changes the
/// output (libplacebo's names and defaults): `black_cutoff`,
/// `contrast_smoothness`, `lut_size`, `mastering_clip`, `film_strength`, the
/// tone and gamut constants, `lut3d_size`, `lut3d_tricubic`,
/// `gamut_expansion` and `inverse_tone_mapping`. vszip's own, last:
/// `cache_frame` (speed/memory only, the output is the same).
pub const params = "clip:vnode;src_csp:int:opt;dst_csp:int:opt;dst_prim:int:opt;" ++
    "src_max:float:opt;src_min:float:opt;dst_max:float:opt;dst_min:float:opt;" ++
    "dynamic_peak_detection:int:opt;smoothing_period:float:opt;" ++
    "scene_threshold_low:float:opt;scene_threshold_high:float:opt;percentile:float:opt;black_cutoff:float:opt;" ++
    "gamut_mapping:int:opt;tone_mapping_function:int:opt;tone_mapping_function_s:data:opt;tone_mapping_param:float:opt;" ++
    "metadata:int:opt;use_dovi:int:opt;dovi_el:vnode:opt;show_clipping:int:opt;contrast_recovery:float:opt;contrast_smoothness:float:opt;" ++
    "lut_size:int:opt;mastering_clip:float:opt;film_strength:float:opt;inverse_tone_mapping:int:opt;" ++
    "knee_adaptation:float:opt;knee_minimum:float:opt;knee_maximum:float:opt;knee_default:float:opt;knee_offset:float:opt;" ++
    "slope_tuning:float:opt;slope_offset:float:opt;spline_contrast:float:opt;reinhard_contrast:float:opt;" ++
    "linear_knee:float:opt;exposure:float:opt;" ++
    "gamut_expansion:int:opt;lut3d_size:int[]:opt;lut3d_tricubic:int:opt;" ++
    "perceptual_deadzone:float:opt;perceptual_strength:float:opt;colorimetric_gamma:float:opt;" ++
    "softclip_knee:float:opt;softclip_desat:float:opt;cache_frame:int:opt;";

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

const Data = struct {
    node: ?*vs.Node = null,
    /// Dolby Vision FEL clip (dovi_el with use_dovi): composed with
    /// src_csp=3, and the RPU source, as in vs-placebo
    el_node: ?*vs.Node = null,
    vi: vs.VideoInfo = .{},
    cfg: filter.Config = undefined,
    window: u32 = 0,
    peaks: ?filter.PeakCache = null,
    gamut: filter.GamutCache = .{},
    /// cache_frame: linear frames kept from detection (or contrast
    /// recovery's feature pass) for the output pass
    lin: ?filter.LinearPool = null,
};

/// A float property value as f32; non-finite values, and finite ones that
/// would overflow f32 to inf, read as absent (0).
fn propF32(v: f64) f32 {
    return if (math.isFinite(v) and @abs(v) <= math.floatMax(f32)) @floatCast(v) else 0;
}

/// A frame property as f32, stored as float or int (sources differ).
fn propNum(props: anytype, comptime key: [:0]const u8) f32 {
    if (props.getFloat(f64, key)) |v| return propF32(v);
    if (props.getInt(i64, key)) |v| return @floatFromInt(v);
    return 0;
}

fn propTriple(props: anytype, comptime key: [:0]const u8) ?[3]f32 {
    const arr = props.getFloatArray(key) orelse return null;
    if (arr.len != 3) return null;
    var r: [3]f32 = undefined;
    for (&r, arr) |*o, v| o.* = propF32(v);
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
            for (arr, 0..) |v, i| m.scene_max[i] = propF32(v);
        }
    }
    m.prim_x = propTriple(props, "MasteringDisplayPrimariesX");
    m.prim_y = propTriple(props, "MasteringDisplayPrimariesY");
    m.white = .{ .x = propNum(props, "MasteringDisplayWhitePointX"), .y = propNum(props, "MasteringDisplayWhitePointY") };
    return m;
}

const dovi = filter.dovi;
const Frame = ZAPI.ZFrame(*const vs.Frame);

const missing_rpu = filter_name ++ ": Clips are missing `DolbyVisionRPU` prop for Dolby Vision mapping!";
const invalid_rpu = filter_name ++ ": invalid `DolbyVisionRPU` prop (failed parsing the RPU).";

/// The frame's DolbyVisionRPU, parsed; null when absent.
fn readRpu(frame: anytype) dovi.Error!?dovi.Rpu {
    const raw = frame.getPropertiesRO().getData("DolbyVisionRPU", 0) orelse return null;
    return try dovi.parse(raw);
}

/// How far back a frame's RPU may defer to earlier ones (use_prev_vdr_rpu_flag,
/// missing or compressed DM data) before the chain counts as broken.
const max_rpu_chain = 300;

const Load = union(enum) {
    ok,
    /// an earlier frame the RPU chain needs, not available yet
    need: c_int,
    err: [:0]const u8,
};

/// Frame i's metadata into `m`. With an RPU, its DM data feeds the HDR
/// metadata whatever src_csp is (as in vs-placebo). With src_csp dovi, the
/// reshaping state is resolved into `dv`: vs-placebo keeps it in a static
/// that every call updates in request order, so an RPU that signals only
/// part of it continues the previous frame's. Here frame i folds the RPUs
/// from the last complete one at or before it, in frame order, which is
/// what vs-placebo computes on sequential requests. With an EL clip the
/// RPUs are read from it (vs-placebo reads them from the EL's props).
fn loadMeta(d: *const Data, zapi: *const ZAPI, frame: anytype, i: c_int, m: *filter.FrameMeta, dv: *dovi.Meta) Load {
    const dovi_src = d.cfg.src_csp == .dovi;
    const el_present = d.el_node != null;
    m.* = readMeta(frame);
    const rpu_read = if (d.el_node) |el| blk: {
        // requested together with the BL frame
        const f: Frame = .init(zapi, zapi.getFrameFilter(i, el) orelse return .{ .need = i });
        defer f.deinit();
        break :blk readRpu(f);
    } else readRpu(frame);
    const rpu = (rpu_read catch return if (dovi_src) .{ .err = invalid_rpu } else .ok) orelse
        return if (dovi_src) .{ .err = missing_rpu } else .ok;
    m.dovi_dm = rpu.dm;
    m.dovi_profile = rpu.profile();
    if (!dovi_src) return .ok;

    const rpu_node = rpuNode(d);
    var base = i;
    if (!rpu.complete()) {
        while (base > 0) {
            if (i - base >= max_rpu_chain) return .{ .err = filter_name ++ ": no complete DolbyVisionRPU within 300 frames." };
            base -= 1;
            // null when not requested yet: request it and come back
            const f: Frame = .init(zapi, zapi.getFrameFilter(base, rpu_node) orelse return .{ .need = base });
            defer f.deinit();
            const r = (readRpu(f) catch return .{ .err = invalid_rpu }) orelse return .{ .err = missing_rpu };
            if (r.complete()) break;
        }
    }
    dv.* = .{};
    var j = base;
    while (j < i) : (j += 1) {
        // fetched and parsed by the walk above
        const f: Frame = .init(zapi, zapi.getFrameFilter(j, rpu_node).?);
        defer f.deinit();
        const r = (readRpu(f) catch unreachable).?;
        dv.update(&r, el_present);
    }
    dv.update(&rpu, el_present);
    m.dovi_meta = dv;
    return .ok;
}

/// The clip the DolbyVisionRPU props are read from.
fn rpuNode(d: *const Data) ?*vs.Node {
    return d.el_node orelse d.node;
}

/// The FEL frame i's planes (requested with the BL frame); the caller frees
/// the frame after processing.
fn elFrame(d: *const Data, zapi: *const ZAPI, i: c_int) ?Frame {
    const el = d.el_node orelse return null;
    return .init(zapi, zapi.getFrameFilter(i, el).?);
}

fn planesIn(comptime T: type, f: anytype, el: ?Frame) filter.Planes(T) {
    return .{
        .ptr = .{ f.getReadSlice2(T, 0).ptr, f.getReadSlice2(T, 1).ptr, f.getReadSlice2(T, 2).ptr },
        .stride = .{ f.getStride2(T, 0), f.getStride2(T, 1), f.getStride2(T, 2) },
        .el = if (el) |e| .{
            .ptr = .{ e.getReadSlice2(u16, 0).ptr, e.getReadSlice2(u16, 1).ptr, e.getReadSlice2(u16, 2).ptr },
            .stride = .{ e.getStride2(u16, 0), e.getStride2(u16, 1), e.getStride2(u16, 2) },
        } else null,
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
                    if (!d.peaks.?.ready(@intCast(i))) {
                        zapi.requestFrameFilter(i, d.node);
                        if (d.el_node) |el| zapi.requestFrameFilter(i, el);
                    }
                }
                zapi.requestFrameFilter(n, d.node);
                if (d.el_node) |el| zapi.requestFrameFilter(n, el);
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const src = zapi.initZFrame(d.node, n);
            defer src.deinit();
            var meta: filter.FrameMeta = undefined;
            var dv: dovi.Meta = undefined;
            // Frames whose RPU continues an earlier one need that frame
            // too: request what is missing and get called again (all
            // requested frames stay available until this request ends).
            // The window frames this request may measure are resolved
            // first, before any PeakCache claim.
            var requested: [filter.Config.max_window + 1]c_int = undefined;
            var nreq: usize = 0;
            const need = struct {
                fn f(z: *const ZAPI, node: ?*vs.Node, list: []c_int, len: *usize, k: c_int) void {
                    if (std.mem.indexOfScalar(c_int, list[0..len.*], k) != null) return;
                    z.requestFrameFilter(k, node);
                    if (len.* < list.len) {
                        list[len.*] = k;
                        len.* += 1;
                    }
                }
            }.f;
            switch (loadMeta(d, &zapi, src, n, &meta, &dv)) {
                .ok => {},
                .need => |k| need(&zapi, rpuNode(d), &requested, &nreq, k),
                .err => |msg| {
                    zapi.setFilterError(msg);
                    return null;
                },
            }
            if (d.cfg.src_csp == .dovi) if (d.peaks) |*peaks| {
                var i = lo;
                while (i < n) : (i += 1) {
                    // not measured now: not measured at .Initial either,
                    // so it was requested
                    if (peaks.ready(@intCast(i))) continue;
                    const f = zapi.initZFrame(d.node, i);
                    defer f.deinit();
                    var m2: filter.FrameMeta = undefined;
                    var dv2: dovi.Meta = undefined;
                    switch (loadMeta(d, &zapi, f, i, &m2, &dv2)) {
                        .ok => {},
                        .need => |k| need(&zapi, rpuNode(d), &requested, &nreq, k),
                        .err => |msg| {
                            zapi.setFilterError(msg);
                            return null;
                        },
                    }
                }
            };
            if (nreq > 0) return null;
            const setup: filter.Setup = .init(&d.cfg, &meta);

            // the frame's linear RGB, when this request decodes it before
            // the output pass (measuring its own frame, contrast recovery)
            var lin: ?*filter.LinearFrame = null;
            defer if (lin) |l| d.lin.?.give(l);

            var detected: ?tc.PeakState = null;
            // Config.peakUsable: no frame is active without the cache
            std.debug.assert(!setup.peak_active or d.peaks != null);
            if (d.peaks) |*peaks| if (setup.peak_active) {
                var entries: [filter.Config.max_window]filter.PeakCache.Entry = undefined;
                const count: usize = @intCast(n - lo + 1);
                // own frame first: earlier ones are likely measured already
                var k = count;
                while (k > 0) {
                    k -= 1;
                    const i: c_int = lo + @as(c_int, @intCast(k));
                    const ui: usize = @intCast(i);
                    const cached = peaks.acquire(ui) catch {
                        zapi.setFilterError(filter_name ++ ": out of memory");
                        return null;
                    };
                    entries[k] = cached orelse blk: {
                        if (i == n) if (d.lin) |*pool| {
                            lin = pool.take();
                        };
                        const e = (if (i == n) measure(d, &zapi, &setup, src, n, lin) else measureOther(d, &zapi, i)) catch |err| {
                            peaks.release(ui);
                            zapi.setFilterError(if (err == error.InvalidRpu) invalid_rpu else filter_name ++ ": out of memory");
                            return null;
                        };
                        peaks.put(ui, e);
                        break :blk e;
                    };
                }
                detected = filter.PeakCache.replay(entries[0..count], &d.cfg.peak);
            };

            const plan = allocator.create(filter.Plan) catch {
                zapi.setFilterError(filter_name ++ ": out of memory");
                return null;
            };
            defer allocator.destroy(plan);
            plan.init(&d.cfg, &setup, detected, &d.gamut) catch {
                zapi.setFilterError(filter_name ++ ": out of memory");
                return null;
            };
            defer plan.deinit(&d.gamut);

            const dst = src.newVideoFrame3(.{ .format = &d.vi.format });
            const out: filter.PlanesOut(T) = .{
                .ptr = .{ dst.getWriteSlice2(T, 0).ptr, dst.getWriteSlice2(T, 1).ptr, dst.getWriteSlice2(T, 2).ptr },
                .stride = .{ dst.getStride2(T, 0), dst.getStride2(T, 1), dst.getStride2(T, 2) },
            };
            const el = elFrame(d, &zapi, n);
            defer if (el) |e| e.deinit();
            if (lin == null and plan.recovery > 0) if (d.lin) |*pool| {
                lin = pool.take();
            };
            filter.process(T, T, &d.cfg, plan, planesIn(T, src, el), out, lin) catch return hz.frameOom(&zapi, filter_name, dst);

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

        fn measure(d: *Data, zapi: *const ZAPI, s: *const filter.Setup, f: anytype, i: c_int, keep: ?*filter.LinearFrame) !filter.PeakCache.Entry {
            const el = elFrame(d, zapi, i);
            defer if (el) |e| e.deinit();
            const m = try filter.measure(T, &d.cfg, s, planesIn(T, f, el), keep);
            return .{ .m = m, .active = s.peak_active };
        }

        fn measureOther(d: *Data, zapi: *const ZAPI, i: c_int) !filter.PeakCache.Entry {
            const f = zapi.initZFrame(d.node, i);
            defer f.deinit();
            var meta: filter.FrameMeta = undefined;
            var dv: dovi.Meta = undefined;
            // resolved before the claim (getFrame), so this cannot need more
            if (loadMeta(d, zapi, f, i, &meta, &dv) != .ok) return error.InvalidRpu;
            const s: filter.Setup = .init(&d.cfg, &meta);
            return measure(d, zapi, &s, f, i, null);
        }
    };
}

fn tonemapFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    if (d.peaks) |*p| p.deinit();
    if (d.lin) |*p| p.deinit();
    d.gamut.deinit();
    zapi.freeNode(d.node);
    zapi.freeNode(d.el_node);
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
    // 9- to 16-bit integers share the u16 carrier
    const int_in = fmt.sampleType == .Integer and fmt.bitsPerSample >= 9 and fmt.bitsPerSample <= 16;
    const f32in = fmt.sampleType == .Float and fmt.bitsPerSample == 32;
    const family_ok = fmt.colorFamily == .RGB or (fmt.colorFamily == .YUV and fmt.subSamplingW <= 1 and fmt.subSamplingH <= 1);
    if (!(int_in or f32in) or !family_ok) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": input must be 9- to 16-bit integer or 32-bit float RGB or YUV (4:4:4, 4:2:2, 4:2:0, 4:4:0).");
    }
    if (vi.width < filter.min_size or vi.height < filter.min_size) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": input must be at least 64x64.");
    }

    const mp = hz.Maps.init(&map_in, &map_out, filter_name);
    const src_csp = mp.getValue(i64, "src_csp", 1, 0, 3) catch return zapi.freeNode(d.node);
    const dst_csp = mp.getValue(i64, "dst_csp", 0, 0, 3) catch return zapi.freeNode(d.node);
    if (src_csp == 0) return fail(map_out, &zapi, d.node, filter_name ++ ": src_csp must be 1 (HDR10), 2 (HLG) or 3 (Dolby Vision).");
    if (dst_csp == 3) return fail(map_out, &zapi, d.node, filter_name ++ ": Dolby Vision output (dst_csp=3) is not supported.");
    // use_dovi only selects the enhancement layer, as in vs-placebo (the
    // RPU metadata applies whenever the prop is present)
    // dovi_el is taken last, after every other argument (see below)
    const use_dovi = (map_in.getValue(i64, "use_dovi") orelse @intFromBool(src_csp == 3)) != 0;
    if (src_csp == 3 and !(fmt.colorFamily == .YUV and int_in)) {
        return fail(map_out, &zapi, d.node, filter_name ++ ": Dolby Vision input must be an integer YUV clip.");
    }

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
            // capped so that five periods fit Config.max_window
            .smoothing_period = @floatCast(mp.getValue(f64, "smoothing_period", 20, 0, filter.Config.max_smoothing_period) catch return zapi.freeNode(d.node)),
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
        .bits = if (f32in) 16 else @intCast(fmt.bitsPerSample),
        .ssw = @intCast(fmt.subSamplingW),
        .ssh = @intCast(fmt.subSamplingH),
    };
    // 0 when detection can't run on any frame: no window, no cache
    d.window = d.cfg.peakWindow();
    // cache_frame (vszip's own): keep the frame's linear RGB from the
    // detection or contrast-recovery pass for the output pass (12 bytes a
    // pixel per frame in flight) instead of decoding it twice
    const cache_frame = (mp.getValue(i64, "cache_frame", 1, 0, 1) catch return zapi.freeNode(d.node)) != 0;
    if (cache_frame and (d.window > 0 or d.cfg.contrast_recovery > 0)) d.lin = .init(&d.cfg);

    d.vi = vi.*;
    _ = zapi.getVideoFormatByID(&d.vi.format, switch (fmt.colorFamily) {
        .RGB => if (f32in) .RGBS else .RGB48,
        else => if (f32in) .YUV444PS else .YUV444P16,
    });

    // Dolby Vision FEL, taken last so no earlier error has to free it.
    // vs-placebo ignores it without use_dovi.
    if (map_in.getNodeVi("dovi_el")) |el_arg| {
        const el, const evi = el_arg;
        if (!use_dovi) {
            zapi.freeNode(el);
        } else {
            const ef = evi.format;
            const el_ok = vapoursynth.vshelper.isConstantVideoFormat(evi) and ef.colorFamily == .YUV and
                ef.sampleType == .Integer and ef.bitsPerSample >= 9 and ef.bitsPerSample <= 16 and
                ef.subSamplingW <= 1 and ef.subSamplingH <= 1;
            d.cfg.el = if (el_ok) filter.ElGeometry.init(
                @intCast(vi.width),
                @intCast(vi.height),
                @intCast(evi.width),
                @intCast(evi.height),
                @intCast(ef.subSamplingW),
                @intCast(ef.subSamplingH),
                @intCast(ef.bitsPerSample),
            ) else null;
            if (d.cfg.el == null) {
                zapi.freeNode(el);
                return fail(map_out, &zapi, d.node, filter_name ++ ": dovi_el must be a 9- to 16-bit integer YUV clip (4:2:0, 4:2:2, 4:4:0 or 4:4:4) of the BL's size or half of it.");
            }
            d.el_node = el;
        }
    }

    if (d.window > 0) {
        d.peaks = filter.PeakCache.init(@intCast(vi.numFrames)) catch {
            zapi.freeNode(d.el_node);
            return fail(map_out, &zapi, d.node, filter_name ++ ": out of memory");
        };
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    // Dolby Vision RPU chains can reach earlier frames too
    const pattern: vs.RequestPattern = if (d.window > 1 or d.cfg.src_csp == .dovi) .General else .StrictSpatial;
    const deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = pattern },
        .{ .source = d.el_node, .requestPattern = pattern },
    };
    const gf: vs.FilterGetFrame = if (f32in) &Tonemap(f32).getFrame else &Tonemap(u16).getFrame;
    zapi.createVideoFilter(out, filter_name, &d.vi, gf, tonemapFree, .Parallel, deps[0..if (d.el_node != null) 2 else 1], data);
}
