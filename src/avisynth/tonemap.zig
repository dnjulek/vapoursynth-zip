//! avszip_Tonemap(clip, int "src_csp", int "dst_csp", int "dst_prim", float "src_max", ...,
//!   bool "cache_frame")
//!
//! vs-placebo's Tonemap on the CPU (filters/tonemap.zig), bit-exact with
//! vszip: same parameters (vszip's int booleans are AviSynth bools), HDR10,
//! HLG and Dolby Vision (the `DolbyVisionRPU` prop, and the FEL as
//! `dovi_el`). Input: 10-16 bit integer or 32-bit float RGB or YUV (4:2:0,
//! 4:2:2, 4:4:4); output RGBP16 / YUV444P16 (float: RGBPS / YUV444PS).
//! Alpha is dropped.
//!
//! Frame n's peak detection replays frames n - window + 1 .. n through the
//! shared `PeakCache` (claim/wait per frame), and a Dolby Vision RPU that
//! continues an earlier one walks back in frame order, as in vszip. AviSynth
//! fetches frames synchronously, so both simply read the frames they need.
//!
//! Planes go to the kernel in place at their own pitch: it reads and writes
//! only active samples, so cropped sources need no canonical copy.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/tonemap.zig");
const tc = filter.tc;
const dovi = filter.dovi;
const hz = @import("helper.zig");

pub const filter_name = "avszip_Tonemap";
pub const params = "c[src_csp]i[dst_csp]i[dst_prim]i[src_max]f[src_min]f[dst_max]f[dst_min]f" ++
    "[dynamic_peak_detection]b[smoothing_period]f[scene_threshold_low]f[scene_threshold_high]f[percentile]f[black_cutoff]f" ++
    "[gamut_mapping]i[tone_mapping_function]i[tone_mapping_function_s]s[tone_mapping_param]f" ++
    "[metadata]i[use_dovi]b[dovi_el]c[show_clipping]b[contrast_recovery]f[contrast_smoothness]f" ++
    "[lut_size]i[mastering_clip]f[film_strength]f[inverse_tone_mapping]b" ++
    "[knee_adaptation]f[knee_minimum]f[knee_maximum]f[knee_default]f[knee_offset]f" ++
    "[slope_tuning]f[slope_offset]f[spline_contrast]f[reinhard_contrast]f[linear_knee]f[exposure]f" ++
    "[gamut_expansion]b[lut3d_size]i*[lut3d_tricubic]b" ++
    "[perceptual_deadzone]f[perceptual_strength]f[colorimetric_gamma]f[softclip_knee]f[softclip_desat]f" ++
    "[cache_frame]b";

const Data = struct {
    /// output format (YUV444 / RGB, 16-bit or float)
    out_format: avs.Format,
    in_format: avs.Format,
    num_frames: c_int,
    /// Dolby Vision FEL clip (dovi_el with use_dovi): composed with
    /// src_csp=3, and the RPU source, as in vs-placebo
    el: avs.OwnedClip = .{},
    cfg: filter.Config,
    window: u32,
    peaks: ?filter.PeakCache = null,
    gamut: filter.GamutCache = .{},
    /// cache_frame: linear frames kept from detection (or contrast
    /// recovery's feature pass) for the output pass
    lin: ?filter.LinearPool = null,

    pub fn deinit(self: *Data) void {
        if (self.peaks) |*p| p.deinit();
        if (self.lin) |*p| p.deinit();
        self.gamut.deinit();
        self.el.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

/// A float property value as f32; non-finite values, and finite ones that
/// would overflow f32 to inf, read as absent (0).
fn propF32(v: f64) f32 {
    return if (math.isFinite(v) and @abs(v) <= math.floatMax(f32)) @floatCast(v) else 0;
}

/// A frame property as f32, stored as float or int (sources differ).
fn propNum(props: avs.PropsRO, comptime key: [:0]const u8) f32 {
    if (props.getFloat(key, 0)) |v| return propF32(v) else |_| {}
    if (props.getInt(key, 0)) |v| return @floatFromInt(v) else |_| {}
    return 0;
}

fn propTriple(props: avs.PropsRO, comptime key: [:0]const u8) ?[3]f32 {
    const arr = props.getFloatArray(key) catch return null;
    if (arr.len != 3) return null;
    var r: [3]f32 = undefined;
    for (&r, arr) |*o, v| o.* = propF32(v);
    return r;
}

fn readMeta(env: avs.Env, frame: avs.Frame) filter.FrameMeta {
    const props = frame.propsRO(env);
    var m: filter.FrameMeta = .{};
    // AviSynth's _ColorRange: 0 = full, 1 = limited
    m.full_range = if (props.getInt("_ColorRange", 0)) |r| r == 0 else |_| null;
    m.chroma_loc = props.getInt("_ChromaLocation", 0) catch null;
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
    } else |_| {}
    m.prim_x = propTriple(props, "MasteringDisplayPrimariesX");
    m.prim_y = propTriple(props, "MasteringDisplayPrimariesY");
    m.white = .{ .x = propNum(props, "MasteringDisplayWhitePointX"), .y = propNum(props, "MasteringDisplayWhitePointY") };
    return m;
}

const missing_rpu = "Clips are missing `DolbyVisionRPU` prop for Dolby Vision mapping!";
const invalid_rpu = "invalid `DolbyVisionRPU` prop (failed parsing the RPU).";

/// The frame's DolbyVisionRPU, parsed; null when absent.
fn readRpu(env: avs.Env, frame: avs.Frame) dovi.Error!?dovi.Rpu {
    const raw = frame.propsRO(env).getData("DolbyVisionRPU", 0) catch return null;
    return try dovi.parse(raw);
}

/// How far back a frame's RPU may defer to earlier ones (use_prev_vdr_rpu_flag,
/// missing or compressed DM data) before the chain counts as broken.
const max_rpu_chain = 300;

/// The clip the DolbyVisionRPU props are read from.
fn rpuClip(d: *const Data, ctx: *avs.FrameCtx) avs.Clip {
    return d.el.clip orelse ctx.child();
}

/// The RPU of frame `i` of the RPU clip (error when missing or invalid).
fn fetchRpu(d: *const Data, ctx: *avs.FrameCtx, i: c_int) !dovi.Rpu {
    const f = try ctx.getFrame(rpuClip(d, ctx), i);
    defer f.release();
    const r = readRpu(ctx.env(), f) catch return ctx.fail(invalid_rpu, .{});
    return r orelse return ctx.fail(missing_rpu, .{});
}

/// Frame i's metadata into `m` (vszip's loadMeta). With an RPU, its DM data
/// feeds the HDR metadata whatever src_csp is (as in vs-placebo). With
/// src_csp dovi, the reshaping state is resolved into `dv`: frame i folds
/// the RPUs from the last complete one at or before it, in frame order.
/// With an EL clip the RPUs are read from it.
fn loadMeta(d: *const Data, ctx: *avs.FrameCtx, frame: avs.Frame, i: c_int, m: *filter.FrameMeta, dv: *dovi.Meta) !void {
    const env = ctx.env();
    const dovi_src = d.cfg.src_csp == .dovi;
    const el_present = d.el.clip != null;
    m.* = readMeta(env, frame);
    const rpu_read = if (d.el.clip) |el| blk: {
        const f = try ctx.getFrame(el, i);
        defer f.release();
        break :blk readRpu(env, f);
    } else readRpu(env, frame);
    const rpu = (rpu_read catch {
        if (dovi_src) return ctx.fail(invalid_rpu, .{});
        return;
    }) orelse {
        if (dovi_src) return ctx.fail(missing_rpu, .{});
        return;
    };
    m.dovi_dm = rpu.dm;
    m.dovi_profile = rpu.profile();
    if (!dovi_src) return;

    var base = i;
    if (!rpu.complete()) {
        while (base > 0) {
            if (i - base >= max_rpu_chain) return ctx.fail("no complete DolbyVisionRPU within 300 frames.", .{});
            base -= 1;
            if ((try fetchRpu(d, ctx, base)).complete()) break;
        }
    }
    dv.* = .{};
    var j = base;
    while (j < i) : (j += 1) {
        const r = try fetchRpu(d, ctx, j);
        dv.update(&r, el_present);
    }
    dv.update(&rpu, el_present);
    m.dovi_meta = dv;
}

fn planesIn(comptime T: type, d: *const Data, f: avs.Frame, el: ?avs.Frame) filter.Planes(T) {
    var out: filter.Planes(T) = undefined;
    for (0..3) |p| {
        const pl = f.plane(T, d.in_format.planeId(p));
        out.ptr[p] = @ptrCast(@alignCast(pl.ptr));
        out.stride[p] = pl.stride();
    }
    out.el = if (el) |e| blk: {
        var ep: filter.ElPlanes = undefined;
        const ids = [3]c_int{ c.AVS_PLANAR_Y, c.AVS_PLANAR_U, c.AVS_PLANAR_V };
        for (0..3) |p| {
            const pl = e.plane(u16, ids[p]);
            ep.ptr[p] = @ptrCast(@alignCast(pl.ptr));
            ep.stride[p] = pl.stride();
        }
        break :blk ep;
    } else null;
    return out;
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
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            var meta: filter.FrameMeta = undefined;
            var dv: dovi.Meta = undefined;
            try loadMeta(d, ctx, src, n, &meta, &dv);
            const setup: filter.Setup = .init(&d.cfg, &meta);

            // the frame's linear RGB, when this request decodes it before
            // the output pass (measuring its own frame, contrast recovery)
            var lin: ?*filter.LinearFrame = null;
            defer if (lin) |l| d.lin.?.give(l);

            var detected: ?tc.PeakState = null;
            // Config.peakUsable: no frame is active without the cache
            std.debug.assert(!setup.peak_active or d.peaks != null);
            if (d.peaks) |*peaks| if (setup.peak_active) {
                const lo: c_int = if (d.window > 1) @max(0, n - @as(c_int, @intCast(d.window)) + 1) else n;
                var entries: [filter.Config.max_window]filter.PeakCache.Entry = undefined;
                const count: usize = @intCast(n - lo + 1);
                // own frame first: earlier ones are likely measured already
                var k = count;
                while (k > 0) {
                    k -= 1;
                    const i: c_int = lo + @as(c_int, @intCast(k));
                    const ui: usize = @intCast(i);
                    const cached = peaks.acquire(ui) catch return ctx.fail("out of memory", .{});
                    entries[k] = cached orelse blk: {
                        if (i == n) if (d.lin) |*pool| {
                            lin = pool.take();
                        };
                        const e = (if (i == n) measure(d, ctx, &setup, src, n, lin) else measureOther(d, ctx, i)) catch |err| {
                            peaks.release(ui);
                            return err;
                        };
                        peaks.put(ui, e);
                        break :blk e;
                    };
                }
                detected = filter.PeakCache.replay(entries[0..count], &d.cfg.peak);
            };

            const plan = std.heap.c_allocator.create(filter.Plan) catch return ctx.fail("out of memory", .{});
            defer std.heap.c_allocator.destroy(plan);
            plan.init(&d.cfg, &setup, detected, &d.gamut) catch return ctx.fail("out of memory", .{});
            defer plan.deinit(&d.gamut);

            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            var out: filter.PlanesOut(T) = undefined;
            for (0..3) |p| {
                const pl = dst.planeMut(T, d.out_format.planeId(p));
                out.ptr[p] = @ptrCast(@alignCast(pl.ptr));
                out.stride[p] = pl.stride();
            }
            const el = if (d.el.clip) |e| try ctx.getFrame(e, n) else null;
            defer if (el) |e| e.release();
            if (lin == null and plan.recovery > 0) if (d.lin) |*pool| {
                lin = pool.take();
            };
            filter.process(T, T, &d.cfg, plan, planesIn(T, d, src, el), out, lin) catch return ctx.fail("out of memory", .{});

            const props = dst.propsRW(ctx.env());
            const dst_trc = setup.dst.trc;
            const full = d.cfg.rgb or d.cfg.float or setup.full;
            props.setInt("_Matrix", if (d.cfg.rgb) 0 else if (dst_trc == .bt1886) 1 else 9, .replace) catch {};
            props.setInt("_Transfer", switch (dst_trc) {
                .bt1886 => 1,
                .pq => 16,
                .hlg => 18,
            }, .replace) catch {};
            props.setInt("_Primaries", vsPrimaries(setup.dst.prim), .replace) catch {};
            hz.setRange(ctx.env(), dst, if (full) .FULL else .LIMITED);
            _ = props.delete("_ChromaLocation");
            return dst;
        }

        fn measure(d: *Data, ctx: *avs.FrameCtx, s: *const filter.Setup, f: avs.Frame, i: c_int, keep: ?*filter.LinearFrame) !filter.PeakCache.Entry {
            const el = if (d.el.clip) |e| try ctx.getFrame(e, i) else null;
            defer if (el) |e| e.release();
            const m = filter.measure(T, &d.cfg, s, planesIn(T, d, f, el), keep) catch return ctx.fail("out of memory", .{});
            return .{ .m = m, .active = s.peak_active };
        }

        fn measureOther(d: *Data, ctx: *avs.FrameCtx, i: c_int) !filter.PeakCache.Entry {
            const f = try ctx.childFrame(i);
            defer f.release();
            var meta: filter.FrameMeta = undefined;
            var dv: dovi.Meta = undefined;
            try loadMeta(d, ctx, f, i, &meta, &dv);
            const s: filter.Setup = .init(&d.cfg, &meta);
            return measure(d, ctx, &s, f, i, null);
        }
    };
}

/// Whether the tone constant that `tone_mapping_param` sets for `f`
/// (pl_tone_map_params_infer) was passed explicitly.
fn paramConstantGiven(arg: anytype, f: tc.ToneFn) bool {
    return switch (f) {
        .st2094_40, .st2094_10 => arg.defined("knee_adaptation"),
        .bt2390 => arg.defined("knee_offset"),
        .spline => arg.defined("spline_contrast"),
        .reinhard => arg.defined("reinhard_contrast"),
        .mobius, .gamma => arg.defined("linear_knee"),
        .linear, .linear_light => arg.defined("exposure"),
        .clip, .bt2446a, .hable => false,
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const fmt = try hz.planarFormat(ctx, vi);

    // 10- to 16-bit integers share the u16 carrier
    const int_in = fmt.sample == .u16;
    const f32in = fmt.isFloat();
    const family_ok = fmt.family == .rgb or (fmt.family == .yuv and fmt.ssw <= 1 and fmt.ssh <= 1);
    if (!(int_in or f32in) or !family_ok) {
        return ctx.fail("input must be 10- to 16-bit integer or 32-bit float RGB or YUV (4:4:4, 4:2:2, 4:2:0).", .{});
    }
    if (vi.width < filter.min_size or vi.height < filter.min_size) return ctx.fail("input must be at least 64x64.", .{});

    const src_csp = try arg.int(i64, "src_csp", 1, 0, 3);
    const dst_csp = try arg.int(i64, "dst_csp", 0, 0, 3);
    if (src_csp == 0) return ctx.fail("src_csp must be 1 (HDR10), 2 (HLG) or 3 (Dolby Vision).", .{});
    if (dst_csp == 3) return ctx.fail("Dolby Vision output (dst_csp=3) is not supported.", .{});
    // use_dovi only selects the enhancement layer, as in vs-placebo (the
    // RPU metadata applies whenever the prop is present)
    const use_dovi = try arg.boolean("use_dovi", src_csp == 3);
    if (src_csp == 3 and !(fmt.family == .yuv and int_in)) return ctx.fail("Dolby Vision input must be an integer YUV clip.", .{});

    const tone_index = try arg.int(i64, "tone_mapping_function", @intFromEnum(tc.ToneFn.spline), 0, tc.ToneFn.names.len - 1);
    var tone_fn: tc.ToneFn = @enumFromInt(tone_index);
    if (try arg.string("tone_mapping_function_s")) |name| {
        tone_fn = tc.ToneFn.fromName(name) orelse return ctx.fail("unknown tone_mapping_function_s '{s}'.", .{name});
    }

    const fmax = 10000.0;
    const dst_prim: ?tc.Prim = if (try arg.optInt(i64, "dst_prim", 0, tc.Prim.count - 1)) |v| @enumFromInt(v) else null;

    // vs-placebo always passes tone_mapping_param (param_def when absent),
    // which overrides the function's tone constant; keep that default, but
    // not over an explicitly passed constant
    const param_def: f32 = if (paramConstantGiven(arg, tone_fn)) 0 else tone_fn.paramDef();

    // lut3d_size: (I, C, h); 0 keeps that axis' default, like PL_DEF
    var lut3d: [3]u32 = .{ 48, 32, 256 };
    var lut_buf: [3]i64 = undefined;
    if (try arg.intArray(i64, "lut3d_size", &lut_buf, math.minInt(i64), math.maxInt(i64))) |a| {
        if (a.len != 3) return ctx.fail("lut3d_size must have 3 values (I, C, h).", .{});
        for (a, &lut3d) |v, *o| {
            if (v == 0) continue;
            if (v < 2 or v > 1024) return ctx.fail("lut3d_size values must be 0 (default) or 2..1024.", .{});
            o.* = @intCast(v);
        }
        // 16 bytes per entry: cap the LUT at 256 MB
        if (@as(u64, lut3d[0]) * lut3d[1] * lut3d[2] > 1 << 24) {
            return ctx.fail("lut3d_size is too large (I * C * h must be at most 16777216).", .{});
        }
    }

    var cfg: filter.Config = .{
        .src_csp = @enumFromInt(src_csp),
        .dst_csp = @enumFromInt(dst_csp),
        .dst_prim = dst_prim,
        .src_max = try arg.float(f32, "src_max", 0, 0, fmax),
        .src_min = try arg.float(f32, "src_min", 0, 0, fmax),
        .dst_max = try arg.float(f32, "dst_max", 0, 0, fmax),
        .dst_min = try arg.float(f32, "dst_min", 0, 0, fmax),
        .peak_detect = try arg.boolean("dynamic_peak_detection", true),
        .peak = .{
            // capped so that five periods fit Config.max_window
            .smoothing_period = try arg.float(f32, "smoothing_period", 20, 0, filter.Config.max_smoothing_period),
            .scene_threshold_low = try arg.float(f32, "scene_threshold_low", 1, 0, 100),
            .scene_threshold_high = try arg.float(f32, "scene_threshold_high", 3, 0, 100),
            .percentile = try arg.float(f32, "percentile", 100, 0, 100),
            .black_cutoff = try arg.float(f32, "black_cutoff", 1, 0, 100),
        },
        .gamut_fn = @enumFromInt(try arg.int(i64, "gamut_mapping", @intFromEnum(tc.GamutFn.perceptual), 0, tc.GamutFn.count - 1)),
        .tone_fn = tone_fn,
        .tone_param = try arg.float(f32, "tone_mapping_param", param_def, 0, 10),
        .metadata = @enumFromInt(try arg.int(i64, "metadata", 0, 0, 4)),
        .show_clipping = try arg.boolean("show_clipping", false),
        .contrast_recovery = try arg.float(f32, "contrast_recovery", 0, 0, 10),
        .contrast_smoothness = try arg.float(f32, "contrast_smoothness", 3.5, 1, 32),
        .lut_size = @intCast(try arg.int(i64, "lut_size", 256, 2, 4096)),
        .mastering_clip = try arg.float(f32, "mastering_clip", 1, 0, 1),
        .film_strength = try arg.float(f32, "film_strength", 0.3, 0, 2),
        .inverse_tone_mapping = try arg.boolean("inverse_tone_mapping", false),
        // ranges as documented on pl_tone_map_constants / pl_gamut_map_constants
        .tone_constants = .{
            .knee_adaptation = try arg.float(f32, "knee_adaptation", 0.4, 0, 1),
            .knee_minimum = try arg.float(f32, "knee_minimum", 0.1, 0, 0.5),
            .knee_maximum = try arg.float(f32, "knee_maximum", 0.8, 0.5, 1),
            .knee_default = try arg.float(f32, "knee_default", 0.4, 0, 1),
            .knee_offset = try arg.float(f32, "knee_offset", 1, 0.5, 2),
            .slope_tuning = try arg.float(f32, "slope_tuning", 1.5, 0, 10),
            .slope_offset = try arg.float(f32, "slope_offset", 0.2, 0, 1),
            .spline_contrast = try arg.float(f32, "spline_contrast", 0.5, 0, 1.5),
            .reinhard_contrast = try arg.float(f32, "reinhard_contrast", 0.5, 0, 1),
            .linear_knee = try arg.float(f32, "linear_knee", 0.3, 0, 1),
            .exposure = try arg.float(f32, "exposure", 1, 0, 10),
        },
        .gamut_expansion = try arg.boolean("gamut_expansion", false),
        .lut3d_size = lut3d,
        .lut3d_tricubic = try arg.boolean("lut3d_tricubic", false),
        .gamut_constants = .{
            .perceptual_deadzone = try arg.float(f32, "perceptual_deadzone", 0.3, 0, 1),
            .perceptual_strength = try arg.float(f32, "perceptual_strength", 0.8, 0, 1),
            .colorimetric_gamma = try arg.float(f32, "colorimetric_gamma", 1.8, 0, 10),
            .softclip_knee = try arg.float(f32, "softclip_knee", 0.7, 0, 1),
            .softclip_desat = try arg.float(f32, "softclip_desat", 0.35, 0, 1),
        },
        .width = @intCast(vi.width),
        .height = @intCast(vi.height),
        .rgb = fmt.family == .rgb,
        .float = f32in,
        .bits = if (f32in) 16 else @intCast(fmt.bits),
        .ssw = @intCast(fmt.ssw),
        .ssh = @intCast(fmt.ssh),
    };
    // cache_frame (vszip's own): keep the frame's linear RGB from the
    // detection or contrast-recovery pass for the output pass
    const cache_frame = try arg.boolean("cache_frame", true);

    // Dolby Vision FEL; vs-placebo ignores it without use_dovi
    var el: ?avs.Clip = null;
    errdefer if (el) |e| e.release();
    if (try arg.optClip("dovi_el")) |e| {
        if (!use_dovi) {
            e.release();
        } else {
            el = e;
            const evi = e.videoInfo();
            const ef: ?avs.Format = avs.Format.fromVideoInfo(evi) catch null;
            const el_ok = if (ef) |f| f.family == .yuv and f.sample == .u16 and f.ssw <= 1 and f.ssh <= 1 else false;
            cfg.el = if (el_ok) filter.ElGeometry.init(
                @intCast(vi.width),
                @intCast(vi.height),
                @intCast(evi.width),
                @intCast(evi.height),
                @intCast(ef.?.ssw),
                @intCast(ef.?.ssh),
                @intCast(ef.?.bits),
            ) else null;
            if (cfg.el == null) {
                return ctx.fail("dovi_el must be a 10- to 16-bit integer YUV clip (4:2:0, 4:2:2 or 4:4:4) of the BL's size or half of it.", .{});
            }
        }
    }

    var d: Data = .{
        .out_format = .{
            .family = fmt.family,
            .sample = if (f32in) .f32 else .u16,
            .bits = if (f32in) 32 else 16,
            .color_planes = 3,
            .has_alpha = false,
            .ssw = 0,
            .ssh = 0,
        },
        .in_format = fmt,
        .num_frames = vi.num_frames,
        .cfg = cfg,
        // 0 when detection can't run on any frame: no window, no cache
        .window = cfg.peakWindow(),
    };
    if (cache_frame and (d.window > 0 or cfg.contrast_recovery > 0)) d.lin = .init(&d.cfg);
    errdefer if (d.lin) |*p| p.deinit();
    if (d.window > 0) d.peaks = filter.PeakCache.init(@intCast(vi.num_frames)) catch return ctx.fail("out of memory", .{});
    errdefer if (d.peaks) |*p| p.deinit();
    d.el = .{ .clip = el };

    const vi_out = hz.withFormat(vi.*, d.out_format);
    const gf = if (f32in) F.getFrame(Tonemap(f32).getFrame) else F.getFrame(Tonemap(u16).getFrame);
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice, .vi = vi_out });
    return avs.Create.returnClip(out);
}
