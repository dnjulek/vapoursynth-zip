//! avszip_SSIMULACRA2(clip reference, clip distorted)
//!
//! Returns `reference` with the per-frame prop SSIMULACRA2. Planar RGB inputs
//! (8-16 bit int, 32-bit float) are read natively, bit-exact with vszip.
//! YUV with 2x or no subsampling (8-16 bit int, 32-bit float) is converted to
//! RGB row by row inside the filter with vszip's code (filters/ssimulacra2.zig
//! YuvSetup: each frame's `_ColorRange`, `_Matrix` (2 or unset: BT.709 above
//! 650 lines, else BT.601) and `_ChromaLocation`, zimg's Bicubic chroma
//! upsampling), and the reference is returned as is. Gray and other YUV
//! layouts are converted to float RGB per frame: an internal stage normalizes
//! each frame with its own `_ColorRange` (limited when unset), AviSynth's
//! ConvertToYUV444 upsamples other subsamplings, then the frame's `_Matrix`
//! applies; such a reference is returned as RGBPS (RGBAPS with its alpha). The
//! sRGB transfer is undone in the filter unless frame 0 carries `_Transfer` =
//! 8 (linear).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter_ssim = @import("../filters/ssimulacra2.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_SSIMULACRA2";
pub const params = "c[distorted]c";
const Scratch = []align(common.vec_len) f32;

/// How one clip's frames are read: native RGB (int via LUT, or f32), or
/// normalized float YUV444/Gray converted to RGB per frame (`yuv`).
const Source = struct {
    kind: filter_ssim.Input.Kind = .f32,
    srgb: bool = true,
    lut: []f32 = &.{},
    yuv: ?Yuv = null,
    /// YUV converted inside the filter (the clip is read as is)
    direct: ?filter_ssim.YuvSetup = null,

    fn deinit(self: *Source) void {
        allocator.free(self.lut);
        if (self.direct) |*ys| ys.deinit();
    }
};

const Yuv = struct {
    gray: bool,
    /// For the matrix fallback.
    height: c_int,
};

const Data = struct {
    other: avs.OwnedClip,
    src1: Source = .{},
    src2: Source = .{},
    /// YUVA reference: its alpha is copied into the RGBAPS output.
    out_alpha: bool = false,
    // Reusable scratch buffers; concurrent calls each pop their own buffer
    // (or allocate one) and push it back when done.
    pool: std.ArrayList(Scratch) = .empty,
    pool_mutex: std.Io.Mutex = .init,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |buf| allocator.free(buf);
        self.pool.deinit(allocator);
        self.src1.deinit();
        self.src2.deinit();
        self.other.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn acquireScratch(d: *Data, size: usize) !Scratch {
    {
        d.pool_mutex.lockUncancelable(common.io);
        defer d.pool_mutex.unlock(common.io);
        if (d.pool.pop()) |buf| return buf;
    }
    return allocator.alignedAlloc(f32, common.alignment, size);
}

fn releaseScratch(d: *Data, buf: Scratch) void {
    d.pool_mutex.lockUncancelable(common.io);
    defer d.pool_mutex.unlock(common.io);
    d.pool.append(allocator, buf) catch allocator.free(buf);
}

/// Input view of a frame's R, G, B planes with one shared byte stride (the
/// planes of a planar RGB frame share a pitch; a mismatching or unaligned
/// one is repacked).
fn input(src: *const Source, frame: avs.Frame, scratch: *hz.Scratch) !filter_ssim.Input {
    const ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };
    var planes: [3][*]const u8 = undefined;
    var pitch: usize = 0;
    inline for (0..3) |p| {
        const raw = frame.plane(u8, ids[p]);
        if (p == 0) pitch = raw.pitch;
        const can = try hz.canonical(u8, raw, pitch, scratch);
        planes[p] = can.ptr;
    }
    return .{ .kind = src.kind, .planes = planes, .stride = pitch, .lut = src.lut, .srgb = src.srgb };
}

/// Input view of a YUV frame converted inside the filter, from that frame's
/// metadata (AviSynth `_ColorRange`: 0 full, 1 limited).
fn directInput(ctx: *avs.FrameCtx, src: *const Source, frame: avs.Frame, n: c_int) !filter_ssim.Input {
    const ys = &src.direct.?;
    const props = frame.propsRO(ctx.env());
    const unset = std.math.minInt(i64);
    const opt = struct {
        fn f(v: i64) ?i64 {
            return if (v == unset) null else v;
        }
    }.f;
    const full = props.intOr("_ColorRange", 1) == 0;
    const m = opt(props.intOr("_Matrix", unset));
    const y = ys.frame(full, m, opt(props.intOr("_ChromaLocation", unset))) orelse
        return ctx.fail("frame {d}: unsupported _Matrix {d} (supported: 1, 4, 5, 6, 7, 9; 2 or unset uses BT.709 above 650 lines, else BT.601).", .{ n, m.? });
    const ids = [3]c_int{ c.AVS_PLANAR_Y, c.AVS_PLANAR_U, c.AVS_PLANAR_V };
    var planes: [3][*]const u8 = undefined;
    var strides: [3]usize = undefined;
    inline for (0..3) |p| {
        const raw = frame.plane(u8, ids[p]);
        planes[p] = raw.ptr;
        strides[p] = raw.pitch;
    }
    return filter_ssim.YuvSetup.input(y, planes, strides, src.srgb);
}

/// YCbCr -> RGB coefficients for normalized full-range input (Y in [0, 1],
/// Cb/Cr in [-0.5, 0.5]): R = Y + cr_r*Cr, G = Y + cb_g*Cb + cr_g*Cr,
/// B = Y + cb_b*Cb. Computed in f64 and rounded once, like zimg.
const Coeffs = struct { cr_r: f32, cb_g: f32, cr_g: f32, cb_b: f32 };

fn coeffsOf(matrix: i64) ?Coeffs {
    const kr: f64, const kb: f64 = switch (matrix) {
        1 => .{ 0.2126, 0.0722 }, // BT.709
        4 => .{ 0.30, 0.11 }, // FCC
        5, 6 => .{ 0.299, 0.114 }, // BT.470BG / ST 170M (BT.601)
        7 => .{ 0.212, 0.087 }, // ST 240M
        9 => .{ 0.2627, 0.0593 }, // BT.2020 non-constant luminance
        else => return null,
    };
    const kg = 1.0 - kr - kb;
    return .{
        .cr_r = @floatCast(2.0 * (1.0 - kr)),
        .cb_g = @floatCast(-2.0 * kb * (1.0 - kb) / kg),
        .cr_g = @floatCast(-2.0 * kr * (1.0 - kr) / kg),
        .cb_b = @floatCast(2.0 * (1.0 - kb)),
    };
}

/// Convert one normalized float YUV444/Gray frame to RGB rows at `dst`
/// (three planes sharing `pitch` bytes), with that frame's `_Matrix`.
fn toRgb(ctx: *avs.FrameCtx, yuv: Yuv, src: avs.Frame, dst: [3][*]u8, pitch: usize, n: c_int) !void {
    const y_pl = src.plane(f32, c.AVS_PLANAR_Y);
    const w = y_pl.width;
    if (yuv.gray) {
        for (0..y_pl.height) |y| {
            for (dst) |plane| {
                const out: [*]f32 = @ptrCast(@alignCast(plane + y * pitch));
                @memcpy(out[0..w], y_pl.row(y));
            }
        }
        return;
    }
    var m = src.propsRO(ctx.env()).intOr("_Matrix", 2);
    if (m == 2) m = if (yuv.height > 650) 1 else 6;
    const k = coeffsOf(m) orelse
        return ctx.fail("frame {d}: unsupported _Matrix {d} (supported: 1, 4, 5, 6, 7, 9; 2 or unset uses BT.709 above 650 lines, else BT.601).", .{ n, m });
    const u_pl = src.plane(f32, c.AVS_PLANAR_U);
    const v_pl = src.plane(f32, c.AVS_PLANAR_V);
    for (0..y_pl.height) |y| {
        const ys = y_pl.row(y);
        const us = u_pl.row(y);
        const vs = v_pl.row(y);
        const r: [*]f32 = @ptrCast(@alignCast(dst[0] + y * pitch));
        const g: [*]f32 = @ptrCast(@alignCast(dst[1] + y * pitch));
        const b: [*]f32 = @ptrCast(@alignCast(dst[2] + y * pitch));
        for (0..w) |x| {
            r[x] = ys[x] + k.cr_r * vs[x];
            g[x] = ys[x] + k.cb_g * us[x] + k.cr_g * vs[x];
            b[x] = ys[x] + k.cb_b * us[x];
        }
    }
}

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const env = ctx.env();
    var scratch: hz.Scratch = .{};
    defer scratch.deinit();
    const w: u32 = @intCast(ctx.vi().width);
    const h: u32 = @intCast(ctx.vi().height);

    // Output: the reference itself (RGB input: only the prop is added), or
    // its per-frame RGB conversion plus its alpha.
    var frame = blk: {
        const ref = try ctx.childFrame(n);
        const yuv = d.src1.yuv orelse break :blk ref;
        defer ref.release();
        const out = try ctx.newFrame(ref);
        errdefer out.release();
        const ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };
        var planes: [3][*]u8 = undefined;
        var pitch: usize = 0;
        for (ids, &planes) |id, *pl| {
            const p = out.planeMut(f32, id);
            pitch = p.pitch; // the planes of a planar RGB frame share a pitch
            pl.* = @ptrCast(p.ptr);
        }
        try toRgb(ctx, yuv, ref, planes, pitch, n);
        if (d.out_alpha) avs.copyPlane(env, out, ref, c.AVS_PLANAR_A);
        const props = out.propsRW(env);
        props.setInt("_Matrix", 0, .replace) catch {};
        props.setInt("_ColorRange", 0, .replace) catch {};
        break :blk out;
    };
    errdefer frame.release();
    const src2 = try ctx.getFrame(d.other.get(), n);
    defer src2.release();

    const in1 = if (d.src1.direct != null) try directInput(ctx, &d.src1, frame, n) else try input(&d.src1, frame, &scratch);
    const in2: filter_ssim.Input = if (d.src2.direct != null) try directInput(ctx, &d.src2, src2, n) else if (d.src2.yuv) |yuv| blk: {
        const pitch: usize = common.ceilN(w * @sizeOf(f32), 64);
        const buf = try scratch.plane(u8, pitch * 3, h);
        const planes = [3][*]u8{ buf.ptr, buf.ptr + pitch * h, buf.ptr + 2 * pitch * h };
        try toRgb(ctx, yuv, src2, planes, pitch, n);
        break :blk .{ .kind = .f32, .planes = .{ planes[0], planes[1], planes[2] }, .stride = pitch, .srgb = d.src2.srgb };
    } else try input(&d.src2, src2, &scratch);

    const buf = acquireScratch(d, filter_ssim.scratchSize(w, h)) catch return ctx.fail("out of memory", .{});
    defer releaseScratch(d, buf);
    const val = filter_ssim.process(in1, in2, w, h, buf);

    try ctx.makePropertyWritable(&frame);
    frame.propsRW(env).setFloat("SSIMULACRA2", val, .replace) catch return ctx.fail("could not set SSIMULACRA2.", .{});
    return frame;
}

fn invokeClip(ctx: *avs.Create, name: [:0]const u8, args: []const c.AVS_Value, names: []const ?[*:0]const u8) !avs.Clip {
    var msg: ?[:0]const u8 = null;
    const v = ctx.env.invoke(name, avs.Value.arrayView(args), names, &msg) catch
        return ctx.fail("internal {s} failed: {s}", .{ name, msg orelse "unknown error" });
    defer v.release();
    if (!v.isClip()) return ctx.fail("internal {s} did not return a clip.", .{name});
    return v.takeClip(ctx.env);
}

/// Internal stage: integer YUV/Gray(+A) to float with each frame's own
/// `_ColorRange` (zimg's normalization: Y in [0, 1], chroma in [-0.5, 0.5];
/// alpha as full-range data). Output frames are tagged full range.
const Norm = struct { format: avs.Format };
const FN = avs.Filter(Norm, filter_name);

fn Normalize(comptime T: type) type {
    return struct {
        fn getFrame(d: *Norm, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const env = ctx.env();
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            const f = d.format;
            const full = src.propsRO(env).intOr("_ColorRange", 1) == 0;
            const sh: u5 = @intCast(f.bits - 8);
            const peak: f64 = @floatFromInt((@as(u32, 1) << @intCast(f.bits)) - 1);
            for (0..f.numPlanes()) |p| {
                const alpha = p >= f.color_planes;
                const chroma = !alpha and p > 0;
                // value = (x - lo) / span, as `x * scale + offset` in f32
                const lo: f64, const span: f64 = if (alpha)
                    .{ 0, peak }
                else if (full)
                    .{ if (chroma) @floatFromInt(@as(u32, 1) << @intCast(f.bits - 1)) else 0, peak }
                else if (chroma)
                    .{ @floatFromInt(@as(u32, 128) << sh), @floatFromInt(@as(u32, 224) << sh) }
                else
                    .{ @floatFromInt(@as(u32, 16) << sh), @floatFromInt(@as(u32, 219) << sh) };
                const scale: f32 = @floatCast(1.0 / span);
                const offset: f32 = @floatCast(-lo / span);
                const id = f.planeId(p);
                const sp = src.plane(T, id);
                const dp = dst.planeMut(f32, id);
                for (0..sp.height) |y| {
                    for (dp.row(y), sp.row(y)) |*o, x| o.* = @as(f32, @floatFromInt(x)) * scale + offset;
                }
            }
            dst.propsRW(env).setInt("_ColorRange", 0, .replace) catch {};
            return dst;
        }
    };
}

/// Owned clip readable by the filter: planar RGB as-is, else normalized
/// float YUV444/Gray (+A) for the per-frame conversion in getFrame.
fn prepare(ctx: *avs.Create, clip: avs.Clip, src: *Source) !avs.Clip {
    const vi = clip.videoInfo();
    const f = try hz.planarFormat(ctx, vi);
    var out = clip.retain();
    errdefer out.release();
    if (f.family == .yuv and filter_ssim.YuvSetup.supports(f.ssw, f.ssh, f.isFloat(), f.bits)) {
        src.srgb = blk: {
            const frame = out.getFrame(0) catch break :blk true;
            defer frame.release();
            break :blk frame.propsRO(ctx.env).intOr("_Transfer", 2) != 8;
        };
        src.kind = .yuv;
        const sample: filter_ssim.YuvSetup.Sample = switch (f.sample) {
            .u8 => .u8,
            .u16 => .u16,
            .f32 => .f32,
        };
        src.direct = try filter_ssim.YuvSetup.init(sample, @intCast(f.bits), @intCast(vi.width), @intCast(vi.height), @intCast(f.ssw), @intCast(f.ssh));
        return out;
    }
    if (f.family != .rgb) {
        // Float first, then chroma upsampling, then the matrix: like zimg, no
        // intermediate integer rounding (measured: max |diff| vs zimg 0.003
        // instead of 0.9 when converting through 8-bit RGB). AviSynth's own
        // ConvertBits/ConvertToPlanarRGB take range and matrix from frame 0
        // (ConvertBits even overwrites later frames' `_ColorRange`), so the
        // metadata-dependent steps run per frame here; ConvertToYUV444 keeps
        // each frame's props and needs no metadata.
        if (!f.isFloat()) {
            var nf = f;
            nf.bits = 32;
            nf.sample = .f32;
            const norm = switch (f.sample) {
                inline .u8, .u16 => |s| FN.getFrame(Normalize(s.Type()).getFrame),
                .f32 => unreachable,
            };
            const flt = try FN.create(ctx.env, out, .{ .format = f }, norm, .{ .mt_mode = .nice, .vi = hz.withFormat(vi.*, nf) });
            out.release();
            out = flt;
        }
        if (f.ssw != 0 or f.ssh != 0) {
            const cv = avs.Value.fromClip(out);
            defer cv.release();
            const a1 = [_]c.AVS_Value{cv.raw};
            const up = try invokeClip(ctx, "ConvertToYUV444", &a1, &.{null});
            out.release();
            out = up;
        }
        src.yuv = .{ .gray = f.family == .gray, .height = vi.height };
    }

    const fmt = try hz.planarFormat(ctx, out.videoInfo());
    src.srgb = blk: {
        const frame = out.getFrame(0) catch break :blk true;
        defer frame.release();
        break :blk frame.propsRO(ctx.env).intOr("_Transfer", 2) != 8;
    };
    if (!fmt.isFloat()) {
        src.kind = if (fmt.sample == .u8) .u8 else .u16;
        src.lut = try allocator.alloc(f32, if (fmt.sample == .u8) 256 else 65536);
        filter_ssim.buildLut(src.lut, @intCast(fmt.bits), src.srgb);
    } else {
        src.kind = .f32;
    }
    return out;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const ref_in = try ctx.clip(0, "reference");
    defer ref_in.release();
    const dist_in = try ctx.clip(1, "distorted");
    defer dist_in.release();
    const vi1 = ref_in.videoInfo();
    const vi2 = dist_in.videoInfo();
    if (vi1.width != vi2.width or vi1.height != vi2.height) return ctx.fail("clips must have the same dimensions.", .{});
    if (vi1.num_frames != vi2.num_frames) return ctx.fail("clips must have the same length.", .{});

    var d: Data = .{ .other = .{} };
    errdefer {
        d.src1.deinit();
        d.src2.deinit();
    }
    const reference = try prepare(ctx, ref_in, &d.src1);
    defer reference.release();
    const distorted = try prepare(ctx, dist_in, &d.src2);
    errdefer distorted.release();
    d.other = .{ .clip = distorted };

    // A YUV reference is returned converted: RGBPS, plus its alpha.
    const out_vi: ?c.AVS_VideoInfo = if (d.src1.yuv != null) blk: {
        const rf = try hz.planarFormat(ctx, reference.videoInfo());
        d.out_alpha = rf.has_alpha;
        break :blk hz.withFormat(reference.videoInfo().*, .{ .family = .rgb, .sample = .f32, .bits = 32, .color_planes = 3, .has_alpha = rf.has_alpha, .ssw = 0, .ssh = 0 });
    } else null;
    const out = try F.create(ctx.env, reference, d, F.getFrame(getFrame), .{ .mt_mode = .nice, .vi = out_vi });
    return avs.Create.returnClip(out);
}
