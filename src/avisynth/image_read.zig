//! vszip_ImageRead(string[] path, bool "validate")
//!
//! Source filter (one frame per path, 30 fps) decoding with zigimg; paths may
//! be http(s) URLs. Output formats: Y8/Y16 (gray), RGBP/RGBP16 (RGB),
//! RGBAP/RGBAP16/RGBAPS when the image has alpha (native alpha plane), with
//! 1/2/4-bit and indexed images expanded to 8 bit like vszip.
//!
//! Differences from vszip (AviSynth has no planar gray+alpha format): a
//! gray+alpha image is returned as Y with its alpha as a separate Y frame in
//! the `_Alpha` frame property, which is vszip's representation for every
//! alpha image. RGB(A) alpha goes to the native A plane instead.
//!
//! Props: zigimg_file_path, zigimg_format, zigimg_bits, and from PNG color
//! chunks _Matrix/_Transfer/_Primaries (same mapping as vszip).

const std = @import("std");
const avs = @import("avisynth");
const c = avs.c;
const ir = @import("../filters/image_read.zig");
const Image = ir.Image;
const PngColor = ir.PngColor;
const loadImage = ir.loadImage;

const hz = @import("helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_ImageRead";
pub const params = "[path]s*[validate]b";

const Layout = enum {
    gray,
    gray_alpha, // Y + _Alpha frame prop
    rgb,
    rgba, // native RGBAP

    fn hasAlpha(self: Layout) bool {
        return self == .gray_alpha or self == .rgba;
    }
};

const Data = struct {
    format: avs.Format,
    vi_gray: c.AVS_VideoInfo,
    layout: Layout,
    paths: [][:0]u8,

    pub fn deinit(self: *Data) void {
        for (self.paths) |p| allocator.free(p);
        allocator.free(self.paths);
    }
};

const F = avs.Filter(Data, filter_name);

/// The frame's planes (plus the `_Alpha` frame for gray+alpha) as targets.
const Dest = struct {
    frame: avs.Frame,
    alpha_frame: ?avs.Frame,
    layout: Layout,

    pub fn targets(self: Dest, comptime T: type) ir.Targets {
        var t: ir.Targets = .{};
        const ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };
        const gray = self.layout == .gray or self.layout == .gray_alpha;
        const p0 = self.frame.planeMut(T, if (gray) c.AVS_PLANAR_Y else ids[0]);
        t.stride = p0.stride();
        t.width = p0.width;
        t.height = p0.height;
        t.p[0] = std.mem.sliceAsBytes(p0.fullSlice());
        if (!gray) {
            for (1..3) |i| {
                const p = self.frame.planeMut(T, ids[i]);
                std.debug.assert(p.pitch == p0.pitch);
                t.p[i] = std.mem.sliceAsBytes(p.fullSlice());
            }
        }
        if (self.layout == .rgba) {
            const a = self.frame.planeMut(T, c.AVS_PLANAR_A);
            std.debug.assert(a.pitch == p0.pitch);
            t.a = std.mem.sliceAsBytes(a.fullSlice());
        } else if (self.alpha_frame) |af| {
            const a = af.planeMut(T, c.AVS_PLANAR_Y);
            std.debug.assert(a.pitch == p0.pitch);
            t.a = std.mem.sliceAsBytes(a.fullSlice());
        }
        return t;
    }
};

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const path = d.paths[@intCast(n)];
    var png_color: ?PngColor = null;
    var image = loadImage(path, &png_color) catch |err|
        return ctx.fail("Couldn't open '{s}' ({any})", .{ path, err });
    defer image.deinit(allocator);
    if (image.width != ctx.vi().width or image.height != ctx.vi().height)
        return ctx.fail("'{s}' is {d}x{d}, the clip is {d}x{d} (use validate=true to check at load time).", .{ path, image.width, image.height, ctx.vi().width, ctx.vi().height });

    const dst = try ctx.newFrame(null);
    errdefer dst.release();
    const alpha_frame: ?avs.Frame = if (d.layout == .gray_alpha) try ctx.newFrameVi(&d.vi_gray, null) else null;
    defer if (alpha_frame) |a| a.release();
    ir.fill(&image, Dest{ .frame = dst, .alpha_frame = alpha_frame, .layout = d.layout });

    const env = ctx.env();
    const props = dst.propsRW(env);
    props.setData("zigimg_file_path", path, .replace) catch {};
    props.setData("zigimg_format", @tagName(image.pixelFormat()), .replace) catch {};
    props.setInt("zigimg_bits", image.pixelFormat().bitsPerChannel(), .replace) catch {};
    if (png_color) |pc| {
        const cp = ir.colorProps(pc);
        props.setInt("_Matrix", if (d.format.family == .gray) 1 else 0, .replace) catch {};
        props.setInt("_Transfer", cp.transfer, .replace) catch {};
        props.setInt("_Primaries", cp.primaries, .replace) catch {};
    }
    if (alpha_frame) |a| {
        hz.setRange(env, a, .FULL);
        props.setFrame("_Alpha", a.ptr, .replace) catch return ctx.fail("could not attach _Alpha.", .{});
    }
    return dst;
}

fn validatePaths(ctx: *avs.Create, paths: []const [:0]u8, image_0: Image) !void {
    const pf = image_0.pixelFormat();
    var png_color: ?PngColor = null;
    for (paths[1..]) |path| {
        var image = loadImage(path, &png_color) catch |err| return ctx.fail("Couldn't open '{s}' ({any})", .{ path, err });
        defer image.deinit(allocator);
        switch (ir.compare(&image_0, &image)) {
            .none => {},
            .dimensions => return ctx.fail("Dimensions do not match ({d}x{d} != {d}x{d}):\n{s}\n{s}", .{ image_0.width, image_0.height, image.width, image.height, paths[0], path }),
            .pixel_format => return ctx.fail("Pixel formats do not match ({s} != {s}):\n{s}\n{s}", .{ @tagName(pf), @tagName(image.pixelFormat()), paths[0], path }),
        }
    }
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const pv = ctx.arg(0);
    if (!pv.isDefined() or pv.len() == 0) return ctx.fail("path is required.", .{});
    const paths = try allocator.alloc([:0]u8, pv.len());
    var filled: usize = 0;
    errdefer {
        for (paths[0..filled]) |p| allocator.free(p);
        allocator.free(paths);
    }
    for (0..pv.len()) |i| {
        const e = pv.at(i);
        if (!e.isString()) return ctx.fail("path[{d}] must be a string.", .{i});
        paths[i] = try allocator.dupeSentinel(u8, e.asString(), 0);
        filled += 1;
    }

    var png_color: ?PngColor = null;
    var image_0 = loadImage(paths[0], &png_color) catch |err| return ctx.fail("Couldn't open '{s}' ({any})", .{ paths[0], err });
    defer image_0.deinit(allocator);

    if ((try ctx.boolean(1, "validate", false)) and paths.len > 1) try validatePaths(ctx, paths, image_0);

    const pf = image_0.pixelFormat();
    if (!ir.supported(pf)) return ctx.fail("Unsupported pixel format '{s}'", .{@tagName(pf)});
    const pi = pf.info();
    const bits = ir.outputBits(pf);
    const alpha = ir.hasAlpha(pf);
    const layout: Layout = if (pf.isGrayscale()) (if (alpha) .gray_alpha else .gray) else (if (alpha) .rgba else .rgb);

    const format: avs.Format = .{
        .family = if (layout == .gray or layout == .gray_alpha) .gray else .rgb,
        .sample = if (pi.variant == .float) .f32 else if (bits == 8) .u8 else .u16,
        .bits = if (pi.variant == .float) 32 else bits,
        .color_planes = if (layout == .gray or layout == .gray_alpha) 1 else 3,
        .has_alpha = layout == .rgba,
        .ssw = 0,
        .ssh = 0,
    };
    var vi = std.mem.zeroes(c.AVS_VideoInfo);
    vi.width = @intCast(image_0.width);
    vi.height = @intCast(image_0.height);
    vi.fps_numerator = 30;
    vi.fps_denominator = 1;
    vi.num_frames = @intCast(paths.len);
    vi.pixel_type = avs.format.pixelType(format) catch return ctx.fail("unsupported image layout.", .{});
    const vi_gray = hz.withFormat(vi, hz.grayOf(format));

    const d: Data = .{ .format = format, .vi_gray = vi_gray, .layout = layout, .paths = paths };
    const out = try F.create(ctx.env, null, d, F.getFrame(getFrame), .{ .mt_mode = .nice, .vi = vi });
    return avs.Create.returnClip(out);
}
