//! ImageRead core shared by the VapourSynth and AviSynth adapters: loading
//! (files and http(s) URLs) and decoding with zigimg, PNG color chunks to
//! H.273 code points, and copying decoded pixels into planar targets.

const std = @import("std");
const zigimg = @import("zigimg");
const common = @import("../common.zig");

pub const Image = zigimg.Image;
const png = zigimg.formats.png;
const allocator = std.heap.c_allocator;

pub const ImgFormat = enum {
    Rgb,
    Rgba,
    Gray,
    GrayAlpha,

    pub fn isGray(self: *const ImgFormat) bool {
        return self.* == .GrayAlpha or self.* == .Gray;
    }
};

/// Pixel formats ImageRead can output; the others are rejected at create.
pub fn supported(pf: zigimg.PixelFormat) bool {
    return switch (pf) {
        .invalid, .bgr555, .rgb332, .rgb555, .rgb565, .sega_grb333, .sega_bgr333, .sega_bgr222, .sega_bgr444 => false,
        else => true,
    };
}

/// The image carries alpha (indexed images always get a palette alpha plane).
pub fn hasAlpha(pf: zigimg.PixelFormat) bool {
    const pi = pf.info();
    return (pi.channel_count == 4) or (pi.channel_count == 2) or pf.isIndexed();
}

/// Output bit depth (1/2/4-bit and indexed images expand to 8 bit).
pub fn outputBits(pf: zigimg.PixelFormat) u8 {
    return @intCast(@max(pf.info().bits_per_channel, 8));
}

/// The output layout a supported pixel format decodes to. A clip's format is
/// fixed by its first image, so every later frame must decode to the same
/// layout (checked per frame: `validate=true` only checks eagerly, and files
/// can change afterwards). Formats differing only in storage (rgb24/bgr24,
/// gray1..gray8) share a layout.
pub const Layout = struct {
    gray: bool,
    bits: u8,
    float: bool,
    alpha: bool,

    pub fn of(pf: zigimg.PixelFormat) Layout {
        return .{
            .gray = pf.isGrayscale(),
            .bits = outputBits(pf),
            .float = pf.info().variant == .float,
            .alpha = hasAlpha(pf),
        };
    }

    pub fn eql(a: Layout, b: Layout) bool {
        return a.gray == b.gray and a.bits == b.bits and a.float == b.float and a.alpha == b.alpha;
    }
};

/// Why a later image cannot be written into the clip's frames, if it can't.
pub fn frameMismatch(image: *const Image, width: u32, height: u32, layout: Layout) ?Mismatch {
    if (image.width != width or image.height != height) return .dimensions;
    const pf = image.pixelFormat();
    if (!supported(pf) or !Layout.of(pf).eql(layout)) return .pixel_format;
    return null;
}

fn scaleSample(comptime T: type, value: anytype) T {
    if (@TypeOf(value) == T) return value;
    return zigimg.color.ScaleValue(T)(value);
}

/// Planar destination: byte views of the color planes (one for gray) and
/// the optional alpha plane, all with the same element stride.
pub const Targets = struct {
    p: [3][]u8 = .{ &.{}, &.{}, &.{} },
    a: ?[]u8 = null,
    stride: usize = 0,
    width: u32 = 0,
    height: u32 = 0,
};

fn slice(comptime T: type, s: []u8) []T {
    const p: [*]T = @ptrCast(@alignCast(s.ptr));
    return p[0 .. s.len / @sizeOf(T)];
}

fn copyPixels(comptime T: type, src: anytype, t: Targets, comptime f: ImgFormat) void {
    const alpha = comptime (f == .Rgba or f == .GrayAlpha);
    const dst_0 = slice(T, t.p[0]);
    const dst_1 = if (!(comptime f.isGray())) slice(T, t.p[1]);
    const dst_2 = if (!(comptime f.isGray())) slice(T, t.p[2]);
    const dst_a: ?[]T = if (alpha) (if (t.a) |a| slice(T, a) else null) else null;
    for (0..t.height) |y| {
        for (0..t.width) |x| {
            const src_index = y * t.width + x;
            const dst_index = y * t.stride + x;
            if (comptime f.isGray()) {
                dst_0[dst_index] = scaleSample(T, src[src_index].value);
            } else {
                dst_0[dst_index] = src[src_index].r;
                dst_1[dst_index] = src[src_index].g;
                dst_2[dst_index] = src[src_index].b;
            }
            if (alpha) {
                if (dst_a) |da| da[dst_index] = switch (f) {
                    .Rgba => src[src_index].a,
                    .GrayAlpha => src[src_index].alpha,
                    else => unreachable,
                };
            }
        }
    }
}

fn copyPixelsIndexed(comptime T: type, src: anytype, t: Targets) void {
    const dst_0 = slice(T, t.p[0]);
    const dst_1 = slice(T, t.p[1]);
    const dst_2 = slice(T, t.p[2]);
    const dst_a: ?[]T = if (t.a) |a| slice(T, a) else null;
    for (0..t.height) |y| {
        for (0..t.width) |x| {
            const src_index = src.indices[y * t.width + x];
            const dst_index = y * t.stride + x;
            dst_0[dst_index] = src.palette[src_index].r;
            dst_1[dst_index] = src.palette[src_index].g;
            dst_2[dst_index] = src.palette[src_index].b;
            if (dst_a) |da| da[dst_index] = src.palette[src_index].a;
        }
    }
}

/// Copy `image` into the planes `dest.targets(T)` returns for the image's
/// sample type T (u8, u16 or f32). Alpha is written when the image has it
/// and the targets provide an alpha plane.
pub fn fill(image: *const Image, dest: anytype) void {
    switch (image.pixels) {
        inline .grayscale1, .grayscale2, .grayscale4, .grayscale8 => |src| copyPixels(u8, src, dest.targets(u8), .Gray),
        .grayscale16 => |src| copyPixels(u16, src, dest.targets(u16), .Gray),
        .grayscale8Alpha => |src| copyPixels(u8, src, dest.targets(u8), .GrayAlpha),
        .grayscale16Alpha => |src| copyPixels(u16, src, dest.targets(u16), .GrayAlpha),
        inline .rgb24, .bgr24 => |src| copyPixels(u8, src, dest.targets(u8), .Rgb),
        inline .rgba32, .bgra32 => |src| copyPixels(u8, src, dest.targets(u8), .Rgba),
        .rgb48 => |src| copyPixels(u16, src, dest.targets(u16), .Rgb),
        .rgba64 => |src| copyPixels(u16, src, dest.targets(u16), .Rgba),
        .float32 => |src| copyPixels(f32, src, dest.targets(f32), .Rgba),
        inline .indexed1, .indexed2, .indexed4, .indexed8 => |src| copyPixelsIndexed(u8, src, dest.targets(u8)),
        .indexed16 => |src| copyPixelsIndexed(u16, src, dest.targets(u16)),
        .invalid, .bgr555, .rgb332, .rgb555, .rgb565, .sega_grb333, .sega_bgr333, .sega_bgr222, .sega_bgr444 => unreachable,
    }
}

// --------------------------------------------------------------------------
// PNG color chunks
// --------------------------------------------------------------------------

const cicp_id: u32 = std.mem.bigToNative(u32, std.mem.bytesToValue(u32, "cICP"));

pub const PngColor = struct {
    gama: ?u32 = null,
    srgb: bool = false,
    chrm: ?[8]u32 = null,
    cicp: ?[4]u8 = null,
};

const PngColorProcessor = struct {
    color: PngColor = .{},

    fn processor(self: *PngColorProcessor, id: u32) png.ReaderProcessor {
        return png.ReaderProcessor.init(id, self, processChunk, null, null);
    }

    fn processChunk(self: *PngColorProcessor, data: *png.ChunkProcessData) Image.ReadError!zigimg.PixelFormat {
        const len = data.chunk_length;
        var reader = data.read_stream.reader();

        if (data.chunk_id == png.Chunks.gAMA.id and len == 4) {
            self.color.gama = try reader.takeInt(u32, .big);
        } else if (data.chunk_id == png.Chunks.sRGB.id and len == 1) {
            _ = try reader.takeByte();
            self.color.srgb = true;
        } else if (data.chunk_id == png.Chunks.cHRM.id and len == 32) {
            var vals: [8]u32 = undefined;
            for (&vals) |*v| v.* = try reader.takeInt(u32, .big);
            self.color.chrm = vals;
        } else if (data.chunk_id == cicp_id and len == 4) {
            var vals: [4]u8 = undefined;
            for (&vals) |*v| v.* = try reader.takeByte();
            self.color.cicp = vals;
        } else {
            try data.read_stream.seekBy(len);
        }

        try data.read_stream.seekBy(@sizeOf(u32));
        return data.current_format;
    }
};

// H.273 code points (the numbering of both hosts' frame props).
const transfer_bt470m = 4;
const transfer_bt470bg = 5;
const transfer_linear = 8;
const transfer_srgb = 13;
const unspecified = 2;
const primaries_bt709 = 1;

/// `_Transfer` / `_Primaries` for a PNG (sRGB / BT.709 by default).
pub const ColorProps = struct {
    transfer: u8 = transfer_srgb,
    primaries: u8 = primaries_bt709,
};

fn near(a: u32, b: u32, tol: u32) bool {
    return @max(a, b) - @min(a, b) <= tol;
}

fn matchChrm(chrm: [8]u32) u8 {
    const candidates = [_]struct { [8]u32, u8 }{
        .{ .{ 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000 }, 1 }, // BT709
        .{ .{ 31270, 32900, 70800, 29200, 17000, 79700, 13100, 4600 }, 9 }, // BT2020
        .{ .{ 31270, 32900, 68000, 32000, 26500, 69000, 15000, 6000 }, 12 }, // ST432_1
        .{ .{ 31400, 35100, 68000, 32000, 26500, 69000, 15000, 6000 }, 11 }, // ST431_2
        .{ .{ 31270, 32900, 63000, 34000, 31000, 59500, 15500, 7000 }, 6 }, // ST170_M
    };

    outer: for (candidates) |cand| {
        for (cand[0], chrm) |ref, val| {
            if (!near(ref, val, 1000)) continue :outer;
        }
        return cand[1];
    }

    return unspecified;
}

/// cICP code points vszip accepts: exactly the members of VapourSynth's
/// ColorPrimaries / TransferCharacteristics enums (others keep the default).
pub fn validPrimaries(v: u8) bool {
    return switch (v) {
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22 => true,
        else => false,
    };
}

pub fn validTransfer(v: u8) bool {
    return switch (v) {
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18 => true,
        else => false,
    };
}

pub fn colorProps(c: PngColor) ColorProps {
    var out: ColorProps = .{};

    if (c.cicp) |ci| {
        if (validPrimaries(ci[0])) out.primaries = ci[0];
        if (validTransfer(ci[1])) out.transfer = ci[1];
        return out;
    }

    if (c.srgb) return out;

    if (c.gama) |g| {
        out.transfer = if (near(g, 100000, 1000))
            transfer_linear
        else if (near(g, 45455, 1000))
            transfer_bt470m
        else if (near(g, 35714, 1000))
            transfer_bt470bg
        else
            unspecified;
    }

    if (c.chrm) |chrm| out.primaries = matchChrm(chrm);

    return out;
}

// --------------------------------------------------------------------------
// Loading
// --------------------------------------------------------------------------

const LoadResult = struct {
    path: []const u8,
    img: Image = undefined,
    color: ?PngColor = null,
    err: ?anyerror = null,
};

fn isUrl(path: []const u8) bool {
    return std.ascii.startsWithIgnoreCase(path, "http://") or
        std.ascii.startsWithIgnoreCase(path, "https://");
}

fn readSource(path: []const u8) ![]u8 {
    if (isUrl(path)) {
        var client: std.http.Client = .{ .allocator = allocator, .io = common.io };
        defer client.deinit();

        var aw: std.Io.Writer.Allocating = .init(allocator);
        defer aw.deinit();

        const res = try client.fetch(.{
            .location = .{ .url = path },
            .response_writer = &aw.writer,
        });
        if (res.status != .ok) return error.HttpRequestFailed;

        return aw.toOwnedSlice();
    }

    return std.Io.Dir.cwd().readFileAlloc(common.io, path, allocator, .unlimited);
}

fn decodeImage(bytes: []const u8, color_out: *?PngColor) !Image {
    if (!std.mem.startsWith(u8, bytes, png.magic_header)) {
        return Image.fromMemory(allocator, bytes);
    }

    var stream = zigimg.io.ReadStream.initMemory(bytes);
    var trns: png.TrnsProcessor = .{};
    var cp: PngColorProcessor = .{};
    var processors = [_]png.ReaderProcessor{
        trns.processor(),
        cp.processor(png.Chunks.gAMA.id),
        cp.processor(png.Chunks.sRGB.id),
        cp.processor(png.Chunks.cHRM.id),
        cp.processor(cicp_id),
    };

    const img = try png.load(&stream, allocator, .{ .processors = processors[0..] });
    color_out.* = cp.color;
    return img;
}

fn loadImageThread(result: *LoadResult) void {
    const bytes = readSource(result.path) catch |err| {
        result.err = err;
        return;
    };
    defer allocator.free(bytes);

    result.img = decodeImage(bytes, &result.color) catch |err| {
        result.err = err;
        return;
    };
}

/// Load and decode `path` on a thread with a large stack (zigimg decoders
/// recurse deeply; host worker threads may have small stacks). PNG color
/// chunks go to `color_out` (null for other formats). Free with
/// `image.deinit(std.heap.c_allocator)`.
pub fn loadImage(path: []const u8, color_out: *?PngColor) !Image {
    var result = LoadResult{ .path = path };
    const thread = try std.Thread.spawn(.{ .stack_size = 8 * 1024 * 1024 }, loadImageThread, .{&result});
    thread.join();
    if (result.err) |err| return err;
    color_out.* = result.color;
    return result.img;
}

/// What `validate=true` rejects in a later path, against the first image.
pub const Mismatch = enum { none, dimensions, pixel_format };

pub fn compare(image_0: *const Image, image: *const Image) Mismatch {
    if (image_0.width != image.width or image_0.height != image.height) return .dimensions;
    if (image_0.pixelFormat() != image.pixelFormat()) return .pixel_format;
    return .none;
}
