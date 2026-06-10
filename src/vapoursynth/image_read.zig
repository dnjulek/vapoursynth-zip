const std = @import("std");

const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;
const zigimg = vszip.zigimg;
const Image = zigimg.Image;
const png = zigimg.formats.png;

const allocator = std.heap.c_allocator;
pub const filter_name = "ImageRead";

const Data = struct {
    vi: vs.VideoInfo = undefined,
    vi_gray: vs.VideoInfo = undefined,
    paths: [][]u8 = undefined,
};

pub const ImgFormat = enum {
    Rgb,
    Rgba,
    Gray,
    GrayAlpha,

    pub fn isGray(self: *const ImgFormat) bool {
        return self.* == .GrayAlpha or self.* == .Gray;
    }
};

pub fn copyPixels(comptime T: type, src: anytype, dst: ZAPI.ZFrame(*vs.Frame), adst: anytype, comptime alpha: bool, comptime f: ImgFormat) void {
    const dst_0 = dst.getWriteSlice2(T, 0);
    const dst_1 = if (!(comptime f.isGray())) dst.getWriteSlice2(T, 1);
    const dst_2 = if (!(comptime f.isGray())) dst.getWriteSlice2(T, 2);
    const dst_a = if (alpha) adst.getWriteSlice2(T, 0);
    const width, const height, const stride = dst.getDimensions2(T, 0);

    var x: u32 = 0;
    while (x < width) : (x += 1) {
        var y: u32 = 0;
        while (y < height) : (y += 1) {
            const src_index = (y * width + x);
            const dst_index = (y * stride + x);

            if (comptime f.isGray()) {
                dst_0[dst_index] = src[src_index].value;
            } else {
                dst_0[dst_index] = src[src_index].r;
                dst_1[dst_index] = src[src_index].g;
                dst_2[dst_index] = src[src_index].b;
            }

            if (alpha) {
                dst_a[dst_index] = switch (f) {
                    .Rgba => src[src_index].a,
                    .GrayAlpha => src[src_index].alpha,
                    else => unreachable,
                };
            }
        }
    }
}

pub fn copyPixelsIndexed(comptime T: type, src: anytype, dst: ZAPI.ZFrame(*vs.Frame), adst: anytype, comptime alpha: bool) void {
    const dst_0 = dst.getWriteSlice2(T, 0);
    const dst_1 = dst.getWriteSlice2(T, 1);
    const dst_2 = dst.getWriteSlice2(T, 2);
    const dst_a = if (alpha) adst.getWriteSlice2(T, 0);
    const width, const height, const stride = dst.getDimensions2(T, 0);

    var x: u32 = 0;
    while (x < width) : (x += 1) {
        var y: u32 = 0;
        while (y < height) : (y += 1) {
            const src_index = src.indices[y * width + x];
            const dst_index = (y * stride + x);

            dst_0[dst_index] = src.palette[src_index].r;
            dst_1[dst_index] = src.palette[src_index].g;
            dst_2[dst_index] = src.palette[src_index].b;
            if (alpha) dst_a[dst_index] = src.palette[src_index].a;
        }
    }
}

const cicp_id: u32 = std.mem.bigToNative(u32, std.mem.bytesToValue(u32, "cICP"));

const PngColor = struct {
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

const ColorProps = struct {
    transfer: vsc.TransferCharacteristics = .IEC_61966_2_1,
    primaries: vsc.ColorPrimaries = .BT709,
};

fn near(a: u32, b: u32, tol: u32) bool {
    return @max(a, b) - @min(a, b) <= tol;
}

fn matchChrm(chrm: [8]u32) vsc.ColorPrimaries {
    const candidates = [_]struct { [8]u32, vsc.ColorPrimaries }{
        .{ .{ 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000 }, .BT709 },
        .{ .{ 31270, 32900, 70800, 29200, 17000, 79700, 13100, 4600 }, .BT2020 },
        .{ .{ 31270, 32900, 68000, 32000, 26500, 69000, 15000, 6000 }, .ST432_1 },
        .{ .{ 31400, 35100, 68000, 32000, 26500, 69000, 15000, 6000 }, .ST431_2 },
        .{ .{ 31270, 32900, 63000, 34000, 31000, 59500, 15500, 7000 }, .ST170_M },
    };

    outer: for (candidates) |cand| {
        for (cand[0], chrm) |ref, val| {
            if (!near(ref, val, 1000)) continue :outer;
        }
        return cand[1];
    }

    return .UNSPECIFIED;
}

fn colorProps(c: PngColor) ColorProps {
    var out: ColorProps = .{};

    if (c.cicp) |ci| {
        if (std.enums.fromInt(vsc.ColorPrimaries, ci[0])) |p| out.primaries = p;
        if (std.enums.fromInt(vsc.TransferCharacteristics, ci[1])) |t| out.transfer = t;
        return out;
    }

    if (c.srgb) return out;

    if (c.gama) |g| {
        out.transfer = if (near(g, 100000, 1000))
            .LINEAR
        else if (near(g, 45455, 1000))
            .BT470_M
        else if (near(g, 35714, 1000))
            .BT470_BG
        else
            .UNSPECIFIED;
    }

    if (c.chrm) |chrm| out.primaries = matchChrm(chrm);

    return out;
}

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
        var client: std.http.Client = .{ .allocator = allocator, .io = vszip.io };
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

    return std.Io.Dir.cwd().readFileAlloc(vszip.io, path, allocator, .unlimited);
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

fn loadImage(path: []const u8, color_out: *?PngColor) !Image {
    var result = LoadResult{ .path = path };
    const thread = std.Thread.spawn(.{ .stack_size = 8 * 1024 * 1024 }, loadImageThread, .{&result}) catch unreachable;
    thread.join();
    if (result.err) |err| return err;
    color_out.* = result.color;
    return result.img;
}

fn Read(comptime alpha: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                var png_color: ?PngColor = null;
                var image = loadImage(d.paths[@intCast(n)], &png_color) catch |err| {
                    const err_msg = std.fmt.allocPrintSentinel(allocator, "{s}: Couldn't open '{s}' ({any})", .{ filter_name, d.paths[@intCast(n)], err }, 0) catch unreachable;
                    zapi.setFilterError(err_msg);
                    allocator.free(err_msg);
                    return null;
                };

                defer image.deinit(allocator);

                const dst = zapi.initZFrameFromVi(&d.vi, null);
                const adst = if (alpha) zapi.initZFrameFromVi(&d.vi_gray, null);

                switch (image.pixels) {
                    .grayscale1 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Gray);
                    },
                    .grayscale2 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Gray);
                    },
                    .grayscale4 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Gray);
                    },
                    .grayscale8 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Gray);
                    },
                    .grayscale16 => |src| {
                        copyPixels(u16, src, dst, adst, alpha, .Gray);
                    },
                    .grayscale8Alpha => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .GrayAlpha);
                    },
                    .grayscale16Alpha => |src| {
                        copyPixels(u16, src, dst, adst, alpha, .GrayAlpha);
                    },
                    .rgb24 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Rgb);
                    },
                    .rgba32 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Rgba);
                    },
                    .bgr24 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Rgb);
                    },
                    .bgra32 => |src| {
                        copyPixels(u8, src, dst, adst, alpha, .Rgba);
                    },
                    .rgb48 => |src| {
                        copyPixels(u16, src, dst, adst, alpha, .Rgb);
                    },
                    .rgba64 => |src| {
                        copyPixels(u16, src, dst, adst, alpha, .Rgba);
                    },
                    .float32 => |src| {
                        copyPixels(f32, src, dst, adst, alpha, .Rgba);
                    },
                    .indexed1 => |src| {
                        copyPixelsIndexed(u8, src, dst, adst, alpha);
                    },
                    .indexed2 => |src| {
                        copyPixelsIndexed(u8, src, dst, adst, alpha);
                    },
                    .indexed4 => |src| {
                        copyPixelsIndexed(u8, src, dst, adst, alpha);
                    },
                    .indexed8 => |src| {
                        copyPixelsIndexed(u8, src, dst, adst, alpha);
                    },
                    .indexed16 => |src| {
                        copyPixelsIndexed(u16, src, dst, adst, alpha);
                    },

                    .invalid, .bgr555, .rgb332, .rgb555, .rgb565, .sega_grb333, .sega_bgr333 => unreachable,
                }

                const props = dst.getPropertiesRW();
                props.setData("zigimg_file_path", @ptrCast(d.paths[@intCast(n)]), .Utf8, .Replace);
                props.setData("zigimg_format", @tagName(image.pixelFormat()), .Utf8, .Replace);
                props.setInt("zigimg_bits", image.pixelFormat().bitsPerChannel(), .Replace);

                if (png_color) |pc| {
                    const cp = colorProps(pc);
                    props.setMatrix(if (d.vi.format.numPlanes == 1) .BT709 else .RGB);
                    props.setTransfer(cp.transfer);
                    props.setPrimaries(cp.primaries);
                }

                if (alpha) {
                    adst.getPropertiesRW().setColorRange(.FULL);
                    props.consumeAlpha(adst.frame);
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn readFree(instance_data: ?*anyopaque, _: ?*vs.Core, _: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));

    for (d.paths) |path| {
        allocator.free(path);
    }

    allocator.free(d.paths);
    allocator.destroy(d);
}

pub fn readCreate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    const paths_in = map_in.getDataArray("path", allocator).?;
    d.paths = allocator.alloc([]u8, paths_in.len) catch unreachable;

    for (paths_in, 0..) |path, i| {
        d.paths[i] = allocator.dupe(u8, path) catch unreachable;
    }

    var transferred = false;
    defer if (!transferred) {
        for (d.paths) |p| allocator.free(p);
        allocator.free(d.paths);
    };

    allocator.free(paths_in);
    var png_color: ?PngColor = null;
    var image_0 = loadImage(d.paths[0], &png_color) catch |err| {
        const err_msg = std.fmt.allocPrintSentinel(allocator, "{s}: Couldn't open '{s}' ({any})", .{ filter_name, d.paths[0], err }, 0) catch unreachable;
        map_out.setError(err_msg);
        allocator.free(err_msg);
        return;
    };
    defer image_0.deinit(allocator);

    const validate = map_in.getBool("validate") orelse false;
    if (validate and d.paths.len > 1) {
        validatePaths(d.paths, map_out, image_0) catch return;
    }

    d.vi = .{
        .format = .{},
        .fpsNum = 30,
        .fpsDen = 1,
        .width = @intCast(image_0.width),
        .height = @intCast(image_0.height),
        .numFrames = @intCast(d.paths.len),
    };

    const pf = image_0.pixelFormat();
    const pi = pf.info();
    const cf: vs.ColorFamily = if (pf.isGrayscale()) .Gray else .RGB;
    const st: vs.SampleType = if (pi.variant == .float) .Float else .Integer;
    const bps: i32 = @max(pi.bits_per_channel, 8);

    _ = zapi.queryVideoFormat(&d.vi.format, cf, st, bps, 0, 0);

    d.vi_gray = d.vi;
    d.vi_gray.format.numPlanes = 1;
    d.vi_gray.format.colorFamily = .Gray;

    switch (pf) {
        .grayscale1, .grayscale2, .grayscale4, .grayscale8, .grayscale16, .grayscale8Alpha, .grayscale16Alpha, .rgb24, .rgba32, .bgr24, .bgra32, .rgb48, .rgba64, .indexed1, .indexed2, .indexed4, .indexed8, .indexed16, .float32 => {},
        .invalid, .bgr555, .rgb332, .rgb555, .rgb565, .sega_grb333, .sega_bgr333 => |f| {
            const err_msg = std.fmt.allocPrintSentinel(allocator, "{s}: Unsupported pixel format '{s}'", .{ filter_name, @tagName(f) }, 0) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return;
        },
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    transferred = true;

    const alpha = (pi.channel_count == 4) or (pi.channel_count == 2) or pf.isIndexed();
    const gf: vs.FilterGetFrame = if (alpha) &Read(true).getFrame else &Read(false).getFrame;

    zapi.createVideoFilter(out, filter_name, &d.vi, gf, readFree, .Unordered, null, data);
}

fn validatePaths(paths: [][]u8, map_out: anytype, image_0: Image) !void {
    const pf = image_0.pixelFormat();

    var png_color: ?PngColor = null;
    for (paths[1..]) |path| {
        var image = loadImage(path, &png_color) catch |err| {
            const err_msg = std.fmt.allocPrintSentinel(allocator, "{s}: Couldn't open '{s}' ({any})", .{ filter_name, path, err }, 0) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return error.openFile;
        };

        if (image_0.width != image.width or image_0.height != image.height) {
            const err_msg = std.fmt.allocPrintSentinel(
                allocator,
                "{s}: Dimensions do not match ({}x{} != {}x{}):\n{s}\n{s}",
                .{ filter_name, image_0.width, image_0.height, image.width, image.height, paths[0], path },
                0,
            ) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return error.dimensions;
        }

        const pf2 = image.pixelFormat();
        if (pf != pf2) {
            const err_msg = std.fmt.allocPrintSentinel(
                allocator,
                "{s}: Pixel formats do not match ({s} != {s}):\n{s}\n{s}",
                .{ filter_name, @tagName(pf), @tagName(pf2), paths[0], path },
                0,
            ) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return error.pixelFormat;
        }

        image.deinit(allocator);
    }
}
