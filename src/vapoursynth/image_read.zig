const std = @import("std");

const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const Image = vszip.zigimg.Image;

const allocator = std.heap.c_allocator;
pub const filter_name = "ImageRead";

const Data = struct {
    vi: vs.VideoInfo = undefined,
    paths: [][]const u8 = undefined,
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

fn Read(comptime alpha: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) ?*const vs.Frame {
            _ = frame_data;
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi);

            if (activation_reason == .Initial) {
                var image = Image.fromFilePath(allocator, d.paths[@intCast(n)]) catch unreachable;
                defer image.deinit();

                const dst = zapi.initZFrameFromVi(&d.vi, frame_ctx, core, .{});
                const adst = if (alpha) zapi.initZFrameFromVi(&d.vi, frame_ctx, core, .{ .cf = .Gray });

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

                    .invalid, .bgr555, .rgb332, .rgb555, .rgb565 => unreachable,
                }

                const props = dst.getPropertiesRW();
                props.setData("zigimg_file_path", @ptrCast(d.paths[@intCast(n)]), .Utf8, .Replace);
                props.setData("zigimg_format", @tagName(image.pixelFormat()), .Utf8, .Replace);
                props.setInt("zigimg_bits", image.pixelFormat().bitsPerChannel(), .Replace);

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

fn readFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = vsapi;
    _ = core;
    const d: *Data = @ptrCast(@alignCast(instance_data));
    allocator.free(d.paths);
    allocator.destroy(d);
}

pub fn readCreate(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.C) void {
    _ = user_data;
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.paths = map_in.getDataArray("path", allocator).?;

    var image_0 = Image.fromFilePath(allocator, d.paths[0]) catch |err| {
        const err_msg = std.fmt.allocPrintZ(allocator, "{s}: Couldn't open '{s}' ({any})", .{ filter_name, d.paths[0], err }) catch unreachable;
        map_out.setError(err_msg);
        allocator.free(err_msg);
        return;
    };
    defer image_0.deinit();

    const pf = image_0.pixelFormat();

    for (d.paths[1..]) |path| {
        var image = Image.fromFilePath(allocator, path) catch |err| {
            const err_msg = std.fmt.allocPrintZ(allocator, "{s}: Couldn't open '{s}' ({any})", .{ filter_name, path, err }) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return;
        };

        if (image_0.width != image.width or image_0.height != image.height) {
            const err_msg = std.fmt.allocPrintZ(
                allocator,
                "{s}: Dimensions do not match ({}x{} != {}x{}):\n{s}\n{s}",
                .{ filter_name, image_0.width, image_0.height, image.width, image.height, d.paths[0], path },
            ) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return;
        }

        const pf2 = image.pixelFormat();
        if (pf != pf2) {
            const err_msg = std.fmt.allocPrintZ(
                allocator,
                "{s}: Pixel formats do not match ({s} != {s}):\n{s}\n{s}",
                .{ filter_name, @tagName(pf), @tagName(pf2), d.paths[0], path },
            ) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return;
        }

        image.deinit();
    }

    d.vi = .{
        .format = .{},
        .fpsNum = 30,
        .fpsDen = 1,
        .width = @intCast(image_0.width),
        .height = @intCast(image_0.height),
        .numFrames = @intCast(d.paths.len),
    };

    const pi = pf.info();
    const cf: vs.ColorFamily = if (pf.isGrayscale()) .Gray else .RGB;
    const st: vs.SampleType = if (pi.variant == .float) .Float else .Integer;
    const bps: i32 = @max(pi.bits_per_channel, 8);

    _ = zapi.queryVideoFormat(&d.vi.format, cf, st, bps, 0, 0, core);

    switch (pf) {
        .grayscale1, .grayscale2, .grayscale4, .grayscale8, .grayscale16, .grayscale8Alpha, .grayscale16Alpha, .rgb24, .rgba32, .bgr24, .bgra32, .rgb48, .rgba64, .indexed1, .indexed2, .indexed4, .indexed8, .indexed16, .float32 => {},
        .invalid, .bgr555, .rgb332, .rgb555, .rgb565 => |f| {
            const err_msg = std.fmt.allocPrintZ(allocator, "{s}: Unsupported pixel format '{s}'", .{ filter_name, @tagName(f) }) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return;
        },
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const alpha = (pi.channel_count == 4) or (pi.channel_count == 2) or pf.isIndexed();
    const gf: vs.FilterGetFrame = if (alpha) &Read(true).getFrame else &Read(false).getFrame;

    zapi.createVideoFilter(out, filter_name, &d.vi, gf, readFree, .Unordered, null, data, core);
}
