const std = @import("std");

const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;
const Image = ir.Image;
const ir = @import("../filters/image_read.zig");
const PngColor = ir.PngColor;
const loadImage = ir.loadImage;

const allocator = std.heap.c_allocator;
pub const filter_name = "ImageRead";

const Data = struct {
    vi: vs.VideoInfo = undefined,
    vi_gray: vs.VideoInfo = undefined,
    paths: [][]u8 = undefined,
};

/// The frame planes (and the separate alpha frame) as ImageRead targets.
const Dest = struct {
    dst: ZAPI.ZFrame(*vs.Frame),
    adst: ?ZAPI.ZFrame(*vs.Frame),
    num_planes: u32,

    pub fn targets(self: Dest, comptime T: type) ir.Targets {
        const width, const height, const stride = self.dst.getDimensions2(T, 0);
        var t: ir.Targets = .{ .stride = stride, .width = width, .height = height };
        for (0..self.num_planes) |p| t.p[p] = std.mem.sliceAsBytes(self.dst.getWriteSlice2(T, @intCast(p)));
        if (self.adst) |a| t.a = std.mem.sliceAsBytes(a.getWriteSlice2(T, 0));
        return t;
    }
};

fn Read(comptime alpha: bool) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                var png_color: ?PngColor = null;
                var image = loadImage(d.paths[@intCast(n)], &png_color) catch |err| {
                    const err_msg = allocator.printSentinel("{s}: Couldn't open '{s}' ({any})", .{ filter_name, d.paths[@intCast(n)], err }, 0) catch unreachable;
                    zapi.setFilterError(err_msg);
                    allocator.free(err_msg);
                    return null;
                };

                defer image.deinit(allocator);

                const dst = zapi.initZFrameFromVi(&d.vi, null);
                const adst = if (alpha) zapi.initZFrameFromVi(&d.vi_gray, null);
                ir.fill(&image, Dest{ .dst = dst, .adst = if (alpha) adst else null, .num_planes = @intCast(d.vi.format.numPlanes) });

                const props = dst.getPropertiesRW();
                props.setData("zigimg_file_path", @ptrCast(d.paths[@intCast(n)]), .Utf8, .Replace);
                props.setData("zigimg_format", @tagName(image.pixelFormat()), .Utf8, .Replace);
                props.setInt("zigimg_bits", image.pixelFormat().bitsPerChannel(), .Replace);

                if (png_color) |pc| {
                    const cp = ir.colorProps(pc);
                    props.setMatrix(if (d.vi.format.numPlanes == 1) .BT709 else .RGB);
                    props.setTransfer(@fromBackingInt(@intCast(cp.transfer)));
                    props.setPrimaries(@fromBackingInt(@intCast(cp.primaries)));
                }

                if (alpha) {
                    adst.getPropertiesRW().setRange(.FULL);
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
        const err_msg = allocator.printSentinel("{s}: Couldn't open '{s}' ({any})", .{ filter_name, d.paths[0], err }, 0) catch unreachable;
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
    const bps: i32 = ir.outputBits(pf);

    _ = zapi.queryVideoFormat(&d.vi.format, cf, st, bps, 0, 0);

    d.vi_gray = d.vi;
    d.vi_gray.format.numPlanes = 1;
    d.vi_gray.format.colorFamily = .Gray;

    if (!ir.supported(pf)) {
        const err_msg = allocator.printSentinel("{s}: Unsupported pixel format '{s}'", .{ filter_name, @tagName(pf) }, 0) catch unreachable;
        map_out.setError(err_msg);
        allocator.free(err_msg);
        return;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    transferred = true;

    const alpha = ir.hasAlpha(pf);
    const gf: vs.FilterGetFrame = if (alpha) &Read(true).getFrame else &Read(false).getFrame;

    zapi.createVideoFilter(out, filter_name, &d.vi, gf, readFree, .Unordered, null, data);
}

fn validatePaths(paths: [][]u8, map_out: anytype, image_0: Image) !void {
    const pf = image_0.pixelFormat();

    var png_color: ?PngColor = null;
    for (paths[1..]) |path| {
        var image = loadImage(path, &png_color) catch |err| {
            const err_msg = allocator.printSentinel("{s}: Couldn't open '{s}' ({any})", .{ filter_name, path, err }, 0) catch unreachable;
            map_out.setError(err_msg);
            allocator.free(err_msg);
            return error.openFile;
        };
        defer image.deinit(allocator);

        switch (ir.compare(&image_0, &image)) {
            .none => {},
            .dimensions => {
                const err_msg = allocator.printSentinel(
                    "{s}: Dimensions do not match ({}x{} != {}x{}):\n{s}\n{s}",
                    .{ filter_name, image_0.width, image_0.height, image.width, image.height, paths[0], path },
                    0,
                ) catch unreachable;
                map_out.setError(err_msg);
                allocator.free(err_msg);
                return error.dimensions;
            },
            .pixel_format => {
                const err_msg = allocator.printSentinel(
                    "{s}: Pixel formats do not match ({s} != {s}):\n{s}\n{s}",
                    .{ filter_name, @tagName(pf), @tagName(image.pixelFormat()), paths[0], path },
                    0,
                ) catch unreachable;
                map_out.setError(err_msg);
                allocator.free(err_msg);
                return error.pixelFormat;
            },
        }
    }
}
