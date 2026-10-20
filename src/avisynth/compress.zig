//! vszip_Compress(clip, int "codec", int "qscale", int "quality", int "dc_prec", bool "chroma")
//!
//! 8-bit Gray/YUV: MPEG-2 (codec=0) or JPEG (codec=1) 8x8 DCT quantization
//! round trip. Alpha (YUVA) passes through.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/compress.zig");
const hz = @import("helper.zig");

const Codec = filter.Codec;
pub const filter_name = "vszip_Compress";
pub const params = "c[codec]i[qscale]i[quality]i[dc_prec]i[chroma]b";

const Data = struct {
    format: avs.Format,
    qt: filter.QuantTables = .{},
    process: [4]bool = .{ true, true, true, false },
};

const F = avs.Filter(Data, filter_name);

fn Compress(comptime codec: Codec) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.process);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.color_planes) |plane| {
                if (!d.process[plane]) continue;
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(u8, id);
                const sp = try hz.canonical(u8, src.plane(u8, id), dp.pitch, &scratch);
                filter.processPlane(codec, sp.fullSlice(), dp.fullSlice(), dp.width, dp.height, dp.stride(), &d.qt, plane != 0);
            }
            return dst;
        }
    };
}

fn intArg(ctx: *avs.Create, i: usize, comptime key: []const u8, default: i64) !i64 {
    return ctx.int(i64, i, key, default, math.minInt(i64), math.maxInt(i64));
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    if (format.sample != .u8 or format.family == .rgb)
        return ctx.fail("only 8-bit integer Gray or YUV formats are supported.", .{});

    var d: Data = .{ .format = format };
    const codec_i = try intArg(ctx, 1, "codec", 0);
    if (codec_i < 0 or codec_i > 1) return ctx.fail("codec must be 0 (mpeg2) or 1 (jpeg).", .{});
    const codec: Codec = if (codec_i == 0) .mpeg2 else .jpeg;

    switch (codec) {
        .mpeg2 => {
            const qscale = try intArg(ctx, 2, "qscale", 8);
            if (qscale < 1 or qscale > 31) return ctx.fail("qscale must be between 1 and 31.", .{});
            const dc_prec = try intArg(ctx, 4, "dc_prec", 0);
            if (dc_prec < 0 or dc_prec > 3) return ctx.fail("dc_prec must be between 0 and 3.", .{});
            d.qt.buildMpeg2(@intCast(qscale), @intCast(dc_prec));
        },
        .jpeg => {
            const quality = try intArg(ctx, 3, "quality", 50);
            if (quality < 1 or quality > 100) return ctx.fail("quality must be between 1 and 100.", .{});
            d.qt.buildJpeg(@intCast(quality));
        },
    }

    const chroma = try ctx.boolean(5, "chroma", true);
    d.process = .{ true, chroma, chroma, false };

    const gf = switch (codec) {
        inline else => |cd| F.getFrame(Compress(cd).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
