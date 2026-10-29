//! avszip_CLAHE(clip, int "limit", int[] "tiles")
//!
//! Every color plane is equalized; the output is tagged full range
//! (`_ColorRange` = 0) like vszip. Float is equalized on 16-bit bins.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/clahe.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_CLAHE";
pub const params = "c[limit]i[tiles]i*";

const Data = struct {
    format: avs.Format,
    limit: u32 = 0,
    tiles: [2]u32 = .{ 0, 0 },
    bits: u5 = 8, // histogram bits: the integer depth, 16 for float
};

const F = avs.Filter(Data, filter_name);

/// `clamp`: 10..14-bit integer clips bin min(v, peak) so stray samples above
/// the nominal peak stay inside their LUT row (8/16-bit need no guard).
fn CLAHE(comptime T: type, comptime clamp: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyAlpha(ctx.env(), dst, src, d.format);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.color_planes) |plane| {
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(T, id);
                const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch);
                try filter.applyCLAHE(T, clamp, sp.fullSlice(), dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height, d.limit, &d.tiles, d.bits, d.format.isChroma(plane));
            }
            hz.setRange(ctx.env(), dst, .FULL);
            return dst;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    var d: Data = .{ .format = format };
    d.bits = if (format.isFloat()) 16 else @intCast(format.bits);
    d.limit = try arg.int(u32, "limit", 7, 0, math.maxInt(u32));

    var tbuf: [2]i64 = undefined;
    const tiles_arr = (try arg.intArray(i64, "tiles", &tbuf, math.minInt(i64), math.maxInt(i64))) orelse blk: {
        tbuf = .{ 3, 3 };
        break :blk tbuf[0..2];
    };
    if (tiles_arr.len < 1) return ctx.fail("tiles array can't have more than 2 values.", .{});
    for (tiles_arr) |t| {
        if (t < 1) return ctx.fail("tiles values must be >= 1.", .{});
        if (t > math.maxInt(u32)) return ctx.fail("tiles must not exceed the (chroma) plane width/height.", .{});
    }
    d.tiles[0] = @intCast(tiles_arr[0]);
    d.tiles[1] = @intCast(if (tiles_arr.len == 2) tiles_arr[1] else tiles_arr[0]);

    const min_w: u32 = @as(u32, @intCast(vi.width)) >> format.ssw;
    const min_h: u32 = @as(u32, @intCast(vi.height)) >> format.ssh;
    if (d.tiles[0] > min_w or d.tiles[1] > min_h)
        return ctx.fail("tiles must not exceed the (chroma) plane width/height.", .{});

    const hist_size: u64 = math.shl(u64, 1, d.bits);
    const tw: u64 = @as(u64, @intCast(vi.width)) / @as(u64, d.tiles[0]);
    const th: u64 = @as(u64, @intCast(vi.height)) / @as(u64, d.tiles[1]);
    const cl: u64 = @as(u64, d.limit) * tw * th / hist_size;
    if (cl > math.maxInt(i32)) return ctx.fail("limit too large for this frame size; reduce limit or increase tiles.", .{});

    const gf = switch (format.sample) {
        .u8 => F.getFrame(CLAHE(u8, false).getFrame),
        .u16 => if (d.bits == 16) F.getFrame(CLAHE(u16, false).getFrame) else F.getFrame(CLAHE(u16, true).getFrame),
        .f32 => F.getFrame(CLAHE(f32, false).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
