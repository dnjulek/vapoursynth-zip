//! avszip_ColorMap(clip, int "color")
//!
//! Y8 in, planar RGB (RGBP) out. Props as vszip: _Matrix=0 (RGB),
//! _Transfer=13 (sRGB), _Primaries=1 (BT.709), _ColorRange=0 (full).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/color_map.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_ColorMap";
pub const params = "c[color]i";

const Data = struct {
    lut: filter.Lut,
};

const F = avs.Filter(Data, filter_name);

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const src = try ctx.childFrame(n);
    defer src.release();
    const dst = try ctx.newFrame(src);
    errdefer dst.release();
    var scratch: hz.Scratch = .{};
    defer scratch.deinit();

    const ids = [3]c_int{ c.AVS_PLANAR_R, c.AVS_PLANAR_G, c.AVS_PLANAR_B };
    var dpl: [3]avs.PlaneMut(u8) = undefined;
    for (&dpl, ids) |*p, id| p.* = dst.planeMut(u8, id);
    // Planar RGB planes of one frame share a pitch; the gray source is
    // brought to it so one row offset addresses all four planes.
    std.debug.assert(dpl[0].pitch == dpl[1].pitch and dpl[0].pitch == dpl[2].pitch);
    const sp = try hz.canonical(u8, src.plane(u8, c.AVS_PLANAR_Y), dpl[0].pitch, &scratch);
    const srcp = sp.fullSlice();
    const dstp = [3][]u8{ dpl[0].fullSlice(), dpl[1].fullSlice(), dpl[2].fullSlice() };
    filter.mapPlane(srcp, dstp, sp.width, sp.height, @intCast(sp.stride()), &d.lut);

    const env = ctx.env();
    const props = dst.propsRW(env);
    props.setInt("_Matrix", 0, .replace) catch {};
    props.setInt("_Transfer", 13, .replace) catch {};
    props.setInt("_Primaries", 1, .replace) catch {};
    props.setInt("_ColorRange", 0, .replace) catch {};
    return dst;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    if (format.family != .gray or format.sample != .u8) return ctx.fail("only Gray8 format is supported.", .{});

    const icolor = try arg.int(i64, "color", 20, math.minInt(i64), math.maxInt(i64));
    if (icolor < 0 or icolor > 21) return ctx.fail("\"color\" should be between 0 and 21.", .{});

    const d: Data = .{ .lut = filter.buildLut(@fromBackingInt(@intCast(@as(i32, @intCast(icolor))))) };

    var rgb = format;
    rgb.family = .rgb;
    rgb.color_planes = 3;
    const out = try F.create(ctx.env, clip, d, F.getFrame(getFrame), .{ .mt_mode = .nice, .vi = hz.withFormat(vi.*, rgb) });
    return avs.Create.returnClip(out);
}
