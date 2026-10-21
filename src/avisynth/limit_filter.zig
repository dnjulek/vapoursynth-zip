//! avszip_LimitFilter(clip flt, clip src, clip "ref", float[] "dark_thr", float[] "bright_thr", float[] "elast", int[] "planes")
//!
//! Thresholds are given on the 8-bit scale and rescaled to the clip's depth
//! (range read from flt's first frame), like vszip. Output props come from flt.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/limit_filter.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_LimitFilter";
pub const params = "c[src]c[ref]c[dark_thr]f*[bright_thr]f*[elast]f*[planes]i*";

const Data = struct {
    format: avs.Format,
    src: avs.OwnedClip,
    ref: avs.OwnedClip = .{},
    dark_thr: [3]f32 = @splat(1),
    bright_thr: [3]f32 = @splat(1),
    elast: [3]f32 = @splat(1),
    planes: [4]bool,

    pub fn deinit(self: *Data) void {
        self.src.deinit();
        self.ref.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn LimitFilter(comptime T: type, comptime refb: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const flt = try ctx.childFrame(n);
            defer flt.release();
            const src = try ctx.getFrame(d.src.get(), n);
            defer src.release();
            const ref = if (refb) try ctx.getFrame(d.ref.get(), n);
            defer if (refb) ref.release();
            const dst = try ctx.newFrame(flt);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, flt, d.format, &d.planes);

            for (0..d.format.color_planes) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const fp = flt.plane(T, id);
                const sp = src.plane(T, id);
                const rp = if (refb) ref.plane(T, id) else sp;
                const dp = dst.planeMut(T, id);
                // Per row: four independent pitches, and no inter-row padding
                // (uninitialized in AviSynth) enters the arithmetic.
                for (0..dp.height) |y| {
                    filter.process(T, fp.row(y), sp.row(y), rp.row(y), dp.row(y), d.dark_thr[plane], d.bright_thr[plane], d.elast[plane]);
                }
            }
            return dst;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const flt = try ctx.clip(0, "flt");
    defer flt.release();
    const src = try ctx.clip(1, "src");
    errdefer src.release();
    const ref = try ctx.optClip(2, "ref");
    errdefer if (ref) |r| r.release();

    const vi = flt.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    try hz.compareClips(ctx, &.{ vi, src.videoInfo(), if (ref) |r| r.videoInfo() else null }, .SAME_LEN, .color_only);

    var d: Data = .{
        .format = format,
        .src = .{ .clip = src },
        .ref = .{ .clip = ref },
        .planes = try ctx.planes(6, format.color_planes, true),
    };
    d.dark_thr = try ctx.perPlaneFloat(f32, 3, "dark_thr", 1, 0, 255);
    d.bright_thr = try ctx.perPlaneFloat(f32, 4, "bright_thr", 1, 0, 255);
    d.elast = try ctx.perPlaneFloat(f32, 5, "elast", 2, 0, math.maxInt(u16));

    const range = hz.getRange(ctx.env, flt, format);
    for (0..3) |i| {
        d.dark_thr[i] = hz.scaleValue(d.dark_thr[i], format, range, .{});
        d.bright_thr[i] = hz.scaleValue(d.bright_thr[i], format, range, .{});
    }

    const gf = switch (format.sample) {
        inline else => |s| if (ref != null)
            F.getFrame(LimitFilter(s.Type(), true).getFrame)
        else
            F.getFrame(LimitFilter(s.Type(), false).getFrame),
    };
    const out = try F.create(ctx.env, flt, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
