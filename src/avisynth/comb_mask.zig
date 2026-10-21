//! avszip_CombMask(clip, int "cthresh", int "mthresh", bool "expand", int "metric")
//!
//! 8-bit only. With mthresh > 0 frame n also reads frame n-1 (n=0 reads
//! itself), like vszip.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/comb_mask.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_CombMask";
pub const params = "c[cthresh]i[mthresh]i[expand]b[metric]i";

const Data = struct {
    format: avs.Format,
    cthresh: i32 = 0,
    cth6: u16 = 0,
    mthresh: u16 = 0,
};

const F = avs.Filter(Data, filter_name);

fn CombMask(comptime metric_1: bool, comptime expand: bool, comptime motion: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const prv = if (motion) try ctx.childFrame(@max(0, n - 1));
            defer if (motion) prv.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            hz.copyAlpha(ctx.env(), dst, src, d.format); // alpha passes through
            for (0..d.format.color_planes) |plane| {
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(u8, id);
                const sp = try hz.canonical(u8, src.plane(u8, id), dp.pitch, &scratch);
                const pp = if (motion) (try hz.canonical(u8, prv.plane(u8, id), dp.pitch, &scratch)).fullSlice();
                try filter.process(sp.fullSlice(), pp, dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height, d.cthresh, d.cth6, d.mthresh, metric_1, expand, motion);
            }
            return dst;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    if (format.sample != .u8) return ctx.fail("only 8 bit int format supported.", .{});

    const cthresh = try ctx.int(i64, 1, "cthresh", 6, math.minInt(i64), math.maxInt(i64));
    const mthresh = try ctx.int(i64, 2, "mthresh", 9, math.minInt(i64), math.maxInt(i64));
    const expand = try ctx.boolean(3, "expand", true);
    const metric_1 = (try ctx.int(i64, 4, "metric", 0, math.minInt(i64), math.maxInt(i64))) != 0;
    const cth_max: i32 = if (metric_1) 65025 else 255;

    if (cthresh > cth_max or cthresh < 0)
        return ctx.fail("cthresh must be between 0 and {d} when metric = {any}.", .{ cth_max, metric_1 });
    if (mthresh > 255 or mthresh < 0) return ctx.fail("mthresh must be between 0 and 255.", .{});
    if ((@as(u32, @intCast(vi.height)) >> format.ssh) < 3)
        return ctx.fail("clip too small; every plane must be at least 3 rows tall.", .{});

    const d: Data = .{
        .format = format,
        .cthresh = @intCast(cthresh),
        .cth6 = if (metric_1) 0 else @intCast(cthresh * 6),
        .mthresh = @intCast(mthresh),
    };
    const gf = switch (metric_1) {
        inline else => |m| switch (expand) {
            inline else => |e| switch (mthresh > 0) {
                inline else => |mo| F.getFrame(CombMask(m, e, mo).getFrame),
            },
        },
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
