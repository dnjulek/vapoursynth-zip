//! avszip_CombMaskMT(clip, int "thY1", int "thY2")

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/comb_mask_mt.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_CombMaskMT";
pub const params = "c[thY1]i[thY2]i";

const Data = struct {
    format: avs.Format,
    thy1: u8 = 0,
    thy2: u8 = 0,
    thr_diff: u8 = 0,
};

const F = avs.Filter(Data, filter_name);

fn CombMaskMT(comptime same_thr: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            hz.copyAlpha(ctx.env(), dst, src, d.format); // alpha passes through
            for (0..d.format.color_planes) |plane| {
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(u8, id);
                const sp = try hz.canonical(u8, src.plane(u8, id), dp.pitch, &scratch);
                filter.process(sp.fullSlice(), dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height, d.thy1, d.thy2, d.thr_diff, same_thr);
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

    const thy1 = try ctx.int(i64, 1, "thY1", 30, math.minInt(i64), math.maxInt(i64));
    const thy2 = try ctx.int(i64, 2, "thY2", 30, math.minInt(i64), math.maxInt(i64));
    if (thy1 > 255 or thy1 < 0) return ctx.fail("thY1 value should be in range [0;255]", .{});
    if (thy2 > 255 or thy2 < 0) return ctx.fail("thY2 value should be in range [0;255]", .{});
    if (thy1 > thy2) return ctx.fail("thY1 can't be greater than thY2", .{});
    if ((@as(u32, @intCast(vi.height)) >> format.ssh) < 3)
        return ctx.fail("clip too small; every plane must be at least 3 rows tall.", .{});

    const d: Data = .{ .format = format, .thy1 = @intCast(thy1), .thy2 = @intCast(thy2), .thr_diff = @intCast(thy2 - thy1) };
    const gf = if (d.thy1 == d.thy2) F.getFrame(CombMaskMT(true).getFrame) else F.getFrame(CombMaskMT(false).getFrame);
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
