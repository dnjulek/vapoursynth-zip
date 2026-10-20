//! vszip_Checkmate(clip, int "thr", int "tmax", int "tthr2")
//!
//! Temporal: frame n reads n-1 and n+1 (and n-2/n+2 when tthr2 > 0), clamped
//! to the clip, exactly like vszip. Results depend only on n, never on request
//! order, so MT_NICE is safe.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/checkmate.zig");
const hz = @import("helper.zig");

pub const filter_name = "vszip_Checkmate";
pub const params = "c[thr]i[tmax]i[tthr2]i";

const Data = struct {
    format: avs.Format,
    num_frames: c_int,
    p: filter.Params = undefined,
};

const F = avs.Filter(Data, filter_name);

fn Checkmate(comptime use_tthr2: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const last = d.num_frames - 1;
            const src_p1 = try ctx.childFrame(@max(0, n - 1));
            defer src_p1.release();
            const src = try ctx.childFrame(n);
            defer src.release();
            const src_n1 = try ctx.childFrame(@min(n + 1, last));
            defer src_n1.release();
            const src_p2 = if (use_tthr2) try ctx.childFrame(@max(0, n - 2));
            defer if (use_tthr2) src_p2.release();
            const src_n2 = if (use_tthr2) try ctx.childFrame(@min(n + 2, last));
            defer if (use_tthr2) src_n2.release();

            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.numPlanes()) |plane| {
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(u8, id);
                const pitch = dp.pitch;
                const dstp = dp.fullSlice();
                const srcp = (try hz.canonical(u8, src.plane(u8, id), pitch, &scratch)).fullSlice();
                const srcp_p1 = (try hz.canonical(u8, src_p1.plane(u8, id), pitch, &scratch)).fullSlice();
                const srcp_n1 = (try hz.canonical(u8, src_n1.plane(u8, id), pitch, &scratch)).fullSlice();
                const srcp_p2 = if (use_tthr2) (try hz.canonical(u8, src_p2.plane(u8, id), pitch, &scratch)).fullSlice();
                const srcp_n2 = if (use_tthr2) (try hz.canonical(u8, src_n2.plane(u8, id), pitch, &scratch)).fullSlice();

                filter.processPlane(use_tthr2, dstp, srcp_p2, srcp_p1, srcp, srcp_n1, srcp_n2, @intCast(dp.stride()), dp.width, dp.height, d.p);
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

    const big = math.maxInt(i32);
    var d: Data = .{ .format = format, .num_frames = vi.num_frames };
    d.p = filter.Params.init(
        try ctx.int(i32, 1, "thr", 12, -big, big),
        try ctx.int(i32, 2, "tmax", 12, -big, big),
        try ctx.int(i32, 3, "tthr2", 0, -big, big),
        @as(u32, @intCast(vi.width)) >> format.ssw,
        @as(u32, @intCast(vi.height)) >> format.ssh,
    ) catch |e| return ctx.fail("{s}", .{filter.Params.message(e)});

    const gf = if (d.p.tthr2 > 0) F.getFrame(Checkmate(true).getFrame) else F.getFrame(Checkmate(false).getFrame);
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
