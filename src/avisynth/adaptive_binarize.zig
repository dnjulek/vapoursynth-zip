//! vszip_AdaptiveBinarize(clip, clip clip2, int "c")
//!
//! 8-bit only: dst = 255 where (clip2 - clip) >= c, else 0; every plane.
//! Output is marked full range (`_ColorRange` = 0) like vszip.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const hz = @import("helper.zig");

pub const filter_name = "vszip_AdaptiveBinarize";
pub const params = "c[clip2]c[c]i";

const Data = struct {
    format: avs.Format,
    other: avs.OwnedClip,
    c: i16 = 0,

    pub fn deinit(self: *Data) void {
        self.other.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

const vec_len = std.simd.suggestVectorLength(i16) orelse 16;
const IV = @Vector(vec_len, i16);
const u8_0: @Vector(vec_len, u8) = @splat(0);
const u8_255: @Vector(vec_len, u8) = @splat(255);

fn binarizeRow(dstp: []u8, srcp: []const u8, srcp2: []const u8, c: i16) void {
    const w = dstp.len;
    const cv: IV = @splat(c);
    var x: usize = 0;
    while (x + vec_len <= w) : (x += vec_len) {
        const s1: IV = @intCast(@as(@Vector(vec_len, u8), srcp[x..][0..vec_len].*));
        const s2: IV = @intCast(@as(@Vector(vec_len, u8), srcp2[x..][0..vec_len].*));
        dstp[x..][0..vec_len].* = @select(u8, (s2 - s1) >= cv, u8_255, u8_0);
    }
    while (x < w) : (x += 1) {
        dstp[x] = if (@as(i16, srcp2[x]) - @as(i16, srcp[x]) >= c) 255 else 0;
    }
}

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const src = try ctx.childFrame(n);
    defer src.release();
    const src2 = try ctx.getFrame(d.other.get(), n);
    defer src2.release();
    const dst = try ctx.newFrame(src);
    errdefer dst.release();

    hz.copyAlpha(ctx.env(), dst, src, d.format); // alpha passes through
    for (0..d.format.color_planes) |plane| {
        const id = d.format.planeId(plane);
        const s1 = src.plane(u8, id);
        const s2 = src2.plane(u8, id);
        const dp = dst.planeMut(u8, id);
        for (0..dp.height) |y| binarizeRow(dp.row(y), s1.row(y), s2.row(y), d.c);
    }
    hz.setRange(ctx.env(), dst, .FULL);
    return dst;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const other = try ctx.clip(1, "clip2");
    errdefer other.release();
    const vi = clip.videoInfo();
    try hz.compareClips(ctx, &.{ vi, other.videoInfo() }, .BIGGER_THAN, .color_only);
    const format = try hz.planarFormat(ctx, vi);
    if (format.sample != .u8) return ctx.fail("only 8 bit int format supported.", .{});

    const c_param = try ctx.int(i64, 2, "c", 3, math.minInt(i64), math.maxInt(i64));
    const d: Data = .{ .format = format, .other = .{ .clip = other }, .c = @intCast(math.clamp(c_param, -256, 256)) };
    const out = try F.create(ctx.env, clip, d, F.getFrame(getFrame), .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
