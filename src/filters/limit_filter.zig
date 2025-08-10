const std = @import("std");

pub fn process(
    comptime T: type,
    fltp: []const T,
    srcp: []const T,
    refp: []const T,
    dstp: []T,
    dark_thr: f32,
    bright_thr: f32,
    elast: f32,
) void {
    for (fltp, srcp, refp, dstp) |f, s, r, *d| {
        const sf: f32 = if (@typeInfo(T) == .int) @floatFromInt(s) else s;
        const ff: f32 = if (@typeInfo(T) == .int) @floatFromInt(f) else f;
        const rf: f32 = if (@typeInfo(T) == .int) @floatFromInt(r) else r;

        const diff_signed: f32 = ff - rf;
        const diff_abs: f32 = @abs(diff_signed);
        const thr1: f32 = if (diff_signed > 0) bright_thr else dark_thr;
        const thr2: f32 = thr1 * elast;

        var out: f32 = undefined;
        if (diff_abs <= thr1) {
            out = ff;
        } else if (diff_abs >= thr2) {
            out = sf;
        } else {
            out = sf + (ff - sf) * (thr2 - diff_abs) / (thr2 - thr1);
        }

        d.* = if (@typeInfo(T) == .int) @intFromFloat(out + 0.5) else @floatCast(out);
    }
}
