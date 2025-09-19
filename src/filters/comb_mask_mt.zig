const std = @import("std");
const math = std.math;

const vec_len = std.simd.suggestVectorLength(i32) orelse 1;
const vec_i32 = @Vector(vec_len, i32);
const vec_u8 = @Vector(vec_len, u8);
const floor: vec_u8 = @splat(0);
const peak: vec_u8 = @splat(255);
const u8_len: vec_i32 = @splat(256);

pub fn process(
    srcp: []const u8,
    dstp: []u8,
    stride: u32,
    width: u32,
    height: u32,
    thresinf: u8,
    thressup: u8,
    thr_diff: u8,
    comptime same_thr: bool,
) void {
    const thresinf_v: vec_u8 = @splat(thresinf);
    const thressup_v: vec_u8 = @splat(thressup);
    const thr_diff_v: vec_i32 = @splat(thr_diff);

    var su = srcp;
    var d = dstp;
    var s = srcp[stride..];
    var sd = s[stride..];

    @memset(d[0..stride], 0);
    d = d[stride..];

    var prod: vec_i32 = undefined;
    var y: u32 = 1;
    while (y < height - 1) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            prod = (@as(vec_i32, su[x..][0..vec_len].*) - @as(vec_i32, s[x..][0..vec_len].*)) *
                (@as(vec_i32, sd[x..][0..vec_len].*) - @as(vec_i32, s[x..][0..vec_len].*));

            const gray: vec_i32 = if (same_thr) floor else @min(((prod - thresinf_v) * u8_len / thr_diff_v), peak);
            const sel: vec_i32 = @select(
                i32,
                prod < thresinf_v,
                floor,
                @select(i32, prod > thressup_v, peak, gray),
            );

            d[x..][0..vec_len].* = @as(vec_u8, @intCast(sel));
        }

        su = su[stride..];
        d = d[stride..];
        s = s[stride..];
        sd = sd[stride..];
    }

    @memset(d[0..stride], 0);
}
