const std = @import("std");
const math = std.math;

const vec_len = std.simd.suggestVectorLength(i16) orelse 1;
const vec_t = @Vector(vec_len, i16);
const shift8: vec_t = @splat(8);
const floor: vec_t = @splat(0);
const peak: vec_t = @splat(255);

pub fn process(srcp: []const u8, dstp: []u8, stride: u32, width: u32, height: u32, thresinf: i16, thressup: i16) void {
    const thresinf_v: vec_t = @splat(thresinf);
    const thressup_v: vec_t = @splat(thressup);

    var su = srcp;
    var d = dstp;
    var s = srcp[stride..];
    var sd = s[stride..];

    @memset(d[0..stride], 0);
    d = d[stride..];

    var prod: vec_t = undefined;
    var y: u32 = 1;
    while (y < height - 1) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += vec_len) {
            prod = (@as(vec_t, su[x..][0..vec_len].*) - @as(vec_t, s[x..][0..vec_len].*)) *
                (@as(vec_t, sd[x..][0..vec_len].*) - @as(vec_t, s[x..][0..vec_len].*));

            const sel: vec_t = @select(
                i16,
                prod < thresinf_v,
                floor,
                @select(i16, prod > thressup_v, peak, prod >> shift8),
            );

            for (0..vec_len) |i| {
                d[x..][i] = @intCast(sel[i]);
            }
        }

        su = su[stride..];
        d = d[stride..];
        s = s[stride..];
        sd = sd[stride..];
    }

    @memset(d[0..stride], 0);
}
