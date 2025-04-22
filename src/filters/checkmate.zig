const std = @import("std");
const math = std.math;
const hz = @import("../helper.zig");

pub fn process(
    dstp: []u8,
    srcp_p2: anytype,
    srcp_p1: []const u8,
    srcp: []const u8,
    srcp_n1: []const u8,
    srcp_n2: anytype,
    stride: u32,
    width: u32,
    thr: i32,
    tmax: i32,
    tthr2: i32,
    comptime use_tthr2: bool,
) void {
    const tmax_multiplier: i32 = @divTrunc((1 << 13), tmax);

    var x: u32 = 0;
    while (x < width) : (x += 1) {
        if (use_tthr2 and
            (@abs(@as(i32, srcp_p1[x]) - srcp_n1[x]) < tthr2 and
                @abs(@as(i32, srcp_p2[x]) - srcp[x]) < tthr2 and
                @abs(@as(i32, srcp[x]) - srcp_n2[x]) < tthr2))
        {
            dstp[x] = @intCast((@as(u16, srcp_p1[x]) + @as(u16, srcp[x]) * 2 + @as(u16, srcp_n1[x])) >> 2);
        } else {
            const stride2: u32 = stride << 1;
            const next_value: i32 = @as(i32, srcp[x]) + @as(i32, srcp_n1[x]);
            const prev_value: i32 = @as(i32, srcp[x]) + @as(i32, srcp_p1[x]);

            const x_left: u32 = if (x < 2) 0 else x - 2;
            const x_right: u32 = if (x > width - 3) width - 1 else x + 2;

            const current_column: i32 = @as(i32, hz.getVal2(u8, srcp.ptr, x, stride2)) + @as(i32, srcp[x]) * 2 + @as(i32, srcp[stride2 + x]);
            const cvl: i32 = hz.getVal2(u8, srcp.ptr, x_left, stride2);
            const cvr: i32 = hz.getVal2(u8, srcp.ptr, x_right, stride2);
            const curr_value: i32 = (-cvl - cvr + @as(i32, srcp[x_left]) * 2 + @as(i32, srcp[x_right]) * 2 - @as(i32, srcp[x_left + stride2]) -
                @as(i32, srcp[x_right + stride2]) + current_column * 2 + @as(i32, srcp[x]) * 12);

            var nc: i32 = hz.getVal2(u8, srcp_n1.ptr, x, stride2);
            var pc: i32 = hz.getVal2(u8, srcp_p1.ptr, x, stride2);
            nc = (nc + @as(i32, srcp_n1[x]) * 2 + @as(i32, srcp_n1[x + stride2])) - current_column;
            pc = (pc + @as(i32, srcp_p1[x]) * 2 + @as(i32, srcp_p1[x + stride2])) - current_column;
            nc = thr + tmax - @as(i32, @intCast(@abs(nc)));
            pc = thr + tmax - @as(i32, @intCast(@abs(pc)));

            const next_weight: i32 = @min(math.clamp(nc, 0, tmax + 1) * tmax_multiplier, 8192);
            const prev_weight: i32 = @min(math.clamp(pc, 0, tmax + 1) * tmax_multiplier, 8192);
            const curr_weight: i32 = (1 << 14) - (next_weight + prev_weight);
            const out: i32 = (curr_weight * @divTrunc(curr_value, 10) + prev_weight * prev_value + next_weight * next_value) >> 15;
            dstp[x] = math.lossyCast(u8, out);
        }
    }
}
