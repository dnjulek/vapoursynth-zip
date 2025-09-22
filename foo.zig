const std = @import("std");
const print = std.debug.print;

fn powfFast(a: f32, b: f32) f32 {
    const u = @as(i32, @bitCast(a));
    const result_bits = @as(i32, @intFromFloat(b * @as(f32, @floatFromInt(u - 1064866805)) + 1064866805.0));
    return @as(f32, @bitCast(result_bits));
}

pub fn main() void {
    const test_cases = [_][2]f32{
        .{ 2.0, 3.0 },
        .{ 5.0, 2.0 },
        .{ 10.0, 0.5 },
        .{ 3.14, 2.0 },
        .{ 2.0, 8.0 },
    };

    print("Testing powf_fast vs standard pow:\n", .{});
    print("{s:<10} {s:<10} {s:<15} {s:<15} {s:<10}\n", .{ "Base", "Exp", "powf_fast", "pow", "Error" });
    print("---------------------------------------------------------------\n", .{});

    for (test_cases) |case| {
        const a = case[0];
        const b = case[1];
        const fast_result = powfFast(a, b);
        const std_result = std.math.pow(f32, a, b);
        const error_pct = @abs(fast_result - std_result) / std_result * 100.0;

        print("{d:<10.2} {d:<10.2} {d:<15.6} {d:<15.6} {d:<10.2}%\n", .{ a, b, fast_result, std_result, error_pct });
    }
}
