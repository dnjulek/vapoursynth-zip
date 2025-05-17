const std = @import("std");
const math = std.math;

const helper = @import("../helper.zig");
const Data = @import("../vapoursynth/planeminmax.zig").Data;

const allocator = std.heap.c_allocator;

const StatsFloat = struct {
    max: f32,
    min: f32,
    diff: f64,
};

const StatsInt = struct {
    max: u16,
    min: u16,
    diff: f64,
};

pub const Stats = union(enum) {
    f: StatsFloat,
    i: StatsInt,
};

pub fn minMaxInt(comptime T: type, src: []const T, stride: u32, w: u32, h: u32, d: *Data) Stats {
    var srcp: []const T = src;
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            accum_buf[v] += 1;
        }
        srcp = srcp[stride..];
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    return .{ .i = .{ .max = retvalmax, .min = retvalmin, .diff = undefined } };
}

pub fn minMaxFloat(comptime T: type, src: []const T, stride: u32, w: u32, h: u32, d: *Data) Stats {
    var srcp: []const T = src;
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            accum_buf[math.lossyCast(u16, (v * 65535.0 + 0.5))] += 1;
        }
        srcp = srcp[stride..];
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    const retvalmaxf = @as(f32, @floatFromInt(retvalmax)) / 65535;
    const retvalminf = @as(f32, @floatFromInt(retvalmin)) / 65535;
    return .{ .f = .{ .max = retvalmaxf, .min = retvalminf, .diff = undefined } };
}

pub fn minMaxIntRef(comptime T: type, src: []const T, ref: []const T, stride: u32, w: u32, h: u32, d: *Data) Stats {
    var srcp: []const T = src;
    var refp: []const T = ref;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: u64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[v] += 1;
            diffacc += helper.absDiff(v, j);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }

    const diff: f64 = @as(f64, @floatFromInt(diffacc)) / total / @as(f64, @floatFromInt(d.peak));
    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    return .{ .i = .{ .max = retvalmax, .min = retvalmin, .diff = diff } };
}

pub fn minMaxFloatRef(comptime T: type, src: []const T, ref: []const T, stride: u32, w: u32, h: u32, d: *Data) Stats {
    var srcp: []const T = src;
    var refp: []const T = ref;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, 32, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[math.lossyCast(u16, (v * 65535.0 + 0.5))] += 1;
            diffacc += helper.absDiff(v, j);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }

    const totalmin: u32 = @intFromFloat(total * d.minthr);
    const totalmax: u32 = @intFromFloat(total * d.maxthr);
    var count: u32 = 0;

    var u: u16 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break u;
    } else d.peak;

    count = 0;
    var i: i32 = @intCast(d.peak);
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    const retvalmaxf = @as(f32, @floatFromInt(retvalmax)) / 65535;
    const retvalminf = @as(f32, @floatFromInt(retvalmin)) / 65535;
    return .{ .f = .{ .max = retvalmaxf, .min = retvalminf, .diff = (diffacc / total) } };
}
