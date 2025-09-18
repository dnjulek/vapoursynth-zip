const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const Data = @import("../vapoursynth/planeminmax.zig").Data;
const vszip = @import("../vszip.zig");
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

pub fn minMaxInt(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, vszip.alignment, 65536) catch unreachable;
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
    var i: i32 = d.peak;
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    props.setInt(d.prop.mi, retvalmin, .Append);
    props.setInt(d.prop.ma, retvalmax, .Append);
}

pub fn minMaxFloat(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    const total: f64 = @floatFromInt(w * h);

    const accum_buf = allocator.alignedAlloc(u32, vszip.alignment, 65536) catch unreachable;
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
    var i: i32 = d.peak;
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    const retvalmaxf = @as(f32, @floatFromInt(retvalmax)) / 65535;
    const retvalminf = @as(f32, @floatFromInt(retvalmin)) / 65535;

    props.setFloat(d.prop.mi, retvalminf, .Append);
    props.setFloat(d.prop.ma, retvalmaxf, .Append);
}

pub fn minMaxIntRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    var refp: []const T = ref;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, vszip.alignment, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[v] += 1;
            diffacc += absDiff(v, j);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }

    const diff: f64 = diffacc / total / d.peakf;
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

    props.setInt(d.prop.mi, retvalmin, .Append);
    props.setInt(d.prop.ma, retvalmax, .Append);
    props.setFloat(d.prop.d, diff, .Append);
}

pub fn minMaxFloatRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    var refp: []const T = ref;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    const accum_buf = allocator.alignedAlloc(u32, vszip.alignment, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    for (accum_buf) |*i| {
        i.* = 0;
    }

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            accum_buf[math.lossyCast(u16, (v * 65535.0 + 0.5))] += 1;
            diffacc += @abs(v - j);
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
    const diff = diffacc / total;

    props.setFloat(d.prop.mi, retvalminf, .Append);
    props.setFloat(d.prop.ma, retvalmaxf, .Append);
    props.setFloat(d.prop.d, diff, .Append);
}

pub fn minMaxNoThr(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    var min: T = if (@typeInfo(T) == .int) math.maxInt(T) else math.inf(T);
    var max: T = if (@typeInfo(T) == .int) 0 else -math.inf(T);
    for (0..h) |_| {
        for (srcp[0..w]) |v| {
            min = @min(min, v);
            max = @max(max, v);
        }
        srcp = srcp[stride..];
    }

    if (@typeInfo(T) == .int) {
        props.setInt(d.prop.mi, min, .Append);
        props.setInt(d.prop.ma, max, .Append);
    } else {
        props.setFloat(d.prop.mi, min, .Append);
        props.setFloat(d.prop.ma, max, .Append);
    }
}

pub fn minMaxNoThrRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    var srcp: []const T = src;
    var refp: []const T = ref;
    var min: T = if (@typeInfo(T) == .int) math.maxInt(T) else math.inf(T);
    var max: T = if (@typeInfo(T) == .int) 0 else -math.inf(T);
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    for (0..h) |_| {
        for (srcp[0..w], refp[0..w]) |v, j| {
            diffacc += if (@typeInfo(T) == .int) absDiff(v, j) else @abs(v - j);
            min = @min(min, v);
            max = @max(max, v);
        }
        srcp = srcp[stride..];
        refp = refp[stride..];
    }

    var diff: f64 = diffacc / total;
    if (@typeInfo(T) == .int) {
        diff /= d.peakf;
    }

    if (@typeInfo(T) == .int) {
        props.setInt(d.prop.mi, min, .Append);
        props.setInt(d.prop.ma, max, .Append);
    } else {
        props.setFloat(d.prop.mi, min, .Append);
        props.setFloat(d.prop.ma, max, .Append);
    }

    props.setFloat(d.prop.d, diff, .Append);
}

fn absDiff(x: anytype, y: anytype) f64 {
    const xf: f64 = @floatFromInt(x);
    const yf: f64 = @floatFromInt(y);
    return @abs(xf - yf);
}
