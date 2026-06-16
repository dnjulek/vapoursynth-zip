const std = @import("std");
const math = std.math;
const allocator = std.heap.c_allocator;

const Data = @import("../vapoursynth/planeminmax.zig").Data;
const vszip = @import("../vszip.zig");
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

inline fn minMaxImpl(comptime T: type, comptime use_ref: bool, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    const is_int = @typeInfo(T) == .int;
    var srcp: []const T = src;
    var refp: []const T = ref;
    const total: f64 = @floatFromInt(w * h);
    var diffacc: f64 = 0;

    const accum_buf: []align(vszip.vec_len) u32 = allocator.alignedAlloc(u32, vszip.alignment, 65536) catch unreachable;
    defer allocator.free(accum_buf);

    @memset(accum_buf, 0);

    for (0..h) |_| {
        if (use_ref) {
            for (srcp[0..w], refp[0..w]) |v, j| {
                const idx = if (is_int) v else math.lossyCast(u16, (@as(f32, v) * 65535.0 + 0.5));
                accum_buf[idx] += 1;
                diffacc += if (is_int) absDiff(v, j) else @abs(v - j);
            }
            refp = refp[stride..];
        } else {
            for (srcp[0..w]) |v| {
                const idx = if (is_int) v else math.lossyCast(u16, (@as(f32, v) * 65535.0 + 0.5));
                accum_buf[idx] += 1;
            }
        }
        srcp = srcp[stride..];
    }

    const totalmin: u32 = @trunc(total * d.minthr);
    const totalmax: u32 = @trunc(total * d.maxthr);
    var count: u32 = 0;

    var u: u32 = 0;
    const retvalmin: u16 = while (u < d.hist_size) : (u += 1) {
        count += accum_buf[u];
        if (count > totalmin) break @intCast(u);
    } else d.peak;

    count = 0;
    var i: i32 = d.peak;
    const retvalmax: u16 = while (i >= 0) : (i -= 1) {
        const ui: u16 = @intCast(i);
        count += accum_buf[ui];
        if (count > totalmax) break ui;
    } else 0;

    if (is_int) {
        props.setInt(d.prop.mi, retvalmin, .Append);
        props.setInt(d.prop.ma, retvalmax, .Append);
    } else {
        props.setFloat(d.prop.mi, @as(f32, @floatFromInt(retvalmin)) / 65535, .Append);
        props.setFloat(d.prop.ma, @as(f32, @floatFromInt(retvalmax)) / 65535, .Append);
    }

    if (use_ref) {
        const diff: f64 = if (is_int) diffacc / total / d.peakf else diffacc / total;
        props.setFloat(d.prop.d, diff, .Append);
    }
}

pub fn minMax(comptime T: type, src: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    minMaxImpl(T, false, src, &.{}, stride, props, w, h, d);
}

pub fn minMaxRef(comptime T: type, src: []const T, ref: []const T, stride: u32, props: *const ZAPI.ZMap(*vs.Map), w: u32, h: u32, d: *Data) void {
    minMaxImpl(T, true, src, ref, stride, props, w, h, d);
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
