const std = @import("std");
const math = std.math;

// const filter = @import("../filters/adg.zig");
const vszip = @import("../vszip.zig");
const hz = @import("../helper.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "ADGMask";

const Data = struct {
    node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    wxh: f32 = 0,
    shift: u4 = 0,
    peak: f32 = 0,
    scaling: f32 = 0,
    float_range: [256]f32 = undefined,
};

fn average(comptime T: type, src: []const T, stride: u32, w: u32, h: u32, wxh: f32, peak: f32) f32 {
    var acc: if (@typeInfo(T) == .float) f64 else u64 = 0;
    var acc2: if (@typeInfo(T) == .float) f64 else u64 = 0;
    var srcp = src;
    var y: u32 = 0;
    while (y < (h - 1)) : (y += 2) {
        const srcp2 = srcp[stride..];
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            acc += srcp[x];
            acc2 += srcp2[x];
        }

        srcp = srcp[(stride << 1)..];
    }

    if (h % 2 != 0) {
        srcp = src[(stride * (h - 1))..];
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            acc += srcp[x];
        }
    }

    acc += acc2;
    if (@typeInfo(T) == .float) {
        return @floatCast(acc / wxh);
    } else {
        return @floatCast(@as(f64, @floatFromInt(acc)) / wxh / peak);
    }
}

pub inline fn process(
    comptime T: type,
    src: []const T,
    dst: []T,
    stride: u32,
    w: u32,
    h: u32,
    float_range: *[256]f32,
    shift: u4,
    scaling: f32,
    wxh: f32,
    peak: f32,
) void {
    const avg = average(T, src, stride, w, h, wxh, peak);
    const scaling2 = avg * avg * scaling;

    const LT: type = if (@typeInfo(T) == .int) T else u16;
    var lut: [256]LT = undefined;
    for (&lut, float_range) |*i, r| {
        const x: f32 = @min(@max(@mulAdd(f32, @exp(scaling2 * @log(r)), peak, 0.5), 0), peak);
        i.* = @intFromFloat(x);
    }

    if (@typeInfo(T) == .int) {
        for (src, dst) |sx, *dx| {
            dx.* = lut[@as(usize, sx) >> shift];
        }
    } else {
        for (src, dst) |sx, *dx| {
            const idx = @as(usize, @intFromFloat(sx * peak)) >> shift;
            dx.* = @as(T, @floatFromInt(lut[idx])) / peak;
        }
    }
}

fn ADG(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(n, d.node);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, n);
                defer src.deinit();
                const dst = src.newVideoFrame();

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp = src.getReadSlice2(T, plane);
                    const dstp = dst.getWriteSlice2(T, plane);
                    const width, const height, const stride = src.getDimensions2(T, plane);

                    process(
                        T,
                        srcp,
                        dstp,
                        stride,
                        width,
                        height,
                        &d.float_range,
                        d.shift,
                        d.scaling,
                        d.wxh,
                        d.peak,
                    );
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    zapi.freeNode(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    _ = map_out; // autofix
    d.node, d.vi = map_in.getNodeVi("clip").?;

    d.scaling = map_in.getValue(f32, "luma_scaling") orelse 8;
    d.shift = @intCast(@min(d.vi.format.bitsPerSample, 16) - 8);
    d.peak = hz.getPeakValue(&d.vi.format, false, .FULL);
    if (d.vi.format.sampleType == .Float) {
        d.peak = std.math.maxInt(u16);
    }

    d.wxh = @floatFromInt(d.vi.width * d.vi.height);
    for (0..256) |i| {
        const x = @as(f32, @floatFromInt(i)) / 256;
        d.float_range[i] = (1 - (x * ((x * ((x * ((x * ((x * 18.188) - 45.47)) + 36.624)) - 9.466)) + 1.124)));
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = switch (d.vi.format.bytesPerSample) {
        1 => &ADG(u8).getFrame,
        2 => &ADG(u16).getFrame,
        4 => &ADG(f32).getFrame,
        else => unreachable,
    };

    zapi.createVideoFilter(out, filter_name, d.vi, gf, free, .Parallel, &deps, data);
}
