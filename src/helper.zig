const std = @import("std");
const math = std.math;

const vszip = @import("vszip.zig");
const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const vsc = vapoursynth.vsconstants;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

pub const BPSType = enum {
    U8,
    U9,
    U10,
    U12,
    U14,
    U16,
    U32,
    F16,
    F32,

    pub fn select(map: ZAPI.ZMap(?*vs.Map), node: ?*vs.Node, vi: *const vs.VideoInfo, comptime name: []const u8) !BPSType {
        var err_msg: ?[:0]const u8 = null;
        errdefer {
            map.setError(err_msg.?);
            map.zapi.freeNode(node);
        }

        if (vi.format.sampleType == .Integer) {
            switch (vi.format.bitsPerSample) {
                8 => return .U8,
                9 => return .U9,
                10 => return .U10,
                12 => return .U12,
                14 => return .U14,
                16 => return .U16,
                32 => return .U32,
                else => return {
                    err_msg = name ++ ": not supported Int format.";
                    return error.format;
                },
            }
        } else {
            switch (vi.format.bitsPerSample) {
                16 => return .F16,
                32 => return .F32,
                else => return {
                    err_msg = name ++ ": not supported Float format.";
                    return error.format;
                },
            }
        }
    }
};

pub const DataType = enum {
    U8,
    U16,
    U32,
    F16,
    F32,

    pub fn select(map: ZAPI.ZMap(?*vs.Map), node: ?*vs.Node, vi: *const vs.VideoInfo, comptime name: []const u8, enable_u32: bool) !DataType {
        var err_msg: ?[:0]const u8 = null;

        errdefer {
            map.setError(err_msg.?);
            map.zapi.freeNode(node);
        }

        if (vi.format.sampleType == .Integer) {
            switch (vi.format.bytesPerSample) {
                1 => return .U8,
                2 => return .U16,
                4 => if (enable_u32) return .U32 else {
                    err_msg = name ++ ": not supported Int format.";
                    return error.format;
                },
                else => {
                    err_msg = name ++ ": not supported Int format.";
                    return error.format;
                },
            }
        } else {
            switch (vi.format.bytesPerSample) {
                2 => return .F16,
                4 => return .F32,
                else => {
                    err_msg = name ++ ": not supported Float format.";
                    return error.format;
                },
            }
        }
    }
};

pub fn absDiff(x: anytype, y: anytype) @TypeOf(x) {
    return if (x > y) (x - y) else (y - x);
}

pub fn mapGetPlanes(in: ZAPI.ZMap(?*const vs.Map), out: ZAPI.ZMap(?*vs.Map), nodes: []const ?*vs.Node, process: []bool, num_planes: c_int, comptime name: []const u8, zapi: *const ZAPI) !void {
    const num_e = in.numElements("planes") orelse return;
    @memset(process, false);

    var err_msg: ?[:0]const u8 = null;
    errdefer {
        out.setError(err_msg.?);
        for (nodes) |node| {
            if (node) |n| {
                zapi.freeNode(n);
            }
        }
    }

    var i: u32 = 0;
    while (i < num_e) : (i += 1) {
        const e = in.getValue2(i32, "planes", i).?;
        if ((e < 0) or (e >= num_planes)) {
            err_msg = name ++ ": plane index out of range";
            return error.ValidationError;
        }

        const ue: u32 = @intCast(e);
        if (process[ue]) {
            err_msg = name ++ ": plane specified twice.";
            return error.ValidationError;
        }

        process[ue] = true;
    }
}

pub const ClipLen = enum {
    SAME_LEN,
    BIGGER_THAN,
    MISMATCH,
};

pub fn compareNodes(out: ZAPI.ZMap(?*vs.Map), nodes: []const ?*vs.Node, len: ClipLen, comptime name: []const u8, zapi: *const ZAPI) !void {
    const vi0 = zapi.getVideoInfo(nodes[0]);
    var err_msg: ?[:0]const u8 = null;
    errdefer {
        out.setError(err_msg.?);
        for (nodes) |node| {
            if (node) |n| {
                zapi.freeNode(n);
            }
        }
    }

    for (nodes[1..]) |node| {
        if (node == null) continue;

        const vi = zapi.getVideoInfo(node);
        if (!vsh.isConstantVideoFormat(vi)) {
            err_msg = name ++ ": all input clips must have constant format.";
            return error.constant_format;
        }
        if ((vi0.width != vi.width) or (vi0.height != vi.height)) {
            err_msg = name ++ ": all input clips must have the same width and height.";
            return error.width_height;
        }
        if (vi0.format.colorFamily != vi.format.colorFamily) {
            err_msg = name ++ ": all input clips must have the same color family.";
            return error.color_family;
        }
        if ((vi0.format.subSamplingW != vi.format.subSamplingW) or (vi0.format.subSamplingH != vi.format.subSamplingH)) {
            err_msg = name ++ ": all input clips must have the same subsampling.";
            return error.subsampling;
        }
        if (vi0.format.bitsPerSample != vi.format.bitsPerSample) {
            err_msg = name ++ ": all input clips must have the same bit depth.";
            return error.bit_depth;
        }

        switch (len) {
            .SAME_LEN => if (vi0.numFrames != vi.numFrames) {
                err_msg = name ++ ": all input clips must have the same length.";
                return error.length;
            },
            .BIGGER_THAN => if (vi0.numFrames > vi.numFrames) {
                err_msg = name ++ ": second clip has less frames than input clip.";
                return error.length;
            },
            .MISMATCH => {},
        }
    }
}

pub fn getHistLen(vi: *const vs.VideoInfo) u32 {
    if (vi.format.sampleType == .Integer) {
        return math.shl(u32, 1, vi.format.bitsPerSample);
    } else {
        return math.maxInt(u16) + 1;
    }
}

pub fn toRGBS(node: ?*vs.Node, zapi: *const ZAPI) ?*vs.Node {
    const vi = zapi.getVideoInfo(node);
    if (zapi.getVideoFormatID(vi) == .RGBS) {
        return node;
    }

    const matrix: i32 = if (vi.height > 650) 1 else 6;
    const args = zapi.createZMap();
    _ = args.consumeNode("clip", node, .Replace);
    args.setInt("matrix_in", matrix, .Replace);
    args.setInt("format", @intFromEnum(vs.PresetVideoFormat.RGBS), .Replace);

    const vsplugin = zapi.getPluginByID2(.Resize);
    const ret = args.invoke(vsplugin, "Bicubic");
    const out = ret.getNode("clip");
    ret.free();
    args.free();
    return out;
}

pub fn getVal(comptime T: type, ptr: anytype, dist: isize) T {
    const adr: isize = @intCast(@intFromPtr(ptr));
    const uadr: usize = @intCast(adr + dist);
    const ptr2: [*]const T = @ptrFromInt(uadr);
    return ptr2[0];
}

pub fn getVal2(comptime T: type, ptr: anytype, x: u32, y: u32) T {
    const ix: i32 = @intCast(x);
    const iy: i32 = @intCast(y);
    const adr: isize = @intCast(@intFromPtr(ptr));
    const uadr: usize = @intCast(adr + ix - iy);
    const ptr2: [*]const T = @ptrFromInt(uadr);
    return ptr2[0];
}

pub fn getColorRange(node: *vs.Node, zapi: *const ZAPI) vsc.ColorRange {
    const frame = zapi.getFrame(0, node, null, 0);
    defer zapi.freeFrame(frame);

    if (frame) |f| {
        const props = zapi.getFramePropertiesRO(f);
        if (props) |p| {
            const color_range = zapi.initZMap(p).getColorRange();
            if (color_range) |cr| return cr;
        }
    }

    const vi = zapi.getVideoInfo(node);
    if (vi.format.colorFamily == .RGB) {
        return .FULL;
    } else {
        return .LIMITED;
    }
}

pub fn getLowestValue(fmt: *const vs.VideoFormat, chroma: bool, range: vsc.ColorRange) f32 {
    if (fmt.sampleType == .Float) {
        return if (chroma) -0.5 else 0.0;
    }

    if (range == .LIMITED) {
        return @floatFromInt(math.shl(i32, 16, fmt.bitsPerSample - 8));
    }

    return 0;
}

pub fn getPeakValue(fmt: *const vs.VideoFormat, chroma: bool, range: vsc.ColorRange) f32 {
    if (fmt.sampleType == .Float) {
        return if (chroma) 0.5 else 1.0;
    }

    if (range == .LIMITED) {
        const a: i32 = if (chroma) 240 else 235;
        return @floatFromInt(math.shl(i32, a, fmt.bitsPerSample - 8));
    }

    return @floatFromInt(math.shl(i32, 1, fmt.bitsPerSample) - 1);
}

const ScaleValue = struct {
    depth_in: i32 = 8,
    sample_type_in: vs.SampleType = .Integer,
    chroma: bool = false,
};

pub fn scaleValue(value: f32, target: *vs.Node, zapi: *const ZAPI, opt: ScaleValue) f32 {
    const depth_in = opt.depth_in;
    const chroma = opt.chroma;

    const fmt_out = zapi.getVideoInfo(target).format;
    var fmt_in = fmt_out;
    fmt_in.bitsPerSample = depth_in;
    fmt_in.sampleType = opt.sample_type_in;

    var out_value = value;
    if (depth_in == fmt_out.bitsPerSample) {
        return out_value;
    }

    const range = getColorRange(target, zapi);
    const input_peak = getPeakValue(&fmt_in, chroma, range);
    const input_lowest = getLowestValue(&fmt_in, chroma, range);
    const output_peak = getPeakValue(&fmt_out, chroma, range);
    const output_lowest = getLowestValue(&fmt_out, chroma, range);
    out_value *= (output_peak - output_lowest) / (input_peak - input_lowest);

    if (fmt_out.sampleType == .Integer) {
        out_value = @max(@min(@round(out_value), getPeakValue(&fmt_out, false, .FULL)), 0);
    }

    return out_value;
}

pub fn getArray(
    comptime T: type,
    default: T,
    min: T,
    max: T,
    comptime key: [:0]const u8,
    comptime filter_name: [:0]const u8,
    in: ZAPI.ZMap(?*const vs.Map),
    out: ZAPI.ZMap(?*vs.Map),
    nodes: []const ?*vs.Node,
    zapi: *const ZAPI,
) ![3]T {
    var err_msg: ?[:0]const u8 = null;
    errdefer {
        out.setError(err_msg.?);
        for (nodes) |node| {
            if (node) |n| zapi.freeNode(n);
        }
        allocator.free(err_msg.?);
    }

    var array: [3]T = undefined;
    const len = in.numElements(key) orelse 0;
    var i: u8 = 0;
    while (i < 3) : (i += 1) {
        if (i < len) {
            array[i] = in.getValue2(T, key, i).?;
        } else if (i == 0) {
            array[i] = default;
        } else {
            array[i] = array[i - 1];
        }

        if (array[i] < min) {
            err_msg = std.fmt.allocPrintSentinel(
                allocator,
                "{s}: {s} value {d} is below minimum {d}.",
                .{ filter_name, key, array[i], min },
                0,
            ) catch return error.outOfMemory;
            return error.invalidArgument;
        }

        if (array[i] > max) {
            err_msg = std.fmt.allocPrintSentinel(
                allocator,
                "{s}: {s} value {d} is above maximum {d}.",
                .{ filter_name, key, array[i], max },
                0,
            ) catch return error.outOfMemory;
            return error.invalidArgument;
        }
    }

    return array;
}

pub const Maps = struct {
    in: *const ZAPI.ZMap(?*const vs.Map),
    out: *const ZAPI.ZMap(?*vs.Map),
    name: [:0]const u8,

    pub fn init(zin: *const ZAPI.ZMap(?*const vs.Map), zout: *const ZAPI.ZMap(?*vs.Map), filter_name: [:0]const u8) Maps {
        return Maps{
            .in = zin,
            .out = zout,
            .name = filter_name,
        };
    }

    pub fn getValue(self: *const Maps, comptime T: type, comptime key: [:0]const u8, default: T, min: T, max: T) !T {
        const is_int = @typeInfo(T) == .int;
        const T2 = if (is_int) i64 else f64;
        const val: T2 = self.in.getValue(T2, key) orelse default;
        if (val < min or val > max) {
            self.out.setError2("{s}: parameter \"{s}={d}\" out of range [{d}..{d}].", .{ self.name, key, val, min, max });
            return error.ParameterOutOfRange;
        }

        return if (is_int) @intCast(val) else @floatCast(val);
    }

    pub fn getArray(self: *const Maps, comptime key: [:0]const u8, max_len: comptime_int, default: [3]f64, min: f64, max: f64) ![3]f64 {
        if (self.in.getFloatArray(key)) |a| {
            if (a.len > max_len) {
                self.out.setError2("{s}: parameter \"{s}\" has too many elements (got {d}, max {d}).", .{ self.name, key, a.len, max_len });
                return error.ParameterOutOfRange;
            }
            var out: [3]f64 = undefined;
            for (0..3) |i| {
                const val = a[@min(i, a.len - 1)];
                if (val < min or val > max) {
                    self.out.setError2("{s}: parameter \"{s}[{d}]={d}\" out of range [{d}..{d}].", .{ self.name, key, i, val, min, max });
                    return error.ParameterOutOfRange;
                }

                out[i] = val;
            }
            return out;
        }

        return default;
    }
};

pub const ditherType = enum {
    none,
    ordered,
    random,
    error_diffusion,

    pub fn toString(self: ditherType) [:0]const u8 {
        return switch (self) {
            .none => "none",
            .ordered => "ordered",
            .random => "random",
            .error_diffusion => "error_diffusion",
        };
    }
};

pub fn bitDepth(bitdepth: u32, node: *vs.Node, dither: ditherType, zapi: *const ZAPI) *vs.Node {
    const vf = zapi.getVideoInfo(node).format;
    if (vf.bitsPerSample == bitdepth) {
        return node;
    }

    const vf_out = vs.makeVideoID(
        vf.colorFamily,
        vf.sampleType,
        bitdepth,
        @intCast(vf.subSamplingW),
        @intCast(vf.subSamplingH),
    );

    const args = zapi.createZMap();
    _ = args.consumeNode("clip", node, .Replace);
    args.setInt("format", vf_out, .Replace);
    args.setData("dither_type", dither.toString(), .Utf8, .Replace);
    const vsplugin = zapi.getPluginByID2(.Resize);
    const ret = args.invoke(vsplugin, "Point");
    const out = ret.getNode("clip").?;
    ret.free();
    args.free();
    return out;
}
