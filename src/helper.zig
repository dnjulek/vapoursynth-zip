const std = @import("std");
const builtin = @import("builtin");
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

    pub fn ToType(comptime self: DataType) type {
        return switch (self) {
            .U8 => u8,
            .U16 => u16,
            .U32 => u32,
            .F16 => f16,
            .F32 => f32,
        };
    }
};

pub fn selectRefFilter(
    comptime Filter: fn (comptime type, comptime bool) type,
    dt: DataType,
    refb: bool,
    comptime enable_u32: bool,
) vs.FilterGetFrame {
    return switch (dt) {
        inline else => |tag| if (comptime tag == .U32 and !enable_u32) unreachable else blk: {
            const T = comptime tag.ToType();
            break :blk if (refb) &Filter(T, true).getFrame else &Filter(T, false).getFrame;
        },
    };
}

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
    args.setInt("format", @backingInt(vs.PresetVideoFormat.RGBS), .Replace);

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

pub fn getRange(node: *vs.Node, zapi: *const ZAPI) vsc.Range {
    const frame = zapi.getFrame(0, node, null);
    defer zapi.freeFrame(frame);

    if (frame) |f| {
        const props = zapi.getFramePropertiesRO(f);
        if (props) |p| {
            const range = zapi.initZMap(p).getRange();
            if (range) |r| return r;
        }
    }

    const vi = zapi.getVideoInfo(node);
    if (vi.format.colorFamily == .RGB) {
        return .FULL;
    } else {
        return .LIMITED;
    }
}

pub fn getLowestValue(fmt: *const vs.VideoFormat, chroma: bool, range: vsc.Range) f32 {
    if (fmt.sampleType == .Float) {
        return if (chroma) -0.5 else 0.0;
    }

    if (range == .LIMITED) {
        return @floatFromInt(math.shl(i64, 16, fmt.bitsPerSample - 8));
    }

    return 0;
}

pub fn getPeakValue(fmt: *const vs.VideoFormat, chroma: bool, range: vsc.Range) f32 {
    if (fmt.sampleType == .Float) {
        return if (chroma) 0.5 else 1.0;
    }

    // i64: a 32-bit integer peak overflows i32 (it used to come out as -1).
    // The f32 result is approximate for 32-bit (2^32 - 1 rounds to 2^32);
    // callers validating 32-bit values must use an exact peak instead.
    if (range == .LIMITED) {
        const a: i64 = if (chroma) 240 else 235;
        return @floatFromInt(math.shl(i64, a, fmt.bitsPerSample - 8));
    }

    return @floatFromInt(math.shl(i64, 1, fmt.bitsPerSample) - 1);
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

    // Float clips are always full range, whatever their _Range prop says (a
    // missing prop reads as LIMITED for non-RGB): scale float targets as full.
    const range: vsc.Range = if (fmt_out.sampleType == .Float) .FULL else getRange(target, zapi);
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

// ---------------------------------------------------------------------------
// Argument validation
//
// Every public float argument must be finite before it is range-checked or
// used: NaN compares false against both bounds of a `v < min or v > max`
// check, so plain range checks accept it, and +-inf / huge values later
// overflow float->int conversions (Debug panic, ReleaseFast UB). Float values
// are therefore read and validated as f64 and only narrowed afterwards.
//
// Narrowing: getArray/getFloat return T only after checking [min, max], so
// with bounds inside T's range the narrowed value is finite. checkFinite alone
// only guarantees a finite f64: a caller that stores into f32/f16 itself must
// also bound the value to that type (e.g. checkArg with +-floatMax(f32)).
//
// Node ownership: getArray, getFloat and mapGetPlanes free `nodes` on error
// (pass `&.{}` when a defer already owns them); checkFinite, checkArg,
// checkArrayLen and Maps only set the error, the caller frees.
// ---------------------------------------------------------------------------

/// Set a formatted error on any out map (no print_buf needed: VapourSynth
/// copies the message, so a stack buffer is fine).
pub fn setArgError(out: anytype, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const msg = std.mem.printSentinel(&buf, fmt, args, 0) catch "invalid argument";
    out.setError(msg);
}

/// Number for an error message: `{d}` for ordinary magnitudes (the format the
/// existing messages and tests use), scientific notation for extreme floats
/// (`{d}` would spell 1e300 out as 300 digits).
fn numStr(buf: []u8, v: anytype) []const u8 {
    if (@typeInfo(@TypeOf(v)) == .float) {
        const a = @abs(v);
        if (math.isFinite(v) and a != 0 and (a >= 1e15 or a < 1e-6)) {
            return std.fmt.bufPrint(buf, "{e}", .{v}) catch "?";
        }
    }
    return std.fmt.bufPrint(buf, "{d}", .{v}) catch "?";
}

/// Reject NaN and +-inf: "<filter>: <key>[<i>] value nan is not a finite
/// number." Integers always pass. Call before any other use of the value.
pub fn checkFinite(value: anytype, filter_name: []const u8, comptime key: []const u8, index: ?usize, out: anytype) error{invalidArgument}!void {
    if (@typeInfo(@TypeOf(value)) != .float) return;
    if (math.isFinite(value)) return;
    if (index) |i| {
        setArgError(out, "{s}: {s}[{d}] value {d} is not a finite number.", .{ filter_name, key, i, value });
    } else {
        setArgError(out, "{s}: {s} value {d} is not a finite number.", .{ filter_name, key, value });
    }
    return error.invalidArgument;
}

/// checkFinite, then `min <= value <= max`, with the messages getArray has
/// always used ("<key> value 3 is above maximum 2.").
pub fn checkArg(comptime T: type, value: T, min: T, max: T, filter_name: []const u8, comptime key: []const u8, out: anytype) error{invalidArgument}!void {
    try checkFinite(value, filter_name, key, null, out);
    var vb: [64]u8 = undefined;
    var bb: [64]u8 = undefined;
    if (value < min) {
        setArgError(out, "{s}: {s} value {s} is below minimum {s}.", .{ filter_name, key, numStr(&vb, value), numStr(&bb, min) });
        return error.invalidArgument;
    }
    if (value > max) {
        setArgError(out, "{s}: {s} value {s} is above maximum {s}.", .{ filter_name, key, numStr(&vb, value), numStr(&bb, max) });
        return error.invalidArgument;
    }
}

/// Reject arrays longer than `max_len`: "<key> has too many elements (got 4,
/// max 3)." Without it, parsers that read indices 0..2 silently ignore a
/// fourth value — non-finite or not.
pub fn checkArrayLen(in: anytype, comptime key: [:0]const u8, max_len: usize, filter_name: []const u8, out: anytype) error{invalidArgument}!void {
    const len: usize = @intCast(in.numElements(key) orelse 0);
    if (len > max_len) {
        setArgError(out, "{s}: {s} has too many elements (got {d}, max {d}).", .{ filter_name, key, len, max_len });
        return error.invalidArgument;
    }
}

/// f64 for float T (validate before narrowing), T otherwise.
fn ReadT(comptime T: type) type {
    return if (@typeInfo(T) == .float) f64 else T;
}

/// Optional array argument (up to 3 values, the last one repeats; `default`
/// when absent), each value checked with checkArg. Frees `nodes` on error.
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
    errdefer for (nodes) |node| {
        if (node) |n| zapi.freeNode(n);
    };

    const R = ReadT(T);
    var array: [3]T = undefined;
    try checkArrayLen(in, key, 3, filter_name, out);
    const len = in.numElements(key) orelse 0;
    var i: u8 = 0;
    while (i < 3) : (i += 1) {
        if (i < len) {
            const v: R = in.getValue2(R, key, i).?;
            try checkArg(R, v, min, max, filter_name, key, out);
            array[i] = if (R == T) v else @floatCast(v);
        } else if (i == 0) {
            try checkArg(T, default, min, max, filter_name, key, out);
            array[i] = default;
        } else {
            array[i] = array[i - 1];
        }
    }

    return array;
}

/// Optional scalar float argument (`default` when absent) checked with
/// checkArg. Frees `nodes` on error.
pub fn getFloat(
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
) !T {
    comptime std.debug.assert(@typeInfo(T) == .float);
    errdefer for (nodes) |node| {
        if (node) |n| zapi.freeNode(n);
    };
    const v: f64 = in.getValue(f64, key) orelse default;
    try checkArg(f64, v, min, max, filter_name, key, out);
    return @floatCast(v);
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
        try checkFinite(val, self.name, key, null, self.out);
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
                try checkFinite(val, self.name, key, @min(i, a.len - 1), self.out);
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

/// [luma, chroma] stride
pub fn strideFromVi(vi: *const vs.VideoInfo) [2]u32 {
    const n: u32 = vsFrameAlignmentT(@intCast(vi.format.bytesPerSample));
    const ssw: u3 = @intCast(vi.format.subSamplingW);
    return .{
        @intCast(ceilN(@intCast(vi.width), n)),
        @intCast(ceilN(@intCast(vi.width >> ssw), n)),
    };
}

pub fn vsFrameAlignmentT(div: u32) u32 {
    return @divExact(vsFrameAlignment(), div);
}

/// VapourSynth aligns every frame's plane rows to `VSFrame::alignment` bytes,
/// which it picks once at startup from the *running* CPU: 64 bytes when AVX-512F
/// is available, otherwise 32 (and 32 on non-x86) — see `alignmentHelper()` in vscore.cpp.
pub fn vsFrameAlignment() u32 {
    if (comptime (builtin.target.cpu.arch == .x86_64 or builtin.target.cpu.arch == .x86)) {
        // Replicates cpufeatures.cpp's `avx512_f` derivation exactly.
        // 1. CPUID.1:ECX must report OSXSAVE(27) + AVX(28).
        const leaf1 = cpuid(1, 0);
        const osxsave_avx: u32 = (1 << 27) | (1 << 28);
        if ((leaf1.ecx & osxsave_avx) != osxsave_avx) return 32;

        // 2. The OS must have enabled AVX state (XCR0 XMM+YMM bits).
        const xcr0 = getXCR0();
        if ((xcr0 & 0x06) != 0x06) return 32;

        // 3. AVX-512F (CPUID.7.0:EBX bit 16) AND OS-enabled AVX-512 state (XCR0
        //    opmask+ZMM bits) => 64-byte frames, matching VSFrame::alignment.
        const leaf7 = cpuid(7, 0);
        if ((leaf7.ebx & (1 << 16)) != 0 and (xcr0 & 0xE0) == 0xE0) return 64;
    }
    return 32;
}

const CpuidLeaf = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32, subleaf: u32) CpuidLeaf {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (subleaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

/// Read XCR0 (low 32 bits hold every AVX / AVX-512 state bit we test).
fn getXCR0() u32 {
    return asm volatile (
        \\ xor %%ecx, %%ecx
        \\ xgetbv
        : [_] "={eax}" (-> u32),
        :
        : .{ .edx = true, .ecx = true });
}

pub inline fn ceilN(x: u32, n: u32) u32 {
    return (x + (n - 1)) & ~(n - 1);
}
