const std = @import("std");
const math = std.math;

const filter_int = @import("../filters/deband_int.zig");
const filter_float = @import("../filters/deband_float.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;
const Maps = hz.Maps;

const allocator = std.heap.c_allocator;
pub const filter_name = "Deband";

const FRAME_LUT_ALIGNMENT = std.simd.suggestVectorLength(u8) orelse 16;
const INTERNAL_BIT_DEPTH: i32 = 16;
const TV_RANGE_Y_MIN: i32 = 16 << (INTERNAL_BIT_DEPTH - 8);
const TV_RANGE_Y_MAX: i32 = 235 << (INTERNAL_BIT_DEPTH - 8);
const TV_RANGE_C_MIN: i32 = TV_RANGE_Y_MIN;
const TV_RANGE_C_MAX: i32 = 240 << (INTERNAL_BIT_DEPTH - 8);
const FULL_RANGE_Y_MIN: i32 = 0;
const FULL_RANGE_Y_MAX: i32 = (1 << INTERNAL_BIT_DEPTH) - 1;
const FULL_RANGE_C_MIN: i32 = FULL_RANGE_Y_MIN;
const FULL_RANGE_C_MAX: i32 = FULL_RANGE_Y_MAX;
const DEFAULT_RANDOM_PARAM: f64 = 1.0;

pub inline fn ceilN(x: anytype, n: anytype) @TypeOf(x) {
    return (x + (n - 1)) & ~(@as(@TypeOf(x), n) - 1);
}

pub const VEC_SIZE = 8;
const i32v = @Vector(VEC_SIZE, i32);
const u32v = @Vector(VEC_SIZE, u32);
const u16v = @Vector(VEC_SIZE, u16);
const i16v = @Vector(VEC_SIZE, i16);
const f32v = @Vector(VEC_SIZE, f32);
const boolv = @Vector(VEC_SIZE, bool);
const vec1_i32: i32v = @splat(1);
const vec2_i32: i32v = @splat(2);
const vec4_i32: i32v = @splat(4);

pub const RandAlgo = enum(i32) {
    old = 0,
    uniform = 1,
    gaussian = 2,
};

pub const Mode = enum(i32) {
    m1 = 1,
    m2 = 2,
    m3 = 3,
    m4 = 4,
    m5 = 5,
    m6 = 6,
    m7 = 7,
};

const ArrT = union(enum) {
    u: [3]u16,
    f: [3]f32,
};

pub const Data = struct {
    node: *vs.Node = undefined,
    vi: *const vs.VideoInfo = undefined,
    tb: TempBuff = .{},

    range: i32 = 15,
    thr: ArrT = undefined,
    thr1: ArrT = undefined,
    thr2: ArrT = undefined,
    grain: ArrT = undefined,
    sample_mode: Mode = .m2,
    seed: i32 = 0,
    blur_first: bool = true,
    dynamic_grain: bool = false,
    keep_tv_range: bool = false,
    random_algo_ref: RandAlgo = .uniform,
    random_algo_grain: RandAlgo = .uniform,
    random_param_ref: f64 = 1.0,
    random_param_grain: f64 = 1.0,
    angle_boost: f32 = 1.5,
    max_angle: f32 = 0.15,
    ssw: u3 = 0,
    ssh: u3 = 0,
    pixel_max: [3]i32 = .{ FULL_RANGE_Y_MAX, FULL_RANGE_C_MAX, FULL_RANGE_C_MAX },
    pixel_min: [3]i32 = .{ FULL_RANGE_Y_MIN, FULL_RANGE_C_MIN, FULL_RANGE_C_MIN },
    deband: [3]bool = .{ true, true, true },
    add_grain: [3]bool = .{ true, true, true },
    process_plane: [3]bool = .{ true, true, true },

    pub fn setData(d: *Data, m: *const Maps) !void {
        const thr_in: [3]f64 = try m.getArray("thr", 3, .{ 0.99, 0.99, 0.99 }, 0, 255);
        const thr1_in: [3]f64 = try m.getArray("thr1", 3, thr_in, 0, 255);
        const thr2_in: [3]f64 = try m.getArray("thr2", 3, thr_in, 0, 255);
        const grain_in: [3]f64 = try m.getArray("grain", 2, .{ 0, 0, 0 }, 0, 255);
        const sample_mode = try m.getValue(i32, "sample_mode", @intFromEnum(d.sample_mode), 1, 7);

        d.range = try m.getValue(i32, "range", d.range, 0, 255);
        d.seed = try m.getValue(i32, "seed", d.seed, math.minInt(i32), math.maxInt(i32));
        d.blur_first = m.in.getBool("blur_first") orelse d.blur_first;
        d.dynamic_grain = m.in.getBool("dynamic_grain") orelse d.dynamic_grain;
        d.keep_tv_range = m.in.getBool("keep_tv_range") orelse d.keep_tv_range;
        d.angle_boost = try m.getValue(f32, "angle_boost", d.angle_boost, 0, math.maxInt(u16));
        d.max_angle = try m.getValue(f32, "max_angle", d.max_angle, 0, 1);
        d.random_param_ref = try m.getValue(f64, "random_param_ref", d.random_param_ref, 0, 255);
        d.random_param_grain = try m.getValue(f64, "random_param_grain", d.random_param_grain, 0, 255);
        const random_algo_ref = try m.getValue(i32, "random_algo_ref", @intFromEnum(d.random_algo_ref), 0, 2);
        const random_algo_grain = try m.getValue(i32, "random_algo_grain", @intFromEnum(d.random_algo_grain), 0, 2);

        d.thr = d.scaleValue(&thr_in);
        d.thr1 = d.scaleValue(&thr1_in);
        d.thr2 = d.scaleValue(&thr2_in);
        d.grain = d.scaleValue(&grain_in);

        d.sample_mode = @enumFromInt(sample_mode);
        d.dynamic_grain = d.dynamic_grain and (grain_in[0] > 0 or grain_in[1] > 0);
        d.random_algo_ref = @enumFromInt(random_algo_ref);
        d.random_algo_grain = @enumFromInt(random_algo_grain);

        for (0..3) |i| {
            d.deband[i] = thr_in[i] > 0;
            d.add_grain[i] = grain_in[@min(i, 1)] > 0;
            d.process_plane[i] = d.deband[i] or d.add_grain[i];
        }

        d.ssw = @intCast(d.vi.format.subSamplingW);
        d.ssh = @intCast(d.vi.format.subSamplingH);

        if (d.keep_tv_range and (d.vi.format.colorFamily == .YUV)) {
            d.pixel_min = .{ TV_RANGE_Y_MIN, TV_RANGE_C_MIN, TV_RANGE_C_MIN };
            d.pixel_max = .{ TV_RANGE_Y_MAX, TV_RANGE_C_MAX, TV_RANGE_C_MAX };
        }
    }

    fn scaleValue(d: *const Data, in: []const f64) ArrT {
        const peak16: f64 = (1 << 16) - 1;
        if (d.vi.format.sampleType == .Integer) {
            var out: [3]u16 = undefined;
            for (0..3) |i| {
                out[i] = @intFromFloat(in[i] * peak16 / 255.0 + 0.5);
            }
            return ArrT{ .u = out };
        } else {
            var out: [3]f32 = undefined;
            for (0..3) |i| {
                out[i] = @floatCast(in[i] / 255.0);
            }
            return ArrT{ .f = out };
        }
    }
};

const TempBuff = struct {
    ref1: [3][]u16 = .{ &.{}, &.{}, &.{} },
    ref2: [3][]u16 = .{ &.{}, &.{}, &.{} },
    grain_int: [3][]i16 = .{ &.{}, &.{}, &.{} },
    grain_float: [3][]f32 = .{ &.{}, &.{}, &.{} },
    grain_offsets: []u32 = &.{},

    pub fn init(d: *const Data) !TempBuff {
        var tb = TempBuff{};
        try tb.initFrameLuts(d);
        return tb;
    }

    fn initFrameLuts(self: *TempBuff, d: *const Data) !void {
        const vi = d.vi;
        const width: i32 = vi.width;
        const height: i32 = vi.height;
        const uwidth: u32 = @intCast(width);
        const uheight: u32 = @intCast(height);
        const mask_w: i32 = (@as(i32, 1) << d.ssw) - 1;
        const mask_h: i32 = (@as(i32, 1) << d.ssh) - 1;
        const is_float = d.vi.format.sampleType == .Float;
        const alignment: u32 = if (is_float) (FRAME_LUT_ALIGNMENT / @sizeOf(f32)) else (FRAME_LUT_ALIGNMENT / @sizeOf(u16));

        const y_stride: u32 = ceilN(uwidth, alignment);
        const c_width: u32 = uwidth >> d.ssw;
        const c_height: u32 = uheight >> d.ssh;
        const c_stride: u32 = ceilN(c_width, alignment);
        const y_size: u32 = y_stride * uheight;
        const c_size: u32 = c_stride * c_height;

        const sizes = [_]u32{ y_size, c_size, c_size };
        for (0..2) |i| {
            self.ref1[i] = try allocator.alignedAlloc(u16, vszip.alignment, sizes[i]);
            self.ref2[i] = try allocator.alignedAlloc(u16, vszip.alignment, sizes[i]);
            @memset(self.ref1[i], 0);
            @memset(self.ref2[i], 0);
        }

        self.ref1[2] = self.ref1[1];
        self.ref2[2] = self.ref2[1];

        var seed = @as(i32, @bitCast(@as(u32, 0x92D68CA2))) - d.seed;
        seed ^= (width << 16) ^ height;
        seed ^= (@as(i32, vi.numFrames) << 16) ^ @as(i32, vi.numFrames);
        var y: u32 = 0;
        while (y < height) : (y += 1) {
            const iy: i32 = @intCast(y);
            const y_row = y * y_stride;
            const c_row = (y >> d.ssh) * c_stride;

            var x: u32 = 0;
            var cx: u32 = 0;
            while (x < width) : (x += 1) {
                const ix: i32 = @intCast(x);
                var val1: u16 = 0;
                var val2: u16 = 0;

                // Consume grain random value to maintain seed sequence compatibility
                _ = randomValue(d.random_algo_grain, &seed, 1, d.random_param_grain);
                const x_range = minMulti(&[_]i32{ d.range, ix, width - ix - 1, -1 });
                const y_range = minMulti(&[_]i32{ d.range, iy, height - iy - 1, -1 });
                var cur_range: i32 = 0;
                switch (d.sample_mode) {
                    .m1 => cur_range = y_range,
                    .m3 => cur_range = x_range,
                    .m2, .m4, .m5, .m6, .m7 => cur_range = minMulti(&[_]i32{ x_range, y_range, -1 }),
                }

                if (cur_range > 0) {
                    const tmp1 = randomValue(d.random_algo_ref, &seed, cur_range, d.random_param_ref);
                    const tmp2 = if (d.sample_mode == .m2) randomValue(d.random_algo_ref, &seed, cur_range, d.random_param_ref) else 0;
                    val1 = @intCast(@abs(tmp1));
                    val2 = @intCast(@abs(tmp2));
                }

                switch (d.sample_mode) {
                    .m1 => {
                        self.ref1[0][y_row + x] = @intCast(val1 * y_stride);
                        self.ref2[0][y_row + x] = 0;
                    },
                    .m2 => {
                        self.ref1[0][y_row + x] = @intCast(y_stride * val2 + val1);
                        self.ref2[0][y_row + x] = @intCast(@abs(@as(i32, val2) - @as(i32, @intCast(y_stride)) * val1));
                    },
                    .m3 => {
                        self.ref1[0][y_row + x] = val1;
                        self.ref2[0][y_row + x] = 0;
                    },
                    .m4, .m5, .m6, .m7 => {
                        self.ref1[0][y_row + x] = @intCast(val1 * y_stride);
                        self.ref2[0][y_row + x] = val1;
                    },
                }

                if (((ix & mask_w) == 0) and ((iy & mask_h) == 0)) {
                    const val1_cw = val1 >> d.ssw;
                    const val1_ch = val1 >> d.ssh;
                    const val2_ch = val2 >> d.ssh;
                    const val2_cw = val2 >> d.ssw;
                    switch (d.sample_mode) {
                        .m1 => {
                            self.ref1[1][c_row + cx] = @intCast(val1_ch * c_stride);
                            self.ref2[1][c_row + cx] = 0;
                        },
                        .m2 => {
                            self.ref1[1][c_row + cx] = @intCast(c_stride * val2_ch + val1_cw);
                            self.ref2[1][c_row + cx] = @intCast(@abs(@as(i32, val2_cw) - @as(i32, @intCast(c_stride)) * val1_ch));
                        },
                        .m3 => {
                            self.ref1[1][c_row + cx] = val1_cw;
                            self.ref2[1][c_row + cx] = 0;
                        },
                        .m4, .m5, .m6, .m7 => {
                            self.ref1[1][c_row + cx] = @intCast(val1_ch * c_stride);
                            self.ref2[1][c_row + cx] = val1_cw;
                        },
                    }
                    // Consume grain random values for chroma to maintain seed sequence
                    _ = randomValue(d.random_algo_grain, &seed, 1, d.random_param_grain);
                    _ = randomValue(d.random_algo_grain, &seed, 1, d.random_param_grain);

                    cx += 1;
                }
            }
        }

        var item_count = width;
        item_count += 255;
        item_count &= @bitCast(@as(u32, 0xffffff80));
        item_count *= height;

        const multiplier: i32 = if (d.dynamic_grain) 3 else 1;
        const total_items: u32 = @intCast(item_count * multiplier);

        for (0..2) |i| {
            if (!d.add_grain[i]) continue;
            if (d.vi.format.sampleType == .Integer) {
                self.grain_int[i] = try allocator.alignedAlloc(i16, vszip.alignment, total_items);
                fillGrainBuffer(self.grain_int[i], d.random_algo_grain, &seed, d.random_param_grain, d.grain.u[i]);
            } else {
                self.grain_float[i] = try allocator.alignedAlloc(f32, vszip.alignment, total_items);
                fillGrainBufferFloat(self.grain_float[i], d.random_algo_grain, &seed, d.random_param_grain, d.grain.f[i]);
            }
        }

        self.grain_int[2] = self.grain_int[1];
        self.grain_float[2] = self.grain_float[1];
        if (d.dynamic_grain) {
            self.grain_offsets = try allocator.alloc(u32, @intCast(vi.numFrames));
            var i: u32 = 0;
            while (i < self.grain_offsets.len) : (i += 1) {
                var offset = item_count + randomValue(.uniform, &seed, item_count, DEFAULT_RANDOM_PARAM);
                offset &= @bitCast(@as(u32, 0xfffffff0));
                self.grain_offsets[i] = @intCast(offset);
            }
        }
    }

    fn deinit(self: *TempBuff, d: *const Data) void {
        if (d.dynamic_grain) allocator.free(self.grain_offsets);
        for (0..2) |i| {
            allocator.free(self.ref1[i]);
            allocator.free(self.ref2[i]);

            if (!d.add_grain[i]) continue;
            if (d.vi.format.sampleType == .Integer)
                allocator.free(self.grain_int[i])
            else
                allocator.free(self.grain_float[i]);
        }
    }
};

fn minMulti(values: []const i32) i32 {
    var result = values[0];
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const current = values[i];
        if (current < 0) break;
        if (current < result) result = current;
    }
    return result;
}

fn randToDouble(rand_num: i32) f64 {
    var raw: u64 = @as(u64, @as(u32, @bitCast(rand_num))) & @as(u64, 0xffffffff);
    raw = (raw << 20) | (raw >> 12);
    raw |= 0x3ff0_0000_0000_0000;
    const val = @as(f64, @bitCast(raw));
    return (val - 1.0) * 2.0 - 1.0;
}

fn randOld(seed: *i32) f64 {
    const useed: u32 = @bitCast(seed.*);
    const tmp: u32 = (((useed << 13) ^ useed) >> 17) ^ (useed << 13) ^ useed;
    seed.* = @bitCast(32 *% tmp ^ tmp);
    return randToDouble(seed.*);
}

fn randUniform(seed: *i32) f64 {
    seed.* = 1664525 *% seed.* +% 1013904223;
    return randToDouble(seed.*);
}

fn randGaussian(seed: *i32, param: f64) f64 {
    while (true) {
        var x: f64 = 0;
        var y: f64 = 0;
        var r2: f64 = 0;
        while (true) {
            x = randUniform(seed);
            y = randUniform(seed);
            r2 = x * x + y * y;
            if (r2 <= 1.0 and r2 != 0.0) break;
        }
        const value = param * y * @sqrt(-2.0 * @log(r2) / r2);
        if (value > -1.0 and value < 1.0) return value;
    }
}

fn randomValue(algo: RandAlgo, seed: *i32, range: i32, param: f64) i32 {
    const value = switch (algo) {
        .old => randOld(seed),
        .uniform => randUniform(seed),
        .gaussian => randGaussian(seed, param),
    };

    std.debug.assert(value >= -1.0 and value <= 1.0);
    return @intFromFloat(@round(value * @as(f64, @floatFromInt(range))));
}

fn fillGrainBuffer(buffer: []i16, algo: RandAlgo, seed: *i32, param: f64, range: i32) void {
    var i: usize = 0;
    while (i < buffer.len) : (i += 1) {
        buffer[i] = @intCast(randomValue(algo, seed, range, param));
    }
}

fn randomValueFloat(algo: RandAlgo, seed: *i32, range: f32, param: f64) f32 {
    const value = switch (algo) {
        .old => randOld(seed),
        .uniform => randUniform(seed),
        .gaussian => randGaussian(seed, param),
    };

    std.debug.assert(value >= -1.0 and value <= 1.0);
    return @floatCast(value * range);
}

fn fillGrainBufferFloat(buffer: []f32, algo: RandAlgo, seed: *i32, param: f64, range: f32) void {
    var i: usize = 0;
    while (i < buffer.len) : (i += 1) {
        buffer[i] = randomValueFloat(algo, seed, range, param);
    }
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.tb.deinit(d);
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    var print_buf = [_]u8{0} ** 512;

    const zapi = ZAPI.init(vsapi, core, null);
    const zin = zapi.initZMap(in);
    const zout = zapi.initZMap2(out, &print_buf);
    const maps = Maps.init(&zin, &zout, filter_name);

    d.node, const vi_in = zin.getNodeVi("clip").?;
    if (!vsh.isConstantVideoFormat(vi_in)) {
        zout.setError2("{s}: clip must have constant format [{}x{}-{}]", .{ filter_name, vi_in.width, vi_in.height, vi_in.format.colorFamily });
        zapi.freeNode(d.node);
        return;
    }

    if (vi_in.format.sampleType == .Float and vi_in.format.bitsPerSample != 32) {
        zout.setError2("{s}: only 32-bit format is supported when float clip", .{filter_name});
        zapi.freeNode(d.node);
        return;
    }

    if (vi_in.format.bitsPerSample < 16) {
        d.node = hz.bitDepth(16, d.node, .none, &zapi);
    }

    d.vi = zapi.getVideoInfo(d.node);
    d.setData(&maps) catch {
        zapi.freeNode(d.node);
        return;
    };

    d.tb = TempBuff.init(&d) catch {
        zout.setError2("{s}: failed to initialize temp buffer (out of memory)", .{filter_name});
        zapi.freeNode(d.node);
        return;
    };

    const data = allocator.create(Data) catch unreachable;
    data.* = d;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    const gf: vs.FilterGetFrame = comptimeGetFrame(
        d.sample_mode,
        d.blur_first,
        d.add_grain,
        d.vi.format.numPlanes,
        d.vi.format.sampleType,
    );

    var out_node = zapi.createVideoFilter2(filter_name, d.vi, gf, free, .Parallel, &deps, data).?;
    if (vi_in.format.bitsPerSample < 16) {
        const bits: u32 = @intCast(vi_in.format.bitsPerSample);
        out_node = hz.bitDepth(bits, out_node, .error_diffusion, &zapi);
    }

    _ = zout.consumeNode("clip", out_node, .Replace);
}

fn comptimeGetFrame(mode: Mode, blur_first: bool, add_grain: [3]bool, num_planes: i32, st: vs.SampleType) vs.FilterGetFrame {
    const comptime_grain: [4][3]bool = .{
        .{ true, false, false },
        .{ false, true, true },
        .{ true, true, true },
        .{ false, false, false },
    };

    var i: u32 = 0;
    const g_idx: u32 = while (i < comptime_grain.len) : (i += 1) {
        if (std.mem.eql(bool, &comptime_grain[i], &add_grain)) break i;
    } else 0;

    const get_frame: vs.FilterGetFrame = switch (num_planes) {
        inline 1...3 => |np| switch (mode) {
            inline else => |m| switch (g_idx) {
                inline 0...3 => |gi| switch (blur_first) {
                    inline false, true => |bf| switch (st) {
                        .Integer => filter_int.F3KDB(m, bf, comptime_grain[gi], np).getFrame,
                        .Float => filter_float.F3KDB(m, bf, comptime_grain[gi], np).getFrame,
                    },
                },
                else => unreachable,
            },
        },
        else => unreachable,
    };

    return get_frame;
}
