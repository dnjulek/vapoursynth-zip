//! Deband (neo_f3kdb) setup: modes, random generators and the per-clip ref /
//! grain tables. Host-independent part of vszip's vapoursynth/deband.zig;
//! the tables are built for the canonical strides in `Data.strides`.

const std = @import("std");
const math = std.math;
const hz = @import("../common.zig");
const angle = @import("deband_angle.zig");

const allocator = hz.kernel_allocator;

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
    // Geometry of the processed (16-bit or float) clip and the canonical plane
    // strides (elements) the ref tables are built for: frames handed to the
    // kernels must use exactly these strides.
    width: i32 = 0,
    height: i32 = 0,
    num_frames: i32 = 0,
    is_int: bool = true,
    is_yuv: bool = false,
    num_planes: u32 = 3,
    strides: [2]u32 = .{ 0, 0 },
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
    pixel_maxf: [3]f32 = .{ 1.0, 1.0, 1.0 },
    pixel_minf: [3]f32 = .{ 0.0, 0.0, 0.0 },
    deband: [3]bool = .{ true, true, true },
    add_grain: [3]bool = .{ true, true, true },
    process_plane: [3]bool = .{ true, true, true },

    /// Parsed user values; `configure` derives everything else exactly as
    /// vszip's setData did.
    pub const Inputs = struct {
        thr: [3]f64,
        thr1: [3]f64,
        thr2: [3]f64,
        grain: [3]f64,
        sample_mode: i32,
        range: i32,
        seed: i32,
        blur_first: bool,
        dynamic_grain: bool,
        keep_tv_range: bool,
        angle_boost: f32,
        max_angle: f32,
        random_param_ref: f64,
        random_param_grain: f64,
        random_algo_ref: i32,
        random_algo_grain: i32,
    };

    pub fn configure(d: *Data, in: Inputs) void {
        d.range = in.range;
        d.seed = in.seed;
        d.blur_first = in.blur_first;
        d.dynamic_grain = in.dynamic_grain;
        d.keep_tv_range = in.keep_tv_range;
        d.angle_boost = in.angle_boost;
        d.max_angle = in.max_angle;
        d.random_param_ref = in.random_param_ref;
        d.random_param_grain = in.random_param_grain;

        d.thr = d.scaleValue(&in.thr);
        d.thr1 = d.scaleValue(&in.thr1);
        d.thr2 = d.scaleValue(&in.thr2);
        d.grain = d.scaleValue(&in.grain);

        d.sample_mode = @fromBackingInt(@intCast(in.sample_mode));
        d.dynamic_grain = d.dynamic_grain and (in.grain[0] > 0 or in.grain[1] > 0);
        d.random_algo_ref = @fromBackingInt(@intCast(in.random_algo_ref));
        d.random_algo_grain = @fromBackingInt(@intCast(in.random_algo_grain));

        for (0..3) |i| {
            d.deband[i] = in.thr[i] > 0;
            d.add_grain[i] = in.grain[@min(i, 1)] > 0;
            d.process_plane[i] = d.deband[i] or d.add_grain[i];
        }

        if (d.keep_tv_range and d.is_yuv) {
            d.pixel_min = .{ TV_RANGE_Y_MIN, TV_RANGE_C_MIN, TV_RANGE_C_MIN };
            d.pixel_max = .{ TV_RANGE_Y_MAX, TV_RANGE_C_MAX, TV_RANGE_C_MAX };
        }

        if (d.is_yuv) {
            d.pixel_minf = .{ 0.0, -0.5, -0.5 };
            d.pixel_maxf = .{ 1.0, 0.5, 0.5 };
        }
    }

    fn scaleValue(d: *const Data, in: []const f64) ArrT {
        const peak16: f64 = (1 << 16) - 1;
        if (d.is_int) {
            var out: [3]u16 = undefined;
            for (0..3) |i| {
                out[i] = @trunc(in[i] * peak16 / 255.0 + 0.5);
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

pub const TempBuff = struct {
    ref1: [3][]i32 = .{ &.{}, &.{}, &.{} },
    ref2: [3][]i32 = .{ &.{}, &.{}, &.{} },
    grain_int: [3][]i16 = .{ &.{}, &.{}, &.{} },
    grain_float: [3][]f32 = .{ &.{}, &.{}, &.{} },
    grain_offsets: []u32 = &.{},
    /// mode 7's angle-plane scratch, shared by concurrent requests
    angles: angle.Pool = .{},

    pub fn init(d: *const Data) !TempBuff {
        var tb = TempBuff{};
        // Unallocated tables are empty slices (and the chroma aliases in slot
        // 2 are never freed), so deinit is valid at any failure point.
        errdefer tb.deinit(d);
        try tb.initFrameLuts(d);
        return tb;
    }

    fn initFrameLuts(self: *TempBuff, d: *const Data) !void {
        const width: i32 = d.width;
        const height: i32 = d.height;
        const strides = d.strides;
        const uheight: u32 = @intCast(height);
        const mask_w: i32 = (@as(i32, 1) << d.ssw) - 1;
        const mask_h: i32 = (@as(i32, 1) << d.ssh) - 1;
        const y_stride: u32 = strides[0];
        const c_height: u32 = uheight >> d.ssh;
        const c_stride: u32 = strides[1];
        const y_size: u32 = y_stride * uheight;
        const c_size: u32 = c_stride * c_height;

        const sizes = [_]u32{ y_size, c_size, c_size };
        for (0..2) |i| {
            self.ref1[i] = try allocator.alloc(i32, sizes[i]);
            self.ref2[i] = try allocator.alloc(i32, sizes[i]);
            @memset(self.ref1[i], 0);
            @memset(self.ref2[i], 0);
        }

        self.ref1[2] = self.ref1[1];
        self.ref2[2] = self.ref2[1];

        const w32: u32 = @bitCast(width);
        const h32: u32 = @bitCast(height);
        const nf32: u32 = @bitCast(@as(i32, d.num_frames));
        var useed: u32 = @as(u32, 0x92D68CA2) -% @as(u32, @bitCast(d.seed));
        useed ^= (w32 << 16) ^ h32;
        useed ^= (nf32 << 16) ^ nf32;
        var seed: i32 = @bitCast(useed);
        var y: u32 = 0;
        while (y < height) : (y += 1) {
            const iy: i32 = @intCast(y);
            const y_row = y * y_stride;
            const c_row = (y >> d.ssh) * c_stride;

            var x: u32 = 0;
            var cx: u32 = 0;
            while (x < width) : (x += 1) {
                const ix: i32 = @intCast(x);
                var val1: i32 = 0;
                var val2: i32 = 0;

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
                    // neo stores refs as `signed char`: info.ref = (signed char)random,
                    // then info.ref = abs(info.ref) which is re-truncated to signed char,
                    // so abs(-128) wraps back to -128 (a NEGATIVE offset that survives).
                    val1 = refEncode(tmp1);
                    val2 = refEncode(tmp2);
                }

                const ystride_i: i32 = @intCast(y_stride);
                switch (d.sample_mode) {
                    .m1 => {
                        self.ref1[0][y_row + x] = val1 * ystride_i;
                        self.ref2[0][y_row + x] = 0;
                    },
                    .m2 => {
                        self.ref1[0][y_row + x] = ystride_i * val2 + val1;
                        // signed ref_pos_2 = val2 - stride*val1 (negative when val1>=1);
                        // process reads src[cur+ref2] / src[cur-ref2] directly.
                        self.ref2[0][y_row + x] = val2 - ystride_i * val1;
                    },
                    .m3 => {
                        self.ref1[0][y_row + x] = val1;
                        self.ref2[0][y_row + x] = 0;
                    },
                    .m4, .m5, .m6, .m7 => {
                        self.ref1[0][y_row + x] = val1 * ystride_i;
                        self.ref2[0][y_row + x] = val1;
                    },
                }

                if (((ix & mask_w) == 0) and ((iy & mask_h) == 0)) {
                    // Arithmetic right shift (val* are signed i32) matches neo's
                    // `signed char >> subsampling`, preserving the -128 sign.
                    const val1_cw = val1 >> d.ssw;
                    const val1_ch = val1 >> d.ssh;
                    const val2_ch = val2 >> d.ssh;
                    const val2_cw = val2 >> d.ssw;
                    const cstride_i: i32 = @intCast(c_stride);
                    switch (d.sample_mode) {
                        .m1 => {
                            self.ref1[1][c_row + cx] = val1_ch * cstride_i;
                            self.ref2[1][c_row + cx] = 0;
                        },
                        .m2 => {
                            self.ref1[1][c_row + cx] = cstride_i * val2_ch + val1_cw;
                            // signed ref_pos_2 for chroma = val2_cw - c_stride*val1_ch
                            self.ref2[1][c_row + cx] = val2_cw - cstride_i * val1_ch;
                        },
                        .m3 => {
                            self.ref1[1][c_row + cx] = val1_cw;
                            self.ref2[1][c_row + cx] = 0;
                        },
                        .m4, .m5, .m6, .m7 => {
                            self.ref1[1][c_row + cx] = val1_ch * cstride_i;
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
            if (!d.add_grain[i]) {
                consumeRandom(total_items, d.random_algo_grain, &seed, d.random_param_grain);
                continue;
            }
            if (d.is_int) {
                self.grain_int[i] = try allocator.alloc(i16, total_items);
                fillGrainBuffer(self.grain_int[i], d.random_algo_grain, &seed, d.random_param_grain, d.grain.u[i]);
            } else {
                self.grain_float[i] = try allocator.alloc(f32, total_items);
                fillGrainBufferFloat(self.grain_float[i], d.random_algo_grain, &seed, d.random_param_grain, d.grain.f[i]);
            }
        }

        self.grain_int[2] = self.grain_int[1];
        self.grain_float[2] = self.grain_float[1];
        if (d.dynamic_grain) {
            self.grain_offsets = try allocator.alloc(u32, @intCast(d.num_frames));
            var i: u32 = 0;
            while (i < self.grain_offsets.len) : (i += 1) {
                var offset = item_count + randomValue(.uniform, &seed, item_count, DEFAULT_RANDOM_PARAM);
                offset &= @bitCast(@as(u32, 0xfffffff0));
                self.grain_offsets[i] = @intCast(offset);
            }
        }
    }

    pub fn deinit(self: *TempBuff, d: *const Data) void {
        self.angles.deinit();
        if (d.dynamic_grain) allocator.free(self.grain_offsets);
        for (0..2) |i| {
            allocator.free(self.ref1[i]);
            allocator.free(self.ref2[i]);

            if (!d.add_grain[i]) continue;
            if (d.is_int)
                allocator.free(self.grain_int[i])
            else
                allocator.free(self.grain_float[i]);
        }
    }
};

fn refEncode(r: i32) i32 {
    const trunc: i8 = @truncate(r);
    const a: u8 = @intCast(@abs(@as(i32, trunc))); // 0..128
    const restore: i8 = @bitCast(a); // 128 -> -128
    return restore;
}

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
    return @round(value * @as(f64, @floatFromInt(range)));
}

fn fillGrainBuffer(buffer: []i16, algo: RandAlgo, seed: *i32, param: f64, range: i32) void {
    var i: usize = 0;
    while (i < buffer.len) : (i += 1) {
        buffer[i] = @intCast(randomValue(algo, seed, range, param));
    }
}

fn consumeRandom(count: u32, algo: RandAlgo, seed: *i32, param: f64) void {
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        _ = randomValue(algo, seed, 0, param);
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
