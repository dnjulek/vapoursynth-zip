const std = @import("std");
const simd = @import("simd.zig");
const ostrom_table = @import("dither_pattern.zig").ostrom_table;

const allocator = std.heap.c_allocator;
const margin = 2;

pub const FmtcMode = enum {
    filter_lite,
    stucki,
    atkinson,
    floyd,
    ostro,

    pub fn numErrLines(comptime self: FmtcMode) u32 {
        return switch (self) {
            .filter_lite, .floyd, .ostro => 1,
            .atkinson, .stucki => 2,
        };
    }
};

const Dir = enum { forward, backward };

inline fn dirIdx(comptime dir: Dir, base: u32, comptime offset: i32) u32 {
    const actual = if (dir == .forward) offset else -offset;
    return @intCast(@as(i32, @intCast(base)) + actual);
}

inline fn pAt(p: anytype, comptime off: comptime_int) @TypeOf(&p[0]) {
    return if (comptime off >= 0) &p[off] else &(p - (-off))[0];
}

inline fn ptrStep(p: anytype, comptime step: comptime_int) @TypeOf(p) {
    return if (comptime step >= 0) p + step else p - (-step);
}

inline fn spreadFilterLiteIntPtr(comptime step: comptime_int, err_i: i32, e0: [*]i16, err_nxt: *i32) void {
    const e1_i: i32 = (err_i + 2) >> 2;
    const e2_i: i32 = err_i - 2 * e1_i;

    err_nxt.* = pAt(e0, step).*;
    const prev: i32 = pAt(e0, -step).*;
    pAt(e0, -step).* = @truncate(prev + e1_i);
    e0[0] = @truncate(e1_i);
    err_nxt.* += e2_i;
}

inline fn spreadFloydIntPtr(comptime step: comptime_int, err_i: i32, e0: [*]i16, err_nxt: *i32) void {
    const e3_i: i32 = (err_i * 4 + 8) >> 4;
    const e5_i: i32 = (err_i * 5 + 8) >> 4;
    const e7_i: i32 = err_i - e3_i - e5_i;

    err_nxt.* = pAt(e0, step).*;
    const prev: i32 = pAt(e0, -step).*;
    pAt(e0, -step).* = @truncate(prev + e3_i);
    const curr: i32 = e0[0];
    e0[0] = @truncate(curr + e5_i);
    pAt(e0, step).* = 0;
    err_nxt.* += e7_i;
}

inline fn spreadOstroIntPtr(comptime step: comptime_int, err_i: i32, src_raw: i32, dif_bits: i32, e0: [*]i16, err_nxt: *i32) void {
    const te = ostrom_table[ostroIdxInt(src_raw, dif_bits)];
    const d: i32 = te.sum;

    const e1_i: i32 = @divTrunc(err_i * te.c0, d);
    const e2_i: i32 = @divTrunc(err_i * te.c1, d);
    const e3_i: i32 = err_i - e1_i - e2_i;

    err_nxt.* = pAt(e0, step).*;
    const prev: i32 = pAt(e0, -step).*;
    pAt(e0, -step).* = @truncate(prev + e2_i);
    e0[0] = @truncate(e3_i);
    err_nxt.* += e1_i;
}

pub const ErrorBuffer = struct {
    line0: []f32,
    line1: []f32,
    line0_i16: []i16,
    line1_i16: []i16,

    pub fn init(n: u32) ErrorBuffer {
        return .{
            .line0 = allocator.alloc(f32, n) catch unreachable,
            .line1 = allocator.alloc(f32, n) catch unreachable,
            .line0_i16 = allocator.alloc(i16, n) catch unreachable,
            .line1_i16 = allocator.alloc(i16, n) catch unreachable,
        };
    }

    pub fn deinit(self: *const ErrorBuffer) void {
        allocator.free(self.line0);
        allocator.free(self.line1);
        allocator.free(self.line0_i16);
        allocator.free(self.line1_i16);
    }

    pub fn clear(self: *const ErrorBuffer) void {
        @memset(self.line0, 0.0);
        @memset(self.line1, 0.0);
        @memset(self.line0_i16, 0);
        @memset(self.line1_i16, 0);
    }
};

pub const IntQuant = struct {
    tmp_shft: u5,
    tmp_invs: u5,
    rcst: i32,
    peak: i32,
    dif_bits: i32,

    pub fn init(bits_in: u6, bits_out: u6) IntQuant {
        const err_res: u6 = 24;
        const dif_bits: u5 = @intCast(bits_in - bits_out);
        const tmp_bits: u6 = if (dif_bits < 6 and bits_in < err_res and bits_out < err_res)
            err_res
        else
            bits_in;
        const tmp_invs: u5 = @intCast(tmp_bits - bits_out);
        return .{
            .tmp_shft = @intCast(tmp_bits - bits_in),
            .tmp_invs = tmp_invs,
            .rcst = @as(i32, 1) << @intCast(tmp_invs - 1),
            .peak = (@as(i32, 1) << @intCast(bits_out)) - 1,
            .dif_bits = dif_bits,
        };
    }
};

const QuantResultInt = struct { pix: i32, err: i32 };

inline fn quantizeInt(comptime shift0: bool, q: IntQuant, src_raw: i32, err_in: i32) QuantResultInt {
    const src_i: i32 = if (shift0) src_raw else src_raw << q.tmp_shft;
    const preq: i32 = src_i + err_in;
    const quant: i32 = (preq + q.rcst) >> q.tmp_invs;
    const pix: i32 = @max(0, @min(quant, q.peak));
    const err: i32 = preq - (quant << q.tmp_invs);
    return .{ .pix = pix, .err = err };
}

const QuantResultFloat = struct { pix: f32, err: f32, src_val: f32 };

inline fn quantizeFloat(comptime T: type, src_raw: T, scale: f32, offset: f32, peak: f32, err_in: f32) QuantResultFloat {
    const src_val: f32 = switch (@typeInfo(T)) {
        .float => @floatCast(src_raw),
        .int => @floatFromInt(src_raw),
        else => @compileError("unsupported source type"),
    };
    const preq: f32 = src_val * scale + offset + err_in;
    const quant = simd.roundNearestEven(preq);
    const pix = @max(0.0, @min(quant, peak));
    return .{ .pix = pix, .err = preq - quant, .src_val = src_val };
}

inline fn ostroIdxInt(src_raw: i32, dif_bits: i32) u32 {
    const t_bits: i32 = 8;
    return @intCast(if (t_bits >= dif_bits)
        (src_raw << @intCast(t_bits - dif_bits)) & 255
    else
        (src_raw >> @intCast(dif_bits - t_bits)) & 255);
}

inline fn ostroIdxFloat(comptime T: type, src_val: f32, scale: f32, offset: f32, dif_bits: i32) u32 {
    if (@typeInfo(T) == .int) {
        return ostroIdxInt(@intFromFloat(src_val), dif_bits);
    } else {
        const src_raw = src_val * scale + offset;
        const idx_i: i32 = @intFromFloat(simd.roundNearestEven(src_raw * 256.0));
        return @intCast(idx_i & 255);
    }
}

inline fn spreadAtkinsonInt(comptime dir: Dir, err_i: i32, line0: []i16, line1: []i16, idx: u32, err_nxt0: *i32, err_nxt1: *i32) void {
    const e1_i: i32 = (err_i + 4) >> 3;

    err_nxt0.* = err_nxt1.* + e1_i;
    err_nxt1.* = line1[dirIdx(dir, idx, 2)] + e1_i;
    const prev: i32 = line0[dirIdx(dir, idx, -1)];
    line0[dirIdx(dir, idx, -1)] = @truncate(prev + e1_i);
    const curr: i32 = line0[idx];
    line0[idx] = @truncate(curr + e1_i);
    const next: i32 = line0[dirIdx(dir, idx, 1)];
    line0[dirIdx(dir, idx, 1)] = @truncate(next + e1_i);
    line1[idx] = @truncate(e1_i);
}

inline fn spreadStuckiInt(comptime dir: Dir, err_i: i32, line0: []i16, line1: []i16, idx: u32, err_nxt0: *i32, err_nxt1: *i32) void {
    const m: i32 = @divTrunc(err_i << 4, 42);
    const e1_i: i32 = (m + 8) >> 4;
    const e2_i: i32 = (m + 4) >> 3;
    const e4_i: i32 = (m + 2) >> 2;
    const sum: i32 = (e1_i << 1) + ((e2_i + e4_i) << 2);
    const e8_i: i32 = (err_i - sum + 1) >> 1;

    err_nxt0.* = err_nxt1.* + e8_i;
    err_nxt1.* = line1[dirIdx(dir, idx, 2)] + e4_i;

    const l0_m2: i32 = line0[dirIdx(dir, idx, -2)];
    line0[dirIdx(dir, idx, -2)] = @truncate(l0_m2 + e2_i);
    const l0_m1: i32 = line0[dirIdx(dir, idx, -1)];
    line0[dirIdx(dir, idx, -1)] = @truncate(l0_m1 + e4_i);
    const l0_0: i32 = line0[idx];
    line0[idx] = @truncate(l0_0 + e8_i);
    const l0_p1: i32 = line0[dirIdx(dir, idx, 1)];
    line0[dirIdx(dir, idx, 1)] = @truncate(l0_p1 + e4_i);
    const l0_p2: i32 = line0[dirIdx(dir, idx, 2)];
    line0[dirIdx(dir, idx, 2)] = @truncate(l0_p2 + e2_i);

    const l1_m2: i32 = line1[dirIdx(dir, idx, -2)];
    line1[dirIdx(dir, idx, -2)] = @truncate(l1_m2 + e1_i);
    const l1_m1: i32 = line1[dirIdx(dir, idx, -1)];
    line1[dirIdx(dir, idx, -1)] = @truncate(l1_m1 + e2_i);
    const l1_0: i32 = line1[idx];
    line1[idx] = @truncate(l1_0 + e4_i);
    const l1_p1: i32 = line1[dirIdx(dir, idx, 1)];
    line1[dirIdx(dir, idx, 1)] = @truncate(l1_p1 + e2_i);
    line1[dirIdx(dir, idx, 2)] = @truncate(e1_i);
}

inline fn spreadFilterLiteFloat(comptime dir: Dir, err: f32, buf: []f32, idx: u32, err_nxt: *f32) void {
    const e1 = err * 0.25;
    const e2 = err * 0.5;

    err_nxt.* = buf[dirIdx(dir, idx, 1)];
    buf[dirIdx(dir, idx, -1)] += e1;
    buf[idx] = e1;
    err_nxt.* += e2;
}

inline fn spreadFloydFloat(comptime dir: Dir, err: f32, buf: []f32, idx: u32, err_nxt: *f32) void {
    const e3 = err * 0.25;
    const e5 = err * (5.0 / 16.0);
    const e7 = err * (7.0 / 16.0);

    err_nxt.* = buf[dirIdx(dir, idx, 1)];
    buf[dirIdx(dir, idx, -1)] += e3;
    buf[idx] += e5;
    buf[dirIdx(dir, idx, 1)] = 0.0;
    err_nxt.* += e7;
}

inline fn spreadAtkinsonFloat(comptime dir: Dir, err: f32, line0: []f32, line1: []f32, idx: u32, err_nxt0: *f32, err_nxt1: *f32) void {
    const e1 = err * 0.125;

    err_nxt0.* = err_nxt1.* + e1;
    err_nxt1.* = line1[dirIdx(dir, idx, 2)] + e1;
    line0[dirIdx(dir, idx, -1)] += e1;
    line0[idx] += e1;
    line0[dirIdx(dir, idx, 1)] += e1;
    line1[idx] = e1;
}

inline fn spreadStuckiFloat(comptime dir: Dir, err: f32, line0: []f32, line1: []f32, idx: u32, err_nxt0: *f32, err_nxt1: *f32) void {
    const e1 = err * (1.0 / 42.0);
    const e2 = err * (2.0 / 42.0);
    const e4 = err * (4.0 / 42.0);
    const e8 = err * (8.0 / 42.0);

    err_nxt0.* = err_nxt1.* + e8;
    err_nxt1.* = line1[dirIdx(dir, idx, 2)] + e4;
    line0[dirIdx(dir, idx, -2)] += e2;
    line0[dirIdx(dir, idx, -1)] += e4;
    line0[idx] += e8;
    line0[dirIdx(dir, idx, 1)] += e4;
    line0[dirIdx(dir, idx, 2)] += e2;
    line1[dirIdx(dir, idx, -2)] += e1;
    line1[dirIdx(dir, idx, -1)] += e2;
    line1[idx] += e4;
    line1[dirIdx(dir, idx, 1)] += e2;
    line1[dirIdx(dir, idx, 2)] = e1;
}

inline fn spreadOstroFloat(comptime T: type, comptime dir: Dir, err: f32, src_val: f32, dif_bits: i32, buf: []f32, idx: u32, err_nxt: *f32, scale: f32, offset: f32) void {
    const te = ostrom_table[ostroIdxFloat(T, src_val, scale, offset, dif_bits)];
    const inv_sum = te.inv_sum;

    const e1 = err * @as(f32, @floatFromInt(te.c0)) * inv_sum;
    const e2 = err * @as(f32, @floatFromInt(te.c1)) * inv_sum;
    const e3 = err - e1 - e2;

    err_nxt.* = buf[dirIdx(dir, idx, 1)];
    buf[dirIdx(dir, idx, -1)] += e2;
    buf[idx] = e3;
    err_nxt.* += e1;
}

const ErrNxtInt = struct { nxt0: i16, nxt1: i16 };
const ErrNxtFloat = struct { nxt0: f32, nxt1: f32 };

pub fn Fmtc(comptime T: type, comptime U: type, comptime mode: FmtcMode) type {
    return struct {
        const num_lines = mode.numErrLines();

        fn processRowIntPtr(comptime dir: Dir, comptime shift0: bool, q: IntQuant, src_row: []const T, dst_row: []U, w: u32, line0: []i16, nxt0_in: i16) i16 {
            const step: comptime_int = if (dir == .forward) 1 else -1;
            const start: usize = if (dir == .forward) 0 else w - 1;

            var err_nxt0: i32 = nxt0_in;
            var sp: [*]const T = src_row.ptr + start;
            var dp: [*]U = dst_row.ptr + start;
            var e0: [*]i16 = line0.ptr + margin + start;

            var rem: u32 = w;
            while (rem != 0) : (rem -= 1) {
                const src_raw: i32 = @intCast(sp[0]);
                const res = quantizeInt(shift0, q, src_raw, err_nxt0);
                dp[0] = @intCast(res.pix);

                switch (comptime mode) {
                    .filter_lite => spreadFilterLiteIntPtr(step, res.err, e0, &err_nxt0),
                    .floyd => spreadFloydIntPtr(step, res.err, e0, &err_nxt0),
                    .ostro => spreadOstroIntPtr(step, res.err, src_raw, q.dif_bits, e0, &err_nxt0),
                    else => unreachable,
                }

                sp = ptrStep(sp, step);
                dp = ptrStep(dp, step);
                e0 = ptrStep(e0, step);
            }

            return @truncate(err_nxt0);
        }

        fn processRowInt(comptime dir: Dir, comptime shift0: bool, q: IntQuant, src_row: []const T, dst_row: []U, w: u32, line0: []i16, line1: []i16, nxt0_in: i16, nxt1_in: i16) ErrNxtInt {
            var err_nxt0: i32 = nxt0_in;
            var err_nxt1: i32 = nxt1_in;

            var i: u32 = 0;
            while (i < w) : (i += 1) {
                const x = if (comptime dir == .forward) i else w - 1 - i;
                const idx: u32 = margin + x;
                const src_raw: i32 = @intCast(src_row[x]);
                const res = quantizeInt(shift0, q, src_raw, err_nxt0);
                dst_row[x] = @intCast(res.pix);

                switch (comptime mode) {
                    .atkinson => spreadAtkinsonInt(dir, res.err, line0, line1, idx, &err_nxt0, &err_nxt1),
                    .stucki => spreadStuckiInt(dir, res.err, line0, line1, idx, &err_nxt0, &err_nxt1),
                    else => unreachable,
                }
            }

            return .{ .nxt0 = @truncate(err_nxt0), .nxt1 = @truncate(err_nxt1) };
        }

        fn processRowFloat(comptime dir: Dir, src_row: []const T, dst_row: []U, w: u32, line0: []f32, line1: []f32, nxt0_in: f32, nxt1_in: f32, scale: f32, offset: f32, peak: f32, dif_bits: i32) ErrNxtFloat {
            var err_nxt0: f32 = nxt0_in;
            var err_nxt1: f32 = nxt1_in;

            var i: u32 = 0;
            while (i < w) : (i += 1) {
                const x = if (comptime dir == .forward) i else w - 1 - i;
                const idx: u32 = margin + x;
                const res = quantizeFloat(T, src_row[x], scale, offset, peak, err_nxt0);
                dst_row[x] = @intFromFloat(res.pix);

                switch (comptime mode) {
                    .filter_lite => spreadFilterLiteFloat(dir, res.err, line0, idx, &err_nxt0),
                    .floyd => spreadFloydFloat(dir, res.err, line0, idx, &err_nxt0),
                    .atkinson => spreadAtkinsonFloat(dir, res.err, line0, line1, idx, &err_nxt0, &err_nxt1),
                    .stucki => spreadStuckiFloat(dir, res.err, line0, line1, idx, &err_nxt0, &err_nxt1),
                    .ostro => spreadOstroFloat(T, dir, res.err, res.src_val, dif_bits, line0, idx, &err_nxt0, scale, offset),
                }
            }

            return .{ .nxt0 = err_nxt0, .nxt1 = err_nxt1 };
        }

        pub fn processPlaneInt(noalias srcp: []const T, noalias dstp: []U, w: u32, h: u32, src_stride: u32, dst_stride: u32, q: IntQuant, err_buf: *const ErrorBuffer) void {
            if (q.tmp_shft == 0) {
                processPlaneIntImpl(true, srcp, dstp, w, h, src_stride, dst_stride, q, err_buf);
            } else {
                processPlaneIntImpl(false, srcp, dstp, w, h, src_stride, dst_stride, q, err_buf);
            }
        }

        fn processPlaneIntImpl(comptime shift0: bool, noalias srcp: []const T, noalias dstp: []U, w: u32, h: u32, src_stride: u32, dst_stride: u32, q: IntQuant, err_buf: *const ErrorBuffer) void {
            var nxt0: i16 = 0;
            var nxt1: i16 = 0;

            var src_row = srcp;
            var dst_row = dstp;
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                const forward = (y & 1) == 0;
                const line0 = if (num_lines == 2 and !forward) err_buf.line1_i16 else err_buf.line0_i16;
                const line1 = if (num_lines == 2 and forward) err_buf.line1_i16 else err_buf.line0_i16;

                if (comptime num_lines == 1) {
                    nxt0 = if (forward)
                        processRowIntPtr(.forward, shift0, q, src_row, dst_row, w, line0, nxt0)
                    else
                        processRowIntPtr(.backward, shift0, q, src_row, dst_row, w, line0, nxt0);
                } else {
                    const result = if (forward)
                        processRowInt(.forward, shift0, q, src_row, dst_row, w, line0, line1, nxt0, nxt1)
                    else
                        processRowInt(.backward, shift0, q, src_row, dst_row, w, line0, line1, nxt0, nxt1);
                    nxt0 = result.nxt0;
                    nxt1 = result.nxt1;
                }

                clearRowEnd(i16, line0, line1, forward, w);
                src_row = src_row[src_stride..];
                dst_row = dst_row[dst_stride..];
            }
        }

        pub fn processPlaneFloat(noalias srcp: []const T, noalias dstp: []U, w: u32, h: u32, src_stride: u32, dst_stride: u32, scale: f32, offset: f32, peak: f32, dif_bits: i32, err_buf: *const ErrorBuffer) void {
            var nxt0: f32 = 0.0;
            var nxt1: f32 = 0.0;

            var src_row = srcp;
            var dst_row = dstp;
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                const forward = (y & 1) == 0;
                const line0 = if (num_lines == 2 and !forward) err_buf.line1 else err_buf.line0;
                const line1 = if (num_lines == 2 and forward) err_buf.line1 else err_buf.line0;

                const result = if (forward)
                    processRowFloat(.forward, src_row, dst_row, w, line0, line1, nxt0, nxt1, scale, offset, peak, dif_bits)
                else
                    processRowFloat(.backward, src_row, dst_row, w, line0, line1, nxt0, nxt1, scale, offset, peak, dif_bits);
                nxt0 = result.nxt0;
                nxt1 = result.nxt1;

                clearRowEnd(f32, line0, line1, forward, w);
                src_row = src_row[src_stride..];
                dst_row = dst_row[dst_stride..];
            }
        }

        fn clearRowEnd(comptime E: type, line0: []E, line1: []E, forward: bool, w: u32) void {
            const zero: E = if (E == f32) 0.0 else 0;
            switch (comptime mode) {
                .filter_lite, .floyd, .ostro => {
                    if (forward) line0[margin + w] = zero else line0[margin - 1] = zero;
                },
                .atkinson => {
                    if (forward) line1[margin + w] = zero else line1[margin - 1] = zero;
                },
                .stucki => {},
            }
        }
    };
}

const f32x8 = @Vector(8, f32);

const FS_ERR_LEFT: f32 = 7.0 / 16.0;
const FS_ERR_TOP_RIGHT: f32 = 3.0 / 16.0;
const FS_ERR_TOP: f32 = 5.0 / 16.0;
const FS_ERR_TOP_LEFT: f32 = 1.0 / 16.0;

pub fn Zimg(comptime T: type, comptime U: type) type {
    return struct {
        fn load8(ptr: []const T) f32x8 {
            const vec: @Vector(8, T) = ptr[0..8].*;
            return switch (@typeInfo(T)) {
                .float => @floatCast(vec),
                .int => @floatFromInt(vec),
                else => @compileError("unsupported source type"),
            };
        }

        fn rotateRightShuffle(v: f32x8, new_val: f32) f32x8 {
            var rotated = @shuffle(f32, v, undefined, [8]i32{ 7, 0, 1, 2, 3, 4, 5, 6 });
            rotated[0] = new_val;
            return rotated;
        }

        fn processRowScalar(
            src_row: []const T,
            dst_row: []U,
            error_top: []const f32,
            error_cur: []f32,
            scale: f32,
            offset: f32,
            peak: f32,
            width: u32,
        ) void {
            var err_left: f32 = error_cur[0];
            var err_top_right: f32 = undefined;
            var err_top: f32 = error_top[0 + 1];
            var err_top_left: f32 = error_top[0];

            var j: u32 = 0;
            while (j < width) : (j += 1) {
                const j_err = j + 1;
                err_top_right = error_top[j_err + 1];

                const val_in: f32 = switch (@typeInfo(T)) {
                    .float => @floatCast(src_row[j]),
                    .int => @floatFromInt(src_row[j]),
                    else => @compileError("unsupported source type"),
                };

                var x: f32 = @mulAdd(f32, val_in, scale, offset);
                var err0: f32 = err_left * FS_ERR_LEFT;
                err0 = @mulAdd(f32, err_top_right, FS_ERR_TOP_RIGHT, err0);
                var err1: f32 = err_top * FS_ERR_TOP;
                err1 = @mulAdd(f32, err_top_left, FS_ERR_TOP_LEFT, err1);

                x += err0 + err1;
                x = @max(0.0, @min(x, peak));

                const q_f: f32 = simd.roundNearestEven(x);
                const new_err = x - q_f;

                dst_row[j] = @intFromFloat(q_f);
                error_cur[j_err] = new_err;

                err_left = new_err;
                err_top_left = err_top;
                err_top = err_top_right;
            }
        }

        fn wavefrontIter(
            v: *f32x8,
            j: u32,
            error_top: []const f32,
            error_cur: []f32,
            peak: f32x8,
            err_left: *f32x8,
            err_top_right: *f32x8,
            err_top: *f32x8,
            err_top_left: *f32x8,
        ) void {
            const j_err = j + 1;
            var err0: f32x8 = err_left.* * @as(f32x8, @splat(FS_ERR_LEFT));
            err0 = @mulAdd(f32x8, err_top_right.*, @splat(FS_ERR_TOP_RIGHT), err0);
            var err1: f32x8 = err_top.* * @as(f32x8, @splat(FS_ERR_TOP));
            err1 = @mulAdd(f32x8, err_top_left.*, @splat(FS_ERR_TOP_LEFT), err1);
            err0 = err0 + err1;

            var x = v.* + err0;
            x = simd.clampV(8, x, @splat(0.0), peak);
            const q: @Vector(8, i32) = simd.cvtRoundI32(8, x);

            const q_f: f32x8 = @floatFromInt(q);
            const new_err = x - q_f;
            v.* = @bitCast(q);
            error_cur[j_err + 0] = new_err[7];
            const next_err = error_top[j_err + 14 + 2];

            err_left.* = new_err;
            err_top_left.* = err_top.*;
            err_top.* = err_top_right.*;
            err_top_right.* = rotateRightShuffle(new_err, next_err);
        }

        fn processWavefront(
            src_rows: [8][]const T,
            dst_rows: [8][]U,
            error_top: []const f32,
            error_cur: []f32,
            error_tmp: *[7][24]f32,
            scale: f32,
            offset: f32,
            peak: f32,
            width: u32,
        ) void {
            const scale_v: f32x8 = @splat(scale);
            const offset_v: f32x8 = @splat(offset);
            const peak_v: f32x8 = @splat(peak);

            processRowScalar(src_rows[0], dst_rows[0], error_top, &error_tmp[0], scale, offset, peak, 14);
            inline for (1..7) |r| {
                processRowScalar(src_rows[r], dst_rows[r], &error_tmp[r - 1], &error_tmp[r], scale, offset, peak, 14 - 2 * r);
            }

            var err_left: f32x8 = .{
                error_tmp[0][13 + 1], error_tmp[1][11 + 1], error_tmp[2][9 + 1], error_tmp[3][7 + 1],
                error_tmp[4][5 + 1],  error_tmp[5][3 + 1],  error_tmp[6][1 + 1], 0.0,
            };
            var err_top_right: f32x8 = .{
                error_top[15 + 1],   error_tmp[0][13 + 1], error_tmp[1][11 + 1], error_tmp[2][9 + 1],
                error_tmp[3][7 + 1], error_tmp[4][5 + 1],  error_tmp[5][3 + 1],  error_tmp[6][1 + 1],
            };
            var err_top: f32x8 = .{
                error_top[14 + 1],   error_tmp[0][12 + 1], error_tmp[1][10 + 1], error_tmp[2][8 + 1],
                error_tmp[3][6 + 1], error_tmp[4][4 + 1],  error_tmp[5][2 + 1],  error_tmp[6][0 + 1],
            };
            var err_top_left: f32x8 = .{
                error_top[13 + 1],   error_tmp[0][11 + 1], error_tmp[1][9 + 1], error_tmp[2][7 + 1],
                error_tmp[3][5 + 1], error_tmp[4][3 + 1],  error_tmp[5][1 + 1], 0.0,
            };

            const vec_count = (width - 14) & ~@as(u32, 7);
            var j: u32 = 0;
            while (j < vec_count) : (j += 8) {
                var v: [8]f32x8 = undefined;
                inline for (0..8) |r| {
                    v[r] = @mulAdd(f32x8, load8(src_rows[r][j + 14 - 2 * r ..]), scale_v, offset_v);
                }

                v = simd.transposeF32Reg(8, v);

                inline for (0..8) |r| {
                    wavefrontIter(&v[r], j + @as(u32, r), error_top, error_cur, peak_v, &err_left, &err_top_right, &err_top, &err_top_left);
                }

                v = simd.transposeF32Reg(8, v);

                inline for (0..8) |r| {
                    const qi: @Vector(8, i32) = @bitCast(v[r]);
                    const out: @Vector(8, U) = @intCast(qi);
                    dst_rows[r][j + 14 - 2 * r ..][0..8].* = out;
                }
            }

            inline for (0..7) |r| {
                error_tmp[r][13 - 2 * r + 1] = err_top_right[r + 1];
                error_tmp[r][12 - 2 * r + 1] = err_top[r + 1];
                if (r < 6) {
                    error_tmp[r][11 - 2 * r + 1] = err_top_left[r + 1];
                } else {
                    error_tmp[6][0] = err_top_left[7];
                }
            }

            processRowScalar(src_rows[0][vec_count + 14 ..], dst_rows[0][vec_count + 14 ..], error_top[vec_count + 14 ..], error_tmp[0][14..], scale, offset, peak, width - vec_count - 14);
            inline for (1..7) |r| {
                const off = 14 - 2 * r;
                processRowScalar(src_rows[r][vec_count + off ..], dst_rows[r][vec_count + off ..], error_tmp[r - 1][off..], error_tmp[r][off..], scale, offset, peak, width - vec_count - @as(u32, off));
            }
            processRowScalar(src_rows[7][vec_count..], dst_rows[7][vec_count..], error_tmp[6][0..], error_cur[vec_count..], scale, offset, peak, width - vec_count);
        }

        pub fn processPlane(noalias srcp: []const T, noalias dstp: []U, w: u32, h: u32, src_stride: u32, dst_stride: u32, scale: f32, offset: f32, peak: f32, error_top: []f32, error_cur: []f32) void {
            @memset(error_top, 0.0);
            @memset(error_cur, 0.0);

            var top = error_top;
            var cur = error_cur;

            var src_rest = srcp;
            var dst_rest = dstp;
            var y: u32 = 0;
            while (y + 8 <= h and w >= 14) : (y += 8) {
                var src_rows: [8][]const T = undefined;
                var dst_rows: [8][]U = undefined;
                var error_tmp: [7][24]f32 = .{.{0} ** 24} ** 7;

                inline for (0..8) |i| {
                    src_rows[i] = src_rest[i * src_stride ..];
                    dst_rows[i] = dst_rest[i * dst_stride ..];
                }

                processWavefront(src_rows, dst_rows, top, cur, &error_tmp, scale, offset, peak, w);
                const t = top;
                top = cur;
                cur = t;
                top[0] = 0.0;
                top[w + 1] = 0.0;
                top[w + 2] = 0.0;
                cur[0] = 0.0;

                src_rest = src_rest[8 * src_stride ..];
                dst_rest = dst_rest[8 * dst_stride ..];
            }

            while (y < h) : (y += 1) {
                cur[0] = 0.0;
                processRowScalar(src_rest, dst_rest, top, cur, scale, offset, peak, w);
                const t = top;
                top = cur;
                cur = t;
                top[w + 1] = 0.0;
                src_rest = src_rest[src_stride..];
                dst_rest = dst_rest[dst_stride..];
            }
        }
    };
}
