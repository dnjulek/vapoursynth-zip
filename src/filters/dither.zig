const std = @import("std");
const simd = @import("simd.zig");
const pattern = @import("dither_pattern.zig");

pub const n_vec = std.simd.suggestVectorLength(f32) orelse 4;
pub const n_u16 = std.simd.suggestVectorLength(u16) orelse 8;
const F32V = @Vector(n_vec, f32);
const U32V = @Vector(n_vec, u32);
const I32V = @Vector(n_vec, i32);

pub const ScaleOffset = struct { scale: f32, offset: f32 };

pub const OrderedMode = enum {
    round,
    zimg_ordered,
    zimg_random,
    fmtc_bayer,
    fmtc_void,
    fmtc_quasi,
};

const zimg_random_offs = [4][2]u32{ .{ 0, 0 }, .{ 32, 12 }, .{ 16, 55 }, .{ 48, 26 } };
const qrs_alpha1: f64 = 1.0 / 1.3247179572447460259609088544781;
const qrs_alpha2: f64 = qrs_alpha1 * qrs_alpha1;
const qrs_sc_l2: u32 = 16;
const qrs_sc_mul: f32 = @floatFromInt(@as(u32, 1) << qrs_sc_l2);
const qrs_shf: u5 = qrs_sc_l2 - 9;
const qrs_inc: u32 = @intFromFloat(qrs_alpha1 * qrs_sc_mul + 0.5);
const qrs_plane_seed = 263;

pub fn quasiRowSeed(y_eff: u32) u32 {
    const v = qrs_alpha2 * @as(f64, @floatFromInt(y_eff)) * @as(f64, qrs_sc_mul);
    const magic: f64 = 6755399441055744.0;
    return @intFromFloat((v + magic) - magic);
}

pub inline fn loadSrc(comptime T: type, ptr: *const [n_vec]T) F32V {
    const vec: @Vector(n_vec, T) = ptr.*;
    return switch (@typeInfo(T)) {
        .float => @floatCast(vec),
        .int => @floatFromInt(vec),
        else => @compileError("unsupported source type"),
    };
}

pub inline fn fmtcIntFromFloat(comptime U: type, val: F32V) @Vector(n_vec, U) {
    const biased = val + @as(F32V, @splat(-32768.0));
    const rounded = simd.cvtRoundI32(n_vec, biased);
    const unbiased = rounded ^ @as(I32V, @splat(0x8000));
    return @truncate(@as(U32V, @bitCast(unbiased)));
}

fn DitherSource(comptime mode: OrderedMode) type {
    return struct {
        row: switch (mode) {
            .round => void,
            .fmtc_quasi => u32,
            else => []const f32,
        },
        hoff: u32,

        const Self = @This();

        inline fn init(y: u32, plane: u32) Self {
            switch (mode) {
                .round => return .{ .row = {}, .hoff = 0 },
                .zimg_ordered => return .{
                    .row = &pattern.zimg_bayer_rot[plane % 4][y & 15],
                    .hoff = 0,
                },
                .zimg_random => return .{
                    .row = &pattern.zimg_blue_noise_f32[(y + zimg_random_offs[plane & 3][1]) & 63],
                    .hoff = zimg_random_offs[plane & 3][0],
                },
                .fmtc_bayer => return .{ .row = &pattern.fmtc_bayer_f32[y & (pattern.fmtc_bayer_len - 1)], .hoff = 0 },
                .fmtc_void => return .{ .row = &pattern.fmtc_void_f32[y & 31], .hoff = 0 },
                .fmtc_quasi => return .{ .row = quasiRowSeed(y + qrs_plane_seed * plane), .hoff = 0 },
            }
        }

        inline fn get(self: Self, x: u32) F32V {
            switch (mode) {
                .round => return @splat(0.0),
                .zimg_ordered => return self.row[x & 15 ..][0..n_vec].*,
                .zimg_random => return self.row[(x + self.hoff) & 63 ..][0..n_vec].*,
                .fmtc_bayer => return self.row[x & (pattern.fmtc_bayer_len - 1) ..][0..n_vec].*,
                .fmtc_void => return self.row[x & 31 ..][0..n_vec].*,
                .fmtc_quasi => {
                    const lanes = std.simd.iota(u32, n_vec) + @as(U32V, @splat(x));
                    const cnt = @as(U32V, @splat(self.row)) +% lanes *% @as(U32V, @splat(qrs_inc));
                    const p = (cnt >> @splat(qrs_shf)) & @as(U32V, @splat(0x1FF));
                    const pi: I32V = @intCast(p);
                    const folded = @select(
                        i32,
                        p > @as(U32V, @splat(255)),
                        @as(I32V, @splat(512 - 128)) - pi,
                        pi - @as(I32V, @splat(128)),
                    );
                    return @as(F32V, @floatFromInt(folded)) * @as(F32V, @splat(1.0 / 256.0));
                },
            }
        }
    };
}

pub fn Ordered(comptime T: type, comptime U: type, comptime mode: OrderedMode) type {
    return struct {
        pub fn processPlane(
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            sc: ScaleOffset,
            peak: f32,
            plane: u32,
            bits_in: u6,
            bits_out: u6,
            int_kernel: bool,
        ) void {
            const fmtc_ik = comptime (T == u16 and
                mode != .zimg_ordered and mode != .zimg_random);
            if (fmtc_ik and int_kernel) {
                intKernelPlane(srcp, dstp, w, h, src_stride, dst_stride, bits_in, bits_out, plane);
            } else {
                std.debug.assert(!int_kernel or (comptime (mode == .zimg_ordered or mode == .zimg_random)));
                processPlaneImpl(srcp, dstp, w, h, src_stride, dst_stride, sc, peak, plane);
            }
        }

        fn intKernelPlane(
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            bits_in: u6,
            bits_out: u6,
            plane: u32,
        ) void {
            switch (@as(u4, @intCast(bits_in - bits_out))) {
                inline 1...8 => |dif| intKernelImpl(dif, srcp, dstp, w, h, src_stride, dst_stride, bits_out, plane),
                else => unreachable,
            }
        }

        fn intKernelImpl(
            comptime dif: u4,
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            bits_out: u6,
            plane: u32,
        ) void {
            if (comptime !(T == u16 and @typeInfo(U) == .int)) unreachable;
            const U16V = @Vector(n_u16, u16);
            const sh: u4 = 8 - dif;
            const rcst: u16 = comptime @as(u16, 1) << (dif - 1);
            const peak: u16 = @intCast((@as(u32, 1) << @intCast(bits_out)) - 1);

            var d_table: [32][32 + pattern.pad]u16 align(64) = undefined;
            const tab_len: u32 = comptime switch (mode) {
                .fmtc_bayer => pattern.fmtc_bayer_len,
                .fmtc_void => 32,
                else => 0,
            };
            if (comptime tab_len != 0) {
                for (0..tab_len) |ty| {
                    for (0..tab_len + pattern.pad) |tx| {
                        const pat: i32 = if (comptime mode == .fmtc_bayer)
                            pattern.fmtc_bayer_i16[ty][tx]
                        else
                            pattern.fmtc_void_i16[ty][tx];
                        d_table[ty][tx] = @intCast((pat >> sh) + rcst);
                    }
                }
            }

            var src_row = srcp;
            var dst_row = dstp;
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                const d_row: [*]const u16 = if (comptime tab_len != 0)
                    @ptrCast(&d_table[y & (tab_len - 1)])
                else
                    undefined;
                const seed: u32 = if (comptime mode == .fmtc_quasi)
                    quasiRowSeed(y + qrs_plane_seed * plane)
                else
                    0;

                var x: u32 = 0;
                while (x < w) : (x += n_u16) {
                    const s: U16V = src_row[x..][0..n_u16].*;
                    const d: U16V = switch (comptime mode) {
                        .round => @splat(rcst),
                        .fmtc_bayer => d_row[x & (pattern.fmtc_bayer_len - 1) ..][0..n_u16].*,
                        .fmtc_void => d_row[x & 31 ..][0..n_u16].*,
                        .fmtc_quasi => quasiDither16(seed, x, sh, rcst),
                        else => unreachable,
                    };
                    const q = @min((s +| d) >> @splat(dif), @as(U16V, @splat(peak)));
                    if (comptime U == u8) {
                        const out: @Vector(n_u16, u8) = @intCast(q);
                        dst_row[x..][0..n_u16].* = out;
                    } else {
                        dst_row[x..][0..n_u16].* = q;
                    }
                }

                src_row = src_row[src_stride..];
                dst_row = dst_row[dst_stride..];
            }
        }

        inline fn quasiDither16(seed: u32, x: u32, sh: u4, rcst: u16) @Vector(n_u16, u16) {
            const U32W = @Vector(n_u16, u32);
            const I32W = @Vector(n_u16, i32);
            const lanes = std.simd.iota(u32, n_u16) + @as(U32W, @splat(x));
            const cnt = @as(U32W, @splat(seed)) +% lanes *% @as(U32W, @splat(qrs_inc));
            const p = (cnt >> @splat(qrs_shf)) & @as(U32W, @splat(0x1FF));
            const pi: I32W = @intCast(p);
            const folded = @select(
                i32,
                p > @as(U32W, @splat(255)),
                @as(I32W, @splat(512 - 128)) - pi,
                pi - @as(I32W, @splat(128)),
            );
            const d = (folded >> @splat(@as(u5, sh))) + @as(I32W, @splat(rcst));
            return @intCast(d);
        }

        inline fn computeVal(ptr: *const [n_vec]T, dsrc: DitherSource(mode), x: u32, scale_v: F32V, offset_v: F32V) F32V {
            const src_vec = loadSrc(T, ptr);
            var val = if (comptime mode == .zimg_ordered or mode == .zimg_random or mode == .round)
                @mulAdd(F32V, src_vec, scale_v, offset_v)
            else
                src_vec * scale_v + offset_v;
            if (comptime mode != .round) val += dsrc.get(x);
            return val;
        }

        fn processPlaneImpl(
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            sc: ScaleOffset,
            peak: f32,
            plane: u32,
        ) void {
            const zimg_style = mode == .zimg_ordered or mode == .zimg_random or mode == .round;
            const scale_v: F32V = @splat(sc.scale);
            const offset_v: F32V = @splat(sc.offset);
            const peak_v: F32V = @splat(peak);
            const zero_v: F32V = @splat(0.0);
            const peak16: u16 = @intFromFloat(peak);
            const peak16_v: @Vector(16, u16) = @splat(peak16);

            var src_row = srcp;
            var dst_row = dstp;
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                const dsrc = DitherSource(mode).init(y, plane);

                var x: u32 = 0;
                if (comptime n_vec == 8 and mode == .zimg_random) {
                    var drow: [8]F32V = undefined;
                    inline for (0..8) |i| drow[i] = dsrc.get(i * 8);
                    while (x + 64 <= src_stride and x + 64 <= dst_stride and x < w) : (x += 64) {
                        inline for (0..4) |i| {
                            const xb = x + i * 16;
                            const lo = @mulAdd(F32V, loadSrc(T, src_row[xb..][0..n_vec]), scale_v, offset_v) + drow[2 * i];
                            const hi = @mulAdd(F32V, loadSrc(T, src_row[xb + n_vec ..][0..n_vec]), scale_v, offset_v) + drow[2 * i + 1];
                            const lo_i = simd.cvtRoundI32(n_vec, lo);
                            const hi_i = simd.cvtRoundI32(n_vec, hi);
                            const w16 = @min(simd.packUsI32ToU16(lo_i, hi_i), peak16_v);
                            if (comptime U == u8) {
                                dst_row[xb..][0..16].* = simd.packU16ToU8(w16);
                            } else {
                                dst_row[xb..][0..16].* = w16;
                            }
                        }
                    }
                }
                if (comptime n_vec == 8) {
                    while (x + 16 <= src_stride and x + 16 <= dst_stride and x < w) : (x += 16) {
                        const lo = computeVal(src_row[x..][0..n_vec], dsrc, x, scale_v, offset_v);
                        const hi = computeVal(src_row[x + n_vec ..][0..n_vec], dsrc, x + n_vec, scale_v, offset_v);
                        var lo_i: I32V = undefined;
                        var hi_i: I32V = undefined;
                        if (comptime zimg_style) {
                            lo_i = simd.cvtRoundI32(n_vec, lo);
                            hi_i = simd.cvtRoundI32(n_vec, hi);
                        } else {
                            const bias: F32V = @splat(-32768.0);
                            const unbias: I32V = @splat(32768);
                            lo_i = simd.cvtRoundI32(n_vec, lo + bias) + unbias;
                            hi_i = simd.cvtRoundI32(n_vec, hi + bias) + unbias;
                        }
                        const w16 = @min(simd.packUsI32ToU16(lo_i, hi_i), peak16_v);
                        if (comptime U == u8) {
                            dst_row[x..][0..16].* = simd.packU16ToU8(w16);
                        } else {
                            dst_row[x..][0..16].* = w16;
                        }
                    }
                }
                while (x < w) : (x += n_vec) {
                    const val = simd.clampV(n_vec, computeVal(src_row[x..][0..n_vec], dsrc, x, scale_v, offset_v), zero_v, peak_v);
                    const out: @Vector(n_vec, U) = if (comptime zimg_style)
                        @intCast(simd.cvtRoundI32(n_vec, val))
                    else
                        fmtcIntFromFloat(U, val);
                    dst_row[x..][0..n_vec].* = out;
                }

                src_row = src_row[src_stride..];
                dst_row = dst_row[dst_stride..];
            }
        }
    };
}

pub fn MulAddUp(comptime T: type, comptime U: type) type {
    return struct {
        pub fn processPlane(
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            gain: u16,
            sub: u16,
        ) void {
            if (comptime U == u16) {
                const U16V = @Vector(n_u16, u16);
                const gv: U16V = @splat(gain);
                const sv: U16V = @splat(sub);
                var src_row = srcp;
                var dst_row = dstp;
                var y: u32 = 0;
                while (y < h) : (y += 1) {
                    var x: u32 = 0;
                    while (x < w) : (x += n_u16) {
                        const raw: @Vector(n_u16, T) = src_row[x..][0..n_u16].*;
                        dst_row[x..][0..n_u16].* = (@as(U16V, raw) *% gv) -| sv;
                    }
                    src_row = src_row[src_stride..];
                    dst_row = dst_row[dst_stride..];
                }
            } else unreachable;
        }
    };
}

pub fn Convert(comptime T: type, comptime U: type) type {
    return struct {
        pub fn processPlane(
            noalias srcp: []const T,
            noalias dstp: []U,
            w: u32,
            h: u32,
            src_stride: u32,
            dst_stride: u32,
            sc: ScaleOffset,
        ) void {
            const scale_v: F32V = @splat(sc.scale);
            const offset_v: F32V = @splat(sc.offset);
            const neutral = sc.scale == 1.0 and sc.offset == 0.0;

            var src_row = srcp;
            var dst_row = dstp;
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += n_vec) {
                    const src_vec = loadSrc(T, src_row[x..][0..n_vec]);
                    const val = if (neutral) src_vec else src_vec * scale_v + offset_v;
                    const out: @Vector(n_vec, U) = @floatCast(val);
                    dst_row[x..][0..n_vec].* = out;
                }
                src_row = src_row[src_stride..];
                dst_row = dst_row[dst_stride..];
            }
        }
    };
}
