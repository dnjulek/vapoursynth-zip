//! BoxBlur with runtime radius size

const std = @import("std");
const math = std.math;

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(u32) orelse 8;

inline fn blurInt(comptime T: type, srcp: []const T, src_step: u32, dstp: []T, dst_step: u32, len: u32, radius: u32) void {
    const ksize: u32 = (radius << 1) + 1;
    const inv: u64 = @divTrunc(((1 << 32) + @as(u64, radius)), ksize);
    var sum: u64 = srcp[radius * src_step];
    const inv2 = inv >> 16;

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        sum += @as(u32, srcp[x * src_step]) << 1;
    }

    sum = (sum * inv + (1 << 31)) >> 16;

    x = 0;
    while (x <= radius) : (x += 1) {
        sum += srcp[(radius + x) * src_step] * inv2;
        sum -= srcp[(radius - x) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }

    while (x < len - radius) : (x += 1) {
        sum += srcp[(radius + x) * src_step] * inv2;
        sum -= srcp[(x - radius - 1) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }

    while (x < len) : (x += 1) {
        sum += srcp[(2 * len - radius - x - 1) * src_step] * inv2;
        sum -= srcp[(x - radius - 1) * src_step] * inv2;
        dstp[x * dst_step] = @intCast(sum >> 16);
    }
}

inline fn blurFloat(comptime T: type, srcp: []const T, src_step: u32, dstp: []T, dst_step: u32, len: u32, radius: u32) void {
    // Accumulate in f32 regardless of T (bit-exact for f32, f16 matches the
    // scalar f32-accumulate-then-narrow reference).
    const ksize: f32 = @floatFromInt(radius * 2 + 1);
    const div: f32 = 1.0 / ksize;
    var sum: f32 = srcp[radius * src_step];

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        const srcv: f32 = srcp[x * src_step];
        sum += srcv * 2;
    }

    sum = sum * div;

    x = 0;
    while (x <= radius) : (x += 1) {
        const src1: f32 = srcp[(radius + x) * src_step];
        const src2: f32 = srcp[(radius - x) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = if (T == f32) sum else @floatCast(sum);
    }

    while (x < len - radius) : (x += 1) {
        const src1: f32 = srcp[(radius + x) * src_step];
        const src2: f32 = srcp[(x - radius - 1) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = if (T == f32) sum else @floatCast(sum);
    }

    while (x < len) : (x += 1) {
        const src1: f32 = srcp[(2 * len - radius - x - 1) * src_step];
        const src2: f32 = srcp[(x - radius - 1) * src_step];
        sum += (src1 - src2) * div;
        dstp[x * dst_step] = if (T == f32) sum else @floatCast(sum);
    }
}

inline fn blur_passes(comptime T: type, srcp: []const T, dstp: []T, step: u32, len: u32, radius: u32, passes: i32, _tmp1: []T, _tmp2: []T) void {
    var tmp1 = _tmp1;
    var tmp2 = _tmp2;
    var p: i32 = passes;

    if (@typeInfo(T) == .int) {
        if (p == 1) {
            // single pass: blur straight into the destination, skipping the
            // tmp copy (identical per-element math)
            blurInt(T, srcp, step, dstp, step, len, radius);
            return;
        }

        blurInt(T, srcp, step, tmp1, 1, len, radius);
        while (p > 2) : (p -= 1) {
            blurInt(T, tmp1, 1, tmp2, 1, len, radius);
            const tmp3 = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp3;
        }

        blurInt(T, tmp1, 1, dstp, step, len, radius);
    } else {
        if (p == 1) {
            blurFloat(T, srcp, step, dstp, step, len, radius);
            return;
        }

        blurFloat(T, srcp, step, tmp1, 1, len, radius);
        while (p > 2) : (p -= 1) {
            blurFloat(T, tmp1, 1, tmp2, 1, len, radius);
            const tmp3 = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp3;
        }

        blurFloat(T, tmp1, 1, dstp, step, len, radius);
    }
}

pub fn hblur(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, temp1: []T, temp2: []T) void {
    if ((passes > 0) and (radius > 0)) {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            blur_passes(
                T,
                srcp[y * stride ..],
                dstp[y * stride ..],
                1,
                w,
                radius,
                passes,
                temp1,
                temp2,
            );
        }
    } else {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            const srcp2 = srcp[(y * stride)..];
            const dstp2 = dstp[(y * stride)..];
            @memcpy(dstp2[0..w], srcp2[0..w]);
        }
    }
}

/// Fused horizontal + single-pass vertical blur. Rows are h-blurred on demand
/// into a ring buffer (2*vradius+2 rows, L2-sized) that feeds the vertical
/// running column sums, so the plane is read once and written once with no
/// intermediate plane. Bit-identical to hblur into a temp plane followed by
/// one vblur sweep. Requires h > 2*vradius + 1 so every mirrored row is still
/// in the ring.
pub fn hvBlurFused(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, temp1: []T, temp2: []T) void {
    const ring_rows: u32 = 2 * vradius + 2;
    const ring_alloc = allocator.alloc(T, ring_rows * w) catch unreachable;
    defer allocator.free(ring_alloc);
    const sums = allocator.alloc(u64, w) catch unreachable;
    defer allocator.free(sums);

    const ring = struct {
        buf: []T,
        rows: u32,
        w: u32,

        inline fn row(self: @This(), j: u32) []T {
            return self.buf[(j % self.rows) * self.w ..][0..self.w];
        }
    }{ .buf = ring_alloc, .rows = ring_rows, .w = w };

    const hb = (hpasses > 0) and (hradius > 0);

    // h-blur source row j into the ring
    const produce = struct {
        inline fn go(r: @TypeOf(ring), src_row: []const T, j: u32, _w: u32, _hradius: u32, _hpasses: i32, t1: []T, t2: []T, _hb: bool) void {
            const dst_row = r.row(j);
            if (_hb) {
                blur_passes(T, src_row, dst_row, 1, _w, _hradius, _hpasses, t1, t2);
            } else {
                @memcpy(dst_row, src_row[0.._w]);
            }
        }
    }.go;

    var j: u32 = 0;
    while (j <= vradius) : (j += 1) {
        produce(ring, srcp[j * stride ..], j, w, hradius, hpasses, temp1, temp2, hb);
    }

    // init running sums: ring[vradius] + 2 * (ring[0] + .. + ring[vradius-1])
    if (@typeInfo(T) == .int) {
        const ksize: u32 = (vradius << 1) + 1;
        const inv: u64 = @divTrunc(((1 << 32) + @as(u64, vradius)), ksize);
        const inv2: u32 = @intCast(inv >> 16);

        {
            const r_row = ring.row(vradius);
            var c: u32 = 0;
            while (c < w) : (c += 1) {
                sums[c] = r_row[c];
            }

            var x: u32 = 0;
            while (x < vradius) : (x += 1) {
                const row = ring.row(x);
                c = 0;
                while (c < w) : (c += 1) {
                    sums[c] += @as(u32, row[c]) << 1;
                }
            }

            c = 0;
            while (c < w) : (c += 1) {
                sums[c] = (sums[c] * inv + (1 << 31)) >> 16;
            }
        }

        var x: u32 = 0;
        while (x <= vradius) : (x += 1) {
            if (x >= 1) produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb);
            rowAddSubInt(T, sums, ring.row(vradius + x), ring.row(vradius - x), dstp[x * stride ..], w, inv2);
        }

        while (x < h - vradius) : (x += 1) {
            produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb);
            rowAddSubInt(T, sums, ring.row(vradius + x), ring.row(x - vradius - 1), dstp[x * stride ..], w, inv2);
        }

        while (x < h) : (x += 1) {
            rowAddSubInt(T, sums, ring.row(2 * h - vradius - x - 1), ring.row(x - vradius - 1), dstp[x * stride ..], w, inv2);
        }
    } else {
        const fsums = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums))[0..w];
        const ksize: f32 = @floatFromInt(vradius * 2 + 1);
        const div: f32 = 1.0 / ksize;

        {
            const r_row = ring.row(vradius);
            var c: u32 = 0;
            while (c < w) : (c += 1) {
                fsums[c] = r_row[c];
            }

            var x: u32 = 0;
            while (x < vradius) : (x += 1) {
                const row = ring.row(x);
                c = 0;
                while (c < w) : (c += 1) {
                    const rv: f32 = row[c];
                    fsums[c] += rv * 2;
                }
            }

            c = 0;
            while (c < w) : (c += 1) {
                fsums[c] = fsums[c] * div;
            }
        }

        var x: u32 = 0;
        while (x <= vradius) : (x += 1) {
            if (x >= 1) produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb);
            rowAddSubFloat(T, fsums, ring.row(vradius + x), ring.row(vradius - x), dstp[x * stride ..], w, div);
        }

        while (x < h - vradius) : (x += 1) {
            produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb);
            rowAddSubFloat(T, fsums, ring.row(vradius + x), ring.row(x - vradius - 1), dstp[x * stride ..], w, div);
        }

        while (x < h) : (x += 1) {
            rowAddSubFloat(T, fsums, ring.row(2 * h - vradius - x - 1), ring.row(x - vradius - 1), dstp[x * stride ..], w, div);
        }
    }
}

/// Vertical blur as `passes` full-plane row-streaming sweeps. Each sweep keeps
/// one running box sum per column (the exact op sequence of blurInt/blurFloat
/// per column, executed for all columns in lockstep row by row), so memory is
/// touched in row order instead of walking down columns.
///
/// Sweeps are always out-of-place, ping-ponging between `tmp` and `dstp` such
/// that the final sweep lands in `dstp`. `first_src` is only read.
pub fn vblur(comptime T: type, first_src: []const T, tmp: []T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32) void {
    if ((passes <= 0) or (radius <= 0)) return;

    const sums = allocator.alloc(u64, w) catch unreachable;
    defer allocator.free(sums);

    var src_cur: []const T = first_src;
    var s: i32 = 1;
    while (s <= passes) : (s += 1) {
        const dst_cur: []T = if (@mod(passes - s, 2) == 0) dstp else tmp;
        if (@typeInfo(T) == .int) {
            vSweepInt(T, src_cur, dst_cur, sums, stride, w, h, radius);
        } else {
            vSweepFloat(T, src_cur, dst_cur, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums))[0..w], stride, w, h, radius);
        }
        src_cur = dst_cur;
    }
}

fn vSweepInt(comptime T: type, src: []const T, dst: []T, sums: []u64, stride: u32, w: u32, h: u32, radius: u32) void {
    const len = h;
    const ksize: u32 = (radius << 1) + 1;
    const inv: u64 = @divTrunc(((1 << 32) + @as(u64, radius)), ksize);
    const inv2: u32 = @intCast(inv >> 16);

    // init running sums: sum = src[radius] + 2 * (src[0] + .. + src[radius-1])
    {
        const r_row = src[radius * stride ..];
        var c: u32 = 0;
        while (c < w) : (c += 1) {
            sums[c] = r_row[c];
        }

        var x: u32 = 0;
        while (x < radius) : (x += 1) {
            const row = src[x * stride ..];
            c = 0;
            while (c < w) : (c += 1) {
                sums[c] += @as(u32, row[c]) << 1;
            }
        }

        c = 0;
        while (c < w) : (c += 1) {
            sums[c] = (sums[c] * inv + (1 << 31)) >> 16;
        }
    }

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        rowAddSubInt(T, sums, src[(radius + x) * stride ..], src[(radius - x) * stride ..], dst[x * stride ..], w, inv2);
    }

    while (x < len - radius) : (x += 1) {
        rowAddSubInt(T, sums, src[(radius + x) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, inv2);
    }

    while (x < len) : (x += 1) {
        rowAddSubInt(T, sums, src[(2 * len - radius - x - 1) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, inv2);
    }
}

inline fn rowAddSubInt(comptime T: type, sums: []u64, add_row: []const T, sub_row: []const T, dst_row: []T, w: u32, inv2: u32) void {
    const U64V = @Vector(vec_len, u64);
    const U32V = @Vector(vec_len, u32);
    const inv2v: U32V = @splat(inv2);
    const shift: @Vector(vec_len, u6) = @splat(16);

    var c: u32 = 0;
    const w_vec = w - (w % vec_len);
    while (c < w_vec) : (c += vec_len) {
        const at: @Vector(vec_len, T) = add_row[c..][0..vec_len].*;
        const st: @Vector(vec_len, T) = sub_row[c..][0..vec_len].*;
        const a32: U32V = @intCast(at);
        const s32: U32V = @intCast(st);
        // pixel * inv2 < 2^31 (inv2 < 2^15, pixel < 2^16), u32 multiply is exact
        const pa: U64V = @intCast(a32 * inv2v);
        const pb: U64V = @intCast(s32 * inv2v);
        var s: U64V = sums[c..][0..vec_len].*;
        s += pa;
        s -= pb;
        sums[c..][0..vec_len].* = s;
        const out: @Vector(vec_len, T) = @intCast(s >> shift);
        dst_row[c..][0..vec_len].* = out;
    }

    while (c < w) : (c += 1) {
        sums[c] += @as(u32, add_row[c]) * @as(u64, inv2);
        sums[c] -= @as(u32, sub_row[c]) * @as(u64, inv2);
        dst_row[c] = @intCast(sums[c] >> 16);
    }
}

fn vSweepFloat(comptime T: type, src: []const T, dst: []T, sums: []f32, stride: u32, w: u32, h: u32, radius: u32) void {
    const len = h;
    const ksize: f32 = @floatFromInt(radius * 2 + 1);
    const div: f32 = 1.0 / ksize;

    {
        const r_row = src[radius * stride ..];
        var c: u32 = 0;
        while (c < w) : (c += 1) {
            sums[c] = r_row[c];
        }

        var x: u32 = 0;
        while (x < radius) : (x += 1) {
            const row = src[x * stride ..];
            c = 0;
            while (c < w) : (c += 1) {
                const rv: f32 = row[c];
                sums[c] += rv * 2;
            }
        }

        c = 0;
        while (c < w) : (c += 1) {
            sums[c] = sums[c] * div;
        }
    }

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        rowAddSubFloat(T, sums, src[(radius + x) * stride ..], src[(radius - x) * stride ..], dst[x * stride ..], w, div);
    }

    while (x < len - radius) : (x += 1) {
        rowAddSubFloat(T, sums, src[(radius + x) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, div);
    }

    while (x < len) : (x += 1) {
        rowAddSubFloat(T, sums, src[(2 * len - radius - x - 1) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, div);
    }
}

inline fn rowAddSubFloat(comptime T: type, sums: []f32, add_row: []const T, sub_row: []const T, dst_row: []T, w: u32, div: f32) void {
    const FV = @Vector(vec_len, f32);
    const divv: FV = @splat(div);

    var c: u32 = 0;
    const w_vec = w - (w % vec_len);
    while (c < w_vec) : (c += vec_len) {
        const at: @Vector(vec_len, T) = add_row[c..][0..vec_len].*;
        const bt: @Vector(vec_len, T) = sub_row[c..][0..vec_len].*;
        const a: FV = if (T == f32) at else @floatCast(at);
        const b: FV = if (T == f32) bt else @floatCast(bt);
        var s: FV = sums[c..][0..vec_len].*;
        s += (a - b) * divv;
        sums[c..][0..vec_len].* = s;
        if (T == f32) {
            dst_row[c..][0..vec_len].* = s;
        } else {
            dst_row[c..][0..vec_len].* = @as(@Vector(vec_len, T), @floatCast(s));
        }
    }

    while (c < w) : (c += 1) {
        const a: f32 = add_row[c];
        const b: f32 = sub_row[c];
        sums[c] += (a - b) * div;
        dst_row[c] = if (T == f32) sums[c] else @floatCast(sums[c]);
    }
}
