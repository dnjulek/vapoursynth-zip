//! BoxBlur with runtime radius size

const std = @import("std");
const math = std.math;
const simd = @import("simd.zig");
const ct = @import("boxblur_comptime.zig");

const allocator = std.heap.c_allocator;

const vec_len = std.simd.suggestVectorLength(u32) orelse 8;

// The narrow/wide divide is picked per plane and threaded on as a *type*, so
// the choice costs one branch per plane rather than one per pixel, and the
// kernels below take it as `mg: anytype`. Float planes are handed `{}` -- their
// kernels never look at it. boxBlurCreate has already rejected any radius with
// no exact magic, hence the unreachables.

/// The running sum is the *raw* window sum, seeded with +radius. ksize is odd,
/// so W/ksize can never land on a tie and `round(W/ksize) == floor((W+radius)/
/// ksize)` exactly; the slide only ever adds and subtracts pixels, so that seed
/// rides along untouched and the rounding costs nothing per pixel.
///
/// Keeping the reciprocal out of the slide is the whole point: scaling inside
/// it (init at one precision, slide at another) let the first window's value
/// leak a DC error into every pixel of the line.
inline fn blurInt(comptime T: type, srcp: []const T, src_step: u32, dstp: []T, dst_step: u32, len: u32, radius: u32, mg: anytype) void {
    var sum: u32 = @as(u32, srcp[radius * src_step]) + radius;

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        sum += @as(u32, srcp[x * src_step]) << 1;
    }

    x = 0;
    while (x <= radius) : (x += 1) {
        sum += srcp[(radius + x) * src_step];
        sum -= srcp[(radius - x) * src_step];
        dstp[x * dst_step] = mg.apply(T, sum);
    }

    if (src_step == 1 and dst_step == 1) {
        // contiguous rows take the vectorized prefix-sum center segment
        // (bit-exact mod-2^32; see boxblur_comptime.slideSumInt)
        sum = ct.slideSumInt(T, srcp, dstp, x, len - radius, radius, mg, sum);
        x = len - radius;
    } else {
        while (x < len - radius) : (x += 1) {
            sum += srcp[(radius + x) * src_step];
            sum -= srcp[(x - radius - 1) * src_step];
            dstp[x * dst_step] = mg.apply(T, sum);
        }
    }

    while (x < len) : (x += 1) {
        sum += srcp[(2 * len - radius - x - 1) * src_step];
        sum -= srcp[(x - radius - 1) * src_step];
        dstp[x * dst_step] = mg.apply(T, sum);
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

inline fn blur_passes(comptime T: type, srcp: []const T, dstp: []T, step: u32, len: u32, radius: u32, passes: i32, _tmp1: []T, _tmp2: []T, mg: anytype) void {
    var tmp1 = _tmp1;
    var tmp2 = _tmp2;
    var p: i32 = passes;

    if (@typeInfo(T) == .int) {
        if (p == 1) {
            // single pass: blur straight into the destination, skipping the
            // tmp copy (identical per-element math)
            blurInt(T, srcp, step, dstp, step, len, radius, mg);
            return;
        }

        blurInt(T, srcp, step, tmp1, 1, len, radius, mg);
        while (p > 2) : (p -= 1) {
            blurInt(T, tmp1, 1, tmp2, 1, len, radius, mg);
            const tmp3 = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp3;
        }

        blurInt(T, tmp1, 1, dstp, step, len, radius, mg);
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
    if (!((passes > 0) and (radius > 0))) {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            const srcp2 = srcp[(y * stride)..];
            const dstp2 = dstp[(y * stride)..];
            @memcpy(dstp2[0..w], srcp2[0..w]);
        }
        return;
    }

    if (@typeInfo(T) == .int) {
        if (ct.narrowMagicFor(T, radius)) |mg| return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, temp1, temp2, mg);
        return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, temp1, temp2, ct.magicFor(T, radius) orelse unreachable);
    }
    return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, temp1, temp2, {});
}

fn hblurGo(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, temp1: []T, temp2: []T, mg: anytype) void {
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
            mg,
        );
    }
}

/// Fused horizontal + single-pass vertical blur. Rows are h-blurred on demand
/// into a ring buffer (2*vradius+2 rows, L2-sized) that feeds the vertical
/// running column sums, so the plane is read once and written once with no
/// intermediate plane. Bit-identical to hblur into a temp plane followed by
/// one vblur sweep. Requires h > 2*vradius + 1 so every mirrored row is still
/// in the ring.
pub fn hvBlurFused(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, temp1: []T, temp2: []T) void {
    if (@typeInfo(T) == .int) {
        // hradius may be 0 when there is no h-blur; that magic is never used.
        const hr = if ((hpasses > 0) and (hradius > 0)) hradius else 1;
        if (ct.narrowMagicFor(T, hr)) |hmg| {
            if (ct.narrowMagicFor(T, vradius)) |vmg| {
                return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, temp1, temp2, hmg, vmg);
            }
        }
        // Asymmetric radii can leave one direction without a narrow magic; take
        // the wide kernel for both rather than instantiate a mixed variant.
        return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, temp1, temp2, ct.magicFor(T, hr) orelse unreachable, ct.magicFor(T, vradius) orelse unreachable);
    }
    return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, temp1, temp2, {}, {});
}

fn hvBlurFusedGo(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, temp1: []T, temp2: []T, hmg: anytype, vmg: anytype) void {
    const ring_rows: u32 = 2 * vradius + 2;
    const ring_alloc = allocator.alloc(T, ring_rows * w) catch unreachable;
    defer allocator.free(ring_alloc);
    const sums = allocator.alloc(u32, w) catch unreachable;
    defer allocator.free(sums);

    const ring = struct {
        buf: []T,
        rows: u32,
        w: u32,

        inline fn row(self: @This(), j: u32) []T {
            // opaquePtr pins the `%` here: without it LLVM re-sinks this
            // (loop-invariant) modulo + address math into the callers' inner
            // per-pixel loops after inlining — an unpipelined 32-bit div per
            // vector block (measured in all four BoxBlurRT variants).
            const s = self.buf[(j % self.rows) * self.w ..][0..self.w];
            return simd.opaquePtr(T, s.ptr)[0..self.w];
        }
    }{ .buf = ring_alloc, .rows = ring_rows, .w = w };

    const hb = (hpasses > 0) and (hradius > 0);

    // h-blur source row j into the ring
    const produce = struct {
        inline fn go(r: @TypeOf(ring), src_row: []const T, j: u32, _w: u32, _hradius: u32, _hpasses: i32, t1: []T, t2: []T, _hb: bool, _mg: anytype) void {
            const dst_row = r.row(j);
            if (_hb) {
                blur_passes(T, src_row, dst_row, 1, _w, _hradius, _hpasses, t1, t2, _mg);
            } else {
                @memcpy(dst_row, src_row[0.._w]);
            }
        }
    }.go;

    var j: u32 = 0;
    while (j <= vradius) : (j += 1) {
        produce(ring, srcp[j * stride ..], j, w, hradius, hpasses, temp1, temp2, hb, hmg);
    }

    // init running sums: ring[vradius] + 2 * (ring[0] + .. + ring[vradius-1])
    if (@typeInfo(T) == .int) {
        {
            // raw column sums, +vradius seed (see blurInt)
            const r_row = ring.row(vradius);
            var c: u32 = 0;
            while (c < w) : (c += 1) {
                sums[c] = @as(u32, r_row[c]) + vradius;
            }

            var x: u32 = 0;
            while (x < vradius) : (x += 1) {
                const row = ring.row(x);
                c = 0;
                while (c < w) : (c += 1) {
                    sums[c] += @as(u32, row[c]) << 1;
                }
            }
        }

        var x: u32 = 0;
        while (x <= vradius) : (x += 1) {
            if (x >= 1) produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb, hmg);
            rowAddSubInt(T, sums, ring.row(vradius + x), ring.row(vradius - x), dstp[x * stride ..], w, vmg);
        }

        while (x < h - vradius) : (x += 1) {
            produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb, hmg);
            rowAddSubInt(T, sums, ring.row(vradius + x), ring.row(x - vradius - 1), dstp[x * stride ..], w, vmg);
        }

        while (x < h) : (x += 1) {
            rowAddSubInt(T, sums, ring.row(2 * h - vradius - x - 1), ring.row(x - vradius - 1), dstp[x * stride ..], w, vmg);
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
            if (x >= 1) produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb, hmg);
            rowAddSubFloat(T, fsums, ring.row(vradius + x), ring.row(vradius - x), dstp[x * stride ..], w, div);
        }

        while (x < h - vradius) : (x += 1) {
            produce(ring, srcp[(vradius + x) * stride ..], vradius + x, w, hradius, hpasses, temp1, temp2, hb, hmg);
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

    if (@typeInfo(T) == .int) {
        if (ct.narrowMagicFor(T, radius)) |mg| return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, mg);
        return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, ct.magicFor(T, radius) orelse unreachable);
    }
    return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, {});
}

fn vblurGo(comptime T: type, first_src: []const T, tmp: []T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, mg: anytype) void {
    const sums = allocator.alloc(u32, w) catch unreachable;
    defer allocator.free(sums);

    var src_cur: []const T = first_src;
    var s: i32 = 1;
    while (s <= passes) : (s += 1) {
        const dst_cur: []T = if (@mod(passes - s, 2) == 0) dstp else tmp;
        if (@typeInfo(T) == .int) {
            vSweepInt(T, src_cur, dst_cur, sums, stride, w, h, radius, mg);
        } else {
            vSweepFloat(T, src_cur, dst_cur, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums))[0..w], stride, w, h, radius);
        }
        src_cur = dst_cur;
    }
}

fn vSweepInt(comptime T: type, src: []const T, dst: []T, sums: []u32, stride: u32, w: u32, h: u32, radius: u32, mg: anytype) void {
    const len = h;

    // init raw running sums, +radius seed (see blurInt):
    //   sum = src[radius] + 2 * (src[0] + .. + src[radius-1]) + radius
    {
        const r_row = src[radius * stride ..];
        var c: u32 = 0;
        while (c < w) : (c += 1) {
            sums[c] = @as(u32, r_row[c]) + radius;
        }

        var x: u32 = 0;
        while (x < radius) : (x += 1) {
            const row = src[x * stride ..];
            c = 0;
            while (c < w) : (c += 1) {
                sums[c] += @as(u32, row[c]) << 1;
            }
        }
    }

    var x: u32 = 0;
    while (x <= radius) : (x += 1) {
        rowAddSubInt(T, sums, src[(radius + x) * stride ..], src[(radius - x) * stride ..], dst[x * stride ..], w, mg);
    }

    while (x < len - radius) : (x += 1) {
        rowAddSubInt(T, sums, src[(radius + x) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, mg);
    }

    while (x < len) : (x += 1) {
        rowAddSubInt(T, sums, src[(2 * len - radius - x - 1) * stride ..], src[(x - radius - 1) * stride ..], dst[x * stride ..], w, mg);
    }
}

inline fn rowAddSubInt(comptime T: type, sums: []u32, add_row: []const T, sub_row: []const T, dst_row: []T, w: u32, mg: anytype) void {
    const U32V = @Vector(vec_len, u32);

    var c: u32 = 0;
    const w_vec = w - (w % vec_len);
    while (c < w_vec) : (c += vec_len) {
        const at: @Vector(vec_len, T) = add_row[c..][0..vec_len].*;
        const st: @Vector(vec_len, T) = sub_row[c..][0..vec_len].*;
        // Sums are raw window sums (+radius seed) in u32: the true value stays
        // < 2^32, so wrapping updates keep every stored value exact.
        var s: U32V = sums[c..][0..vec_len].*;
        s = s +% @as(U32V, @intCast(at));
        s = s -% @as(U32V, @intCast(st));
        sums[c..][0..vec_len].* = s;
        dst_row[c..][0..vec_len].* = ct.scaleVec(T, vec_len, s, mg);
    }

    while (c < w) : (c += 1) {
        sums[c] = sums[c] +% @as(u32, add_row[c]) -% @as(u32, sub_row[c]);
        dst_row[c] = mg.apply(T, sums[c]);
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
