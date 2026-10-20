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
// kernels never look at it. Create has already rejected any radius with
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

/// Rows per float h-blur strip = f32 lanes. Each lane carries one row.
const strip_n = std.simd.suggestVectorLength(f32) orelse 1;
const StripV = @Vector(strip_n, f32);
const use_strip = strip_n > 1;

/// Every scratch buffer of the runtime paths, carved from one block that
/// getFrame allocates once per frame and reuses for all planes (the paths
/// used to malloc/free two to four buffers per plane). Sized for the widest
/// plane; each path uses a prefix. Every element is written before it is
/// read, so the reuse is deterministic.
pub fn Scratch(comptime T: type) type {
    return struct {
        const Self = @This();
        block: []align(64) u8,
        temp1: []T, // blur_passes row ping-pong (tmp_size)
        temp2: []T,
        sums: []u32, // vertical running column sums (u32 or f32 bits)
        cols_a: []StripV, // column-major strip, lane l = row l
        cols_b: []StripV,
        ring: []T, // hvBlurFused / vChainFused h-blurred rows
        chain_sums: []u32, // vChainFused: running sums per vertical pass
        chain_ring: []T, // vChainFused: output rows of every pass but the last

        pub fn init(w: u32, tmp_size: u32, vradius: u32, vpasses: u32) !Self {
            const chain_rows: usize = 2 * @as(usize, vradius) + 2;
            const ring_rows: usize = 2 * @as(usize, vradius) + 2 + (if (use_strip) strip_n else 0);
            const cols_len: usize = if (use_strip) @max(w, 1) else 0;
            const sizes = [_]usize{
                tmp_size * @sizeOf(T),
                tmp_size * @sizeOf(T),
                @as(usize, w) * @sizeOf(u32),
                cols_len * @sizeOf(StripV),
                cols_len * @sizeOf(StripV),
                ring_rows * w * @sizeOf(T),
                @as(usize, vpasses) * w * @sizeOf(u32),
                @as(usize, vpasses -| 1) * chain_rows * w * @sizeOf(T),
            };
            var total: usize = 0;
            for (sizes) |sz| total += std.mem.alignForward(usize, sz, 64);
            const block = try allocator.alignedAlloc(u8, .@"64", total);
            var off: usize = 0;
            var parts: [sizes.len][]align(64) u8 = undefined;
            for (sizes, &parts) |sz, *part| {
                part.* = @alignCast(block[off..][0..sz]);
                off += std.mem.alignForward(usize, sz, 64);
            }
            return .{
                .block = block,
                .temp1 = std.mem.bytesAsSlice(T, parts[0]),
                .temp2 = std.mem.bytesAsSlice(T, parts[1]),
                .sums = std.mem.bytesAsSlice(u32, parts[2]),
                .cols_a = std.mem.bytesAsSlice(StripV, parts[3]),
                .cols_b = std.mem.bytesAsSlice(StripV, parts[4]),
                .ring = std.mem.bytesAsSlice(T, parts[5]),
                .chain_sums = std.mem.bytesAsSlice(u32, parts[6]),
                .chain_ring = std.mem.bytesAsSlice(T, parts[7]),
            };
        }

        pub fn deinit(self: Self) void {
            allocator.free(self.block);
        }
    };
}

/// Float h-blur of `strip_n` rows at once, bit-identical to running
/// blur_passes on each row. blurFloat's running sum is one loop-carried
/// vaddss chain per row (~3 cycles/px on Zen 3, the whole cost of the float
/// runtime h-pass). Here the strip is transposed into column-major vectors
/// (lane l = row l), the *same* recurrence runs on vectors — per lane the
/// identical f32 op sequence, just strip_n rows advancing together — and the
/// result is transposed back. Transposes are in-register (simd.transposeF32Reg,
/// pinned asm at 8/16 lanes). Multi-pass keeps the per-pass rounding to T of
/// the row path (its tmp rows are []T).
fn hblurStripFloat(comptime T: type, src_rows: [strip_n][*]const T, dst_rows: [strip_n][*]T, len: u32, radius: u32, passes: i32, cols_a: []StripV, cols_b: []StripV) void {
    const N = strip_n;
    const V = StripV;
    // gather: rows -> columns
    var c: u32 = 0;
    while (c + N <= len) : (c += N) {
        var rows: [N]V = undefined;
        inline for (0..N) |l| {
            const raw: @Vector(N, T) = src_rows[l][c..][0..N].*;
            rows[l] = if (T == f32) raw else @floatCast(raw);
        }
        const cols = simd.transposeF32Reg(N, rows);
        inline for (0..N) |i| cols_a[c + i] = cols[i];
    }
    while (c < len) : (c += 1) {
        var v: V = undefined;
        inline for (0..N) |l| v[l] = src_rows[l][c];
        cols_a[c] = v;
    }

    var in_cols = cols_a;
    var out_cols = cols_b;
    var p: i32 = passes;
    while (p > 0) : (p -= 1) {
        blurCols(in_cols[0..len], out_cols[0..len], len, radius);
        if (T != f32 and p > 1) {
            // row path stores intermediate passes as T
            for (out_cols[0..len]) |*v| v.* = @floatCast(@as(@Vector(N, T), @floatCast(v.*)));
        }
        const t = in_cols;
        in_cols = out_cols;
        out_cols = t;
    }
    const res = in_cols;

    // scatter: columns -> rows
    c = 0;
    while (c + N <= len) : (c += N) {
        var cols: [N]V = undefined;
        inline for (0..N) |i| cols[i] = res[c + i];
        const rows = simd.transposeF32Reg(N, cols);
        inline for (0..N) |l| {
            dst_rows[l][c..][0..N].* = if (T == f32) rows[l] else @as(@Vector(N, T), @floatCast(rows[l]));
        }
    }
    while (c < len) : (c += 1) {
        const v = res[c];
        inline for (0..N) |l| dst_rows[l][c] = if (T == f32) v[l] else @floatCast(v[l]);
    }
}

/// blurFloat's exact op sequence on column vectors (lane = row).
fn blurCols(src: []const StripV, dst: []StripV, len: u32, radius: u32) void {
    const ksize: f32 = @floatFromInt(radius * 2 + 1);
    const div: StripV = @splat(1.0 / ksize);
    const two: StripV = @splat(2);
    var sum: StripV = src[radius];

    var x: u32 = 0;
    while (x < radius) : (x += 1) {
        sum += src[x] * two;
    }
    sum = sum * div;

    x = 0;
    while (x <= radius) : (x += 1) {
        sum += (src[radius + x] - src[radius - x]) * div;
        dst[x] = sum;
    }
    while (x < len - radius) : (x += 1) {
        sum += (src[radius + x] - src[x - radius - 1]) * div;
        dst[x] = sum;
    }
    while (x < len) : (x += 1) {
        sum += (src[2 * len - radius - x - 1] - src[x - radius - 1]) * div;
        dst[x] = sum;
    }
}

pub fn hblur(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, sc: *const Scratch(T)) void {
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
        if (ct.narrowMagicFor(T, radius)) |mg| return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, sc, mg);
        return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, sc, ct.magicFor(T, radius) orelse unreachable);
    }
    return hblurGo(T, srcp, dstp, stride, w, h, radius, passes, sc, {});
}

fn hblurGo(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, sc: *const Scratch(T), mg: anytype) void {
    const temp1 = sc.temp1;
    const temp2 = sc.temp2;
    var y: u32 = 0;
    if (@typeInfo(T) == .float and use_strip) {
        const ca = sc.cols_a[0..w];
        const cb = sc.cols_b[0..w];
        while (y + strip_n <= h) : (y += strip_n) {
            var sr: [strip_n][*]const T = undefined;
            var dr: [strip_n][*]T = undefined;
            inline for (0..strip_n) |l| {
                sr[l] = srcp[(y + l) * stride ..].ptr;
                dr[l] = dstp[(y + l) * stride ..].ptr;
            }
            hblurStripFloat(T, sr, dr, w, radius, passes, ca, cb);
        }
    }
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

/// h-blurred source rows on demand, in a ring of 2*vradius + 2 rows (plus
/// strip_n look-ahead when float strips produce strip_n rows at a time).
/// Rows are requested in increasing order and each is produced once. Feeds
/// the fused vertical passes (hvBlurFused, vChainFused).
fn HRows(comptime T: type, comptime MG: type) type {
    return struct {
        const Self = @This();
        buf: []T,
        rows: u32,
        w: u32,
        h: u32,
        src: []const T,
        stride: u32,
        hradius: u32,
        hpasses: i32,
        hb: bool,
        strips: bool,
        t1: []T,
        t2: []T,
        ca: []StripV,
        cb: []StripV,
        mg: MG,
        next: u32 = 0,

        fn init(sc: *const Scratch(T), src: []const T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, mg: MG) Self {
            const hb = (hpasses > 0) and (hradius > 0);
            const strips = @typeInfo(T) == .float and use_strip and hb;
            const rows: u32 = 2 * vradius + 2 + (if (strips) @as(u32, strip_n) else 0);
            return .{
                .buf = sc.ring[0 .. rows * w],
                .rows = rows,
                .w = w,
                .h = h,
                .src = src,
                .stride = stride,
                .hradius = hradius,
                .hpasses = hpasses,
                .hb = hb,
                .strips = strips,
                .t1 = sc.temp1,
                .t2 = sc.temp2,
                .ca = if (strips) sc.cols_a[0..w] else &.{},
                .cb = if (strips) sc.cols_b[0..w] else &.{},
                .mg = mg,
            };
        }

        inline fn row(self: *const Self, j: u32) []T {
            // opaquePtr pins the `%` here: without it LLVM re-sinks this
            // (loop-invariant) modulo + address math into the callers' inner
            // per-pixel loops after inlining — an unpipelined 32-bit div per
            // vector block (measured in all four BoxBlurRT variants).
            const s = self.buf[(j % self.rows) * self.w ..][0..self.w];
            return simd.opaquePtr(T, s.ptr)[0..self.w];
        }

        /// Produce rows up to j; with float strips the first request of a
        /// strip produces the whole strip ahead.
        inline fn produce(self: *Self, j: u32) void {
            while (self.next <= j) {
                const k = self.next;
                if (comptime @typeInfo(T) == .float and use_strip) {
                    if (self.strips and k + strip_n <= self.h) {
                        var sr: [strip_n][*]const T = undefined;
                        var dr: [strip_n][*]T = undefined;
                        inline for (0..strip_n) |l| {
                            sr[l] = self.src[(k + l) * self.stride ..].ptr;
                            dr[l] = self.row(k + @as(u32, l)).ptr;
                        }
                        hblurStripFloat(T, sr, dr, self.w, self.hradius, self.hpasses, self.ca, self.cb);
                        self.next = k + strip_n;
                        continue;
                    }
                }
                const src_row = self.src[k * self.stride ..];
                const dst_row = self.row(k);
                if (self.hb) {
                    blur_passes(T, src_row, dst_row, 1, self.w, self.hradius, self.hpasses, self.t1, self.t2, self.mg);
                } else {
                    @memcpy(dst_row, src_row[0..self.w]);
                }
                self.next = k + 1;
            }
        }
    };
}

/// Fused horizontal + single-pass vertical blur. Rows are h-blurred on demand
/// into a ring buffer (2*vradius+2 rows, L2-sized) that feeds the vertical
/// running column sums, so the plane is read once and written once with no
/// intermediate plane. Bit-identical to hblur into a temp plane followed by
/// one vblur sweep. Requires h > 2*vradius + 1 so every mirrored row is still
/// in the ring.
pub fn hvBlurFused(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, sc: *const Scratch(T)) void {
    if (@typeInfo(T) == .int) {
        // hradius may be 0 when there is no h-blur; that magic is never used.
        const hr = if ((hpasses > 0) and (hradius > 0)) hradius else 1;
        if (ct.narrowMagicFor(T, hr)) |hmg| {
            if (ct.narrowMagicFor(T, vradius)) |vmg| {
                return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, sc, hmg, vmg);
            }
        }
        // Asymmetric radii can leave one direction without a narrow magic; take
        // the wide kernel for both rather than instantiate a mixed variant.
        return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, sc, ct.magicFor(T, hr) orelse unreachable, ct.magicFor(T, vradius) orelse unreachable);
    }
    return hvBlurFusedGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, sc, {}, {});
}

fn hvBlurFusedGo(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, sc: *const Scratch(T), hmg: anytype, vmg: anytype) void {
    var hr = HRows(T, @TypeOf(hmg)).init(sc, srcp, stride, w, h, hradius, hpasses, vradius, hmg);
    const sums = sc.sums[0..w];

    var j: u32 = 0;
    while (j <= vradius) : (j += 1) {
        hr.produce(j);
    }

    // init running sums: ring[vradius] + 2 * (ring[0] + .. + ring[vradius-1])
    if (@typeInfo(T) == .int) {
        {
            // raw column sums, +vradius seed (see blurInt)
            const r_row = hr.row(vradius);
            var c: u32 = 0;
            while (c < w) : (c += 1) {
                sums[c] = @as(u32, r_row[c]) + vradius;
            }

            var x: u32 = 0;
            while (x < vradius) : (x += 1) {
                const row = hr.row(x);
                c = 0;
                while (c < w) : (c += 1) {
                    sums[c] += @as(u32, row[c]) << 1;
                }
            }
        }

        var x: u32 = 0;
        while (x <= vradius) : (x += 1) {
            if (x >= 1) hr.produce(vradius + x);
            rowAddSubInt(T, sums, hr.row(vradius + x), hr.row(vradius - x), dstp[x * stride ..], w, vmg);
        }

        while (x < h - vradius) : (x += 1) {
            hr.produce(vradius + x);
            rowAddSubInt(T, sums, hr.row(vradius + x), hr.row(x - vradius - 1), dstp[x * stride ..], w, vmg);
        }

        while (x < h) : (x += 1) {
            rowAddSubInt(T, sums, hr.row(2 * h - vradius - x - 1), hr.row(x - vradius - 1), dstp[x * stride ..], w, vmg);
        }
    } else {
        const fsums = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums))[0..w];
        const ksize: f32 = @floatFromInt(vradius * 2 + 1);
        const div: f32 = 1.0 / ksize;

        {
            const r_row = hr.row(vradius);
            var c: u32 = 0;
            while (c < w) : (c += 1) {
                fsums[c] = r_row[c];
            }

            var x: u32 = 0;
            while (x < vradius) : (x += 1) {
                const row = hr.row(x);
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
            if (x >= 1) hr.produce(vradius + x);
            rowAddSubFloat(T, fsums, hr.row(vradius + x), hr.row(vradius - x), dstp[x * stride ..], w, div);
        }

        while (x < h - vradius) : (x += 1) {
            hr.produce(vradius + x);
            rowAddSubFloat(T, fsums, hr.row(vradius + x), hr.row(x - vradius - 1), dstp[x * stride ..], w, div);
        }

        while (x < h) : (x += 1) {
            rowAddSubFloat(T, fsums, hr.row(2 * h - vradius - x - 1), hr.row(x - vradius - 1), dstp[x * stride ..], w, div);
        }
    }
}

/// Multi-pass vertical blur (after an optional h-blur) as one row stream: pass
/// s consumes pass s-1's rows from a ring of 2*vradius + 2 rows (its window at
/// output row x is input rows [x-vradius-1, x+vradius], edge mirrors
/// included), the first pass reads h-blurred rows from HRows (or the source
/// rows directly), and the last writes dst. The plane form ping-ponged full
/// planes through a scratch frame (passes+1 plane round trips) and went
/// DRAM-bound with every frame worker active. Each pass runs vSweep's
/// sequence (same sum seed, same add/sub rows in the same row order, output
/// rounded to T between passes), so the result is bit-identical. Requires
/// h >= 2*vradius + 2, like hvBlurFused.
pub fn vChainFused(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, vpasses: u32, sc: *const Scratch(T)) error{OutOfMemory}!void {
    if (@typeInfo(T) == .int) {
        const hr = if ((hpasses > 0) and (hradius > 0)) hradius else 1;
        if (ct.narrowMagicFor(T, hr)) |hmg| {
            if (ct.narrowMagicFor(T, vradius)) |vmg| {
                return vChainGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, vpasses, sc, hmg, vmg);
            }
        }
        return vChainGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, vpasses, sc, ct.magicFor(T, hr) orelse unreachable, ct.magicFor(T, vradius) orelse unreachable);
    }
    return vChainGo(T, srcp, dstp, stride, w, h, hradius, hpasses, vradius, vpasses, sc, {}, {});
}

fn vChainGo(comptime T: type, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, hradius: u32, hpasses: i32, vradius: u32, vpasses: u32, sc: *const Scratch(T), hmg: anytype, vmg: anytype) error{OutOfMemory}!void {
    std.debug.assert(vpasses >= 1 and h >= 2 * vradius + 2);
    const HR = HRows(T, @TypeOf(hmg));
    const Chain = struct {
        const Self = @This();
        hr: HR,
        hb: bool,
        src: []const T,
        dst: []T,
        stride: u32,
        w: u32,
        len: u32,
        r: u32,
        last: u32,
        ring_rows: u32,
        sums: []u32,
        ring: []T,
        next: []u32,
        mg: @TypeOf(vmg),

        fn ringRow(self: *const Self, s: u32, j: u32) []T {
            const base = self.ring[s * self.ring_rows * self.w ..];
            const row = base[(j % self.ring_rows) * self.w ..][0..self.w];
            return simd.opaquePtr(T, row.ptr)[0..self.w];
        }

        // row j of pass s's input, produced on demand
        fn input(self: *Self, s: u32, j: u32) []const T {
            if (s == 0) {
                if (!self.hb) return self.src[j * self.stride ..][0..self.w];
                self.hr.produce(j);
                return self.hr.row(j);
            }
            while (self.next[s - 1] <= j) self.step(s - 1);
            return self.ringRow(s - 1, j);
        }

        // next output row of pass s (vSweepInt / vSweepFloat, one row)
        fn step(self: *Self, s: u32) void {
            const x = self.next[s];
            const r = self.r;
            const wd = self.w;
            const sums = self.sums[s * wd ..][0..wd];
            if (x == 0) {
                _ = self.input(s, r);
                const r_row = self.input(s, r);
                if (@typeInfo(T) == .int) {
                    for (0..wd) |c| sums[c] = @as(u32, r_row[c]) + r;
                    var k: u32 = 0;
                    while (k < r) : (k += 1) {
                        const row = self.input(s, k);
                        for (0..wd) |c| sums[c] += @as(u32, row[c]) << 1;
                    }
                } else {
                    const fs = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums));
                    const div: f32 = 1.0 / @as(f32, @floatFromInt(r * 2 + 1));
                    for (0..wd) |c| fs[c] = r_row[c];
                    var k: u32 = 0;
                    while (k < r) : (k += 1) {
                        const row = self.input(s, k);
                        for (0..wd) |c| {
                            const rv: f32 = row[c];
                            fs[c] += rv * 2;
                        }
                    }
                    for (0..wd) |c| fs[c] = fs[c] * div;
                }
            }
            const len = self.len;
            const add = if (x < len - r) r + x else 2 * len - r - x - 1;
            const sub = if (x <= r) r - x else x - r - 1;
            const a = self.input(s, add);
            const b = self.input(s, sub);
            const out = if (s == self.last) self.dst[x * self.stride ..][0..wd] else self.ringRow(s, x);
            if (@typeInfo(T) == .int) {
                rowAddSubInt(T, sums, a, b, out, wd, self.mg);
            } else {
                const div: f32 = 1.0 / @as(f32, @floatFromInt(r * 2 + 1));
                rowAddSubFloat(T, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(sums)), a, b, out, wd, div);
            }
            self.next[s] = x + 1;
        }
    };

    var next_buf: [64]u32 = undefined;
    const next_heap = if (vpasses > next_buf.len) (try allocator.alloc(u32, vpasses)) else next_buf[0..vpasses];
    defer if (vpasses > next_buf.len) allocator.free(next_heap);
    @memset(next_heap, 0);
    var ch: Chain = .{
        .hr = HR.init(sc, srcp, stride, w, h, hradius, hpasses, vradius, hmg),
        .hb = (hpasses > 0) and (hradius > 0),
        .src = srcp,
        .dst = dstp,
        .stride = stride,
        .w = w,
        .len = h,
        .r = vradius,
        .last = vpasses - 1,
        .ring_rows = 2 * vradius + 2,
        .sums = sc.chain_sums[0 .. vpasses * w],
        .ring = sc.chain_ring,
        .next = next_heap,
        .mg = vmg,
    };
    while (ch.next[ch.last] < h) ch.step(ch.last);
}

/// Vertical blur as `passes` full-plane row-streaming sweeps. Each sweep keeps
/// one running box sum per column (the exact op sequence of blurInt/blurFloat
/// per column, executed for all columns in lockstep row by row), so memory is
/// touched in row order instead of walking down columns.
///
/// Sweeps are always out-of-place, ping-ponging between `tmp` and `dstp` such
/// that the final sweep lands in `dstp`. `first_src` is only read.
pub fn vblur(comptime T: type, first_src: []const T, tmp: []T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, sc: *const Scratch(T)) void {
    if ((passes <= 0) or (radius <= 0)) return;

    if (@typeInfo(T) == .int) {
        if (ct.narrowMagicFor(T, radius)) |mg| return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, sc.sums[0..w], mg);
        return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, sc.sums[0..w], ct.magicFor(T, radius) orelse unreachable);
    }
    return vblurGo(T, first_src, tmp, dstp, stride, w, h, radius, passes, sc.sums[0..w], {});
}

fn vblurGo(comptime T: type, first_src: []const T, tmp: []T, dstp: []T, stride: u32, w: u32, h: u32, radius: u32, passes: i32, sums: []u32, mg: anytype) void {
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
    // Pack groups: one saturating pack + full-width store per 4 (u8) / 2
    // (u16) vectors instead of a per-vector narrow (see ct.packGroup).
    const g = comptime ct.packGroup(T, vec_len);
    if (comptime g > 1) {
        const w_blk = w - (w % (g * vec_len));
        while (c < w_blk) : (c += g * vec_len) {
            var q: [g]U32V = undefined;
            inline for (0..g) |k| {
                const ck = c + k * vec_len;
                const at: @Vector(vec_len, T) = add_row[ck..][0..vec_len].*;
                const st: @Vector(vec_len, T) = sub_row[ck..][0..vec_len].*;
                var s: U32V = sums[ck..][0..vec_len].*;
                s = s +% @as(U32V, @intCast(at));
                s = s -% @as(U32V, @intCast(st));
                sums[ck..][0..vec_len].* = s;
                q[k] = ct.scaleVec32(T, vec_len, s, mg);
            }
            ct.storePacked(T, vec_len, dst_row[c..], q);
        }
    }
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

// --------------------------------------------------------------------------
// Parameters and per-plane dispatch (shared by the VapourSynth and AviSynth
// adapters)
// --------------------------------------------------------------------------

pub const Params = struct {
    hradius: u32,
    hpasses: i32,
    vradius: u32,
    vpasses: i32,

    pub fn hblurOn(p: Params) bool {
        return (p.hradius > 0) and (p.hpasses > 0);
    }

    pub fn vblurOn(p: Params) bool {
        return (p.vradius > 0) and (p.vpasses > 0);
    }

    pub const Error = error{ Nothing, HRadius, VRadius, Magic };

    /// Message for a `check` error (callers add their filter-name prefix).
    pub fn message(e: Error) [:0]const u8 {
        return switch (e) {
            error.Nothing => "nothing to be performed",
            error.HRadius => "hradius too large; 2*hradius must be < the (smallest processed) plane width.",
            error.VRadius => "vradius too large; 2*vradius must be < the (smallest processed) plane height.",
            error.Magic => "radius too large to divide exactly at this bit depth.",
        };
    }

    /// vszip's create checks, in order. `int_bytes` is the integer sample
    /// size (1 or 2), 0 for float; planes not in `planes` are not checked.
    pub fn check(p: Params, int_bytes: u8, planes: []const bool, plane_w: []const u32, plane_h: []const u32) Error!void {
        const hb = p.hblurOn();
        const vb = p.vblurOn();
        if (!vb and !hb) return error.Nothing;

        for (planes, plane_w, plane_h) |on, pw, ph| {
            if (!on) continue;
            if (hb and (@as(u64, p.hradius) * 2 >= pw)) return error.HRadius;
            if (vb and (@as(u64, p.vradius) * 2 >= ph)) return error.VRadius;
        }

        // The integer kernels divide the raw window sum by ksize with an exact
        // multiply-and-shift; past radius 23331 (u16) no exact magic exists. The
        // plane-size checks above already rule this out at any sane resolution,
        // but reject it here rather than let getFrame silently lose precision.
        const magic_ok = switch (int_bytes) {
            inline 1, 2 => |b| blk: {
                const T = if (b == 1) u8 else u16;
                const h_ok = !hb or (ct.magicFor(T, p.hradius) != null);
                const v_ok = !vb or (ct.magicFor(T, p.vradius) != null);
                break :blk h_ok and v_ok;
            },
            else => true,
        };
        if (!magic_ok) return error.Magic;
    }

    /// CT (comptime-radius tap kernels) vs RT (running-sum) selection, measured
    /// 2026-07-03 on Zen 3 (callgrind Ir at every radius + 3-rep wall-clock):
    ///  - integer: RT wins at every radius (Ir -7..-19%, wall >= CT incl. the
    ///    default r=1), so int always takes RT;
    ///  - float: CT's O(ksize) tap kernels only beat the (scalar-h) running sum
    ///    below ~r9 on this box; from r>=9 RT is 1.5-2.7x. CT therefore remains
    ///    for float radius 1..8 only.
    ///  - !hblur/!vblur must take RT: CT's hvBlur applies both directions
    ///    unconditionally (an hpasses=0 request used to get h-blurred anyway).
    /// Boundary moves change outputs slightly (CT and RT differ in pass order,
    /// fixed-point rounding and edge-mirror convention: interior <=2 LSB int /
    /// ~1e-5 f32, edge bands up to ~0.1% of range) — the same difference that
    /// always existed across the old r22 boundary.
    pub fn useRT(p: Params, is_float: bool) bool {
        return !p.hblurOn() or !p.vblurOn() or !is_float or
            (p.hradius != p.vradius) or (p.hradius > 8) or
            (p.hpasses > 1) or (p.vpasses > 1);
    }
};

/// One plane through the RT kernels. `tmp.get()` returns a full temporary
/// plane (same stride) and is only called for the multi-pass vblur fallback
/// on short planes, so hosts can create it lazily.
pub fn blurPlane(comptime T: type, p: Params, srcp: []const T, dstp: []T, stride: u32, w: u32, h: u32, sc: *const Scratch(T), tmp: anytype) error{OutOfMemory}!void {
    const hb = p.hblurOn();
    if (p.vblurOn()) {
        if ((p.vpasses == 1) and (!hb)) {
            vblur(T, srcp, dstp, dstp, stride, w, h, p.vradius, 1, sc);
        } else if ((p.vpasses == 1) and (h >= 2 * p.vradius + 2)) {
            hvBlurFused(T, srcp, dstp, stride, w, h, p.hradius, p.hpasses, p.vradius, sc);
        } else if (h >= 2 * p.vradius + 2) {
            try vChainFused(T, srcp, dstp, stride, w, h, p.hradius, p.hpasses, p.vradius, @intCast(p.vpasses), sc);
        } else {
            // multi-pass vblur ping-pongs between full planes
            const tmpp: []T = try tmp.get();
            if (hb) {
                // land the hblur result so the vblur sweeps end in dstp
                const h_out: []T = if (@mod(p.vpasses, 2) == 1) tmpp else dstp;
                hblur(T, srcp, h_out, stride, w, h, p.hradius, p.hpasses, sc);
                vblur(T, h_out, tmpp, dstp, stride, w, h, p.vradius, p.vpasses, sc);
            } else {
                vblur(T, srcp, tmpp, dstp, stride, w, h, p.vradius, p.vpasses, sc);
            }
        }
    } else {
        hblur(T, srcp, dstp, stride, w, h, p.hradius, p.hpasses, sc);
    }
}
