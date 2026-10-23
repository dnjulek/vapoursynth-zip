//! NLMeans: KNLMeansCL-compatible spatio-temporal non-local means, ported
//! from vszipcu's CUDA NLMeans (itself a port of KNLMeansCL).
//!
//! Algorithm (per output pixel x of frame layer `d`, all values normalized
//! to [0, 1] like OpenCL UNORM images, zero outside the frame):
//!
//!   for every offset q = (qx, qy, qz) in the half search space
//!   (qz*area + qy*side + qx < 0, qz in [-m, 0], m = min(d, n)):
//!     Wc(x)   = weight(patch distance between (d, x) and (d+qz, x+q))
//!     Wm(x-q) = weight(patch distance between (d-qz, x-q) and (d, x))
//!     acc    += Wc(x)*src(d+qz, x+q) + Wm(x-q)*src(d-qz, x-q)
//!     accw   += Wc(x) + Wm(x-q),   u5 = max(u5, Wc(x), Wm(x-q))
//!   out = (src(d, x)*wref*u5 + acc) / (wref*u5 + accw)
//!
//! A weight image W(y) is zero when its own pixel y lies outside the frame,
//! and the patch distance sums per-pixel distances that are zero when the
//! first operand lies outside the frame (the second operand reads the zero
//! border). The GPU's tap order and rounding points are kept: the patch sum
//! is a sequential horizontal sum of 2s+1 terms followed by a sequential
//! vertical sum of 2s+1 row sums. A running (sliding) sum would be cheaper
//! but drifts: the weight argument is the sum times ~190 at h=1.2, so a few
//! ulps of drift on a large sum visibly change the small ones next to it.
//! With wmode 1..3 the output is bit-identical to the GPU port; wmode 0
//! differs only through exp (see `weight`).
//!
//! CPU layout: the frame is cut into tiles. Each tile copies its padded
//! window of every used layer into f32 scratch (converted once, zero border
//! included), then sweeps the offsets. Per offset, weight rows come out of a
//! row pipeline (`Pipe`) in batches and are accumulated into the tile's
//! accumulators right away, so the per-offset work stays in L1 and only the
//! tile window and accumulators live in L2. RAM sees about one read of each
//! input plane per output frame.

const std = @import("std");
const common = @import("../common.zig");
const vcl = @import("../vcl.zig");
const simd = @import("simd.zig");

const allocator = common.kernel_allocator;

/// f32 lanes of the compile target's suggested vector.
pub const N: u32 = common.vec_len / 4;
const V = @Vector(N, f32);

pub const max_d = 16;
pub const max_a = 64;
pub const max_s = 8;

/// Which planes form the joint patch distance (KNLMeansCL `channels`).
pub const Ref = enum(u2) {
    luma,
    chroma,
    yuv,
    rgb,

    pub fn channels(self: Ref) u32 {
        return switch (self) {
            .luma => 1,
            .chroma => 2,
            .yuv, .rgb => 3,
        };
    }
};

pub const Config = struct {
    d: u32,
    a: u32,
    s: u32,
    wmode: u32,
    ref: Ref,
    wref: f32,
    /// NLM_H2_INV_NORM: weight argument = patch sum * k.
    k: f32,
    /// wmode 0 exponent multiplier: exp(-sum*k) = 2^(sum*k2), k2 = -k*log2(e).
    k2: f32,
    /// processed plane geometry
    width: u32,
    height: u32,
    /// tile size (output pixels); tw is a multiple of N
    tw: u32,
    th: u32,

    pub const Error = error{ D, A, S, H, WMode, Wref, Window };

    pub const Args = struct {
        d: i64 = 1,
        a: i64 = 2,
        s: i64 = 4,
        h: f64 = 1.2,
        wmode: i64 = 0,
        wref: f64 = 1.0,
        ref: Ref = .luma,
        width: u32,
        height: u32,
    };

    pub fn init(args: Args) Error!Config {
        if (args.d < 0 or args.d > max_d) return error.D;
        if (args.a < 1 or args.a > max_a) return error.A;
        if (args.s < 0 or args.s > max_s) return error.S;
        if (args.wmode < 0 or args.wmode > 3) return error.WMode;
        if (!std.math.isFinite(args.h) or args.h <= 0 or args.h > std.math.floatMax(f32)) return error.H;
        if (!std.math.isFinite(args.wref) or args.wref < 0 or args.wref > std.math.floatMax(f32)) return error.Wref;
        const d: u32 = @intCast(args.d);
        const a: u32 = @intCast(args.a);
        const s: u32 = @intCast(args.s);
        if (2 * a + 1 > args.width or 2 * a + 1 > args.height) return error.Window;

        // NLM_NORM / (NLM_LEGACY * NLM_H * NLM_H * NLM_S_SIZE), evaluated in
        // f32 left to right exactly as the CUDA/OpenCL preprocessor constant.
        const h: f32 = @floatCast(args.h);
        const s_size: f32 = @floatFromInt((2 * s + 1) * (2 * s + 1));
        const k: f32 = (255.0 * 255.0) / (3.0 * h * h * s_size);
        // h so small that k overflows would turn every weight into NaN
        if (!std.math.isFinite(k)) return error.H;
        // wmode 0 folds log2(e) into k, which overflows ~1.44x earlier
        // (s=4: h below ~1.1e-18). An infinite k2 makes a zero patch
        // distance 0*-inf = NaN, which exp2Bounded maps to weight 0, so
        // even perfect matches would drop out.
        const k2: f32 = -k * std.math.log2e;
        if (args.wmode == 0 and !std.math.isFinite(k2)) return error.H;

        const tw, const th = tileSize(d, a, s, args.ref.channels(), args.width, args.height);
        return .{
            .d = d,
            .a = a,
            .s = s,
            .wmode = @intCast(args.wmode),
            .ref = args.ref,
            .wref = @floatCast(args.wref),
            .k = k,
            .k2 = k2,
            .width = args.width,
            .height = args.height,
            .tw = tw,
            .th = th,
        };
    }

    pub fn message(e: Error) []const u8 {
        return switch (e) {
            error.D => "d must be 0..16.",
            error.A => "a must be 1..64.",
            error.S => "s must be 0..8.",
            error.H => "h must be > 0 (and large enough that 1/h^2 stays finite).",
            error.WMode => "wmode must be 0..3.",
            error.Wref => "wref must be >= 0.",
            error.Window => "research window (2*a+1) larger than the processed plane.",
        };
    }

    fn channels(self: *const Config) u32 {
        return self.ref.channels();
    }
};

/// Tile size in output pixels. Bigger tiles amortize the per-offset fixed
/// costs (2s extra horizontal-sum rows per pipeline, row setup, the a+s
/// window margins), but every thread sweeps its tile window and accumulators
/// once per offset, so past ~1.5 MB per thread the 16 threads start fighting
/// over L3. Measured on a 5700X3D (1080p, 1 and 16 threads): 512x64 was the
/// best or tied-best tile for every d/a/s tried, and full-width tiles lost
/// ~28% at 16 threads while still winning at 1 thread.
fn tileSize(d: u32, a: u32, s: u32, C: u32, width: u32, height: u32) struct { u32, u32 } {
    const budget: usize = 1536 * 1024;
    const layers: usize = 2 * d + 1;
    const p: usize = a + s;
    const min_tw: u32 = @max(4 * N, std.mem.alignForward(u32, 4 * a, N));
    var tw: u32 = @max(min_tw, @min(512, ceilN(width)));
    var th: u32 = @max(1, @min(64, height));
    const bytes = struct {
        fn f(layers_: usize, C_: usize, p_: usize, tw_: u32, th_: u32) usize {
            const window = layers_ * C_ * (th_ + 2 * p_) * (tw_ + 2 * p_ + 2 * N);
            const acc = (C_ + 2) * @as(usize, th_) * tw_;
            return 4 * (window + acc);
        }
    }.f;
    while (bytes(layers, C, p, tw, th) > budget and th > 16) th /= 2;
    while (bytes(layers, C, p, tw, th) > budget and tw / 2 >= min_tw) tw = ceilN(tw / 2);
    return .{ tw, th };
}

inline fn ceilN(x: u32) u32 {
    return std.mem.alignForward(u32, x, N);
}

/// Rows per pipeline step. The vertical patch sums of R adjacent output
/// rows share R+2s row loads instead of R*(2s+1); one row at a time they
/// were L1-load bound (9 loads per output vector at s=4), with R=4 they run
/// ~2x faster. R=4 with 2 vectors per row keeps the 8 accumulators in
/// registers on AVX2.
const batch = 4;

/// Per-thread working memory for one plane geometry.
pub const Scratch = struct {
    mem: []align(common.vec_len) f32,
    /// source window, layers * channels planes of bh rows x bw
    src: []f32,
    /// weight window from rclip (aliases `src` without rclip)
    wsrc: []f32,
    dist: []f32,
    /// horizontal patch-sum rings of the two weight pipelines
    ring_a: []f32,
    ring_b: []f32,
    /// weight rows: a ring of a+batch rows (spatial offsets) and a batch
    w_a: []f32,
    w_b: []f32,
    /// tile accumulators, row-major: per tile row, C value rows, the
    /// weight-sum row and the max-weight row (`as` floats each)
    acc: []f32,
    out: []f32,

    bw: u32,
    bh: u32,
    /// row stride of weight rows and rings
    rs: u32,
    as: u32,

    pub fn create(cfg: *const Config, has_ref: bool) error{OutOfMemory}!*Scratch {
        const C = cfg.channels();
        const layers = 2 * cfg.d + 1;
        const p = cfg.a + cfg.s;
        const taps = 2 * cfg.s + 1;
        const bw = ceilN(cfg.tw + 2 * p) + N;
        const bh = cfg.th + 2 * p;
        const rs = ceilN(cfg.tw + cfg.a) + N;
        const as = ceilN(cfg.tw);

        const src_len: usize = @as(usize, layers) * C * bh * bw;
        const ring_len: usize = @as(usize, taps + batch - 1) * rs;
        const lens = [_]usize{
            src_len,
            if (has_ref) src_len else 0,
            rs + 2 * cfg.s + N,
            ring_len,
            ring_len,
            @as(usize, cfg.a + batch) * rs,
            @as(usize, batch) * rs,
            @as(usize, C + 2) * cfg.th * as,
            @as(usize, C) * as,
        };
        var total: usize = 0;
        for (lens) |l| total += std.mem.alignForward(usize, l, common.vec_len / @sizeOf(f32));

        const self = try allocator.create(Scratch);
        errdefer allocator.destroy(self);
        const mem = try allocator.alignedAlloc(f32, common.alignment, total);
        // Lanes past a row's width are computed but never stored to the
        // output; zero everything once so they never see NaN garbage.
        @memset(mem, 0);

        var off: usize = 0;
        var parts: [lens.len][]f32 = undefined;
        for (lens, 0..) |l, i| {
            parts[i] = mem[off..][0..l];
            off += std.mem.alignForward(usize, l, common.vec_len / @sizeOf(f32));
        }
        self.* = .{
            .mem = mem,
            .src = parts[0],
            .wsrc = if (has_ref) parts[1] else parts[0],
            .dist = parts[2],
            .ring_a = parts[3],
            .ring_b = parts[4],
            .w_a = parts[5],
            .w_b = parts[6],
            .acc = parts[7],
            .out = parts[8],
            .bw = bw,
            .bh = bh,
            .rs = rs,
            .as = as,
        };
        return self;
    }

    pub fn destroy(self: *Scratch) void {
        allocator.free(self.mem);
        allocator.destroy(self);
    }
};

/// One input plane: data plus stride in samples.
pub fn PlaneIn(comptime T: type) type {
    return struct { data: []const T, stride: u32 };
}

pub fn PlaneOut(comptime T: type) type {
    return struct { data: []T, stride: u32 };
}

/// Tile-local geometry and buffer views shared by the stages.
const Ctx = struct {
    width: i32,
    height: i32,
    s: u32,
    p: i32,
    bw: u32,
    bh: u32,
    rs: u32,
    as: u32,
    /// floats between two (layer, channel) planes of the source window
    plane: usize,
    C: u32,
    /// sum -> weight argument (wmode 1..3) and sum -> exp2 argument (wmode 0)
    k: V,
    k2: V,
    x0: i32,
    y0: i32,
    tw: u32,
    th: u32,
    src: [*]f32,
    wsrc: [*]f32,
    dist: [*]f32,

    /// index of frame position (x, y) in channel 0 of window layer `t`
    inline fn idx(cx: *const Ctx, t: u32, x: i32, y: i32) usize {
        const row: usize = @intCast(y - cx.y0 + cx.p);
        const col: usize = @intCast(x - cx.x0 + cx.p);
        return @as(usize, t) * cx.C * cx.plane + row * cx.bw + col;
    }
};

inline fn load(p: [*]const f32) V {
    return p[0..N].*;
}

inline fn store(p: [*]f32, v: V) void {
    p[0..N].* = v;
}

inline fn vmax(a: V, b: V) V {
    // maxps directly: inputs are never NaN when the source is finite, and
    // @max's NaN contract would add a compare+blend.
    return @select(f32, a > b, a, b);
}

/// Per-pixel joint distance between window pixels `a` and `b` (channel 0
/// addresses; further channels are `plane` floats apart). The fused
/// multiply-adds are the ones nvcc contracts the CUDA expressions into
/// (read from the NVRTC PTX, sm_86, default -fmad=true); the GPU's
/// wmode 1..3 bit-exactness depends on them.
inline fn pixDist(comptime ref: Ref, a: [*]const f32, b: [*]const f32, plane: usize) V {
    switch (ref) {
        .luma => {
            const d0 = load(a) - load(b);
            return @as(V, @splat(3.0)) * (d0 * d0);
        },
        .chroma => {
            // 1.5f * (du*du + dv*dv): fma(du, du, dv*dv) * 1.5
            const du = load(a) - load(b);
            const dv = load(a + plane) - load(b + plane);
            return @as(V, @splat(1.5)) * simd.madd(V, du, du, dv * dv);
        },
        .yuv => {
            // dy*dy + du*du + dv*dv: fma(dv, dv, fma(dy, dy, du*du))
            const dy = load(a) - load(b);
            const du = load(a + plane) - load(b + plane);
            const dv = load(a + 2 * plane) - load(b + 2 * plane);
            return simd.madd(V, dv, dv, simd.madd(V, dy, dy, du * du));
        },
        .rgb => {
            // (2/3 + m)*(dr*dr) + (4/3)*(dg*dg) + (1 - m)*(db*db):
            // fma(1 - m, db*db, fma(dr*dr, 2/3 + m, (4/3)*(dg*dg)))
            const ar = load(a);
            const br = load(b);
            const dr = ar - br;
            const dg = load(a + plane) - load(b + plane);
            const db = load(a + 2 * plane) - load(b + 2 * plane);
            const m_red = (ar + br) / @as(V, @splat(6.0));
            const t1 = @as(V, @splat(4.0 / 3.0)) * (dg * dg);
            const t01 = simd.madd(V, dr * dr, @as(V, @splat(2.0 / 3.0)) + m_red, t1);
            const dist = simd.madd(V, @as(V, @splat(1.0)) - m_red, db * db, t01);
            // A red mean above 3 or below -2 (extended-range input, e.g. PQ
            // converted to linear light) turns a coefficient negative. Then
            // weights exceed 1 and the sums overflow to inf/NaN in every
            // wmode. Clamping keeps every weight <= 1. For [0, 1] input every
            // term is >= +0, so the clamp is a no-op there and GPU parity
            // holds; outside that range the GPU itself returns non-finite
            // output, so there is nothing to match.
            return vmax(dist, @splat(0.0));
        },
    }
}

/// Degree of the exp2 polynomial for wmode 0. Measured at 1080p, d=1 a=2
/// s=4, against VCL2 exp_f (1 thread / 16 threads, fps):
///   VCL2 exp_f 6.81 / 57.9, degree 6 7.32 / 62.5, degree 5 7.54 / 64.9,
///   degree 4 8.04 / 67.2, degree 3 8.16 / 68.5.
/// Degree 4 (22.6 ulp on 2^y, weight error < 3e-6 relative) changes the
/// output by at most ~1.3e-6 in float and +-1 LSB on <= 0.002% of 16-bit
/// pixels against an exact (f64) exp; the GPU's own expf is ~3e-7 off.
const exp_degree = 4;

inline fn weight(comptime wmode: u32, sum: V, k: V, k2: V) V {
    switch (wmode) {
        0 => {
            // exp(-sum*k) computed as 2^(sum * -k*log2(e)). The GPU rounds
            // sum*k before expf; folding log2(e) into the constant moves
            // that rounding (<= 1 ulp of the argument).
            // Distances are >= 0 for every `ref` (RGB clamps in pixDist),
            // so y <= 0 and the weight is in [0, 1].
            return vcl.exp2Bounded(exp_degree, sum * k2);
        },
        else => {
            // the product is rounded on its own (the GPU pins it with __fmul_rn)
            const arg = sum * k;
            const one: V = @splat(1.0);
            // fdimf(1, arg)
            var c = @select(f32, one > arg, one - arg, @as(V, @splat(0.0)));
            if (wmode >= 2) c = c * c;
            if (wmode == 3) {
                c = c * c;
                c = c * c;
            }
            return c;
        },
    }
}

/// In-place weights over `no` floats. Four independent vectors per step:
/// one exp is a long dependency chain, and interleaving them in program
/// order lets the scheduler overlap chains (one at a time was ~13% slower).
inline fn weightRow(comptime wmode: u32, out: [*]f32, no: u32, k: V, k2: V) void {
    var i: u32 = 0;
    while (i + 4 * N <= no) : (i += 4 * N) {
        var v: [4]V = undefined;
        inline for (0..4) |j| v[j] = load(out + i + j * N);
        inline for (0..4) |j| v[j] = weight(wmode, v[j], k, k2);
        inline for (0..4) |j| store(out + i + j * N, v[j]);
    }
    while (i < no) : (i += N) store(out + i, weight(wmode, load(out + i), k, k2));
}

/// Walks `len` floats (a multiple of N) in blocks of 8, 4, 2, 1 vectors.
const Blocks = struct {
    i: u32 = 0,
    len: u32,

    fn init(len: u32) Blocks {
        return .{ .len = len };
    }

    fn next(self: *Blocks) ?struct { i: u32, u: u32 } {
        const left = (self.len - self.i) / N;
        if (left == 0) return null;
        const u: u32 = if (left >= 8) 8 else if (left >= 4) 4 else if (left >= 2) 2 else 1;
        defer self.i += u * N;
        return .{ .i = self.i, .u = u };
    }
};

/// out[k] = p[k*N] + p[k*N + 1] + ... (taps terms, summed in that order)
/// for U adjacent vectors: the horizontal patch-row sum. The U independent
/// chains hide the add latency; what remains is load-bound (taps unaligned
/// loads per output, some splitting a cache line). Building the shifted taps
/// in registers (vperm2f128 + vpalignr) instead ran slower on Zen 3, and so
/// did a second, 32-byte shifted copy of the row that avoids split loads
/// (the extra store cost more than the splits), and so did vsumBlock's
/// laundered-FMA trick for every other tap (-13% at 1T, -7% at 16T: the
/// loop is load-bound, 16 load-adds per 2 taps, so moving adds to the FMA
/// pipes only lengthens the chains).
inline fn blockSum(comptime U: u32, p: [*]const f32, taps: u32) [U]V {
    var acc: [U]V = undefined;
    inline for (0..U) |k| acc[k] = load(p + k * N);
    var j: u32 = 1;
    while (j < taps) : (j += 1) {
        inline for (0..U) |k| acc[k] += load(p + j + k * N);
    }
    return acc;
}

inline fn storeBlock(comptime U: u32, p: [*]f32, v: [U]V) void {
    inline for (0..U) |k| store(p + k * N, v[k]);
}

/// outs[r] = rows[r] + rows[r+1] + ... + rows[r+taps-1] (in that order).
fn vsumRows(comptime R: u32, comptime taps: u32, rows: *const [batch + 2 * max_s][*]const f32, outs: *const [R][*]f32, no: u32) void {
    const U: u32 = if (R >= 4) 2 else if (R == 2) 4 else 8;
    var i: u32 = 0;
    while (i + U * N <= no) : (i += U * N) vsumBlock(R, U, taps, rows, outs, i);
    while (i < no) : (i += N) vsumBlock(R, 1, taps, rows, outs, i);
}

inline fn vsumBlock(comptime R: u32, comptime U: u32, comptime taps: u32, rows: *const [batch + 2 * max_s][*]const f32, outs: *const [R][*]f32, i: u32) void {
    // Every other tap goes through an FMA with a laundered 1.0: x*1 + acc
    // rounds exactly like acc + x but issues on the multiply pipes, so the
    // adds use all 4 FP pipes on Zen 3 (2 of them are add-only). LLVM would
    // fold a visible 1.0 back into an add. With the loads shared by 4 rows
    // the sums are add-bound; this took them from 3.00 to 2.16 tsc/vector.
    const one = simd.opaqueF32V(N, @splat(1.0));
    var acc: [R][U]V = undefined;
    inline for (0..R + taps - 1) |j| {
        var x: [U]V = undefined;
        inline for (0..U) |u| x[u] = load(rows[j] + i + u * N);
        inline for (0..R) |r| {
            if (j >= r and j < r + taps) {
                inline for (0..U) |u| {
                    if (j == r) {
                        acc[r][u] = x[u];
                    } else if ((j - r) % 2 == 0) {
                        acc[r][u] = simd.madd(V, x[u], one, acc[r][u]);
                    } else {
                        acc[r][u] += x[u];
                    }
                }
            }
        }
    }
    inline for (0..R) |r| inline for (0..U) |u| store(outs[r] + i + u * N, acc[r][u]);
}

/// Produces the rows of one weight image W, in increasing y, over a fixed
/// column range: W(y) compares window layer `t` at y with layer `t2` at
/// y + (qx, qy). Horizontal patch-row sums go into a ring of 2s+batch rows
/// and each batch of weight rows is summed, weighted and handed to the
/// accumulator while still in L1. (Whole-region passes - all hsum rows, then
/// all vertical sums, then all weights - streamed every pass through L2.)
const Pipe = struct {
    ring: [*]f32,
    t: u32,
    t2: u32,
    qx: i32,
    qy: i32,
    x0: i32,
    w: u32,
    /// next hsum row (frame y) to compute, and its ring slot
    next: i32,
    slot: u32,

    fn init(ring: [*]f32, cx: *const Ctx, t: u32, t2: u32, qx: i32, qy: i32, x0: i32, w: u32, y_first: i32) Pipe {
        return .{
            .ring = ring,
            .t = t,
            .t2 = t2,
            .qx = qx,
            .qy = qy,
            .x0 = x0,
            .w = w,
            .next = y_first - @as(i32, @intCast(cx.s)),
            .slot = 0,
        };
    }

    fn hsumRow(self: *Pipe, comptime ref: Ref, cx: *const Ctx, yy: i32) void {
        const s = cx.s;
        const taps = 2 * s + 1;
        const no = ceilN(self.w);
        const r0 = self.ring + @as(usize, self.slot) * cx.rs;
        if (yy < 0 or yy >= cx.height) {
            @memset(r0[0..no], 0);
            return;
        }

        // per-pixel distances over the patch-extended columns; zero where
        // the first operand is outside the frame
        const span = self.w + 2 * s;
        const xs: i32 = self.x0 - @as(i32, @intCast(s));
        const pa = cx.wsrc + cx.idx(self.t, xs, yy);
        const pb = cx.wsrc + cx.idx(self.t2, xs + self.qx, yy + self.qy);
        const nd = ceilN(span);
        var i: u32 = 0;
        while (i < nd) : (i += N) store(cx.dist + i, pixDist(ref, pa + i, pb + i, cx.plane));
        if (xs < 0) @memset(cx.dist[0..@min(@as(u32, @intCast(-xs)), span)], 0);
        const z_hi: i32 = cx.width - xs;
        if (z_hi < @as(i32, @intCast(span))) @memset(cx.dist[@intCast(@max(z_hi, 0))..span], 0);

        var blk = Blocks.init(no);
        while (blk.next()) |b| switch (b.u) {
            inline 1, 2, 4, 8 => |u| storeBlock(u, r0 + b.i, blockSum(u, cx.dist + b.i, taps)),
            else => unreachable,
        };
    }

    /// Writes W rows y .. y+R-1 (ceil(w) floats each) to `outs`. Batches
    /// must be requested in increasing, consecutive y.
    fn produce(self: *Pipe, comptime ref: Ref, comptime wmode: u32, comptime R: u32, cx: *const Ctx, y: i32, outs: *const [R][*]f32) void {
        const s: i32 = @intCast(cx.s);
        const nring = 2 * cx.s + batch;
        while (self.next <= y + @as(i32, R - 1) + s) : (self.next += 1) {
            self.hsumRow(ref, cx, self.next);
            self.slot = if (self.slot + 1 == nring) 0 else self.slot + 1;
        }

        // ring rows y-s .. y+R-1+s, oldest first; `next` is one past them
        const nrows = R + 2 * cx.s;
        var rows: [batch + 2 * max_s][*]const f32 = undefined;
        var sl = (self.slot + nring - nrows) % nring;
        for (0..nrows) |j| {
            rows[j] = self.ring + @as(usize, sl) * cx.rs;
            sl = if (sl + 1 == nring) 0 else sl + 1;
        }
        const no = ceilN(self.w);
        switch (cx.s) {
            inline 0...max_s => |cs| vsumRows(R, 2 * cs + 1, &rows, outs, no),
            else => unreachable,
        }

        inline for (0..R) |r| {
            const yy = y + @as(i32, r);
            const out = outs[r];
            if (yy < 0 or yy >= cx.height) {
                @memset(out[0..no], 0);
            } else {
                // weights in their own loop: fused into the sums, the exp
                // constants spill on AVX2's 16 ymm registers
                weightRow(wmode, out, no, cx.k, cx.k2);
                // pixels outside the frame have weight 0
                if (self.x0 < 0) @memset(out[0..@min(@as(u32, @intCast(-self.x0)), self.w)], 0);
                const z_hi: i32 = cx.width - self.x0;
                if (z_hi < @as(i32, @intCast(self.w))) @memset(out[@intCast(@max(z_hi, 0))..self.w], 0);
            }
        }
    }
};

/// Adds offset q's contribution to tile row `y`: Wc(x) from `wc` and
/// Wm(x - q) from `wm`, both positioned at the tile's first column.
inline fn accumulateRow(comptime C: u32, cx: *const Ctx, tp: u32, tm: u32, qx: i32, qy: i32, y: i32, wc: [*]const f32, wm: [*]const f32, acc_row: [*]f32) void {
    const sp = cx.src + cx.idx(tp, cx.x0 + qx, y + qy);
    const sm = cx.src + cx.idx(tm, cx.x0 - qx, y - qy);
    const as = cx.as;
    const nv = ceilN(cx.tw);
    var i: u32 = 0;
    while (i < nv) : (i += N) {
        const wc_v = load(wc + i);
        const wm_v = load(wm + i);
        inline for (0..C) |c| {
            const pa = acc_row + c * as + i;
            const pq = load(sp + c * cx.plane + i);
            const mq = load(sm + c * cx.plane + i);
            store(pa, load(pa) + simd.madd(V, wc_v, pq, wm_v * mq));
        }
        const pw = acc_row + C * as + i;
        store(pw, load(pw) + (wc_v + wm_v));
        const pu5 = acc_row + (C + 1) * as + i;
        store(pu5, vmax(wc_v, vmax(wm_v, load(pu5))));
    }
}

/// Converts a source plane window into the f32 scratch: rows/cols outside
/// the frame become 0.
fn loadWindow(comptime T: type, cx: *const Ctx, dst: [*]f32, src: PlaneIn(T), peak: f32) void {
    const bx0: i32 = cx.x0 - cx.p;
    const by0: i32 = cx.y0 - cx.p;
    const c_lo: u32 = @intCast(std.math.clamp(-bx0, 0, @as(i32, @intCast(cx.bw))));
    const c_hi: u32 = @intCast(std.math.clamp(cx.width - bx0, 0, @as(i32, @intCast(cx.bw))));
    var r: u32 = 0;
    while (r < cx.bh) : (r += 1) {
        const row = dst[@as(usize, r) * cx.bw ..][0..cx.bw];
        const yy: i32 = by0 + @as(i32, @intCast(r));
        if (yy < 0 or yy >= cx.height) {
            @memset(row, 0);
            continue;
        }
        @memset(row[0..c_lo], 0);
        @memset(row[c_hi..], 0);
        const sx: usize = @intCast(bx0 + @as(i32, @intCast(c_lo)));
        const srow = src.data[@as(usize, @intCast(yy)) * src.stride + sx ..][0 .. c_hi - c_lo];
        const out = row[c_lo..c_hi];
        switch (@typeInfo(T)) {
            // UNORM decode is a division (not a reciprocal multiply), like
            // read_imagef / the GPU port's LOADU1.
            .int => for (out, srow) |*o, v| {
                o.* = @as(f32, @floatFromInt(v)) / peak;
            },
            else => for (out, srow) |*o, v| {
                o.* = @floatCast(v);
            },
        }
    }
}

/// out = (center*m + acc)/den per channel, converted into `dst` rows.
fn finish(comptime T: type, comptime C: u32, cx: *const Ctx, d: u32, wref: f32, acc: [*]const f32, tmp: [*]f32, dst: [C]PlaneOut(T), peak: f32) void {
    const as = cx.as;
    const nv = ceilN(cx.tw);
    var y: u32 = 0;
    while (y < cx.th) : (y += 1) {
        const yy: i32 = cx.y0 + @as(i32, @intCast(y));
        const ar = acc + @as(usize, y) * (C + 2) * as;
        const ctr = cx.src + cx.idx(d, cx.x0, yy);
        var i: u32 = 0;
        while (i < nv) : (i += N) {
            // wref*u5 stays a rounded product (the GPU pins it with __fmul_rn)
            const m = @as(V, @splat(wref)) * load(ar + (C + 1) * as + i);
            const den = m + load(ar + C * as + i);
            inline for (0..C) |c| {
                const center = load(ctr + c * cx.plane + i);
                const v = simd.madd(V, center, m, load(ar + c * as + i)) / den;
                // den is >= wref*2^-23 > 0 unless wref == 0 and every weight
                // underflowed; keep the source pixel there instead of NaN.
                store(tmp + c * as + i, @select(f32, den > @as(V, @splat(0.0)), v, center));
            }
        }

        const row_off = @as(usize, @intCast(yy));
        const x0: usize = @intCast(cx.x0);
        inline for (0..C) |c| {
            const o = dst[c].data[row_off * dst[c].stride + x0 ..][0..cx.tw];
            const t = tmp[c * as ..][0..cx.tw];
            switch (@typeInfo(T)) {
                .int => for (o, t) |*ov, tv| {
                    // __float2int_rn(x * peak) clamped to [0, peak]; NaN -> 0
                    var v = tv * peak;
                    v = if (v > 0) v else 0;
                    v = @min(v, peak);
                    // round half to even (the magic add is exact below 2^22)
                    v = (v + 12582912.0) - 12582912.0;
                    ov.* = @intFromFloat(v);
                },
                else => for (o, t) |*ov, tv| {
                    ov.* = @floatCast(tv);
                },
            }
        }
    }
}

/// Filters one frame. `srcs`/`refs` hold `(2d+1) * C` planes indexed
/// `layer * C + channel`; only layers `d-m ..= d+m` are read. `refs` is
/// null without rclip. `peak` is the integer format's maximum (unused for
/// float formats).
pub fn processFrame(
    comptime T: type,
    comptime ref: Ref,
    comptime wmode: u32,
    cfg: *const Config,
    sc: *Scratch,
    srcs: []const PlaneIn(T),
    refs: ?[]const PlaneIn(T),
    dst: [ref.channels()]PlaneOut(T),
    m: u32,
    peak: f32,
) void {
    const C = comptime ref.channels();
    std.debug.assert(m <= cfg.d);
    std.debug.assert(cfg.ref == ref and cfg.wmode == wmode);
    std.debug.assert(srcs.len == (2 * cfg.d + 1) * C);

    const a: i32 = @intCast(cfg.a);
    const side: i32 = 2 * a + 1;
    const area: i32 = side * side;
    const d = cfg.d;
    const di: i32 = @intCast(d);
    const mi: i32 = @intCast(m);

    var cx: Ctx = .{
        .width = @intCast(cfg.width),
        .height = @intCast(cfg.height),
        .s = cfg.s,
        .p = @intCast(cfg.a + cfg.s),
        .bw = sc.bw,
        .bh = sc.bh,
        .rs = sc.rs,
        .as = sc.as,
        .plane = @as(usize, sc.bh) * sc.bw,
        .C = C,
        .k = @splat(cfg.k),
        .k2 = @splat(cfg.k2),
        .x0 = 0,
        .y0 = 0,
        .tw = 0,
        .th = 0,
        .src = sc.src.ptr,
        .wsrc = sc.wsrc.ptr,
        .dist = sc.dist.ptr,
    };

    const acc = sc.acc.ptr;
    const acc_stride: usize = (C + 2) * @as(usize, sc.as);
    const rs: usize = sc.rs;

    var ty: u32 = 0;
    while (ty < cfg.height) : (ty += cfg.th) {
        var tx: u32 = 0;
        while (tx < cfg.width) : (tx += cfg.tw) {
            cx.x0 = @intCast(tx);
            cx.y0 = @intCast(ty);
            cx.tw = @min(cfg.tw, cfg.width - tx);
            cx.th = @min(cfg.th, cfg.height - ty);
            const y0 = cx.y0;
            const y_end: i32 = y0 + @as(i32, @intCast(cx.th));

            // source windows of the used layers
            var t: u32 = d - m;
            while (t <= d + m) : (t += 1) {
                inline for (0..C) |c| {
                    const off = (@as(usize, t) * C + c) * cx.plane;
                    loadWindow(T, &cx, sc.src.ptr + off, srcs[t * C + c], peak);
                    if (refs) |rp| loadWindow(T, &cx, sc.wsrc.ptr + off, rp[t * C + c], peak);
                }
            }

            // accumulators: values and weight sum 0, max weight seeded with
            // FLT_EPSILON so den > 0
            for (0..cx.th) |y| {
                const ar = acc[y * acc_stride ..][0..acc_stride];
                @memset(ar[0 .. (C + 1) * sc.as], 0);
                @memset(ar[(C + 1) * sc.as ..], 0x1p-23);
            }

            var qz: i32 = -mi;
            while (qz <= 0) : (qz += 1) {
                const tp: u32 = @intCast(di + qz);
                const tm: u32 = @intCast(di - qz);
                var qy: i32 = -a;
                while (qy <= a) : (qy += 1) {
                    var qx: i32 = -a;
                    while (qx <= a) : (qx += 1) {
                        if (qz * area + qy * side + qx >= 0) break;
                        if (qz == 0) {
                            // Wc and Wm are the same image (qy <= 0 here):
                            // one pipeline over the union of the tile's and
                            // the tile-q's columns runs |qy| rows ahead and
                            // keeps the last |qy|+batch weight rows in a ring
                            // (row r lives in slot (r - y0) mod nring).
                            const lag: i32 = -qy;
                            const nring: u32 = @as(u32, @intCast(lag)) + batch;
                            const ux0 = cx.x0 - @max(qx, 0);
                            var pipe = Pipe.init(sc.ring_a.ptr, &cx, d, d, qx, qy, ux0, cx.tw + @abs(qx), y0);
                            const off_c: usize = @intCast(cx.x0 - ux0);
                            const off_m: usize = @intCast(cx.x0 - qx - ux0);
                            const wring = sc.w_a.ptr;
                            var y: i32 = y0;
                            while (y < y0 + lag) : (y += 1) {
                                pipe.produce(ref, wmode, 1, &cx, y, &.{wring + @as(usize, @intCast(y - y0)) * rs});
                            }
                            y = y0;
                            while (y < y_end) {
                                const step: u32 = if (y_end - y >= batch) batch else 1;
                                switch (step) {
                                    inline 1, batch => |R| {
                                        var outs: [R][*]f32 = undefined;
                                        inline for (0..R) |r| outs[r] = wring + (@as(u32, @intCast(y + lag - y0 + @as(i32, r))) % nring) * rs;
                                        pipe.produce(ref, wmode, R, &cx, y + lag, &outs);
                                        inline for (0..R) |r| {
                                            const yr = y + @as(i32, r);
                                            const wc = wring + (@as(u32, @intCast(yr - y0)) % nring) * rs;
                                            const ar = acc + @as(usize, @intCast(yr - y0)) * acc_stride;
                                            accumulateRow(C, &cx, d, d, qx, qy, yr, wc + off_c, outs[r] + off_m, ar);
                                        }
                                    },
                                    else => unreachable,
                                }
                                y += @intCast(step);
                            }
                        } else {
                            var pc = Pipe.init(sc.ring_a.ptr, &cx, d, tp, qx, qy, cx.x0, cx.tw, y0);
                            var pm = Pipe.init(sc.ring_b.ptr, &cx, tm, d, qx, qy, cx.x0 - qx, cx.tw, y0 - qy);
                            var y: i32 = y0;
                            while (y < y_end) {
                                const step: u32 = if (y_end - y >= batch) batch else 1;
                                switch (step) {
                                    inline 1, batch => |R| {
                                        var oc: [R][*]f32 = undefined;
                                        var om: [R][*]f32 = undefined;
                                        inline for (0..R) |r| {
                                            oc[r] = sc.w_a.ptr + r * rs;
                                            om[r] = sc.w_b.ptr + r * rs;
                                        }
                                        pc.produce(ref, wmode, R, &cx, y, &oc);
                                        pm.produce(ref, wmode, R, &cx, y - qy, &om);
                                        inline for (0..R) |r| {
                                            const yr = y + @as(i32, r);
                                            const ar = acc + @as(usize, @intCast(yr - y0)) * acc_stride;
                                            accumulateRow(C, &cx, tp, tm, qx, qy, yr, oc[r], om[r], ar);
                                        }
                                    },
                                    else => unreachable,
                                }
                                y += @intCast(step);
                            }
                        }
                    }
                }
            }

            finish(T, C, &cx, d, cfg.wref, acc, sc.out.ptr, dst, peak);
        }
    }
}
