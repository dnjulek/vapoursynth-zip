const std = @import("std");
const simd = @import("simd.zig");
const math = std.math;
const vcl = @import("../vcl.zig");
const srgb = @import("srgb.zig");

const allocator = std.heap.c_allocator;

const ksize = 9;
const radius = 4;
const vec_size: comptime_int = std.simd.suggestVectorLength(f32) orelse 8;
const weight_pruning: f64 = 0.01;

const SkipInfo = struct {
    ssim: bool,
    artifact: bool,
    detailloss: bool,

    fn all(self: SkipInfo) bool {
        return self.ssim and self.artifact and self.detailloss;
    }
};

const skip_table: [3][6]SkipInfo = blk: {
    var table: [3][6]SkipInfo = undefined;
    var plane: usize = 0;
    while (plane < 3) : (plane += 1) {
        var scale: usize = 0;
        while (scale < 6) : (scale += 1) {
            const base = plane * 36 + scale * 6;
            table[plane][scale] = .{
                .ssim = weight[base + 0] <= weight_pruning and weight[base + 3] <= weight_pruning,
                .artifact = weight[base + 1] <= weight_pruning and weight[base + 4] <= weight_pruning,
                .detailloss = weight[base + 2] <= weight_pruning and weight[base + 5] <= weight_pruning,
            };
        }
    }
    break :blk table;
};

/// One input clip as the scale-0 sweep reads it: three RGB planes in their
/// native sample type, decoded to linear light one row at a time (so no
/// linear plane is ever stored).
pub const Input = struct {
    pub const Kind = enum { u8, u16, f32, yuv };
    kind: Kind,
    planes: [3][*]const u8,
    stride: usize, // bytes (RGB kinds)
    /// integer kinds: sample -> linear light (see buildLut)
    lut: []const f32 = &.{},
    /// f32 and yuv kinds: decode with the sRGB transfer (false: already linear)
    srgb: bool = true,
    /// yuv kind: per-plane strides in bytes, and the conversion
    strides: [3]usize = .{ 0, 0, 0 },
    yuv: Yuv = undefined,

    /// The three linear RGB rows of source row y.
    fn rows(self: *const Input, y: u32, dst: [3][]f32, w: u32, h: u32, tmp: [4][]f32) void {
        if (self.kind == .yuv) {
            yuvRow(self, y, dst, w, h, tmp);
            for (0..3) |c| {
                if (self.srgb) linearizeRow(dst[c][0..w], dst[c][0..w]);
            }
            return;
        }
        for (0..3) |c| self.row(c, y, dst[c], w);
    }

    fn row(self: Input, c: usize, y: u32, dst: []f32, w: u32) void {
        const base = self.planes[c] + y * self.stride;
        switch (self.kind) {
            .u8 => for (dst[0..w], base[0..w]) |*o, v| {
                o.* = self.lut[v];
            },
            .u16 => {
                const sp: [*]const u16 = @ptrCast(@alignCast(base));
                for (dst[0..w], sp[0..w]) |*o, v| o.* = self.lut[v];
            },
            .f32 => {
                const sp: [*]const f32 = @ptrCast(@alignCast(base));
                if (self.srgb) linearizeRow(sp[0..w], dst[0..w]) else @memcpy(dst[0..w], sp[0..w]);
            },
            .yuv => unreachable,
        }
    }
};

/// YUV input converted to RGB inside the scale-0 sweep, one row at a time,
/// instead of through a resize.Bicubic(format=RGBS) node: those intermediate
/// 1080p RGBS frames cost ~2.3 GB at 16 threads (they queue behind this slow
/// consumer) and a third of the throughput. Same steps as zimg's graph:
/// samples normalized to float (range from the frame), chroma upsampled with
/// Bicubic b=0 c=0.5 (Catmull-Rom) at zimg's chroma-location phases with
/// mirrored borders, then the YUV -> RGB matrix. Not bit-exact with zimg
/// (rounding order), ~1e-7 on the RGB values.
pub const Yuv = struct {
    sample: enum { u8, u16, f32 },
    /// 2x chroma subsampling per axis
    ssw: bool,
    ssh: bool,
    cw: u32,
    ch: u32,
    /// normalized value = v * mul + add (float samples: 1, 0)
    y_mul: f32 = 1,
    y_add: f32 = 0,
    c_mul: f32 = 1,
    c_add: f32 = 0,
    /// R = Y + cr_r*V, G = Y + cb_g*U + cr_g*V, B = Y + cb_b*U
    cr_r: f32,
    cb_g: f32,
    cr_g: f32,
    cb_b: f32,
    /// ssw: taps of each output column (len = luma width)
    htab: []const Tap = &.{},
    /// ssh: chroma-location shift of the vertical source position
    vshift: f64 = 0,

    /// Range normalization of a `bits`-deep integer clip (zimg's: limited
    /// Y (v - 16k)/219k, C (v - 128k)/224k with k = 2^(bits-8); full Y
    /// v/(2^bits - 1), C (v - 2^(bits-1))/(2^bits - 1)).
    pub fn setRange(y: *Yuv, bits_in: u8, full: bool) void {
        if (y.sample == .f32) return;
        std.debug.assert(bits_in >= 8 and bits_in <= 16);
        const bits: u5 = @intCast(bits_in);
        const k: f64 = @floatFromInt(@as(u32, 1) << (bits - 8));
        const peak: f64 = @floatFromInt((@as(u32, 1) << bits) - 1);
        if (full) {
            y.y_mul = @floatCast(1.0 / peak);
            y.y_add = 0;
            y.c_mul = @floatCast(1.0 / peak);
            y.c_add = @floatCast(-128.0 * k / peak);
        } else {
            y.y_mul = @floatCast(1.0 / (219.0 * k));
            y.y_add = @floatCast(-16.0 / 219.0);
            y.c_mul = @floatCast(1.0 / (224.0 * k));
            y.c_add = @floatCast(-128.0 / 224.0);
        }
    }

    /// Luma weights (kr, kb) of a VapourSynth/zimg non-constant-luminance
    /// matrix, null for the others (RGB, YCgCo, 2020 CL, ICtCp, ...).
    pub fn matrixWeights(m: i64) ?[2]f64 {
        return switch (m) {
            1 => .{ 0.2126, 0.0722 }, // BT.709
            4 => .{ 0.30, 0.11 }, // FCC
            5, 6 => .{ 0.299, 0.114 }, // BT.470BG, ST 170M
            7 => .{ 0.212, 0.087 }, // ST 240M
            9 => .{ 0.2627, 0.0593 }, // BT.2020 NCL
            else => null,
        };
    }

    /// YUV -> RGB coefficients for luma weights kr, kb (ncl).
    pub fn setMatrix(y: *Yuv, kr: f64, kb: f64) void {
        const kg = 1.0 - kr - kb;
        y.cr_r = @floatCast(2.0 * (1.0 - kr));
        y.cb_b = @floatCast(2.0 * (1.0 - kb));
        y.cb_g = @floatCast(-2.0 * kb * (1.0 - kb) / kg);
        y.cr_g = @floatCast(-2.0 * kr * (1.0 - kr) / kg);
    }

    inline fn sampleV(y: *const Yuv, base: [*]const u8, x: usize, comptime n: comptime_int, mul: f32, add: f32) @Vector(n, f32) {
        const V = @Vector(n, f32);
        const raw: V = switch (y.sample) {
            .u8 => @floatFromInt(@as(@Vector(n, u8), base[x..][0..n].*)),
            .u16 => @floatFromInt(@as(@Vector(n, u16), @as([*]const u16, @ptrCast(@alignCast(base)))[x..][0..n].*)),
            .f32 => return @as([*]const f32, @ptrCast(@alignCast(base)))[x..][0..n].*,
        };
        return simd.madd(V, raw, @splat(mul), @splat(add));
    }

    inline fn sample1(y: *const Yuv, base: [*]const u8, x: usize, mul: f32, add: f32) f32 {
        return sampleV(y, base, x, 1, mul, add)[0];
    }
};

/// Per-clip state of a YUV input; `frame` makes each frame's `Yuv` from its
/// metadata, with the rules of VapourSynth's resize (R80): the frame's
/// _Matrix over the default (BT.709 above 650 rows, else BT.601, toRGBS's
/// choice), range limited and chroma left-sited when unset.
pub const YuvSetup = struct {
    template: Yuv,
    /// integer sample depth (8-16; 32 for float, unused)
    bits: u8,
    default_matrix: i64,
    htab_left: []Tap = &.{},
    htab_center: []Tap = &.{},

    pub const Sample = @FieldType(Yuv, "sample");

    /// The formats the in-filter conversion takes: 2x or no subsampling per
    /// axis, 8-16 bit int or f32 samples.
    pub fn supports(ssw: u32, ssh: u32, float: bool, bits: u32) bool {
        return ssw <= 1 and ssh <= 1 and (if (float) bits == 32 else bits >= 8 and bits <= 16);
    }

    pub fn init(sample: Sample, bits: u8, w: u32, h: u32, ssw: u1, ssh: u1) error{OutOfMemory}!YuvSetup {
        var s: YuvSetup = .{
            .template = .{
                .sample = sample,
                .ssw = ssw == 1,
                .ssh = ssh == 1,
                .cw = w >> ssw,
                .ch = h >> ssh,
                .cr_r = undefined,
                .cb_g = undefined,
                .cr_g = undefined,
                .cb_b = undefined,
            },
            .bits = bits,
            .default_matrix = if (h > 650) 1 else 6,
        };
        if (ssw == 1) {
            s.htab_left = try allocator.alloc(Tap, w);
            errdefer allocator.free(s.htab_left);
            s.htab_center = try allocator.alloc(Tap, w);
            // chroma k sits at luma 2k (left) or 2k + 0.5 (center)
            buildTaps(s.htab_left, s.template.cw, 0.25);
            buildTaps(s.htab_center, s.template.cw, 0);
        }
        return s;
    }

    pub fn deinit(s: *YuvSetup) void {
        allocator.free(s.htab_left);
        allocator.free(s.htab_center);
        s.htab_left = &.{};
        s.htab_center = &.{};
    }

    /// Whether a frame's _Matrix (null: unset) can be converted.
    pub fn matrixOk(matrix: ?i64) bool {
        const m = matrix orelse return true;
        return m == 2 or Yuv.matrixWeights(m) != null;
    }

    /// The conversion of one frame; null for an unsupported _Matrix.
    /// `loc`: zimg chroma location (0 left, 1 center, 2 top-left, 3 top,
    /// 4 bottom-left, 5 bottom).
    pub fn frame(s: *const YuvSetup, full: bool, matrix: ?i64, loc: ?i64) ?Yuv {
        var y = s.template;
        y.setRange(s.bits, full);
        const m = matrix orelse 2;
        const kw = if (m == 2) Yuv.matrixWeights(s.default_matrix).? else Yuv.matrixWeights(m) orelse return null;
        y.setMatrix(kw[0], kw[1]);
        const l = loc orelse 0;
        y.htab = if (l == 1 or l == 3 or l == 5) s.htab_center else s.htab_left;
        y.vshift = switch (l) {
            2, 3 => 0.25,
            4, 5 => -0.25,
            else => 0,
        };
        return y;
    }

    /// Input view of a YUV frame's planes (per-plane byte strides).
    pub fn input(y: Yuv, planes: [3][*]const u8, strides: [3]usize, is_srgb: bool) Input {
        return .{ .kind = .yuv, .planes = planes, .stride = 0, .strides = strides, .srgb = is_srgb, .yuv = y };
    }
};

/// One output sample's 4 source taps (mirrored at the borders).
pub const Tap = struct { idx: [4]u32, w: [4]f32 };

/// Catmull-Rom (Bicubic b=0, c=0.5).
fn catmullRom(x: f64) f64 {
    const a = @abs(x);
    if (a < 1) return (1.5 * a - 2.5) * a * a + 1;
    if (a < 2) return ((-0.5 * a + 2.5) * a - 4) * a + 2;
    return 0;
}

/// zimg's source position of output sample i when upsampling src_n -> dst_n
/// with the chroma-location shift (in source samples).
fn srcPos(i: u32, src_n: u32, dst_n: u32, shift: f64) f64 {
    return (@as(f64, @floatFromInt(i)) + 0.5) * @as(f64, @floatFromInt(src_n)) / @as(f64, @floatFromInt(dst_n)) - 0.5 + shift;
}

/// zimg's border: mirrored with the edge sample repeated (-1 -> 0, n -> n-1).
fn mirror(j: i64, n: u32) u32 {
    const last: i64 = @as(i64, n) - 1;
    var m = j;
    if (m < 0) m = -m - 1;
    if (m > last) m = 2 * last + 1 - m;
    return @intCast(std.math.clamp(m, 0, last));
}

fn tapAt(pos: f64, src_n: u32) Tap {
    const base: i64 = @as(i64, @intFromFloat(@floor(pos))) - 1;
    var t: Tap = undefined;
    var wd: [4]f64 = undefined;
    var sum: f64 = 0;
    for (0..4) |k| {
        const j = base + @as(i64, @intCast(k));
        wd[k] = catmullRom(pos - @as(f64, @floatFromInt(j)));
        sum += wd[k];
        t.idx[k] = mirror(j, src_n);
    }
    for (0..4) |k| t.w[k] = @floatCast(wd[k] / sum);
    return t;
}

/// Horizontal taps for every output column (len taps = dst_n).
pub fn buildTaps(taps: []Tap, src_n: u32, shift: f64) void {
    const dst_n: u32 = @intCast(taps.len);
    for (taps, 0..) |*t, i| t.* = tapAt(srcPos(@intCast(i), src_n, dst_n, shift), src_n);
}

/// RGB row r (not linearized) of a YUV input into out[0..3]; tmp holds 4 rows.
fn yuvRow(in: *const Input, r: u32, out: [3][]f32, w: u32, h: u32, tmp: [4][]f32) void {
    const y = &in.yuv;
    const FV = @Vector(vec_size, f32);
    // chroma rows at chroma width, normalized then vertically filtered
    const vt: Tap = if (y.ssh) tapAt(srcPos(r, y.ch, h, y.vshift), y.ch) else undefined;
    for (0..2) |c| {
        const dst = tmp[c][0..y.cw];
        const p = c + 1;
        if (y.ssh) {
            var rows: [4][*]const u8 = undefined;
            for (0..4) |k| rows[k] = in.planes[p] + vt.idx[k] * in.strides[p];
            var x: usize = 0;
            while (x + vec_size <= y.cw) : (x += vec_size) {
                var acc: FV = @as(FV, @splat(vt.w[0])) * y.sampleV(rows[0], x, vec_size, y.c_mul, y.c_add);
                inline for (1..4) |k| acc = simd.madd(FV, @splat(vt.w[k]), y.sampleV(rows[k], x, vec_size, y.c_mul, y.c_add), acc);
                dst[x..][0..vec_size].* = acc;
            }
            while (x < y.cw) : (x += 1) {
                var acc: f32 = vt.w[0] * y.sample1(rows[0], x, y.c_mul, y.c_add);
                inline for (1..4) |k| acc = simd.madd(f32, vt.w[k], y.sample1(rows[k], x, y.c_mul, y.c_add), acc);
                dst[x] = acc;
            }
        } else {
            const row = in.planes[p] + r * in.strides[p];
            var x: usize = 0;
            while (x + vec_size <= y.cw) : (x += vec_size) dst[x..][0..vec_size].* = y.sampleV(row, x, vec_size, y.c_mul, y.c_add);
            while (x < y.cw) : (x += 1) dst[x] = y.sample1(row, x, y.c_mul, y.c_add);
        }
    }
    var u: []const f32 = tmp[0];
    var v: []const f32 = tmp[1];
    if (y.ssw) {
        // only the first cw samples are written: the slice length is the
        // chroma width upsampleRow bounds its vector interior by
        upsampleRow(tmp[0][0..y.cw], tmp[2], y.htab, w);
        upsampleRow(tmp[1][0..y.cw], tmp[3], y.htab, w);
        u = tmp[2];
        v = tmp[3];
    }
    const yrow = in.planes[0] + r * in.strides[0];
    var x: usize = 0;
    while (x + vec_size <= w) : (x += vec_size) {
        const yv = y.sampleV(yrow, x, vec_size, y.y_mul, y.y_add);
        const uv: FV = u[x..][0..vec_size].*;
        const vv: FV = v[x..][0..vec_size].*;
        out[0][x..][0..vec_size].* = simd.madd(FV, @splat(y.cr_r), vv, yv);
        out[1][x..][0..vec_size].* = simd.madd(FV, @splat(y.cr_g), vv, simd.madd(FV, @splat(y.cb_g), uv, yv));
        out[2][x..][0..vec_size].* = simd.madd(FV, @splat(y.cb_b), uv, yv);
    }
    while (x < w) : (x += 1) {
        const yv = y.sample1(yrow, x, y.y_mul, y.y_add);
        out[0][x] = simd.madd(f32, y.cr_r, v[x], yv);
        out[1][x] = simd.madd(f32, y.cr_g, v[x], simd.madd(f32, y.cb_g, u[x], yv));
        out[2][x] = simd.madd(f32, y.cb_b, u[x], yv);
    }
}

/// 2x horizontal upsampling through the tap table. Interior output pairs
/// (2k, 2k+1) have fixed phases, so they run as two vector filters on shifted
/// loads, interleaved; columns whose taps were mirrored use the table. Both
/// paths take the same weights in the same order. `src` is exactly the
/// chroma row (its length is the chroma width); `tab` has w entries.
fn upsampleRow(src: []const f32, dst: []f32, tab: []const Tap, w: u32) void {
    const FV = @Vector(vec_size, f32);
    const n = vec_size;
    // the pair at k = 2 is past any left mirroring for both phases
    const k0: u32 = 2;
    // narrower rows have no pair at k0 to take the phases from
    if (w < 2 * k0 + 2) {
        for (dst[0..w], tab[0..w]) |*d, t| d.* = tapRow(src, t);
        return;
    }
    const e = tab[2 * k0];
    const o = tab[2 * k0 + 1];
    const oe: i64 = @as(i64, e.idx[0]) - k0;
    const oo: i64 = @as(i64, o.idx[0]) - k0;
    const cw: u32 = @intCast(src.len);
    // interior pairs: every tap of both phases inside [0, cw)
    const hi_tap: i64 = @max(oe, oo) + 3;
    const k_end: i64 = @min(@as(i64, cw) - 1 - hi_tap + 1, @as(i64, w / 2));
    for (0..2 * k0) |x| dst[x] = tapRow(src, tab[x]);
    var k: i64 = k0;
    while (k + n <= k_end) : (k += n) {
        const ke: usize = @intCast(k + oe);
        const ko: usize = @intCast(k + oo);
        var ev: FV = @as(FV, @splat(e.w[0])) * @as(FV, src[ke..][0..n].*);
        var od: FV = @as(FV, @splat(o.w[0])) * @as(FV, src[ko..][0..n].*);
        inline for (1..4) |j| {
            ev = simd.madd(FV, @splat(e.w[j]), src[ke + j ..][0..n].*, ev);
            od = simd.madd(FV, @splat(o.w[j]), src[ko + j ..][0..n].*, od);
        }
        const lo = comptime blk: {
            var m: [n]i32 = undefined;
            for (0..n / 2) |i| {
                m[2 * i] = i;
                m[2 * i + 1] = ~@as(i32, i);
            }
            break :blk m;
        };
        const hi = comptime blk: {
            var m: [n]i32 = undefined;
            for (0..n / 2) |i| {
                m[2 * i] = n / 2 + i;
                m[2 * i + 1] = ~@as(i32, n / 2 + i);
            }
            break :blk m;
        };
        const xo: usize = @intCast(2 * k);
        dst[xo..][0..n].* = @shuffle(f32, ev, od, lo);
        dst[xo + n ..][0..n].* = @shuffle(f32, ev, od, hi);
    }
    var x: u32 = @intCast(2 * k);
    while (x < w) : (x += 1) dst[x] = tapRow(src, tab[x]);
}

inline fn tapRow(src: []const f32, t: Tap) f32 {
    var acc: f32 = t.w[0] * src[t.idx[0]];
    inline for (1..4) |j| acc = simd.madd(f32, t.w[j], src[t.idx[j]], acc);
    return acc;
}

// Encoded-value domain before the sRGB decode: [0, 1], what libjxl sees.
// Its input is always PNG samples, so out-of-range float values (e.g. YUV ->
// RGB overshoot) are clamped exactly as saving the clip to PNG would; left
// unclamped, the odd extension (negative light) and the polynomial outside
// its fit range derail the score by up to hundreds of points. In range this
// is libjxl's result unchanged.
const in_lo: f32 = 0.0;
const in_hi: f32 = 1.0;

inline fn decode(v: anytype) @TypeOf(v) {
    const V = @TypeOf(v);
    return srgb.toLinear(@min(@max(v, @as(V, @splat(in_lo))), @as(V, @splat(in_hi))));
}

fn linearizeRow(src: []const f32, dst: []f32) void {
    const FV = @Vector(vec_size, f32);
    var x: usize = 0;
    while (x + vec_size <= src.len) : (x += vec_size) {
        const v: FV = src[x..][0..vec_size].*;
        dst[x..][0..vec_size].* = decode(v);
    }
    while (x < src.len) : (x += 1) {
        dst[x] = decode(@as(@Vector(1, f32), @splat(src[x])))[0];
    }
}

/// Integer sample -> linear light table: v * (1 / peak) in f32, as libjxl
/// loads 8/16-bit samples, then the sRGB decode unless `is_srgb` is false.
/// Sized for the sample type (256 / 65536) so out-of-range samples above
/// the nominal peak stay in bounds.
pub fn buildLut(lut: []f32, bits: u5, is_srgb: bool) void {
    const scale: f32 = @as(f32, 1.0) / @as(f32, @floatFromInt((@as(u32, 1) << bits) - 1));
    for (lut, 0..) |*o, i| {
        const x = @as(f32, @floatFromInt(i)) * scale;
        o.* = if (is_srgb) decode(@as(@Vector(1, f32), @splat(x)))[0] else x;
    }
}

/// Row buffers of the fused per-scale sweep (sweepScale), carved from the
/// tail of the scratch block. Rows are w + 64 long: blurH's input row is read
/// past w by its vector body.
const RowScratch = struct {
    // Rings keyed by source row % ring_n: an output row's 9-tap window spans
    // at most 9 consecutive source rows.
    const ring_n = 10;
    // tmp/mu1/mu2/s12/d2 + 4 YUV-conversion rows + zero row + linear-RGB and
    // XYB rings (2 images x 3 planes each) + product rings (2 x 3 planes)
    const rows = 10 + 18 * ring_n;
    tmp: []f32,
    /// all zeros: the vertical taps outside the plane read it
    zero: []f32,
    yuv: [4][]f32,
    mu1: []f32,
    mu2: []f32,
    s12: []f32,
    d2: []f32, // blur((a-b)^2)
    lin: [2][3][]f32,
    xyb: [2][3][]f32,
    p12: [3][]f32, // a*b
    pd2: [3][]f32, // (a-b)^2
    rl: usize,

    fn init(buf: []f32, w: u32) RowScratch {
        const rl: usize = @as(usize, w) + 64;
        const ring = ring_n * rl;
        var rs: RowScratch = undefined;
        rs.rl = rl;
        rs.tmp = buf[0..rl];
        rs.mu1 = buf[rl..][0..rl];
        rs.mu2 = buf[2 * rl ..][0..rl];
        rs.s12 = buf[3 * rl ..][0..rl];
        rs.d2 = buf[4 * rl ..][0..rl];
        for (0..4) |k| rs.yuv[k] = buf[(5 + k) * rl ..][0..rl];
        rs.zero = buf[9 * rl ..][0..rl];
        @memset(rs.zero, 0.0);
        var off: usize = 10 * rl;
        for (0..2) |img| {
            for (0..3) |p| {
                rs.lin[img][p] = buf[off..][0..ring];
                rs.xyb[img][p] = buf[off + ring ..][0..ring];
                off += 2 * ring;
            }
        }
        for (0..3) |p| {
            rs.p12[p] = buf[off..][0..ring];
            rs.pd2[p] = buf[off + ring ..][0..ring];
            off += 2 * ring;
        }
        return rs;
    }

    inline fn at(self: RowScratch, ring: []f32, r: u32) []f32 {
        return ring[(r % ring_n) * self.rl ..][0..self.rl];
    }

    /// A ring row as a blur tap: the zero row for `no_row`.
    inline fn tap(self: RowScratch, ring: []f32, r: u32) []const f32 {
        return if (r == no_row) self.zero else self.at(ring, r);
    }
};

/// Float row stride of the (virtual) full-resolution plane; only fixes the
/// strides of the half-resolution chain buffers, never the result.
fn baseStride(w: u32) u32 {
    return std.mem.alignForward(u32, w, 16);
}

/// Scratch buffer size (in f32 elements) needed by `process` for a given plane geometry.
pub fn scratchSize(w: u32, h: u32) usize {
    const stride = baseStride(w);
    const b_size: usize = @as(usize, (stride + 1) / 2) * ((h + 1) / 2);
    return 6 * b_size + RowScratch.rows * (@as(usize, w) + 64);
}

pub fn process(in1: Input, in2: Input, w: u32, h: u32, scratch: []f32) f64 {
    const stride = baseStride(w);
    // The downscale chain buffers only ever hold scale >= 1 (half resolution).
    const b_size: u32 = ((stride + 1) / 2) * ((h + 1) / 2);

    const srcp1b = [3][]f32{ scratch[0 * b_size .. 1 * b_size], scratch[1 * b_size .. 2 * b_size], scratch[2 * b_size .. 3 * b_size] };
    const srcp2b = [3][]f32{ scratch[3 * b_size .. 4 * b_size], scratch[4 * b_size .. 5 * b_size], scratch[5 * b_size .. 6 * b_size] };
    const rs = RowScratch.init(scratch[6 * b_size ..], w);

    var plane_avg_ssim: [6][6]f64 = undefined;
    var plane_avg_edge: [6][12]f64 = undefined;
    var stride2 = stride;
    var w2 = w;
    var h2 = h;

    const n_scales = numScales(w, h);
    var scale: u32 = 0;
    while (scale < n_scales) : (scale += 1) {
        if (scale > 0) {
            // scale 1 was downscaled by the scale-0 sweep, straight from the
            // linearized input rows
            if (scale > 1) {
                downscale(srcp1b, srcp1b, stride2, w2, h2);
                downscale(srcp2b, srcp2b, stride2, w2, h2);
            }
            stride2 = @divTrunc((stride2 + 1), 2);
            w2 = @divTrunc((w2 + 1), 2);
            h2 = @divTrunc((h2 + 1), 2);
        }

        const one_per_pixels: f64 = 1.0 / @as(f64, @floatFromInt(w2 * h2));
        // skip_table follows the 6-scale weight layout; with fewer scales the
        // weights shift (see score), so every sub-score is computed
        var active: [3]Active = @splat(.{ .ssim = true, .edge = true });
        if (n_scales == 6) for (&active, 0..) |*a, plane| {
            const skip = skip_table[plane][scale];
            a.* = .{ .ssim = !skip.ssim, .edge = !skip.artifact or !skip.detailloss };
        };

        var ssim: [3]SsimAcc = @splat(.{});
        var edge: [3]EdgeAcc = @splat(.{});
        const in_b1 = [3][]const f32{ srcp1b[0], srcp1b[1], srcp1b[2] };
        const in_b2 = [3][]const f32{ srcp2b[0], srcp2b[1], srcp2b[2] };
        if (scale == 0) {
            const s0: Scale0 = .{ .in = .{ in1, in2 }, .ds = .{ srcp1b, srcp2b }, .ds_stride = (stride + 1) / 2 };
            sweepScale(s0, in_b1, in_b2, stride2, w2, h2, active, rs, &ssim, &edge);
        } else {
            sweepScale(null, in_b1, in_b2, stride2, w2, h2, active, rs, &ssim, &edge);
        }

        for (0..3) |plane| {
            if (active[plane].ssim) {
                ssim[plane].finish(one_per_pixels, plane_avg_ssim[scale][plane * 2 ..][0..2]);
            } else {
                plane_avg_ssim[scale][plane * 2] = 0.0;
                plane_avg_ssim[scale][plane * 2 + 1] = 0.0;
            }

            if (active[plane].edge) {
                edge[plane].finish(one_per_pixels, plane_avg_edge[scale][plane * 4 ..][0..4]);
            } else {
                plane_avg_edge[scale][plane * 4 + 0] = 0.0;
                plane_avg_edge[scale][plane * 4 + 1] = 0.0;
                plane_avg_edge[scale][plane * 4 + 2] = 0.0;
                plane_avg_edge[scale][plane * 4 + 3] = 0.0;
            }
        }
    }

    return score(plane_avg_ssim, plane_avg_edge, n_scales);
}

/// Scales libjxl computes: it stops once the previous scale is under 8 px in
/// either dimension (scale 0 always runs).
fn numScales(w: u32, h: u32) u32 {
    var n: u32 = 1;
    var w2 = w;
    var h2 = h;
    while (n < 6 and w2 >= 8 and h2 >= 8) : (n += 1) {
        w2 = (w2 + 1) / 2;
        h2 = (h2 + 1) / 2;
    }
    return n;
}

fn downscale(src: [3][]const f32, dst: [3][]f32, src_stride: u32, in_w: u32, in_h: u32) void {
    const out_h = @divTrunc((in_h + 1), 2);
    const dst_stride = @divTrunc((src_stride + 1), 2);
    for (0..3) |plane| {
        var oy: u32 = 0;
        while (oy < out_h) : (oy += 1) {
            const row0 = src[plane][(oy * 2) * src_stride ..];
            const row1: ?[]const f32 = if (oy * 2 + 1 < in_h) src[plane][(oy * 2 + 1) * src_stride ..] else null;
            downscaleRow(row0, row1, dst[plane][oy * dst_stride ..], in_w);
        }
    }
}

/// One output row of the 2x2 box downscale. `row1` is null on the last row of
/// an odd height, where the window's lower row clamps to `row0`.
fn downscaleRow(row0: []const f32, row1: ?[]const f32, dstp: []f32, in_w: u32) void {
    const out_w = @divTrunc((in_w + 1), 2);
    const normalize: f32 = 1.0 / 4.0;
    const FV = @Vector(vec_size, f32);
    const normv: FV = @splat(normalize);
    // ox values whose 2x2 window needs no edge clamping
    const full_w: u32 = in_w / 2;

    var ox: u32 = 0;
    if (row1) |r1| {
        while (ox + vec_size <= full_w) : (ox += vec_size) {
            const a0: FV = row0[ox * 2 ..][0..vec_size].*;
            const b0: FV = row0[ox * 2 + vec_size ..][0..vec_size].*;
            const a1: FV = r1[ox * 2 ..][0..vec_size].*;
            const b1: FV = r1[ox * 2 + vec_size ..][0..vec_size].*;
            // asm deinterleave: the @shuffle form makes LLVM shred
            // each plain load into partial loads + blends, rebuilt per
            // shuffle (measured ~55 instr/block vs ~14 at 8 lanes, and
            // again at 16 lanes on znver4).
            const e0 = simd.evenLanesF32(vec_size, a0, b0);
            const o0 = simd.oddLanesF32(vec_size, a0, b0);
            const e1 = simd.evenLanesF32(vec_size, a1, b1);
            const o1 = simd.oddLanesF32(vec_size, a1, b1);
            // same add order as the scalar 2x2 loop
            const sum = ((e0 + o0) + e1) + o1;
            dstp[ox..][0..vec_size].* = sum * normv;
        }
    }
    const r1 = row1 orelse row0;
    while (ox < out_w) : (ox += 1) {
        var sum: f32 = 0.0;
        inline for (0..2) |iy| {
            const r = if (iy == 0) row0 else r1;
            inline for (0..2) |ix| {
                const x: u32 = @min(ox * 2 + ix, in_w - 1);
                sum += r[x];
            }
        }
        dstp[ox] = sum * normalize;
    }
}

fn multiplyRow(src1: []const f32, src2: []const f32, dst: []f32, w: u32) void {
    const Vec = @Vector(vec_size, f32);
    var x: u32 = 0;
    while (x + vec_size <= w) : (x += vec_size) {
        const a: Vec = src1[x..][0..vec_size].*;
        const b: Vec = src2[x..][0..vec_size].*;
        dst[x..][0..vec_size].* = a * b;
    }
    while (x < w) : (x += 1) {
        dst[x] = src1[x] * src2[x];
    }
}

fn subSquareRow(src1: []const f32, src2: []const f32, dst: []f32, w: u32) void {
    const Vec = @Vector(vec_size, f32);
    var x: u32 = 0;
    while (x + vec_size <= w) : (x += vec_size) {
        const a: Vec = src1[x..][0..vec_size].*;
        const b: Vec = src2[x..][0..vec_size].*;
        const v = a - b;
        dst[x..][0..vec_size].* = v * v;
    }
    while (x < w) : (x += 1) {
        const v = src1[x] - src2[x];
        dst[x] = v * v;
    }
}

/// One edge output of the horizontal blur: taps outside [0, w) are zero.
inline fn blurEdge(srcp: []const f32, kernel: [ksize]f32, j: i32, w: i32) f32 {
    var sum: f32 = 0.0;
    var k: i32 = 0;
    while (k < ksize) : (k += 1) {
        const idx = j - radius + k;
        if (idx >= 0 and idx < w) sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
    }
    return sum;
}

inline fn blurH(srcp: []f32, dstp: []f32, kernel: [ksize]f32, w: i32) void {
    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) dstp[@intCast(j)] = blurEdge(srcp, kernel, j, w);

    j = radius;
    const center_end: i32 = w - @min(w, radius);
    // srcp is the padded blur row buffer, so vector loads past w are safe.
    // Hoist each tap's base pointer out of the j-loop and launder it once. The
    // ksize taps are overlapping comptime-offset loads that LLVM would fuse
    // into shuffle chains (~2x instructions, shuffle-port-bound); making the
    // bases mutually opaque blocks that fusion. Doing it once here (rather than
    // laundering every load inside the loop, as loaduOpaque did) lets each load
    // fold into its vmulps as [tap_k + j*4] instead of re-materializing
    // tap_k + (j-radius) per tap per iteration (removes 9 lea + 8 add/iter).
    var taps: [ksize][*]const f32 = undefined;
    inline for (0..ksize) |k| {
        taps[k] = simd.opaquePtr(f32, srcp.ptr + k);
    }
    while (j + vec_size <= center_end) : (j += vec_size) {
        const base: u32 = @intCast(j - radius);
        var acc: @Vector(vec_size, f32) = @splat(0.0);
        inline for (0..ksize) |k| {
            const kv: @Vector(vec_size, f32) = @splat(kernel[k]);
            const sv: @Vector(vec_size, f32) = (taps[k] + base)[0..vec_size].*;
            acc = acc + kv * sv;
        }
        dstp[@intCast(j)..][0..vec_size].* = acc;
    }
    while (j < center_end) : (j += 1) {
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += kernel[@intCast(k)] * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) dstp[@intCast(j)] = blurEdge(srcp, kernel, j, w);
}

inline fn blurV(src: anytype, dstp: []f32, kernel: [ksize]f32, w: u32) void {
    const Vec = @Vector(vec_size, f32);
    var j: u32 = 0;
    while (j + vec_size <= w) : (j += vec_size) {
        var accum: Vec = @splat(0.0);
        inline for (0..ksize) |k| {
            const kv: Vec = @splat(kernel[k]);
            const sv: Vec = src[k][j..][0..vec_size].*;
            accum = simd.madd(Vec, kv, sv, accum);
        }
        dstp[j..][0..vec_size].* = accum;
    }
    while (j < w) : (j += 1) {
        var accum: f32 = 0.0;
        var k: u32 = 0;
        while (k < ksize) : (k += 1) {
            accum += kernel[k] * src[k][j];
        }
        dstp[j] = accum;
    }
}

/// libjxl's blur as a 9-tap FIR. libjxl runs CreateRecursiveGaussian(1.5)
/// (Charalampidis 2016: three truncated-cosine recursions, radius 5) with
/// zero padding; its impulse response is exactly these 9 taps (the
/// recursion matches this kernel to 4e-15 in float64), so the FIR gives
/// libjxl's result at the cost of the plain Gaussian kernel used before
/// (sampled sigma 1.5, edges mirrored), which put scores up to ~8 points
/// away from libjxl's on 640x360 and ~5 on 1080p. Rows and columns outside
/// the plane count as zero, as in libjxl.
const blur_kernel = [ksize]f32{
    0.009414367377758026123046875,
    0.036011114716529846191406250,
    0.109335370361804962158203125,
    0.212928593158721923828125000,
    0.264621108770370483398437500,
    0.212928593158721923828125000,
    0.109335370361804962158203125,
    0.036011114716529846191406250,
    0.009414367377758026123046875,
};

/// Marks a vertical tap outside the plane (zero padded, see blur_kernel).
const no_row = std.math.maxInt(u32);

/// Source rows of the 9-tap vertical window for output row i; `no_row`
/// outside [0, ih).
fn blurRows(i: i32, ih: i32) [ksize]u32 {
    var rows: [ksize]u32 = undefined;
    for (&rows, 0..) |*r, k| {
        const row = i - radius + @as(i32, @intCast(k));
        r.* = if (row >= 0 and row < ih) @intCast(row) else no_row;
    }
    return rows;
}

/// One output row of the separable 9x9 blur: vertical taps into `tmp`, then
/// the horizontal pass into `dst`.
inline fn blurRow(srcp: [ksize][]const f32, dst: []f32, tmp: []f32, w: u32) void {
    blurV(srcp, tmp, blur_kernel, w);
    blurH(tmp, dst, blur_kernel, @intCast(w));
}

const Active = struct { ssim: bool, edge: bool };

/// Fused per-scale sweep. Per output row: the XYB rows of the 9-tap window
/// are made on demand from the linear RGB input (toXYBRow, once per source
/// row, all three planes of both images), then per plane the four blurs
/// (mu1, mu2, blur(a*b), blur((a-b)^2)) and the ssim/edge reductions run on a
/// few cache-resident rows. The plane-at-a-time form wrote 6 full XYB planes
/// plus ~20 full-plane f32 passes per scale and went DRAM-bound with every
/// frame worker active. Bit-exact against it: every row is the same ops on
/// the same rows, and each plane's accumulators see rows top to bottom.
/// Scale 0 of sweepScale: rows come from the native-format inputs, are
/// linearized once into the `lin` rings, and the 2x2 downscale feeding scale 1
/// is emitted from those same rows (so the full-resolution linear image never
/// exists as a plane).
const Scale0 = struct {
    in: [2]Input,
    ds: [2][3][]f32,
    ds_stride: u32,
};

fn sweepScale(s0: ?Scale0, in1: [3][]const f32, in2: [3][]const f32, stride: u32, w: u32, h: u32, active: [3]Active, rs: RowScratch, ssim: *[3]SsimAcc, edge: *[3]EdgeAcc) void {
    const n = RowScratch.ring_n;
    const any_active = active[0].ssim or active[0].edge or active[1].ssim or active[1].edge or active[2].ssim or active[2].edge;
    if (s0 == null and !any_active) return;
    var next_ds: u32 = 0; // scale 0: next downscaled row to emit
    const ds_h = (h + 1) / 2;
    var tag_xyb: [n]u32 = @splat(std.math.maxInt(u32));
    var tag_prod: [3][n]u32 = @splat(@splat(std.math.maxInt(u32)));
    const ih: i32 = @intCast(h);
    var i: i32 = 0;
    while (i < ih) : (i += 1) {
        const rows = blurRows(i, ih);
        inline for (0..ksize) |k| {
            const r = rows[k];
            if (r != no_row and tag_xyb[r % n] != r) {
                if (s0) |z| {
                    inline for (0..2) |img| {
                        var lr: [3][]f32 = undefined;
                        inline for (0..3) |c| lr[c] = rs.at(rs.lin[img][c], r);
                        z.in[img].rows(r, lr, w, h, rs.yuv);
                        toXYBRow(.{ lr[0], lr[1], lr[2] }, .{ rs.at(rs.xyb[img][0], r), rs.at(rs.xyb[img][1], r), rs.at(rs.xyb[img][2], r) }, w);
                    }
                } else {
                    const off = r * stride;
                    toXYBRow(.{ in1[0][off..], in1[1][off..], in1[2][off..] }, .{ rs.at(rs.xyb[0][0], r), rs.at(rs.xyb[0][1], r), rs.at(rs.xyb[0][2], r) }, w);
                    toXYBRow(.{ in2[0][off..], in2[1][off..], in2[2][off..] }, .{ rs.at(rs.xyb[1][0], r), rs.at(rs.xyb[1][1], r), rs.at(rs.xyb[1][2], r) }, w);
                }
                tag_xyb[r % n] = r;
            }
        }
        const ui: u32 = @intCast(i);

        if (s0) |z| {
            // rows <= min(i + radius, h - 1) are now linearized and still in
            // the ring; emit every downscaled row whose 2x2 window is ready
            const ready = @min(ui + radius, h - 1);
            while (next_ds < ds_h and @min(2 * next_ds + 1, h - 1) <= ready) : (next_ds += 1) {
                const y0 = 2 * next_ds;
                const has1 = y0 + 1 < h;
                inline for (0..2) |img| {
                    inline for (0..3) |c| {
                        const ring = rs.lin[img][c];
                        downscaleRow(rs.at(ring, y0), if (has1) rs.at(ring, y0 + 1) else null, z.ds[img][c][next_ds * z.ds_stride ..], w);
                    }
                }
            }
            if (!any_active) continue;
        }

        for (0..3) |p| {
            const act = active[p];
            if (!(act.ssim or act.edge)) continue;
            var r1: [ksize][]const f32 = undefined;
            var r2: [ksize][]const f32 = undefined;
            inline for (0..ksize) |k| {
                r1[k] = rs.tap(rs.xyb[0][p], rows[k]);
                r2[k] = rs.tap(rs.xyb[1][p], rows[k]);
            }
            blurRow(r1, rs.mu1, rs.tmp, w);
            blurRow(r2, rs.mu2, rs.tmp, w);

            if (act.ssim) {
                var q12: [ksize][]const f32 = undefined;
                var qd2: [ksize][]const f32 = undefined;
                inline for (0..ksize) |k| {
                    const r = rows[k];
                    if (r == no_row) {
                        q12[k] = rs.zero;
                        qd2[k] = rs.zero;
                    } else {
                        const p12 = rs.at(rs.p12[p], r);
                        const pd2 = rs.at(rs.pd2[p], r);
                        if (tag_prod[p][r % n] != r) {
                            multiplyRow(r1[k], r2[k], p12, w);
                            subSquareRow(r1[k], r2[k], pd2, w);
                            tag_prod[p][r % n] = r;
                        }
                        q12[k] = p12;
                        qd2[k] = pd2;
                    }
                }
                blurRow(q12, rs.s12, rs.tmp, w);
                blurRow(qd2, rs.d2, rs.tmp, w);
                ssim[p].row(rs.d2, rs.s12, rs.mu1, rs.mu2, w);
            }

            if (act.edge) edge[p].row(rs.at(rs.xyb[0][p], ui), rs.at(rs.xyb[1][p], ui), rs.mu1, rs.mu2, w);
        }
    }
}

const K_D0: f32 = 0.0037930734;
const K_D1: f32 = math.cbrt(K_D0);

const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;

const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;

const K_M20: f32 = 0.24342269;
const K_M21: f32 = 0.20476745;
const K_M22: f32 = 1.0 - K_M20 - K_M21;

const OPSIN_ABSORBANCE_MATRIX = [9]f32{ K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22 };
const OPSIN_ABSORBANCE_BIAS: f32 = K_D0;

/// One row of the linear RGB -> XYB transform (all three planes).
fn toXYBRow(srcp: [3][]const f32, dstp: [3][]f32, w: u32) void {
    const FV = @Vector(vec_size, f32);
    const zero: FV = @splat(0.0);
    const bias: FV = @splat(OPSIN_ABSORBANCE_BIAS);
    const kd1: FV = @splat(K_D1);
    const half: FV = @splat(0.5);
    const c14: FV = @splat(14.0);
    const c042: FV = @splat(0.42);
    const c055: FV = @splat(0.55);
    const c001: FV = @splat(0.01);
    const m00: FV = @splat(OPSIN_ABSORBANCE_MATRIX[0]);
    const m01: FV = @splat(OPSIN_ABSORBANCE_MATRIX[1]);
    const m02: FV = @splat(OPSIN_ABSORBANCE_MATRIX[2]);
    const m10: FV = @splat(OPSIN_ABSORBANCE_MATRIX[3]);
    const m11: FV = @splat(OPSIN_ABSORBANCE_MATRIX[4]);
    const m12: FV = @splat(OPSIN_ABSORBANCE_MATRIX[5]);
    const m20: FV = @splat(OPSIN_ABSORBANCE_MATRIX[6]);
    const m21: FV = @splat(OPSIN_ABSORBANCE_MATRIX[7]);
    const m22: FV = @splat(OPSIN_ABSORBANCE_MATRIX[8]);

    {
        if (w >= vec_size) {
            // the final vector backsteps to w - vec_size (overlapping recompute)
            // so every pixel takes the identical SIMD path
            var x: u32 = 0;
            while (true) {
                if (x + vec_size > w) x = w - vec_size;

                const r: FV = srcp[0][x..][0..vec_size].*;
                const g: FV = srcp[1][x..][0..vec_size].*;
                const b: FV = srcp[2][x..][0..vec_size].*;

                const ox = simd.madd(FV, m00, r, simd.madd(FV, m01, g, simd.madd(FV, m02, b, bias)));
                const oy = simd.madd(FV, m10, r, simd.madd(FV, m11, g, simd.madd(FV, m12, b, bias)));
                const oz = simd.madd(FV, m20, r, simd.madd(FV, m21, g, simd.madd(FV, m22, b, bias)));
                const cx = vcl.cbrt(@max(ox, zero)) - kd1;
                const cy = vcl.cbrt(@max(oy, zero)) - kd1;
                const cz = vcl.cbrt(@max(oz, zero)) - kd1;

                const xv = half * (cx - cy);
                const yv = half * (cx + cy);
                dstp[0][x..][0..vec_size].* = xv * c14 + c042;
                dstp[1][x..][0..vec_size].* = yv + c001;
                dstp[2][x..][0..vec_size].* = (cz - yv) + c055;

                if (x + vec_size >= w) break;
                x += vec_size;
            }
        } else {
            // degenerate width: single-lane vectors keep cbrt rounding identical
            var x: u32 = 0;
            while (x < w) : (x += 1) {
                const F1 = @Vector(1, f32);
                const r: F1 = @splat(srcp[0][x]);
                const g: F1 = @splat(srcp[1][x]);
                const b: F1 = @splat(srcp[2][x]);
                const b1: F1 = @splat(OPSIN_ABSORBANCE_BIAS);
                const ox = simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[0]), r, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[1]), g, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[2]), b, b1)));
                const oy = simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[3]), r, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[4]), g, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[5]), b, b1)));
                const oz = simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[6]), r, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[7]), g, simd.madd(F1, @splat(OPSIN_ABSORBANCE_MATRIX[8]), b, b1)));
                const cx = vcl.cbrt(@max(ox, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const cy = vcl.cbrt(@max(oy, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const cz = vcl.cbrt(@max(oz, @as(F1, @splat(0.0)))) - @as(F1, @splat(K_D1));
                const xs = 0.5 * (cx[0] - cy[0]);
                const ys = 0.5 * (cx[0] + cy[0]);
                dstp[0][x] = xs * 14.0 + 0.42;
                dstp[1][x] = ys + 0.01;
                dstp[2][x] = (cz[0] - ys) + 0.55;
            }
        }
    }
}

fn tothe4th(y: f64) f64 {
    var x = y * y;
    x *= x;
    return x;
}

const AccFV = @Vector(vec_size, f32);
const AccDV = @Vector(vec_size, f64);

/// SSIM-map accumulators, fed one row at a time (rows top to bottom; the
/// per-lane vector sums and the scalar-tail sums are reduced only in finish).
///
/// The denominator sigma1^2 + sigma2^2 + C2 is formed as
/// blur((a-b)^2) - (mu1-mu2)^2 + num_s, since sigma1^2 + sigma2^2 =
/// [blur((a-b)^2) - (mu1-mu2)^2] + 2 sigma12. libjxl's blur(a^2) - mu1^2 +
/// blur(b^2) - mu2^2 cancels terms near 0.7 down to variances near 1e-6 in
/// flat areas, and the former blur((a+b)^2) form terms near 3; their f32
/// noise, clipped by max(d, 0), became a positive bias (about 1 point on a
/// real 1080p frame, 7 on a flat one). Here only the covariance cancels, as
/// in libjxl; scores land within ~0.003 of a float64 evaluation on real
/// frames.
const SsimAcc = struct {
    sum0v: AccDV = @splat(0.0),
    sum1v: AccDV = @splat(0.0),
    sum1: [2]f64 = .{ 0.0, 0.0 },

    /// `d2p`: blur((a-b)^2), `s12p`: blur(a*b).
    fn row(self: *SsimAcc, d2p: []const f32, s12p: []const f32, mu1p: []const f32, mu2p: []const f32, w: u32) void {
        const onef: AccFV = @splat(1.0);
        const twof: AccFV = @splat(2.0);
        const e9f: AccFV = @splat(0.0009);
        const oned: AccDV = @splat(1.0);
        const zerod: AccDV = @splat(0.0);

        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const m1: AccFV = mu1p[x..][0..vec_size].*;
            const m2: AccFV = mu2p[x..][0..vec_size].*;
            const s12v: AccFV = s12p[x..][0..vec_size].*;
            const d2v: AccFV = d2p[x..][0..vec_size].*;
            const m12 = m1 * m2;
            const m_diff = m1 - m2;
            const num_m: AccDV = @floatCast(simd.madd(AccFV, m_diff, -m_diff, onef));
            const num_s32 = simd.madd(AccFV, (s12v - m12), twof, e9f);
            const num_s: AccDV = @floatCast(num_s32);
            const denom_s: AccDV = @floatCast((d2v - m_diff * m_diff) + num_s32);
            const d1: AccDV = @max(oned - ((num_m * num_s) / denom_s), zerod);

            self.sum0v += d1;
            var t4 = d1 * d1;
            t4 *= t4;
            self.sum1v += t4;
        }

        while (x < w) : (x += 1) {
            const m1: f32 = mu1p[x];
            const m2: f32 = mu2p[x];
            const m12 = m1 * m2;
            const m_diff = m1 - m2;
            const num_m: f64 = simd.madd(f32, m_diff, -m_diff, 1.0);
            const num_s32 = simd.madd(f32, (s12p[x] - m12), 2.0, 0.0009);
            const num_s: f64 = num_s32;
            const denom_s: f64 = (d2p[x] - m_diff * m_diff) + num_s32;
            const d1: f64 = @max(1.0 - ((num_m * num_s) / denom_s), 0.0);

            self.sum1[0] += d1;
            self.sum1[1] += tothe4th(d1);
        }
    }

    fn finish(self: *SsimAcc, one_per_pixels: f64, out: *[2]f64) void {
        self.sum1[0] += @reduce(.Add, self.sum0v);
        self.sum1[1] += @reduce(.Add, self.sum1v);
        out[0] = one_per_pixels * self.sum1[0];
        out[1] = @sqrt(@sqrt(one_per_pixels * self.sum1[1]));
    }
};

/// Edge-map accumulators (artifact / detail-loss), same row-feeding scheme.
const EdgeAcc = struct {
    art0: AccDV = @splat(0.0),
    art1: AccDV = @splat(0.0),
    det0: AccDV = @splat(0.0),
    det1: AccDV = @splat(0.0),
    sum2: [4]f64 = .{ 0.0, 0.0, 0.0, 0.0 },

    fn row(self: *EdgeAcc, im1p: []const f32, im2p: []const f32, mu1p: []const f32, mu2p: []const f32, w: u32) void {
        const oned: AccDV = @splat(1.0);
        const zerod: AccDV = @splat(0.0);

        var x: u32 = 0;
        while (x + vec_size <= w) : (x += vec_size) {
            const v1: AccFV = im1p[x..][0..vec_size].*;
            const v2: AccFV = im2p[x..][0..vec_size].*;
            const m1: AccFV = mu1p[x..][0..vec_size].*;
            const m2: AccFV = mu2p[x..][0..vec_size].*;
            const n2: AccDV = @floatCast(@abs(v2 - m2));
            const n1: AccDV = @floatCast(@abs(v1 - m1));
            const d1: AccDV = (oned + n2) / (oned + n1) - oned;

            const artifact = @max(d1, zerod);
            self.art0 += artifact;
            var a4 = artifact * artifact;
            a4 *= a4;
            self.art1 += a4;

            const detail_lost = @max(-d1, zerod);
            self.det0 += detail_lost;
            var l4 = detail_lost * detail_lost;
            l4 *= l4;
            self.det1 += l4;
        }

        while (x < w) : (x += 1) {
            const d1: f64 = (1.0 + @as(f64, @abs(im2p[x] - mu2p[x]))) /
                (1.0 + @as(f64, @abs(im1p[x] - mu1p[x]))) - 1.0;
            const artifact: f64 = @max(d1, 0.0);
            self.sum2[0] += artifact;
            self.sum2[1] += tothe4th(artifact);
            const detail_lost: f64 = @max(-d1, 0.0);
            self.sum2[2] += detail_lost;
            self.sum2[3] += tothe4th(detail_lost);
        }
    }

    fn finish(self: *EdgeAcc, one_per_pixels: f64, out: *[4]f64) void {
        self.sum2[0] += @reduce(.Add, self.art0);
        self.sum2[1] += @reduce(.Add, self.art1);
        self.sum2[2] += @reduce(.Add, self.det0);
        self.sum2[3] += @reduce(.Add, self.det1);
        out[0] = one_per_pixels * self.sum2[0];
        out[1] = @sqrt(@sqrt(one_per_pixels * self.sum2[1]));
        out[2] = one_per_pixels * self.sum2[2];
        out[3] = @sqrt(@sqrt(one_per_pixels * self.sum2[3]));
    }
};

/// Weighted sum of the sub-scores, as libjxl's Msssim::Score: the weight index
/// runs on through the scales actually computed, so with fewer than 6 scales
/// (a side of 112 px or less) the later planes take shifted weights.
/// libjxl's quirk, kept so small clips score as libjxl scores them.
fn score(plane_avg_ssim: [6][6]f64, plane_avg_edge: [6][12]f64, n_scales: u32) f64 {
    var ssim: f64 = 0.0;

    var i: u32 = 0;
    var plane: u32 = 0;

    while (plane < 3) : (plane += 1) {
        var s: u32 = 0;
        while (s < n_scales) : (s += 1) {
            var n: u32 = 0;
            while (n < 2) : (n += 1) {
                ssim = simd.madd(f64, weight[i], @abs(plane_avg_ssim[s][plane * 2 + n]), ssim);
                i += 1;
                ssim = simd.madd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n]), ssim);
                i += 1;
                ssim = simd.madd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n + 2]), ssim);
                i += 1;
            }
        }
    }

    ssim *= 0.9562382616834844;
    ssim = (6.248496625763138e-5 * ssim * ssim) * ssim +
        2.326765642916932 * ssim -
        0.020884521182843837 * ssim * ssim;

    if (ssim > 0.0) {
        ssim = math.pow(f64, ssim, 0.6276336467831387) * -10.0 + 100.0;
    } else {
        ssim = 100.0;
    }

    return ssim;
}

const weight = [108]f64{
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
};
