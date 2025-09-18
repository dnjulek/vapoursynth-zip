const std = @import("std");
const math = std.math;
const vszip = @import("../vszip.zig");

const allocator = std.heap.c_allocator;

const ksize = 9;
const radius = 4;

pub fn process(srcp1: [3][]const f32, srcp2: [3][]const f32, stride: u32, w: u32, h: u32) f64 {
    const wh: u32 = stride * h;
    const temp_alloc = allocator.alignedAlloc(f32, vszip.alignment, wh * 18) catch unreachable;
    defer allocator.free(temp_alloc);
    var temp = temp_alloc[0..];

    var temp6x3: [6][3][]f32 = undefined;
    var x: u32 = 0;
    for (0..6) |i| {
        for (0..3) |ii| {
            temp6x3[i][ii] = temp[x..(x + wh)];
            x += wh;
        }
    }

    const srcp1b = temp6x3[0];
    const srcp2b = temp6x3[1];
    const tmpp1 = temp6x3[2];
    const tmpp2 = temp6x3[3];

    const tmpp3 = temp6x3[4][0];
    const tmpps11 = temp6x3[4][1];
    const tmpps22 = temp6x3[4][2];
    const tmpps12 = temp6x3[5][0];
    const tmppmu1 = temp6x3[5][1];

    for (0..3) |i| {
        @memcpy(srcp1b[i], srcp1[i]);
        @memcpy(srcp2b[i], srcp2[i]);
    }

    var plane_avg_ssim: [6][6]f64 = undefined;
    var plane_avg_edge: [6][12]f64 = undefined;
    var stride2 = stride;
    var w2 = w;
    var h2 = h;

    var scale: u32 = 0;
    while (scale < 6) : (scale += 1) {
        if (scale > 0) {
            downscale(srcp1b, srcp1b, stride2, w2, h2);
            downscale(srcp2b, srcp2b, stride2, w2, h2);
            stride2 = @divTrunc((stride2 + 1), 2);
            w2 = @divTrunc((w2 + 1), 2);
            h2 = @divTrunc((h2 + 1), 2);
        }

        const one_per_pixels: f64 = 1.0 / @as(f64, @floatFromInt(w2 * h2));
        toXYB(srcp1b, tmpp1, stride2, w2, h2);
        toXYB(srcp2b, tmpp2, stride2, w2, h2);

        var plane: u32 = 0;
        while (plane < 3) : (plane += 1) {
            multiply(tmpp1[plane], tmpp1[plane], tmpp3, stride2, w2, h2);
            blur(tmpp3, tmpps11, stride2, w2, h2);

            multiply(tmpp2[plane], tmpp2[plane], tmpp3, stride2, w2, h2);
            blur(tmpp3, tmpps22, stride2, w2, h2);

            multiply(tmpp1[plane], tmpp2[plane], tmpp3, stride2, w2, h2);
            blur(tmpp3, tmpps12, stride2, w2, h2);

            blur(tmpp1[plane], tmppmu1, stride2, w2, h2);
            blur(tmpp2[plane], tmpp3, stride2, w2, h2);

            ssimMap(
                tmpps11,
                tmpps22,
                tmpps12,
                tmppmu1,
                tmpp3,
                stride2,
                w2,
                h2,
                plane,
                one_per_pixels,
                &plane_avg_ssim[scale],
            );

            edgeMap(
                tmpp1[plane],
                tmpp2[plane],
                tmppmu1,
                tmpp3,
                stride2,
                w2,
                h2,
                plane,
                one_per_pixels,
                &plane_avg_edge[scale],
            );
        }
    }

    return score(plane_avg_ssim, plane_avg_edge);
}

inline fn downscale(src: [3][]f32, dst: [3][]f32, src_stride: u32, in_w: u32, in_h: u32) void {
    const fscale: f32 = 2.0;
    const uscale: u32 = 2;
    const out_w = @divTrunc((in_w + uscale - 1), uscale);
    const out_h = @divTrunc((in_h + uscale - 1), uscale);
    const dst_stride = @divTrunc((src_stride + uscale - 1), uscale);
    const normalize: f32 = 1.0 / (fscale * fscale);

    var plane: u32 = 0;
    while (plane < 3) : (plane += 1) {
        const srcp = src[plane];
        var dstp = dst[plane];
        var oy: u32 = 0;
        while (oy < out_h) : (oy += 1) {
            var ox: u32 = 0;
            while (ox < out_w) : (ox += 1) {
                var sum: f32 = 0.0;
                var iy: u32 = 0;
                while (iy < uscale) : (iy += 1) {
                    var ix: u32 = 0;
                    while (ix < uscale) : (ix += 1) {
                        const x: u32 = @min((ox * uscale + ix), (in_w - 1));
                        const y: u32 = @min((oy * uscale + iy), (in_h - 1));
                        sum += srcp[y * src_stride + x];
                    }
                }
                dstp[ox] = sum * normalize;
            }
            dstp = dstp[dst_stride..];
        }
    }
}

const vec_t: type = @Vector(16, f32);

inline fn multiplyVec(src1: anytype, src2: anytype, dst: []f32) void {
    dst[0..16].* = @as(vec_t, src1[0..16].*) * @as(vec_t, src2[0..16].*);
}

pub inline fn multiply(src1: []const f32, src2: []const f32, dst: []f32, stride: u32, w: u32, h: u32) void {
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var srcp1 = src1[(y * stride)..];
        var srcp2 = src2[(y * stride)..];
        var dstp = dst[(y * stride)..];
        var x: u32 = 0;
        while (x < w) : (x += 16) {
            const x2: u32 = x + 16;
            multiplyVec(srcp1[x..x2], srcp2[x..x2], dstp[x..x2]);
        }
    }
}

fn blurH(srcp: []f32, dstp: []f32, kernel: [ksize]f32, w: i32) void {
    var j: i32 = 0;
    while (j < @min(w, radius)) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = radius;
    while (j < w - @min(w, radius)) : (j += 1) {
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < ksize) : (k += 1) {
            sum += kernel[@intCast(k)] * srcp[@intCast(j - radius + k)];
        }

        dstp[@intCast(j)] = sum;
    }

    j = @max(radius, w - @min(w, radius));
    while (j < w) : (j += 1) {
        const dist_from_right: i32 = w - 1 - j;
        var sum: f32 = 0.0;
        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const idx: i32 = if (j < radius - k) @min(radius - k - j, w - 1) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const idx: i32 = if (dist_from_right < k - radius) (j - @min(k - radius - dist_from_right, j)) else (j - radius + k);
            sum += kernel[@intCast(k)] * srcp[@intCast(idx)];
        }

        dstp[@intCast(j)] = sum;
    }
}

inline fn blurV(src: anytype, dstp: []f32, kernel: [ksize]f32, w: u32) void {
    var j: u32 = 0;
    while (j < w) : (j += 1) {
        var accum: f32 = 0.0;
        var k: u32 = 0;
        while (k < 9) : (k += 1) {
            accum += kernel[k] * src[k][j];
        }

        dstp[j] = accum;
    }
}

pub inline fn blur(src: []const f32, dst: []f32, stride: u32, w: u32, h: u32) void {
    const kernel = [ksize]f32{
        0.0076144188642501831054687500,
        0.0360749699175357818603515625,
        0.1095860823988914489746093750,
        0.2134445458650588989257812500,
        0.2665599882602691650390625000,
        0.2134445458650588989257812500,
        0.1095860823988914489746093750,
        0.0360749699175357818603515625,
        0.0076144188642501831054687500,
    };

    var i: i32 = 0;
    const ih: i32 = @bitCast(h);
    while (i < h) : (i += 1) {
        const ui: u32 = @bitCast(i);
        var srcp: [ksize][]const f32 = undefined;
        const dstp: []f32 = dst[(ui * stride)..];
        const dist_from_bottom: i32 = ih - 1 - i;

        const tmp = allocator.alignedAlloc(f32, vszip.alignment, w) catch unreachable;
        defer allocator.free(tmp);

        var k: i32 = 0;
        while (k < radius) : (k += 1) {
            const row: i32 = if (i < radius - k) (@min(radius - k - i, ih - 1)) else (i - radius + k);
            const urow: u32 = @bitCast(row);
            srcp[@intCast(k)] = src[(urow * stride)..];
        }

        k = radius;
        while (k < ksize) : (k += 1) {
            const row: i32 = if (dist_from_bottom < k - radius) (i - @min(k - radius - dist_from_bottom, i)) else (i - radius + k);
            const urow: u32 = @bitCast(row);
            srcp[@intCast(k)] = src[(urow * stride)..];
        }

        blurV(srcp, tmp, kernel, w);
        blurH(tmp, dstp, kernel, @intCast(w));
    }
}

const K_D0: f32 = 0.0037930734;
const K_D1: f32 = std.math.lossyCast(f32, math.cbrt(@as(f32, K_D0)));

const V00: vec_t = @splat(@as(f32, 0.0));
const V05: vec_t = @splat(@as(f32, 0.5));
const V10: vec_t = @splat(@as(f32, 1.0));
const V11: vec_t = @splat(@as(f32, 1.1));

const V001: vec_t = @splat(@as(f32, 0.01));
const V005: vec_t = @splat(@as(f32, 0.05));
const V055: vec_t = @splat(@as(f32, 0.55));
const V042: vec_t = @splat(@as(f32, 0.42));
const V140: vec_t = @splat(@as(f32, 14.0));

const K_M02: vec_t = @splat(@as(f32, 0.078));
const K_M00: vec_t = @splat(@as(f32, 0.30));
const K_M01: vec_t = V10 - K_M02 - K_M00;

const K_M12: vec_t = @splat(@as(f32, 0.078));
const K_M10: vec_t = @splat(@as(f32, 0.23));
const K_M11: vec_t = V10 - K_M12 - K_M10;

const K_M20: vec_t = @splat(@as(f32, 0.24342269));
const K_M21: vec_t = @splat(@as(f32, 0.20476745));
const K_M22: vec_t = V10 - K_M20 - K_M21;

const OPSIN_ABSORBANCE_MATRIX = [_]vec_t{ K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22 };
const OPSIN_ABSORBANCE_BIAS: vec_t = @splat(K_D0);
const ABSORBANCE_BIAS: vec_t = @splat(-K_D1);

inline fn cbrtVec(x: vec_t) vec_t {
    var out: vec_t = undefined;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        out[i] = std.math.lossyCast(f32, math.cbrt(@as(f32, x[i])));
    }

    return out;
}

inline fn mixedToXYB(mixed: [3]vec_t) [3]vec_t {
    var out: [3]vec_t = undefined;
    out[0] = V05 * (mixed[0] - mixed[1]);
    out[1] = V05 * (mixed[0] + mixed[1]);
    out[2] = mixed[2];
    return out;
}

inline fn opsinAbsorbance(rgb: [3]vec_t) [3]vec_t {
    var out: [3]vec_t = undefined;
    out[0] = @mulAdd(
        vec_t,
        OPSIN_ABSORBANCE_MATRIX[0],
        rgb[0],
        @mulAdd(
            vec_t,
            OPSIN_ABSORBANCE_MATRIX[1],
            rgb[1],
            @mulAdd(
                vec_t,
                OPSIN_ABSORBANCE_MATRIX[2],
                rgb[2],
                OPSIN_ABSORBANCE_BIAS,
            ),
        ),
    );

    out[1] = @mulAdd(
        vec_t,
        OPSIN_ABSORBANCE_MATRIX[3],
        rgb[0],
        @mulAdd(
            vec_t,
            OPSIN_ABSORBANCE_MATRIX[4],
            rgb[1],
            @mulAdd(
                vec_t,
                OPSIN_ABSORBANCE_MATRIX[5],
                rgb[2],
                OPSIN_ABSORBANCE_BIAS,
            ),
        ),
    );

    out[2] = @mulAdd(
        vec_t,
        OPSIN_ABSORBANCE_MATRIX[6],
        rgb[0],
        @mulAdd(
            vec_t,
            OPSIN_ABSORBANCE_MATRIX[7],
            rgb[1],
            @mulAdd(
                vec_t,
                OPSIN_ABSORBANCE_MATRIX[8],
                rgb[2],
                OPSIN_ABSORBANCE_BIAS,
            ),
        ),
    );

    return out;
}

inline fn linearRGBtoXYB(input: [3]vec_t) [3]vec_t {
    var mixed: [3]vec_t = opsinAbsorbance(input);

    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        const pred: @Vector(16, bool) = mixed[i] < V00;
        mixed[i] = @select(f32, pred, V00, mixed[i]);
        mixed[i] = cbrtVec(mixed[i]) + ABSORBANCE_BIAS;
    }

    mixed = mixedToXYB(mixed);
    return mixed;
}

inline fn makePositiveXYB(xyb: *[3]vec_t) void {
    xyb[2] = (xyb[2] - xyb[1]) + V055;
    xyb[0] = xyb[0] * V140 + V042;
    xyb[1] += V001;
}

inline fn xybVec(src: [3][]const f32, dst: [3][]f32) void {
    var out: [3]vec_t = undefined;
    const rgb = [3]vec_t{
        src[0][0..16].*,
        src[1][0..16].*,
        src[2][0..16].*,
    };

    out = linearRGBtoXYB(rgb);
    makePositiveXYB(&out);

    for (dst, 0..) |p, i| {
        p[0..16].* = out[i];
    }
}

pub inline fn toXYB(_srcp: [3][]const f32, _dstp: [3][]f32, stride: u32, w: u32, h: u32) void {
    var srcp = _srcp;
    var dstp = _dstp;
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        var x: u32 = 0;
        while (x < w) : (x += 16) {
            const x2: u32 = x + 16;
            const srcps = [3][]const f32{
                srcp[0][x..x2],
                srcp[1][x..x2],
                srcp[2][x..x2],
            };

            const dstps = [3][]f32{
                dstp[0][x..x2],
                dstp[1][x..x2],
                dstp[2][x..x2],
            };

            xybVec(srcps, dstps);
        }

        var i: u32 = 0;
        while (i < 3) : (i += 1) {
            srcp[i] = srcp[i][stride..];
            dstp[i] = dstp[i][stride..];
        }
    }
}

inline fn tothe4th(y: f64) f64 {
    var x = y * y;
    x *= x;
    return x;
}

inline fn ssimMap(
    s11: []f32,
    s22: []f32,
    s12: []f32,
    mu1: []f32,
    mu2: []f32,
    stride: u32,
    w: u32,
    h: u32,
    plane: u32,
    one_per_pixels: f64,
    plane_avg_ssim: []f64,
) void {
    var sum1 = [2]f64{ 0.0, 0.0 };
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const s11p = s11[(y * stride)..];
        const s22p = s22[(y * stride)..];
        const s12p = s12[(y * stride)..];
        const mu1p = mu1[(y * stride)..];
        const mu2p = mu2[(y * stride)..];

        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const m1: f32 = mu1p[x];
            const m2: f32 = mu2p[x];
            const m11 = m1 * m1;
            const m22 = m2 * m2;
            const m12 = m1 * m2;
            const m_diff = m1 - m2;
            const num_m: f64 = @mulAdd(f32, m_diff, -m_diff, 1.0);
            const num_s: f64 = @mulAdd(f32, (s12p[x] - m12), 2.0, 0.0009);
            const denom_s: f64 = (s11p[x] - m11) + (s22p[x] - m22) + 0.0009;
            const d1: f64 = @max(1.0 - ((num_m * num_s) / denom_s), 0.0);

            sum1[0] += d1;
            sum1[1] += tothe4th(d1);
        }
    }

    plane_avg_ssim[plane * 2] = one_per_pixels * sum1[0];
    plane_avg_ssim[plane * 2 + 1] = @sqrt(@sqrt(one_per_pixels * sum1[1]));
}

inline fn edgeMap(
    im1: []f32,
    im2: []f32,
    mu1: []f32,
    mu2: []f32,
    stride: u32,
    w: u32,
    h: u32,
    plane: u32,
    one_per_pixels: f64,
    plane_avg_edge: []f64,
) void {
    var sum2 = [4]f64{ 0.0, 0.0, 0.0, 0.0 };
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const im1p = im1[(y * stride)..];
        const im2p = im2[(y * stride)..];
        const mu1p = mu1[(y * stride)..];
        const mu2p = mu2[(y * stride)..];

        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const d1: f64 = (1.0 + @as(f64, @abs(im2p[x] - mu2p[x]))) /
                (1.0 + @as(f64, @abs(im1p[x] - mu1p[x]))) - 1.0;
            const artifact: f64 = @max(d1, 0.0);
            sum2[0] += artifact;
            sum2[1] += tothe4th(artifact);
            const detail_lost: f64 = @max(-d1, 0.0);
            sum2[2] += detail_lost;
            sum2[3] += tothe4th(detail_lost);
        }
    }

    plane_avg_edge[plane * 4] = one_per_pixels * sum2[0];
    plane_avg_edge[plane * 4 + 1] = @sqrt(@sqrt(one_per_pixels * sum2[1]));
    plane_avg_edge[plane * 4 + 2] = one_per_pixels * sum2[2];
    plane_avg_edge[plane * 4 + 3] = @sqrt(@sqrt(one_per_pixels * sum2[3]));
}

inline fn score(plane_avg_ssim: [6][6]f64, plane_avg_edge: [6][12]f64) f64 {
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

    var ssim: f64 = 0.0;

    var i: u32 = 0;
    var plane: u32 = 0;

    while (plane < 3) : (plane += 1) {
        var s: u32 = 0;
        while (s < 6) : (s += 1) {
            var n: u32 = 0;
            while (n < 2) : (n += 1) {
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_ssim[s][plane * 2 + n]), ssim);
                i += 1;
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n]), ssim);
                i += 1;
                ssim = @mulAdd(f64, weight[i], @abs(plane_avg_edge[s][plane * 4 + n + 2]), ssim);
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
