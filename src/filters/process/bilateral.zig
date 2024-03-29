const std = @import("std");
const math = std.math;
const helper = @import("../../helper.zig");
const BilateralData = @import("../bilateral.zig").BilateralData;

const allocator = std.heap.c_allocator;

pub fn gaussianFunctionSpatialLUTGeneration(gs_lut: [*]f32, upper: usize, sigmaS: f64) void {
    var y: usize = 0;
    while (y < upper) : (y += 1) {
        var x: usize = 0;
        while (x < upper) : (x += 1) {
            gs_lut[y * upper + x] = math.lossyCast(f32, @exp(@as(f64, @floatFromInt(x * x + y * y)) / (sigmaS * sigmaS * -2.0)));
        }
    }
}

pub fn gaussianFunctionRangeLUTGeneration(gr_lut: [*]f32, range: usize, sigmaR: f64) void {
    const levels: usize = range + 1;
    const range_f: f64 = @floatFromInt(range);
    const upper: usize = @intFromFloat(@min(range_f, (sigmaR * 8.0 * range_f + 0.5)));

    var i: usize = 0;
    while (i <= upper) : (i += 1) {
        const j: f64 = @as(f64, @floatFromInt(i)) / range_f;
        gr_lut[i] = math.lossyCast(f32, normalizedGaussianFunction(j, sigmaR));
    }

    if (i < levels) {
        const upperLUTvalue: f32 = gr_lut[upper];
        while (i < levels) : (i += 1) {
            gr_lut[i] = upperLUTvalue;
        }
    }
}

fn normalizedGaussianFunction(y: f64, sigma: f64) f64 {
    const x = y / sigma;
    return @exp(x * x / -2) / (math.sqrt(2.0 * math.pi) * sigma);
}

fn stride_cal(comptime T: type, width: usize) usize {
    const alignment: usize = 32 / @sizeOf(T);
    return if (width % alignment == 0) width else (width / alignment + 1) * alignment;
}

fn data2buff(comptime T: type, dst: [*]T, src: [*]const T, radius: usize, bufheight: usize, bufwidth: usize, bufstride: usize, height: usize, width: usize, stride: usize) void {
    var srcp = src;
    var dstp = dst;

    var y: usize = 0;
    while (y < height) : (y += 1) {
        dstp = dst + (radius + y) * bufstride;
        srcp = src + y * stride;

        var x: usize = 0;
        while (x < radius) : (x += 1) {
            dstp[x] = srcp[0];
        }
        var tmpp = dstp + radius;
        @memcpy(tmpp[0..width], srcp);

        x = radius + width;
        while (x < bufwidth) : (x += 1) {
            dstp[x] = srcp[width - 1];
        }
    }

    srcp = dst + radius * bufstride;
    y = 0;
    while (y < radius) : (y += 1) {
        dstp = dst + y * bufstride;
        @memcpy(dstp[0..bufwidth], srcp);
    }

    srcp = dst + (radius + height - 1) * bufstride;
    y = radius + height;

    while (y < bufheight) : (y += 1) {
        dstp = dst + y * bufstride;
        @memcpy(dstp[0..bufwidth], srcp);
    }
}

fn recursiveGaussianParameters(sigma: f64, B: *f32, B1: *f32, B2: *f32, B3: *f32) void {
    const q: f64 = if (sigma < 2.5) (3.97156 - 4.14554 * math.sqrt(1 - 0.26891 * sigma)) else 0.98711 * sigma - 0.96330;

    const b0: f64 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    const b1: f64 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    const b2: f64 = -(1.4281 * q * q + 1.26661 * q * q * q);
    const b3: f64 = 0.422205 * q * q * q;

    B.* = @floatCast(1 - (b1 + b2 + b3) / b0);
    B1.* = @floatCast(b1 / b0);
    B2.* = @floatCast(b2 / b0);
    B3.* = @floatCast(b3 / b0);
}

fn recursiveGaussian2DVertical(output: [*]f32, input: [*]const f32, height: u32, width: u32, stride: usize, B: f32, B1: f32, B2: f32, B3: f32) void {
    if (output != input) {
        @memcpy(output[0..width], input);
    }

    for (0..height) |j| {
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var x0: usize = lower;
        var x1: usize = if (j < 1) x0 else (x0 - stride);
        var x2: usize = if (j < 2) x1 else (x1 - stride);
        var x3: usize = if (j < 3) x2 else (x2 - stride);

        while (x0 < upper) : ({
            x0 += 1;
            x1 += 1;
            x2 += 1;
            x3 += 1;
        }) {
            const P3: f32 = output[x3];
            const P2: f32 = output[x2];
            const P1: f32 = output[x1];
            const P0: f32 = input[x0];
            output[x0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    }

    var i: i32 = @intCast(height - 1);
    while (i >= 0) : (i -= 1) {
        const j: usize = @intCast(i);
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var x0: usize = lower;
        var x1: usize = if (j >= height - 1) x0 else (x0 + stride);
        var x2: usize = if (j >= height - 2) x1 else (x1 + stride);
        var x3: usize = if (j >= height - 3) x2 else (x2 + stride);

        while (x0 < upper) : ({
            x0 += 1;
            x1 += 1;
            x2 += 1;
            x3 += 1;
        }) {
            const P3: f32 = output[x3];
            const P2: f32 = output[x2];
            const P1: f32 = output[x1];
            const P0: f32 = output[x0];
            output[x0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    }
}

fn recursiveGaussian2DHorizontal(output: [*]f32, input: [*]const f32, height: u32, width: u32, stride: usize, B: f32, B1: f32, B2: f32, B3: f32) void {
    for (0..height) |j| {
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var i: i32 = @intCast(lower);
        var P0: f32 = undefined;
        var P1: f32 = input[@intCast(i)];
        var P2: f32 = P1;
        var P3: f32 = P2;
        output[@intCast(i)] = P3;
        i += 1;

        while (i < upper) : (i += 1) {
            P0 = B * input[@intCast(i)] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[@intCast(i)] = P0;
        }

        i -= 1;

        P1 = output[@intCast(i)];
        P2 = P1;
        P3 = P2;
        i -= 1;
        while (i >= lower) : (i -= 1) {
            P0 = B * output[@intCast(i)] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[@intCast(i)] = P0;
        }
    }
}

pub fn Bilateral2D_1(comptime T: type, srcp: [*]const T, dstp: [*]T, refp: [*]const T, stride: usize, width: u32, height: u32, plane: u32, d: *BilateralData) void {
    const sigma: f64 = d.sigmaS[plane];
    const PBFICnum = d.PBFICnum[plane];
    const pcount: usize = stride * height;
    const gr_lut: []f32 = d.gr_lut[plane];
    const PBFICk: []T = allocator.alloc(T, PBFICnum) catch unreachable;
    defer allocator.free(PBFICk);

    const PBFICnumF: f32 = @floatFromInt(PBFICnum);
    for (PBFICk, 0..) |*i, k| {
        const kF: f32 = @floatFromInt(k);
        i.* = @intFromFloat(d.peak * kF / (PBFICnumF - 1) + 0.5);
    }

    var B: f32 = undefined;
    var B1: f32 = undefined;
    var B2: f32 = undefined;
    var B3: f32 = undefined;
    recursiveGaussianParameters(sigma, &B, &B1, &B2, &B3);

    const PBFIC: [][]f32 = allocator.alloc([]f32, PBFICnum) catch unreachable;
    const wk: []f32 = allocator.alignedAlloc(f32, 32, pcount) catch unreachable;
    const jk: []f32 = allocator.alignedAlloc(f32, 32, pcount) catch unreachable;
    defer allocator.free(wk);
    defer allocator.free(jk);

    for (0..PBFICnum) |k| {
        PBFIC[k] = allocator.alignedAlloc(f32, 32, pcount) catch unreachable;
        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                wk[i] = gr_lut[helper.absDiff(PBFICk[k], refp[i])];
                jk[i] = wk[i] * @as(f32, @floatFromInt(srcp[i]));
            }
        }

        recursiveGaussian2DHorizontal(wk.ptr, wk.ptr, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(wk.ptr, wk.ptr, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DHorizontal(jk.ptr, jk.ptr, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(jk.ptr, jk.ptr, height, width, stride, B, B1, B2, B3);

        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                PBFIC[k][i] = if (wk[i] == 0) 0 else (jk[i] / wk[i]);
            }
        }
    }

    for (0..height) |j| {
        var i = stride * j;
        const upper = i + width;
        while (i < upper) : (i += 1) {
            var k: u32 = 0;
            while (k < (PBFICnum - 2)) : (k += 1) {
                if ((refp[i] < PBFICk[k + 1]) and (refp[i] >= PBFICk[k])) {
                    break;
                }
            }

            const refF: f32 = @floatFromInt(refp[i]);
            const PBFICk0F: f32 = @floatFromInt(PBFICk[k]);
            const PBFICk1F: f32 = @floatFromInt(PBFICk[k + 1]);
            const vf: f32 = ((PBFICk1F - refF) * PBFIC[k][i] + (refF - PBFICk0F) * PBFIC[k + 1][i]) / (PBFICk1F - PBFICk0F);
            dstp[i] = @intFromFloat(math.clamp(vf + 0.5, 0, d.peak));
        }
    }

    for (PBFIC) |i| {
        allocator.free(i);
    }

    allocator.free(PBFIC);
}

pub fn Bilateral2D_2(comptime T: type, dst: [*]T, src: [*]const T, gs_lut: [*]f32, gr_lut: [*]f32, stride: usize, width: u32, height: u32, radius: u32, samplestep: u32, peak: f32) void {
    var srcp: [*]const T = src;
    var dstp: [*]T = dst;
    const radius2: usize = radius + 1;
    const bufheight: usize = height + radius * 2;
    const bufwidth: usize = width + radius * 2;
    const bufstride: usize = stride_cal(T, bufwidth);

    const srcbuff_arr = allocator.alignedAlloc(T, 32, bufheight * bufstride) catch unreachable;
    defer allocator.free(srcbuff_arr);
    const srcbuff: [*]T = srcbuff_arr.ptr;

    data2buff(T, srcbuff, srcp, radius, bufheight, bufwidth, bufstride, height, width, stride);

    var swei: f32 = undefined;
    var rwei1: f32 = undefined;
    var rwei2: f32 = undefined;
    var rwei3: f32 = undefined;
    var rwei4: f32 = undefined;
    var weight_sum: f32 = undefined;
    var sum: f32 = undefined;

    var y: usize = 0;
    while (y < height) : (y += 1) {
        const tmp1: usize = (radius + y) * bufstride;
        var x: usize = 0;
        while (x < width) : (x += 1) {
            const tmp2: usize = radius + x + tmp1;
            const cx: T = srcp[x];

            weight_sum = gs_lut[0] * gr_lut[0];
            sum = @as(f32, @floatFromInt(srcp[x])) * weight_sum;

            var yy: usize = 1;
            while (yy < radius2) : (yy += samplestep) {
                const tmp3: usize = yy * bufstride;

                var xx: usize = 1;
                while (xx < radius2) : (xx += samplestep) {
                    const cxx1: T = srcbuff[tmp2 + tmp3 + xx];
                    const cxx2: T = srcbuff[tmp2 + tmp3 - xx];
                    const cxx3: T = srcbuff[tmp2 - tmp3 - xx];
                    const cxx4: T = srcbuff[tmp2 - tmp3 + xx];
                    const cxx1f: f32 = @floatFromInt(cxx1);
                    const cxx2f: f32 = @floatFromInt(cxx2);
                    const cxx3f: f32 = @floatFromInt(cxx3);
                    const cxx4f: f32 = @floatFromInt(cxx4);

                    swei = gs_lut[yy * radius2 + xx];
                    rwei1 = gr_lut[helper.absDiff(cx, cxx1)];
                    rwei2 = gr_lut[helper.absDiff(cx, cxx2)];
                    rwei3 = gr_lut[helper.absDiff(cx, cxx3)];
                    rwei4 = gr_lut[helper.absDiff(cx, cxx4)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1f * rwei1 + cxx2f * rwei2 + cxx3f * rwei3 + cxx4f * rwei4);
                }
            }
            dstp[x] = @intFromFloat(math.clamp(sum / weight_sum + 0.5, 0.0, peak));
        }
        srcp += stride;
        dstp += stride;
    }
}

pub fn Bilateral2D_2ref(comptime T: type, dst: [*]T, src: [*]const T, ref: [*]const T, gs_lut: [*]f32, gr_lut: [*]f32, stride: usize, width: u32, height: u32, radius: u32, samplestep: u32, peak: f32) void {
    var srcp: [*]const T = src;
    var refp: [*]const T = ref;
    var dstp: [*]T = dst;
    const radius2: usize = radius + 1;
    const bufheight: usize = height + radius * 2;
    const bufwidth: usize = width + radius * 2;
    const bufstride: usize = stride_cal(T, bufwidth);

    const srcbuff_arr = allocator.alignedAlloc(T, 32, bufheight * bufstride) catch unreachable;
    const refbuff_arr = allocator.alignedAlloc(T, 32, bufheight * bufstride) catch unreachable;
    defer allocator.free(srcbuff_arr);
    defer allocator.free(refbuff_arr);
    const srcbuff: [*]T = srcbuff_arr.ptr;
    const refbuff: [*]T = refbuff_arr.ptr;

    data2buff(T, srcbuff, srcp, radius, bufheight, bufwidth, bufstride, height, width, stride);
    data2buff(T, refbuff, refp, radius, bufheight, bufwidth, bufstride, height, width, stride);

    var y: usize = 0;
    while (y < height) : (y += 1) {
        const tmp1: usize = (radius + y) * bufstride;
        var x: usize = 0;
        while (x < width) : (x += 1) {
            const tmp2: usize = radius + x + tmp1;
            const cx: T = refp[x];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = @as(f32, @floatFromInt(srcp[x])) * weight_sum;

            var yy: usize = 1;
            while (yy < radius2) : (yy += samplestep) {
                const tmp3: usize = yy * bufstride;

                var xx: usize = 1;
                while (xx < radius2) : (xx += samplestep) {
                    const cxx1: T = refbuff[tmp2 + tmp3 + xx];
                    const cxx2: T = refbuff[tmp2 + tmp3 - xx];
                    const cxx3: T = refbuff[tmp2 - tmp3 - xx];
                    const cxx4: T = refbuff[tmp2 - tmp3 + xx];
                    const cxx1f: f32 = @floatFromInt(srcbuff[tmp2 + tmp3 + xx]);
                    const cxx2f: f32 = @floatFromInt(srcbuff[tmp2 + tmp3 - xx]);
                    const cxx3f: f32 = @floatFromInt(srcbuff[tmp2 - tmp3 - xx]);
                    const cxx4f: f32 = @floatFromInt(srcbuff[tmp2 - tmp3 + xx]);

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[helper.absDiff(cx, cxx1)];
                    const rwei2 = gr_lut[helper.absDiff(cx, cxx2)];
                    const rwei3 = gr_lut[helper.absDiff(cx, cxx3)];
                    const rwei4 = gr_lut[helper.absDiff(cx, cxx4)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1f * rwei1 + cxx2f * rwei2 + cxx3f * rwei3 + cxx4f * rwei4);
                }
            }
            dstp[x] = @intFromFloat(math.clamp(sum / weight_sum + 0.5, 0.0, peak));
        }
        srcp += stride;
        refp += stride;
        dstp += stride;
    }
}
