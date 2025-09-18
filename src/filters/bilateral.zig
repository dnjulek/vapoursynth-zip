const std = @import("std");
const math = std.math;

const hz = @import("../helper.zig");
const Data = @import("../vapoursynth/bilateral.zig").Data;
const vszip = @import("../vszip.zig");

const allocator = std.heap.c_allocator;

pub fn bilateral(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, w: u32, h: u32, plane: u32, comptime join: bool, d: *Data) void {
    if (@typeInfo(T) == .int) {
        if (d.algorithm[plane] == 1) {
            bilateralAlg1(T, srcp, refp, dstp, stride, w, h, plane, d);
        } else {
            if (join) {
                bilateralAlg2Ref(T, srcp, refp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane], d.peak);
            } else {
                bilateralAlg2(T, srcp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane], d.peak);
            }
        }
    } else {
        if (d.algorithm[plane] == 1) {
            bilateralAlg1Float(T, srcp, refp, dstp, stride, w, h, plane, d);
        } else {
            if (join) {
                bilateralAlg2RefFloat(T, srcp, refp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane]);
            } else {
                bilateralAlg2Float(T, srcp, dstp, d.gs_lut[plane], d.gr_lut[plane], stride, w, h, d.radius[plane], d.step[plane]);
            }
        }
    }
}

fn bilateralAlg1(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, width: u32, height: u32, plane: u32, d: *Data) void {
    const sigma: f64 = d.sigmaS[plane];
    const PBFICnum = d.PBFICnum[plane];
    const pcount: u32 = stride * height;
    const gr_lut: []f32 = d.gr_lut[plane];
    const PBFICk: []T = allocator.alloc(T, PBFICnum) catch unreachable;
    defer allocator.free(PBFICk);

    const PBFICnumF: f32 = @floatFromInt(PBFICnum);
    for (PBFICk, 0..) |*i, k| {
        const kF: f32 = @floatFromInt(k);
        i.* = math.lossyCast(T, d.peak * kF / (PBFICnumF - 1) + 0.5);
    }

    var B: f32 = undefined;
    var B1: f32 = undefined;
    var B2: f32 = undefined;
    var B3: f32 = undefined;
    recursiveGaussianParameters(sigma, &B, &B1, &B2, &B3);

    const PBFIC: [][]f32 = allocator.alloc([]f32, PBFICnum) catch unreachable;
    const wk: []f32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    const jk: []f32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    defer allocator.free(wk);
    defer allocator.free(jk);

    for (0..PBFICnum) |k| {
        PBFIC[k] = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                wk[i] = gr_lut[hz.absDiff(PBFICk[k], refp[i])];
                jk[i] = wk[i] * @as(f32, @floatFromInt(srcp[i]));
            }
        }

        recursiveGaussian2DHorizontal(wk, wk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(wk, wk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DHorizontal(jk, jk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(jk, jk, height, width, stride, B, B1, B2, B3);

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

            const iF: f32 = @floatFromInt(refp[i]);
            const PBFICk0F: f32 = @floatFromInt(PBFICk[k]);
            const PBFICk1F: f32 = @floatFromInt(PBFICk[k + 1]);
            const vf: f32 = ((PBFICk1F - iF) * PBFIC[k][i] + (iF - PBFICk0F) * PBFIC[k + 1][i]) / (PBFICk1F - PBFICk0F);
            dstp[i] = @intFromFloat(math.clamp(vf + 0.5, 0, d.peak));
        }
    }

    for (PBFIC) |i| {
        allocator.free(i);
    }

    allocator.free(PBFIC);
}

fn bilateralAlg2(comptime T: type, src: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius: u32, step: u32, peak: f32) void {
    const radius2: u32 = radius + 1;

    var y: u32 = radius;
    while (y < height - radius) : (y += 1) {
        var x: u32 = radius;
        while (x < width - radius) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = src[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = @as(f32, @floatFromInt(src[xy])) * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys - yys)..];
                const line_b = src[(ys + yys)..];

                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const cxx1: T = line_a[x + xx];
                    const cxx2: T = line_b[x + xx];
                    const cxx3: T = line_a[x - xx];
                    const cxx4: T = line_b[x - xx];
                    const cxx1f: f32 = @floatFromInt(cxx1);
                    const cxx2f: f32 = @floatFromInt(cxx2);
                    const cxx3f: f32 = @floatFromInt(cxx3);
                    const cxx4f: f32 = @floatFromInt(cxx4);

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[hz.absDiff(cx, cxx1)];
                    const rwei2 = gr_lut[hz.absDiff(cx, cxx2)];
                    const rwei3 = gr_lut[hz.absDiff(cx, cxx3)];
                    const rwei4 = gr_lut[hz.absDiff(cx, cxx4)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1f * rwei1 + cxx2f * rwei2 + cxx3f * rwei3 + cxx4f * rwei4);
                }
            }

            dst[xy] = @intFromFloat(math.clamp(sum / weight_sum + 0.5, 0.0, peak));
        }
    }

    alg2Edges(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, peak, 0, 0, radius, width);
    alg2Edges(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, peak, height - radius, 0, height, width);
    alg2Edges(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, 0, height - radius, radius);
    alg2Edges(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, width - radius, height - radius, width);
}

fn bilateralAlg2Ref(comptime T: type, src: []const T, ref: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius: u32, step: u32, peak: f32) void {
    const radius2: u32 = radius + 1;

    var y: u32 = radius;
    while (y < height - radius) : (y += 1) {
        var x: u32 = radius;
        while (x < width - radius) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = ref[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = @as(f32, @floatFromInt(src[xy])) * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys - yys)..];
                const line_b = src[(ys + yys)..];
                const line_ar = ref[(ys - yys)..];
                const line_br = ref[(ys + yys)..];
                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const cxx1r: T = line_ar[x + xx];
                    const cxx2r: T = line_br[x + xx];
                    const cxx3r: T = line_ar[x - xx];
                    const cxx4r: T = line_br[x - xx];
                    const cxx1f: f32 = @floatFromInt(line_a[x + xx]);
                    const cxx2f: f32 = @floatFromInt(line_b[x + xx]);
                    const cxx3f: f32 = @floatFromInt(line_a[x - xx]);
                    const cxx4f: f32 = @floatFromInt(line_b[x - xx]);

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[hz.absDiff(cx, cxx1r)];
                    const rwei2 = gr_lut[hz.absDiff(cx, cxx2r)];
                    const rwei3 = gr_lut[hz.absDiff(cx, cxx3r)];
                    const rwei4 = gr_lut[hz.absDiff(cx, cxx4r)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1f * rwei1 + cxx2f * rwei2 + cxx3f * rwei3 + cxx4f * rwei4);
                }
            }

            dst[xy] = @intFromFloat(math.clamp(sum / weight_sum + 0.5, 0.0, peak));
        }
    }

    alg2Edges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, 0, 0, radius, width);
    alg2Edges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, height - radius, 0, height, width);
    alg2Edges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, 0, height - radius, radius);
    alg2Edges(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, peak, radius, width - radius, height - radius, width);
}

fn bilateralAlg1Float(comptime T: type, srcp: []const T, refp: []const T, dstp: []T, stride: u32, width: u32, height: u32, plane: u32, d: *Data) void {
    const sigma: f64 = d.sigmaS[plane];
    const PBFICnum = d.PBFICnum[plane];
    const pcount: u32 = stride * height;
    const gr_lut: []f32 = d.gr_lut[plane];
    const PBFICk: []T = allocator.alloc(T, PBFICnum) catch unreachable;
    defer allocator.free(PBFICk);

    const PBFICnumF: T = @floatFromInt(PBFICnum - 1);
    for (PBFICk, 0..) |*i, k| {
        const kF: T = @floatFromInt(k);
        i.* = kF / PBFICnumF;
    }

    var B: f32 = undefined;
    var B1: f32 = undefined;
    var B2: f32 = undefined;
    var B3: f32 = undefined;
    recursiveGaussianParameters(sigma, &B, &B1, &B2, &B3);

    const PBFIC: [][]f32 = allocator.alloc([]f32, PBFICnum) catch unreachable;
    const wk: []f32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    const jk: []f32 = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
    defer allocator.free(wk);
    defer allocator.free(jk);

    for (0..PBFICnum) |k| {
        PBFIC[k] = allocator.alignedAlloc(f32, vszip.alignment, pcount) catch unreachable;
        for (0..height) |j| {
            var i = stride * j;
            const upper = i + width;
            while (i < upper) : (i += 1) {
                wk[i] = gr_lut[@intFromFloat(@as(f32, @abs(PBFICk[k] - refp[i])) * 65535 + 0.5)];
                jk[i] = wk[i] * srcp[i];
            }
        }

        recursiveGaussian2DHorizontal(wk, wk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(wk, wk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DHorizontal(jk, jk, height, width, stride, B, B1, B2, B3);
        recursiveGaussian2DVertical(jk, jk, height, width, stride, B, B1, B2, B3);

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

            const iF: f32 = refp[i];
            const PBFICk0F: f32 = PBFICk[k];
            const PBFICk1F: f32 = PBFICk[k + 1];
            dstp[i] = @floatCast(((PBFICk1F - iF) * PBFIC[k][i] + (iF - PBFICk0F) * PBFIC[k + 1][i]) / (PBFICk1F - PBFICk0F));
        }
    }

    for (PBFIC) |i| {
        allocator.free(i);
    }

    allocator.free(PBFIC);
}

fn bilateralAlg2Float(comptime T: type, src: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius: u32, step: u32) void {
    const radius2: u32 = radius + 1;

    var y: u32 = radius;
    while (y < height - radius) : (y += 1) {
        var x: u32 = radius;
        while (x < width - radius) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = src[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = cx * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys - yys)..];
                const line_b = src[(ys + yys)..];

                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const cxx1: T = line_a[x + xx];
                    const cxx2: T = line_b[x + xx];
                    const cxx3: T = line_a[x - xx];
                    const cxx4: T = line_b[x - xx];

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx1)) * 65535 + 0.5)];
                    const rwei2 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx2)) * 65535 + 0.5)];
                    const rwei3 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx3)) * 65535 + 0.5)];
                    const rwei4 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx4)) * 65535 + 0.5)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1 * rwei1 + cxx2 * rwei2 + cxx3 * rwei3 + cxx4 * rwei4);
                }
            }

            dst[xy] = @floatCast(sum / weight_sum);
        }
    }

    alg2EdgesFloat(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, 0, 0, radius, width);
    alg2EdgesFloat(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, height - radius, 0, height, width);
    alg2EdgesFloat(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, radius, 0, height - radius, radius);
    alg2EdgesFloat(T, dst, src, src, gs_lut, gr_lut, stride, width, height, radius2, step, radius, width - radius, height - radius, width);
}

fn bilateralAlg2RefFloat(comptime T: type, src: []const T, ref: []const T, dst: []T, gs_lut: []f32, gr_lut: []f32, stride: u32, width: u32, height: u32, radius: u32, step: u32) void {
    const radius2: u32 = radius + 1;

    var y: u32 = radius;
    while (y < height - radius) : (y += 1) {
        var x: u32 = radius;
        while (x < width - radius) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = ref[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = src[xy] * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys - yys)..];
                const line_b = src[(ys + yys)..];
                const line_ar = ref[(ys - yys)..];
                const line_br = ref[(ys + yys)..];

                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const cxx1r: T = line_ar[x + xx];
                    const cxx2r: T = line_br[x + xx];
                    const cxx3r: T = line_ar[x - xx];
                    const cxx4r: T = line_br[x - xx];
                    const cxx1: T = line_a[x + xx];
                    const cxx2: T = line_b[x + xx];
                    const cxx3: T = line_a[x - xx];
                    const cxx4: T = line_b[x - xx];

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx1r)) * 65535 + 0.5)];
                    const rwei2 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx2r)) * 65535 + 0.5)];
                    const rwei3 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx3r)) * 65535 + 0.5)];
                    const rwei4 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx4r)) * 65535 + 0.5)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1 * rwei1 + cxx2 * rwei2 + cxx3 * rwei3 + cxx4 * rwei4);
                }
            }

            dst[xy] = @floatCast(sum / weight_sum);
        }
    }

    alg2EdgesFloat(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, 0, 0, radius, width);
    alg2EdgesFloat(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, height - radius, 0, height, width);
    alg2EdgesFloat(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, radius, 0, height - radius, radius);
    alg2EdgesFloat(T, dst, src, ref, gs_lut, gr_lut, stride, width, height, radius2, step, radius, width - radius, height - radius, width);
}

fn alg2Edges(
    comptime T: type,
    dst: []T,
    src: []const T,
    ref: []const T,
    gs_lut: []f32,
    gr_lut: []f32,
    stride: u32,
    width: u32,
    height: u32,
    radius2: u32,
    step: u32,
    peak: f32,
    y_start: u32,
    x_start: u32,
    y_end: u32,
    x_end: u32,
) void {
    var y: u32 = y_start;
    const max_line = stride * (height - 1);
    while (y < y_end) : (y += 1) {
        var x: u32 = x_start;
        while (x < x_end) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = ref[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = @as(f32, @floatFromInt(src[xy])) * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys -| yys)..];
                const line_b = src[@min(ys + yys, max_line)..];
                const line_ar = ref[(ys -| yys)..];
                const line_br = ref[@min(ys + yys, max_line)..];
                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const xxa = @min(x + xx, width - 1);
                    const xxb = x -| xx;
                    const cxx1: T = line_ar[xxa];
                    const cxx2: T = line_br[xxa];
                    const cxx3: T = line_ar[xxb];
                    const cxx4: T = line_br[xxb];
                    const cxx1f: f32 = @floatFromInt(line_a[xxa]);
                    const cxx2f: f32 = @floatFromInt(line_b[xxa]);
                    const cxx3f: f32 = @floatFromInt(line_a[xxb]);
                    const cxx4f: f32 = @floatFromInt(line_b[xxb]);

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[hz.absDiff(cx, cxx1)];
                    const rwei2 = gr_lut[hz.absDiff(cx, cxx2)];
                    const rwei3 = gr_lut[hz.absDiff(cx, cxx3)];
                    const rwei4 = gr_lut[hz.absDiff(cx, cxx4)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1f * rwei1 + cxx2f * rwei2 + cxx3f * rwei3 + cxx4f * rwei4);
                }
            }

            dst[xy] = @intFromFloat(math.clamp(sum / weight_sum + 0.5, 0.0, peak));
        }
    }
}

fn alg2EdgesFloat(
    comptime T: type,
    dst: []T,
    src: []const T,
    ref: []const T,
    gs_lut: []f32,
    gr_lut: []f32,
    stride: u32,
    width: u32,
    height: u32,
    radius2: u32,
    step: u32,
    y_start: u32,
    x_start: u32,
    y_end: u32,
    x_end: u32,
) void {
    var y: u32 = y_start;
    const max_line = stride * (height - 1);
    while (y < y_end) : (y += 1) {
        var x: u32 = x_start;
        while (x < x_end) : (x += 1) {
            const ys = y * stride;
            const xy = x + ys;
            const cx: T = ref[xy];
            var weight_sum = gs_lut[0] * gr_lut[0];
            var sum = src[xy] * weight_sum;

            var yy: u32 = 1;
            while (yy < radius2) : (yy += step) {
                const yys: u32 = yy * stride;
                const line_a = src[(ys -| yys)..];
                const line_b = src[@min(ys + yys, max_line)..];
                const line_ar = ref[(ys -| yys)..];
                const line_br = ref[@min(ys + yys, max_line)..];
                var xx: u32 = 1;
                while (xx < radius2) : (xx += step) {
                    const xxa = @min(x + xx, width - 1);
                    const xxb = x -| xx;
                    const cxx1r: T = line_ar[xxa];
                    const cxx2r: T = line_br[xxa];
                    const cxx3r: T = line_ar[xxb];
                    const cxx4r: T = line_br[xxb];
                    const cxx1: T = line_a[xxa];
                    const cxx2: T = line_b[xxa];
                    const cxx3: T = line_a[xxb];
                    const cxx4: T = line_b[xxb];

                    const swei = gs_lut[yy * radius2 + xx];
                    const rwei1 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx1r)) * 65535 + 0.5)];
                    const rwei2 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx2r)) * 65535 + 0.5)];
                    const rwei3 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx3r)) * 65535 + 0.5)];
                    const rwei4 = gr_lut[@intFromFloat(@as(f32, @abs(cx - cxx4r)) * 65535 + 0.5)];
                    weight_sum += swei * (rwei1 + rwei2 + rwei3 + rwei4);
                    sum += swei * (cxx1 * rwei1 + cxx2 * rwei2 + cxx3 * rwei3 + cxx4 * rwei4);
                }
            }

            dst[xy] = @floatCast(sum / weight_sum);
        }
    }
}

pub fn gaussianFunctionSpatialLUTGeneration(gs_lut: []f32, upper: u32, sigmaS: f64) void {
    var y: u32 = 0;
    while (y < upper) : (y += 1) {
        var x: u32 = 0;
        while (x < upper) : (x += 1) {
            gs_lut[y * upper + x] = math.lossyCast(f32, @exp(@as(f64, @floatFromInt(x * x + y * y)) / (sigmaS * sigmaS * -2.0)));
        }
    }
}

pub fn gaussianFunctionRangeLUTGeneration(gr_lut: []f32, range: f64, sigmaR: f64) void {
    const upper: u32 = @intFromFloat(@min(range, (sigmaR * 8.0 * range + 0.5)));

    var i: u32 = 0;
    while (i <= upper) : (i += 1) {
        const j: f64 = @as(f64, @floatFromInt(i)) / range;
        gr_lut[i] = math.lossyCast(f32, normalizedGaussianFunction(j, sigmaR));
    }

    if (i < gr_lut.len) {
        const upperLUTvalue: f32 = gr_lut[upper];
        while (i < gr_lut.len) : (i += 1) {
            gr_lut[i] = upperLUTvalue;
        }
    }
}

fn normalizedGaussianFunction(y: f64, sigma: f64) f64 {
    const x = y / sigma;
    return @exp(x * x / -2) / (math.sqrt(2.0 * math.pi) * sigma);
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

fn recursiveGaussian2DVertical(output: []f32, input: []const f32, height: u32, width: u32, stride: u32, B: f32, B1: f32, B2: f32, B3: f32) void {
    if (output.ptr != input.ptr) {
        @memcpy(output[0..width], input[0..width]);
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

    var i: i32 = @bitCast(height - 1);
    while (i >= 0) : (i -= 1) {
        const j: u32 = @bitCast(i);
        const lower: u32 = stride * j;
        const upper: u32 = lower + width;

        var x0: u32 = lower;
        var x1: u32 = if (j >= height - 1) x0 else (x0 + stride);
        var x2: u32 = if (j >= height - 2) x1 else (x1 + stride);
        var x3: u32 = if (j >= height - 3) x2 else (x2 + stride);

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

fn recursiveGaussian2DHorizontal(output: []f32, input: []const f32, height: u32, width: u32, stride: u32, B: f32, B1: f32, B2: f32, B3: f32) void {
    for (0..height) |j| {
        const lower: usize = stride * j;
        const upper: usize = lower + width;

        var i: isize = @bitCast(lower);
        var P0: f32 = undefined;
        var P1: f32 = input[@bitCast(i)];
        var P2: f32 = P1;
        var P3: f32 = P2;
        output[@bitCast(i)] = P3;
        i += 1;

        while (i < upper) : (i += 1) {
            P0 = B * input[@bitCast(i)] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[@bitCast(i)] = P0;
        }

        i -= 1;

        P1 = output[@bitCast(i)];
        P2 = P1;
        P3 = P2;
        i -= 1;
        while (i >= lower) : (i -= 1) {
            P0 = B * output[@bitCast(i)] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[@bitCast(i)] = P0;
        }
    }
}
