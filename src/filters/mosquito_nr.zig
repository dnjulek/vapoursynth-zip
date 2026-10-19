const std = @import("std");
const simd = @import("simd.zig");

pub fn MosquitoNR(comptime T: type) type {
    return struct {
        const I = if (T == u8) i16 else i32;
        const VL = std.simd.suggestVectorLength(I) orelse 8;
        const V = @Vector(VL, I);
        const VT = @Vector(VL, T);
        const VShift = @Vector(VL, std.math.Log2Int(I));

        // Laundered: smooth()'s direction passes load 9-25 windows at fixed
        // +-1/+-2 column offsets; unlaundered, LLVM fuses them into
        // vpermd/vpalignr shuffle chains (measured).
        inline fn vload(s: []const I, i: usize) V {
            return simd.loaduOpaque(I, VL, s[i..][0..VL]);
        }
        inline fn vstore(s: []I, i: usize, v: V) void {
            s[i..][0..VL].* = v;
        }
        inline fn vshr(v: V, comptime n: comptime_int) V {
            return v >> @as(VShift, @splat(n));
        }
        inline fn vsplat(comptime n: I) V {
            return @splat(n);
        }
        inline fn vabs(v: V) V {
            return @max(v, -%v);
        }

        inline fn w32(x: I) i32 {
            return x;
        }

        // fwdH/invH vector regime: pinned to the shapes of the shared shuffle
        // helpers (16x i16 / 8x i32), independent of VL so a wider-vector
        // build cannot desync from them. The scalar loops stay in place as
        // the semantic reference and the non-AVX2 path.
        const HVL = if (I == i16) 16 else 8;
        const HV = @Vector(HVL, I);
        const HVShift = @Vector(HVL, std.math.Log2Int(I));
        const use_hvec = if (I == i16) simd.use_avx2 else simd.f32LanesAccelerated(HVL);

        // Laundered: fwdH/invH load the same span at +1/+2 element offsets
        // (row / row+2, hdr-1 / hdr / hdr+1); same fusion pessimization as
        // vload above.
        inline fn hload(s: []const I, i: usize) HV {
            return simd.loaduOpaque(I, HVL, s[i..][0..HVL]);
        }
        inline fn hstore(s: []I, i: usize, v: HV) void {
            s[i..][0..HVL].* = v;
        }
        inline fn hshr(v: HV, comptime n: comptime_int) HV {
            return v >> @as(HVShift, @splat(n));
        }
        inline fn hEven(a: HV, b: HV) HV {
            return if (I == i16)
                simd.evenLanesI16x16(a, b)
            else
                @bitCast(simd.evenLanesF32(HVL, @bitCast(a), @bitCast(b)));
        }
        inline fn hDeinterleave(a: HV, b: HV) [2]HV {
            return if (I == i16)
                simd.deinterleaveI16x16(a, b)
            else
                .{ @bitCast(simd.evenLanesF32(HVL, @bitCast(a), @bitCast(b))), @bitCast(simd.oddLanesF32(HVL, @bitCast(a), @bitCast(b))) };
        }
        inline fn hInterleave(e: HV, o: HV) [2]HV {
            return if (I == i16) simd.interleaveI16x16(e, o) else blk: {
                const p = simd.interleaveF32(HVL, @bitCast(e), @bitCast(o));
                break :blk .{ @bitCast(p[0]), @bitCast(p[1]) };
            };
        }

        // u8 at 16 (AVX2) / 32 (AVX512BW) lanes: one vpmaddwd per 8 / 16 pixels on interleaved (c, X)
        // pairs, no widening (|coef1*c| + |k2*X| <= 2 * 256 * 4080 and no
        // -32768 pairs); the generic form is its reference.
        inline fn blendX(c: V, x: V, kc: i32, kx: i32, comptime rnd: i32, comptime sh: u5) V {
            if (comptime I == i16 and ((VL == 16 and simd.use_avx2) or (VL == 32 and simd.use_avx512bw)))
                return simd.maddPairShrI16(VL, c, x, @intCast(kc), @intCast(kx), rnd, sh);
            const W = @Vector(VL, i32);
            const r = (@as(W, c) * @as(W, @splat(kc)) + @as(W, x) * @as(W, @splat(kx)) + @as(W, @splat(rnd))) >> @splat(sh);
            return if (I == i32) r else @intCast(r);
        }

        // Every direction's blend shares one formula: with coef1 == 2*coef0
        // and k3 == 2*k2, the 1/2-weight group rewrites exactly as
        //   (coef0*c + k2*S + rnd0) >> sh0 == (coef1*c + k2*(2*S) + 2*rnd0) >> (sh0+1)
        // (numerator doubled, one more floor shift), and the 1/4-weight
        // group as (coef1*c + k2*(2*T + Q) + rnd1) >> sh1. So only the tap
        // sum X differs per direction: the argmin picks X alongside the SAD,
        // and the multiply/round/shift runs once per pixel instead of once
        // per candidate. Range: |X| <= 8 * (65535 << 4) (u16: i32 lanes,
        // coef1*c + k2*X < 2^30) and X <= 8 * 4080 = 32640 (u8: fits i16).
        fn smoothRow(pr: [5][]const I, orow: []I, wv: usize, strength: i32, comptime radius: u32) void {
            const sh = if (radius == 1) 7 else 8;
            const coef1: i32 = if (radius == 1) 128 - 4 * strength else 256 - 8 * strength;
            const rnd: i32 = 1 << (sh - 1);

            const w = orow.len;
            const rm2 = pr[0];
            const rm1 = pr[1];
            const r0 = pr[2];
            const rp1 = pr[3];
            const rp2 = pr[4];

            // Runs over the padded width wv (the pl rows' round-up slack holds
            // valid samples); the last partial vector stores via a copy.
            var x: usize = 0;
            while (x < wv) : (x += VL) {
                const cx = x + 2;
                var sad: [8]V = undefined;
                var xs: [8]V = undefined;
                const c = vload(r0, cx);
                if (radius == 1) {
                    const p_m10 = vload(r0, cx - 1);
                    const p_p10 = vload(r0, cx + 1);
                    const p_0m1 = vload(rm1, cx);
                    const p_0p1 = vload(rp1, cx);
                    const p_m1m1 = vload(rm1, cx - 1);
                    const p_p1p1 = vload(rp1, cx + 1);
                    const p_p1m1 = vload(rm1, cx + 1);
                    const p_m1p1 = vload(rp1, cx - 1);
                    sad[0] = vabs(p_m10 -% c) +% vabs(p_p10 -% c);
                    sad[4] = vabs(vshr(p_m10 +% p_m1m1, 1) -% c) +% vabs(vshr(p_p10 +% p_p1p1, 1) -% c);
                    sad[1] = vabs(p_m1m1 -% c) +% vabs(p_p1p1 -% c);
                    sad[5] = vabs(vshr(p_m1m1 +% p_0m1, 1) -% c) +% vabs(vshr(p_p1p1 +% p_0p1, 1) -% c);
                    sad[2] = vabs(p_0m1 -% c) +% vabs(p_0p1 -% c);
                    sad[6] = vabs(vshr(p_0m1 +% p_p1m1, 1) -% c) +% vabs(vshr(p_0p1 +% p_m1p1, 1) -% c);
                    sad[3] = vabs(p_p1m1 -% c) +% vabs(p_m1p1 -% c);
                    sad[7] = vabs(vshr(p_p10 +% p_p1m1, 1) -% c) +% vabs(vshr(p_m10 +% p_m1p1, 1) -% c);
                    const s0 = p_m10 +% p_p10;
                    const s1 = p_m1m1 +% p_p1p1;
                    const s2 = p_0m1 +% p_0p1;
                    const s3 = p_p1m1 +% p_m1p1;
                    xs = .{ s0 +% s0, s1 +% s1, s2 +% s2, s3 +% s3, s0 +% s1, s1 +% s2, s3 +% s2, s3 +% s0 };
                } else {
                    const p_m10 = vload(r0, cx - 1);
                    const p_p10 = vload(r0, cx + 1);
                    const p_m20 = vload(r0, cx - 2);
                    const p_p20 = vload(r0, cx + 2);
                    const p_0m1 = vload(rm1, cx);
                    const p_0p1 = vload(rp1, cx);
                    const p_0m2 = vload(rm2, cx);
                    const p_0p2 = vload(rp2, cx);
                    const p_m1m1 = vload(rm1, cx - 1);
                    const p_p1p1 = vload(rp1, cx + 1);
                    const p_p1m1 = vload(rm1, cx + 1);
                    const p_m1p1 = vload(rp1, cx - 1);
                    const p_m2m2 = vload(rm2, cx - 2);
                    const p_p2p2 = vload(rp2, cx + 2);
                    const p_p2m2 = vload(rm2, cx + 2);
                    const p_m2p2 = vload(rp2, cx - 2);
                    const p_m2m1 = vload(rm1, cx - 2);
                    const p_p2p1 = vload(rp1, cx + 2);
                    const p_m1m2 = vload(rm2, cx - 1);
                    const p_p1p2 = vload(rp2, cx + 1);
                    const p_p1m2 = vload(rm2, cx + 1);
                    const p_m1p2 = vload(rp2, cx - 1);
                    const p_p2m1 = vload(rm1, cx + 2);
                    const p_m2p1 = vload(rp1, cx - 2);
                    sad[0] = vabs(p_m10 -% c) +% vabs(p_p10 -% c) +% vabs(p_m20 -% c) +% vabs(p_p20 -% c);
                    sad[4] = vabs(p_m2m1 -% c) +% vabs(p_p2p1 -% c) +%
                        vabs(vshr(p_m10 +% p_m1m1, 1) -% c) +% vabs(vshr(p_p10 +% p_p1p1, 1) -% c);
                    sad[1] = vabs(p_m1m1 -% c) +% vabs(p_p1p1 -% c) +% vabs(p_m2m2 -% c) +% vabs(p_p2p2 -% c);
                    sad[5] = vabs(p_m1m2 -% c) +% vabs(p_p1p2 -% c) +%
                        vabs(vshr(p_m1m1 +% p_0m1, 1) -% c) +% vabs(vshr(p_p1p1 +% p_0p1, 1) -% c);
                    sad[2] = vabs(p_0m1 -% c) +% vabs(p_0p1 -% c) +% vabs(p_0m2 -% c) +% vabs(p_0p2 -% c);
                    sad[6] = vabs(p_p1m2 -% c) +% vabs(p_m1p2 -% c) +%
                        vabs(vshr(p_0m1 +% p_p1m1, 1) -% c) +% vabs(vshr(p_0p1 +% p_m1p1, 1) -% c);
                    sad[3] = vabs(p_p1m1 -% c) +% vabs(p_m1p1 -% c) +% vabs(p_p2m2 -% c) +% vabs(p_m2p2 -% c);
                    sad[7] = vabs(p_p2m1 -% c) +% vabs(p_m2p1 -% c) +%
                        vabs(vshr(p_p1m1 +% p_p10, 1) -% c) +% vabs(vshr(p_m1p1 +% p_m10, 1) -% c);
                    // inner 2-tap sums shared by both weight groups
                    const q0 = p_m10 +% p_p10;
                    const q1 = p_m1m1 +% p_p1p1;
                    const q2 = p_0m1 +% p_0p1;
                    const q3 = p_p1m1 +% p_m1p1;
                    const s0 = (p_m20 +% p_p20) +% q0;
                    const s1 = (p_m2m2 +% p_p2p2) +% q1;
                    const s2 = (p_0m2 +% p_0p2) +% q2;
                    const s3 = (p_p2m2 +% p_m2p2) +% q3;
                    const t4 = p_m2m1 +% p_p2p1;
                    const t5 = p_m1m2 +% p_p1p2;
                    const t6 = p_p1m2 +% p_m1p2;
                    const t7 = p_p2m1 +% p_m2p1;
                    xs = .{
                        s0 +% s0,                 s1 +% s1,                 s2 +% s2,                 s3 +% s3,
                        (t4 +% t4) +% (q1 +% q0), (t5 +% t5) +% (q1 +% q2), (t6 +% t6) +% (q3 +% q2), (t7 +% t7) +% (q3 +% q0),
                    };
                }

                // strict < keeps the first minimum, as the old direction index did
                var bv = sad[0];
                var bx = xs[0];
                inline for (1..8) |id| {
                    const lt = sad[id] < bv;
                    bx = @select(I, lt, xs[id], bx);
                    bv = @select(I, lt, sad[id], bv);
                }
                const out = @select(I, bv == vsplat(0), c, blendX(c, bx, coef1, strength, rnd, sh)); // flat keeps c
                if (x + VL <= w) {
                    vstore(orow, x, out);
                } else {
                    const tail: [VL]I = out;
                    @memcpy(orow[x..w], tail[0 .. w - x]);
                }
            }
        }

        // One vertical lifting step of the 5/3 integer wavelet on a row:
        // out = base -/+ ((a + b) >> sh). Forward predict is (sub, 1), forward
        // update (add, 2), inverse update (sub, 2), inverse predict (add, 1);
        // process() resolves the row edge mirrors when picking a and b.
        fn liftRow(comptime sub: bool, comptime sh: comptime_int, base: []const I, a: []const I, b: []const I, out: []I) void {
            const w = out.len;
            var x: usize = 0;
            while (x + VL <= w) : (x += VL) {
                const t = vshr(vload(a, x) +% vload(b, x), sh);
                vstore(out, x, if (sub) vload(base, x) -% t else vload(base, x) +% t);
            }
            while (x < w) : (x += 1) {
                const t = (a[x] +% b[x]) >> sh;
                out[x] = if (sub) base[x] -% t else base[x] +% t;
            }
        }

        // The wavelet edge clamps only fire at the first/last element of a
        // row (Create guarantees w >= 4), so the vector bodies below are
        // branch-free: i == 0 is peeled, the right edge stays in the scalar
        // tails, and the vector trip bounds keep every lane unclamped and
        // every (laundered, overlapping) load in bounds.
        fn fwdH(in: []const I, ha: []I, hd: []I, w: usize, rows: usize, comptime need_ha: bool) void {
            std.debug.assert(w >= 4);
            const naw = (w + 1) / 2;
            const ndw = w / 2;
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const row = in[r * w ..];
                const hdr = hd[r * ndw ..];
                var i: usize = 0;
                if (comptime use_hvec) {
                    while (i + HVL < ndw) : (i += HVL) {
                        const eo = hDeinterleave(hload(row, 2 * i), hload(row, 2 * i + HVL));
                        const e2 = hEven(hload(row, 2 * i + 2), hload(row, 2 * i + 2 + HVL));
                        hstore(hdr, i, eo[1] -% hshr(eo[0] +% e2, 1));
                    }
                }
                while (i < ndw) : (i += 1) {
                    const c0 = row[2 * i];
                    const c2 = row[if (2 * i + 2 < w) 2 * i + 2 else w - 2];
                    hdr[i] = row[2 * i + 1] -% ((c0 +% c2) >> 1);
                }
                if (comptime !need_ha) continue;
                const har = ha[r * naw ..];
                i = 0;
                if (comptime use_hvec) {
                    har[0] = row[0] +% ((hdr[0] +% hdr[0]) >> 2);
                    i = 1;
                    while (i + HVL <= ndw) : (i += HVL) {
                        const e = hEven(hload(row, 2 * i), hload(row, 2 * i + HVL));
                        hstore(har, i, e +% hshr(hload(hdr, i - 1) +% hload(hdr, i), 2));
                    }
                }
                while (i < naw) : (i += 1) {
                    const il = if (i >= 1) i - 1 else 0;
                    const ir = if (i < ndw) i else ndw - 1;
                    har[i] = row[2 * i] +% ((hdr[il] +% hdr[ir]) >> 2);
                }
            }
        }

        fn invH(ha: []const I, hd: []const I, out: []I, w: usize, rows: usize) void {
            std.debug.assert(w >= 4);
            const naw = (w + 1) / 2;
            const ndw = w / 2;
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const har = ha[r * naw ..];
                const hdr = hd[r * ndw ..];
                const orow = out[r * w ..];
                var i: usize = 0;
                if (comptime use_hvec) {
                    orow[0] = har[0] -% ((hdr[0] +% hdr[0]) >> 2);
                    i = 1;
                    while (i + HVL < ndw) : (i += HVL) {
                        // Odd outputs need the even outputs one lane to the
                        // right; e2 recomputes them from source (identical
                        // ops on identical inputs, so identical values)
                        // instead of carrying a lane-shift across blocks.
                        const dl = hload(hdr, i - 1);
                        const d0 = hload(hdr, i);
                        const dr = hload(hdr, i + 1);
                        const e = hload(har, i) -% hshr(dl +% d0, 2);
                        const e2 = hload(har, i + 1) -% hshr(d0 +% dr, 2);
                        const o = d0 +% hshr(e +% e2, 1);
                        const lohi = hInterleave(e, o);
                        hstore(orow, 2 * i, lohi[0]);
                        hstore(orow, 2 * i + HVL, lohi[1]);
                    }
                }
                var j: usize = i;
                while (j < naw) : (j += 1) {
                    const il = if (j >= 1) j - 1 else 0;
                    const ir = if (j < ndw) j else ndw - 1;
                    orow[2 * j] = har[j] -% ((hdr[il] +% hdr[ir]) >> 2);
                }
                // odd 0 reads even 1, so it peels after the even tail
                if (comptime use_hvec) orow[1] = hdr[0] +% ((orow[0] +% orow[2]) >> 1);
                j = i;
                while (j < ndw) : (j += 1) {
                    const xl = orow[2 * j];
                    const xr = orow[if (2 * j + 2 < w) 2 * j + 2 else w - 2];
                    orow[2 * j + 1] = hdr[j] +% ((xl +% xr) >> 1);
                }
            }
        }

        // 12-bit fixed point -> output row (vectorized; scalar tail)
        fn storeRow(src: []const I, drow: [*]T, max_val: I) void {
            const w = src.len;
            const lo: V = @splat(0);
            const hi: V = @splat(max_val);
            var x: usize = 0;
            while (x + VL <= w) : (x += VL) {
                const v = @min(@max(vshr(vload(src, x) +% vsplat(8), 4), lo), hi);
                drow[x..][0..VL].* = simd.intCastSignedToUnsigned(VT, v);
            }
            while (x < w) : (x += 1) {
                var v = (src[x] +% 8) >> 4;
                if (v < 0) v = 0;
                if (v > max_val) v = max_val;
                drow[x] = @intCast(v);
            }
        }

        // Ring of the 12-bit source rows with the 2-sample mirror border.
        // Padded row q (= y + 2, -2 <= y <= h + 1) lives in slot q % n and
        // mirrors like the full padded plane did (-2 -> 2, h -> h-2, ...);
        // its columns hold the row at +2, mirrored at both ends, and the
        // round-up slack to wv + 4 repeats the last sample.
        const PadRing = struct {
            const n = 8; // live window: q in [2j+1, 2j+6] per output pair j
            buf: []I,
            pw: usize,
            w: usize,
            h: usize,
            srcp: [*]const T,
            stride: usize,
            next: usize = 0,

            fn row(self: *const PadRing, q: usize) []const I {
                return self.buf[(q % n) * self.pw ..][0..self.pw];
            }
            // source row y, unpadded
            fn data(self: *const PadRing, y: usize) []const I {
                return self.row(y + 2)[2..][0..self.w];
            }
            fn fillTo(self: *PadRing, q_end: usize) void {
                const w = self.w;
                while (self.next < q_end) : (self.next += 1) {
                    const q = self.next;
                    const ys = if (q < 2) 2 - q else if (q >= self.h + 2) 2 * self.h - q else q - 2;
                    const srow = self.srcp[ys * self.stride ..];
                    const dst = self.buf[(q % n) * self.pw ..][0..self.pw];
                    var x: usize = 0;
                    while (x + VL <= w) : (x += VL) {
                        const s: VT = srow[x..][0..VL].*;
                        dst[2 + x ..][0..VL].* = @as(V, @intCast(s)) << @as(VShift, @splat(4));
                    }
                    while (x < w) : (x += 1)
                        dst[2 + x] = @as(I, @intCast(srow[x])) << 4;
                    dst[0] = dst[4]; // col -2 -> +2
                    dst[1] = dst[3]; // col -1 -> +1
                    dst[w + 2] = dst[w]; // col w  -> w-2
                    dst[w + 3] = dst[w - 1]; // col w+1 -> w-3
                    for (dst[w + 4 ..]) |*e| e.* = dst[w - 1]; // round-up slack (dir ignored)
                }
            }
        };

        // Row-streamed pipeline: smooth -> forward 5/3 of the smoothed and
        // the original plane -> recombine -> inverse -> store, all through
        // small row rings. The whole-plane form kept ~12 full i16 planes per
        // frame (~25 MB at 1080p u8) and went DRAM-bound with every frame
        // worker active (16T slower than 8T). Each row is computed by the
        // same integer ops from the same inputs, so the output is identical.
        fn stream(pr: *PadRing, dstp: [*]T, stride: usize, strength: i32, restore: i32, max_val: I, wv: usize, a: std.mem.Allocator, comptime radius: u32) !void {
            const w = pr.w;
            const h = pr.h;
            const blur = try a.alloc(I, 4 * w); // blur row y in slot y % 4

            if (restore == 0) {
                for (0..h) |y| {
                    pr.fillTo(y + 5);
                    const orow = blur[0..w];
                    smoothRow(.{ pr.row(y), pr.row(y + 1), pr.row(y + 2), pr.row(y + 3), pr.row(y + 4) }, orow, wv, strength, radius);
                    storeRow(orow, dstp + y * stride, max_val);
                }
                return;
            }

            const na = @divCeil(h, 2);
            const nd = h / 2;
            const naw = @divCeil(w, 2);
            const ndw = w / 2;
            const vd_b = try a.alloc(I, 2 * w); // vertical detail rows j % 2
            const vd_o = try a.alloc(I, 2 * w);
            const even = try a.alloc(I, 2 * w); // reconstructed even rows j % 2
            const va = try a.alloc(I, w); // vertical approximation, then va_rec
            const odd = try a.alloc(I, w);
            const ll_o = try a.alloc(I, naw);
            const ll_b = try a.alloc(I, naw);
            const hd_b = try a.alloc(I, ndw);
            const hd_o = try a.alloc(I, ndw); // discarded

            const R = struct {
                inline fn at(buf: []I, i: usize, len: usize, comptime slots: usize) []I {
                    return buf[(i % slots) * len ..][0..len];
                }
            };

            var by: usize = 0; // next blur row to produce
            for (0..na) |j| {
                const last = @min(2 * j + 2, h - 1);
                while (by <= last) : (by += 1) {
                    pr.fillTo(by + 5);
                    smoothRow(.{ pr.row(by), pr.row(by + 1), pr.row(by + 2), pr.row(by + 3), pr.row(by + 4) }, R.at(blur, by, w, 4), wv, strength, radius);
                }

                if (j < nd) {
                    const r2 = if (2 * j + 2 < h) 2 * j + 2 else h - 2;
                    liftRow(true, 1, R.at(blur, 2 * j + 1, w, 4), R.at(blur, 2 * j, w, 4), R.at(blur, r2, w, 4), R.at(vd_b, j, w, 2));
                    liftRow(true, 1, pr.data(2 * j + 1), pr.data(2 * j), pr.data(r2), R.at(vd_o, j, w, 2));
                }
                const jl = if (j >= 1) j - 1 else 0;
                const jr = if (j < nd) j else nd - 1;

                // horizontal split of both approximation rows
                liftRow(false, 2, R.at(blur, 2 * j, w, 4), R.at(vd_b, jl, w, 2), R.at(vd_b, jr, w, 2), va);
                if (restore != 128) fwdH(va, ll_b, hd_b, w, 1, true) else fwdH(va, ll_b, hd_b, w, 1, false);
                liftRow(false, 2, pr.data(2 * j), R.at(vd_o, jl, w, 2), R.at(vd_o, jr, w, 2), va);
                fwdH(va, ll_o, hd_o, w, 1, true);

                // restore == 128 (default) keeps ll = ll_o untouched, so the
                // smoothed approximation band is never computed.
                if (restore != 128) {
                    const inv: i32 = 128 - restore;
                    for (ll_o, ll_b) |*o, b| {
                        o.* = @intCast((restore * w32(o.*) + inv * w32(b) + 64) >> 7);
                    }
                }

                invH(ll_o, hd_b, va, w, 1);
                const e = R.at(even, j, w, 2);
                liftRow(true, 2, va, R.at(vd_b, jl, w, 2), R.at(vd_b, jr, w, 2), e);
                storeRow(e, dstp + 2 * j * stride, max_val);
                if (j >= 1) {
                    liftRow(false, 1, R.at(vd_b, j - 1, w, 2), R.at(even, j - 1, w, 2), e, odd);
                    storeRow(odd, dstp + (2 * j - 1) * stride, max_val);
                }
            }
            if (nd == na) { // even h: odd row h-1 mirrors its lower even neighbour (h -> h-2)
                const e = R.at(even, nd - 1, w, 2);
                liftRow(false, 1, R.at(vd_b, nd - 1, w, 2), e, e, odd);
                storeRow(odd, dstp + (h - 1) * stride, max_val);
            }
        }

        pub fn process(
            noalias dstp: [*]T,
            stride: usize,
            noalias srcp: [*]const T,
            width: usize,
            height: usize,
            strength: i32,
            restore: i32,
            radius: u32,
            bits: u6,
            _: bool,
            alloc: std.mem.Allocator,
        ) !void {
            const max_val: I = @intCast((@as(i32, 1) << @as(u5, @intCast(bits))) - 1);

            if (strength == 0) {
                var y: usize = 0;
                while (y < height) : (y += 1)
                    @memcpy(dstp[y * stride ..][0..width], srcp[y * stride ..][0..width]);
                return;
            }

            // alloc is the caller's per-frame arena, reused across frames;
            // every buffer is fully written before it is read, so the reuse
            // is deterministic.
            const wv = @divCeil(width, VL) * VL;
            const pw = wv + 4;
            var pr: PadRing = .{
                .buf = try alloc.alloc(I, PadRing.n * pw),
                .pw = pw,
                .w = width,
                .h = height,
                .srcp = srcp,
                .stride = stride,
            };
            if (radius == 1) {
                try stream(&pr, dstp, stride, strength, restore, max_val, wv, alloc, 1);
            } else {
                try stream(&pr, dstp, stride, strength, restore, max_val, wv, alloc, 2);
            }
        }
    };
}
