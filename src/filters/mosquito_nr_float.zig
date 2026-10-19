const std = @import("std");
const simd = @import("simd.zig");

pub const MosquitoNRFloat = struct {
    const I = f32;
    const VL = std.simd.suggestVectorLength(f32) orelse 8;
    const V = @Vector(VL, f32);
    const VI = @Vector(VL, i32);

    // Laundered: same +-1/+-2 sliding-window fusion as the int variant.
    inline fn vload(s: []const I, i: usize) V {
        return simd.loaduOpaque(I, VL, s[i..][0..VL]);
    }

    // fwdH/invH vector regime: engaged whenever the shared f32 lane helpers
    // have a pinned-asm path at this VL (8-lane AVX2 or 16-lane AVX-512). The
    // scalar loops stay in place as the semantic reference and fallback.
    const use_hvec = simd.f32LanesAccelerated(VL);

    inline fn interleave(e: V, o: V) [2]V {
        return simd.interleaveF32(VL, e, o);
    }
    inline fn vstore(s: []I, i: usize, v: V) void {
        s[i..][0..VL].* = v;
    }
    inline fn vhalf(v: V) V {
        return v * @as(V, @splat(0.5));
    }

    // ---- direction-aware smoothing ------------------------------------------

    // Every candidate has the shape ((k*c + A) + k2*S) * inv with k/inv
    // fixed per weight group (dirs 0-3: coef0, inv_lo; 4-7: coef1, inv_hi),
    // S the direction's tap sum and A = k3*T the radius-2 outer pair of the
    // 1/4-weight group. The argmin picks S and T alongside the SAD and the
    // candidate is evaluated once per pixel. Bit-exact against the
    // per-direction expressions: S/T keep their left-to-right add order, the
    // products are the same ops on the same operands, and the 1/2-weight
    // group's A is k3 * -0.0 = -0.0 (k3 = 2*strength > 0), which is an exact
    // additive identity for every value including +-0 and NaN.
    fn smoothRow(pr: [5][]const I, orow: []I, wv: usize, strength: i32, comptime radius: u32) void {
        const s: f32 = @floatFromInt(strength);
        const k0: V = @splat(if (radius == 1) 64 - 2 * s else 128 - 4 * s);
        const k1: V = @splat(if (radius == 1) 128 - 4 * s else 256 - 8 * s);
        const k2: V = @splat(s);
        const k3: V = @splat(2 * s);
        // divisor reciprocals (radius 1: 64/128, radius 2: 128/256)
        const ilo: V = @splat(if (radius == 1) 1.0 / 64.0 else 1.0 / 128.0);
        const ihi: V = @splat(if (radius == 1) 1.0 / 128.0 else 1.0 / 256.0);
        const nz: V = @splat(-0.0);

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
            var ss: [8]V = undefined;
            var ts: [8]V = @splat(nz);
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
                sad[0] = @abs(p_m10 - c) + @abs(p_p10 - c);
                sad[4] = @abs(vhalf(p_m10 + p_m1m1) - c) + @abs(vhalf(p_p10 + p_p1p1) - c);
                sad[1] = @abs(p_m1m1 - c) + @abs(p_p1p1 - c);
                sad[5] = @abs(vhalf(p_m1m1 + p_0m1) - c) + @abs(vhalf(p_p1p1 + p_0p1) - c);
                sad[2] = @abs(p_0m1 - c) + @abs(p_0p1 - c);
                sad[6] = @abs(vhalf(p_0m1 + p_p1m1) - c) + @abs(vhalf(p_0p1 + p_m1p1) - c);
                sad[3] = @abs(p_p1m1 - c) + @abs(p_m1p1 - c);
                sad[7] = @abs(vhalf(p_p10 + p_p1m1) - c) + @abs(vhalf(p_m10 + p_m1p1) - c);
                ss = .{
                    p_m10 + p_p10,
                    p_m1m1 + p_p1p1,
                    p_0m1 + p_0p1,
                    p_p1m1 + p_m1p1,
                    p_m1m1 + p_m10 + p_p10 + p_p1p1,
                    p_m1m1 + p_0m1 + p_0p1 + p_p1p1,
                    p_p1m1 + p_0m1 + p_0p1 + p_m1p1,
                    p_p1m1 + p_p10 + p_m10 + p_m1p1,
                };
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
                sad[0] = @abs(p_m10 - c) + @abs(p_p10 - c) + @abs(p_m20 - c) + @abs(p_p20 - c);
                sad[4] = @abs(p_m2m1 - c) + @abs(p_p2p1 - c) +
                    @abs(vhalf(p_m10 + p_m1m1) - c) + @abs(vhalf(p_p10 + p_p1p1) - c);
                sad[1] = @abs(p_m1m1 - c) + @abs(p_p1p1 - c) + @abs(p_m2m2 - c) + @abs(p_p2p2 - c);
                sad[5] = @abs(p_m1m2 - c) + @abs(p_p1p2 - c) +
                    @abs(vhalf(p_m1m1 + p_0m1) - c) + @abs(vhalf(p_p1p1 + p_0p1) - c);
                sad[2] = @abs(p_0m1 - c) + @abs(p_0p1 - c) + @abs(p_0m2 - c) + @abs(p_0p2 - c);
                sad[6] = @abs(p_p1m2 - c) + @abs(p_m1p2 - c) +
                    @abs(vhalf(p_0m1 + p_p1m1) - c) + @abs(vhalf(p_0p1 + p_m1p1) - c);
                sad[3] = @abs(p_p1m1 - c) + @abs(p_m1p1 - c) + @abs(p_p2m2 - c) + @abs(p_m2p2 - c);
                sad[7] = @abs(p_p2m1 - c) + @abs(p_m2p1 - c) +
                    @abs(vhalf(p_p1m1 + p_p10) - c) + @abs(vhalf(p_m1p1 + p_m10) - c);
                ss = .{
                    p_m20 + p_m10 + p_p10 + p_p20,
                    p_m2m2 + p_m1m1 + p_p1p1 + p_p2p2,
                    p_0m2 + p_0m1 + p_0p1 + p_0p2,
                    p_p2m2 + p_p1m1 + p_m1p1 + p_m2p2,
                    p_m1m1 + p_m10 + p_p10 + p_p1p1,
                    p_m1m1 + p_0m1 + p_0p1 + p_p1p1,
                    p_p1m1 + p_0m1 + p_0p1 + p_m1p1,
                    p_p1m1 + p_p10 + p_m10 + p_m1p1,
                };
                ts[4] = p_m2m1 + p_p2p1;
                ts[5] = p_m1m2 + p_p1p2;
                ts[6] = p_p1m2 + p_m1p2;
                ts[7] = p_p2m1 + p_m2p1;
            }

            // strict < keeps the first minimum, as the old direction index did
            var bv = sad[0];
            var bs = ss[0];
            var bt = nz;
            var bi: VI = @splat(0);
            inline for (1..8) |id| {
                const lt = sad[id] < bv;
                bs = @select(f32, lt, ss[id], bs);
                if (radius != 1 and id >= 4) bt = @select(f32, lt, ts[id], bt);
                bi = @select(i32, lt, @as(VI, @splat(@as(i32, id))), bi);
                bv = @select(f32, lt, sad[id], bv);
            }
            const hi = bi >= @as(VI, @splat(4));
            const kc = @select(f32, hi, k1, k0) * c;
            const num = if (radius == 1) kc + k2 * bs else (kc + k3 * bt) + k2 * bs;
            const res = num * @select(f32, hi, ihi, ilo);
            const out = @select(f32, bv == @as(V, @splat(0.0)), c, res); // flat keeps c
            if (x + VL <= w) {
                vstore(orow, x, out);
            } else {
                const tail: [VL]I = out;
                @memcpy(orow[x..w], tail[0 .. w - x]);
            }
        }
    }

    // ---- separable CDF 5/3 wavelet (exact /2 and /4 in float) ---------------

    // One vertical lifting step on a row: out = base -/+ (a + b) * k. Forward
    // predict is (sub, 0.5), forward update (add, 0.25), inverse update
    // (sub, 0.25), inverse predict (add, 0.5); process() resolves the row
    // edge mirrors when picking a and b.
    fn liftRow(comptime sub: bool, comptime k: f32, base: []const I, a: []const I, b: []const I, out: []I) void {
        const w = out.len;
        var x: usize = 0;
        while (x + VL <= w) : (x += VL) {
            const t = (vload(a, x) + vload(b, x)) * @as(V, @splat(k));
            vstore(out, x, if (sub) vload(base, x) - t else vload(base, x) + t);
        }
        while (x < w) : (x += 1) {
            const t = (a[x] + b[x]) * k;
            out[x] = if (sub) base[x] - t else base[x] + t;
        }
    }

    // Same structure as the int core's fwdH/invH: edge clamps only fire at
    // the first/last element (Create guarantees w >= 4), so i == 0 is peeled,
    // the right edge stays in the scalar tails, and every vector lane
    // computes the exact scalar expression op-for-op (bit-exact; no
    // accumulation across elements).
    fn fwdH(in: []const I, ha: []I, hd: []I, w: usize, rows: usize, comptime need_ha: bool) void {
        std.debug.assert(w >= 4);
        const naw = (w + 1) / 2;
        const ndw = w / 2;
        const quarter: V = @splat(0.25);
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const row = in[r * w ..];
            const hdr = hd[r * ndw ..];
            var i: usize = 0;
            if (comptime use_hvec) {
                while (i + VL < ndw) : (i += VL) {
                    const a = vload(row, 2 * i);
                    const b = vload(row, 2 * i + VL);
                    const e = simd.evenLanesF32(VL, a, b);
                    const o = simd.oddLanesF32(VL, a, b);
                    const e2 = simd.evenLanesF32(VL, vload(row, 2 * i + 2), vload(row, 2 * i + 2 + VL));
                    vstore(hdr, i, o - vhalf(e + e2));
                }
            }
            while (i < ndw) : (i += 1) {
                const c0 = row[2 * i];
                const c2 = row[if (2 * i + 2 < w) 2 * i + 2 else w - 2];
                hdr[i] = row[2 * i + 1] - (c0 + c2) * 0.5;
            }
            if (comptime !need_ha) continue;
            const har = ha[r * naw ..];
            i = 0;
            if (comptime use_hvec) {
                har[0] = row[0] + (hdr[0] + hdr[0]) * 0.25;
                i = 1;
                while (i + VL <= ndw) : (i += VL) {
                    const e = simd.evenLanesF32(VL, vload(row, 2 * i), vload(row, 2 * i + VL));
                    vstore(har, i, e + (vload(hdr, i - 1) + vload(hdr, i)) * quarter);
                }
            }
            while (i < naw) : (i += 1) {
                const il = if (i >= 1) i - 1 else 0;
                const ir = if (i < ndw) i else ndw - 1;
                har[i] = row[2 * i] + (hdr[il] + hdr[ir]) * 0.25;
            }
        }
    }

    fn invH(ha: []const I, hd: []const I, out: []I, w: usize, rows: usize) void {
        std.debug.assert(w >= 4);
        const naw = (w + 1) / 2;
        const ndw = w / 2;
        const quarter: V = @splat(0.25);
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const har = ha[r * naw ..];
            const hdr = hd[r * ndw ..];
            const orow = out[r * w ..];
            var i: usize = 0;
            if (comptime use_hvec) {
                orow[0] = har[0] - (hdr[0] + hdr[0]) * 0.25;
                i = 1;
                while (i + VL < ndw) : (i += VL) {
                    // Odd outputs need the even outputs one lane to the
                    // right; e2 recomputes them from source (identical ops
                    // on identical inputs, so identical bits) instead of
                    // carrying a lane-shift across blocks.
                    const dl = vload(hdr, i - 1);
                    const d0 = vload(hdr, i);
                    const dr = vload(hdr, i + 1);
                    const e = vload(har, i) - (dl + d0) * quarter;
                    const e2 = vload(har, i + 1) - (d0 + dr) * quarter;
                    const o = d0 + vhalf(e + e2);
                    const lohi = interleave(e, o);
                    vstore(orow, 2 * i, lohi[0]);
                    vstore(orow, 2 * i + VL, lohi[1]);
                }
            }
            var j: usize = i;
            while (j < naw) : (j += 1) {
                const il = if (j >= 1) j - 1 else 0;
                const ir = if (j < ndw) j else ndw - 1;
                orow[2 * j] = har[j] - (hdr[il] + hdr[ir]) * 0.25;
            }
            // odd 0 reads even 1, so it peels after the even tail
            if (comptime use_hvec) orow[1] = hdr[0] + (orow[0] + orow[2]) * 0.5;
            j = i;
            while (j < ndw) : (j += 1) {
                const xl = orow[2 * j];
                const xr = orow[if (2 * j + 2 < w) 2 * j + 2 else w - 2];
                orow[2 * j + 1] = hdr[j] + (xl + xr) * 0.5;
            }
        }
    }

    // ---- public entry --------------------------------------------------------

    // store with clamp to the valid float range: luma [0, 1], chroma [-0.5, 0.5]
    fn storeRow(src: []const I, drow: [*]f32, lo_s: f32, hi_s: f32) void {
        const w = src.len;
        const lo: V = @splat(lo_s);
        const hi: V = @splat(hi_s);
        var x: usize = 0;
        while (x + VL <= w) : (x += VL)
            drow[x..][0..VL].* = @min(@max(vload(src, x), lo), hi);
        while (x < w) : (x += 1)
            drow[x] = @min(@max(src[x], lo_s), hi_s);
    }

    // Ring of reflection-padded source rows, same layout and mirroring as
    // the int core's PadRing (no scaling: [2 left pad][w][2 right pad][slack]).
    const PadRing = struct {
        const n = 8; // live window: q in [2j+1, 2j+6] per output pair j
        buf: []I,
        pw: usize,
        w: usize,
        h: usize,
        srcp: [*]const f32,
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
                const dst = self.buf[(q % n) * self.pw ..][0..self.pw];
                @memcpy(dst[2..][0..w], self.srcp[ys * self.stride ..][0..w]);
                dst[0] = dst[4];
                dst[1] = dst[3];
                dst[w + 2] = dst[w];
                dst[w + 3] = dst[w - 1];
                for (dst[w + 4 ..]) |*e| e.* = dst[w - 1];
            }
        }
    };

    // Row-streamed pipeline, same schedule as the int core's stream() (see
    // there): every row is the same float ops on the same inputs as the
    // whole-plane form, so the output is bit-identical.
    fn stream(pr: *PadRing, dstp: [*]f32, stride: usize, strength: i32, restore: i32, lo_s: f32, hi_s: f32, wv: usize, a: std.mem.Allocator, comptime radius: u32) !void {
        const w = pr.w;
        const h = pr.h;
        const blur = try a.alloc(I, 4 * w); // blur row y in slot y % 4

        if (restore == 0) {
            for (0..h) |y| {
                pr.fillTo(y + 5);
                const orow = blur[0..w];
                smoothRow(.{ pr.row(y), pr.row(y + 1), pr.row(y + 2), pr.row(y + 3), pr.row(y + 4) }, orow, wv, strength, radius);
                storeRow(orow, dstp + y * stride, lo_s, hi_s);
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
        const wo: f32 = @as(f32, @floatFromInt(restore)) / 128.0;
        const wb: f32 = 1.0 - wo;

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
                liftRow(true, 0.5, R.at(blur, 2 * j + 1, w, 4), R.at(blur, 2 * j, w, 4), R.at(blur, r2, w, 4), R.at(vd_b, j, w, 2));
                liftRow(true, 0.5, pr.data(2 * j + 1), pr.data(2 * j), pr.data(r2), R.at(vd_o, j, w, 2));
            }
            const jl = if (j >= 1) j - 1 else 0;
            const jr = if (j < nd) j else nd - 1;

            liftRow(false, 0.25, R.at(blur, 2 * j, w, 4), R.at(vd_b, jl, w, 2), R.at(vd_b, jr, w, 2), va);
            if (restore != 128) fwdH(va, ll_b, hd_b, w, 1, true) else fwdH(va, ll_b, hd_b, w, 1, false);
            liftRow(false, 0.25, pr.data(2 * j), R.at(vd_o, jl, w, 2), R.at(vd_o, jr, w, 2), va);
            fwdH(va, ll_o, hd_o, w, 1, true);

            // restore == 128 (default) keeps ll = ll_o untouched, so the
            // smoothed approximation band is never computed.
            if (restore != 128) {
                for (ll_o, ll_b) |*o, b| o.* = wo * o.* + wb * b;
            }

            invH(ll_o, hd_b, va, w, 1);
            const e = R.at(even, j, w, 2);
            liftRow(true, 0.25, va, R.at(vd_b, jl, w, 2), R.at(vd_b, jr, w, 2), e);
            storeRow(e, dstp + 2 * j * stride, lo_s, hi_s);
            if (j >= 1) {
                liftRow(false, 0.5, R.at(vd_b, j - 1, w, 2), R.at(even, j - 1, w, 2), e, odd);
                storeRow(odd, dstp + (2 * j - 1) * stride, lo_s, hi_s);
            }
        }
        if (nd == na) { // even h: odd row h-1 mirrors its lower even neighbour (h -> h-2)
            const e = R.at(even, nd - 1, w, 2);
            liftRow(false, 0.5, R.at(vd_b, nd - 1, w, 2), e, e, odd);
            storeRow(odd, dstp + (h - 1) * stride, lo_s, hi_s);
        }
    }

    pub fn process(
        noalias dstp: [*]f32,
        stride: usize, // elements
        noalias srcp: [*]const f32,
        width: usize,
        height: usize,
        strength: i32,
        restore: i32,
        radius: u32,
        _: u6,
        chroma: bool,
        alloc: std.mem.Allocator,
    ) !void {
        if (strength == 0) {
            var y: usize = 0;
            while (y < height) : (y += 1)
                @memcpy(dstp[y * stride ..][0..width], srcp[y * stride ..][0..width]);
            return;
        }

        // alloc is the caller's per-frame arena, reused across frames; every
        // buffer is fully written before it is read, so the reuse is
        // deterministic.
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
        const lo_s: f32 = if (chroma) -0.5 else 0.0;
        const hi_s: f32 = if (chroma) 0.5 else 1.0;
        if (radius == 1) {
            try stream(&pr, dstp, stride, strength, restore, lo_s, hi_s, wv, alloc, 1);
        } else {
            try stream(&pr, dstp, stride, strength, restore, lo_s, hi_s, wv, alloc, 2);
        }
    }
};
