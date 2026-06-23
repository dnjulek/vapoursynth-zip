const std = @import("std");

pub const MosquitoNRFloat = struct {
    const I = f32;
    const VL = std.simd.suggestVectorLength(f32) orelse 8;
    const V = @Vector(VL, f32);
    const VI = @Vector(VL, i32);

    inline fn vload(s: []const I, i: usize) V {
        return s[i..][0..VL].*;
    }
    inline fn vstore(s: []I, i: usize, v: V) void {
        s[i..][0..VL].* = v;
    }
    inline fn vhalf(v: V) V {
        return v * @as(V, @splat(0.5));
    }

    // ---- direction-aware smoothing ------------------------------------------

    fn smooth(pl: []const I, pw: usize, dirs: []i32, blur: []I, wv: usize, w: usize, h: usize, strength: i32, comptime radius: u32) void {
        const G = struct {
            inline fn p(row: []const I, idx: usize) f32 {
                return row[idx];
            }
        };
        const s: f32 = @floatFromInt(strength);
        const coef0: f32 = if (radius == 1) 64 - 2 * s else 128 - 4 * s;
        const coef1: f32 = if (radius == 1) 128 - 4 * s else 256 - 8 * s;
        const coef2: f32 = s;
        // divisor reciprocals (radius 1: 64/128, radius 2: 128/256)
        const inv_lo: f32 = if (radius == 1) 1.0 / 64.0 else 1.0 / 128.0;
        const inv_hi: f32 = if (radius == 1) 1.0 / 128.0 else 1.0 / 256.0;

        var y: usize = 0;
        while (y < h) : (y += 1) {
            const cy = y + 2;
            const rm2 = pl[(cy - 2) * pw ..];
            const rm1 = pl[(cy - 1) * pw ..];
            const r0 = pl[cy * pw ..];
            const rp1 = pl[(cy + 1) * pw ..];
            const rp2 = pl[(cy + 2) * pw ..];

            // --- vectorized direction pass into the row-local dirs buffer ---
            var x: usize = 0;
            while (x < wv) : (x += VL) {
                const cx = x + 2;
                var sad: [8]V = undefined;

                if (radius == 1) {
                    const c = vload(r0, cx);
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
                } else {
                    const c = vload(r0, cx);
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
                }

                var bv = sad[0];
                var bi: VI = @splat(0);
                inline for (1..8) |id| {
                    const lt = sad[id] < bv;
                    bi = @select(i32, lt, @as(VI, @splat(@as(i32, id))), bi);
                    bv = @select(f32, lt, sad[id], bv);
                }
                const flat = bv == @as(V, @splat(0.0));
                dirs[x..][0..VL].* = @select(i32, flat, @as(VI, @splat(8)), bi);
            }

            // --- scalar blend pass (data-dependent) ---
            const orow = blur[y * w ..];
            var bx: usize = 0;
            if (radius == 1) {
                while (bx < w) : (bx += 1) {
                    const cx = bx + 2;
                    const c = G.p(r0, cx);
                    orow[bx] = switch (dirs[bx]) {
                        0 => (coef0 * c + coef2 * (G.p(r0, cx - 1) + G.p(r0, cx + 1))) * inv_lo,
                        1 => (coef0 * c + coef2 * (G.p(rm1, cx - 1) + G.p(rp1, cx + 1))) * inv_lo,
                        2 => (coef0 * c + coef2 * (G.p(rm1, cx) + G.p(rp1, cx))) * inv_lo,
                        3 => (coef0 * c + coef2 * (G.p(rm1, cx + 1) + G.p(rp1, cx - 1))) * inv_lo,
                        4 => (coef1 * c + coef2 * (G.p(rm1, cx - 1) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(rp1, cx + 1))) * inv_hi,
                        5 => (coef1 * c + coef2 * (G.p(rm1, cx - 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx + 1))) * inv_hi,
                        6 => (coef1 * c + coef2 * (G.p(rm1, cx + 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx - 1))) * inv_hi,
                        7 => (coef1 * c + coef2 * (G.p(rm1, cx + 1) + G.p(r0, cx + 1) + G.p(r0, cx - 1) + G.p(rp1, cx - 1))) * inv_hi,
                        else => c, // 8 == flat
                    };
                }
            } else {
                const coef3: f32 = 2 * s;
                while (bx < w) : (bx += 1) {
                    const cx = bx + 2;
                    const c = G.p(r0, cx);
                    orow[bx] = switch (dirs[bx]) {
                        0 => (coef0 * c + coef2 * (G.p(r0, cx - 2) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(r0, cx + 2))) * inv_lo,
                        1 => (coef0 * c + coef2 * (G.p(rm2, cx - 2) + G.p(rm1, cx - 1) + G.p(rp1, cx + 1) + G.p(rp2, cx + 2))) * inv_lo,
                        2 => (coef0 * c + coef2 * (G.p(rm2, cx) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp2, cx))) * inv_lo,
                        3 => (coef0 * c + coef2 * (G.p(rm2, cx + 2) + G.p(rm1, cx + 1) + G.p(rp1, cx - 1) + G.p(rp2, cx - 2))) * inv_lo,
                        4 => (coef1 * c + coef3 * (G.p(rm1, cx - 2) + G.p(rp1, cx + 2)) + coef2 * (G.p(rm1, cx - 1) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(rp1, cx + 1))) * inv_hi,
                        5 => (coef1 * c + coef3 * (G.p(rm2, cx - 1) + G.p(rp2, cx + 1)) + coef2 * (G.p(rm1, cx - 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx + 1))) * inv_hi,
                        6 => (coef1 * c + coef3 * (G.p(rm2, cx + 1) + G.p(rp2, cx - 1)) + coef2 * (G.p(rm1, cx + 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx - 1))) * inv_hi,
                        7 => (coef1 * c + coef3 * (G.p(rm1, cx + 2) + G.p(rp1, cx - 2)) + coef2 * (G.p(rm1, cx + 1) + G.p(r0, cx + 1) + G.p(r0, cx - 1) + G.p(rp1, cx - 1))) * inv_hi,
                        else => c, // 8 == flat
                    };
                }
            }
        }
    }

    // ---- separable CDF 5/3 wavelet (exact /2 and /4 in float) ---------------

    fn fwdV(in: []const I, istride: usize, ibase: usize, va: []I, vd: []I, w: usize, h: usize) void {
        const na = (h + 1) / 2;
        const nd = h / 2;
        var j: usize = 0;
        while (j < nd) : (j += 1) {
            const r0 = ibase + (2 * j) * istride;
            const r1 = ibase + (2 * j + 1) * istride;
            const r2 = ibase + (if (2 * j + 2 < h) 2 * j + 2 else h - 2) * istride;
            var x: usize = 0;
            while (x + VL <= w) : (x += VL)
                vstore(vd, j * w + x, vload(in, r1 + x) - vhalf(vload(in, r0 + x) + vload(in, r2 + x)));
            while (x < w) : (x += 1)
                vd[j * w + x] = in[r1 + x] - (in[r0 + x] + in[r2 + x]) * 0.5;
        }
        j = 0;
        while (j < na) : (j += 1) {
            const r0 = ibase + (2 * j) * istride;
            const jl = if (j >= 1) j - 1 else 0;
            const jr = if (j < nd) j else nd - 1;
            var x: usize = 0;
            while (x + VL <= w) : (x += VL)
                vstore(va, j * w + x, vload(in, r0 + x) + (vload(vd, jl * w + x) + vload(vd, jr * w + x)) * @as(V, @splat(0.25)));
            while (x < w) : (x += 1)
                va[j * w + x] = in[r0 + x] + (vd[jl * w + x] + vd[jr * w + x]) * 0.25;
        }
    }

    fn invV(va: []const I, vd: []const I, out: []I, w: usize, h: usize) void {
        const na = (h + 1) / 2;
        const nd = h / 2;
        var j: usize = 0;
        while (j < na) : (j += 1) {
            const jl = if (j >= 1) j - 1 else 0;
            const jr = if (j < nd) j else nd - 1;
            const orow = (2 * j) * w;
            var x: usize = 0;
            while (x + VL <= w) : (x += VL)
                vstore(out, orow + x, vload(va, j * w + x) - (vload(vd, jl * w + x) + vload(vd, jr * w + x)) * @as(V, @splat(0.25)));
            while (x < w) : (x += 1)
                out[orow + x] = va[j * w + x] - (vd[jl * w + x] + vd[jr * w + x]) * 0.25;
        }
        j = 0;
        while (j < nd) : (j += 1) {
            const r0 = (2 * j) * w;
            const r2 = (if (2 * j + 2 < h) 2 * j + 2 else h - 2) * w;
            const orow = (2 * j + 1) * w;
            var x: usize = 0;
            while (x + VL <= w) : (x += VL)
                vstore(out, orow + x, vload(vd, j * w + x) + vhalf(vload(out, r0 + x) + vload(out, r2 + x)));
            while (x < w) : (x += 1)
                out[orow + x] = vd[j * w + x] + (out[r0 + x] + out[r2 + x]) * 0.5;
        }
    }

    fn fwdH(in: []const I, ha: []I, hd: []I, w: usize, rows: usize) void {
        const naw = (w + 1) / 2;
        const ndw = w / 2;
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const row = in[r * w ..];
            const hdr = hd[r * ndw ..];
            const har = ha[r * naw ..];
            var i: usize = 0;
            while (i < ndw) : (i += 1) {
                const c0 = row[2 * i];
                const c2 = row[if (2 * i + 2 < w) 2 * i + 2 else w - 2];
                hdr[i] = row[2 * i + 1] - (c0 + c2) * 0.5;
            }
            i = 0;
            while (i < naw) : (i += 1) {
                const il = if (i >= 1) i - 1 else 0;
                const ir = if (i < ndw) i else ndw - 1;
                har[i] = row[2 * i] + (hdr[il] + hdr[ir]) * 0.25;
            }
        }
    }

    fn invH(ha: []const I, hd: []const I, out: []I, w: usize, rows: usize) void {
        const naw = (w + 1) / 2;
        const ndw = w / 2;
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const har = ha[r * naw ..];
            const hdr = hd[r * ndw ..];
            const orow = out[r * w ..];
            var i: usize = 0;
            while (i < naw) : (i += 1) {
                const il = if (i >= 1) i - 1 else 0;
                const ir = if (i < ndw) i else ndw - 1;
                orow[2 * i] = har[i] - (hdr[il] + hdr[ir]) * 0.25;
            }
            i = 0;
            while (i < ndw) : (i += 1) {
                const xl = orow[2 * i];
                const xr = orow[if (2 * i + 2 < w) 2 * i + 2 else w - 2];
                orow[2 * i + 1] = hdr[i] + (xl + xr) * 0.5;
            }
        }
    }

    // ---- public entry --------------------------------------------------------

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

        const w = width;
        const h = height;
        const na_h = (h + 1) / 2;
        const nd_h = h / 2;
        const na_w = (w + 1) / 2;
        const nd_w = w / 2;
        const wv = ((w + VL - 1) / VL) * VL;

        var arena_state = std.heap.ArenaAllocator.init(alloc);
        defer arena_state.deinit();
        const a = arena_state.allocator();

        const blur = try a.alloc(I, w * h);

        // reflection-padded original, straight from src (no scaling). Same layout
        // as the integer core: [2 left pad][w][2 right pad][round-up slack].
        const pw = wv + 4;
        const ph = h + 4;
        const pl_base = 2 * pw + 2;
        const pl = try a.alloc(I, pw * ph);
        {
            var y: usize = 0;
            while (y < h) : (y += 1) {
                const srow = srcp[y * stride ..];
                const dst = pl[(y + 2) * pw ..];
                @memcpy(dst[2..][0..w], srow[0..w]);
                dst[0] = dst[4];
                dst[1] = dst[3];
                dst[w + 2] = dst[w];
                dst[w + 3] = dst[w - 1];
                for (dst[w + 4 .. pw]) |*e| e.* = dst[w - 1];
            }
            @memcpy(pl[0..pw], pl[4 * pw ..][0..pw]);
            @memcpy(pl[pw..][0..pw], pl[3 * pw ..][0..pw]);
            @memcpy(pl[(h + 2) * pw ..][0..pw], pl[h * pw ..][0..pw]);
            @memcpy(pl[(h + 3) * pw ..][0..pw], pl[(h - 1) * pw ..][0..pw]);
        }

        const dirs = try a.alloc(i32, wv);
        if (radius == 1) smooth(pl, pw, dirs, blur, wv, w, h, strength, 1) else smooth(pl, pw, dirs, blur, wv, w, h, strength, 2);

        const out12 = blur;

        if (restore != 0) {
            const va_o = try a.alloc(I, na_h * w);
            const va_b = try a.alloc(I, na_h * w);
            const vd_b = try a.alloc(I, nd_h * w);
            const ll_o = try a.alloc(I, na_h * na_w);
            const ll_b = try a.alloc(I, na_h * na_w);
            const hd_b = try a.alloc(I, na_h * nd_w);
            const va_rec = try a.alloc(I, na_h * w);
            const scratch_vd = try a.alloc(I, nd_h * w);
            const scratch_hd = try a.alloc(I, na_h * nd_w);

            fwdV(pl, pw, pl_base, va_o, scratch_vd, w, h);
            fwdH(va_o, ll_o, scratch_hd, w, na_h);
            fwdV(blur, w, 0, va_b, vd_b, w, h);

            const ll: []const I = ll_o;
            fwdH(va_b, ll_b, hd_b, w, na_h);
            if (restore != 128) {
                const wo: f32 = @as(f32, @floatFromInt(restore)) / 128.0;
                const wb: f32 = 1.0 - wo;
                for (ll_o, ll_b) |*o, b| o.* = wo * o.* + wb * b;
            }

            invH(ll, hd_b, va_rec, w, na_h);
            invV(va_rec, vd_b, out12, w, h);
        }

        // store with clamp to the valid float range: luma [0, 1], chroma [-0.5, 0.5]
        {
            const lo_s: f32 = if (chroma) -0.5 else 0.0;
            const hi_s: f32 = if (chroma) 0.5 else 1.0;
            const lo: V = @splat(lo_s);
            const hi: V = @splat(hi_s);
            var y: usize = 0;
            while (y < h) : (y += 1) {
                const drow = dstp[y * stride ..];
                var x: usize = 0;
                while (x + VL <= w) : (x += VL)
                    drow[x..][0..VL].* = @min(@max(vload(out12, y * w + x), lo), hi);
                while (x < w) : (x += 1)
                    drow[x] = @min(@max(out12[y * w + x], lo_s), hi_s);
            }
        }
    }
};
