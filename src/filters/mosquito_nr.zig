const std = @import("std");

pub fn MosquitoNR(comptime T: type) type {
    return struct {
        const I = if (T == u8) i16 else i32;
        const VL = std.simd.suggestVectorLength(I) orelse 8;
        const V = @Vector(VL, I);
        const VT = @Vector(VL, T);
        const VShift = @Vector(VL, std.math.Log2Int(I));

        inline fn vload(s: []const I, i: usize) V {
            return s[i..][0..VL].*;
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

        fn smooth(pl: []const I, pw: usize, dirs: []I, blur: []I, wv: usize, w: usize, h: usize, strength: i32, comptime radius: u32) void {
            const G = struct {
                inline fn p(row: []const I, idx: usize) i32 {
                    return row[idx];
                }
            };
            const coef0: i32 = if (radius == 1) 64 - 2 * strength else 128 - 4 * strength;
            const coef1: i32 = if (radius == 1) 128 - 4 * strength else 256 - 8 * strength;
            const coef2: i32 = strength;

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
                        sad[0] = vabs(p_m10 -% c) +% vabs(p_p10 -% c);
                        sad[4] = vabs(vshr(p_m10 +% p_m1m1, 1) -% c) +% vabs(vshr(p_p10 +% p_p1p1, 1) -% c);
                        sad[1] = vabs(p_m1m1 -% c) +% vabs(p_p1p1 -% c);
                        sad[5] = vabs(vshr(p_m1m1 +% p_0m1, 1) -% c) +% vabs(vshr(p_p1p1 +% p_0p1, 1) -% c);
                        sad[2] = vabs(p_0m1 -% c) +% vabs(p_0p1 -% c);
                        sad[6] = vabs(vshr(p_0m1 +% p_p1m1, 1) -% c) +% vabs(vshr(p_0p1 +% p_m1p1, 1) -% c);
                        sad[3] = vabs(p_p1m1 -% c) +% vabs(p_m1p1 -% c);
                        sad[7] = vabs(vshr(p_p10 +% p_p1m1, 1) -% c) +% vabs(vshr(p_m10 +% p_m1p1, 1) -% c);
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
                    }

                    var bv = sad[0];
                    var bi: V = @splat(0);
                    inline for (1..8) |id| {
                        const lt = sad[id] < bv;
                        bi = @select(I, lt, vsplat(@as(I, id)), bi);
                        bv = @select(I, lt, sad[id], bv);
                    }
                    const flat = bv == vsplat(0);
                    vstore(dirs, x, @select(I, flat, vsplat(8), bi));
                }

                const orow = blur[y * w ..];
                var bx: usize = 0;
                if (radius == 1) {
                    while (bx < w) : (bx += 1) {
                        const cx = bx + 2;
                        const c = G.p(r0, cx);
                        orow[bx] = @intCast(switch (dirs[bx]) {
                            0 => (coef0 * c + coef2 * (G.p(r0, cx - 1) + G.p(r0, cx + 1)) + 32) >> 6,
                            1 => (coef0 * c + coef2 * (G.p(rm1, cx - 1) + G.p(rp1, cx + 1)) + 32) >> 6,
                            2 => (coef0 * c + coef2 * (G.p(rm1, cx) + G.p(rp1, cx)) + 32) >> 6,
                            3 => (coef0 * c + coef2 * (G.p(rm1, cx + 1) + G.p(rp1, cx - 1)) + 32) >> 6,
                            4 => (coef1 * c + coef2 * (G.p(rm1, cx - 1) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(rp1, cx + 1)) + 64) >> 7,
                            5 => (coef1 * c + coef2 * (G.p(rm1, cx - 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx + 1)) + 64) >> 7,
                            6 => (coef1 * c + coef2 * (G.p(rm1, cx + 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx - 1)) + 64) >> 7,
                            7 => (coef1 * c + coef2 * (G.p(rm1, cx + 1) + G.p(r0, cx + 1) + G.p(r0, cx - 1) + G.p(rp1, cx - 1)) + 64) >> 7,
                            else => c, // 8 == flat
                        });
                    }
                } else {
                    const coef3: i32 = 2 * strength;
                    while (bx < w) : (bx += 1) {
                        const cx = bx + 2;
                        const c = G.p(r0, cx);
                        orow[bx] = @intCast(switch (dirs[bx]) {
                            0 => (coef0 * c + coef2 * (G.p(r0, cx - 2) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(r0, cx + 2)) + 64) >> 7,
                            1 => (coef0 * c + coef2 * (G.p(rm2, cx - 2) + G.p(rm1, cx - 1) + G.p(rp1, cx + 1) + G.p(rp2, cx + 2)) + 64) >> 7,
                            2 => (coef0 * c + coef2 * (G.p(rm2, cx) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp2, cx)) + 64) >> 7,
                            3 => (coef0 * c + coef2 * (G.p(rm2, cx + 2) + G.p(rm1, cx + 1) + G.p(rp1, cx - 1) + G.p(rp2, cx - 2)) + 64) >> 7,
                            4 => (coef1 * c + coef3 * (G.p(rm1, cx - 2) + G.p(rp1, cx + 2)) + coef2 * (G.p(rm1, cx - 1) + G.p(r0, cx - 1) + G.p(r0, cx + 1) + G.p(rp1, cx + 1)) + 128) >> 8,
                            5 => (coef1 * c + coef3 * (G.p(rm2, cx - 1) + G.p(rp2, cx + 1)) + coef2 * (G.p(rm1, cx - 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx + 1)) + 128) >> 8,
                            6 => (coef1 * c + coef3 * (G.p(rm2, cx + 1) + G.p(rp2, cx - 1)) + coef2 * (G.p(rm1, cx + 1) + G.p(rm1, cx) + G.p(rp1, cx) + G.p(rp1, cx - 1)) + 128) >> 8,
                            7 => (coef1 * c + coef3 * (G.p(rm1, cx + 2) + G.p(rp1, cx - 2)) + coef2 * (G.p(rm1, cx + 1) + G.p(r0, cx + 1) + G.p(r0, cx - 1) + G.p(rp1, cx - 1)) + 128) >> 8,
                            else => c, // 8 == flat
                        });
                    }
                }
            }
        }

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
                    vstore(vd, j * w + x, vload(in, r1 + x) -% vshr(vload(in, r0 + x) +% vload(in, r2 + x), 1));
                while (x < w) : (x += 1)
                    vd[j * w + x] = in[r1 + x] -% ((in[r0 + x] +% in[r2 + x]) >> 1);
            }
            j = 0;
            while (j < na) : (j += 1) {
                const r0 = ibase + (2 * j) * istride;
                const jl = if (j >= 1) j - 1 else 0;
                const jr = if (j < nd) j else nd - 1;
                var x: usize = 0;
                while (x + VL <= w) : (x += VL)
                    vstore(va, j * w + x, vload(in, r0 + x) +% vshr(vload(vd, jl * w + x) +% vload(vd, jr * w + x), 2));
                while (x < w) : (x += 1)
                    va[j * w + x] = in[r0 + x] +% ((vd[jl * w + x] +% vd[jr * w + x]) >> 2);
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
                    vstore(out, orow + x, vload(va, j * w + x) -% vshr(vload(vd, jl * w + x) +% vload(vd, jr * w + x), 2));
                while (x < w) : (x += 1)
                    out[orow + x] = va[j * w + x] -% ((vd[jl * w + x] +% vd[jr * w + x]) >> 2);
            }
            j = 0;
            while (j < nd) : (j += 1) {
                const r0 = (2 * j) * w;
                const r2 = (if (2 * j + 2 < h) 2 * j + 2 else h - 2) * w;
                const orow = (2 * j + 1) * w;
                var x: usize = 0;
                while (x + VL <= w) : (x += VL)
                    vstore(out, orow + x, vload(vd, j * w + x) +% vshr(vload(out, r0 + x) +% vload(out, r2 + x), 1));
                while (x < w) : (x += 1)
                    out[orow + x] = vd[j * w + x] +% ((out[r0 + x] +% out[r2 + x]) >> 1);
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
                    hdr[i] = row[2 * i + 1] -% ((c0 +% c2) >> 1);
                }
                i = 0;
                while (i < naw) : (i += 1) {
                    const il = if (i >= 1) i - 1 else 0;
                    const ir = if (i < ndw) i else ndw - 1;
                    har[i] = row[2 * i] +% ((hdr[il] +% hdr[ir]) >> 2);
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
                    orow[2 * i] = har[i] -% ((hdr[il] +% hdr[ir]) >> 2);
                }
                i = 0;
                while (i < ndw) : (i += 1) {
                    const xl = orow[2 * i];
                    const xr = orow[if (2 * i + 2 < w) 2 * i + 2 else w - 2];
                    orow[2 * i + 1] = hdr[i] +% ((xl +% xr) >> 1);
                }
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

            const pw = wv + 4;
            const ph = h + 4;
            const pl_base = 2 * pw + 2;
            const pl = try a.alloc(I, pw * ph);
            {
                var y: usize = 0;
                while (y < h) : (y += 1) {
                    const srow = srcp[y * stride ..];
                    const dst = pl[(y + 2) * pw ..];
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
                    for (dst[w + 4 .. pw]) |*e| e.* = dst[w - 1]; // round-up slack (dir ignored)
                }
                // top/bottom reflected rows (whole rows incl. padding)
                @memcpy(pl[0..pw], pl[4 * pw ..][0..pw]); // row -2 -> +2
                @memcpy(pl[pw..][0..pw], pl[3 * pw ..][0..pw]); // row -1 -> +1
                @memcpy(pl[(h + 2) * pw ..][0..pw], pl[h * pw ..][0..pw]); // row h   -> h-2
                @memcpy(pl[(h + 3) * pw ..][0..pw], pl[(h - 1) * pw ..][0..pw]); // row h+1 -> h-3
            }

            const dirs = try a.alloc(I, wv); // small per-row scratch (stays in cache)
            if (radius == 1) {
                smooth(pl, pw, dirs, blur, wv, w, h, strength, 1);
            } else {
                smooth(pl, pw, dirs, blur, wv, w, h, strength, 2);
            }

            const out12 = blur; // reused below as the reconstruction target

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

                fwdV(pl, pw, pl_base, va_o, scratch_vd, w, h); // original: read pl interior
                fwdH(va_o, ll_o, scratch_hd, w, na_h);

                fwdV(blur, w, 0, va_b, vd_b, w, h); // smoothed: tight plane

                const ll: []const I = ll_o;
                fwdH(va_b, ll_b, hd_b, w, na_h);
                if (restore != 128) {
                    const inv: i32 = 128 - restore;
                    for (ll_o, ll_b) |*o, b| {
                        o.* = @intCast((restore * w32(o.*) + inv * w32(b) + 64) >> 7);
                    }
                }

                invH(ll, hd_b, va_rec, w, na_h);
                invV(va_rec, vd_b, out12, w, h);
            }

            // store 12-bit fixed point -> output (vectorized; scalar tail)
            {
                const lo: V = @splat(0);
                const hi: V = @splat(max_val);
                var y: usize = 0;
                while (y < h) : (y += 1) {
                    const drow = dstp[y * stride ..];
                    var x: usize = 0;
                    while (x + VL <= w) : (x += VL) {
                        const v = @min(@max(vshr(vload(out12, y * w + x) +% vsplat(8), 4), lo), hi);
                        drow[x..][0..VL].* = @as(VT, @intCast(v));
                    }
                    while (x < w) : (x += 1) {
                        var v = (out12[y * w + x] +% 8) >> 4;
                        if (v < 0) v = 0;
                        if (v > max_val) v = max_val;
                        drow[x] = @intCast(v);
                    }
                }
            }
        }
    };
}
