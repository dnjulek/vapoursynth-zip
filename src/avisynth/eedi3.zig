//! vszip_EEDI3 / vszip_EEDI3H(clip, int field, bool "dh", float "alpha", float "beta", float "gamma",
//!   int "nrad", int "mdis", bool "hp", int "vcheck", float "vthresh0", float "vthresh1",
//!   float "vthresh2", clip "sclip", clip "mclip")
//!
//! Edge-directed interpolation (EEDI3H: along the width). field 2/3 bob:
//! output has twice the frames and frame rate; frame n interpolates source
//! frame n/2, alternating field. The output is frame-based, `_FieldBased` is
//! set to 0 and `_DurationNum/_DurationDen` halve when bobbing, as in vszip.
//!
//! Scratch: vszip keyed per-thread scratch by thread id; here each request
//! checks a scratch block out of a mutex-guarded pool and returns it (exclusive
//! per request, no lock held while processing).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const core = @import("../filters/eedi3.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
const Scratch = core.Scratch;
const n_vec = core.n_vec;
const processPlane = core.processPlane;

const shared_params = "c[field]i[dh]b[alpha]f[beta]f[gamma]f[nrad]i[mdis]i[hp]b[vcheck]i[vthresh0]f[vthresh1]f[vthresh2]f[sclip]c[mclip]c";

pub const V = Impl(false);
pub const H = Impl(true);

const PoolEntry = struct { s: *Scratch, stride: u32 };

const Data = struct {
    format: avs.Format,
    k: core.Data,
    sclip: avs.OwnedClip = .{},
    mclip: avs.OwnedClip = .{},

    pool_lock: std.Io.Mutex = .init,
    pool: std.ArrayList(PoolEntry) = .empty,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |e| core.freeScratch(e.s);
        self.pool.deinit(allocator);
        self.sclip.deinit();
        self.mclip.deinit();
    }

    fn acquire(self: *Data, stride: u32, w: u32, n_interp: u32, srcT_rows: u32, dstT_rows: u32, elem: u32) !PoolEntry {
        {
            self.pool_lock.lockUncancelable(common.io);
            defer self.pool_lock.unlock(common.io);
            if (self.pool.pop()) |e| {
                if (e.stride >= stride) return e;
                core.freeScratch(e.s); // too small for this frame's geometry
            }
        }
        return .{ .s = try core.allocScratch(w, stride, n_interp, self.k.hp, srcT_rows, dstT_rows, elem), .stride = stride };
    }

    fn release(self: *Data, e: PoolEntry) void {
        self.pool_lock.lockUncancelable(common.io);
        defer self.pool_lock.unlock(common.io);
        self.pool.append(allocator, e) catch core.freeScratch(e.s);
    }
};

fn reduce(num: *u64, den: *u64) void {
    const g = std.math.gcd(num.*, den.*);
    if (g > 1) {
        num.* /= g;
        den.* /= g;
    }
}

fn Impl(comptime horizontal: bool) type {
    return struct {
        pub const filter_name = if (horizontal) "vszip_EEDI3H" else "vszip_EEDI3";
        pub const params = shared_params;
        const F = avs.Filter(Data, filter_name);

        fn Get(comptime T: type) type {
            return struct {
                fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
                    const env = ctx.env();
                    const k = &d.k;
                    const src_n: c_int = if (k.field > 1) @divTrunc(n, 2) else n;
                    const src = try ctx.childFrame(src_n);
                    defer src.release();
                    const scp: ?avs.Frame = if (k.vcheck > 0 and d.sclip.clip != null) try ctx.getFrame(d.sclip.get(), n) else null;
                    defer if (scp) |s| s.release();
                    const mcp: ?avs.Frame = if (d.mclip.clip != null) try ctx.getFrame(d.mclip.get(), src_n) else null;
                    defer if (mcp) |m| m.release();
                    const dst = try ctx.newFrame(src);
                    errdefer dst.release();
                    hz.copyAlpha(env, dst, src, d.format); // alpha is not interpolated; only when sizes match
                    var scratch_bufs: hz.Scratch = .{};
                    defer scratch_bufs.deinit();

                    var field: u8 = k.field & 1;
                    switch (src.propsRO(env).intOr("_FieldBased", 0)) {
                        1 => field = 0, // bottom field first
                        2 => field = 1, // top field first
                        else => {},
                    }
                    if (k.field > 1) field = @as(u8, @intCast(n & 1)) ^ field;

                    // Size the scratch off plane 0. The internal pipeline is
                    // vertical: the horizontal path sizes its line buffers off
                    // the source HEIGHT (the transposed line length).
                    const sp0 = src.plane(T, d.format.planeId(0));
                    const dp0 = dst.planeMut(T, d.format.planeId(0));
                    var alloc_w: u32 = undefined;
                    var alloc_stride: u32 = undefined;
                    var alloc_n_interp: u32 = undefined;
                    var srcT_rows: u32 = 0;
                    var dstT_rows: u32 = 0;
                    if (horizontal) {
                        alloc_w = sp0.height;
                        alloc_stride = std.mem.alignForward(u32, sp0.height, n_vec);
                        alloc_n_interp = if (k.dh) sp0.width else sp0.width / 2;
                        srcT_rows = sp0.width;
                        dstT_rows = if (k.dh) sp0.width * 2 else sp0.width;
                    } else {
                        alloc_w = sp0.width;
                        alloc_stride = @intCast(dp0.stride());
                        alloc_n_interp = if (k.dh) sp0.height else sp0.height / 2;
                    }
                    const entry = d.acquire(alloc_stride, alloc_w, alloc_n_interp, srcT_rows, dstT_rows, @sizeOf(T)) catch
                        return ctx.fail("failed to allocate memory.", .{});
                    defer d.release(entry);
                    const scratch = entry.s;

                    // mclip is always a single Gray plane; it drives every plane.
                    const mask_plane: ?avs.Plane(u8) = if (mcp) |m| blk: {
                        const raw = m.plane(u8, c.AVS_PLANAR_Y);
                        break :blk try hz.canonical(u8, raw, raw.pitch, &scratch_bufs);
                    } else null;
                    const maskp: ?[]const u8 = if (mask_plane) |m| m.fullSlice() else null;
                    const mask_stride: u32 = if (mask_plane) |m| @intCast(m.stride()) else 0;

                    for (0..d.format.color_planes) |plane| {
                        const id = d.format.planeId(plane);
                        const dp = dst.planeMut(T, id);
                        if (horizontal) {
                            const raw = src.plane(T, id);
                            const sp = try hz.canonical(T, raw, raw.pitch, &scratch_bufs);
                            const src_w = sp.width;
                            const src_h = sp.height;
                            const dst_w = dp.width;
                            const Lstride: u32 = std.mem.alignForward(u32, src_h, n_vec);
                            const srcT: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.srcT));
                            const dstT: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.dstT));
                            core.transposeAny(T, srcT, Lstride, sp.fullSlice(), @intCast(sp.stride()), src_w, src_h);

                            const maskT: ?[]const u8 = if (maskp) |mp| blk: {
                                core.transposeBlocked(u8, scratch.maskT, Lstride, mp, mask_stride, src_w, src_h);
                                break :blk scratch.maskT;
                            } else null;

                            const scpT: ?[]const T = if (k.vcheck > 0) (if (scp) |s| blk: {
                                const sraw = s.plane(T, id);
                                const scan = try hz.canonical(T, sraw, sraw.pitch, &scratch_bufs);
                                const scpT_buf: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.scpT));
                                core.transposeAny(T, scpT_buf, Lstride, scan.fullSlice(), @intCast(scan.stride()), dst_w, src_h);
                                break :blk scpT_buf;
                            } else null) else null;

                            processPlane(T, k, scratch, srcT, dstT, scpT, maskT, Lstride, field, src_h, Lstride, src_w, dst_w);
                            core.transposeAny(T, dp.fullSlice(), @intCast(dp.stride()), dstT, Lstride, src_h, dst_w);
                        } else {
                            // One line stride for src, dst and sclip: bring
                            // them to the destination's pitch.
                            const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch_bufs);
                            const scpp: ?[]const T = if (scp) |s| (try hz.canonical(T, s.plane(T, id), dp.pitch, &scratch_bufs)).fullSlice() else null;
                            processPlane(T, k, scratch, sp.fullSlice(), dp.fullSlice(), scpp, maskp, mask_stride, field, sp.width, @intCast(dp.stride()), sp.height, dp.height);
                        }
                    }

                    const props = dst.propsRW(env);
                    props.setInt("_FieldBased", 0, .replace) catch {};
                    if (k.field > 1) {
                        const ro = props.ro();
                        if (ro.getInt("_DurationNum", 0)) |num0| {
                            if (ro.getInt("_DurationDen", 0)) |den0| {
                                var num: u64 = @intCast(@max(num0, 0));
                                var den: u64 = @as(u64, @intCast(@max(den0, 0))) * 2;
                                reduce(&num, &den);
                                props.setInt("_DurationNum", @intCast(num), .replace) catch {};
                                props.setInt("_DurationDen", @intCast(den), .replace) catch {};
                            } else |_| {}
                        } else |_| {}
                    }
                    return dst;
                }
            };
        }

        fn mclipTo8(ctx: *avs.Create, clip: avs.Clip) !avs.Clip {
            const cv = avs.Value.fromClip(clip);
            defer cv.release();
            const t = c.avs_new_value_bool(1);
            const args = [_]c.AVS_Value{ cv.raw, c.avs_new_value_int(8), c.avs_new_value_int(0), t, t };
            const names = [_]?[*:0]const u8{ null, null, "dither_type", "fulls", "fulld" };
            var msg: ?[:0]const u8 = null;
            const v = ctx.env.invoke("vszip_Dither", avs.Value.arrayView(&args), &names, &msg) catch
                return ctx.fail("could not convert mclip to 8 bit: {s}", .{msg orelse "unknown error"});
            defer v.release();
            return v.takeClip(ctx.env);
        }

        pub fn create(ctx: *avs.Create) !avs.Value {
            const clip = try ctx.clip(0, "clip");
            defer clip.release();
            const vi_in = clip.videoInfo();
            const format = try hz.planarFormat(ctx, vi_in);
            var vi = vi_in.*;
            const big = math.maxInt(i32);
            const fmax = math.floatMax(f32);
            const eargs: core.Data.Args = .{
                .field = try ctx.int(i32, 1, "field", null, -big, big),
                .dh = try ctx.boolean(2, "dh", false),
                .alpha = try ctx.float(f32, 3, "alpha", 0.2, -fmax, fmax),
                .beta = try ctx.float(f32, 4, "beta", 0.25, -fmax, fmax),
                .gamma = try ctx.float(f32, 5, "gamma", 20.0, -fmax, fmax),
                .nrad = try ctx.int(i32, 6, "nrad", 2, -big, big),
                .mdis = try ctx.int(i32, 7, "mdis", 20, -big, big),
                .hp = try ctx.boolean(8, "hp", false),
                .vcheck = try ctx.int(i32, 9, "vcheck", 2, -big, big),
                .vthresh0 = try ctx.float(f32, 10, "vthresh0", 32.0, -fmax, fmax),
                .vthresh1 = try ctx.float(f32, 11, "vthresh1", 64.0, -fmax, fmax),
                .vthresh2 = try ctx.float(f32, 12, "vthresh2", 4.0, -fmax, fmax),
            };
            const field = eargs.field;
            const vcheck = eargs.vcheck;
            if (core.Data.check(eargs, horizontal, if (horizontal) vi.width else vi.height)) |msg| return ctx.fail("{s}", .{msg});
            if (eargs.dh and format.has_alpha) return ctx.fail("dh=true is not supported for clips with an alpha plane.", .{});

            var mclip: ?avs.Clip = try ctx.optClip(14, "mclip");
            errdefer if (mclip) |m| m.release();
            if (mclip) |m| {
                const mvi = m.videoInfo();
                const mf = try hz.planarFormat(ctx, mvi);
                // Unlike eedi3m, the mask must be Gray: a single plane drives
                // every processed plane.
                if (mf.family != .gray) return ctx.fail("mclip must be Gray.", .{});
                if (mvi.width != vi.width or mvi.height != vi.height) return ctx.fail("mclip's dimensions don't match.", .{});
                if (mvi.num_frames != vi.num_frames) return ctx.fail("mclip's number of frames doesn't match.", .{});
                if (mf.sample != .u8) {
                    const m8 = try mclipTo8(ctx, m);
                    m.release();
                    mclip = m8;
                }
            }

            if (field > 1) {
                if (vi.num_frames > math.maxInt(i32) / 2) return ctx.fail("resulting clip is too long.", .{});
                vi.num_frames *= 2;
                var num: u64 = @as(u64, vi.fps_numerator) * 2;
                var den: u64 = vi.fps_denominator;
                reduce(&num, &den);
                if (num > math.maxInt(c_uint) or den > math.maxInt(c_uint)) return ctx.fail("resulting frame rate overflows.", .{});
                vi.fps_numerator = @intCast(num);
                vi.fps_denominator = @intCast(den);
            }
            if (eargs.dh) {
                if (horizontal) vi.width *= 2 else vi.height *= 2;
            }
            c.avs_set_field_based(&vi, 0);

            const sclip: ?avs.Clip = if (vcheck > 0) try ctx.optClip(13, "sclip") else null;
            errdefer if (sclip) |s| s.release();
            if (sclip) |s| {
                const svi = s.videoInfo();
                if (svi.pixel_type != vi.pixel_type or svi.width != vi.width or svi.height != vi.height or
                    svi.fps_numerator != vi.fps_numerator or svi.fps_denominator != vi.fps_denominator)
                    return ctx.fail("sclip's format and dimensions don't match.", .{});
                if (svi.num_frames != vi.num_frames) return ctx.fail("sclip's number of frames doesn't match.", .{});
            }

            const k = core.Data.init(eargs, horizontal, format.isFloat(), format.bits);
            const d: Data = .{ .format = format, .k = k, .sclip = .{ .clip = sclip }, .mclip = .{ .clip = mclip } };
            const gf = switch (format.sample) {
                inline else => |s| F.getFrame(Get(s.Type()).getFrame),
            };
            const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice, .vi = vi });
            return avs.Create.returnClip(out);
        }
    };
}
