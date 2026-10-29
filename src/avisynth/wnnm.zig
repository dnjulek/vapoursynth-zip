//! avszip_WNNM(clip, float[] "sigma", int "block_size", int "block_step", int "group_size",
//!   int "bm_range", int "radius", int "ps_num", int "ps_range", bool "residual",
//!   bool "adaptive_aggregation", clip "rclip")
//!
//! 32-bit float only. radius = 0: one spatial stage. radius > 0: the same
//! two-stage graph as vszip: a raw stage writes per-frame numerator /
//! denominator slabs for the 2r+1 temporal window into a tall float frame, and
//! an aggregation stage sums the slabs of neighbouring raw frames and divides.
//! The raw stage is registered as `avszip_WNNMRaw` (same parameters) and
//! instantiated through avs_invoke so the host puts its frame cache in front
//! of it: every raw frame is read by up to 2r+1 output frames.
//!
//! Scratch: vszip keyed scratch by thread id; here each request checks a
//! block out of a mutex-guarded pool (exclusive per request).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/wnnm.zig");
const bc = @import("../filters/bm3d_common.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
const max_window = filter.max_window;
const Config = filter.Config;
const Bufs = filter.Bufs;
const processPlane = filter.processPlane;

const shared_params = "c[sigma]f*[block_size]i[block_step]i[group_size]i[bm_range]i[radius]i[ps_num]i[ps_range]i" ++
    "[residual]b[adaptive_aggregation]b[rclip]c";

/// Exclusive-checkout pool of `*Item`.
fn Pool(comptime Item: type) type {
    return struct {
        const Self = @This();
        lock: std.Io.Mutex = .init,
        items: std.ArrayList(Item) = .empty,

        fn take(self: *Self) ?Item {
            self.lock.lockUncancelable(common.io);
            defer self.lock.unlock(common.io);
            return self.items.pop();
        }

        /// Returns false when the item could not be kept (caller frees it).
        fn put(self: *Self, it: Item) bool {
            self.lock.lockUncancelable(common.io);
            defer self.lock.unlock(common.io);
            self.items.append(allocator, it) catch return false;
            return true;
        }
    };
}

const Data = struct {
    format: avs.Format,
    cfg: Config,
    num_frames: c_int,
    ref: avs.OwnedClip = .{},
    pool: Pool(*Bufs) = .{},

    pub fn deinit(self: *Data) void {
        for (self.pool.items.items) |b| b.destroy();
        self.pool.items.deinit(allocator);
        self.ref.deinit();
    }

    fn acquire(self: *Data) !*Bufs {
        return self.pool.take() orelse Bufs.create(&self.cfg);
    }

    fn release(self: *Data, b: *Bufs) void {
        if (!self.pool.put(b)) b.destroy();
    }
};

fn Wnnm(comptime BS: ?u32) type {
    return struct {
        const FS = avs.Filter(Data, "avszip_WNNM");
        const FR = avs.Filter(Data, "avszip_WNNMRaw");

        fn spatial(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const ref = if (d.ref.clip) |r| try ctx.getFrame(r, n) else src.retain();
            defer ref.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            const process: [4]bool = .{ d.cfg.process[0], d.cfg.process[1], d.cfg.process[2], false };
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &process);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            const bufs = d.acquire() catch return ctx.fail("out of memory", .{});
            defer d.release(bufs);

            for (0..d.format.color_planes) |plane| {
                if (!d.cfg.process[plane]) continue;
                defer scratch.reset(); // copies are per plane
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(f32, id);
                const sp = try hz.canonical(f32, src.plane(f32, id), dp.pitch, &scratch);
                const rp = try hz.canonical(f32, ref.plane(f32, id), dp.pitch, &scratch);
                const w = dp.width;
                const h = dp.height;
                const stride: u32 = @intCast(dp.stride());
                const intermediate = bufs.intermediate[0 .. 2 * h * w];
                @memset(intermediate, 0.0);
                processPlane(BS, &d.cfg, bufs, &.{sp.fullSlice()}, &.{rp.fullSlice()}, intermediate.ptr, w, h, w, h, stride, d.cfg.sigma[plane], {}) catch
                    return ctx.fail("out of memory", .{});
                filter.aggregation(dp.fullSlice(), intermediate, w, h, stride);
            }
            return dst;
        }

        fn raw(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const radius: i32 = @intCast(d.cfg.radius);
            const num_frames: usize = 2 * d.cfg.radius + 1;
            var srcs: [max_window]avs.Frame = undefined;
            var refs: [max_window]avs.Frame = undefined;
            var got: usize = 0;
            defer for (0..got) |i| {
                srcs[i].release();
                refs[i].release();
            };
            for (0..num_frames) |i| {
                const fid = bc.clampFrame(n, @as(i64, @intCast(i)) - radius, d.num_frames);
                const s = try ctx.childFrame(fid);
                const r = if (d.ref.clip) |rc| ctx.getFrame(rc, fid) catch |e| {
                    s.release();
                    return e;
                } else s.retain();
                srcs[i] = s;
                refs[i] = r;
                got += 1;
            }

            // The tall raw frame: props from the window's center frame.
            const dst = try ctx.newFrame(srcs[d.cfg.radius]);
            errdefer dst.release();
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            const bufs = d.acquire() catch return ctx.fail("out of memory", .{});
            defer d.release(bufs);

            var srcps: [max_window][]const f32 = undefined;
            var refps: [max_window][]const f32 = undefined;
            for (0..d.format.color_planes) |plane| {
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(f32, id);
                if (!d.cfg.process[plane]) continue;
                defer scratch.reset(); // copies are per plane
                const pitch = dp.pitch;
                const center = srcs[d.cfg.radius].plane(f32, id);
                const w = center.width;
                const h = center.height;
                for (0..num_frames) |i| {
                    srcps[i] = (try hz.canonical(f32, srcs[i].plane(f32, id), pitch, &scratch)).fullSlice();
                    refps[i] = (try hz.canonical(f32, refs[i].plane(f32, id), pitch, &scratch)).fullSlice();
                }
                const stride: u32 = @intCast(dp.stride());
                const dstp = dp.fullSlice();
                @memset(dstp[0 .. 2 * num_frames * h * stride], 0.0);
                processPlane(BS, &d.cfg, bufs, srcps[0..num_frames], refps[0..num_frames], dstp.ptr, stride, h, w, h, stride, d.cfg.sigma[plane], {}) catch
                    return ctx.fail("out of memory", .{});
            }
            return dst;
        }
    };
}

fn selectSpatial(bs: u32) avs.filter.RawGetFrame {
    return switch (bs) {
        inline 4, 6, 8, 16 => |b| Wnnm(b).FS.getFrame(Wnnm(b).spatial),
        else => Wnnm(null).FS.getFrame(Wnnm(null).spatial),
    };
}

fn selectRaw(bs: u32) avs.filter.RawGetFrame {
    return switch (bs) {
        inline 4, 6, 8, 16 => |b| Wnnm(b).FR.getFrame(Wnnm(b).raw),
        else => Wnnm(null).FR.getFrame(Wnnm(null).raw),
    };
}

// --------------------------------------------------------------------------
// Aggregation stage (radius > 0)
// --------------------------------------------------------------------------

const VagData = struct {
    format: avs.Format,
    raw: avs.OwnedClip,
    radius: u32,
    process: [3]bool,
    max_w: u32,
    num_frames: c_int,
    pool: Pool([]align(common.vec_len) f32) = .{},

    pub fn deinit(self: *VagData) void {
        for (self.pool.items.items) |b| allocator.free(b);
        self.pool.items.deinit(allocator);
        self.raw.deinit();
    }
};

const FV = avs.Filter(VagData, "avszip_WNNM");

fn vagGetFrame(d: *VagData, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    const radius: i32 = @intCast(d.radius);
    const num_frames: usize = 2 * d.radius + 1;
    const src = try ctx.childFrame(n);
    defer src.release();

    var raws: [max_window]avs.Frame = undefined;
    var got: usize = 0;
    defer for (raws[0..got]) |f| f.release();
    for (0..num_frames) |i| {
        const fid = bc.clampFrame(n, @as(i64, @intCast(i)) - radius, d.num_frames);
        raws[i] = try ctx.getFrame(d.raw.get(), fid);
        got += 1;
    }

    const dst = try ctx.newFrame(src);
    errdefer dst.release();
    const process: [4]bool = .{ d.process[0], d.process[1], d.process[2], false };
    hz.copyUnprocessed(ctx.env(), dst, src, d.format, &process);

    const buffer = d.pool.take() orelse (allocator.alignedAlloc(f32, common.alignment, 2 * d.max_w) catch return ctx.fail("out of memory", .{}));
    defer if (!d.pool.put(buffer)) allocator.free(buffer);

    for (0..d.format.color_planes) |plane| {
        if (!d.process[plane]) continue;
        const id = d.format.planeId(plane);
        const dp = dst.planeMut(f32, id);
        const w = dp.width;
        const h = dp.height;
        const num = buffer[0..w];
        const den = buffer[d.max_w..][0..w];

        for (0..h) |y| {
            @memset(num, 0.0);
            @memset(den, 0.0);
            for (0..num_frames) |i| {
                const rp = raws[i].plane(f32, id);
                // clamp(2r - i, n - nf + 1 + r, n + r), in i64
                const slab = bc.aggZ(@intCast(i), n, d.num_frames, radius);
                const base_row = @as(usize, @intCast(slab)) * 2 * h + y;
                filter.vagAccumulate(num, rp.row(base_row));
                filter.vagAccumulate(den, rp.row(base_row + h));
            }
            filter.vagDivide(dp.row(y), num, den);
        }
    }
    return dst;
}

// --------------------------------------------------------------------------
// Create
// --------------------------------------------------------------------------

fn intArg(arg: anytype, comptime key: []const u8, default: i64) !i64 {
    return arg.int(i64, key, default, math.minInt(i64), math.maxInt(i64));
}

/// Parsed config, or null when every plane is skipped (pass-through).
fn parse(ctx: *avs.Create, vi: *const c.AVS_VideoInfo, format: avs.Format) !?Config {
    const arg = ctx.named(shared_params);
    const fmax = math.floatMax(f32);
    var plane_w: [3]u32 = @splat(0);
    var plane_h: [3]u32 = @splat(0);
    for (0..format.color_planes) |i| {
        plane_w[i] = format.planeWidth(@intCast(vi.width), i);
        plane_h[i] = format.planeHeight(@intCast(vi.height), i);
    }
    return Config.init(.{
        .sigma = try arg.perPlaneFloat(f32, "sigma", 3.0, -fmax, fmax),
        .block_size = try intArg(arg, "block_size", 8),
        .block_step = if (arg.defined("block_step")) try intArg(arg, "block_step", 0) else null,
        .group_size = try intArg(arg, "group_size", 8),
        .bm_range = try intArg(arg, "bm_range", 7),
        .radius = try intArg(arg, "radius", 0),
        .ps_num = try intArg(arg, "ps_num", 2),
        .ps_range = try intArg(arg, "ps_range", 4),
        .residual = try arg.boolean("residual", false),
        .adaptive = try arg.boolean("adaptive_aggregation", true),
        .num_planes = format.color_planes,
        .plane_w = plane_w,
        .plane_h = plane_h,
    }) catch |e| ctx.fail("{s}", .{Config.message(e)});
}

fn openRef(ctx: *avs.Create, vi: *const c.AVS_VideoInfo) !?avs.Clip {
    const arg = ctx.named(shared_params);
    const ref = try arg.optClip("rclip");
    if (ref) |r| {
        const rvi = r.videoInfo();
        if (rvi.pixel_type != vi.pixel_type or rvi.width != vi.width or rvi.height != vi.height or
            rvi.num_frames != vi.num_frames or rvi.fps_numerator != vi.fps_numerator or rvi.fps_denominator != vi.fps_denominator)
        {
            r.release();
            return ctx.fail("\"rclip\" must be of the same format, dimensions and number of frames as \"clip\".", .{});
        }
    }
    return ref;
}

fn checkFormat(ctx: *avs.Create, vi: *const c.AVS_VideoInfo) !avs.Format {
    const format = try hz.planarFormat(ctx, vi);
    if (!format.isFloat()) return ctx.fail("only constant format 32 bit float input supported.", .{});
    return format;
}

pub const Main = struct {
    pub const filter_name = "avszip_WNNM";
    pub const params = shared_params;

    pub fn create(ctx: *avs.Create) !avs.Value {
        const clip = try ctx.clip(0, "clip");
        defer clip.release();
        const vi = clip.videoInfo();
        const format = try checkFormat(ctx, vi);
        const cfg = (try parse(ctx, vi, format)) orelse return avs.Create.returnClip(clip.retain());

        if (cfg.radius == 0) {
            const ref = try openRef(ctx, vi);
            errdefer if (ref) |r| r.release();
            const d: Data = .{ .format = format, .cfg = cfg, .num_frames = vi.num_frames, .ref = .{ .clip = ref } };
            const out = try Wnnm(null).FS.create(ctx.env, clip, d, selectSpatial(cfg.bs), .{ .mt_mode = .nice });
            return avs.Create.returnClip(out);
        }

        // Raw stage through the host (cached), with exactly our arguments.
        var vals: [16]c.AVS_Value = undefined;
        var names: [16]?[*:0]const u8 = undefined;
        const na = ctx.forwardArgs(shared_params, &vals, &names);
        var msg: ?[:0]const u8 = null;
        const rv = ctx.env.invoke(Raw.filter_name, avs.Value.arrayView(vals[0..na]), names[0..na], &msg) catch
            return ctx.fail("internal {s} failed: {s}", .{ Raw.filter_name, msg orelse "unknown error" });
        defer rv.release();
        if (!rv.isClip()) return ctx.fail("internal {s} did not return a clip.", .{Raw.filter_name});
        const raw_clip = rv.takeClip(ctx.env);
        errdefer raw_clip.release();

        const vag: VagData = .{
            .format = format,
            .raw = .{ .clip = raw_clip },
            .radius = cfg.radius,
            .process = cfg.process,
            .max_w = cfg.max_iw,
            .num_frames = vi.num_frames,
        };
        const out = try FV.create(ctx.env, clip, vag, FV.getFrame(vagGetFrame), .{ .mt_mode = .nice });
        return avs.Create.returnClip(out);
    }
};

/// Internal raw stage (radius > 0); public so the host can cache it.
pub const Raw = struct {
    pub const filter_name = "avszip_WNNMRaw";
    pub const params = shared_params;

    pub fn create(ctx: *avs.Create) !avs.Value {
        const clip = try ctx.clip(0, "clip");
        defer clip.release();
        const vi = clip.videoInfo();
        const format = try checkFormat(ctx, vi);
        const cfg = (try parse(ctx, vi, format)) orelse return ctx.fail("nothing to process.", .{});
        if (cfg.radius == 0) return ctx.fail("radius must be > 0 for the raw stage.", .{});
        const ref = try openRef(ctx, vi);
        errdefer if (ref) |r| r.release();

        var vi_raw = vi.*;
        vi_raw.height *= @intCast(2 * (2 * cfg.radius + 1));
        const d: Data = .{ .format = format, .cfg = cfg, .num_frames = vi.num_frames, .ref = .{ .clip = ref } };
        const out = try Wnnm(null).FR.create(ctx.env, clip, d, selectRaw(cfg.bs), .{ .mt_mode = .nice, .vi = vi_raw });
        return avs.Create.returnClip(out);
    }
};
