//! avszip_NLMeans(clip, int "d", int "a", int "s", float "h", string "channels",
//!   int "wmode", float "wref", clip "rclip")
//!
//! KNLMeansCL-compatible non-local means (filters/nlmeans.zig), bit-exact
//! with vszip. 8-16 bit integer and 32-bit float; alpha passes through.
//! Frame n reads frames n-m .. n+m with m = min(d, n), the far side clamped
//! to the last frame (KNLMeansCL's window, as in vszip).
//!
//! Planes go to the kernel in place at their own pitch: the kernel copies a
//! padded f32 window per tile and reads only active samples, so cropped or
//! foreign-pitch sources need no canonical copy.
//!
//! Scratch: vszip keys it by thread id; here each request checks a block out
//! of a mutex-guarded pool (exclusive per request).

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/nlmeans.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_NLMeans";
pub const params = "c[d]i[a]i[s]i[h]f[channels]s[wmode]i[wref]f[rclip]c";

const Ref = filter.Ref;
const max_window = 2 * filter.max_d + 1;

const Data = struct {
    format: avs.Format,
    cfg: filter.Config,
    plane0: u32,
    peak: f32,
    num_frames: c_int,
    ref: avs.OwnedClip = .{},
    lock: std.Io.Mutex = .init,
    pool: std.ArrayList(*filter.Scratch) = .empty,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |sc| sc.destroy();
        self.pool.deinit(allocator);
        self.ref.deinit();
    }

    fn acquire(self: *Data) !*filter.Scratch {
        {
            self.lock.lockUncancelable(common.io);
            defer self.lock.unlock(common.io);
            if (self.pool.pop()) |sc| return sc;
        }
        return filter.Scratch.create(&self.cfg, self.ref.clip != null);
    }

    fn release(self: *Data, sc: *filter.Scratch) void {
        self.lock.lockUncancelable(common.io);
        defer self.lock.unlock(common.io);
        self.pool.append(allocator, sc) catch sc.destroy();
    }
};

const F = avs.Filter(Data, filter_name);

fn view(comptime T: type, p: avs.Plane(T)) filter.PlaneIn(T) {
    return .{ .data = p.span(), .stride = @intCast(p.stride()) };
}

fn NLMeans(comptime T: type, comptime ref: Ref, comptime wmode: u32) type {
    return struct {
        const C = ref.channels();

        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const m: i32 = @min(@as(i32, @intCast(d.cfg.d)), n);
            const count: usize = @intCast(2 * m + 1);
            var srcs: [max_window]avs.Frame = undefined;
            var refs: [max_window]avs.Frame = undefined;
            var got_src: usize = 0;
            var got_ref: usize = 0;
            defer {
                for (srcs[0..got_src]) |f| f.release();
                for (refs[0..got_ref]) |f| f.release();
            }
            for (0..count) |i| {
                const fid = common.clampFrame(n, @as(i64, @intCast(i)) - m, d.num_frames);
                srcs[i] = try ctx.childFrame(fid);
                got_src += 1;
                if (d.ref.clip) |rc| {
                    refs[i] = try ctx.getFrame(rc, fid);
                    got_ref += 1;
                }
            }

            const center = srcs[@intCast(m)];
            const dst = try ctx.newFrame(center);
            errdefer dst.release();
            var process: [4]bool = @splat(false);
            for (0..C) |c| process[d.plane0 + c] = true;
            hz.copyUnprocessed(ctx.env(), dst, center, d.format, &process);

            const sc = d.acquire() catch return ctx.fail("out of memory", .{});
            defer d.release(sc);

            // window layer t = d + k holds frame n + k
            const In = filter.PlaneIn(T);
            var src_planes: [max_window * 3]In = undefined;
            var ref_planes: [max_window * 3]In = undefined;
            const n_planes = (2 * d.cfg.d + 1) * C;
            @memset(src_planes[0..n_planes], .{ .data = &.{}, .stride = 0 });
            @memset(ref_planes[0..n_planes], .{ .data = &.{}, .stride = 0 });
            const t0: usize = d.cfg.d - @as(u32, @intCast(m));
            for (0..count) |i| {
                for (0..C) |c| {
                    const id = d.format.planeId(d.plane0 + c);
                    src_planes[(t0 + i) * C + c] = view(T, srcs[i].plane(T, id));
                    if (got_ref > 0) ref_planes[(t0 + i) * C + c] = view(T, refs[i].plane(T, id));
                }
            }

            var dst_planes: [C]filter.PlaneOut(T) = undefined;
            for (0..C) |c| {
                const p = dst.planeMut(T, d.format.planeId(d.plane0 + c));
                dst_planes[c] = .{ .data = p.span(), .stride = @intCast(p.stride()) };
            }

            filter.processFrame(
                T,
                ref,
                wmode,
                &d.cfg,
                sc,
                src_planes[0..n_planes],
                if (got_ref > 0) ref_planes[0..n_planes] else null,
                dst_planes,
                @intCast(m),
                d.peak,
            );
            return dst;
        }
    };
}

/// KNLMeansCL `channels`: which planes form the joint distance, and the
/// first of them.
fn parseChannels(ctx: *avs.Create, format: avs.Format, str: []const u8) !struct { Ref, u32 } {
    const eq = std.ascii.eqlIgnoreCase;
    switch (format.family) {
        .gray => {
            if (eq(str, "Y") or eq(str, "auto")) return .{ .luma, 0 };
            return ctx.fail("'channels' must be 'Y' with Gray.", .{});
        },
        .yuv => {
            if (eq(str, "YUV")) {
                if (format.ssw != 0 or format.ssh != 0) {
                    return ctx.fail("'channels'='YUV' requires 4:4:4 (the joint patch distance needs one pixel lattice; run 'Y' and 'UV' passes for subsampled formats).", .{});
                }
                return .{ .yuv, 0 };
            }
            if (eq(str, "UV")) return .{ .chroma, 1 };
            if (eq(str, "Y") or eq(str, "auto")) return .{ .luma, 0 };
            return ctx.fail("'channels' must be 'YUV', 'Y' or 'UV' with YUV.", .{});
        },
        .rgb => {
            if (eq(str, "RGB") or eq(str, "auto")) return .{ .rgb, 0 };
            return ctx.fail("'channels' must be 'RGB' with RGB.", .{});
        },
    }
}

fn intArg(ctx: *avs.Create, i: usize, comptime key: []const u8, default: i64) !i64 {
    return ctx.int(i64, i, key, default, math.minInt(i64), math.maxInt(i64));
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    const ref, const plane0 = try parseChannels(ctx, format, (try ctx.string(5, "channels")) orelse "auto");
    const fmax: f64 = math.floatMax(f32);
    const cfg = filter.Config.init(.{
        .d = try intArg(ctx, 1, "d", 1),
        .a = try intArg(ctx, 2, "a", 2),
        .s = try intArg(ctx, 3, "s", 4),
        .h = try ctx.float(f64, 4, "h", 1.2, 0, fmax),
        .wmode = try intArg(ctx, 6, "wmode", 0),
        .wref = try ctx.float(f64, 7, "wref", 1.0, 0, fmax),
        .ref = ref,
        .width = format.planeWidth(@intCast(vi.width), plane0),
        .height = format.planeHeight(@intCast(vi.height), plane0),
    }) catch |e| return ctx.fail("{s}", .{filter.Config.message(e)});

    const rclip = try ctx.optClip(8, "rclip");
    errdefer if (rclip) |r| r.release();
    if (rclip) |r| try hz.compareClips(ctx, &.{ vi, r.videoInfo() }, .SAME_LEN, .color_only);

    const d: Data = .{
        .format = format,
        .cfg = cfg,
        .plane0 = plane0,
        .peak = @floatCast(format.peak()),
        .num_frames = vi.num_frames,
        .ref = .{ .clip = rclip },
    };
    const gf = switch (format.sample) {
        inline else => |st| switch (ref) {
            inline else => |r| switch (cfg.wmode) {
                inline 0...3 => |w| F.getFrame(NLMeans(st.Type(), r, w).getFrame),
                else => unreachable,
            },
        },
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
