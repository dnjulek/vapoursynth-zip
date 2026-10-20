//! vszip_MosquitoNR(clip, int[] "strength", int[] "restore", int[] "radius", int[] "planes")

const std = @import("std");
const avs = @import("avisynth");

const core = @import("../filters/mosquito_nr.zig");
const core_float = @import("../filters/mosquito_nr_float.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_MosquitoNR";
pub const params = "c[strength]i*[restore]i*[radius]i*[planes]i*";

const Data = struct {
    format: avs.Format,
    strength: [3]i32,
    restore: [3]i32,
    radius: [3]u32,
    planes: [4]bool,

    // Pool of per-frame wavelet scratch arenas (~42 MB at 1080p u16), reused
    // across frames. Each in-flight request checks one out exclusively; the
    // mutex only guards the pool (never held across upstream requests or
    // processing). Reuse is deterministic: process() writes every scratch
    // element before reading it.
    pool_mutex: std.Io.Mutex = .init,
    pool: std.ArrayList(std.heap.ArenaAllocator.State) = .empty,

    pub fn deinit(self: *Data) void {
        for (self.pool.items) |state| state.promote(allocator).deinit();
        self.pool.deinit(allocator);
    }
};

const F = avs.Filter(Data, filter_name);

fn Filter(comptime T: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.planes);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            const proc = if (T == f32) core_float.MosquitoNRFloat.process else core.MosquitoNR(T).process;

            var arena = blk: {
                d.pool_mutex.lockUncancelable(common.io);
                defer d.pool_mutex.unlock(common.io);
                break :blk (d.pool.pop() orelse std.heap.ArenaAllocator.State.init).promote(allocator);
            };
            defer {
                _ = arena.reset(.retain_capacity);
                d.pool_mutex.lockUncancelable(common.io);
                defer d.pool_mutex.unlock(common.io);
                d.pool.append(allocator, arena.state) catch arena.deinit();
            }

            for (0..d.format.color_planes) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(T, id);
                const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch);
                const dptr: [*]T = @ptrCast(dp.ptr);
                const sptr: [*]const T = @ptrCast(sp.ptr);
                proc(dptr, dp.stride(), sptr, dp.width, dp.height, d.strength[plane], d.restore[plane], d.radius[plane], @intCast(d.format.bits), plane > 0, arena.allocator()) catch
                    return ctx.fail("out of memory", .{});
            }
            return dst;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clip");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);
    if (format.family == .rgb) return ctx.fail("input must be YUV or Gray.", .{});

    var planes = try ctx.planes(4, format.color_planes, false);
    if (!ctx.defined(4)) planes[0] = true; // vszip default: luma only

    for (0..format.color_planes) |p| {
        if (!planes[p]) continue;
        if (format.planeWidth(@intCast(vi.width), p) < 4 or format.planeHeight(@intCast(vi.height), p) < 4)
            return ctx.fail("input is too small (need at least 4x4 per processed plane).", .{});
    }

    const strength = try ctx.perPlaneInt(i32, 1, "strength", 16, 0, 32);
    const restore = try ctx.perPlaneInt(i32, 2, "restore", 128, 0, 128);
    const radius = try ctx.perPlaneInt(u32, 3, "radius", 2, 1, 2);

    const d: Data = .{ .format = format, .strength = strength, .restore = restore, .radius = radius, .planes = planes };
    const gf = switch (format.sample) {
        inline else => |s| F.getFrame(Filter(s.Type()).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
