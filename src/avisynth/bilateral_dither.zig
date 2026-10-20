//! vszip_BilateralDither(clip, clip "ref", int[] "radius", float[] "thr", float[] "flat", float[] "wmin", float[] "subspl", int[] "planes")

const std = @import("std");
const avs = @import("avisynth");

const filter = @import("../filters/bilateral_dither.zig");
const subspl = @import("../filters/bilateral_dither_subspl.zig");
const hz = @import("helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_BilateralDither";
pub const params = "c[ref]c[radius]i*[thr]f*[flat]f*[wmin]f*[subspl]f*[planes]i*";

const Data = struct {
    format: avs.Format,
    ref: avs.OwnedClip = .{},
    planes: [4]bool,
    cfg: filter.Config = .{},

    pub fn deinit(self: *Data) void {
        self.ref.deinit();
        self.cfg.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn GetFrame(comptime T: type, comptime has_ref: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const ref = if (has_ref) try ctx.getFrame(d.ref.get(), n);
            defer if (has_ref) ref.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.planes);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.color_planes) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(T, id);
                const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch);
                const refp: ?[]const T = if (has_ref) (try hz.canonical(T, ref.plane(T, id), dp.pitch, &scratch)).fullSlice() else null;
                filter.processPlane(T, sp.fullSlice(), refp, dp.fullSlice(), dp.width, dp.height, @intCast(dp.stride()), d.cfg.rh[plane], d.cfg.rv[plane], d.cfg.m[plane], d.cfg.wmax[plane], d.cfg.sum_w_min[plane], d.cfg.peak, d.cfg.offsets[plane], d.cfg.k[plane]);
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
    const is_int = !format.isFloat();

    const radius = try ctx.perPlaneInt(i32, 2, "radius", 16, 2, 16384);
    const thr = try ctx.perPlaneFloat(f32, 3, "thr", 2.5, 0, 65535);
    const flat = try ctx.perPlaneFloat(f32, 4, "flat", 0.4, 0, 1);
    const wmin = try ctx.perPlaneFloat(f32, 5, "wmin", 0, 0, 65535);
    const subspl_arg = try ctx.perPlaneFloat(f32, 6, "subspl", 0, 0, 4096);

    filter.Config.checkSize(vi.width, vi.height) catch |e| return ctx.fail("{s}", .{filter.Config.message(e)});

    var d: Data = .{ .format = format, .planes = try ctx.planes(7, format.color_planes, true) };
    var plane_w: [3]i32 = @splat(0);
    var plane_h: [3]i32 = @splat(0);
    for (0..format.color_planes) |p| {
        plane_w[p] = @intCast(format.planeWidth(@intCast(vi.width), p));
        plane_h[p] = @intCast(format.planeHeight(@intCast(vi.height), p));
    }
    d.cfg = filter.Config.init(.{
        .is_int = is_int,
        .bits = format.bits,
        .planes = .{ d.planes[0], d.planes[1], d.planes[2] },
        .num_planes = format.color_planes,
        .plane_w = plane_w,
        .plane_h = plane_h,
        .radius = radius,
        .thr = thr,
        .flat = flat,
        .wmin = wmin,
        .subspl = subspl_arg,
    }) catch |e| return ctx.fail("{s}", .{filter.Config.message(e)});
    errdefer d.cfg.deinit();

    const ref = try ctx.optClip(1, "ref");
    errdefer if (ref) |r| r.release();
    if (ref) |r| {
        const rvi = r.videoInfo();
        if (rvi.pixel_type != vi.pixel_type or rvi.width != vi.width or rvi.height != vi.height or
            rvi.num_frames != vi.num_frames or rvi.fps_numerator != vi.fps_numerator or rvi.fps_denominator != vi.fps_denominator)
            return ctx.fail("\"ref\" must have the same format and dimensions as \"clip\"", .{});
    }
    d.ref = .{ .clip = ref };

    const gf = switch (format.sample) {
        inline else => |s| if (ref != null)
            F.getFrame(GetFrame(s.Type(), true).getFrame)
        else
            F.getFrame(GetFrame(s.Type(), false).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
