//! vszip_Bilateral(clip, clip "ref", float[] "sigmaS", float[] "sigmaR", int[] "planes", int[] "algorithm", int[] "PBFICnum")

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/bilateral.zig");
const hz = @import("helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_Bilateral";
pub const params = "c[ref]c[sigmaS]f*[sigmaR]f*[planes]i*[algorithm]i*[PBFICnum]i*";
const max_sigma_s = filter.Data.max_sigma_s;

const Data = struct {
    format: avs.Format,
    ref: avs.OwnedClip = .{},
    k: filter.Data,

    pub fn deinit(self: *Data) void {
        self.ref.deinit();
        self.k.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn Bilateral(comptime T: type, comptime join: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const ref = if (join) try ctx.getFrame(d.ref.get(), n) else src;
            defer if (join) ref.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            const process: [4]bool = .{ d.k.planes[0], d.k.planes[1], d.k.planes[2], false };
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &process);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.color_planes) |plane| {
                if (!d.k.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(T, id);
                const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch);
                const rp = if (join) try hz.canonical(T, ref.plane(T, id), dp.pitch, &scratch) else sp;
                try filter.bilateral(T, sp.fullSlice(), rp.fullSlice(), dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height, @intCast(plane), join, &d.k);
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

    const yuv = format.family == .yuv;
    const hist_len: u32 = if (format.isFloat()) 65536 else math.shl(u32, 1, format.bits);

    // sigmaS: hand-parsed for the derived chroma default.
    var sbuf: [3]f64 = undefined;
    const sigma_given = (try ctx.floatArray(f64, 2, "sigmaS", &sbuf, -math.floatMax(f64), math.floatMax(f64))) orelse sbuf[0..0];
    var sigma_s: [3]f64 = undefined;
    for (0..3) |i| {
        sigma_s[i] = if (i < sigma_given.len) sigma_given[i] else filter.Data.sigmaSDefault(&sigma_s, i, yuv, format.ssw, format.ssh);
        if (sigma_s[i] < 0) return ctx.fail("Invalid \"sigmaS\" assigned, must be non-negative float number", .{});
        // The spatial radius derivation runs in i32/u32; bound sigma so none
        // of it can overflow.
        if (sigma_s[i] > max_sigma_s) return ctx.fail("Invalid \"sigmaS\" assigned, must be at most {d}", .{max_sigma_s});
    }

    const sigma_r = try ctx.perPlaneFloat(f64, 3, "sigmaR", 0.02, 0, math.floatMax(f64));
    const algorithm = try ctx.perPlaneInt(i32, 5, "algorithm", 0, 0, 2);
    const pbfic_num = try ctx.perPlaneInt(u32, 6, "PBFICnum", 0, 0, 256);
    const planes4 = try ctx.planes(4, format.color_planes, true);

    var plane_w: [3]u32 = undefined;
    var plane_h: [3]u32 = undefined;
    for (0..3) |p| {
        plane_w[p] = format.planeWidth(@intCast(vi.width), p);
        plane_h[p] = format.planeHeight(@intCast(vi.height), p);
    }

    const ref = try ctx.optClip(1, "ref");
    errdefer if (ref) |r| r.release();
    if (ref) |r| try hz.compareClips(ctx, &.{ vi, r.videoInfo() }, .BIGGER_THAN, .color_only);

    var k = filter.Data.init(.{
        .sigmaS = sigma_s,
        .sigmaR = sigma_r,
        .algorithm = algorithm,
        .PBFICnum = pbfic_num,
        .planes = .{ planes4[0], planes4[1], planes4[2] },
        .yuv = yuv,
        .plane_w = plane_w,
        .plane_h = plane_h,
        .hist_len = hist_len,
    }) catch |e| return ctx.fail("{s}", .{filter.Data.message(e)});
    errdefer k.deinit();

    const d: Data = .{ .format = format, .ref = .{ .clip = ref }, .k = k };
    const gf = switch (format.sample) {
        inline else => |s| if (ref != null)
            F.getFrame(Bilateral(s.Type(), true).getFrame)
        else
            F.getFrame(Bilateral(s.Type(), false).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
