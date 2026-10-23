//! avszip_BoxBlur(clip, int[] "planes", int "hradius", int "hpasses", int "vradius", int "vpasses")

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const boxblur_ct = @import("../filters/boxblur_comptime.zig");
const boxblur_rt = @import("../filters/boxblur_runtime.zig");
const hz = @import("helper.zig");

pub const filter_name = "avszip_BoxBlur";
pub const params = "c[planes]i*[hradius]i[hpasses]i[vradius]i[vpasses]i";

const Data = struct {
    format: avs.Format,
    width: u32,
    p: boxblur_rt.Params = undefined,
    tmp_size: u32 = 0,
    planes: [4]bool,
};

const F = avs.Filter(Data, filter_name);

fn BoxBlurCT(comptime T: type, comptime radius: comptime_int) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
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
                try boxblur_ct.hvBlur(T, radius, sp.fullSlice(), dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height);
            }
            return dst;
        }
    };
}

fn BoxBlurRT(comptime T: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyUnprocessed(ctx.env(), dst, src, d.format, &d.planes);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            // one scratch block per frame, shared by every plane (plane 0 is the widest)
            const sc = boxblur_rt.Scratch(T).init(d.width, d.tmp_size, d.p.vradius, @intCast(@max(d.p.vpasses, 1))) catch
                return ctx.fail("out of memory", .{});
            defer sc.deinit();

            for (0..d.format.color_planes) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const dp = dst.planeMut(T, id);
                const sp = try hz.canonical(T, src.plane(T, id), dp.pitch, &scratch);
                const tmp = struct {
                    scratch: *hz.Scratch,
                    stride: usize,
                    h: usize,

                    pub fn get(self: @This()) error{OutOfMemory}![]T {
                        return self.scratch.plane(T, self.stride, self.h);
                    }
                }{ .scratch = &scratch, .stride = dp.stride(), .h = dp.height };
                try boxblur_rt.blurPlane(T, d.p, sp.fullSlice(), dp.fullSlice(), @intCast(dp.stride()), dp.width, dp.height, &sc, tmp);
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

    var d: Data = .{
        .format = format,
        .width = @intCast(vi.width),
        .tmp_size = @intCast(@max(vi.width, vi.height)),
        .planes = try ctx.planes(1, format.color_planes, true),
    };
    d.p = .{
        .hradius = try ctx.int(u32, 2, "hradius", 1, 0, math.maxInt(i32)),
        .hpasses = try ctx.int(i32, 3, "hpasses", 1, math.minInt(i32), math.maxInt(i32)),
        .vradius = try ctx.int(u32, 4, "vradius", 1, 0, math.maxInt(i32)),
        .vpasses = try ctx.int(i32, 5, "vpasses", 1, math.minInt(i32), math.maxInt(i32)),
    };

    const np = format.color_planes;
    var pw: [3]u32 = undefined;
    var ph: [3]u32 = undefined;
    for (0..np) |p| {
        pw[p] = format.planeWidth(@intCast(vi.width), p);
        ph[p] = format.planeHeight(@intCast(vi.height), p);
    }
    const int_bytes: u8 = if (format.isFloat()) 0 else @intCast(format.sample.bytes());
    d.p.check(int_bytes, d.planes[0..np], pw[0..np], ph[0..np]) catch |e| return ctx.fail("{s}", .{boxblur_rt.Params.message(e)});

    // CT vs RT: see boxblur_rt.Params.useRT.
    const use_rt = d.p.useRT(format.isFloat());
    const gf = if (use_rt) switch (format.sample) {
        inline else => |s| F.getFrame(BoxBlurRT(s.Type()).getFrame),
    } else switch (d.p.hradius) {
        inline 1...boxblur_rt.Params.ct_max_radius => |r| F.getFrame(BoxBlurCT(f32, r).getFrame),
        else => unreachable,
    };

    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
