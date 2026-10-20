//! vszip_XPSNR(clip reference, clip distorted, bool "temporal", bool "verbose")
//!
//! Returns `distorted` with per-frame props XPSNR_Y/U/V. A lower-depth input
//! is first raised to the other's depth (vszip_Dither, plain rounding).
//!
//! Aggregate (verbose) statistics differ from vszip on purpose: vszip adds to
//! running totals on every evaluation, so a frame recomputed by the cache is
//! counted twice. Here each frame index stores its result once and the
//! average printed when the filter is freed is reduced over the evaluated
//! frames in index order (deterministic, repeat-safe). Partial renders report
//! how many of the clip's frames were evaluated.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const filter = @import("../filters/xpsnr.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "vszip_XPSNR";
pub const params = "c[distorted]c[temporal]b[verbose]b";

const PerFrame = struct {
    done: bool = false,
    wdist: [3]f64 = .{ 0, 0, 0 },
    xpsnr: [3]f64 = .{ 0, 0, 0 },
};

const Data = struct {
    format: avs.Format,
    reference: avs.OwnedClip,
    depth: u6 = 0,
    num_comps: u8 = 0,
    frame_rate: u32 = 0,
    max_error_64: u64 = 0,
    width: [3]u32 = .{ 0, 0, 0 },
    height: [3]u32 = .{ 0, 0, 0 },
    temporal: bool = true,
    verbose: bool = true,

    mutex: std.Io.Mutex = .init,
    results: []PerFrame,

    pub fn deinit(self: *Data) void {
        if (self.verbose) self.report();
        self.reference.deinit();
        allocator.free(self.results);
    }

    fn report(self: *Data) void {
        var sum_wdist = [3]f64{ 0, 0, 0 };
        var sum_xpsnr = [3]f64{ 0, 0, 0 };
        var count: u64 = 0;
        for (self.results) |r| {
            if (!r.done) continue;
            count += 1;
            for (0..self.num_comps) |i| {
                sum_wdist[i] += r.wdist[i];
                sum_xpsnr[i] += r.xpsnr[i];
            }
        }
        if (count == 0) return;
        filter.printAverage(common.io, count, self.results.len, sum_wdist, sum_xpsnr, self.num_comps, self.width, self.height, self.max_error_64);
    }
};

const F = avs.Filter(Data, filter_name);

fn XPSNR(comptime T: type) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const env = ctx.env();
            const ref_clip = d.reference.get();
            const src1 = try ctx.getFrame(ref_clip, n);
            defer src1.release();
            var dst = try ctx.childFrame(n); // distorted
            errdefer dst.release();

            // Temporal activity uses the actual previous reference frame(s); a
            // missing one (n == 0/1) means "all zeros", as in vszip.
            const p1 = d.temporal and n > 0;
            const p2 = d.temporal and d.frame_rate >= 32 and n > 1;
            const src_p1: ?avs.Frame = if (p1) try ctx.getFrame(ref_clip, n - 1) else null;
            defer if (src_p1) |f| f.release();
            const src_p2: ?avs.Frame = if (p2) try ctx.getFrame(ref_clip, n - 2) else null;
            defer if (src_p2) |f| f.release();

            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            var orgp: [3][]const T = .{ &.{}, &.{}, &.{} };
            var recp: [3][]const T = .{ &.{}, &.{}, &.{} };
            var strides: [3]u32 = .{ 0, 0, 0 };
            var prv1: ?[]const T = null;
            var prv2: ?[]const T = null;
            for (0..d.num_comps) |p| {
                const id = d.format.planeId(p);
                const raw = src1.plane(T, id);
                const org = try hz.canonical(T, raw, raw.pitch, &scratch);
                orgp[p] = org.fullSlice();
                recp[p] = (try hz.canonical(T, dst.plane(T, id), org.pitch, &scratch)).fullSlice();
                strides[p] = @intCast(org.stride());
                if (p == 0) {
                    if (src_p1) |f| prv1 = (try hz.canonical(T, f.plane(T, id), org.pitch, &scratch)).fullSlice();
                    if (src_p2) |f| prv2 = (try hz.canonical(T, f.plane(T, id), org.pitch, &scratch)).fullSlice();
                }
            }

            var wsse64 = [3]u64{ 0, 0, 0 };
            var cur_xpsnr = [3]f64{ math.inf(f64), math.inf(f64), math.inf(f64) };
            try filter.getWSSE(T, orgp, recp, prv1, prv2, &wsse64, d.width, d.height, strides, d.depth, d.num_comps, d.frame_rate, d.temporal);

            var rec: PerFrame = .{ .done = true };
            for (0..d.num_comps) |i| {
                const sqrt_wsse: f64 = @sqrt(@as(f64, @floatFromInt(wsse64[i])));
                cur_xpsnr[i] = filter.getFrameXPSNR(sqrt_wsse, d.width[i], d.height[i], d.max_error_64);
                rec.wdist[i] = sqrt_wsse;
                rec.xpsnr[i] = cur_xpsnr[i];
            }
            {
                d.mutex.lockUncancelable(common.io);
                defer d.mutex.unlock(common.io);
                d.results[@intCast(n)] = rec;
            }

            try ctx.makePropertyWritable(&dst);
            const props = dst.propsRW(env);
            props.setFloat("XPSNR_Y", cur_xpsnr[0], .replace) catch return ctx.fail("could not set XPSNR_Y.", .{});
            props.setFloat("XPSNR_U", cur_xpsnr[1], .replace) catch return ctx.fail("could not set XPSNR_U.", .{});
            props.setFloat("XPSNR_V", cur_xpsnr[2], .replace) catch return ctx.fail("could not set XPSNR_V.", .{});
            return dst;
        }
    };
}

fn raiseDepth(ctx: *avs.Create, clip: avs.Clip, bits: u8) !avs.Clip {
    const cv = avs.Value.fromClip(clip);
    defer cv.release();
    const args = [_]c.AVS_Value{ cv.raw, c.avs_new_value_int(bits), c.avs_new_value_int(0) };
    const names = [_]?[*:0]const u8{ null, null, "dither_type" };
    var msg: ?[:0]const u8 = null;
    const v = ctx.env.invoke("vszip_Dither", avs.Value.arrayView(&args), &names, &msg) catch
        return ctx.fail("internal vszip_Dither failed: {s}", .{msg orelse "unknown error"});
    defer v.release();
    return v.takeClip(ctx.env);
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const ref_in = try ctx.clip(0, "reference");
    defer ref_in.release();
    const dist_in = try ctx.clip(1, "distorted");
    defer dist_in.release();

    const f1 = try hz.planarFormat(ctx, ref_in.videoInfo());
    const f2 = try hz.planarFormat(ctx, dist_in.videoInfo());
    if (f1.family != .yuv) return ctx.fail("only supports YUV format clips", .{});
    if (f1.bits != 8 and f1.bits != 10) return ctx.fail("only supports 8 or 10 bit clips", .{});
    const vi1 = ref_in.videoInfo();
    if ((vi1.width & 1) != 0 or (vi1.height & 1) != 0) return ctx.fail("only supports even width and height", .{});
    if (f2.isFloat()) return ctx.fail("distorted must be an integer clip.", .{});

    const reference = if (f1.bits < f2.bits) try raiseDepth(ctx, ref_in, f2.bits) else ref_in.retain();
    errdefer reference.release();
    const distorted = if (f1.bits > f2.bits) try raiseDepth(ctx, dist_in, f1.bits) else dist_in.retain();
    defer distorted.release();

    const vi = reference.videoInfo();
    const vi2 = distorted.videoInfo();
    try hz.compareClips(ctx, &.{ vi, vi2 }, .SAME_LEN, .color_only);
    const format = try hz.planarFormat(ctx, vi);

    var d: Data = .{ .format = format, .reference = .{ .clip = reference }, .results = undefined };
    d.temporal = try ctx.boolean(2, "temporal", true);
    d.verbose = try ctx.boolean(3, "verbose", true);
    d.depth = @intCast(format.bits);
    d.max_error_64 = filter.maxError64(d.depth);
    d.frame_rate = if (vi2.fps_denominator != 0)
        @intCast(vi2.fps_numerator / vi2.fps_denominator)
    else if (vi.fps_denominator != 0)
        @intCast(vi.fps_numerator / vi.fps_denominator)
    else
        0;
    d.num_comps = format.color_planes;
    for (0..d.num_comps) |p| {
        d.width[p] = format.planeWidth(@intCast(vi.width), p);
        d.height[p] = format.planeHeight(@intCast(vi.height), p);
    }
    d.results = try allocator.alloc(PerFrame, @intCast(vi.num_frames));
    @memset(d.results, .{});
    errdefer allocator.free(d.results);

    const gf = if (format.sample == .u8) F.getFrame(XPSNR(u8).getFrame) else F.getFrame(XPSNR(u16).getFrame);
    const out = try F.create(ctx.env, distorted, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
