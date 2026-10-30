//! Host-free tests: every kernel in `filters/` is compiled here without the
//! VapourSynth or AviSynth binding (this test root imports neither), which
//! proves the shared kernels carry no host dependency. Generic kernels are
//! additionally instantiated for the sample types the plugins use, so lazy
//! analysis cannot hide a host import or an invalid body.

const std = @import("std");

const kernels = .{
    @import("filters/adaptive_grain_mask.zig"),
    @import("filters/bilateral.zig"),
    @import("filters/bilateral_dither.zig"),
    @import("filters/bilateral_dither_subspl.zig"),
    @import("filters/bm3d_common.zig"),
    @import("filters/bm3d.zig"),
    @import("filters/boxblur_comptime.zig"),
    @import("filters/boxblur_runtime.zig"),
    @import("filters/butteraugli.zig"),
    @import("filters/checkmate.zig"),
    @import("filters/clahe.zig"),
    @import("filters/color_map.zig"),
    @import("filters/comb_mask.zig"),
    @import("filters/comb_mask_mt.zig"),
    @import("filters/compress.zig"),
    @import("filters/deband_angle.zig"),
    @import("filters/deband_float.zig"),
    @import("filters/deband_int.zig"),
    @import("filters/deband_setup.zig"),
    @import("filters/dither.zig"),
    @import("filters/dither_errdiff.zig"),
    @import("filters/dither_pattern.zig"),
    @import("filters/eedi3.zig"),
    @import("filters/image_read.zig"),
    @import("filters/limit_filter.zig"),
    @import("filters/limiter.zig"),
    @import("filters/mosquito_nr.zig"),
    @import("filters/mosquito_nr_float.zig"),
    @import("filters/nlmeans.zig"),
    @import("filters/dovi.zig"),
    @import("filters/planeaverage.zig"),
    @import("filters/planeminmax.zig"),
    @import("filters/simd.zig"),
    @import("filters/srgb.zig"),
    @import("filters/ssimulacra2.zig"),
    @import("filters/tonemap.zig"),
    @import("filters/tonemap_color.zig"),
    @import("filters/wnnm.zig"),
    @import("filters/xpsnr.zig"),
    @import("vcl.zig"),
    @import("common.zig"),
};

test "kernels compile without a host binding" {
    inline for (kernels) |k| std.testing.refAllDecls(k);
}

const limiter = @import("filters/limiter.zig");
const planeaverage = @import("filters/planeaverage.zig");
const boxblur_rt = @import("filters/boxblur_runtime.zig");
const limit_filter = @import("filters/limit_filter.zig");

test "limiter clampSlice matches scalar clamp at SIMD boundaries" {
    var src: [131]u16 = undefined;
    var dst: [131]u16 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast((i * 997) % 65536);
    inline for (.{ 1, 15, 16, 17, 63, 64, 65, 131 }) |n| {
        limiter.clampSlice(u16, dst[0..n], src[0..n], 1000, 50000);
        for (dst[0..n], src[0..n]) |d, s| try std.testing.expectEqual(@min(@max(1000, s), 50000), d);
    }
    var fs: [37]f32 = undefined;
    var fd: [37]f32 = undefined;
    for (&fs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 18.0 - 0.5;
    limiter.clampSlice(f32, &fd, &fs, -0.25, 0.75);
    for (fd, fs) |d, s| try std.testing.expectEqual(@min(@max(-0.25, s), 0.75), d);
}

test "planeaverage matches a scalar oracle with stride padding" {
    const w = 37;
    const h = 5;
    const stride = 48;
    var buf: [stride * h]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @intCast((i * 31 + 7) % 256);
    var sum: u64 = 0;
    for (0..h) |y| for (0..w) |x| {
        sum += buf[y * stride + x];
    };
    const want = @as(f64, @floatFromInt(sum)) / (w * h) / 255.0;
    const ex = planeaverage.Exclude{ .i = &.{} };
    try std.testing.expectApproxEqRel(want, planeaverage.average(u8, &buf, stride, w, h, ex, 255), 1e-12);
}

const boxblur_ct = @import("filters/boxblur_comptime.zig");
const bilateral = @import("filters/bilateral.zig");
const eedi3 = @import("filters/eedi3.zig");
const planeminmax = @import("filters/planeminmax.zig");

/// Non-generic wrappers around representative generic kernels. Taking their
/// address forces semantic analysis of the wrapper bodies, i.e. of each
/// kernel instantiated for `T` (taking `&kernel` of a generic function does
/// not). The plugin builds instantiate every kernel they use; this keeps the
/// host-free check honest for the shared drivers.
fn Inst(comptime T: type) type {
    return struct {
        const Tmp = struct {
            pub fn get(_: @This()) error{OutOfMemory}![]T {
                return &.{};
            }
        };
        const Sink = struct {
            pub fn setInt(_: *@This(), _: [:0]const u8, _: i64, _: anytype) void {}
            pub fn setFloat(_: *@This(), _: [:0]const u8, _: f64, _: anytype) void {}
        };

        fn blurPlane(p: boxblur_rt.Params, s: []const T, d: []T, sc: *const boxblur_rt.Scratch(T)) !void {
            try boxblur_rt.blurPlane(T, p, s, d, 1, 1, 1, sc, Tmp{});
        }
        fn hvBlur(s: []const T, d: []T) !void {
            try boxblur_ct.hvBlur(T, 3, s, d, 1, 1, 1);
        }
        fn bilat(s: []const T, d: []T, k: *bilateral.Data) !void {
            try bilateral.bilateral(T, s, s, d, 1, 1, 1, 0, false, k);
        }
        fn interp(k: *const eedi3.Data, sc: *eedi3.Scratch, s: []const T, d: []T) void {
            eedi3.processPlane(T, k, sc, s, d, null, null, 0, 0, 1, 1, 1, 1);
        }
        fn minMax(s: []const T, k: *planeminmax.Data) !void {
            var sink: Sink = .{};
            try planeminmax.minMax(T, s, 1, &sink, 1, 1, k);
        }
    };
}

test "generic kernels instantiate for every plugin sample type" {
    inline for (.{ u8, u16, f32 }) |T| {
        const I = Inst(T);
        _ = &I.blurPlane;
        _ = &I.bilat;
        _ = &I.interp;
        _ = &I.minMax;
        if (T == f32) _ = &I.hvBlur; // CT kernels are float-only in the plugins
        const s: []const T = &.{};
        const d: []T = &.{};
        limit_filter.process(T, s, s, s, d, 1, 1, 2);
        limiter.clampSlice(T, d, s, 0, 1);
    }
}

// ---------------------------------------------------------------------------
// Allocation-failure cleanup: fail every allocation position in turn and
// require that nothing stays allocated (after the error, and after normal
// destruction once construction succeeds).
// ---------------------------------------------------------------------------

const common = @import("common.zig");
const deband_setup = @import("filters/deband_setup.zig");
const xpsnr = @import("filters/xpsnr.zig");

fn faultSweep(ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) !void {
    var i: usize = 0;
    while (true) : (i += 1) {
        var fa = std.testing.FailingAllocator.init(std.heap.c_allocator, .{ .fail_index = i });
        common.test_hooks.backing = fa.allocator();
        defer common.test_hooks.backing = std.heap.c_allocator;
        const r = run(ctx);
        std.testing.expectEqual(fa.allocated_bytes, fa.freed_bytes) catch |e| {
            std.debug.print("fail_index {d}: {d} bytes left allocated\n", .{ i, fa.allocated_bytes - fa.freed_bytes });
            return e;
        };
        if (r) |_| {
            if (!fa.has_induced_failure) {
                try std.testing.expect(i > 0); // the sweep exercised some allocation
                return;
            }
        } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
    }
}

test "eedi3 allocScratch releases partial construction" {
    const Case = struct { hp: bool, t_rows: u32 };
    inline for (.{ Case{ .hp = false, .t_rows = 0 }, Case{ .hp = true, .t_rows = 16 } }) |cs| {
        try faultSweep(cs, struct {
            fn run(c: Case) anyerror!void {
                const s = try eedi3.allocScratch(64, 64, 8, c.hp, c.t_rows, 2 * c.t_rows, 2);
                eedi3.freeScratch(s);
            }
        }.run);
    }
}

test "deband TempBuff.init releases partial construction" {
    inline for (.{ true, false }) |is_int| {
        var d: deband_setup.Data = .{ .width = 32, .height = 16, .num_frames = 3, .is_int = is_int, .is_yuv = true, .strides = .{ 64, 64 }, .ssw = 1, .ssh = 1 };
        d.configure(.{
            .thr = .{ 1, 1, 1 },
            .thr1 = .{ 1, 1, 1 },
            .thr2 = .{ 1, 1, 1 },
            .grain = .{ 4, 4, 4 },
            .sample_mode = 2,
            .range = 15,
            .seed = 0,
            .blur_first = true,
            .dynamic_grain = true,
            .keep_tv_range = false,
            .angle_boost = 1.5,
            .max_angle = 0.15,
            .random_param_ref = 1,
            .random_param_grain = 1,
            .random_algo_ref = 1,
            .random_algo_grain = 1,
        });
        try faultSweep(&d, struct {
            fn run(dd: *deband_setup.Data) anyerror!void {
                var tb = try deband_setup.TempBuff.init(dd);
                tb.deinit(dd);
            }
        }.run);
    }
}

test "bilateral init and PBFIC release partial allocations" {
    try faultSweep({}, struct {
        fn run(_: void) anyerror!void {
            var k = try bilateral.Data.init(.{
                .sigmaS = .{ 3, 3, 3 },
                .sigmaR = .{ 0.02, 0.02, 0.02 },
                .algorithm = .{ 1, 2, 0 },
                .PBFICnum = .{ 0, 0, 0 },
                .planes = .{ true, true, false },
                .yuv = true,
                .plane_w = .{ 32, 32, 32 },
                .plane_h = .{ 32, 32, 32 },
                .hist_len = 256,
            });
            defer k.deinit();
            var src: [32 * 32]u8 = @splat(100);
            var dst: [32 * 32]u8 = undefined;
            try bilateral.bilateral(u8, &src, &src, &dst, 32, 32, 32, 0, false, &k);
        }
    }.run);
}

test "xpsnr getWSSE releases partial allocations" {
    try faultSweep({}, struct {
        fn run(_: void) anyerror!void {
            const w = 256;
            const h = 128;
            var org: [w * h]u8 = @splat(50);
            var rec: [w * h]u8 = @splat(52);
            var wsse = [3]u64{ 0, 0, 0 };
            try xpsnr.getWSSE(u8, .{ &org, &.{}, &.{} }, .{ &rec, &.{}, &.{} }, null, null, &wsse, .{ w, 0, 0 }, .{ h, 0, 0 }, .{ w, 0, 0 }, 8, 1, 24, false);
        }
    }.run);
}

const nlmeans = @import("filters/nlmeans.zig");

test "nlmeans validates its config and bounds the tile working set" {
    try std.testing.expectError(error.A, nlmeans.Config.init(.{ .a = 0, .width = 64, .height = 64 }));
    try std.testing.expectError(error.Window, nlmeans.Config.init(.{ .a = 40, .width = 64, .height = 64 }));
    // 1e-30 squared underflows f32: 1/h^2 would be inf and every weight NaN
    try std.testing.expectError(error.H, nlmeans.Config.init(.{ .h = 1e-30, .width = 64, .height = 64 }));
    const big = try nlmeans.Config.init(.{ .d = 16, .ref = .rgb, .width = 1920, .height = 1080 });
    try std.testing.expect(big.tw < 512 and big.th == 16);
    const small = try nlmeans.Config.init(.{ .width = 20, .height = 9 });
    try std.testing.expect(small.tw >= 20 and small.th == 9);
}

test "nlmeans keeps a flat interior flat" {
    // Away from the zero border every candidate equals the pixel, so the
    // weighted mean must reproduce it. 70x37 spans a partial tile, a
    // partial vector and a 4-row batch remainder.
    const w = 70;
    const h = 37;
    const cfg = try nlmeans.Config.init(.{ .d = 1, .width = w, .height = h });
    const sc = try nlmeans.Scratch.create(&cfg, false);
    defer sc.destroy();
    var src: [w * h]u8 = @splat(77);
    var dst: [w * h]u8 = @splat(0);
    const In = nlmeans.PlaneIn(u8);
    const planes: [3]In = @splat(.{ .data = &src, .stride = w });
    nlmeans.processFrame(u8, .luma, 0, &cfg, sc, &planes, null, .{.{ .data = &dst, .stride = w }}, 1, 255);
    for (2..h - 2) |y| for (2..w - 2) |x| {
        try std.testing.expectEqual(@as(u8, 77), dst[y * w + x]);
    };
}

const wnnm = @import("filters/wnnm.zig");

test "wnnm search runs are the windows' union in (y, x) order" {
    const gpa = std.testing.allocator;
    var runs: std.ArrayList(wnnm.Run) = .empty;
    defer runs.deinit(gpa);
    const w = 40;
    const h = 30;
    const bs = 8;
    const cols = w - bs + 1;
    const rows = h - bs + 1;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    for (0..500) |_| {
        const n = rand.intRangeAtMost(usize, 1, 6);
        const r = rand.intRangeAtMost(i32, 0, 12);
        var centers: [6]wnnm.Match = undefined;
        var inside: [rows * cols]bool = @splat(false);
        for (centers[0..n]) |*c| {
            c.* = .{ .err = 0, .x = rand.intRangeLessThan(i32, 0, cols), .y = rand.intRangeLessThan(i32, 0, rows), .t = 0 };
            var y = @max(c.y - r, 0);
            while (y <= @min(c.y + r, rows - 1)) : (y += 1) {
                var x = @max(c.x - r, 0);
                while (x <= @min(c.x + r, cols - 1)) : (x += 1) inside[@intCast(y * cols + x)] = true;
            }
        }
        try wnnm.generateSearchRuns(&centers, n, bs, w, h, @intCast(r), &runs, gpa);

        var total: u32 = 0;
        var prev: i32 = -1;
        for (runs.items, 0..) |run, k| {
            try std.testing.expect(run.len > 0);
            // maximal: runs on one row never touch
            if (k > 0 and runs.items[k - 1].y == run.y) {
                try std.testing.expect(runs.items[k - 1].x + @as(i32, @intCast(runs.items[k - 1].len)) < run.x);
            }
            for (0..run.len) |j| {
                const x = run.x + @as(i32, @intCast(j));
                const pos = run.y * cols + x;
                try std.testing.expect(pos > prev and inside[@intCast(pos)]);
                try std.testing.expectEqual(wnnm.Loc{ .x = x, .y = run.y }, wnnm.runLoc(runs.items, total));
                prev = pos;
                total += 1;
            }
        }
        try std.testing.expectEqual(std.mem.count(bool, &inside, &.{true}), total);
    }
}

test "nlmeans Scratch.create releases partial construction" {
    inline for (.{ false, true }) |has_ref| {
        try faultSweep(has_ref, struct {
            fn run(r: bool) anyerror!void {
                const cfg = nlmeans.Config.init(.{ .width = 64, .height = 48 }) catch unreachable;
                const sc = try nlmeans.Scratch.create(&cfg, r);
                sc.destroy();
            }
        }.run);
    }
}
