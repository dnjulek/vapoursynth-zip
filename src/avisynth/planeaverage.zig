//! avszip_PlaneAverage(clip clipa, int[] "exclude", clip "clipb", int[] "planes", string "prop")
//!
//! Property-only: pixels pass through (shared, copy-on-write props). Writes
//! `<prop>Avg` (and `<prop>Diff` with clipb) per selected plane, appended in
//! plane order, exactly like vszip.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/planeaverage.zig");
const hz = @import("helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_PlaneAverage";
pub const params = "c[exclude]i*[clipb]c[planes]i*[prop]s";

const Data = struct {
    format: avs.Format,
    ref: avs.OwnedClip = .{},
    exclude: filter.Exclude,
    peak: f32,
    planes: [4]bool,
    prop_d: [:0]u8,
    prop_a: [:0]u8,

    pub fn deinit(self: *Data) void {
        self.ref.deinit();
        switch (self.exclude) {
            .i => |e| allocator.free(e),
            .f => |e| allocator.free(e),
        }
        allocator.free(self.prop_d);
        allocator.free(self.prop_a);
    }
};

const F = avs.Filter(Data, filter_name);

fn PlaneAverage(comptime T: type, comptime refb: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const env = ctx.env();
            var frame = try ctx.childFrame(n);
            errdefer frame.release();
            const ref = if (refb) try ctx.getFrame(d.ref.get(), n);
            defer if (refb) ref.release();

            // Share the pixels; only the property map gets a private copy.
            try ctx.makePropertyWritable(&frame);
            var sink: hz.PropSink = .{ .rw = frame.propsRW(env) };
            _ = sink.rw.delete(d.prop_d);
            _ = sink.rw.delete(d.prop_a);

            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.numPlanes()) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const sp = frame.plane(T, id);
                const srcp = try hz.canonical(T, sp, sp.pitch, &scratch);
                const stride: u32 = @intCast(srcp.stride());
                var avg: f64 = undefined;
                if (refb) {
                    const refp = try hz.canonical(T, ref.plane(T, id), srcp.pitch, &scratch);
                    const stats = filter.averageRef(T, srcp.fullSlice(), refp.fullSlice(), stride, srcp.width, srcp.height, d.exclude, d.peak);
                    sink.setFloat(d.prop_d, stats.diff, .Append);
                    avg = stats.avg;
                } else {
                    avg = filter.average(T, srcp.fullSlice(), stride, srcp.width, srcp.height, d.exclude, d.peak);
                }
                sink.setFloat(d.prop_a, avg, .Append);
            }
            if (sink.failed) return ctx.fail("could not set the frame properties.", .{});
            return frame;
        }
    };
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clipa");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    if (!arg.defined("exclude")) return ctx.fail("exclude is required (use exclude=[-1] to exclude nothing).", .{});

    const ref = try arg.optClip("clipb");
    errdefer if (ref) |r| r.release();
    var planes = try arg.planes("planes", format.numPlanes(), false);
    if (!arg.defined("planes")) planes[0] = true; // vszip default: planes=[0]
    // A selected alpha plane is read from clipb too.
    if (ref) |r| try hz.compareClips(ctx, &.{ vi, r.videoInfo() }, .BIGGER_THAN, if (planes[3]) .same else .color_only);

    const prop_in = (try arg.string("prop")) orelse "psm";
    const prop_d = try hz.dupeSuffix(prop_in, "Diff");
    errdefer allocator.free(prop_d);
    const prop_a = try hz.dupeSuffix(prop_in, "Avg");
    errdefer allocator.free(prop_a);

    const ex = arg.arg("exclude");
    var exclude: filter.Exclude = undefined;
    if (format.isFloat()) {
        const f = try allocator.alloc(f32, ex.len());
        errdefer allocator.free(f);
        for (f, 0..) |*df, k| {
            const e = ex.at(k);
            if (!e.isInt()) return ctx.fail("exclude[{d}] must be an integer.", .{k});
            df.* = @floatFromInt(e.asInt());
        }
        exclude = .{ .f = f };
    } else {
        // The kernels compare the i32-widened pixel against i32 excludes, so a
        // value outside T's storage range can never match: strip them (the
        // never-match sentinel is exclude=[-1]) so getFrame takes the
        // compare-free fast path on an empty slice.
        const tmax: i64 = if (format.sample == .u8) math.maxInt(u8) else math.maxInt(u16);
        var m: usize = 0;
        for (0..ex.len()) |k| {
            const e = ex.at(k);
            if (!e.isInt()) return ctx.fail("exclude[{d}] must be an integer.", .{k});
            if (e.asInt() >= 0 and e.asInt() <= tmax) m += 1;
        }
        const ei = try allocator.alloc(i32, m);
        var idx: usize = 0;
        for (0..ex.len()) |k| {
            const e = ex.at(k).asInt();
            if (e >= 0 and e <= tmax) {
                ei[idx] = @intCast(e);
                idx += 1;
            }
        }
        exclude = .{ .i = ei };
    }
    errdefer switch (exclude) {
        .i => |e| allocator.free(e),
        .f => |e| allocator.free(e),
    };

    const d: Data = .{
        .format = format,
        .ref = .{ .clip = ref },
        .exclude = exclude,
        .peak = if (format.isFloat()) @floatFromInt((@as(u64, 1) << 32) - 1) else @floatCast(format.peak()),
        .planes = planes,
        .prop_d = prop_d,
        .prop_a = prop_a,
    };
    const gf = switch (format.sample) {
        inline else => |s| if (ref != null)
            F.getFrame(PlaneAverage(s.Type(), true).getFrame)
        else
            F.getFrame(PlaneAverage(s.Type(), false).getFrame),
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
