//! vszip_PlaneMinMax(clip clipa, float "minthr", float "maxthr", clip "clipb", int[] "planes", string "prop")
//!
//! Property-only: writes `<prop>Min`/`<prop>Max` (and `<prop>Diff` with clipb)
//! per selected plane, appended in plane order, exactly like vszip.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");

const filter = @import("../filters/planeminmax.zig");
const hz = @import("helper.zig");

pub const filter_name = "vszip_PlaneMinMax";
pub const params = "c[minthr]f[maxthr]f[clipb]c[planes]i*[prop]s";

const Data = struct {
    format: avs.Format,
    ref: avs.OwnedClip = .{},
    planes: [4]bool,
    k: filter.Data,

    pub fn deinit(self: *Data) void {
        self.ref.deinit();
        self.k.deinit();
    }
};

const F = avs.Filter(Data, filter_name);

fn PlaneMinMax(comptime T: type, comptime refb: bool, comptime no_thr: bool) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const env = ctx.env();
            var frame = try ctx.childFrame(n);
            errdefer frame.release();
            const ref = if (refb) try ctx.getFrame(d.ref.get(), n);
            defer if (refb) ref.release();

            try ctx.makePropertyWritable(&frame);
            var sink: hz.PropSink = .{ .rw = frame.propsRW(env) };
            _ = sink.rw.delete(d.k.prop.d);
            _ = sink.rw.delete(d.k.prop.ma);
            _ = sink.rw.delete(d.k.prop.mi);

            var scratch: hz.Scratch = .{};
            defer scratch.deinit();

            for (0..d.format.numPlanes()) |plane| {
                if (!d.planes[plane]) continue;
                const id = d.format.planeId(plane);
                const sp = frame.plane(T, id);
                const srcp = try hz.canonical(T, sp, sp.pitch, &scratch);
                const stride: u32 = @intCast(srcp.stride());
                const w = srcp.width;
                const h = srcp.height;
                if (refb) {
                    const refp = try hz.canonical(T, ref.plane(T, id), srcp.pitch, &scratch);
                    if (no_thr) {
                        filter.minMaxNoThrRef(T, srcp.fullSlice(), refp.fullSlice(), stride, &sink, w, h, &d.k);
                    } else {
                        try filter.minMaxRef(T, srcp.fullSlice(), refp.fullSlice(), stride, &sink, w, h, &d.k);
                    }
                } else {
                    if (no_thr) {
                        filter.minMaxNoThr(T, srcp.fullSlice(), stride, &sink, w, h, &d.k);
                    } else {
                        try filter.minMax(T, srcp.fullSlice(), stride, &sink, w, h, &d.k);
                    }
                }
            }
            if (sink.failed) return ctx.fail("could not set the frame properties.", .{});
            return frame;
        }
    };
}

fn getThr(ctx: *avs.Create, i: usize, comptime key: []const u8) !f32 {
    const v = ctx.arg(i);
    if (!v.isDefined()) return 0;
    const thr = v.asFloat();
    if (!math.isFinite(thr)) return ctx.fail(key ++ " value {d} is not a finite number.", .{thr});
    if (thr < 0 or thr > 1) return ctx.fail(key ++ " should be a float between 0.0 and 1.0", .{});
    return @floatCast(thr);
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const clip = try ctx.clip(0, "clipa");
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    const ref = try ctx.optClip(3, "clipb");
    errdefer if (ref) |r| r.release();
    var planes = try ctx.planes(4, format.numPlanes(), false);
    if (!ctx.defined(4)) planes[0] = true; // vszip default: planes=[0]
    // A selected alpha plane is read from clipb too.
    if (ref) |r| try hz.compareClips(ctx, &.{ vi, r.videoInfo() }, .BIGGER_THAN, if (planes[3]) .same else .color_only);

    const maxthr = try getThr(ctx, 2, "maxthr");
    const minthr = try getThr(ctx, 1, "minthr");

    const no_thr = maxthr == 0 and minthr == 0;
    if ((planes[1] or planes[2]) and !no_thr and format.family == .yuv and format.isFloat())
        return ctx.fail("you can't use maxthr/minthr with float chroma, use planes=[0] or maxthr/minthr=0", .{});

    const prop_in = (try ctx.string(5, "prop")) orelse "psm";
    var k = try filter.Data.init(format.isFloat(), format.bits, minthr, maxthr, prop_in);
    errdefer k.deinit();

    const d: Data = .{ .format = format, .ref = .{ .clip = ref }, .planes = planes, .k = k };
    const gf = switch (format.sample) {
        inline else => |s| switch (ref != null) {
            inline else => |refb| switch (no_thr) {
                inline else => |nt| F.getFrame(PlaneMinMax(s.Type(), refb, nt).getFrame),
            },
        },
    };
    const out = try F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
