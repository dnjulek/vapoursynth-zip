//! avszip_RFS(clip clipa, clip clipb, int[] "frames", bool "mismatch", int[] "planes")
//!
//! Replace the listed frames of clipa with clipb's. Differences from vszip:
//! AviSynth has no variable-format clips, so `mismatch=true` only tolerates a
//! frame-rate difference (output keeps clipa's rate); size/format mismatches
//! are always errors. Audio and parity come from clipa.
//!
//! Property policy (same as vszip's ShufflePlanes path): a fully replaced
//! frame is clipb's frame (its props); with `planes`, the props come from the
//! clip that supplies plane 0.

const std = @import("std");
const avs = @import("avisynth");

const hz = @import("helper.zig");

const allocator = std.heap.c_allocator;
pub const filter_name = "avszip_RFS";
pub const params = "c[clipb]c[frames]i*[mismatch]b[planes]i*";

const Data = struct {
    format: avs.Format,
    other: avs.OwnedClip,
    replace: []bool,
    /// Planes taken from clipb on replaced frames; all true = whole frame.
    planes: [4]bool,

    pub fn deinit(self: *Data) void {
        self.other.deinit();
        allocator.free(self.replace);
    }
};

const F = avs.Filter(Data, filter_name);

fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
    if (!d.replace[@intCast(n)]) return ctx.childFrame(n);

    const np = d.format.numPlanes();
    const whole = for (d.planes[0..np]) |p| {
        if (!p) break false;
    } else true;
    if (whole) return ctx.getFrame(d.other.get(), n);

    const a = try ctx.childFrame(n);
    defer a.release();
    const b = try ctx.getFrame(d.other.get(), n);
    defer b.release();
    const dst = try ctx.newFrame(if (d.planes[0]) b else a);
    const env = ctx.env();
    for (0..np) |p| {
        const id = d.format.planeId(p);
        avs.copyPlane(env, dst, if (d.planes[p]) b else a, id);
    }
    return dst;
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const arg = ctx.named(params);
    const clip = try ctx.clip(0, "clipa");
    defer clip.release();
    const other = try arg.clip("clipb");
    errdefer other.release();

    const vi = clip.videoInfo().*;
    const vi2 = other.videoInfo();
    const format = try hz.planarFormat(ctx, &vi);
    const format2 = try hz.planarFormat(ctx, vi2);
    const mismatch = try arg.boolean("mismatch", false);

    if (vi.width != vi2.width or vi.height != vi2.height) {
        if (mismatch) return ctx.fail("clip dimensions don't match; AviSynth clips cannot have variable dimensions.", .{});
        return ctx.fail("Clip dimensions don't match, enable mismatch if you want variable format.", .{});
    }
    if (!format.eql(format2)) {
        if (mismatch) return ctx.fail("clip formats don't match; AviSynth clips cannot have variable format.", .{});
        return ctx.fail("Clip formats don't match, enable mismatch if you want variable format.", .{});
    }
    if (!mismatch and (vi.fps_numerator != vi2.fps_numerator or vi.fps_denominator != vi2.fps_denominator))
        return ctx.fail("Clip frame rates don't match, enable mismatch if you want variable format.", .{});

    // planes: only meaningful for multi-plane clips (vszip ignores it on gray).
    var planes: [4]bool = @splat(true);
    const np = format.numPlanes();
    if (np > 1 and arg.defined("planes")) {
        planes = @splat(false);
        const v = arg.arg("planes");
        for (0..v.len()) |k| {
            const e = v.at(k);
            if (!e.isInt()) return ctx.fail("planes[{d}] must be an integer.", .{k});
            const p = e.asInt();
            if (p < 0 or p >= np) return ctx.fail("plane index out of range.", .{});
            planes[@intCast(p)] = true;
        }
    }

    if (!arg.defined("frames")) return ctx.fail("frames is required.", .{});
    const replace = try allocator.alloc(bool, @intCast(vi.num_frames));
    errdefer allocator.free(replace);
    @memset(replace, false);
    const frames = arg.arg("frames");
    for (0..frames.len()) |k| {
        const e = frames.at(k);
        if (!e.isInt()) return ctx.fail("frames[{d}] must be an integer.", .{k});
        const f = e.asInt();
        if (f < 0) return ctx.fail("frame index ({d}) must be non-negative.", .{f});
        if (f >= vi.num_frames) return ctx.fail("frame index ({d}) > last frame index ({d}).", .{ f, vi.num_frames - 1 });
        replace[@intCast(f)] = true;
    }

    const d: Data = .{ .format = format, .other = .{ .clip = other }, .replace = replace, .planes = planes };
    const out = try F.create(ctx.env, clip, d, F.getFrame(getFrame), .{ .mt_mode = .nice });
    return avs.Create.returnClip(out);
}
