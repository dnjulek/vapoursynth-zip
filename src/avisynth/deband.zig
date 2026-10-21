//! avszip_Deband(clip, int "range", float[] "thr", float[] "grain", int "sample_mode", int "seed",
//!   bool "blur_first", bool "dynamic_grain", bool "keep_tv_range", int "random_algo_ref",
//!   int "random_algo_grain", float "random_param_ref", float "random_param_grain",
//!   float[] "thr1", float[] "thr2", float "angle_boost", float "max_angle")
//!
//! neo_f3kdb core at 16-bit integer or 32-bit float. Like vszip, lower integer
//! depths are processed at 16 bit: up with `avszip_Dither(16, dither_type=0)`,
//! back down with zimg error diffusion (`dither_type=3`), using the range of
//! the source's first frame.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;

const setup = @import("../filters/deband_setup.zig");
const filter_int = @import("../filters/deband_int.zig");
const filter_float = @import("../filters/deband_float.zig");
const hz = @import("helper.zig");
const common = @import("../common.zig");

pub const filter_name = "avszip_Deband";
pub const params = "c[range]i[thr]f*[grain]f*[sample_mode]i[seed]i[blur_first]b[dynamic_grain]b[keep_tv_range]b" ++
    "[random_algo_ref]i[random_algo_grain]i[random_param_ref]f[random_param_grain]f[thr1]f*[thr2]f*[angle_boost]f[max_angle]f";

const Mode = setup.Mode;

const Data = struct {
    format: avs.Format,
    k: setup.Data,

    pub fn deinit(self: *Data) void {
        self.k.tb.deinit(&self.k);
    }
};

const F = avs.Filter(Data, filter_name);

/// Canonical plane stride (elements): rows padded to 64 bytes. The ref tables
/// encode offsets with it, so source and destination must use it exactly.
fn canonStride(width: u32, bytes: u32) u32 {
    return common.ceilN(width * bytes, 64) / bytes;
}

fn F3KDB(comptime T: type, comptime mode: Mode, comptime blur_first: bool, comptime add_grain: [3]bool, comptime np: comptime_int) type {
    return struct {
        fn getFrame(d: *Data, ctx: *avs.FrameCtx, n: c_int) !avs.Frame {
            const src = try ctx.childFrame(n);
            defer src.release();
            const dst = try ctx.newFrame(src);
            errdefer dst.release();
            hz.copyAlpha(ctx.env(), dst, src, d.format);
            var scratch: hz.Scratch = .{};
            defer scratch.deinit();
            const k = &d.k;

            inline for (0..np) |plane| {
                const id = d.format.planeId(plane);
                const stride: u32 = k.strides[if (plane == 0) 0 else 1];
                const pitch = @as(usize, stride) * @sizeOf(T);
                const sp = try hz.canonical(T, src.plane(T, id), pitch, &scratch);
                const dpl = dst.planeMut(T, id);
                // Normally the output frame already has the canonical pitch;
                // otherwise process into a canonical temp and copy.
                const direct = dpl.pitch == pitch;
                const dst_slice: []T = if (direct) dpl.fullSlice() else try scratch.plane(T, stride, dpl.height);

                var grain = if (add_grain[plane]) (if (T == f32) k.tb.grain_float[plane] else k.tb.grain_int[plane]);
                if (add_grain[plane] and k.dynamic_grain) {
                    const offset = k.tb.grain_offsets[@intCast(n)];
                    grain = grain[offset..];
                }

                if (T == f32) {
                    filter_float.processPlane(sp.fullSlice(), dst_slice, k.tb.ref1[plane], k.tb.ref2[plane], grain, stride, dpl.width, dpl.height, k.thr.f[plane], k.thr1.f[plane], k.thr2.f[plane], k.angle_boost, k.max_angle, &k.tb.angles, k.pixel_minf[plane], k.pixel_maxf[plane], mode, blur_first, add_grain[plane]);
                } else {
                    filter_int.processPlane(sp.fullSlice(), dst_slice, k.tb.ref1[plane], k.tb.ref2[plane], grain, stride, dpl.width, dpl.height, k.thr.u[plane], k.thr1.u[plane], k.thr2.u[plane], k.pixel_min[plane], k.pixel_max[plane], k.angle_boost, k.max_angle, &k.tb.angles, mode, blur_first, add_grain[plane]);
                }

                if (!direct) {
                    const tmp: [*]const u8 = @ptrCast(dst_slice.ptr);
                    ctx.env().bitBlt(dpl.ptr, dpl.pitch, tmp, pitch, dpl.rowBytes(), dpl.height);
                }
            }
            return dst;
        }
    };
}

fn selectGetFrame(comptime T: type, mode: Mode, blur_first: bool, add_grain: [3]bool, num_planes: u32) avs.filter.RawGetFrame {
    const comptime_grain: [4][3]bool = .{
        .{ true, false, false },
        .{ false, true, true },
        .{ true, true, true },
        .{ false, false, false },
    };
    const g_idx: u32 = hz.planeIndex(comptime_grain, add_grain, 0);
    return switch (num_planes) {
        inline 1, 3 => |np| switch (mode) {
            inline else => |m| switch (g_idx) {
                inline 0...3 => |gi| switch (blur_first) {
                    inline else => |bf| F.getFrame(F3KDB(T, m, bf, comptime_grain[gi], np).getFrame),
                },
                else => unreachable,
            },
        },
        else => unreachable,
    };
}

fn floatArg(ctx: *avs.Create, i: usize, comptime key: []const u8, default: f64, min: f64, max: f64) !f64 {
    return ctx.float(f64, i, key, default, min, max);
}

/// vszip's Maps.getArray: up to `max_len` values, the last one repeating;
/// `default` when absent.
fn arrayArg(ctx: *avs.Create, i: usize, comptime key: []const u8, comptime max_len: usize, default: [3]f64, min: f64, max: f64) ![3]f64 {
    var buf: [max_len]f64 = undefined;
    const got = (try ctx.floatArray(f64, i, key, &buf, min, max)) orelse return default;
    if (got.len == 0) return default;
    var out: [3]f64 = undefined;
    for (0..3) |k| out[k] = got[@min(k, got.len - 1)];
    return out;
}

fn intArg(ctx: *avs.Create, i: usize, comptime key: []const u8, default: i64, min: i64, max: i64) !i32 {
    return @intCast(try ctx.int(i64, i, key, default, min, max));
}

/// Call another script function from create; returns an owned clip.
fn invokeClip(ctx: *avs.Create, name: [:0]const u8, args: []const c.AVS_Value, names: []const ?[*:0]const u8) !avs.Clip {
    var msg: ?[:0]const u8 = null;
    const v = ctx.env.invoke(name, avs.Value.arrayView(args), names, &msg) catch
        return ctx.fail("internal {s} failed: {s}", .{ name, msg orelse "unknown error" });
    defer v.release();
    if (!v.isClip()) return ctx.fail("internal {s} did not return a clip.", .{name});
    return v.takeClip(ctx.env);
}

/// `range`: the source's tagged range, or null to let avszip_Dither use its
/// defaults (limited for YUV/Gray, full for RGB: zimg's defaults too) without
/// stamping a range property.
fn ditherTo(ctx: *avs.Create, clip: avs.Clip, bits: c_int, dither_type: c_int, range: ?hz.Range) !avs.Clip {
    const cv = avs.Value.fromClip(clip);
    defer cv.release();
    if (range) |r| {
        const full = c.avs_new_value_bool(@intFromBool(r == .FULL));
        const args = [_]c.AVS_Value{ cv.raw, c.avs_new_value_int(bits), c.avs_new_value_int(dither_type), full, full };
        const names = [_]?[*:0]const u8{ null, null, "dither_type", "fulls", "fulld" };
        return invokeClip(ctx, "avszip_Dither", &args, &names);
    }
    const args = [_]c.AVS_Value{ cv.raw, c.avs_new_value_int(bits), c.avs_new_value_int(dither_type) };
    const names = [_]?[*:0]const u8{ null, null, "dither_type" };
    return invokeClip(ctx, "avszip_Dither", &args, &names);
}

pub fn create(ctx: *avs.Create) !avs.Value {
    const input = try ctx.clip(0, "clip");
    defer input.release();
    const vi_in = input.videoInfo();
    const format_in = try hz.planarFormat(ctx, vi_in);

    const thr_in = try arrayArg(ctx, 2, "thr", 3, .{ 0.99, 0.99, 0.99 }, 0, 255);
    const inputs: setup.Data.Inputs = .{
        .range = try intArg(ctx, 1, "range", 15, 0, 255),
        .thr = thr_in,
        .grain = try arrayArg(ctx, 3, "grain", 2, .{ 0, 0, 0 }, 0, 127),
        .sample_mode = try intArg(ctx, 4, "sample_mode", 2, 1, 7),
        .seed = try intArg(ctx, 5, "seed", 0, math.minInt(i32), math.maxInt(i32)),
        .blur_first = try ctx.boolean(6, "blur_first", true),
        .dynamic_grain = try ctx.boolean(7, "dynamic_grain", false),
        .keep_tv_range = try ctx.boolean(8, "keep_tv_range", false),
        .random_algo_ref = try intArg(ctx, 9, "random_algo_ref", 1, 0, 2),
        .random_algo_grain = try intArg(ctx, 10, "random_algo_grain", 1, 0, 2),
        .random_param_ref = try floatArg(ctx, 11, "random_param_ref", 1.0, 0, 255),
        .random_param_grain = try floatArg(ctx, 12, "random_param_grain", 1.0, 0, 255),
        .thr1 = try arrayArg(ctx, 13, "thr1", 3, thr_in, 0, 255),
        .thr2 = try arrayArg(ctx, 14, "thr2", 3, thr_in, 0, 255),
        .angle_boost = @floatCast(try floatArg(ctx, 15, "angle_boost", 1.5, 0, math.maxInt(u16))),
        .max_angle = @floatCast(try floatArg(ctx, 16, "max_angle", 0.15, 0, 1)),
    };

    // Integer clips below 16 bit are processed at 16 bit (vszip's path).
    const low_bits = !format_in.isFloat() and format_in.bits < 16;
    const full = hz.getRangeProp(ctx.env, input);
    const clip = if (low_bits) try ditherTo(ctx, input, 16, 0, full) else input.retain();
    defer clip.release();
    const vi = clip.videoInfo();
    const format = try hz.planarFormat(ctx, vi);

    const bytes: u32 = format.sample.bytes();
    const w: u32 = @intCast(vi.width);
    var k: setup.Data = .{
        .width = vi.width,
        .height = vi.height,
        .num_frames = vi.num_frames,
        .is_int = !format.isFloat(),
        .is_yuv = format.family == .yuv,
        .num_planes = format.color_planes,
        .strides = .{ canonStride(w, bytes), canonStride(w >> format.ssw, bytes) },
        .ssw = format.ssw,
        .ssh = format.ssh,
    };
    k.configure(inputs);
    k.tb = setup.TempBuff.init(&k) catch return ctx.fail("failed to initialize temp buffer (out of memory)", .{});
    var d: Data = .{ .format = format, .k = k };

    const gf = switch (format.sample) {
        .f32 => selectGetFrame(f32, k.sample_mode, k.blur_first, k.add_grain, format.color_planes),
        else => selectGetFrame(u16, k.sample_mode, k.blur_first, k.add_grain, format.color_planes),
    };
    // On success the filter owns the tables (no errdefer past this point).
    const flt = F.create(ctx.env, clip, d, gf, .{ .mt_mode = .nice }) catch |e| {
        d.deinit();
        return e;
    };

    if (!low_bits) return avs.Create.returnClip(flt);
    defer flt.release();
    const down = try ditherTo(ctx, flt, format_in.bits, 3, full);
    return avs.Create.returnClip(down);
}
