//! AviSynth-side helpers shared by the adapters in `avisynth/`: format and
//! clip validation, range/peak/scale conventions (the arithmetic is shared
//! with the VapourSynth side in common.zig) and plane plumbing over
//! independent pitches.

const std = @import("std");
const math = std.math;
const avs = @import("avisynth");
const c = avs.c;
const common = @import("../common.zig");

pub const Format = avs.Format;

/// Integer bit depths AviSynth+ provides, plus float. 9-bit and u32/f16
/// formats of vszip have no AviSynth+ planar equivalent.
pub const BPSType = enum {
    U8,
    U10,
    U12,
    U14,
    U16,
    F32,

    pub fn fromFormat(f: Format) BPSType {
        return switch (f.bits) {
            8 => .U8,
            10 => .U10,
            12 => .U12,
            14 => .U14,
            16 => .U16,
            else => .F32,
        };
    }
};

/// Planar format of `clip` or a create error.
pub fn planarFormat(ctx: *avs.Create, vi: *const c.AVS_VideoInfo) error{Failed}!Format {
    if (vi.width == 0) return ctx.fail("clip must have video.", .{});
    return Format.fromVideoInfo(vi) catch ctx.fail("only planar Y, YUV(A) and RGB(A) formats are supported.", .{});
}

pub const ClipLen = enum {
    SAME_LEN,
    BIGGER_THAN,
    MISMATCH,
};

/// Which planes of the secondary clips a caller reads (VapourSynth formats
/// have no alpha, so vszip's compareNodes has no such rule).
pub const AlphaRule = enum {
    /// Only color planes are read: alpha presence may differ.
    color_only,
    /// The alpha plane is read too: every clip must have alpha iff `vis[0]` has.
    same,
};

/// vszip's compareNodes: every clip must match `vis[0]` in size, family,
/// subsampling and depth (and length per `len`), plus alpha presence when
/// `alpha == .same`.
pub fn compareClips(ctx: *avs.Create, vis: []const ?*const c.AVS_VideoInfo, len: ClipLen, alpha: AlphaRule) error{Failed}!void {
    const vi0 = vis[0].?;
    const f0 = try planarFormat(ctx, vi0);
    for (vis[1..]) |opt| {
        const vi = opt orelse continue;
        const f = try planarFormat(ctx, vi);
        if (vi0.width != vi.width or vi0.height != vi.height)
            return ctx.fail("all input clips must have the same width and height.", .{});
        if (f0.family != f.family)
            return ctx.fail("all input clips must have the same color family.", .{});
        if (f0.ssw != f.ssw or f0.ssh != f.ssh)
            return ctx.fail("all input clips must have the same subsampling.", .{});
        if (f0.bits != f.bits)
            return ctx.fail("all input clips must have the same bit depth.", .{});
        if (alpha == .same and f0.has_alpha != f.has_alpha)
            return ctx.fail("all input clips must have an alpha plane, or none.", .{});
        switch (len) {
            .SAME_LEN => if (vi0.num_frames != vi.num_frames)
                return ctx.fail("all input clips must have the same length.", .{}),
            .BIGGER_THAN => if (vi0.num_frames > vi.num_frames)
                return ctx.fail("second clip has less frames than input clip.", .{}),
            .MISMATCH => {},
        }
    }
}

pub const Range = enum { FULL, LIMITED };

/// `_ColorRange` of frame 0 (0 = full, 1 = limited: same convention in
/// AviSynth+ and VapourSynth), else full for RGB and limited otherwise.
pub fn getRange(env: avs.Env, clip: avs.Clip, f: Format) Range {
    if (clip.getFrame(0)) |frame| {
        defer frame.release();
        if (frame.propsRO(env).getInt("_ColorRange", 0)) |r| {
            return if (r == 0) .FULL else .LIMITED;
        } else |_| {}
    } else |_| {}
    return if (f.family == .rgb) .FULL else .LIMITED;
}

/// `_ColorRange` of frame 0 when present (no format-based fallback).
pub fn getRangeProp(env: avs.Env, clip: avs.Clip) ?Range {
    const frame = clip.getFrame(0) catch return null;
    defer frame.release();
    const r = frame.propsRO(env).getInt("_ColorRange", 0) catch return null;
    return if (r == 0) .FULL else .LIMITED;
}

fn depthOf(f: Format) common.Depth {
    return .{ .bits = f.bits, .float = f.isFloat() };
}

pub fn getLowestValue(f: Format, chroma: bool, range: Range) f32 {
    return common.getLowestValue(depthOf(f), chroma, range == .LIMITED);
}

pub fn getPeakValue(f: Format, chroma: bool, range: Range) f32 {
    return common.getPeakValue(depthOf(f), chroma, range == .LIMITED);
}

pub const ScaleValue = struct {
    depth_in: u8 = 8,
    float_in: bool = false,
    chroma: bool = false,
};

/// vszip's scaleValue: rescale `value` given at `opt.depth_in` to the target
/// format. `range_in` is the target's range (float targets always scale as full).
pub fn scaleValue(value: f32, target: Format, range_in: Range, opt: ScaleValue) f32 {
    if (opt.depth_in == target.bits) return value;
    const range: Range = if (target.isFloat()) .FULL else range_in;
    const in: common.Depth = .{ .bits = opt.depth_in, .float = opt.float_in };
    return common.rescale(value, in, depthOf(target), opt.chroma, range == .LIMITED);
}

/// Copy every logical plane not selected in `process` from `src` to `dst`
/// (AviSynth has no VapourSynth-style plane sharing between frames).
pub fn copyUnprocessed(env: avs.Env, dst: avs.Frame, src: avs.Frame, f: Format, process: []const bool) void {
    for (0..f.numPlanes()) |p| {
        if (p < process.len and process[p]) continue;
        avs.copyPlane(env, dst, src, f.planeId(p));
    }
}

/// Copy the planes beyond the color planes (alpha) unchanged.
pub fn copyAlpha(env: avs.Env, dst: avs.Frame, src: avs.Frame, f: Format) void {
    if (f.has_alpha) avs.copyPlane(env, dst, src, c.AVS_PLANAR_A);
}

/// Run an element-wise `kernel(dst, src, ctx)` over a plane. Matching pitches
/// process the whole span in one call (inter-row padding of the owned output
/// is written, which is harmless); otherwise row by row.
pub fn elementwise(
    comptime T: type,
    dst: avs.PlaneMut(T),
    src: avs.Plane(T),
    kctx: anytype,
    comptime kernel: fn ([]T, []const T, @TypeOf(kctx)) void,
) void {
    std.debug.assert(dst.width == src.width and dst.height == src.height);
    if (dst.pitch == src.pitch) {
        kernel(dst.span(), src.span(), kctx);
    } else {
        for (0..src.height) |y| kernel(dst.row(y), src.row(y), kctx);
    }
}

/// Run `kernel(dst_row, src_row, ctx)` strictly per row (never over padding).
/// Required for kernels that index tables with sample values: AviSynth leaves
/// inter-row padding uninitialized, so garbage there could index out of range.
pub fn rowwise(
    comptime T: type,
    comptime U: type,
    dst: avs.PlaneMut(T),
    src: avs.Plane(U),
    kctx: anytype,
    comptime kernel: fn ([]T, []const U, @TypeOf(kctx)) void,
) void {
    std.debug.assert(dst.width == src.width and dst.height == src.height);
    for (0..src.height) |y| kernel(dst.row(y), src.row(y), kctx);
}

/// Output video info with a different planar format (same size, timing).
pub fn withFormat(vi: c.AVS_VideoInfo, f: Format) c.AVS_VideoInfo {
    var out = vi;
    out.pixel_type = avs.format.pixelType(f) catch unreachable;
    return out;
}

/// Gray format with the same sample type and depth as `f`.
pub fn grayOf(f: Format) Format {
    var g = f;
    g.family = .gray;
    g.color_planes = 1;
    g.has_alpha = false;
    g.ssw = 0;
    g.ssh = 0;
    return g;
}

/// Set `_ColorRange` (0 = full, 1 = limited) on a writable frame.
pub fn setRange(env: avs.Env, frame: avs.Frame, range: Range) void {
    frame.propsRW(env).setInt("_ColorRange", if (range == .FULL) 0 else 1, .replace) catch {};
}

/// Index of `planes` in a `comptime_planes`-style table (vszip's plane-mask
/// monomorphization); `fallback` when absent.
pub fn planeIndex(comptime table: anytype, planes: [3]bool, fallback: u32) u32 {
    for (table, 0..) |t, i| {
        if (std.mem.eql(bool, &t, &planes)) return @intCast(i);
    }
    return fallback;
}

/// Per-call scratch for canonical source copies and temp planes; `deinit`
/// (or `reset`) frees them all. Grows as needed: temporal filters can hold
/// many copies (WNNM: 2 per frame of a 31-frame window per plane).
pub const Scratch = struct {
    bufs: std.ArrayList([]align(common.vec_len) u8) = .empty,
    /// Canonical copies made so far, so a plane requested again (WNNM's
    /// reference when it is the source, clamped edge frames of a temporal
    /// window) shares one copy. Keyed by the source plane's address and
    /// geometry: valid while the source frames stay alive, so `reset` before
    /// releasing a source frame that a later `canonical` call could alias.
    memo: std.ArrayList(Memo) = .empty,

    const Memo = struct { src: usize, pitch: usize, rows: usize, row_bytes: usize, want: usize, out: [*]align(common.vec_len) u8 };

    pub fn deinit(self: *Scratch) void {
        for (self.bufs.items) |b| std.heap.c_allocator.free(b);
        self.bufs.deinit(std.heap.c_allocator);
        self.memo.deinit(std.heap.c_allocator);
        self.* = .{};
    }

    /// Free every copy but keep the bookkeeping capacity (per-plane reuse).
    pub fn reset(self: *Scratch) void {
        for (self.bufs.items) |b| std.heap.c_allocator.free(b);
        self.bufs.clearRetainingCapacity();
        self.memo.clearRetainingCapacity();
    }

    /// Zeroed full plane of `stride * height` elements (temp planes for
    /// multi-pass kernels), freed by `deinit`/`reset`.
    pub fn plane(self: *Scratch, comptime T: type, stride: usize, height: usize) error{OutOfMemory}![]T {
        const buf = try self.alloc(stride * height * @sizeOf(T));
        @memset(buf, 0);
        const p: [*]T = @ptrCast(@alignCast(buf.ptr));
        return p[0 .. stride * height];
    }

    fn alloc(self: *Scratch, n: usize) error{OutOfMemory}![]align(common.vec_len) u8 {
        try self.bufs.ensureUnusedCapacity(std.heap.c_allocator, 1);
        const b = try std.heap.c_allocator.alignedAlloc(u8, common.alignment, n);
        self.bufs.appendAssumeCapacity(b);
        return b;
    }
};

/// Source plane in vszip's canonical layout: pitch `want_pitch` (normally the
/// output frame's) and a `vec_len`-aligned start, so `fullSlice()` is safe and
/// single-stride kernels can address source and destination alike. Freshly
/// allocated upstream frames already qualify (no copy); horizontally cropped
/// or foreign-pitch planes are repacked into `scratch` (one copy per distinct
/// source plane, see `Scratch.memo`).
pub fn canonical(comptime T: type, src: avs.Plane(T), want_pitch: usize, scratch: *Scratch) error{OutOfMemory}!avs.Plane(T) {
    if (src.pitch == want_pitch and src.isAligned(common.vec_len)) return src;
    std.debug.assert(want_pitch >= src.rowBytes());
    const key: Scratch.Memo = .{
        .src = @intFromPtr(src.ptr),
        .pitch = src.pitch,
        .rows = src.height,
        .row_bytes = src.rowBytes(),
        .want = want_pitch,
        .out = undefined,
    };
    for (scratch.memo.items) |m| {
        if (m.src == key.src and m.pitch == key.pitch and m.rows == key.rows and m.row_bytes == key.row_bytes and m.want == key.want)
            return .{ .ptr = @ptrCast(@alignCast(m.out)), .pitch = want_pitch, .width = src.width, .height = src.height };
    }
    try scratch.memo.ensureUnusedCapacity(std.heap.c_allocator, 1);
    const buf = try scratch.alloc(want_pitch * @max(src.height, 1));
    for (0..src.height) |y| {
        const d = buf[y * want_pitch ..][0..want_pitch];
        @memcpy(d[0..src.rowBytes()], std.mem.sliceAsBytes(src.row(y)));
        @memset(d[src.rowBytes()..], 0);
    }
    var m = key;
    m.out = buf.ptr;
    scratch.memo.appendAssumeCapacity(m);
    return .{ .ptr = @ptrCast(@alignCast(buf.ptr)), .pitch = want_pitch, .width = src.width, .height = src.height };
}

/// Adapter from vszip kernels' `props.setInt/setFloat(key, v, .Append)` calls
/// to an AviSynth writable property map. A failed set is remembered and
/// reported by the adapter after the kernel returns.
pub const PropSink = struct {
    rw: avs.PropsRW,
    failed: bool = false,

    pub fn setInt(self: *PropSink, k: [:0]const u8, v: i64, mode: anytype) void {
        self.rw.setInt(k, v, appendMode(mode)) catch {
            self.failed = true;
        };
    }

    pub fn setFloat(self: *PropSink, k: [:0]const u8, v: f64, mode: anytype) void {
        self.rw.setFloat(k, v, appendMode(mode)) catch {
            self.failed = true;
        };
    }

    fn appendMode(comptime mode: anytype) avs.props.AppendMode {
        return switch (mode) {
            .Append => .append,
            .Replace => .replace,
            else => @compileError("unsupported append mode"),
        };
    }
};

/// Owned copy of a string argument with a suffix (property names).
pub fn dupeSuffix(prefix: []const u8, comptime suffix: []const u8) error{OutOfMemory}![:0]u8 {
    return std.fmt.allocPrintSentinel(std.heap.c_allocator, "{s}" ++ suffix, .{prefix}, 0);
}
