const std = @import("std");

const filter = @import("../filters/butteraugli.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Butteraugli";
pub const params = "reference:vnode;distorted:vnode;intensity_target:float:opt;hf_asymmetry:float:opt;xmul:float:opt;distmap:int:opt;";
const Scratch = []align(vszip.vec_len) f32;

const Source = struct {
    srgb: bool = true,
    lut: []f32 = &.{},
};

const Data = struct {
    node1: ?*vs.Node = null,
    node2: ?*vs.Node = null,
    vi_out: vs.VideoInfo = undefined,
    kind: filter.Input.Kind = .f32,
    bits: u6 = 32,
    src1: Source = .{},
    src2: Source = .{},
    params: filter.Params = .{},
    distmap: u2 = 0,
    good: f64 = 0,
    bad: f64 = 0,
    w: u32 = 0,
    h: u32 = 0,
    // Reusable scratch blocks; getFrame runs concurrently, so each call pops
    // its own block (or allocates one) and pushes it back when done.
    pool: std.ArrayList(Scratch) = .empty,
    pool_mutex: std.Io.Mutex = .init,
};

fn acquireScratch(d: *Data) Scratch {
    {
        d.pool_mutex.lockUncancelable(vszip.io);
        defer d.pool_mutex.unlock(vszip.io);
        if (d.pool.pop()) |buf| return buf;
    }
    const buf = allocator.alignedAlloc(f32, vszip.alignment, filter.scratchSize(d.w, d.h)) catch unreachable;
    // the kernel's padding columns and Malta border are never written
    @memset(buf, 0);
    return buf;
}

fn releaseScratch(d: *Data, buf: Scratch) void {
    d.pool_mutex.lockUncancelable(vszip.io);
    defer d.pool_mutex.unlock(vszip.io);
    d.pool.append(allocator, buf) catch unreachable;
}

fn input(d: *const Data, s: *const Source, f: anytype) filter.Input {
    return .{
        .kind = d.kind,
        .planes = .{ f.getReadSlice(0).ptr, f.getReadSlice(1).ptr, f.getReadSlice(2).ptr },
        .stride = f.getStride(0),
        .lut = s.lut,
        .srgb = s.srgb,
    };
}

fn butteraugliGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node1);
        zapi.requestFrameFilter(n, d.node2);
    } else if (activation_reason == .AllFramesReady) {
        const src1 = zapi.initZFrame(d.node1, n);
        const src2 = zapi.initZFrame(d.node2, n);
        defer src1.deinit();
        defer src2.deinit();

        const scratch = acquireScratch(d);
        defer releaseScratch(d, scratch);
        var ctx = filter.Context.init(scratch, d.w, d.h);
        const res = filter.run(&ctx, input(d, &d.src1, &src1), input(d, &d.src2, &src2), d.params);

        const dst = switch (d.distmap) {
            0 => src1.copyFrame(),
            else => src1.newVideoFrame3(.{ .format = &d.vi_out.format }),
        };
        const props = dst.getPropertiesRW();
        switch (d.distmap) {
            1 => {
                writeHeatmap(d, &dst, &ctx);
                // butteraugli_main saves these colors as an sRGB image; the
                // reference's color tags (e.g. a linear _Transfer) don't apply
                props.setInt("_Transfer", 13, .Replace); // IEC 61966-2-1 (sRGB)
                props.setInt("_Primaries", 1, .Replace); // BT.709
                props.setInt("_Matrix", 0, .Replace); // RGB
                props.setRange(.FULL);
            },
            2 => {
                writeRaw(d, &dst, &ctx);
                // plain numbers, not light: drop the inherited color tags
                // (_ColorRange is left alone: deleting the alias deletes _Range)
                props.deleteKey("_Transfer");
                props.deleteKey("_Primaries");
                props.deleteKey("_Matrix");
            },
            else => {},
        }
        props.setFloat("Butteraugli", res.max, .Replace);
        props.setFloat("Butteraugli3Norm", res.norm3, .Replace);
        return dst.frame;
    }
    return null;
}

/// butteraugli_main's --distmap colors, in the input's RGB format.
fn writeHeatmap(d: *const Data, dst: anytype, ctx: *const filter.Context) void {
    const map = ctx.diffmap();
    const ms = ctx.full.stride;
    switch (d.kind) {
        inline else => |k| {
            const T = switch (k) {
                .u8 => u8,
                .u16 => u16,
                .f32 => f32,
            };
            var planes: [3][]T = undefined;
            for (0..3) |c| planes[c] = dst.getWriteSlice2(T, @intCast(c));
            const stride = dst.getStride2(T, 0);
            const peak: f32 = if (T == f32) 1.0 else @floatFromInt((@as(u32, 1) << @intCast(d.bits)) - 1);
            for (0..d.h) |y| {
                for (0..d.w) |x| {
                    const rgb = filter.heatmapRgb(map[y * ms + x], d.good, d.bad);
                    for (0..3) |c| {
                        planes[c][y * stride + x] = if (T == f32) rgb[c] else @intFromFloat(@round(rgb[c] * peak));
                    }
                }
            }
        },
    }
}

fn writeRaw(d: *const Data, dst: anytype, ctx: *const filter.Context) void {
    const map = ctx.diffmap();
    const ms = ctx.full.stride;
    const out = dst.getWriteSlice2(f32, 0);
    const stride = dst.getStride2(f32, 0);
    for (0..d.h) |y| @memcpy(out[y * stride ..][0..d.w], map[y * ms ..][0..d.w]);
}

fn butteraugliFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    for (d.pool.items) |buf| allocator.free(buf);
    d.pool.deinit(allocator);
    allocator.free(d.src1.lut);
    allocator.free(d.src2.lut);
    zapi.freeNode(d.node1);
    zapi.freeNode(d.node2);
    allocator.destroy(d);
}

/// Reads frame 0's _Transfer (linear input skips the sRGB decode). Null with
/// the upstream message in `err` if that frame fails.
fn frame0Srgb(node: ?*vs.Node, zapi: *const ZAPI, err: []u8) ?bool {
    const frame = zapi.getFrame(0, node, err) orelse return null;
    defer zapi.freeFrame(frame);
    return zapi.initZMap(zapi.getFramePropertiesRO(frame)).getTransfer() != .LINEAR;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    var print_buf: [256]u8 = undefined;
    const map_out = zapi.initZMap2(out, &print_buf);

    d.node1, const vi1 = map_in.getNodeVi("reference").?;
    d.node2, const vi2 = map_in.getNodeVi("distorted").?;
    const nodes = [_]?*vs.Node{ d.node1, d.node2 };
    const fail = struct {
        fn f(mo: anytype, z: *const ZAPI, ns: []const ?*vs.Node, comptime msg: [:0]const u8) void {
            mo.setError(filter_name ++ ": " ++ msg);
            for (ns) |nd| z.freeNode(nd);
        }
    }.f;

    if (!vsh.isConstantVideoFormat(vi1) or !vsh.isSameVideoInfo(vi1, vi2)) {
        return fail(&map_out, &zapi, &nodes, "both clips must have the same constant format and dimensions.");
    }
    if (vi1.numFrames != vi2.numFrames) return fail(&map_out, &zapi, &nodes, "clips must have the same length.");
    const fmt = vi1.format;
    if (fmt.colorFamily != .RGB) return fail(&map_out, &zapi, &nodes, "only RGB input is supported.");
    if (fmt.sampleType == .Float and fmt.bitsPerSample != 32) return fail(&map_out, &zapi, &nodes, "half-float (f16) input is not supported.");
    // 17-32 bit integers are 4-byte samples the u16 path can't read
    if (fmt.sampleType == .Integer and fmt.bitsPerSample > 16) return fail(&map_out, &zapi, &nodes, "integer input must be 8-16 bit.");
    if (vi1.width < 64 or vi1.height < 64) return fail(&map_out, &zapi, &nodes, "clips must be at least 64x64.");

    // validated as f64 against the decimal bounds, then narrowed: an f32
    // bound (0.001f > 0.001) would reject the documented endpoint
    d.params.intensity_target = @floatCast(hz.getFloat(f64, 80.0, 0.001, 10000.0, "intensity_target", filter_name, map_in, map_out, &nodes, &zapi) catch return);
    d.params.hf_asymmetry = @floatCast(hz.getFloat(f64, 1.0, 0.001, 1000.0, "hf_asymmetry", filter_name, map_in, map_out, &nodes, &zapi) catch return);
    d.params.xmul = @floatCast(hz.getFloat(f64, 1.0, 0.0, 1000.0, "xmul", filter_name, map_in, map_out, &nodes, &zapi) catch return);
    const dm = map_in.getValue(i64, "distmap") orelse 0;
    if (dm < 0 or dm > 2) return fail(&map_out, &zapi, &nodes, "distmap must be 0 (off), 1 (heatmap) or 2 (raw).");
    d.distmap = @intCast(dm);

    var err_buf: [512]u8 = @splat(0);
    const s1 = frame0Srgb(d.node1, &zapi, &err_buf);
    const s2 = if (s1 != null) frame0Srgb(d.node2, &zapi, &err_buf) else null;
    if (s1 == null or s2 == null) {
        var buf: [640]u8 = undefined;
        const msg = std.mem.printSentinel(&buf, filter_name ++ ": frame 0 failed: {s}", .{std.mem.sliceTo(&err_buf, 0)}, 0) catch filter_name ++ ": frame 0 failed.";
        map_out.setError(msg);
        for (nodes) |nd| zapi.freeNode(nd);
        return;
    }
    d.src1.srgb = s1.?;
    d.src2.srgb = s2.?;

    d.w = @intCast(vi1.width);
    d.h = @intCast(vi1.height);
    d.bits = @intCast(fmt.bitsPerSample);
    d.kind = if (fmt.sampleType == .Float) .f32 else if (fmt.bytesPerSample == 1) .u8 else .u16;
    if (d.kind != .f32) {
        const lut_len: usize = if (d.kind == .u8) 256 else 65536;
        for ([_]*Source{ &d.src1, &d.src2 }) |s| {
            s.lut = allocator.alloc(f32, lut_len) catch unreachable;
            filter.buildLut(s.lut, @intCast(d.bits), s.srgb);
        }
    }
    d.good = filter.fuzzyInverse(1.5);
    d.bad = filter.fuzzyInverse(0.5);

    d.vi_out = vi1.*;
    if (d.distmap == 2) _ = zapi.queryVideoFormat(&d.vi_out.format, .Gray, .Float, 32, 0, 0);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    var deps = [_]vs.FilterDependency{
        .{ .source = d.node1, .requestPattern = .StrictSpatial },
        .{ .source = d.node2, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, filter_name, &data.vi_out, butteraugliGetFrame, butteraugliFree, .Parallel, &deps, data);
}
