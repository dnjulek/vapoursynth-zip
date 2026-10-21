const std = @import("std");
const math = std.math;

const filter = @import("../filters/nlmeans.zig");
const hz = @import("../helper.zig");
const vszip = @import("../vszip.zig");

const vapoursynth = vszip.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "NLMeans";
pub const params = "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;" ++
    "wmode:int:opt;wref:float:opt;rclip:vnode:opt;";

const Ref = filter.Ref;
const max_planes = (2 * filter.max_d + 1) * 3;

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    cfg: filter.Config = undefined,
    plane0: u32 = 0,
    peak: f32 = 1.0,

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *filter.Scratch) = undefined,
};

fn acquireScratch(d: *Data) ?*filter.Scratch {
    const tid = std.Thread.getCurrentId();
    d.pool_lock.lockUncancelable(vszip.io);
    defer d.pool_lock.unlock(vszip.io);
    if (d.pool.get(tid)) |sc| return sc;
    const sc = filter.Scratch.create(&d.cfg, d.ref_node != null) catch return null;
    d.pool.put(tid, sc) catch {
        sc.destroy();
        return null;
    };
    return sc;
}

fn NLMeans(comptime T: type, comptime ref: Ref, comptime wmode: u32) type {
    return struct {
        const C = ref.channels();
        const In = filter.PlaneIn(T);
        const Out = filter.PlaneOut(T);

        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core, frame_ctx);
            // KNLMeansCL's window: radius min(d, n) on both sides, the far
            // side clamped to the last frame.
            const m: i32 = @min(@as(i32, @intCast(d.cfg.d)), n);
            const last = d.vi.numFrames - 1;

            if (activation_reason == .Initial) {
                var k: i32 = -m;
                while (k <= m) : (k += 1) {
                    const fid = math.clamp(n + k, 0, last);
                    zapi.requestFrameFilter(fid, d.node);
                    if (d.ref_node != null) zapi.requestFrameFilter(fid, d.ref_node);
                }
                return null;
            }
            if (activation_reason != .AllFramesReady) return null;

            const ZF = @TypeOf(zapi.initZFrame(d.node, n));
            var srcs: [2 * filter.max_d + 1]ZF = undefined;
            var refs: [2 * filter.max_d + 1]ZF = undefined;
            const count: usize = @intCast(2 * m + 1);
            for (0..count) |i| {
                const fid = math.clamp(n - m + @as(i32, @intCast(i)), 0, last);
                srcs[i] = zapi.initZFrame(d.node, fid);
                if (d.ref_node != null) refs[i] = zapi.initZFrame(d.ref_node, fid);
            }
            defer for (0..count) |i| {
                srcs[i].deinit();
                if (d.ref_node != null) refs[i].deinit();
            };

            const center = srcs[@intCast(m)];
            var process: [3]bool = @splat(false);
            for (0..C) |c| process[d.plane0 + c] = true;
            const dst = center.newVideoFrame2(process);

            const sc = acquireScratch(d) orelse return hz.frameOom(&zapi, filter_name, dst);

            // window layer t = d + k holds frame n + k
            var src_planes: [max_planes]In = undefined;
            var ref_planes: [max_planes]In = undefined;
            const n_planes = (2 * d.cfg.d + 1) * C;
            @memset(src_planes[0..n_planes], .{ .data = &.{}, .stride = 0 });
            @memset(ref_planes[0..n_planes], .{ .data = &.{}, .stride = 0 });
            const t0: usize = d.cfg.d - @as(u32, @intCast(m));
            for (0..count) |i| {
                for (0..C) |c| {
                    const p = d.plane0 + @as(u32, @intCast(c));
                    src_planes[(t0 + i) * C + c] = .{ .data = srcs[i].getReadSlice2(T, p), .stride = srcs[i].getStride2(T, p) };
                    if (d.ref_node != null) {
                        ref_planes[(t0 + i) * C + c] = .{ .data = refs[i].getReadSlice2(T, p), .stride = refs[i].getStride2(T, p) };
                    }
                }
            }

            var dst_planes: [C]Out = undefined;
            for (0..C) |c| {
                const p = d.plane0 + @as(u32, @intCast(c));
                dst_planes[c] = .{ .data = dst.getWriteSlice2(T, p), .stride = dst.getStride2(T, p) };
            }

            filter.processFrame(
                T,
                ref,
                wmode,
                &d.cfg,
                sc,
                src_planes[0..n_planes],
                if (d.ref_node != null) ref_planes[0..n_planes] else null,
                dst_planes,
                @intCast(m),
                d.peak,
            );
            return dst.frame;
        }
    };
}

fn selectGetFrame(dt: hz.DataType, ref: Ref, wmode: u32) vs.FilterGetFrame {
    return switch (dt) {
        .U32 => unreachable,
        inline else => |tag| switch (ref) {
            inline else => |r| switch (wmode) {
                inline 0...3 => |w| &NLMeans(tag.ToType(), r, w).getFrame,
                else => unreachable,
            },
        },
    };
}

fn nlmFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);

    var it = d.pool.valueIterator();
    while (it.next()) |sc| sc.*.destroy();
    d.pool.deinit();

    zapi.freeNode(d.node);
    zapi.freeNode(d.ref_node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;
    const fmt = d.vi.format;

    if (!vapoursynth.vshelper.isConstantVideoFormat(d.vi)) {
        map_out.setError(filter_name ++ ": only constant format input supported.");
        zapi.freeNode(d.node);
        return;
    }
    const dt = hz.DataType.select(map_out, d.node, d.vi, filter_name, false) catch return;
    if (fmt.sampleType == .Integer) {
        d.peak = @floatFromInt((@as(u32, 1) << @intCast(fmt.bitsPerSample)) - 1);
    }

    const fmax: f64 = math.floatMax(f32);
    const h = hz.getFloat(f64, 1.2, 0, fmax, "h", filter_name, map_in, map_out, &.{d.node}, &zapi) catch return;
    const wref = hz.getFloat(f64, 1.0, 0, fmax, "wref", filter_name, map_in, map_out, &.{d.node}, &zapi) catch return;

    // channels
    const eq = std.ascii.eqlIgnoreCase;
    const chstr = map_in.getData("channels", 0) orelse "auto";
    var ref: Ref = .luma;
    const ch_err: ?[:0]const u8 = switch (fmt.colorFamily) {
        .Gray => if (eq(chstr, "Y") or eq(chstr, "auto")) null else filter_name ++ ": 'channels' must be 'Y' with Gray.",
        .YUV => blk: {
            if (eq(chstr, "YUV")) {
                if (fmt.subSamplingW != 0 or fmt.subSamplingH != 0) {
                    break :blk filter_name ++ ": 'channels'='YUV' requires 4:4:4 (the joint patch distance needs one pixel lattice; run 'Y' and 'UV' passes for subsampled formats).";
                }
                ref = .yuv;
            } else if (eq(chstr, "UV")) {
                ref = .chroma;
                d.plane0 = 1;
            } else if (!(eq(chstr, "Y") or eq(chstr, "auto"))) {
                break :blk filter_name ++ ": 'channels' must be 'YUV', 'Y' or 'UV' with YUV.";
            }
            break :blk null;
        },
        .RGB => blk: {
            if (!(eq(chstr, "RGB") or eq(chstr, "auto"))) break :blk filter_name ++ ": 'channels' must be 'RGB' with RGB.";
            ref = .rgb;
            break :blk null;
        },
        else => filter_name ++ ": unsupported color family.",
    };
    if (ch_err) |msg| {
        map_out.setError(msg);
        zapi.freeNode(d.node);
        return;
    }

    const ssw: u5 = @intCast(fmt.subSamplingW);
    const ssh: u5 = @intCast(fmt.subSamplingH);
    const pw: u32 = if (ref == .chroma) @as(u32, @intCast(d.vi.width)) >> ssw else @intCast(d.vi.width);
    const ph: u32 = if (ref == .chroma) @as(u32, @intCast(d.vi.height)) >> ssh else @intCast(d.vi.height);

    d.cfg = filter.Config.init(.{
        .d = map_in.getValue(i64, "d") orelse 1,
        .a = map_in.getValue(i64, "a") orelse 2,
        .s = map_in.getValue(i64, "s") orelse 4,
        .h = h,
        .wmode = map_in.getValue(i64, "wmode") orelse 0,
        .wref = wref,
        .ref = ref,
        .width = pw,
        .height = ph,
    }) catch |e| {
        hz.setArgError(map_out, "{s}: {s}", .{ filter_name, filter.Config.message(e) });
        zapi.freeNode(d.node);
        return;
    };

    d.ref_node = map_in.getNode("rclip");
    if (d.ref_node != null) {
        const ref_vi = zapi.getVideoInfo(d.ref_node);
        if (!vapoursynth.vshelper.isSameVideoInfo(d.vi, ref_vi) or ref_vi.numFrames != d.vi.numFrames) {
            map_out.setError(filter_name ++ ": 'rclip' must match the source clip's format, dimensions and frame count.");
            zapi.freeNode(d.node);
            zapi.freeNode(d.ref_node);
            return;
        }
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    data.pool_lock = .init;
    data.pool = std.AutoHashMap(std.Thread.Id, *filter.Scratch).init(allocator);

    const rp: vs.RequestPattern = if (d.cfg.d > 0) .General else .StrictSpatial;
    var deps_buf: [2]vs.FilterDependency = .{
        .{ .source = d.node, .requestPattern = rp },
        .{ .source = d.ref_node, .requestPattern = rp },
    };
    const deps = if (d.ref_node != null) deps_buf[0..2] else deps_buf[0..1];
    zapi.createVideoFilter(out, filter_name, d.vi, selectGetFrame(dt, ref, d.cfg.wmode), nlmFree, .Parallel, deps, data);
    if (map_out.getError() != null) nlmFree(data, core, vsapi);
}
