const std = @import("std");
const vapoursynth = @import("vapoursynth");
const hz = @import("../helper.zig");
const math = std.math;

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const core = @import("../filters/eedi3.zig");

const io = core.io;
const allocator = core.allocator;
const Scratch = core.Scratch;
const processPlane = core.processPlane;
const transposeAny = core.transposeAny;
const transposeBlocked = core.transposeBlocked;
const allocScratch = core.allocScratch;
const freeScratch = core.freeScratch;
const n_vec = core.n_vec;

const Data = struct {
    node: ?*vs.Node = null,
    sclip: ?*vs.Node = null,
    mclip: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,
    k: core.Data = .{},

    pool_lock: std.Io.Mutex = .init,
    pool: std.AutoHashMap(std.Thread.Id, *Scratch) = undefined,
};

fn Filter(comptime T: type) type {
    return struct {
        pub fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
            const d: *Data = @ptrCast(@alignCast(instance_data));
            const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);
            const src_n: c_int = if (d.k.field > 1) @divTrunc(n, 2) else n;

            if (activation_reason == .Initial) {
                zapi.requestFrameFilter(src_n, d.node);
                if (d.k.vcheck > 0 and d.sclip != null) zapi.requestFrameFilter(n, d.sclip);
                if (d.mclip != null) zapi.requestFrameFilter(src_n, d.mclip);
            } else if (activation_reason == .AllFramesReady) {
                const src = zapi.initZFrame(d.node, src_n);
                defer src.deinit();

                const scp = if (d.k.vcheck > 0 and d.sclip != null) zapi.initZFrame(d.sclip, n) else null;
                defer if (scp) |s| s.deinit();

                const mcp = if (d.mclip != null) zapi.initZFrame(d.mclip, src_n) else null;
                defer if (mcp) |m| m.deinit();

                const dst = if (d.k.horizontal)
                    src.newVideoFrame3(.{ .width = d.vi.width })
                else
                    src.newVideoFrame3(.{ .height = d.vi.height });
                const dst_props = dst.getPropertiesRW();

                var field: u8 = d.k.field & 1;
                switch (dst_props.getFieldBased() orelse .PROGRESSIVE) {
                    .BOTTOM => field = 0,
                    .TOP => field = 1,
                    else => {},
                }
                if (d.k.field > 1) field = @as(u8, @intCast(n & 1)) ^ field;

                // Size the per-thread scratch off plane 0. The internal pipeline is
                // vertical, so the horizontal path sizes its line buffers off the source
                // HEIGHT (the transposed line length) and requests the srcT/dstT frames.
                const p0_w: u32, const p0_h: u32, const p0_stride: u32 = src.getDimensions2(T, 0);
                var alloc_w: u32 = undefined;
                var alloc_stride: u32 = undefined;
                var alloc_n_interp: u32 = undefined;
                var srcT_rows: u32 = 0;
                var dstT_rows: u32 = 0;
                if (d.k.horizontal) {
                    alloc_w = p0_h;
                    alloc_stride = std.mem.alignForward(u32, p0_h, n_vec);
                    alloc_n_interp = if (d.k.dh) p0_w else p0_w / 2;
                    srcT_rows = p0_w;
                    dstT_rows = if (d.k.dh) p0_w * 2 else p0_w;
                } else {
                    alloc_w = p0_w;
                    alloc_stride = p0_stride;
                    alloc_n_interp = if (d.k.dh) p0_h else p0_h / 2;
                }

                const tid = std.Thread.getCurrentId();
                const scratch: *Scratch = blk: {
                    d.pool_lock.lockUncancelable(io);
                    defer d.pool_lock.unlock(io);
                    if (d.pool.get(tid)) |s| break :blk s;
                    const s = allocScratch(alloc_w, alloc_stride, alloc_n_interp, d.k.hp, srcT_rows, dstT_rows, @sizeOf(T)) catch {
                        zapi.setFilterError(if (d.k.horizontal) "EEDI3H: failed to allocate memory." else "EEDI3: failed to allocate memory.");
                        dst.deinit();
                        return null;
                    };
                    d.pool.put(tid, s) catch unreachable;
                    break :blk s;
                };

                var plane: u32 = 0;
                while (plane < d.vi.format.numPlanes) : (plane += 1) {
                    const srcp: []const T = src.getReadSlice2(T, plane);
                    const dstp: []T = dst.getWriteSlice2(T, plane);
                    const scpp: ?[]const T = if (scp) |s| s.getReadSlice2(T, plane) else null;
                    // mclip is always a single Gray plane (see createImpl); the same mask
                    // drives every processed plane.
                    const maskp: ?[]const u8 = if (mcp) |m| m.getReadSlice2(u8, 0) else null;
                    const mask_stride: u32 = if (mcp) |m| m.getStride2(u8, 0) else 0;

                    if (d.k.horizontal) {
                        const src_w: u32, const src_h: u32, const src_stride: u32 = src.getDimensions2(T, plane);
                        const dst_w: u32 = dst.getWidth(plane);
                        const dst_stride: u32 = dst.getStride2(T, plane);
                        const Lstride: u32 = std.mem.alignForward(u32, src_h, n_vec);

                        // The only strided work: bring src (and mclip/sclip) into
                        // column-major layout, run the vertical pipeline on contiguous
                        // columns, then transpose the result back into the dst frame.
                        const srcT: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.srcT));
                        const dstT: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.dstT));
                        transposeAny(T, srcT, Lstride, srcp, src_stride, src_w, src_h);

                        const maskT: ?[]const u8 = if (maskp) |mp| blk: {
                            transposeBlocked(u8, scratch.maskT, Lstride, mp, mask_stride, src_w, src_h);
                            break :blk scratch.maskT;
                        } else null;

                        const scpT: ?[]const T = if (d.k.vcheck > 0) (if (scpp) |sp| blk: {
                            const scpT_buf: []T = @alignCast(std.mem.bytesAsSlice(T, scratch.scpT));
                            const scp_stride: u32 = scp.?.getStride2(T, plane);
                            transposeAny(T, scpT_buf, Lstride, sp, scp_stride, dst_w, src_h);
                            break :blk scpT_buf;
                        } else null) else null;

                        processPlane(T, &d.k, scratch, srcT, dstT, scpT, maskT, Lstride, field, src_h, Lstride, src_w, dst_w);

                        transposeAny(T, dstp, dst_stride, dstT, Lstride, src_h, dst_w);
                    } else {
                        const w: u32, const src_h: u32, const stride: u32 = src.getDimensions2(T, plane);
                        const dst_h: u32 = dst.getHeight(plane);
                        processPlane(T, &d.k, scratch, srcp, dstp, scpp, maskp, mask_stride, field, w, stride, src_h, dst_h);
                    }
                }

                dst_props.setFieldBased(.PROGRESSIVE);

                if (d.k.field > 1) {
                    var duration_num = dst_props.getDurationNum();
                    var duration_den = dst_props.getDurationDen();
                    if (duration_num != null and duration_den != null) {
                        vsh.muldivRational(&duration_num.?, &duration_den.?, 1, 2);
                        dst_props.setDurationNum(duration_num.?);
                        dst_props.setDurationDen(duration_den.?);
                    }
                }

                return dst.frame;
            }

            return null;
        }
    };
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    var it = d.pool.valueIterator();
    while (it.next()) |s| freeScratch(s.*);
    d.pool.deinit();
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.sclip);
    vsapi.?.freeNode.?(d.mclip);
    allocator.destroy(d);
}

pub fn createEEDI3(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(false, in, out, ud, core_ptr, vsapi);
}

pub fn createEEDI3H(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(true, in, out, ud, core_ptr, vsapi);
}

fn createImpl(comptime horizontal: bool, in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) void {
    const filter_name = if (horizontal) "EEDI3H" else "EEDI3";

    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core_ptr, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_in = map_in.getNodeVi("clip").?;
    d.vi = vi_in.*;

    const vcheck = map_in.getValue(i32, "vcheck") orelse 2;
    d.sclip = if (vcheck > 0) map_in.getNode("sclip") else null;
    d.mclip = map_in.getNode("mclip");

    var keep_node = false;
    defer if (!keep_node) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.sclip);
        zapi.freeNode(d.mclip);
    };

    if ((d.vi.format.sampleType == .Integer and d.vi.format.bitsPerSample > 16) or
        (d.vi.format.sampleType == .Float and d.vi.format.bitsPerSample != 32))
    {
        map_out.setError(filter_name ++ ": only 8-16 bit integer and 32-bit float input is supported.");
        return;
    }

    const field = map_in.getValue(i32, "field") orelse 0;
    const mdis = map_in.getValue(i32, "mdis") orelse 20;
    const nrad = map_in.getValue(i32, "nrad") orelse 2;

    // Finite only here (nodes are freed by the defer above); the specific
    // ranges below keep their own messages.
    const fmax = math.floatMax(f32);
    const alpha = hz.getFloat(f32, 0.2, -fmax, fmax, "alpha", filter_name, map_in, map_out, &.{}, &zapi) catch return;
    const beta = hz.getFloat(f32, 0.25, -fmax, fmax, "beta", filter_name, map_in, map_out, &.{}, &zapi) catch return;
    const gamma = hz.getFloat(f32, 20.0, -fmax, fmax, "gamma", filter_name, map_in, map_out, &.{}, &zapi) catch return;

    const dh = map_in.getBool("dh") orelse false;
    const hp = map_in.getBool("hp") orelse false;
    const vthresh0 = hz.getFloat(f32, 32.0, -fmax, fmax, "vthresh0", filter_name, map_in, map_out, &.{}, &zapi) catch return;
    const vthresh1 = hz.getFloat(f32, 64.0, -fmax, fmax, "vthresh1", filter_name, map_in, map_out, &.{}, &zapi) catch return;
    const vthresh2 = hz.getFloat(f32, 4.0, -fmax, fmax, "vthresh2", filter_name, map_in, map_out, &.{}, &zapi) catch return;

    const params: core.Data.Args = .{
        .field = field,
        .dh = dh,
        .alpha = alpha,
        .beta = beta,
        .gamma = gamma,
        .nrad = nrad,
        .mdis = mdis,
        .hp = hp,
        .vcheck = vcheck,
        .vthresh0 = vthresh0,
        .vthresh1 = vthresh1,
        .vthresh2 = vthresh2,
    };
    // The interpolated axis is the height for EEDI3, the width for EEDI3H.
    if (core.Data.check(params, horizontal, if (horizontal) d.vi.width else d.vi.height)) |msg| {
        hz.setArgError(map_out, "{s}: {s}", .{ filter_name, msg });
        return;
    }

    if (d.mclip != null) {
        const mvi = zapi.getVideoInfo(d.mclip);
        // Unlike eedi3m, the mask must be Gray: a single plane drives every
        // processed plane (see getFrame), so a multi-plane mask is rejected.
        if (mvi.format.colorFamily != .Gray) {
            map_out.setError(filter_name ++ ": mclip must be Gray.");
            return;
        }

        if (mvi.width != d.vi.width or mvi.height != d.vi.height) {
            map_out.setError(filter_name ++ ": mclip's dimensions don't match.");
            return;
        }

        if (mvi.numFrames != d.vi.numFrames) {
            map_out.setError(filter_name ++ ": mclip's number of frames doesn't match.");
            return;
        }

        if (mvi.format.bitsPerSample != 8 or mvi.format.sampleType != .Integer) {
            const args = zapi.createZMap();
            defer args.free();
            _ = args.consumeNode("clip", d.mclip, .Replace);
            d.mclip = null; // ownership of the node ref transferred to `args`
            args.setRange(.FULL);
            var ret = args.invoke(zapi.getPluginByID2(.Std), "SetFrameProps");

            args.clear();
            _ = args.consumeNode("clip", ret.getNode("clip"), .Replace);
            ret.free();
            args.setVideoFormat("format", .Gray8, .Replace);
            ret = args.invoke(zapi.getPluginByID2(.Resize), "Point");
            defer ret.free();

            if (ret.getError()) |err| {
                map_out.setError(err);
                return;
            }

            d.mclip = ret.getNode("clip");
        }
    }

    if (field > 1) {
        if (d.vi.numFrames > std.math.maxInt(i32) / 2) {
            map_out.setError(filter_name ++ ": resulting clip is too long.");
            return;
        }

        d.vi.numFrames *= 2;
        vsh.muldivRational(&d.vi.fpsNum, &d.vi.fpsDen, 2, 1);
    }

    if (dh) {
        if (horizontal) d.vi.width *= 2 else d.vi.height *= 2;
    }

    if (vcheck > 0 and d.sclip != null) {
        if (!vsh.isSameVideoInfo(zapi.getVideoInfo(d.sclip), &d.vi)) {
            map_out.setError(filter_name ++ ": sclip's format and dimensions don't match.");
            return;
        }
        if (zapi.getVideoInfo(d.sclip).numFrames != d.vi.numFrames) {
            map_out.setError(filter_name ++ ": sclip's number of frames doesn't match.");
            return;
        }
    }

    const float = d.vi.format.sampleType == .Float;
    d.k = core.Data.init(params, horizontal, float, @intCast(d.vi.format.bitsPerSample));

    d.pool_lock = .init;
    d.pool = std.AutoHashMap(std.Thread.Id, *Scratch).init(allocator);

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep_node = true;

    const dep_sources = [_]?*vs.Node{ d.node, d.sclip, d.mclip };
    var dep_buf: [3]vs.FilterDependency = undefined;
    var ndeps: usize = 0;
    for (dep_sources) |source| {
        if (source != null) {
            dep_buf[ndeps] = .{ .source = source, .requestPattern = .StrictSpatial };
            ndeps += 1;
        }
    }
    const get_frame: vs.FilterGetFrame = if (d.vi.format.sampleType == .Float)
        &Filter(f32).getFrame
    else switch (d.vi.format.bytesPerSample) {
        1 => &Filter(u8).getFrame,
        else => &Filter(u16).getFrame,
    };
    zapi.createVideoFilter(out, filter_name, &d.vi, get_frame, free, .Parallel, dep_buf[0..ndeps], data);
}

pub const args_string = "clip:vnode;field:int;dh:int:opt;alpha:float:opt;beta:float:opt;gamma:float:opt;nrad:int:opt;mdis:int:opt;hp:int:opt;vcheck:int:opt;vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;mclip:vnode:opt;";
