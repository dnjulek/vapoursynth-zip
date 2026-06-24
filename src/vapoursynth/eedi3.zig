const std = @import("std");
const vapoursynth = @import("vapoursynth");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const core = @import("../filters/eedi3.zig");

const io = core.io;
const allocator = core.allocator;
const Scratch = core.Scratch;
const Data = core.Data;
const interpLine = core.interpLine;
const interpLineHP = core.interpLineHP;
const fillPaddedRow = core.fillPaddedRow;
const buildBmask = core.buildBmask;
const vcheckLine = core.vcheckLine;
const srcCol = core.srcCol;
const transposeF32 = core.transposeF32;
const transposeBlocked = core.transposeBlocked;
const allocScratch = core.allocScratch;
const freeScratch = core.freeScratch;
const n_vec = core.n_vec;

fn processPlane(
    d: *const Data,
    scratch: *Scratch,
    srcl: []const f32,
    dstl: []f32,
    scpl: ?[]const f32,
    maskl: ?[]const u8,
    mask_stride: u32,
    field: u8,
    L: u32,
    lstride: u32,
    n_src: u32,
    n_dst: u32,
) void {
    const n_src_i: i32 = @intCast(n_src);
    const n_interp: u32 = if (d.dh) n_src else n_src / 2;

    // Copy the kept field of lines straight across (contiguous in either layout).
    if (d.dh) {
        var k: u32 = 0;
        while (k < n_src) : (k += 1) {
            const dl = 2 * k + (1 - field);
            @memcpy(dstl[dl * lstride ..][0..L], srcl[k * lstride ..][0..L]);
        }
    } else {
        var k: u32 = 1 - field;
        while (k < n_src) : (k += 2) {
            @memcpy(dstl[k * lstride ..][0..L], srcl[k * lstride ..][0..L]);
        }
    }

    var p3p: []f32 = scratch.r3p;
    var p1p: []f32 = scratch.r1p;
    var p1n: []f32 = scratch.r1n;
    var p3n: []f32 = scratch.r3n;
    var interp_off: u32 = 0;
    var line: u32 = field;
    while (line < n_dst) : (line += 2) {
        const line_i: i32 = @intCast(line);

        if (interp_off == 0) {
            fillPaddedRow(p3p, srcl[srcCol(d.dh, line_i - 3, n_src_i) * lstride ..][0..L], L);
            fillPaddedRow(p1p, srcl[srcCol(d.dh, line_i - 1, n_src_i) * lstride ..][0..L], L);
            fillPaddedRow(p1n, srcl[srcCol(d.dh, line_i + 1, n_src_i) * lstride ..][0..L], L);
            fillPaddedRow(p3n, srcl[srcCol(d.dh, line_i + 3, n_src_i) * lstride ..][0..L], L);
        } else {
            std.mem.swap([]f32, &p3p, &p1p);
            std.mem.swap([]f32, &p1p, &p1n);
            std.mem.swap([]f32, &p1n, &p3n);
            fillPaddedRow(p3n, srcl[srcCol(d.dh, line_i + 3, n_src_i) * lstride ..][0..L], L);
        }

        const bmask_row: ?[]const bool = if (maskl) |mp| blk: {
            const mrow: u32 = if (d.dh) interp_off else line;
            buildBmask(scratch.bmask, mp[mrow * mask_stride ..][0..L], L, d.mdis);
            break :blk scratch.bmask[0..L];
        } else null;

        const out_line = dstl[line * lstride ..];
        if (d.hp) {
            interpLineHP(
                p3p,
                p1p,
                p1n,
                p3n,
                scratch.hp3p,
                scratch.hp1p,
                scratch.hp1n,
                scratch.hp3n,
                out_line,
                scratch.pbackt,
                scratch.fpath,
                scratch.t_costs,
                scratch.dmap[interp_off * lstride ..],
                lstride,
                L,
                d.mdis,
                d.nrad,
                d.alpha,
                d.beta,
                d.gamma,
                d.one_minus_ab,
                bmask_row,
            );
        } else {
            interpLine(
                p3p,
                p1p,
                p1n,
                p3n,
                out_line,
                scratch.pbackt,
                scratch.fpath,
                scratch.t_base,
                scratch.t_costs,
                scratch.dmap[interp_off * lstride ..],
                lstride,
                L,
                d.mdis,
                d.nrad,
                d.alpha,
                d.beta,
                d.gamma,
                d.one_minus_ab,
                bmask_row,
                scratch.block_active,
            );
        }
        interp_off += 1;
    }

    if (d.vcheck > 0) {
        vcheckLine(srcl, dstl, scpl, scratch.dmap, scratch.tline, field, L, n_dst, n_src, lstride, n_interp, d);
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);
    const src_n: c_int = if (d.field > 1) @divTrunc(n, 2) else n;

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(src_n, d.node);
        if (d.vcheck > 0 and d.sclip != null) zapi.requestFrameFilter(n, d.sclip);
        if (d.mclip != null) zapi.requestFrameFilter(src_n, d.mclip);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, src_n);
        defer src.deinit();

        const scp = if (d.vcheck > 0 and d.sclip != null) zapi.initZFrame(d.sclip, n) else null;
        defer if (scp) |s| s.deinit();

        const mcp = if (d.mclip != null) zapi.initZFrame(d.mclip, src_n) else null;
        defer if (mcp) |m| m.deinit();

        const dst = if (d.horizontal)
            src.newVideoFrame3(.{ .width = d.vi.width })
        else
            src.newVideoFrame3(.{ .height = d.vi.height });
        const dst_props = dst.getPropertiesRW();

        var field: u8 = d.field & 1;
        switch (dst_props.getFieldBased() orelse .PROGRESSIVE) {
            .BOTTOM => field = 0,
            .TOP => field = 1,
            else => {},
        }
        if (d.field > 1) field = @as(u8, @intCast(n & 1)) ^ field;

        // Size the per-thread scratch off plane 0. The internal pipeline is
        // vertical, so the horizontal path sizes its line buffers off the source
        // HEIGHT (the transposed line length) and requests the srcT/dstT frames.
        const p0_w: u32, const p0_h: u32, const p0_stride: u32 = src.getDimensions2(f32, 0);
        var alloc_w: u32 = undefined;
        var alloc_stride: u32 = undefined;
        var alloc_n_interp: u32 = undefined;
        var srcT_rows: u32 = 0;
        var dstT_rows: u32 = 0;
        if (d.horizontal) {
            alloc_w = p0_h;
            alloc_stride = std.mem.alignForward(u32, p0_h, n_vec);
            alloc_n_interp = if (d.dh) p0_w else p0_w / 2;
            srcT_rows = p0_w;
            dstT_rows = if (d.dh) p0_w * 2 else p0_w;
        } else {
            alloc_w = p0_w;
            alloc_stride = p0_stride;
            alloc_n_interp = if (d.dh) p0_h else p0_h / 2;
        }

        const tid = std.Thread.getCurrentId();
        const scratch: *Scratch = blk: {
            d.pool_lock.lockUncancelable(io);
            defer d.pool_lock.unlock(io);
            if (d.pool.get(tid)) |s| break :blk s;
            const s = allocScratch(alloc_w, alloc_stride, alloc_n_interp, d.hp, srcT_rows, dstT_rows) catch {
                zapi.setFilterError(if (d.horizontal) "EEDI3H: failed to allocate memory." else "EEDI3: failed to allocate memory.");
                dst.deinit();
                return null;
            };
            d.pool.put(tid, s) catch unreachable;
            break :blk s;
        };

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const srcp: []const f32 = src.getReadSlice2(f32, plane);
            const dstp: []f32 = dst.getWriteSlice2(f32, plane);
            const scpp: ?[]const f32 = if (scp) |s| s.getReadSlice2(f32, plane) else null;
            // mclip is always a single Gray plane (see createImpl); the same mask
            // drives every processed plane.
            const maskp: ?[]const u8 = if (mcp) |m| m.getReadSlice2(u8, 0) else null;
            const mask_stride: u32 = if (mcp) |m| m.getStride2(u8, 0) else 0;

            if (d.horizontal) {
                const src_w: u32, const src_h: u32, const src_stride: u32 = src.getDimensions2(f32, plane);
                const dst_w: u32 = dst.getWidth(plane);
                const dst_stride: u32 = dst.getStride2(f32, plane);
                const Lstride: u32 = std.mem.alignForward(u32, src_h, n_vec);

                // The only strided work: bring src (and mclip/sclip) into
                // column-major layout, run the vertical pipeline on contiguous
                // columns, then transpose the result back into the dst frame.
                const srcT: []f32 = scratch.srcT;
                const dstT: []f32 = scratch.dstT;
                transposeF32(srcT, Lstride, srcp, src_stride, src_w, src_h);

                const maskT: ?[]const u8 = if (maskp) |mp| blk: {
                    transposeBlocked(u8, scratch.maskT, Lstride, mp, mask_stride, src_w, src_h);
                    break :blk scratch.maskT;
                } else null;

                const scpT: ?[]const f32 = if (d.vcheck > 0) (if (scpp) |sp| blk: {
                    const scp_stride: u32 = scp.?.getStride2(f32, plane);
                    transposeF32(scratch.scpT, Lstride, sp, scp_stride, dst_w, src_h);
                    break :blk scratch.scpT;
                } else null) else null;

                processPlane(d, scratch, srcT, dstT, scpT, maskT, Lstride, field, src_h, Lstride, src_w, dst_w);

                transposeF32(dstp, dst_stride, dstT, Lstride, src_h, dst_w);
            } else {
                const w: u32, const src_h: u32, const stride: u32 = src.getDimensions2(f32, plane);
                const dst_h: u32 = dst.getHeight(plane);
                processPlane(d, scratch, srcp, dstp, scpp, maskp, mask_stride, field, w, stride, src_h, dst_h);
            }
        }

        dst_props.setFieldBased(.PROGRESSIVE);

        if (d.field > 1) {
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
    const axis_name = if (horizontal) "width" else "height";

    var d: Data = .{};
    d.horizontal = horizontal;

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

    if (d.vi.format.sampleType != .Float or d.vi.format.bitsPerSample != 32) {
        map_out.setError(filter_name ++ ": only 32-bit float input is supported.");
        return;
    }

    const field = map_in.getValue(i32, "field") orelse 0;
    const mdis = map_in.getValue(i32, "mdis") orelse 20;
    const nrad = map_in.getValue(i32, "nrad") orelse 2;

    d.alpha = map_in.getValue(f32, "alpha") orelse 0.2;
    d.beta = map_in.getValue(f32, "beta") orelse 0.25;
    d.gamma = map_in.getValue(f32, "gamma") orelse 20.0;

    d.dh = map_in.getBool("dh") orelse false;
    d.hp = map_in.getBool("hp") orelse false;
    d.vthresh0 = map_in.getValue(f32, "vthresh0") orelse 32.0;
    d.vthresh1 = map_in.getValue(f32, "vthresh1") orelse 64.0;
    d.vthresh2 = map_in.getValue(f32, "vthresh2") orelse 4.0;

    // The interpolated axis is the height for EEDI3, the width for EEDI3H.
    const interp_axis: i32 = if (horizontal) d.vi.width else d.vi.height;

    if (field < 0 or field > 3) {
        map_out.setError(filter_name ++ ": field must be 0, 1, 2, or 3.");
        return;
    }

    if (d.dh and field > 1) {
        map_out.setError(filter_name ++ ": field must be 0 or 1 when dh=True.");
        return;
    }

    if (!d.dh and (interp_axis & 1) != 0) {
        map_out.setError(filter_name ++ ": " ++ axis_name ++ " must be mod 2 when dh=False.");
        return;
    }

    if (d.alpha < 0.0 or d.alpha > 1.0) {
        map_out.setError(filter_name ++ ": alpha must be between 0.0 and 1.0 (inclusive).");
        return;
    }

    if (d.beta < 0.0 or d.beta > 1.0) {
        map_out.setError(filter_name ++ ": beta must be between 0.0 and 1.0 (inclusive).");
        return;
    }

    if (d.alpha + d.beta > 1.0) {
        map_out.setError(filter_name ++ ": alpha + beta must be less than or equal to 1.0.");
        return;
    }

    if (d.gamma < 0.0) {
        map_out.setError(filter_name ++ ": gamma must be greater than or equal to 0.0.");
        return;
    }

    if (nrad < 0 or nrad > 3) {
        map_out.setError(filter_name ++ ": nrad must be between 0 and 3 (inclusive).");
        return;
    }

    if (mdis < 1 or mdis > 40) {
        map_out.setError(filter_name ++ ": mdis must be between 1 and 40 (inclusive).");
        return;
    }

    if (vcheck < 0 or vcheck > 3) {
        map_out.setError(filter_name ++ ": vcheck must be 0, 1, 2, or 3.");
        return;
    }

    if (vcheck > 0 and (d.vthresh0 <= 0.0 or d.vthresh1 <= 0.0 or d.vthresh2 <= 0.0)) {
        map_out.setError(filter_name ++ ": vthresh0, vthresh1 and vthresh2 must be greater than 0.0.");
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
            args.setInt("_Range", 1, .Replace);
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

    if (d.dh) {
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

    d.field = @intCast(field);
    d.mdis = @intCast(mdis);
    d.nrad = @intCast(nrad);
    d.vcheck = @intCast(vcheck);
    d.one_minus_ab = 1.0 - d.alpha - d.beta;
    d.alpha /= 3.0;
    d.beta /= 255.0;
    d.gamma /= 255.0;
    d.vthresh0 /= 255.0;
    d.vthresh1 /= 255.0;
    d.rcpVthresh0 = 1.0 / d.vthresh0;
    d.rcpVthresh1 = 1.0 / d.vthresh1;
    d.rcpVthresh2 = 1.0 / d.vthresh2;

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
    zapi.createVideoFilter(out, filter_name, &d.vi, getFrame, free, .Parallel, dep_buf[0..ndeps], data);
}

pub const args_string = "clip:vnode;field:int;dh:int:opt;alpha:float:opt;beta:float:opt;gamma:float:opt;nrad:int:opt;mdis:int:opt;hp:int:opt;vcheck:int:opt;vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;mclip:vnode:opt;";
