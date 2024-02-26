const vszip = @import("vszip.zig");
const vs = vszip.vs;
const vsh = vszip.vsh;

pub const DataType = enum(c_int) {
    U8 = 1,
    U16 = 2,
    F32 = 4,
};

pub fn absDiff(x: anytype, y: anytype) @TypeOf(x) {
    return if (x > y) (x - y) else (y - x);
}

pub fn mapGetPlanes(in: ?*const vs.Map, out: ?*vs.Map, nodes: []?*vs.Node, process: []bool, num_planes: c_int, comptime name: [*]const u8, vsapi: ?*const vs.API) !void {
    const num_e = vsapi.?.mapNumElements.?(in, "planes");
    if (num_e < 1) {
        return;
    }

    for (process) |*p| {
        p.* = false;
    }

    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        for (nodes) |node| {
            vsapi.?.freeNode.?(node);
        }
    }

    var i: c_int = 0;
    while (i < num_e) : (i += 1) {
        const e: i32 = vsh.mapGetN(i32, in, "planes", i, vsapi).?;
        if ((e < 0) or (e >= num_planes)) {
            err_msg = name ++ ": plane index out of range";
            return error.ValidationError;
        }

        const ue: u32 = @intCast(e);
        if (process[ue]) {
            err_msg = name ++ ": plane specified twice.";
            return error.ValidationError;
        }

        process[ue] = true;
    }
}

pub fn compareNodes(out: ?*vs.Map, node1: *vs.Node, node2: ?*vs.Node, comptime name: [*]const u8, vsapi: ?*const vs.API) !void {
    if (node2 == null) {
        return;
    }

    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        vsapi.?.freeNode.?(node1);
        vsapi.?.freeNode.?(node2);
    }

    const vi1 = vsapi.?.getVideoInfo.?(node1);
    const vi2 = vsapi.?.getVideoInfo.?(node2);
    if (!vsh.isSameVideoInfo(vi1, vi2) or !vsh.isConstantVideoFormat(vi2)) {
        err_msg = name ++ ": both input clips must have the same format.";
        return error.node;
    }
}

pub fn newVideoFrame(src: ?*const vs.Frame, core: ?*vs.Core, vsapi: ?*const vs.API) ?*vs.Frame {
    return vsapi.?.newVideoFrame.?(
        vsapi.?.getVideoFrameFormat.?(src),
        vsapi.?.getFrameWidth.?(src, 0),
        vsapi.?.getFrameHeight.?(src, 0),
        src,
        core,
    );
}

pub fn newVideoFrame2(src: ?*const vs.Frame, process: []bool, core: ?*vs.Core, vsapi: ?*const vs.API) ?*vs.Frame {
    var planes = [_]c_int{ 0, 1, 2 };
    var cp_planes = [_]?*const vs.Frame{
        if (process[0]) null else src,
        if (process[1]) null else src,
        if (process[2]) null else src,
    };

    return vsapi.?.newVideoFrame2.?(
        vsapi.?.getVideoFrameFormat.?(src),
        vsapi.?.getFrameWidth.?(src, 0),
        vsapi.?.getFrameHeight.?(src, 0),
        &cp_planes[0],
        &planes[0],
        src,
        core,
    );
}
