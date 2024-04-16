const std = @import("std");
const vszip = @import("vszip.zig");
const vs = vszip.vs;
const vsh = vszip.vsh;
const zapi = vszip.zapi;

pub const DataType = enum(c_int) {
    U8 = 1,
    U16 = 2,
    F32 = 4,
};

pub fn absDiff(x: anytype, y: anytype) @TypeOf(x) {
    return if (x > y) (x - y) else (y - x);
}

pub fn mapGetPlanes(in: ?*const vs.Map, out: ?*vs.Map, nodes: []?*vs.Node, process: []bool, num_planes: c_int, comptime name: []const u8, vsapi: ?*const vs.API) !void {
    const num_e = vsapi.?.mapNumElements.?(in, "planes");
    if (num_e < 1) {
        return;
    }

    @memset(process, false);

    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        for (nodes) |node| vsapi.?.freeNode.?(node);
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

pub fn compareNodes(out: ?*vs.Map, node1: ?*vs.Node, node2: ?*vs.Node, vi1: *const vs.VideoInfo, vi2: *const vs.VideoInfo, comptime name: []const u8, vsapi: ?*const vs.API) !void {
    if (node2 == null) {
        return;
    }

    var err_msg: ?[*]const u8 = null;
    errdefer {
        vsapi.?.mapSetError.?(out, err_msg.?);
        vsapi.?.freeNode.?(node1);
        vsapi.?.freeNode.?(node2);
    }

    if (!vsh.isSameVideoInfo(vi1, vi2) or !vsh.isConstantVideoFormat(vi2)) {
        err_msg = name ++ ": both input clips must have the same format.";
        return error.node;
    }
}
