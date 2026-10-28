//! Dolby Vision RPU parsing and metadata mapping, a native port of the part
//! of libdovi (quietvoid's dovi_tool, `dolby_vision/src/rpu/`) that
//! vs-placebo's Tonemap uses, plus vs-placebo's `create_dovi_meta`
//! (src/dovi_meta.h, the libplacebo API >= 387 branches).
//!
//! Parsing follows `DoviRpu::parse_unspec62_nalu`: strip the NAL header,
//! remove the emulation-prevention bytes, check the 0x19 prefix, the 0x80
//! terminator and the CRC32 (MPEG-2), then read the header, the data
//! mapping (reshaping curves and NLQ) and `vdr_dm_data` (matrices, source
//! PQ range, L1 and L6 from the CM v2.9 and v4.0 extension blocks). The
//! other extension blocks are skipped by their signalled length instead of
//! being parsed and validated one by one; writing, editing and conversion
//! are not ported.

const std = @import("std");
const math = std.math;

const common = @import("../common.zig");

const allocator = common.kernel_allocator;

pub const Error = error{ InvalidRpu, OutOfMemory };

/// MSB-first bit reader over the unescaped RPU.
const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    fn available(r: *const Reader) usize {
        return r.data.len * 8 - r.pos;
    }

    fn aligned(r: *const Reader) bool {
        return r.pos & 7 == 0;
    }

    fn bit(r: *Reader) Error!bool {
        if (r.pos >= r.data.len * 8) return error.InvalidRpu;
        const b = (r.data[r.pos >> 3] >> @intCast(7 - (r.pos & 7))) & 1;
        r.pos += 1;
        return b != 0;
    }

    fn bits(r: *Reader, n: u32) Error!u64 {
        std.debug.assert(n <= 64);
        if (n > r.available()) return error.InvalidRpu;
        var v: u64 = 0;
        for (0..n) |_| v = (v << 1) | @intFromBool(try r.bit());
        return v;
    }

    fn int(r: *Reader, comptime T: type) Error!T {
        const U = @Int(.unsigned, @bitSizeOf(T));
        return @bitCast(@as(U, @intCast(try r.bits(@bitSizeOf(T)))));
    }

    /// unsigned Exp-Golomb
    fn ue(r: *Reader) Error!u64 {
        var zeros: u32 = 0;
        while (!try r.bit()) {
            zeros += 1;
            if (zeros > 63) return error.InvalidRpu;
        }
        if (zeros == 0) return 0;
        return ((@as(u64, 1) << @intCast(zeros)) - 1) + try r.bits(zeros);
    }

    /// signed Exp-Golomb
    fn se(r: *Reader) Error!i64 {
        const k = try r.ue();
        const m: i64 = @intCast((k >> 1) + (k & 1));
        return if (k & 1 == 1) m else -m;
    }

    fn skip(r: *Reader, n: u64) Error!void {
        if (n > r.available()) return error.InvalidRpu;
        r.pos += @intCast(n);
    }
};

pub const Header = struct {
    rpu_format: u16 = 0,
    vdr_rpu_profile: u8 = 0,
    vdr_rpu_level: u8 = 0,
    vdr_seq_info_present_flag: bool = false,
    coefficient_data_type: u8 = 0,
    coefficient_log2_denom: u64 = 0,
    /// bits of each coefficient's fraction
    coefficient_log2_denom_length: u32 = 0,
    bl_video_full_range_flag: bool = false,
    bl_bit_depth_minus8: u64 = 0,
    el_bit_depth_minus8: u64 = 0,
    vdr_bit_depth_minus8: u64 = 0,
    reserved_zero_3bits: u8 = 0,
    el_spatial_resampling_filter_flag: bool = false,
    disable_residual_flag: bool = false,
    vdr_dm_metadata_present_flag: bool = false,
    use_prev_vdr_rpu_flag: bool = false,

    fn parse(r: *Reader) Error!Header {
        if (try r.bits(6) != 2) return error.InvalidRpu; // rpu_type
        var h: Header = .{
            .rpu_format = @intCast(try r.bits(11)),
            .vdr_rpu_profile = @intCast(try r.bits(4)),
            .vdr_rpu_level = @intCast(try r.bits(4)),
            .vdr_seq_info_present_flag = try r.bit(),
        };
        if (h.vdr_seq_info_present_flag) {
            _ = try r.bit(); // chroma_resampling_explicit_filter_flag
            h.coefficient_data_type = @intCast(try r.bits(2));
            if (h.coefficient_data_type == 0) h.coefficient_log2_denom = try r.ue();
            _ = try r.bits(2); // vdr_rpu_normalized_idc
            h.bl_video_full_range_flag = try r.bit();
            if (h.rpu_format & 0x700 == 0) {
                h.bl_bit_depth_minus8 = try r.ue();
                // the low 8 bits; the next ones are ext_mapping_idc
                h.el_bit_depth_minus8 = (try r.ue()) & 0xff;
                h.vdr_bit_depth_minus8 = try r.ue();
                _ = try r.bit(); // spatial_resampling_filter_flag
                h.reserved_zero_3bits = @intCast(try r.bits(3));
                h.el_spatial_resampling_filter_flag = try r.bit();
                h.disable_residual_flag = try r.bit();
            }
            h.coefficient_log2_denom_length = switch (h.coefficient_data_type) {
                0 => if (h.coefficient_log2_denom > 32) return error.InvalidRpu else @intCast(h.coefficient_log2_denom),
                1 => 32,
                else => return error.InvalidRpu,
            };
        }
        h.vdr_dm_metadata_present_flag = try r.bit();
        h.use_prev_vdr_rpu_flag = try r.bit();
        if (h.use_prev_vdr_rpu_flag) _ = try r.ue(); // prev_vdr_rpu_id
        return h;
    }

    /// RpuDataHeader::get_dovi_profile
    pub fn profile(h: *const Header) u8 {
        return switch (h.vdr_rpu_profile) {
            0 => if (h.bl_video_full_range_flag) 5 else 0,
            1 => if (h.el_spatial_resampling_filter_flag and !h.disable_residual_flag)
                (if (h.vdr_bit_depth_minus8 == 4) 7 else 4)
            else
                8,
            else => 0,
        };
    }

    fn validate(h: *const Header) Error!void {
        switch (h.profile()) {
            5 => if (h.vdr_rpu_profile != 0 or !h.bl_video_full_range_flag) return error.InvalidRpu,
            7, 8 => if (h.vdr_rpu_profile != 1) return error.InvalidRpu,
            else => {},
        }
        if (h.vdr_rpu_level != 0 or h.bl_bit_depth_minus8 != 2 or h.el_bit_depth_minus8 != 2 or
            h.vdr_bit_depth_minus8 > 6 or h.coefficient_log2_denom > 23) return error.InvalidRpu;
    }
};

/// Per-component pivots are limited to 9 by libplacebo's pl_reshape_data.
pub const max_pivots = 9;
pub const max_pieces = max_pivots - 1;
pub const mmr_coeffs = 7;

pub const Curve = struct {
    num_pivots: u32 = 0,
    /// as signalled: each one is a delta from the previous
    pivots: [max_pivots]u16 = @splat(0),
    mmr: bool = false,
    /// polynomial: order 1 or 2; MMR: 1 to 3
    order: [max_pieces]u8 = @splat(0),
    /// polynomial: up to 3 coefficients; MMR: the constant
    coef_int: [max_pieces][3]i64 = @splat(@splat(0)),
    coef: [max_pieces][3]u64 = @splat(@splat(0)),
    mmr_int: [max_pieces][3][mmr_coeffs]i64 = @splat(@splat(@splat(0))),
    mmr_frac: [max_pieces][3][mmr_coeffs]u64 = @splat(@splat(@splat(0))),
};

pub const Nlq = struct {
    offset: [3]u16 = @splat(0),
    vdr_in_max_int: [3]u64 = @splat(0),
    vdr_in_max: [3]u64 = @splat(0),
    slope_int: [3]u64 = @splat(0),
    slope: [3]u64 = @splat(0),
    threshold_int: [3]u64 = @splat(0),
    threshold: [3]u64 = @splat(0),

    /// RpuDataNlq::is_mel
    fn isMel(n: *const Nlq) bool {
        for (0..3) |c| {
            if (n.offset[c] != 0 or n.vdr_in_max_int[c] != 1 or n.vdr_in_max[c] != 0 or n.slope_int[c] != 0 or
                n.slope[c] != 0 or n.threshold_int[c] != 0 or n.threshold[c] != 0) return false;
        }
        return true;
    }
};

pub const Mapping = struct {
    curves: [3]Curve = @splat(.{}),
    /// NLQ (profile 7): the method is always linear deadzone
    nlq: ?Nlq = null,

    fn parse(r: *Reader, h: *const Header) Error!Mapping {
        var m: Mapping = .{};
        _ = try r.ue(); // vdr_rpu_id
        const color_space = try r.ue();
        const chroma_format = try r.ue();
        if (color_space != 0 or chroma_format != 0) return error.InvalidRpu;
        const bl_bits: u32 = @intCast(h.bl_bit_depth_minus8 + 8);
        const frac_bits = h.coefficient_log2_denom_length;
        for (&m.curves) |*c| {
            const n = try r.ue() + 2;
            if (n > max_pivots) return error.InvalidRpu;
            c.num_pivots = @intCast(n);
            for (c.pivots[0..c.num_pivots]) |*p| p.* = @intCast(try r.bits(bl_bits));
        }
        var nlq_present = false;
        if (h.rpu_format & 0x700 == 0 and !h.disable_residual_flag) {
            if (try r.bits(3) != 0) return error.InvalidRpu; // nlq_method_idc: linear deadzone
            _ = try r.bits(bl_bits); // nlq_pred_pivot_value
            _ = try r.bits(bl_bits);
            nlq_present = true;
        }
        _ = try r.ue(); // num_x_partitions_minus1
        _ = try r.ue(); // num_y_partitions_minus1

        for (&m.curves) |*c| {
            for (0..c.num_pivots - 1) |i| {
                const mapping_idc = try r.ue();
                if (mapping_idc > 1) return error.InvalidRpu;
                // vs-placebo reads one method per component (libdovi's last
                // piece); mixed pieces would index the wrong arrays there
                if (i == 0) c.mmr = mapping_idc == 1 else if (c.mmr != (mapping_idc == 1)) return error.InvalidRpu;
                if (!c.mmr) {
                    const order_minus1 = try r.ue();
                    if (order_minus1 > 1) return error.InvalidRpu;
                    // linear interpolation: unsupported by libdovi too
                    if (order_minus1 == 0 and try r.bit()) return error.InvalidRpu;
                    c.order[i] = @intCast(order_minus1 + 1);
                    for (0..c.order[i] + 1) |k| {
                        if (h.coefficient_data_type == 0) c.coef_int[i][k] = try r.se();
                        c.coef[i][k] = try r.bits(frac_bits);
                    }
                } else {
                    const order_minus1 = try r.bits(2);
                    if (order_minus1 > 2) return error.InvalidRpu;
                    c.order[i] = @intCast(order_minus1 + 1);
                    if (h.coefficient_data_type == 0) c.coef_int[i][0] = try r.se();
                    c.coef[i][0] = try r.bits(frac_bits);
                    for (0..c.order[i]) |j| {
                        for (0..mmr_coeffs) |k| {
                            if (h.coefficient_data_type == 0) c.mmr_int[i][j][k] = try r.se();
                            c.mmr_frac[i][j][k] = try r.bits(frac_bits);
                        }
                    }
                }
            }
        }

        if (nlq_present) {
            var q: Nlq = .{};
            const el_bits: u32 = @intCast(h.el_bit_depth_minus8 + 8);
            for (0..3) |c| {
                q.offset[c] = @intCast(try r.bits(el_bits));
                if (h.coefficient_data_type == 0) q.vdr_in_max_int[c] = try r.ue();
                q.vdr_in_max[c] = try r.bits(frac_bits);
                if (h.coefficient_data_type == 0) q.slope_int[c] = try r.ue();
                q.slope[c] = try r.bits(frac_bits);
                if (h.coefficient_data_type == 0) q.threshold_int[c] = try r.ue();
                q.threshold[c] = try r.bits(frac_bits);
            }
            m.nlq = q;
        }
        return m;
    }

    /// FEL when the NLQ is not the MEL no-op (get_enhancement_layer_type)
    pub fn fel(m: *const Mapping) bool {
        return if (m.nlq) |*q| !q.isMel() else false;
    }
};

pub const Level1 = struct { min_pq: u16, max_pq: u16, avg_pq: u16 };
pub const Level6 = struct { max_luminance: u16, min_luminance: u16, max_cll: u16, max_fall: u16 };

pub const DmData = struct {
    /// compressed DM data carries none of the fields below except L1/L6
    compressed: bool = false,
    ycc_to_rgb_coef: [9]i16 = @splat(0),
    ycc_to_rgb_offset: [3]u32 = @splat(0),
    rgb_to_lms_coef: [9]i16 = @splat(0),
    source_min_pq: u16 = 0,
    source_max_pq: u16 = 0,
    /// the last of each found in the CM v2.9, then v4.0 extension blocks
    level1: ?Level1 = null,
    level6: ?Level6 = null,

    fn parse(r: *Reader, h: *const Header) Error!DmData {
        var d: DmData = .{ .compressed = h.reserved_zero_3bits == 1 };
        const affected_dm_metadata_id = try r.ue();
        _ = try r.ue(); // current_dm_metadata_id
        _ = try r.ue(); // scene_refresh_flag
        if (affected_dm_metadata_id > 15) return error.InvalidRpu;
        if (!d.compressed) {
            for (&d.ycc_to_rgb_coef) |*v| v.* = try r.int(i16);
            for (&d.ycc_to_rgb_offset) |*v| v.* = try r.int(u32);
            for (&d.rgb_to_lms_coef) |*v| v.* = try r.int(i16);
            const signal_eotf = try r.int(u16);
            const p0 = try r.int(u16);
            const p1 = try r.int(u16);
            const p2 = try r.int(u32);
            const signal_bit_depth = try r.bits(5);
            _ = try r.bits(2 + 2 + 2); // color space, chroma format, full range
            d.source_min_pq = @intCast(try r.bits(12));
            d.source_max_pq = @intCast(try r.bits(12));
            _ = try r.bits(10); // source_diagonal
            if (signal_bit_depth < 8 or signal_bit_depth > 16) return error.InvalidRpu;
            if (p0 == 0 and p1 == 0 and p2 == 0 and signal_eotf != 65535) return error.InvalidRpu;
        }
        try d.parseBlocks(r);
        // CM v4.0 blocks follow when at least L254 + CRC32 + 0x80 fit
        if (r.available() >= 56) try d.parseBlocks(r);
        return d;
    }

    /// One `ext_metadata_block` list: L1 and L6 are read, the others skipped.
    fn parseBlocks(d: *DmData, r: *Reader) Error!void {
        const num = try r.ue();
        while (!r.aligned()) {
            if (try r.bit()) return error.InvalidRpu; // dm_alignment_zero_bit
        }
        for (0..num) |_| {
            const len = try r.ue();
            const level = try r.bits(8);
            if (len > r.available() / 8) return error.InvalidRpu;
            const start = r.pos;
            switch (level) {
                1 => {
                    if (len != 5) return error.InvalidRpu;
                    d.level1 = .{ .min_pq = @intCast(try r.bits(12)), .max_pq = @intCast(try r.bits(12)), .avg_pq = @intCast(try r.bits(12)) };
                },
                6 => {
                    if (len != 8) return error.InvalidRpu;
                    d.level6 = .{
                        .max_luminance = try r.int(u16),
                        .min_luminance = try r.int(u16),
                        .max_cll = try r.int(u16),
                        .max_fall = try r.int(u16),
                    };
                },
                else => {},
            }
            // the rest of the block, alignment bits included
            r.pos = start;
            try r.skip(len * 8);
        }
    }
};

pub const Rpu = struct {
    header: Header,
    /// absent with use_prev_vdr_rpu_flag
    mapping: ?Mapping = null,
    dm: ?DmData = null,

    pub fn profile(r: *const Rpu) u8 {
        return r.header.profile();
    }

    /// A full RPU (its own mapping and uncompressed DM data): its Meta
    /// does not depend on earlier frames.
    pub fn complete(r: *const Rpu) bool {
        return r.mapping != null and r.dm != null and !r.dm.?.compressed;
    }
};

/// DoviRpu::validated_trimmed_data: the payload from its 0x19 prefix on,
/// past a start code or NAL header.
fn trimmed(data: []const u8) Error![]const u8 {
    if (data.len < 25) return error.InvalidRpu;
    const d = data;
    if (std.mem.eql(u8, d[0..5], &.{ 0, 0, 0, 1, 25 })) return d[4..];
    if (std.mem.eql(u8, d[0..5], &.{ 0, 0, 1, 25, 8 })) return d[3..];
    if (std.mem.eql(u8, d[0..5], &.{ 0, 1, 25, 8, 9 }) or std.mem.eql(u8, d[0..5], &.{ 124, 1, 25, 8, 9 })) return d[2..];
    if (std.mem.eql(u8, d[0..4], &.{ 1, 25, 8, 9 })) return d[1..];
    if (std.mem.eql(u8, d[0..3], &.{ 25, 8, 9 })) return d;
    return error.InvalidRpu;
}

/// clear_start_code_emulation_prevention_3_byte; returns the length.
fn unescape(src: []const u8, dst: []u8) usize {
    if (src.len <= 2) {
        @memcpy(dst[0..src.len], src);
        return src.len;
    }
    dst[0] = src[0];
    dst[1] = src[1];
    var n: usize = 2;
    for (2..src.len) |i| {
        if (!(src[i - 2] == 0 and src[i - 1] == 0 and src[i] == 3)) {
            dst[n] = src[i];
            n += 1;
        }
    }
    return n;
}

/// DoviRpu::parse_unspec62_nalu: the `DolbyVisionRPU` frame property.
pub fn parse(data: []const u8) Error!Rpu {
    const t = try trimmed(data);
    var stack: [1024]u8 = undefined;
    const heap = if (t.len > stack.len) try allocator.alloc(u8, t.len) else null;
    defer if (heap) |b| allocator.free(b);
    const buf = heap orelse stack[0..t.len];
    const len = unescape(t, buf);
    return parseUnescaped(buf[0..len]);
}

/// DoviRpu::parse: CRC check, then the payload.
fn parseUnescaped(data: []const u8) Error!Rpu {
    // trailing zero bytes are ignored
    var end = data.len;
    while (end > 0 and data[end - 1] == 0) end -= 1;
    // prefix + payload + CRC32 + terminator
    if (end < 7 or data[end - 1] != 0x80) return error.InvalidRpu;
    const crc_start = end - 5;
    const crc = std.hash.crc.@"CRC-32/MPEG-2".hash(data[1..crc_start]);
    if (std.mem.readInt(u32, data[crc_start..][0..4], .big) != crc) return error.InvalidRpu;

    var r: Reader = .{ .data = data[0..end] };
    if (try r.bits(8) != 25) return error.InvalidRpu;
    var rpu: Rpu = .{ .header = try Header.parse(&r) };
    try rpu.header.validate();
    if (!rpu.header.use_prev_vdr_rpu_flag) {
        rpu.mapping = try Mapping.parse(&r, &rpu.header);
        // profile-specific NLQ presence (RpuDataMapping::validate)
        switch (rpu.profile()) {
            5, 8 => if (rpu.mapping.?.nlq != null) return error.InvalidRpu,
            7 => if (rpu.mapping.?.nlq == null) return error.InvalidRpu,
            else => {},
        }
    }
    if (rpu.header.vdr_dm_metadata_present_flag) rpu.dm = try DmData.parse(&r, &rpu.header);
    while (!r.aligned()) {
        if (try r.bit()) return error.InvalidRpu; // rpu_alignment_zero_bit
    }
    // unparsed data may sit between the payload and the CRC32
    if (r.available() < 40) return error.InvalidRpu;
    return rpu;
}

// ------------------------------------------------------------ metadata

/// libplacebo's pl_reshape_data for one component.
pub const Reshape = struct {
    /// 0: no reshaping
    num_pivots: u32 = 0,
    /// cumulative, in units of the BL code range (/ 2^bl_bit_depth)
    pivots: [max_pivots]f32 = @splat(0),
    mmr: bool = false,
    poly_coeffs: [max_pieces][3]f32 = @splat(@splat(0)),
    mmr_constant: [max_pieces]f32 = @splat(0),
    mmr_order: [max_pieces]u8 = @splat(0),
    /// orders past mmr_order are zero
    mmr_coeffs: [max_pieces][3][mmr_coeffs]f32 = @splat(@splat(@splat(0))),
};

pub const NlqData = struct { offset: f32 = 0, hdr_in_max: f32 = 0, slope: f32 = 0, threshold: f32 = 0 };

/// libplacebo's pl_dovi_metadata.
pub const Meta = struct {
    comp: [3]Reshape = @splat(.{}),
    nonlinear_offset: [3]f32 = @splat(0),
    /// YCbCr -> nonlinear RGB, and RGB -> LMS (row-major)
    nonlinear: [3][3]f32 = @splat(@splat(0)),
    linear: [3][3]f32 = @splat(@splat(0)),
    nlq_active: bool = false,
    nlq: [3]NlqData = @splat(.{}),

    /// vs-placebo's create_dovi_meta: applies `rpu` over the previous
    /// state, which keeps whatever the RPU does not signal (the mapping
    /// with use_prev_vdr_rpu_flag, the matrices without DM data).
    /// `el_present`: an enhancement layer clip is attached.
    pub fn update(m: *Meta, rpu: *const Rpu, el_present: bool) void {
        const h = &rpu.header;
        if (h.use_prev_vdr_rpu_flag) return;
        if (rpu.mapping) |*map| {
            const bl_bits: u5 = @intCast(h.bl_bit_depth_minus8 + 8);
            // f32 arithmetic throughout, as in the C code
            const coef_scale: f32 = 1.0 / @as(f32, @floatFromInt(@as(u64, 1) << @intCast(h.coefficient_log2_denom)));
            const scaled = struct {
                fn f(ipart: i64, fpart: u64, scale: f32) f32 {
                    return @as(f32, @floatFromInt(ipart)) + scale * @as(f32, @floatFromInt(fpart));
                }
            }.f;
            for (&m.comp, &map.curves) |*cmp, *curve| {
                cmp.num_pivots = curve.num_pivots;
                cmp.mmr = curve.mmr;
                var pivot: u16 = 0;
                for (0..curve.num_pivots) |i| {
                    pivot +%= curve.pivots[i];
                    cmp.pivots[i] = @as(f32, @floatFromInt(pivot)) / @as(f32, @floatFromInt(@as(u32, 1) << bl_bits));
                }
                for (0..curve.num_pivots - 1) |i| {
                    cmp.poly_coeffs[i] = @splat(0);
                    if (!curve.mmr) {
                        for (0..@as(usize, curve.order[i]) + 1) |k| cmp.poly_coeffs[i][k] = scaled(curve.coef_int[i][k], curve.coef[i][k], coef_scale);
                    } else {
                        cmp.mmr_constant[i] = scaled(curve.coef_int[i][0], curve.coef[i][0], coef_scale);
                        cmp.mmr_order[i] = curve.order[i];
                        for (0..3) |j| for (0..mmr_coeffs) |k| {
                            cmp.mmr_coeffs[i][j][k] = if (j < curve.order[i]) scaled(curve.mmr_int[i][j][k], curve.mmr_frac[i][j][k], coef_scale) else 0;
                        };
                    }
                }
            }
            m.nlq_active = el_present and !h.disable_residual_flag and map.fel();
            if (m.nlq_active) {
                const q = &map.nlq.?;
                // the C code stores the unsigned integer parts in int64_t
                for (&m.nlq, 0..) |*d, c| d.* = .{
                    .offset = @floatFromInt(q.offset[c]),
                    .hdr_in_max = scaled(@bitCast(q.vdr_in_max_int[c]), q.vdr_in_max[c], coef_scale),
                    .slope = scaled(@bitCast(q.slope_int[c]), q.slope[c], coef_scale),
                    .threshold = scaled(@bitCast(q.threshold_int[c]), q.threshold[c], coef_scale),
                };
            }
        }
        // compressed DM data repeats the previous metadata: libdovi leaves
        // its matrices zero (vs-placebo would decode black), keep the
        // previous ones instead
        if (h.vdr_dm_metadata_present_flag) if (rpu.dm) |*dm| if (!dm.compressed) {
            for (&m.nonlinear_offset, dm.ycc_to_rgb_offset) |*o, v| o.* = @as(f32, @floatFromInt(v)) / @as(f32, 1 << 28);
            for (0..9) |i| {
                m.nonlinear[i / 3][i % 3] = @floatCast(@as(f64, @floatFromInt(dm.ycc_to_rgb_coef[i])) / 8192.0);
                m.linear[i / 3][i % 3] = @floatCast(@as(f64, @floatFromInt(dm.rgb_to_lms_coef[i])) / 16384.0);
            }
        };
    }
};

fn hexBytes(comptime hex: []const u8) [hex.len / 2]u8 {
    @setEvalBranchQuota(100_000);
    var out: [hex.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch unreachable;
    return out;
}

// sample RPUs from dovi_tool's assets/tests (HEVC unspec62 NAL units)
const test_profile5 = hexBytes("00000001190809004061b6506f003ff801ffc00fffd000001000001b00000302000003036000004000006880000c7c1a448003f16c110c8000042fa95c0000030000200000030020000003010ae7fa8ffa8ffa8d0ae7fa8ffa8ffa8d0ae7fffc00000300000300000300019081f738054630080169b99980c028218008008005b6854100020200000300080500000300000300000300120c1f40006400000300007b68c48480");
const test_profile8 = hexBytes("0000000119080908406136506f003ff801ffc00fffd0000008000006800000400000340000030200000301a2566000035ea2566f9fceb1c256644ca00000100000030080000003008000000301c36224301860a5e308e0514000001a63e5affff000000300000300000300060207dce0151830080169b99980c028218008008005b6854100020200000300080500000300000300000300120c1f4000640000030000d2d3f72280");
const test_fel = hexBytes("000000011908090840613650ae20002008020080200802007f801ffc00ffc001fffa0000030100000300d0000008000006800000400000340000030200000301a000001000000d0000030080000068000004000003034000002000000a86d9e67b2f863efc189e6029e98c06b8c566a7cd5f8dffcb6d8c39e38a9981b86ca6790f405f434cb7d80e9e04c349fd3bb77ec2cd5de3a147d44c0a2e8062164a029b5d59500ed38f58c46c4ec13a14582cff3023536d57bbb6e411483b92966413698fce6bc31bd620fadd405bbef4643160b1f01bf2eefbf2a5e2f3ccb4abf3cc3983779e7568311b004800004004004000004012000010010010000010048000040040040000072566000035ea2566f9fceb1c256644ca0000030100000300080000030008000003001c36224301860a5e308e051400000301a63e5affff000003000003000003000060200f80e1518030080059ca1200c028218df82580080061400002022a512088050000030002281150120c07d000020d60015eb8f29d8280");
const test_mel = hexBytes("000000011908090840613650af003ff801ffc00ffc001fffa000001000000d0000030080000068000004000003000040000020000020000003000400000302000003020000030000400000200000200000392b300001af512b37cfe758e12b3226500000080000030040000003004000000300e1b112180c3052f1847028a000000d31f2d7fff8000003000003000003000301007c070a8c300800410a9e00c028215858004af8006cb00002021a8959880500000300000300000300120c07d0000205d2005e258dd4a680");
const test_profile4 = hexBytes("000000011908090840613670ae70007f1fc7f1fc7f1fd05801ffc00ffc001fff506808db3cbce7eb99b150c9b1940da8aff645c543c9b34afca2ca1e7f5477c28bdade1797913b566bd205cef4b8a2f3e1524634d22526592195da3d6ae528380000052f5fdf57d3a1636edae60998c0f05c61fe45c392884b80f504fe7bb4fbd7038425b14008bf13d98b9f55fe8cda707da6436304c63d5c9d82c1ce8c1d5b69927dde10fead4ec00000400000285f056a4ec50d30d69a5400a542e182e6212c2cba48204fbe0c0168699b74ccd2c3b92dd60234a92c2a18fbb57de30110714b020031669805c415335b308a302a990a379665034a0000030200000300dea6c769800c648000007220e483e0013de000001da830ff18004f6800000e4ace0000732c4acff24dddc24ace87b800000800000300400000030040000003002daa4bcc068a14105fc00c320000054e7ab333347bf601c8002a3968e040fb9c42a2003008007ffa0000c0282180080080080080010002025d65d600aa9f1db080");
const test_cmv40 = hexBytes("0000000119080908406136506f003ff801ffc00fffd0000008000006800000400000340000030200000301c95980000d7a8959be7f3ac7095991328000004000000302000003000200000300070d8890c06182978c23814500000300698f96bfffc000000300000300000300180803e038546030080041099980c02b2380080080080080040002020000030008050000030002301180120c07d0000207d0032070301c0040040000b08ff8008008008008008004120060affbb800002285808800003fe000208d90ba8480");
const test_trailing = hexBytes("000000011908090840613650af003ff801ffc00ffc001fffa000001000000d0000030080000068000004000003020090000080080080000080240000200200200000200900000800800800000e4acc00006bd44acdf3f9d6384acc8994000003020000030010000003001000000300386c4486030c14bc611c0a28000003034c7cb5fffe0000030000030000030000c040fb9c02a3003008064ca1bb80c028216c380065b8009d510002020b08fe080500000300005402a0120c1f400064085a05ba62834e73800000");

test "dovi_tool sample RPUs: profile and EL type" {
    const Case = struct { data: []const u8, profile: u8, fel: ?bool };
    const cases = [_]Case{
        .{ .data = &test_profile4, .profile = 4, .fel = null },
        .{ .data = &test_profile5, .profile = 5, .fel = null },
        .{ .data = &test_profile8, .profile = 8, .fel = null },
        .{ .data = &test_fel, .profile = 7, .fel = true },
        .{ .data = &test_mel, .profile = 7, .fel = false },
        .{ .data = &test_cmv40, .profile = 8, .fel = null },
        .{ .data = &test_trailing, .profile = 7, .fel = null },
    };
    for (cases, 0..) |cs, i| {
        errdefer std.debug.print("case {d}\n", .{i});
        const rpu = try parse(cs.data);
        try std.testing.expectEqual(cs.profile, rpu.profile());
        try std.testing.expect(rpu.complete());
        if (cs.fel) |fel| try std.testing.expectEqual(fel, rpu.mapping.?.fel());
        try std.testing.expect(rpu.dm.?.level1 != null);
        // the whole NAL unit or just the payload
        _ = try parse(cs.data[4..]);
    }
}

test "profile 8.1 RPU maps to identity reshaping and the BT.2020 matrices" {
    const rpu = try parse(&test_profile8);
    var m: Meta = .{};
    m.update(&rpu, false);
    for (m.comp) |c| {
        try std.testing.expectEqual(@as(u32, 2), c.num_pivots);
        try std.testing.expect(!c.mmr);
        try std.testing.expectEqual(@as(f32, 0), c.pivots[0]);
        try std.testing.expectEqual(@as(f32, 1023.0 / 1024.0), c.pivots[1]);
        // s -> 0 + 1 * s
        try std.testing.expectEqual([3]f32{ 0, 1, 0 }, c.poly_coeffs[0]);
    }
    // limited-range YCbCr -> RGB, with the levels folded in: Y gain ~255/219
    try std.testing.expect(m.nonlinear[0][0] > 1.1 and m.nonlinear[0][0] < 1.2);
    try std.testing.expect(!m.nlq_active);
    const dm = rpu.dm.?;
    try std.testing.expect(dm.source_max_pq > dm.source_min_pq);
}

test "partial RPUs keep the previous state" {
    const full = try parse(&test_profile8);
    var m: Meta = .{};
    m.update(&full, false);
    const before = m;
    // compressed DM data (libdovi leaves its matrices zero)
    var compressed = full;
    compressed.dm.?.compressed = true;
    compressed.dm.?.ycc_to_rgb_coef = @splat(0);
    compressed.dm.?.rgb_to_lms_coef = @splat(0);
    try std.testing.expect(!compressed.complete());
    m.update(&compressed, false);
    try std.testing.expectEqual(before.nonlinear, m.nonlinear);
    try std.testing.expectEqual(before.linear, m.linear);
    // use_prev_vdr_rpu_flag: nothing changes
    var prev = full;
    prev.header.use_prev_vdr_rpu_flag = true;
    prev.mapping = null;
    m.update(&prev, false);
    try std.testing.expectEqual(before.comp[0].pivots, m.comp[0].pivots);
    try std.testing.expect(!prev.complete());
}

test "corrupted RPUs are rejected" {
    var bad = test_profile8;
    bad[60] ^= 0x10; // CRC mismatch
    try std.testing.expectError(error.InvalidRpu, parse(&bad));
    try std.testing.expectError(error.InvalidRpu, parse(test_profile8[0..30]));
    const junk: [30]u8 = @splat(7);
    try std.testing.expectError(error.InvalidRpu, parse(&junk));
}
