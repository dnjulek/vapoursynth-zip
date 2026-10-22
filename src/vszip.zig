//! vszip: VapourSynth Zig Image Process.
//!
//! Every function is registered in the `vszip` namespace; the AviSynth+ port
//! (`avszip.zig`) registers the same filters as `avszip_<Filter>`.

const std = @import("std");
pub const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");

// Shared with the kernels and the AviSynth adapters (see common.zig).
const common = @import("common.zig");
pub const vec_len = common.vec_len;
pub const alignment = common.alignment;
pub const io = common.io;

/// Adapter modules; each declares `filter_name`, `params` and `create`.
const modules = .{
    @import("vapoursynth/adaptive_binarize.zig"),
    @import("vapoursynth/adaptive_grain_mask.zig"),
    @import("vapoursynth/bilateral.zig"),
    @import("vapoursynth/bilateral_dither.zig"),
    @import("vapoursynth/bm3d.zig").BM3D,
    @import("vapoursynth/bm3d.zig").BM3Dv2,
    @import("vapoursynth/bm3d.zig").VAggregate,
    @import("vapoursynth/bm3d_classic.zig").Basic,
    @import("vapoursynth/bm3d_classic.zig").Final,
    @import("vapoursynth/bm3d_classic.zig").VBasic,
    @import("vapoursynth/bm3d_classic.zig").VFinal,
    @import("vapoursynth/bm3d_classic.zig").VAggregate,
    @import("vapoursynth/bm3d_classic.zig").RGB2OPP,
    @import("vapoursynth/bm3d_classic.zig").OPP2RGB,
    @import("vapoursynth/boxblur.zig"),
    @import("vapoursynth/checkmate.zig"),
    @import("vapoursynth/clahe.zig"),
    @import("vapoursynth/color_map.zig"),
    @import("vapoursynth/compress.zig"),
    @import("vapoursynth/comb_mask_mt.zig"),
    @import("vapoursynth/comb_mask.zig"),
    @import("vapoursynth/deband.zig"),
    @import("vapoursynth/dither.zig"),
    @import("vapoursynth/eedi3.zig").V,
    @import("vapoursynth/eedi3.zig").H,
    @import("vapoursynth/image_read.zig"),
    @import("vapoursynth/limit_filter.zig"),
    @import("vapoursynth/limiter.zig"),
    @import("vapoursynth/mosquito_nr.zig"),
    @import("vapoursynth/nlmeans.zig"),
    @import("vapoursynth/packrgb.zig"),
    @import("vapoursynth/planeaverage.zig"),
    @import("vapoursynth/planeminmax.zig"),
    @import("vapoursynth/rfs.zig"),
    @import("vapoursynth/ssimulacra2.zig"),
    @import("vapoursynth/wnnm.zig"),
    @import("vapoursynth/xpsnr.zig"),
};

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszip", "vszip", "VapourSynth Zig Image Process", zon.version, plugin, vspapi);
    inline for (modules) |m| {
        if (comptime enabled(m.filter_name)) {
            ZAPI.Plugin.function(m.filter_name, m.params, "clip:vnode;", m.create, plugin, vspapi);
        }
    }
}

/// `-Dfilters=A,B` (dev builds) registers only those filters; a module that
/// is not registered is never referenced, so its kernels are not compiled.
fn enabled(comptime name: []const u8) bool {
    comptime {
        if (zon.filters.len == 0) return true;
        @setEvalBranchQuota(100_000);
        var it = std.mem.tokenizeScalar(u8, zon.filters, ',');
        while (it.next()) |raw| {
            const want = std.mem.trim(u8, raw, " ");
            var known = false;
            for (modules) |m| known = known or std.ascii.eqlIgnoreCase(m.filter_name, want);
            if (!known) @compileError("-Dfilters: unknown filter '" ++ want ++ "' (names as registered, e.g. NLMeans, EEDI3H)");
            if (std.ascii.eqlIgnoreCase(want, name)) return true;
        }
        return false;
    }
}
