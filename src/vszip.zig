const std = @import("std");
pub const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
pub const zigimg = @import("zigimg");
const zon = @import("zon");

const adaptive_binarize = @import("vapoursynth/adaptive_binarize.zig");
const bilateral = @import("vapoursynth/bilateral.zig");
const boxblur = @import("vapoursynth/boxblur.zig");
const checkmate = @import("vapoursynth/checkmate.zig");
const clahe = @import("vapoursynth/clahe.zig");
const color_map = @import("vapoursynth/color_map.zig");
const comb_mask_mt = @import("vapoursynth/comb_mask_mt.zig");
const comb_mask = @import("vapoursynth/comb_mask.zig");
const deband = @import("vapoursynth/deband.zig");
const image_read = @import("vapoursynth/image_read.zig");
const limit_filter = @import("vapoursynth/limit_filter.zig");
const limiter = @import("vapoursynth/limiter.zig");
const packrgb = @import("vapoursynth/packrgb.zig");
const pavg = @import("vapoursynth/planeaverage.zig");
const pmm = @import("vapoursynth/planeminmax.zig");
const rfs = @import("vapoursynth/rfs.zig");
const ssimulacra2 = @import("vapoursynth/ssimulacra2.zig");
const xpsnr = @import("vapoursynth/xpsnr.zig");

pub const vec_len = std.simd.suggestVectorLength(u8) orelse 32;
pub const alignment = std.mem.Alignment.fromByteUnits(vec_len);

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszip", "vszip", "VapourSynth Zig Image Process", zon.version, plugin, vspapi);

    ZAPI.Plugin.function(
        adaptive_binarize.filter_name,
        "clip:vnode;clip2:vnode;c:int:opt;",
        "clip:vnode;",
        adaptive_binarize.adaptiveBinarizeCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        bilateral.filter_name,
        "clip:vnode;ref:vnode:opt;sigmaS:float[]:opt;sigmaR:float[]:opt;planes:int[]:opt;algorithm:int[]:opt;PBFICnum:int[]:opt",
        "clip:vnode;",
        bilateral.bilateralCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        boxblur.filter_name,
        "clip:vnode;planes:int[]:opt;hradius:int:opt;hpasses:int:opt;vradius:int:opt;vpasses:int:opt",
        "clip:vnode;",
        boxblur.boxBlurCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        checkmate.filter_name,
        "clip:vnode;thr:int:opt;tmax:int:opt;tthr2:int:opt;",
        "clip:vnode;",
        checkmate.checkmateCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        clahe.filter_name,
        "clip:vnode;limit:int:opt;tiles:int[]:opt",
        "clip:vnode;",
        clahe.claheCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        color_map.filter_name,
        "clip:vnode;color:int:opt;",
        "clip:vnode;",
        color_map.colorMapCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        comb_mask_mt.filter_name,
        "clip:vnode;thY1:int:opt;thY2:int:opt;",
        "clip:vnode;",
        comb_mask_mt.combMaskMTCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        comb_mask.filter_name,
        "clip:vnode;cthresh:int:opt;mthresh:int:opt;expand:int:opt;metric:int:opt;",
        "clip:vnode;",
        comb_mask.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        deband.filter_name,
        "clip:vnode;range:int:opt;thr:float[]:opt;grain:float[]:opt;sample_mode:int:opt;seed:int:opt;blur_first:int:opt;dynamic_grain:int:opt;" ++
            "keep_tv_range:int:opt;random_algo_ref:int:opt;random_algo_grain:int:opt;random_param_ref:float:opt;random_param_grain:float:opt;" ++
            "thr1:float[]:opt;thr2:float[]:opt;angle_boost:float:opt;max_angle:float:opt;",
        "clip:vnode;",
        deband.create,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        image_read.filter_name,
        "path:data[];validate:int:opt;",
        "clip:vnode;",
        image_read.readCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        limit_filter.filter_name,
        "flt:vnode;src:vnode;ref:vnode:opt;dark_thr:float[]:opt;bright_thr:float[]:opt;elast:float[]:opt;planes:int[]:opt;",
        "clip:vnode;",
        limit_filter.limitFilterCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        limiter.filter_name,
        "clip:vnode;min:float[]:opt;max:float[]:opt;tv_range:int:opt;mask:int:opt;planes:int[]:opt;",
        "clip:vnode;",
        limiter.limiterCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        packrgb.filter_name,
        "clip:vnode;",
        "clip:vnode;",
        packrgb.packrgbCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        pavg.filter_name,
        "clipa:vnode;exclude:int[];clipb:vnode:opt;planes:int[]:opt;prop:data:opt;",
        "clip:vnode;",
        pavg.planeAverageCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        pmm.filter_name,
        "clipa:vnode;minthr:float:opt;maxthr:float:opt;clipb:vnode:opt;planes:int[]:opt;prop:data:opt;",
        "clip:vnode;",
        pmm.planeMinMaxCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        rfs.filter_name,
        "clipa:vnode;clipb:vnode;frames:int[];mismatch:int:opt;planes:int[]:opt;",
        "clip:vnode;",
        rfs.rfsCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        ssimulacra2.filter_name,
        "reference:vnode;distorted:vnode;",
        "clip:vnode;",
        ssimulacra2.ssimulacraCreate,
        plugin,
        vspapi,
    );
    ZAPI.Plugin.function(
        xpsnr.filter_name,
        "reference:vnode;distorted:vnode;temporal:int:opt;verbose:int:opt;",
        "clip:vnode;",
        xpsnr.xpsnrCreate,
        plugin,
        vspapi,
    );
}
