pub const vapoursynth = @import("vapoursynth");
pub const zigimg = @import("zigimg");
const vs = vapoursynth.vapoursynth4;

const adaptive_binarize = @import("vapoursynth/adaptive_binarize.zig");
const bilateral = @import("vapoursynth/bilateral.zig");
const boxblur = @import("vapoursynth/boxblur.zig");
const checkmate = @import("vapoursynth/checkmate.zig");
const clahe = @import("vapoursynth/clahe.zig");
const comb_mask_mt = @import("vapoursynth/comb_mask_mt.zig");
const image_read = @import("vapoursynth/image_read.zig");
const limiter = @import("vapoursynth/limiter.zig");
const metrics = @import("vapoursynth/metrics.zig");
const packrgb = @import("vapoursynth/packrgb.zig");
const pavg = @import("vapoursynth/planeaverage.zig");
const pmm = @import("vapoursynth/planeminmax.zig");
const rfs = @import("vapoursynth/rfs.zig");

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    _ = vspapi.configPlugin.?(
        "com.julek.vszip",
        "vszip",
        "VapourSynth Zig Image Process",
        vs.makeVersion(5, 0),
        vs.VAPOURSYNTH_API_VERSION,
        0,
        plugin,
    );

    _ = vspapi.registerFunction.?(
        adaptive_binarize.filter_name,
        "clip:vnode;clip2:vnode;c:int:opt;",
        "clip:vnode;",
        adaptive_binarize.adaptiveBinarizeCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        bilateral.filter_name,
        "clip:vnode;ref:vnode:opt;sigmaS:float[]:opt;sigmaR:float[]:opt;planes:int[]:opt;algorithm:int[]:opt;PBFICnum:int[]:opt",
        "clip:vnode;",
        bilateral.bilateralCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        boxblur.filter_name,
        "clip:vnode;planes:int[]:opt;hradius:int:opt;hpasses:int:opt;vradius:int:opt;vpasses:int:opt",
        "clip:vnode;",
        boxblur.boxBlurCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        checkmate.filter_name,
        "clip:vnode;thr:int:opt;tmax:int:opt;tthr2:int:opt;",
        "clip:vnode;",
        checkmate.checkmateCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        clahe.filter_name,
        "clip:vnode;limit:int:opt;tiles:int[]:opt",
        "clip:vnode;",
        clahe.claheCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        comb_mask_mt.filter_name,
        "clip:vnode;thY1:int:opt;thY2:int:opt;",
        "clip:vnode;",
        comb_mask_mt.combMaskMTCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        image_read.filter_name,
        "path:data[];",
        "clip:vnode;",
        image_read.readCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        limiter.filter_name,
        "clip:vnode;min:float[]:opt;max:float[]:opt;tv_range:int:opt;planes:int[]:opt;",
        "clip:vnode;",
        limiter.limiterCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        metrics.filter_name,
        "reference:vnode;distorted:vnode;mode:int:opt;",
        "clip:vnode;",
        metrics.metricsCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        packrgb.filter_name,
        "clip:vnode;",
        "clip:vnode;",
        packrgb.packrgbCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        pavg.filter_name,
        "clipa:vnode;exclude:int[];clipb:vnode:opt;planes:int[]:opt;prop:data:opt;",
        "clip:vnode;",
        pavg.planeAverageCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        pmm.filter_name,
        "clipa:vnode;minthr:float:opt;maxthr:float:opt;clipb:vnode:opt;planes:int[]:opt;prop:data:opt;",
        "clip:vnode;",
        pmm.planeMinMaxCreate,
        null,
        plugin,
    );
    _ = vspapi.registerFunction.?(
        rfs.filter_name,
        "clipa:vnode;clipb:vnode;frames:int[];mismatch:int:opt;planes:int[]:opt;",
        "clip:vnode;",
        rfs.rfsCreate,
        null,
        plugin,
    );
}
