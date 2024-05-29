const vapoursynth = @import("vapoursynth");
const bilateral = @import("vapoursynth/bilateral.zig");
const boxblur = @import("vapoursynth/boxblur.zig");
const rfs = @import("vapoursynth/rfs.zig");
const pavg = @import("vapoursynth/planeaverage.zig");
const pmm = @import("vapoursynth/planeminmax.zig");

pub const vs = vapoursynth.vapoursynth4;
pub const vsh = vapoursynth.vshelper;
pub const zapi = vapoursynth.zigapi;

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    _ = vspapi.configPlugin.?("com.julek.vszip", "vszip", "VapourSynth Zig Image Process", vs.makeVersion(1, 0), vs.VAPOURSYNTH_API_VERSION, 0, plugin);
    _ = vspapi.registerFunction.?(bilateral.filter_name, "clip:vnode;ref:vnode:opt;sigmaS:float[]:opt;sigmaR:float[]:opt;planes:int[]:opt;algorithm:int[]:opt;PBFICnum:int[]:opt", "clip:vnode;", bilateral.bilateralCreate, null, plugin);
    _ = vspapi.registerFunction.?(boxblur.filter_name, "clip:vnode;planes:int[]:opt;hradius:int:opt;hpasses:int:opt;vradius:int:opt;vpasses:int:opt", "clip:vnode;", boxblur.boxBlurCreate, null, plugin);
    _ = vspapi.registerFunction.?(pavg.filter_name, "clipa:vnode;exclude:int[];clipb:vnode:opt;planes:int[]:opt;prop:data:opt;", "clip:vnode;", pavg.planeAverageCreate, null, plugin);
    _ = vspapi.registerFunction.?(pmm.filter_name, "clipa:vnode;minthr:float:opt;maxthr:float:opt;clipb:vnode:opt;planes:int[]:opt;prop:data:opt;", "clip:vnode;", pmm.planeMinMaxCreate, null, plugin);
    _ = vspapi.registerFunction.?(rfs.filter_name, "clipa:vnode;clipb:vnode;frames:int[];mismatch:int:opt;planes:int[]:opt;", "clip:vnode;", rfs.rfsCreate, null, plugin);
}
