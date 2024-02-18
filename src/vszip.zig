const vapoursynth = @import("vapoursynth");
const rfs = @import("rfs.zig");
const pavg = @import("planeaverage.zig");

pub const vs = vapoursynth.vapoursynth4;
pub const vsh = vapoursynth.vshelper;

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    _ = vspapi.configPlugin.?("com.julek.zip", "zip", "Zig Image Process", vs.makeVersion(1, 0), vs.VAPOURSYNTH_API_VERSION, 0, plugin);
    _ = vspapi.registerFunction.?(pavg.filter_name, "clipa:vnode;exclude:int[];clipb:vnode:opt;planes:int[]:opt;", "clip:vnode;", pavg.planeAverageCreate, null, plugin);
    _ = vspapi.registerFunction.?(rfs.filter_name, "clip_a:vnode;clip_b:vnode;frames:int[];mismatch:int:opt;planes:int[]:opt;", "clip:vnode;", rfs.rfsCreate, null, plugin);
}
