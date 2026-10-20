//! avszip: AviSynth+ port of vszip (VapourSynth Zig Image Process).
//!
//! Every function is registered as `vszip_<Filter>` with vszip's argument
//! names and defaults (see README for the per-filter mapping). AviSynth+ also
//! exposes each one as `avszip_vszip_<Filter>` (DLL-basename alias).

const std = @import("std");
pub const avs = @import("avisynth");
const c = avs.c;
const zon = @import("zon");

/// Adapter modules; each declares `filter_name`, `params` and `create`.
const modules = .{
    @import("avisynth/limiter.zig"),
    @import("avisynth/planeaverage.zig"),
    @import("avisynth/planeminmax.zig"),
    @import("avisynth/rfs.zig"),
    @import("avisynth/boxblur.zig"),
    @import("avisynth/adaptive_binarize.zig"),
    @import("avisynth/adaptive_grain_mask.zig"),
    @import("avisynth/color_map.zig"),
    @import("avisynth/comb_mask.zig"),
    @import("avisynth/comb_mask_mt.zig"),
    @import("avisynth/limit_filter.zig"),
    @import("avisynth/compress.zig"),
    @import("avisynth/bilateral.zig"),
    @import("avisynth/bilateral_dither.zig"),
    @import("avisynth/mosquito_nr.zig"),
    @import("avisynth/dither.zig"),
    @import("avisynth/deband.zig"),
    @import("avisynth/checkmate.zig"),
    @import("avisynth/clahe.zig"),
    @import("avisynth/xpsnr.zig"),
    @import("avisynth/ssimulacra2.zig"),
    @import("avisynth/eedi3.zig").V,
    @import("avisynth/eedi3.zig").H,
    @import("avisynth/wnnm.zig").Main,
    @import("avisynth/wnnm.zig").Raw,
    @import("avisynth/image_read.zig"),
};

/// Interface floor: V11 (AviSynth+ 3.7.4+): 64-bit aware values through
/// `avisynth_c_plugin_init2`, frame properties and property writability.
const load_options: avs.LoadOptions = .{ .interface = 11 };

const functions = blk: {
    var list: [modules.len]avs.Function = undefined;
    for (modules, 0..) |m, i| {
        list[i] = .{ .name = m.filter_name, .params = m.params, .apply = avs.applyFn(m.filter_name, m.create) };
    }
    break :blk list;
};

const names = blk: {
    var list: [modules.len][:0]const u8 = undefined;
    for (modules, 0..) |m, i| list[i] = m.filter_name;
    break :blk list;
};

const version_string = std.fmt.comptimePrint("avszip {d}.{d}.{d} (vszip port)", .{ zon.version.major, zon.version.minor, zon.version.patch });

export fn avisynth_c_plugin_init2(env: ?*c.AVS_ScriptEnvironment) callconv(.c) [*:0]const u8 {
    const e = env orelse return "avszip: plugin init called without a script environment.";
    // A failed load leaves no API to register the filters, and LoadPlugin
    // still returns true: register stubs that raise the reason instead
    // ("avszip: cannot run: ... requires interface >= 11 ...").
    _ = avs.load(e, load_options) catch {
        _ = avs.registerFailureStubs(e, "avszip", &names, avs.getLastError());
        return avs.getLastError().ptr;
    };
    if (avs.register(.init(e), &functions)) |err| return err;
    return version_string;
}
