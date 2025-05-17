const std = @import("std");

pub const required_zig_version = std.SemanticVersion{
    .major = 0,
    .minor = 14,
    .patch = 0,
};

pub fn build(b: *std.Build) void {
    ensureZigVersion() catch return;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "vszip",
        .root_source_file = b.path("src/vszip.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    lib.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    const vapoursynth_dep = b.dependency("vapoursynth", .{
        .target = target,
        .optimize = optimize,
    });

    lib.root_module.addImport("vapoursynth", vapoursynth_dep.module("vapoursynth"));
    lib.linkLibC();

    if (lib.root_module.optimize == .ReleaseFast) {
        lib.root_module.strip = true;
    }

    b.installArtifact(lib);
}

fn ensureZigVersion() !void {
    var installed_ver = @import("builtin").zig_version;
    installed_ver.build = null;

    if (installed_ver.order(required_zig_version).compare(.neq)) {
        std.log.err("\n" ++
            \\---------------------------------------------------------------------------
            \\
            \\Installed Zig compiler version is not supported.
            \\
            \\Required version: {any}
            \\Installed version: {any}
            \\
            \\Please install version {any} and try again.
            \\https://ziglang.org/download/
            \\
            \\---------------------------------------------------------------------------
            \\
        , .{ required_zig_version, installed_ver, required_zig_version });
        return error.ZigVersion;
    }
}
