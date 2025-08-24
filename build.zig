const std = @import("std");

pub const zig_min_version = std.SemanticVersion{
    .major = 0,
    .minor = 15,
    .patch = 1,
};

pub const zig_max_version = std.SemanticVersion{
    .major = 0,
    .minor = 15,
    .patch = 99,
};

pub fn build(b: *std.Build) void {
    ensureZigVersion() catch return;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "vszip",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vszip.zig"),
            .target = target,
            .optimize = optimize,
        }),
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

    if (installed_ver.order(zig_min_version).compare(.lt) or installed_ver.order(zig_max_version).compare(.gt)) {
        std.log.err("\n" ++
            \\---------------------------------------------------------------------------
            \\
            \\Installed Zig compiler version is incompatible.
            \\
            \\Min. supported version: {any}
            \\Max. supported version: {any}
            \\Installed version: {any}
            \\
            \\Please install compatible version and try again.
            \\https://ziglang.org/download/
            \\
            \\---------------------------------------------------------------------------
            \\
        , .{ zig_min_version, zig_max_version, installed_ver });
        return error.ZigVersion;
    }
}
