const std = @import("std");
const zon = @import("build.zig.zon");

pub fn build(b: *std.Build) !void {
    ensureZigVersion(try .parse(zon.minimum_zig_version)) catch return;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The VapourSynth plugin is the default product (pip/hatch builds run a
    // plain `zig build`); the AviSynth+ plugin is opt-in.
    const build_vs = b.option(bool, "vapoursynth", "Build the VapourSynth plugin (default: true)") orelse true;
    const build_avs = b.option(bool, "avisynth", "Build the AviSynth+ plugin (default: false)") orelse false;

    const options = b.addOptions();
    const version = try std.SemanticVersion.parse(zon.version);
    options.addOption(std.SemanticVersion, "version", version);
    // Dev builds: register only these filters (comma-separated, e.g.
    // -Dfilters=NLMeans,Limiter); the rest is never analyzed or compiled.
    const filters = b.option([]const u8, "filters", "Register only these VapourSynth filters (comma-separated; default: all)") orelse "";
    options.addOption([]const u8, "filters", filters);

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    if (build_vs) {
        const lib = b.addLibrary(.{
            .name = "vszip",
            .linkage = .dynamic,
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/vszip.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });

        lib.use_llvm = true;
        lib.root_module.addOptions("zon", options);
        lib.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

        const vapoursynth_dep = b.dependency("vapoursynth", .{
            .target = target,
            .optimize = optimize,
        });

        lib.root_module.addImport("vapoursynth", vapoursynth_dep.module("vapoursynth"));
        lib.root_module.link_libc = true;

        if (lib.root_module.optimize == .fast) {
            lib.root_module.strip = true;
        }

        b.installArtifact(lib);
    }

    if (build_avs) {
        // Lazy: fetched only for -Davisynth builds, so VapourSynth-only builds
        // (and the sdist/wheel builds) never need it.
        if (b.lazyDependency("avisynth", .{
            .target = target,
            .optimize = optimize,
        })) |avisynth_dep| {
            const avisynth_mod = avisynth_dep.builder.modules.get("avisynth") orelse {
                std.log.err("-Davisynth: the avisynth dependency (build.zig.zon) has no \"avisynth\" module; point it at an avisynth-zig checkout or `zig fetch --save git+https://github.com/dnjulek/avisynth-zig.git`", .{});
                return error.MissingAvisynthModule;
            };
            const lib = b.addLibrary(.{
                .name = "avszip",
                .linkage = .dynamic,
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/avszip.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });

            lib.use_llvm = true;
            lib.root_module.addOptions("zon", options);
            lib.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));
            lib.root_module.addImport("avisynth", avisynth_mod);
            lib.root_module.link_libc = true;

            if (lib.root_module.optimize == .fast) {
                lib.root_module.strip = true;
            }

            b.installArtifact(lib);
        }
    }

    // Host-free unit tests: every shared kernel compiles without a host binding.
    const unit = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/kernel_tests.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    unit.root_module.link_libc = true;
    unit.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));
    const test_step = b.step("test", "Run host-free kernel tests");
    test_step.dependOn(&b.addRunArtifact(unit).step);
}

fn ensureZigVersion(min_zig_version: std.SemanticVersion) !void {
    var installed_ver = @import("builtin").zig_version;
    installed_ver.build = null;

    if (installed_ver.order(min_zig_version) == .lt) {
        std.log.err("\n" ++
            \\---------------------------------------------------------------------------
            \\
            \\Installed Zig compiler version is too old.
            \\
            \\Min. required version: {any}
            \\Installed version: {any}
            \\
            \\Please install newer version and try again.
            \\Latest version can be found here: https://ziglang.org/download/
            \\
            \\---------------------------------------------------------------------------
            \\
        , .{ min_zig_version, installed_ver });
        return error.ZigIsTooOld;
    }
}
