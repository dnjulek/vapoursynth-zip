import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

library_suffixes = frozenset({'.so', '.dll', '.dylib'})

cpus = [
    {'cpu': 'x86_64'}, 
    {'cpu': 'haswell', 'opt_suffix': 'avx2'},
    {'cpu': 'znver4-sse4a', 'opt_suffix': 'zn4'}, # "-sse4a" means *remove* the use of SSE4a (which Intel never implemented)
]

targets = {
    # Linux
    'aarch64-linux-gnu': {
        'zig_target': 'aarch64-linux-gnu.2.17', 
        'python_platform_tag': 'manylinux_2_17_aarch64',
        'basename': 'libvszip',
    },
    'aarch64-linux-musl': {
        'zig_target': 'aarch64-linux-musl', 
        'python_platform_tag': 'musllinux_1_2_aarch64',
        'basename': 'libvszip',
    },
    'x86_64-linux-gnu': {
        'zig_target': 'x86_64-linux-gnu.2.17', 
        'python_platform_tag': 'manylinux_2_17_x86_64',
        'basename': 'libvszip',
        'cpus': cpus,
    },
    'x86_64-linux-musl': {
        'zig_target': 'x86_64-linux-musl', 
        'python_platform_tag': 'musllinux_1_2_x86_64',
        'basename': 'libvszip',
        'cpus': cpus, 
    },

    # Mac
    'aarch64-macos': {
        'zig_target': 'aarch64-macos', 
        'python_platform_tag': 'macosx_11_0_arm64',
        'basename': 'libvszip',
    },
    'x86_64-macos': {
        'zig_target': 'x86_64-macos', 
        'python_platform_tag': 'macosx_11_0_x86_64',
        'basename': 'libvszip',
    },

    # Windows
    'x86_64-windows': {
        'zig_target': 'x86_64-windows', 
        'python_platform_tag': 'win_amd64',
        'basename': 'vszip',
        'cpus': cpus, 
    },
}


class CustomHook(BuildHookInterface[Any]):
    """
    Custom build hook to compile the Zig project and package the resulting binaries.
    """

    source_dir = Path("zig-out")
    target_dir = Path("vapoursynth/plugins/vszip")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Called before the build process starts.
        Sets build metadata and executes the Zig compilation.
        """
        # https://hatch.pypa.io/latest/plugins/builder/wheel/#build-data
        build_data["pure_python"] = False

        # Ensure the target directory exists
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Cross-compilation support
        if "ZTARGET" in os.environ:
            # Ex: ZTARGET="x86_64-linux-gnu"
            ztarget = os.environ["ZTARGET"]

            if ztarget not in targets:
                raise ValueError(f"Unsupported target {ztarget}")

            target = targets[ztarget]
            zig_target = target['zig_target']
            python_platform_tag = target['python_platform_tag']

            build_data["tag"] = f"py3-none-{python_platform_tag}"

            if 'cpus' in target: 
                # Build all optimization levels of the library
                for cpu_spec in target['cpus']:
                    subprocess.run([sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast", f"-Dtarget={zig_target}", f"-Dcpu={cpu_spec['cpu']}"], check=True)

                    for file_path in self.source_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix in library_suffixes:
                            if 'opt_suffix' in cpu_spec:
                                name = file_path.stem + f".{cpu_spec['opt_suffix']}" + file_path.suffix
                                shutil.copy2(file_path, Path(self.target_dir, name))
                            else:
                                shutil.copy2(file_path, self.target_dir)
            else:
                subprocess.run([sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast", f"-Dtarget={zig_target}"], check=True)

                for file_path in self.source_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix in library_suffixes:
                        shutil.copy2(file_path, self.target_dir)

            # Write a manifest to ensure instruction set-based loading works as desired
            # https://github.com/vapoursynth/vapoursynth/discussions/1196
            manifest_path = Path(self.target_dir, "manifest.vs")
            with open(manifest_path, "wt") as manifest:
                manifest.writelines([
                    "[VapourSynth Manifest V1]\n"
                    f"{target['basename']}\n"
                ])

        # Build for *this* machine
        else:
            build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"

            subprocess.run([sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast"], check=True)
        
            # Copy the compiled binaries
            for file_path in self.source_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in library_suffixes:
                    shutil.copy2(file_path, self.target_dir)


    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """
        Called after the build process finishes.
        Cleans up temporary build artifacts.
        """
        shutil.rmtree(self.target_dir.parent, ignore_errors=True)
