#!/bin/sh

cd ..
rm -rf ./zig-out ./.zig-cache ./build
mkdir build

zig build -Doptimize=ReleaseFast -Dtarget=x86_64-windows -Dcpu=x86_64_v3
zip -9 -j build/vapoursynth-zip-r$1-windows-x86_64.zip zig-out/bin/vszip.dll

zig build -Doptimize=ReleaseFast -Dtarget=x86_64-windows -Dcpu=znver3
zip -9 -j build/vapoursynth-zip-r$1-windows-x86_64-znver3.zip zig-out/bin/vszip.dll

zig build -Doptimize=ReleaseFast -Dtarget=x86_64-windows -Dcpu=znver4
zip -9 -j build/vapoursynth-zip-r$1-windows-x86_64-znver4.zip zig-out/bin/vszip.dll

zig build -Doptimize=ReleaseFast -Dtarget=x86_64-macos
zip -9 -j build/vapoursynth-zip-r$1-macos-x86_64.zip zig-out/lib/libvszip.dylib

zig build -Doptimize=ReleaseFast -Dtarget=aarch64-macos
zip -9 -j build/vapoursynth-zip-r$1-macos-aarch64.zip zig-out/lib/libvszip.dylib

zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-gnu.2.17 -Dcpu=x86_64_v3
zip -9 -j build/vapoursynth-zip-r$1-linux-gnu-x86_64.zip zig-out/lib/libvszip.so

zig build -Doptimize=ReleaseFast -Dtarget=aarch64-linux-gnu.2.17
zip -9 -j build/vapoursynth-zip-r$1-linux-gnu-aarch64.zip zig-out/lib/libvszip.so