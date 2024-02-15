#!/bin/sh

set -x

wget https://github.com/ziglang/www.ziglang.org/raw/master/data/releases.json
json_file="releases.json"

VER=$(jq -r '.master.version' "${json_file}")
ZNAME="zig-linux-x86_64-${VER}"

wget "https://ziglang.org/builds/${ZNAME}.tar.xz"
tar -xf "${ZNAME}.tar.xz"

"${ZNAME}/zig" build -Doptimize=ReleaseFast

sudo mv zig-out/lib/vszip.so /usr/lib/vapoursynth
