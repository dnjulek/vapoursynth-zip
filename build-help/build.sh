#!/bin/sh

if ! command -v wget > /dev/null; then
  echo "Error: wget is not installed. Please install wget first."
  exit 1
fi

if ! command -v jq > /dev/null; then
  echo "Error: jq is not installed. Please install jq first."
  exit 1
fi

json_file="index.json" 

if [ -e "${json_file}" ]
then
    echo "Using cached Zig index.json"
else
    echo "Downloading index.json..."
    wget https://ziglang.org/download/index.json
fi

VER=$(jq -r '.master.version' "${json_file}")
ZNAME="zig-linux-x86_64-${VER}"

if [ -e "${ZNAME}" ]
then
    echo "Using cached ${ZNAME}"
else
    echo "Downloading ${ZNAME}..."
    wget "https://ziglang.org/builds/${ZNAME}.tar.xz"
    tar -xf "${ZNAME}.tar.xz"
fi

cd ..

echo "Building..."
"build-help/${ZNAME}/zig" build -Doptimize=ReleaseFast

echo "Installing libvszip.so to /usr/lib/vapoursynth"
if [ -e /usr/lib/vapoursynth ]
then
    sudo cp zig-out/lib/libvszip.so /usr/lib/vapoursynth
else
    sudo mkdir /usr/lib/vapoursynth
    sudo cp zig-out/lib/libvszip.so /usr/lib/vapoursynth
fi
