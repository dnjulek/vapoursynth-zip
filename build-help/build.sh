#!/bin/sh

if ! command -v wget > /dev/null; then
  echo "Error: wget is not installed. Please install wget first."
  exit 1
fi

if ! command -v jq > /dev/null; then
  echo "Error: jq is not installed. Please install jq first."
  exit 1
fi

ZNAME="zig-linux-x86_64-0.14.0"

if [ -e "${ZNAME}" ]
then
    echo "Using cached ${ZNAME}"
else
    echo "Downloading ${ZNAME}..."
    wget "https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz"
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
