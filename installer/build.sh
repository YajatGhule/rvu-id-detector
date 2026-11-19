#!/bin/bash
# Builds dist/RVU-ID-Detector-Setup.exe on Linux.
# Needs the Zig compiler for cross compiling:  uv tool install ziglang  (or pip install ziglang)
# Usage: bash installer/build.sh [path/to/python-with-ziglang]
set -e
cd "$(dirname "$0")/.."
PY="${1:-python3}"
mkdir -p dist
"$PY" -m ziglang cc -target x86_64-windows-gnu -O2 -o dist/launcher.exe installer/launcher.c
git archive --format=zip -o dist/project.zip HEAD
size=$(stat -c %s dist/project.zip)
{
  cat dist/launcher.exe dist/project.zip
  printf 'RVUPAYLD'
  "$PY" -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<Q', $size))"
} > dist/RVU-ID-Detector-Setup.exe
rm -f dist/launcher.exe dist/launcher.pdb dist/project.zip
ls -la dist/RVU-ID-Detector-Setup.exe
