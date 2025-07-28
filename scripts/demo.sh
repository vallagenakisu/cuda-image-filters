#!/usr/bin/env bash
# Render every filter over one image into out/, so you can flip through them.
#
#   ./scripts/demo.sh                      uses assets/sample.png
#   ./scripts/demo.sh photo.jpg            uses your own image
#   BACKEND=cpu ./scripts/demo.sh          run on the CPU instead
set -euo pipefail

BIN=${BIN:-./cuda-filters}
INPUT=${1:-assets/sample.png}
OUTDIR=${OUTDIR:-out}
BACKEND=${BACKEND:-cuda}

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not found - build it first (cmake --build build, or make)" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "error: no such image: $INPUT" >&2
  echo "       run 'make sample' to generate assets/sample.png" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
echo "input:   $INPUT"
echo "backend: $BACKEND"
echo

run() {
  local name=$1; shift
  printf '%-14s -> %s\n' "$name" "$OUTDIR/$name.png"
  "$BIN" -i "$INPUT" -o "$OUTDIR/$name.png" --backend "$BACKEND" -q "$@"
}

run grayscale   -f grayscale
run invert      -f invert
run blur-r2     -f gaussian -r 2
run blur-r8     -f gaussian -r 8
run box-r4      -f box -r 4
run sharpen     -f sharpen -r 2 --amount 1.5
run emboss      -f emboss
run laplacian   -f laplacian
run sobel       -f sobel
run tonemap     -f tonemap --exposure 2.0 --white 3.0

for k in kernels/*.kernel; do
  run "custom-$(basename "$k" .kernel)" -k "$k"
done

echo
echo "done - $(find "$OUTDIR" -name '*.png' | wc -l | tr -d ' ') images in $OUTDIR/"
