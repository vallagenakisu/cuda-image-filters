#!/usr/bin/env bash
# Full test pass.
#
# Stage 1 is host-only and runs anywhere. Stage 2 needs a GPU and checks every
# CUDA filter against the CPU reference with --verify, which is the check that
# actually catches kernel indexing bugs.
set -uo pipefail

BIN=${BIN:-./cuda-filters}
TESTBIN=${TESTBIN:-./build/test_kernels}
INPUT=${INPUT:-assets/sample.png}
TMP=${TMP:-$(mktemp -d)}
failures=0

pass() { printf '  \033[32mpass\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; failures=$((failures + 1)); }

echo "== stage 1: unit tests (no GPU needed) =="
if [[ -x "$TESTBIN" ]]; then
  if "$TESTBIN" > "$TMP/units.log" 2>&1; then
    pass "convolution matrix builders ($(grep -c '^pass' "$TMP/units.log") assertions)"
  else
    fail "convolution matrix builders"
    cat "$TMP/units.log"
  fi
else
  echo "  skip  $TESTBIN not built (cmake --build build --target test_kernels, or make test)"
fi

if command -v python3 > /dev/null 2>&1; then
  if python3 tests/test_gui.py > "$TMP/gui.log" 2>&1; then
    pass "gui logic ($(grep -c '^pass' "$TMP/gui.log") assertions)"
  else
    fail "gui logic"
    cat "$TMP/gui.log"
  fi
else
  echo "  skip  python3 not found, gui tests not run"
fi

echo
echo "== stage 2: GPU vs CPU reference (needs a CUDA device) =="
if [[ ! -x "$BIN" ]]; then
  echo "  skip  $BIN not built"
  exit $((failures > 0))
fi
if ! "$BIN" --list-devices 2>&1 | grep -q "compute capability"; then
  echo "  skip  no CUDA device visible"
  exit $((failures > 0))
fi
if [[ ! -f "$INPUT" ]]; then
  echo "  skip  no input image at $INPUT (run 'make sample')"
  exit $((failures > 0))
fi

# --verify exits non-zero when the GPU result drifts further from the reference
# than float rounding can explain.
check() {
  local label=$1; shift
  if "$BIN" -i "$INPUT" -o "$TMP/out.png" --verify -q "$@" > "$TMP/v.log" 2>&1; then
    pass "$label"
  else
    fail "$label"
    sed 's/^/        /' "$TMP/v.log"
  fi
}

check "grayscale"                  -f grayscale
check "invert"                     -f invert
check "sobel"                      -f sobel
check "tonemap"                    -f tonemap --exposure 1.8
check "gaussian r=3 naive"         -f gaussian -r 3 --method naive
check "gaussian r=3 tiled"         -f gaussian -r 3 --method tiled
check "gaussian r=3 separable"     -f gaussian -r 3 --method separable
check "gaussian r=8 tiled"         -f gaussian -r 8 --method tiled
check "box r=5 tiled"              -f box -r 5 --method tiled
check "sharpen r=2 tiled"          -f sharpen -r 2 --method tiled
check "emboss tiled"               -f emboss --method tiled
check "laplacian naive"            -f laplacian --method naive
check "custom outline"             -k kernels/outline.kernel
check "custom motion blur (9x9)"   -k kernels/motion_blur.kernel
check "rgba input"                 -f gaussian -r 4 --channels 4
check "grayscale input"            -f gaussian -r 4 --channels 1
check "non-square block 32x8"      -f gaussian -r 3 --block 32x8
check "small block 8x8"            -f gaussian -r 6 --block 8x8

echo
if [[ $failures -eq 0 ]]; then
  echo "all tests passed"
else
  echo "$failures test(s) failed"
fi
exit $((failures > 0))
