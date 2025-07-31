#!/usr/bin/env python3
"""Tests for the GUI's pure logic: argument construction, kernel validation
and output parsing.

No server and no GPU - these are the parts most likely to be wrong, and they
are separated from the HTTP layer so they can be tested directly.

    python3 tests/test_gui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gui"))

from server import build_command, parse_output, validate_kernel_text  # noqa: E402

FAILURES = []


def check(cond, label, detail=""):
    print(("pass  " if cond else "FAIL  ") + label + ("" if cond else f"   <- {detail}"))
    if not cond:
        FAILURES.append(label)


def cmd(req, kernel=None):
    return " ".join(build_command("BIN", req, "in.png", "out.png", kernel))


def test_kernel_validation():
    print("\n-- kernel validation (mirrors src/conv_kernel.cpp) --")

    ok, msg, info = validate_kernel_text("-1 -1 -1\n-1 8 -1\n-1 -1 -1")
    check(ok, "3x3 outline accepted", msg)
    check(info["side"] == 3 and info["radius"] == 1, "reports 3x3 radius 1", info)
    check(abs(info["sum"]) < 1e-6, "outline sums to 0", info)

    ok, _, info = validate_kernel_text("name box\nnormalize 1\n" + "1 " * 25)
    check(ok and info["side"] == 5, "5x5 accepted")
    check(abs(info["sum"] - 1.0) < 1e-6, "normalize reports an effective sum of 1", info)
    check(info["name"] == "box", "name directive parsed", info)

    ok, _, info = validate_kernel_text("bias 128\n-2 -1 0\n-1 0 1\n0 1 2")
    check(ok and info["bias"] == 128.0, "bias directive parsed", info)

    # Comments must not be mistaken for weights.
    ok, _, info = validate_kernel_text("# a comment 9 9 9\n1 0 0\n0 1 0\n0 0 1  # trailing 5")
    check(ok and info["side"] == 3, "comments stripped, including trailing ones", info)

    for text, label in [
        ("1 2 3 4", "non-square weight count rejected"),
        ("1 2 3 4\n5 6 7 8\n9 1 2 3\n4 5 6 7", "even-sided matrix rejected"),
        ("# nothing here", "empty matrix rejected"),
        ("1 0 0\n0 zzz 0\n0 0 1", "junk token rejected"),
        ("normalize 1\n-1 -1 -1\n-1 8 -1\n-1 -1 -1", "normalize on a sum-zero matrix rejected"),
        ("", "empty string rejected"),
    ]:
        ok, msg, _ = validate_kernel_text(text)
        check(not ok, label, "unexpectedly accepted")
        check(bool(msg), f"  ...with a message: {msg!r}" if ok else f"  ...with a message: {msg!r}")

    # 65x65 is the documented limit; 67x67 is past it.
    ok, _, _ = validate_kernel_text("1 " * (65 * 65))
    check(ok, "65x65 (radius 32) accepted, the documented limit")
    ok, msg, _ = validate_kernel_text("1 " * (67 * 67))
    check(not ok and "32" in msg, "67x67 rejected as past the radius limit", msg)


def test_command_building():
    print("\n-- command construction --")

    base = {"backend": "cuda", "method": "auto", "block": "16x16"}

    c = cmd(dict(base, filter="gaussian", radius=5, sigma=2.0))
    check("-f gaussian" in c and "-r 5" in c and "-s 2.0" in c, "gaussian passes -f, -r, -s", c)

    c = cmd(dict(base, filter="grayscale"))
    check("-r " not in c and "-s " not in c, "grayscale passes no convolution params", c)
    check("--method" not in c, "grayscale passes no --method (it is not a convolution)", c)

    c = cmd(dict(base, filter="sharpen", radius=2, sigma=0, amount=1.5))
    check("--amount 1.5" in c, "sharpen passes --amount", c)

    c = cmd(dict(base, filter="tonemap", exposure=2.0, white=3.0, gamma=2.2))
    check(all(f in c for f in ("--exposure 2.0", "--white 3.0", "--gamma 2.2")),
          "tonemap passes its three params", c)
    check("-r " not in c, "tonemap passes no radius", c)

    c = cmd(dict(base, filter="custom"), kernel="k.kernel")
    check("-k k.kernel" in c and "-f custom" not in c, "custom passes -k and not -f", c)

    try:
        cmd(dict(base, filter="custom"))
        check(False, "custom without a kernel path raises")
    except ValueError:
        check(True, "custom without a kernel path raises")

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, method="tiled", block="32x8"))
    check("--method tiled" in c and "--block 32x8" in c, "cuda passes --method and --block", c)

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, backend="cpu", method="tiled"))
    check("--backend cpu" in c and "--method" not in c, "cpu backend drops --method", c)

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, block="bogus"))
    check("--block" not in c, "a malformed block string is dropped, not forwarded", c)

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, verify=True))
    check("--verify" in c, "verify flag forwarded", c)
    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, backend="cpu", verify=True))
    check("--verify" not in c, "verify dropped on the cpu backend (nothing to compare)", c)

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, bench=50))
    check("--bench 50" in c, "bench count forwarded", c)

    c = cmd(dict(base, filter="gaussian", radius=3, sigma=0, backend="../../evil"))
    check("../../evil" not in c, "an unknown backend string is not forwarded", c)


def test_output_parsing():
    print("\n-- output parsing --")

    sample = (
        "loaded  assets/sample.png  (1024x768x3)\n"
        "kernel  gaussian 9x9 (sum 1.000, separable)\n"
        "gpu     separable upload 0.412 ms | kernel 0.087 ms | download 0.395 ms | total 0.894 ms\n"
        "verify  exact match: every one of 2359296 channel values is identical\n"
        "wrote   out.png  (1024x768x3)\n"
    )
    info = parse_output(sample)
    check(info["loaded"] == "1024x768x3", "image dimensions parsed", info)
    check(info["kernel"].startswith("gaussian 9x9"), "kernel description parsed", info)
    check(info["timings"]["method"] == "separable", "method parsed", info)
    check(abs(info["timings"]["kernel_ms"] - 0.087) < 1e-9, "kernel time parsed", info)
    check(abs(info["timings"]["total_ms"] - 0.894) < 1e-9, "total time parsed", info)
    check(info["verify"].startswith("exact match"), "verify line parsed", info)

    info = parse_output("loaded  x.png  (64x64x1)\ncpu     reference backend\nwrote   out.png  (64x64x1)\n")
    check(info["timings"] is None, "cpu output yields no gpu timings", info)
    check(info["loaded"] == "64x64x1", "cpu output still yields dimensions", info)

    info = parse_output("")
    check(all(v is None for v in info.values()), "empty output parses to all-None", info)


if __name__ == "__main__":
    test_kernel_validation()
    test_command_building()
    test_output_parsing()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("all GUI logic tests pass")
