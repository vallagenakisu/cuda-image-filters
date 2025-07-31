#!/usr/bin/env python3
"""Local web UI for cuda-image-filters.

Runs a small HTTP server on localhost that drives the `cuda-filters` binary:
pick an image, pick a filter or write your own convolution matrix, see the
result and the kernel timings side by side.

Standard library only - no pip install. The browser does the image decoding
and scaling, which is why there is no Pillow dependency.

    python3 gui/server.py                  # opens http://127.0.0.1:8765
    python3 gui/server.py --port 9000
    python3 gui/server.py --binary ./build/cuda-filters
    python3 gui/server.py --no-browser

The server binds to 127.0.0.1 only. It shells out to a local binary with
user-supplied arguments, so do not expose it to a network.
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_DIR = Path(__file__).resolve().parent

# Filters that take no convolution matrix, so the kernel editor is hidden and
# --method does not apply to them.
POINTWISE = {"grayscale", "invert", "sobel", "tonemap", "passthrough"}

# Which parameter controls each filter actually uses, so the UI can show only
# the relevant ones instead of a wall of inputs.
FILTER_PARAMS = {
    "grayscale": [],
    "invert": [],
    "box": ["radius"],
    "gaussian": ["radius", "sigma"],
    "sharpen": ["radius", "sigma", "amount"],
    "emboss": [],
    "laplacian": [],
    "sobel": [],
    "tonemap": ["exposure", "white", "gamma"],
    "custom": [],
}

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
RUN_TIMEOUT_SECONDS = 300


# --------------------------------------------------------------------------
# Pure helpers. These hold the logic worth testing, and none of them touch the
# network or the filesystem, so tests/test_gui.py can exercise them directly.
# --------------------------------------------------------------------------

def find_binary(explicit=None):
    """Locate the cuda-filters executable.

    Checks an explicit path first, then the layouts cmake and make produce on
    each platform, then PATH.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    candidates = [
        REPO_ROOT / "cuda-filters",
        REPO_ROOT / "build" / "cuda-filters",
        REPO_ROOT / "build" / "Release" / "cuda-filters.exe",
        REPO_ROOT / "build" / "Debug" / "cuda-filters.exe",
        REPO_ROOT / "cuda-filters.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c

    found = shutil.which("cuda-filters")
    return Path(found) if found else None


def validate_kernel_text(text):
    """Parse a .kernel file body the same way src/conv_kernel.cpp does.

    Returns (ok, message, info). Mirroring the C++ rules here means the UI can
    reject a bad matrix instantly instead of round-tripping to the binary for
    an error.
    """
    name = "custom"
    bias = 0.0
    normalize = False
    values = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0]
        tokens = line.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "name" and i + 1 < len(tokens):
                name = tokens[i + 1]
                i += 2
            elif tok == "bias" and i + 1 < len(tokens):
                try:
                    bias = float(tokens[i + 1])
                except ValueError:
                    return False, f"line {lineno}: bias needs a number", None
                i += 2
            elif tok == "normalize" and i + 1 < len(tokens):
                normalize = tokens[i + 1] in ("1", "true", "yes")
                i += 2
            else:
                try:
                    values.append(float(tok))
                except ValueError:
                    return False, f"line {lineno}: '{tok}' is neither a directive nor a number", None
                i += 1

    if not values:
        return False, "no weights found", None

    side = round(len(values) ** 0.5)
    if side * side != len(values):
        return False, f"{len(values)} weights do not form a square matrix", None
    if side % 2 == 0:
        return False, f"matrix is {side}x{side}; the side must be odd so there is a centre tap", None
    if (side - 1) // 2 > 32:
        return False, f"radius {(side - 1) // 2} exceeds the limit of 32", None

    total = sum(values)
    if normalize and abs(total) < 1e-6:
        return False, "normalize requested but the weights sum to zero", None

    effective = 1.0 if normalize else total
    return True, "", {
        "name": name,
        "side": side,
        "radius": (side - 1) // 2,
        "sum": round(effective, 4),
        "bias": bias,
        "normalize": normalize,
    }


def build_command(binary, req, input_path, output_path, kernel_path=None):
    """Turn a request dict from the browser into an argv list.

    Kept separate from the HTTP layer so the argument construction - the part
    most likely to be wrong - is directly testable.
    """
    filt = req.get("filter", "gaussian")
    cmd = [str(binary), "-i", str(input_path), "-o", str(output_path)]

    if filt == "custom":
        if not kernel_path:
            raise ValueError("custom filter needs a kernel file")
        cmd += ["-k", str(kernel_path)]
    else:
        cmd += ["-f", filt]

    used = FILTER_PARAMS.get(filt, [])
    if "radius" in used or filt == "custom":
        pass  # radius comes from the matrix itself for custom kernels
    if "radius" in used:
        cmd += ["-r", str(int(req.get("radius", 2)))]
    if "sigma" in used:
        cmd += ["-s", str(float(req.get("sigma", 0.0)))]
    if "amount" in used:
        cmd += ["--amount", str(float(req.get("amount", 1.0)))]
    if "exposure" in used:
        cmd += ["--exposure", str(float(req.get("exposure", 1.0)))]
    if "white" in used:
        cmd += ["--white", str(float(req.get("white", 4.0)))]
    if "gamma" in used:
        cmd += ["--gamma", str(float(req.get("gamma", 2.2)))]

    backend = req.get("backend", "cuda")
    if backend in ("cuda", "cpu"):
        cmd += ["--backend", backend]

    # --method only means anything for a convolution, and only on the GPU.
    method = req.get("method", "auto")
    if filt not in POINTWISE and backend == "cuda" and method in ("auto", "naive", "tiled", "separable"):
        cmd += ["--method", method]

    block = req.get("block", "16x16")
    if backend == "cuda" and re.fullmatch(r"\d+x\d+", str(block)):
        cmd += ["--block", str(block)]

    if req.get("channels") in (1, 3, 4, "1", "3", "4"):
        cmd += ["--channels", str(req["channels"])]

    if req.get("verify") and backend == "cuda":
        cmd += ["--verify"]

    if req.get("bench"):
        cmd += ["--bench", str(int(req["bench"]))]

    return cmd


def parse_output(stdout):
    """Pull the interesting numbers out of the binary's console output."""
    info = {"kernel": None, "timings": None, "verify": None, "loaded": None}

    m = re.search(r"^loaded\s+\S+\s+\((\d+x\d+x\d+)\)", stdout, re.M)
    if m:
        info["loaded"] = m.group(1)

    m = re.search(r"^kernel\s+(.+)$", stdout, re.M)
    if m:
        info["kernel"] = m.group(1).strip()

    m = re.search(
        r"^gpu\s+(\S+)\s+upload ([\d.]+) ms \| kernel ([\d.]+) ms \| "
        r"download ([\d.]+) ms \| total ([\d.]+) ms",
        stdout, re.M)
    if m:
        info["timings"] = {
            "method": m.group(1),
            "upload_ms": float(m.group(2)),
            "kernel_ms": float(m.group(3)),
            "download_ms": float(m.group(4)),
            "total_ms": float(m.group(5)),
        }

    m = re.search(r"^verify\s+(.+)$", stdout, re.M)
    if m:
        info["verify"] = m.group(1).strip()

    return info


def list_presets():
    """The .kernel files shipped in kernels/, with their contents."""
    out = []
    kdir = REPO_ROOT / "kernels"
    if kdir.is_dir():
        for p in sorted(kdir.glob("*.kernel")):
            try:
                out.append({"name": p.stem, "text": p.read_text()})
            except OSError:
                continue
    return out


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Session:
    """Everything one running server instance owns."""

    def __init__(self, binary):
        self.binary = binary
        self.dir = Path(tempfile.mkdtemp(prefix="cuda-filters-ui-"))
        self.input_path = None
        self.input_name = None
        self.preview = None
        self.last_output = None
        self.counter = 0
        self.lock = threading.Lock()

    def next_output(self):
        # A fresh filename each run, so the browser never shows a cached result.
        with self.lock:
            self.counter += 1
            return self.dir / f"out_{self.counter}.png"

    def run(self, cmd):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=RUN_TIMEOUT_SECONDS, cwd=str(REPO_ROOT))
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {RUN_TIMEOUT_SECONDS}s"
        except OSError as exc:
            return 127, "", f"could not run {cmd[0]}: {exc}"
        return proc.returncode, proc.stdout, proc.stderr


class Handler(BaseHTTPRequestHandler):
    session = None
    server_version = "cuda-filters-ui"

    def log_message(self, fmt, *args):
        # The default logs every request to stderr, which buries the one line
        # the user actually needs (the URL).
        pass

    # -- helpers ----------------------------------------------------------

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return None
        return json.loads(self.rfile.read(length).decode())

    def _serve_file(self, path):
        path = Path(path)
        # Only ever serve out of the session dir or the repo's assets.
        allowed = (self.session.dir.resolve(), (REPO_ROOT / "assets").resolve())
        try:
            resolved = path.resolve()
        except OSError:
            return self._send(404, {"error": "not found"})
        if not any(str(resolved).startswith(str(a)) for a in allowed):
            return self._send(403, {"error": "forbidden"})
        if not resolved.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self._send(200, resolved.read_bytes(), ctype)

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/":
            index = GUI_DIR / "index.html"
            if not index.is_file():
                return self._send(500, "gui/index.html is missing", "text/plain")
            return self._send(200, index.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/state":
            return self._send(200, {
                "binary": str(self.session.binary) if self.session.binary else None,
                "filters": list(FILTER_PARAMS.keys()),
                "filter_params": FILTER_PARAMS,
                "pointwise": sorted(POINTWISE),
                "presets": list_presets(),
                "has_input": self.session.input_path is not None,
                # So a page reload restores what you were looking at.
                "preview": f"/file/{self.session.preview.name}" if self.session.preview else None,
                "input_name": self.session.input_name,
                "last_output": self.session.last_output,
                "sample": str((REPO_ROOT / "assets" / "sample.png").resolve())
                          if (REPO_ROOT / "assets" / "sample.png").is_file() else None,
            })

        if path == "/api/devices":
            if not self.session.binary:
                return self._send(200, {"ok": False, "output": "binary not found"})
            code, out, err = self.session.run([str(self.session.binary), "--list-devices"])
            return self._send(200, {"ok": code == 0, "output": (out + err).strip()})

        if path.startswith("/file/"):
            return self._serve_file(self.session.dir / path[len("/file/"):])

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/upload":
            return self._upload()
        if path == "/api/use-sample":
            return self._use_sample()
        if path == "/api/validate-kernel":
            body = self._read_json() or {}
            ok, msg, info = validate_kernel_text(body.get("text", ""))
            return self._send(200, {"ok": ok, "message": msg, "info": info})
        if path == "/api/apply":
            return self._apply()
        if path == "/api/save-kernel":
            return self._save_kernel()

        return self._send(404, {"error": "not found"})

    # -- route bodies -----------------------------------------------------

    def _install_input(self, data, filename):
        suffix = Path(filename).suffix.lower() or ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".ppm", ".pgm", ".pnm", ".psd"):
            return None, f"unsupported file type '{suffix}'"

        src = self.session.dir / f"input{suffix}"
        src.write_bytes(data)
        self.session.input_path = src

        # Browsers cannot display BMP/TGA/PSD reliably, so make a PNG preview
        # using the project's own passthrough filter. It doubles as a check
        # that the binary runs at all.
        preview = self.session.dir / "preview.png"
        if self.session.binary:
            code, out, err = self.session.run([
                str(self.session.binary), "-i", str(src), "-o", str(preview),
                "-f", "passthrough", "--backend", "cpu", "-q"])
            if code != 0:
                return None, (err or out or "could not decode image").strip()
        else:
            shutil.copyfile(src, preview)

        self.session.preview = preview
        self.session.input_name = filename
        self.session.last_output = None
        return preview, None

    def _upload(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._send(400, {"ok": False, "error": "empty upload"})
        if length > MAX_UPLOAD_BYTES:
            return self._send(413, {"ok": False, "error": "image larger than 64 MB"})

        filename = self.headers.get("X-Filename", "upload.png")
        data = self.rfile.read(length)
        preview, err = self._install_input(data, filename)
        if err:
            return self._send(200, {"ok": False, "error": err})
        return self._send(200, {"ok": True, "preview": f"/file/{preview.name}",
                                "name": filename})

    def _use_sample(self):
        sample = REPO_ROOT / "assets" / "sample.png"
        if not sample.is_file():
            return self._send(200, {"ok": False,
                                    "error": "assets/sample.png is missing - run 'make sample'"})
        preview, err = self._install_input(sample.read_bytes(), "sample.png")
        if err:
            return self._send(200, {"ok": False, "error": err})
        return self._send(200, {"ok": True, "preview": f"/file/{preview.name}",
                                "name": "sample.png"})

    def _save_kernel(self):
        body = self._read_json() or {}
        name = re.sub(r"[^A-Za-z0-9_-]", "", body.get("name", "")) or "untitled"
        ok, msg, _ = validate_kernel_text(body.get("text", ""))
        if not ok:
            return self._send(200, {"ok": False, "error": msg})
        target = REPO_ROOT / "kernels" / f"{name}.kernel"
        try:
            target.write_text(body.get("text", ""))
        except OSError as exc:
            return self._send(200, {"ok": False, "error": str(exc)})
        return self._send(200, {"ok": True, "path": str(target)})

    def _apply(self):
        req = self._read_json()
        if req is None:
            return self._send(400, {"ok": False, "error": "bad request"})
        if not self.session.binary:
            return self._send(200, {"ok": False, "error":
                                    "cuda-filters binary not found - build it first"})
        if not self.session.input_path:
            return self._send(200, {"ok": False, "error": "no image loaded"})

        kernel_path = None
        if req.get("filter") == "custom":
            ok, msg, _ = validate_kernel_text(req.get("kernel_text", ""))
            if not ok:
                return self._send(200, {"ok": False, "error": f"kernel: {msg}"})
            kernel_path = self.session.dir / "custom.kernel"
            kernel_path.write_text(req.get("kernel_text", ""))

        out_path = self.session.next_output()
        try:
            cmd = build_command(self.session.binary, req, self.session.input_path,
                                out_path, kernel_path)
        except (ValueError, TypeError) as exc:
            return self._send(200, {"ok": False, "error": str(exc)})

        code, stdout, stderr = self.session.run(cmd)
        payload = {
            "ok": code == 0,
            "exit_code": code,
            "command": " ".join(cmd),
            "stdout": stdout,
            "stderr": stderr,
            "info": parse_output(stdout),
        }
        # --bench writes no image, so only advertise a result when one exists.
        if code == 0 and out_path.is_file():
            payload["output"] = f"/file/{out_path.name}"
            self.session.last_output = payload["output"]
        if code != 0 and not stderr:
            payload["error"] = f"exit code {code}"
        elif code != 0:
            payload["error"] = stderr.strip()
        return self._send(200, payload)


def main():
    ap = argparse.ArgumentParser(description="Local web UI for cuda-image-filters")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--binary", help="path to the cuda-filters executable")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    binary = find_binary(args.binary)
    Handler.session = Session(binary)

    url = f"http://127.0.0.1:{args.port}/"
    # flush=True so the banner appears immediately even when stdout is a pipe
    # (make ui, tee, an IDE terminal) rather than sitting in a buffer.
    say = lambda line: print(line, flush=True)
    say(f"cuda-image-filters UI  ->  {url}")
    if binary:
        say(f"binary                 :  {binary}")
    else:
        say("binary                 :  NOT FOUND")
        say("                          build it first (cmake --build build, or make),")
        say("                          or pass --binary <path>")
    say(f"scratch dir            :  {Handler.session.dir}")
    say("ctrl-c to stop")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
        shutil.rmtree(Handler.session.dir, ignore_errors=True)


if __name__ == "__main__":
    main()
