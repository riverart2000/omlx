#!/usr/bin/env python3
"""omlx-image — local HiDream-O1 text-to-image microservice (port 8400).

Mirrors the pattern of the TTS Studio (:8200) and memory (:8300) services:
a tiny stdlib HTTP server with an async job API and a single serialized
worker (image generation is compute-bound, so we run one at a time).

Endpoints:
  GET  /health             -> service + model status
  GET  /info               -> defaults, presets, sizes
  POST /generate           -> {prompt,width,height,steps,seed,...} => {job_id}
  GET  /status?id=JOB      -> job progress / result
  GET  /files/<name>.png   -> the rendered image
  POST /load               -> warm-load the model now (optional)
"""
import json
import os
import base64
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import hidream_engine as eng

HOST = os.environ.get("IMAGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("IMAGE_PORT", "8400"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TMP_DIR = os.path.join(OUT_DIR, "_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

MFLUX_KONTEXT_BIN = os.environ.get(
    "MFLUX_KONTEXT_BIN",
    os.path.join(BASE_DIR, ".venv-mflux", "bin", "mflux-generate-kontext"),
)
KONTEXT_MODEL = os.environ.get(
    "KONTEXT_MODEL",
    "akx/FLUX.1-Kontext-dev-mflux-4bit",
)
KONTEXT_BASE_MODEL = os.environ.get("KONTEXT_BASE_MODEL", "dev")

MAX_DIM = 2048
MIN_DIM = 256
PRESETS = {
    "1:1":  (1024, 1024),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "4:5":  (1024, 1280),
    "3:2":  (1216, 832),
}

_jobs = {}
_jobs_lock = threading.Lock()
_work_q = []
_work_cv = threading.Condition()


def _decode_data_url_png(raw: str) -> bytes:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty reference image payload")
    if "," in s and "base64" in s[:80].lower():
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64 reference image: {e}") from e


def _run_kontext(jid, opts):
    from PIL import Image

    if not os.path.isfile(MFLUX_KONTEXT_BIN):
        raise RuntimeError(f"missing kontext binary: {MFLUX_KONTEXT_BIN}")

    ref_bytes = _decode_data_url_png(opts.get("reference_image", ""))
    ref_name = f"{jid}_ref.png"
    out_name = f"img_{uuid.uuid4().hex[:12]}.png"
    ref_path = os.path.join(TMP_DIR, ref_name)
    out_path = os.path.join(OUT_DIR, out_name)
    with open(ref_path, "wb") as f:
        f.write(ref_bytes)

    t0 = time.time()
    cmd = [
        MFLUX_KONTEXT_BIN,
        "--model", KONTEXT_MODEL,
        "--base-model", KONTEXT_BASE_MODEL,
        "--image-path", ref_path,
        "--prompt", opts["prompt"],
        "--steps", str(opts["steps"]),
        "--guidance", str(opts["guidance"]),
        "--seed", str(opts["seed"]),
        "--height", str(opts["height"]),
        "--width", str(opts["width"]),
        "--output", out_path,
    ]
    proc = subprocess.run(
        cmd, cwd=BASE_DIR, capture_output=True, text=True, check=False
    )
    try:
        os.remove(ref_path)
    except OSError:
        pass
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError("kontext failed: " + " | ".join(tail))
    if not os.path.isfile(out_path):
        raise RuntimeError("kontext failed: output image not found")
    img = Image.open(out_path)
    w, h = img.size
    elapsed = round(time.time() - t0, 1)
    return out_name, int(w), int(h), elapsed


def _set(jid, **kw):
    with _jobs_lock:
        j = _jobs.setdefault(jid, {})
        j.update(kw)


def _get(jid):
    with _jobs_lock:
        return dict(_jobs.get(jid, {})) or None


def _sys_stats():
    """Best-effort Apple-Silicon GPU + unified-memory stats (no sudo needed)."""
    out = {"gpu_util": None, "gpu_mem_gb": None,
           "mem_used_gb": None, "mem_total_gb": None}
    try:
        io = subprocess.check_output(
            ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
            text=True, timeout=2, stderr=subprocess.DEVNULL)
        m = re.search(r'"Device Utilization %"=(\d+)', io)
        if m:
            out["gpu_util"] = int(m.group(1))
        m = re.search(r'"In use system memory"=(\d+)', io)
        if m:
            out["gpu_mem_gb"] = round(int(m.group(1)) / 1e9, 1)
    except Exception:
        pass
    try:
        total = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip())
        out["mem_total_gb"] = round(total / 1e9, 1)
    except Exception:
        pass
    try:
        vm = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        page = 4096
        pm = re.search(r'page size of (\d+) bytes', vm)
        if pm:
            page = int(pm.group(1))

        def _pg(name):
            mm = re.search(name + r':\s+(\d+)\.', vm)
            return int(mm.group(1)) if mm else 0
        used_pages = (_pg("Pages active") + _pg("Pages wired down")
                      + _pg("Pages occupied by compressor"))
        out["mem_used_gb"] = round(used_pages * page / 1e9, 1)
    except Exception:
        pass
    return out


def _worker():
    while True:
        with _work_cv:
            while not _work_q:
                _work_cv.wait()
            jid = _work_q.pop(0)
        _run_job(jid)


def _run_job(jid):
    job = _get(jid)
    if not job:
        return
    opts = job["opts"]
    try:
        if opts["engine"] == "kontext":
            _set(jid, stage="kontext generating", progress=8)
            name, w, h, elapsed = _run_kontext(jid, opts)
            _set(jid, stage="saving", progress=97)
        else:
            if not eng.weights_ready():
                raise RuntimeError("model weights are still downloading — "
                                   "try again once the download completes")
            if not eng.is_loaded():
                _set(jid, stage="loading model (first run, ~30s)", progress=2)
            _set(jid, stage="generating", progress=5)

            steps = opts["steps"]

            def prog(done, total):
                _set(jid, stage=f"denoising {done}/{total}",
                     progress=int(5 + 90 * done / max(1, total)))

            t0 = time.time()
            arr, w, h = eng.generate(
                prompt=opts["prompt"], width=opts["width"], height=opts["height"],
                steps=steps, seed=opts["seed"], snap=opts["snap"],
                blend_seams=opts["blend_seams"], progress=prog)
            from PIL import Image
            _set(jid, stage="saving", progress=97)
            name = "img_" + uuid.uuid4().hex[:12] + ".png"
            path = os.path.join(OUT_DIR, name)
            Image.fromarray(arr).save(path)
            elapsed = round(time.time() - t0, 1)

        result = {
            "filename": name,
            "url": f"/files/{name}",
            "width": int(w), "height": int(h),
            "steps": opts["steps"], "seed": opts["seed"],
            "prompt": opts["prompt"],
            "engine": opts["engine"],
            "seconds": elapsed,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        _set(jid, status="done", stage="done", progress=100, result=result)
    except Exception as e:
        _set(jid, status="error", stage="error",
             error=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = "omlx-image/1.0"

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            kontext_ready = os.path.isfile(MFLUX_KONTEXT_BIN)
            return self._json(200, {
                "ok": True,
                "scripts_available": eng.scripts_available(),
                "weights_ready": eng.weights_ready(),
                "model_loaded": eng.is_loaded(),
                "kontext_ready": kontext_ready,
                "kontext_model": KONTEXT_MODEL,
                "queue": len(_work_q),
            })
        if path == "/info":
            return self._json(200, {
                "model": "HiDream-O1-Image-Dev (MLX bf16)",
                "kontext_model": KONTEXT_MODEL,
                "engines": ["hidream", "kontext"],
                "presets": PRESETS, "default_steps": 28,
                "min_dim": MIN_DIM, "max_dim": MAX_DIM,
                "weights_ready": eng.weights_ready(),
                "model_loaded": eng.is_loaded(),
            })
        if path == "/status":
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            job = _get(jid)
            if not job:
                return self._json(404, {"error": "unknown job"})
            return self._json(200, {k: job.get(k) for k in
                                    ("status", "stage", "progress",
                                     "result", "error")})
        if path == "/stats":
            return self._json(200, _sys_stats())
        if path.startswith("/files/"):
            return self._serve_file(os.path.basename(path))
        return self._json(404, {"error": "not found"})

    def _serve_file(self, name):
        fp = os.path.join(OUT_DIR, name)
        if not (name.endswith(".png") and os.path.isfile(fp)):
            return self._json(404, {"error": "not found"})
        data = open(fp, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/load":
            try:
                if not eng.weights_ready():
                    return self._json(409, {"error": "weights not ready"})
                threading.Thread(target=eng.load, daemon=True).start()
                return self._json(200, {"ok": True, "loading": True})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/generate":
            d = self._body()
            prompt = (d.get("prompt") or "").strip()
            if not prompt:
                return self._json(400, {"error": "prompt is required"})
            preset = d.get("preset")
            if preset in PRESETS:
                w, h = PRESETS[preset]
            else:
                w = int(d.get("width", 1024))
                h = int(d.get("height", 1024))
            w = max(MIN_DIM, min(MAX_DIM, w))
            h = max(MIN_DIM, min(MAX_DIM, h))
            engine = str(d.get("engine", "hidream")).strip().lower()
            if engine not in ("hidream", "kontext"):
                return self._json(400, {"error": "engine must be hidream or kontext"})
            opts = {
                "engine": engine,
                "prompt": prompt[:2000],
                "width": w, "height": h,
                "steps": max(4, min(50, int(d.get("steps", 28)))),
                "seed": int(d.get("seed", 32)),
                "snap": bool(d.get("snap", False)),
                "blend_seams": max(0, min(4, int(d.get("blend_seams", 0)))),
                "guidance": float(d.get("guidance", 2.8)),
                "reference_image": d.get("reference_image", ""),
            }
            if engine == "kontext" and not opts["reference_image"]:
                return self._json(400, {"error": "reference_image is required for kontext"})
            jid = "img_" + uuid.uuid4().hex[:12]
            _set(jid, status="running", stage="queued", progress=0,
                 result=None, error=None, opts=opts)
            with _work_cv:
                _work_q.append(jid)
                _work_cv.notify()
            return self._json(200, {"ok": True, "job_id": jid})
        return self._json(404, {"error": "not found"})


def main():
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-image ready at http://{HOST}:{PORT} "
          f"(weights_ready={eng.weights_ready()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
