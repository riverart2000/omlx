#!/usr/bin/env python3
"""omlx-video — local Wan2.2-TI2V-5B text+image-to-video microservice (port 8500).

Mirrors the omlx-image (:8400) pattern: a tiny stdlib HTTP server with an async
job API and a single serialized worker (video generation is very compute-bound,
so we run exactly one at a time). Generation is delegated to mlx-video's
`mlx_video.models.wan_2.generate` CLI running in this service's own venv.

Quality-first defaults (per user): 720p 1280x704, 24fps, unipc scheduler, 40
steps. Frame count must be 4n+1; duration_seconds is converted to the nearest
valid frame count at 24fps.

Endpoints:
  GET  /health             -> service + weights status
  GET  /info               -> defaults, presets, limits
  POST /generate           -> {prompt, image?, width, height, duration_seconds,
                              steps, guide_scale, seed, ...} => {job_id}
  GET  /status?id=JOB      -> job progress / result
  GET  /files/<name>.mp4   -> the rendered clip
"""
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOST = os.environ.get("VIDEO_HOST", "127.0.0.1")
PORT = int(os.environ.get("VIDEO_PORT", "8500"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

MODEL_DIR = os.environ.get(
    "WAN_MODEL_DIR",
    "/Users/joebains/.omlx/models/Anes1032/Wan2.2-TI2V-5B-mlx-q8",
)
VENV_PY = os.environ.get("VIDEO_PY", os.path.join(BASE_DIR, ".venv", "bin", "python"))

FPS = 24  # Wan2.2 TI2V output is fixed 24fps.
# Quality-first defaults. Target 512p (short side 512), all quality knobs maxed.
DEF_WIDTH = 896
DEF_HEIGHT = 512
DEF_STEPS = 60           # max-quality by default (server clamps to [10,60])
DEF_GUIDE = 5.0
DEF_SCHEDULER = "unipc"  # official / highest-quality 2nd-order solver
DEF_TILING = "none"      # no VAE tiling => no seams => highest fidelity (64GB M5)
DEF_TRIM_FIRST = 0       # extra discarded temporal chunks (first-frame artifact fix)
DEF_SECONDS = 3.5  # ~81 frames
MAX_SECONDS = 8.0  # keep render time + memory sane on 64GB M5
MIN_SECONDS = 1.0
MAX_DIM = 1280
MIN_DIM = 256

TILING_MODES = ("auto", "none", "default", "aggressive", "conservative",
                "spatial", "temporal")

# Aspect presets per resolution tier (width, height) — all divisible by 32.
# 512p (short side ~512) is the default; 720p available for higher quality.
RES_PRESETS = {
    "512p": {
        "16:9": (896, 512),
        "9:16": (512, 896),
        "1:1":  (512, 512),
        "4:5":  (512, 640),
        "3:2":  (768, 512),
    },
    "720p": {
        "16:9": (1280, 704),
        "9:16": (704, 1280),
        "1:1":  (960, 960),
        "4:5":  (832, 1040),
        "3:2":  (1152, 768),
    },
}
DEF_RES = "512p"
# Backward-compatible flat preset map (defaults to the 512p tier).
PRESETS = RES_PRESETS[DEF_RES]

_jobs = {}
_jobs_lock = threading.Lock()
_work_q = []
_work_cv = threading.Condition()


def _weights_ready() -> bool:
    need = ["model.safetensors", "t5_encoder.safetensors", "vae.safetensors",
            "config.json"]
    return all(os.path.isfile(os.path.join(MODEL_DIR, n)) for n in need)


def _scripts_available() -> bool:
    return os.path.isfile(VENV_PY)


def _frames_for_seconds(sec: float) -> int:
    """Wan2.2 requires num_frames == 4n+1. Convert seconds@24fps to nearest valid."""
    sec = max(MIN_SECONDS, min(MAX_SECONDS, float(sec)))
    raw = int(round(sec * FPS))
    n = max(1, round((raw - 1) / 4))
    return int(4 * n + 1)


def _even32(v: int) -> int:
    v = max(MIN_DIM, min(MAX_DIM, int(v)))
    return (v // 32) * 32


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


def _set(jid, **kw):
    with _jobs_lock:
        j = _jobs.setdefault(jid, {})
        j.update(kw)


def _get(jid):
    with _jobs_lock:
        return dict(_jobs.get(jid, {})) or None


def _worker():
    while True:
        with _work_cv:
            while not _work_q:
                _work_cv.wait()
            jid = _work_q.pop(0)
        _run_job(jid)


# Progress lines from mlx-video look like "Step 12/40" or a tqdm bar; we scan
# stderr/stdout for "x/y" to estimate progress.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _run_generate(jid, opts):
    out_name = "vid_" + uuid.uuid4().hex[:12] + ".mp4"
    out_path = os.path.join(OUT_DIR, out_name)
    cmd = [
        VENV_PY, "-m", "mlx_video.models.wan_2.generate",
        "--model-dir", MODEL_DIR,
        "--prompt", opts["prompt"],
        "--width", str(opts["width"]),
        "--height", str(opts["height"]),
        "--num-frames", str(opts["num_frames"]),
        "--steps", str(opts["steps"]),
        "--guide-scale", str(opts["guide_scale"]),
        "--scheduler", opts["scheduler"],
        "--seed", str(opts["seed"]),
        "--output-path", out_path,
    ]
    if opts.get("tiling"):
        cmd += ["--tiling", opts["tiling"]]
    if opts.get("shift") is not None:
        cmd += ["--shift", str(opts["shift"])]
    if int(opts.get("trim_first_frames") or 0) > 0:
        cmd += ["--trim-first-frames", str(int(opts["trim_first_frames"]))]
    ref_path = None
    if opts.get("image"):
        ref_bytes = _decode_data_url_png(opts["image"])
        ref_path = os.path.join(TMP_DIR, f"{jid}_ref.png")
        with open(ref_path, "wb") as f:
            f.write(ref_bytes)
        cmd += ["--image", ref_path]
    if opts.get("negative_prompt"):
        cmd += ["--negative-prompt", opts["negative_prompt"]]

    total = opts["steps"]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 40:
            tail.pop(0)
        m = _STEP_RE.search(line)
        if m:
            done, tot = int(m.group(1)), int(m.group(2))
            if tot in (total, total + 1) or tot == total:
                _set(jid, stage=f"diffusing {done}/{tot}",
                     progress=int(5 + 88 * done / max(1, tot)))
    proc.wait()
    if ref_path:
        try:
            os.remove(ref_path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError("wan2.2 generate failed: "
                           + " | ".join(tail[-8:]))
    if not os.path.isfile(out_path):
        raise RuntimeError("wan2.2 generate failed: no output produced — "
                           + " | ".join(tail[-6:]))
    elapsed = round(time.time() - t0, 1)
    return out_name, elapsed


def _run_job(jid):
    job = _get(jid)
    if not job:
        return
    opts = job["opts"]
    try:
        if not _weights_ready():
            raise RuntimeError("Wan2.2 weights are still downloading — try "
                               "again once model.safetensors finishes")
        _set(jid, stage="loading model + encoders (first run is slow)", progress=3)
        name, elapsed = _run_generate(jid, opts)
        _set(jid, stage="saving", progress=96)
        result = {
            "filename": name,
            "url": f"/files/{name}",
            "width": opts["width"], "height": opts["height"],
            "num_frames": opts["num_frames"], "fps": FPS,
            "duration": round(opts["num_frames"] / FPS, 2),
            "steps": opts["steps"], "seed": opts["seed"],
            "mode": "i2v" if opts.get("image") else "t2v",
            "prompt": opts["prompt"],
            "seconds": elapsed,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        _set(jid, status="done", stage="done", progress=100, result=result)
    except Exception as e:
        _set(jid, status="error", stage="error",
             error=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = "omlx-video/1.0"

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

    def _read_body(self):
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
            return self._json(200, {
                "ok": True,
                "scripts_available": _scripts_available(),
                "weights_ready": _weights_ready(),
                "model_dir": MODEL_DIR,
                "queue": len(_work_q),
            })
        if path == "/info":
            return self._json(200, {
                "model": "Wan2.2-TI2V-5B (MLX q8)",
                "modes": ["t2v", "i2v"],
                "fps": FPS,
                "presets": PRESETS,
                "res_presets": RES_PRESETS,
                "res_default": DEF_RES,
                "tiling_modes": list(TILING_MODES),
                "defaults": {
                    "res": DEF_RES,
                    "width": DEF_WIDTH, "height": DEF_HEIGHT,
                    "steps": DEF_STEPS, "guide_scale": DEF_GUIDE,
                    "scheduler": DEF_SCHEDULER, "tiling": DEF_TILING,
                    "trim_first_frames": DEF_TRIM_FIRST,
                    "duration_seconds": DEF_SECONDS,
                },
                "limits": {"min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS,
                           "min_dim": MIN_DIM, "max_dim": MAX_DIM,
                           "min_steps": 10, "max_steps": 60},
                "weights_ready": _weights_ready(),
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
        if path.startswith("/files/"):
            return self._serve_file(os.path.basename(path))
        return self._json(404, {"error": "not found"})

    def _serve_file(self, name):
        fp = os.path.join(OUT_DIR, name)
        if not (name.endswith(".mp4") and os.path.isfile(fp)):
            return self._json(404, {"error": "not found"})
        data = open(fp, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/generate":
            d = self._read_body()
            prompt = (d.get("prompt") or "").strip()
            if not prompt:
                return self._json(400, {"error": "prompt is required"})
            res = str(d.get("res") or DEF_RES)
            if res not in RES_PRESETS:
                res = DEF_RES
            aspect = d.get("aspect") or d.get("preset")
            if aspect in RES_PRESETS[res]:
                w, h = RES_PRESETS[res][aspect]
            elif aspect in PRESETS:
                w, h = PRESETS[aspect]
            else:
                w = _even32(d.get("width", DEF_WIDTH))
                h = _even32(d.get("height", DEF_HEIGHT))
            if "duration_seconds" in d or "num_frames" not in d:
                num_frames = _frames_for_seconds(
                    d.get("duration_seconds", DEF_SECONDS))
            else:
                nf = int(d.get("num_frames", 81))
                num_frames = nf if (nf - 1) % 4 == 0 else _frames_for_seconds(nf / FPS)
            tiling = str(d.get("tiling", DEF_TILING))
            if tiling not in TILING_MODES:
                tiling = DEF_TILING
            shift = d.get("shift", None)
            try:
                shift = float(shift) if shift not in (None, "") else None
            except (TypeError, ValueError):
                shift = None
            opts = {
                "prompt": prompt[:2000],
                "image": d.get("image", ""),
                "negative_prompt": (d.get("negative_prompt") or "").strip(),
                "width": w, "height": h,
                "num_frames": num_frames,
                "steps": max(10, min(60, int(d.get("steps", DEF_STEPS)))),
                "guide_scale": float(d.get("guide_scale", DEF_GUIDE)),
                "scheduler": str(d.get("scheduler", DEF_SCHEDULER)),
                "tiling": tiling,
                "shift": shift,
                "trim_first_frames": max(0, min(4, int(d.get("trim_first_frames", DEF_TRIM_FIRST)))),
                "seed": int(d.get("seed", 42)),
            }
            if opts["scheduler"] not in ("euler", "dpm++", "unipc"):
                opts["scheduler"] = DEF_SCHEDULER
            jid = "vid_" + uuid.uuid4().hex[:12]
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
    print(f"omlx-video ready at http://{HOST}:{PORT} "
          f"(weights_ready={_weights_ready()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
