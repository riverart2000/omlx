#!/usr/bin/env python3
"""omlx-video — local Wan2.2-I2V-A14B image-to-video microservice (port 8500).

Mirrors the omlx-image (:8400) pattern: a tiny stdlib HTTP server with an async
job API and a single serialized worker (video generation is very compute-bound,
so we run exactly one at a time). Generation is delegated to mlx-video's
`mlx_video.models.wan_2.generate` CLI running in this service's own venv.

Model: Wan2.2-I2V-A14B (MLX q8) — a much higher-quality 14B dual-model than the
old TI2V-5B. It is IMAGE-TO-VIDEO only, so every request MUST include a start
image (the UI's "reference / first frame"). Native output is 16fps.

Quality-first defaults: 512p (short side 512, user-selectable up to 720p), 16fps,
unipc scheduler, 60 steps. Frame count must be 4n+1; duration_seconds is
converted to the nearest valid frame count at 16fps.

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
import shutil
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
SAVE_DIR = os.environ.get("VIDEO_SAVE_DIR", "/Users/joebains/Movies")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

MODEL_DIR = os.environ.get(
    "WAN_MODEL_DIR",
    "/Users/joebains/.omlx/models/Anes1032/Wan2.2-I2V-A14B-mlx-q8",
)
VENV_PY = os.environ.get("VIDEO_PY", os.path.join(BASE_DIR, ".venv", "bin", "python"))

# Wan2.2-I2V-A14B (dual-model, image-to-video only). Weight files differ from the
# old single-model 5B (high_noise_model + low_noise_model instead of model).
MODEL_LABEL = "Wan2.2-I2V-A14B (MLX q8)"
MODEL_WEIGHT_FILES = ["high_noise_model.safetensors", "low_noise_model.safetensors",
                      "t5_encoder.safetensors", "vae.safetensors", "config.json"]
I2V_ONLY = True  # this model has no pure text-to-video path; a start image is required
DUAL_MODEL = True  # A14B is a high/low-noise dual model; guide_scale must be a pair

FPS = 16  # Wan2.2-I2V-A14B native output is 16fps.
# Quality-first defaults. Target 512p (short side 512), all quality knobs maxed.
DEF_WIDTH = 896
DEF_HEIGHT = 512
DEF_STEPS = 60           # max-quality by default (server clamps to [10,60])
DEF_GUIDE = 3.5          # A14B dual-model config default (sample_guide_scale=[3.5,3.5])
DEF_SCHEDULER = "unipc"  # official / highest-quality 2nd-order solver
DEF_TILING = "auto"      # SAFE default: bounds VAE-decode peak memory so a
                         # long/high-res clip can't exhaust unified memory and
                         # hang the machine. "none" (seam-free max fidelity) is
                         # still selectable for short/low-res clips.
DEF_TRIM_FIRST = 0       # extra discarded temporal chunks (first-frame artifact fix)
DEF_SECONDS = 3.5  # ~81 frames
MAX_SECONDS = 8.0  # keep render time + memory sane on 64GB M5
MIN_SECONDS = 1.0
MAX_DIM = 1280
MIN_DIM = 256

# VAE decode with tiling="none" materializes all frames at once. Empirically a
# 512x896x49 clip (~22.5M px*frames) decodes near ~30GB and is the crash edge on
# a 64GB machine that's also running the LLM/app. Cap tiling="none" to clips
# below this pixel*frame budget; heavier clips are forced to safe tiling.
NONE_TILING_PX_BUDGET = 512 * 512 * 33  # ~8.6M px*frames (short/low-res only)

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

# ---------------------------------------------------------------------------
# Second engine: LongCat-Video (text-to-video, long clips). A completely
# separate 13.6B diffusers-style stack living in its own repo + venv. Selected
# per-request via `engine: "longcat"`; Wan2.2 (`engine: "wan"`, the default)
# stays the image-to-video path. LongCat generates 15fps long video by chaining
# 93-frame segments (each conditioned on the previous segment's last 13 frames),
# using the cfg_step_lora fast path (8 steps, collapsed CFG) — the only path
# fast enough to be usable.
# ---------------------------------------------------------------------------
LC_REPO_DIR = os.environ.get("LONGCAT_REPO_DIR", "/Users/joebains/longcat-video-mlx")
LC_VENV_PY = os.environ.get(
    "LONGCAT_PY", os.path.join(LC_REPO_DIR, ".venv", "bin", "python"))
# Parent dir that contains LongCat-Video-<variant>/ (VARIANT_DIRNAMES resolves it).
LC_WEIGHTS_DIR = os.environ.get(
    "LONGCAT_WEIGHTS_DIR", "/Users/joebains/.omlx/models/mlx-community")
LC_VARIANT = os.environ.get("LONGCAT_VARIANT", "q8")
LC_MODEL_LABEL = f"LongCat-Video ({LC_VARIANT}) — text-to-video, long clips"

LC_FPS = 15                    # LongCat native output is 15fps.
LC_SEG_FRAMES = 93             # frames generated per segment
LC_COND_FRAMES = 13            # frames of overlap conditioning between segments
LC_SEG_STRIDE = LC_SEG_FRAMES - LC_COND_FRAMES   # 80 net-new frames per extra segment
LC_STEPS = 8                   # cfg_step_lora fast path
LC_MIN_SECONDS = 5.0
LC_MAX_SECONDS = 30.0          # user target ceiling; caps runaway multi-hour renders
LC_DEF_SECONDS = 15.0
LC_MAX_SEGMENTS = 6            # 6 segments ≈ 32.9s ≈ 30s preset

# 480p is LongCat's default / sweet-spot resolution. Dims must be /16 (VAE).
LC_RES_PRESETS = {
    "480p": {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1":  (512, 512),
        "4:5":  (512, 640),
        "3:2":  (720, 480),
    },
}
LC_DEF_RES = "480p"
LC_DEF_ASPECT = "9:16"
# Named length presets for the UI, in 5-second blocks (seconds -> segments:
# 5→1, 10→2, 15→3, 20→4, 25→5, 30→6 via _lc_segments_for_seconds).
LC_LENGTH_PRESETS = [
    {"label": "5s",  "seconds": 5},
    {"label": "10s", "seconds": 10},
    {"label": "15s", "seconds": 15},
    {"label": "20s", "seconds": 20},
    {"label": "25s", "seconds": 25},
    {"label": "30s", "seconds": 30},
]

# --- Memory safety (LongCat) ----------------------------------------------
# A LongCat 480p q8 run needs a large chunk of unified memory (DiT ~15GB +
# UMT5 encoder + VAE-decode peak). Running it ON TOP of the already-loaded LLM
# has hard-crashed the whole Mac (kernel panic -> reboot). Two guards:
#  1. Pre-flight: refuse to launch unless at least LC_REQUIRED_FREE_GB is
#     genuinely free right now.
#  2. In-child MLX cap: bound this run's MLX memory so an overshoot raises a
#     catchable error instead of exhausting RAM (see the venv sitecustomize.py).
LC_REQUIRED_FREE_GB = float(os.environ.get("LONGCAT_REQUIRED_FREE_GB", "34"))
# Leave this much of the measured-free memory as OS/other-process headroom;
# the child's MLX hard limit = (available_at_launch - this).
LC_MLX_HEADROOM_GB = float(os.environ.get("LONGCAT_MLX_HEADROOM_GB", "8"))
LC_MLX_CACHE_GB = float(os.environ.get("LONGCAT_MLX_CACHE_GB", "1"))

_jobs = {}
_jobs_lock = threading.Lock()
_work_q = []
_work_cv = threading.Condition()


def _weights_ready() -> bool:
    return all(os.path.isfile(os.path.join(MODEL_DIR, n))
               for n in MODEL_WEIGHT_FILES)


def _scripts_available() -> bool:
    return os.path.isfile(VENV_PY)


def _lc_weights_ready() -> bool:
    """LongCat weights present: <LC_WEIGHTS_DIR>/LongCat-Video-<variant>/dit exists."""
    d = os.path.join(LC_WEIGHTS_DIR, f"LongCat-Video-{LC_VARIANT}")
    return os.path.isdir(os.path.join(d, "dit"))


def _lc_available() -> bool:
    return os.path.isfile(LC_VENV_PY) and _lc_weights_ready()


def _avail_mem_gb() -> float:
    """Best-effort 'genuinely reclaimable' unified memory, in GB.

    free + inactive + speculative + purgeable pages — memory the kernel can
    hand out without swapping. Used as the LongCat launch guardrail. Returns a
    large number on failure so we never wrongly block (the in-child MLX cap is
    the backstop)."""
    try:
        vm = subprocess.check_output(["vm_stat"], text=True, timeout=3)
    except Exception:
        return 1e9
    page = 4096
    pm = re.search(r"page size of (\d+) bytes", vm)
    if pm:
        page = int(pm.group(1))

    def _pg(name):
        mm = re.search(name + r":\s+(\d+)\.", vm)
        return int(mm.group(1)) if mm else 0

    pages = (_pg("Pages free") + _pg("Pages inactive")
             + _pg("Pages speculative") + _pg("Pages purgeable"))
    return round(pages * page / 1e9, 1)


def _lc_segments_for_seconds(sec: float) -> int:
    """Chained-segment count for a target duration at LC_FPS.

    frames(s) = 93 + (s-1)*80. Solve for the fewest segments that reach the
    requested seconds (round up so we never undershoot), clamped to
    [1, LC_MAX_SEGMENTS]."""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        sec = LC_DEF_SECONDS
    sec = max(LC_MIN_SECONDS, min(LC_MAX_SECONDS, sec))
    target = sec * LC_FPS
    if target <= LC_SEG_FRAMES:
        return 1
    import math
    segs = int(math.ceil((target - LC_SEG_FRAMES) / LC_SEG_STRIDE)) + 1
    return max(1, min(LC_MAX_SEGMENTS, segs))


def _lc_frames(segments: int) -> int:
    return LC_SEG_FRAMES + max(0, segments - 1) * LC_SEG_STRIDE


def _even16(v: int) -> int:
    v = max(MIN_DIM, min(MAX_DIM, int(v)))
    return (v // 16) * 16


def _frames_for_seconds(sec: float) -> int:
    """Wan2.2 requires num_frames == 4n+1. Convert seconds@16fps to nearest valid."""
    sec = max(MIN_SECONDS, min(MAX_SECONDS, float(sec)))
    raw = int(round(sec * FPS))
    n = max(1, round((raw - 1) / 4))
    return int(4 * n + 1)


def _even32(v: int) -> int:
    v = max(MIN_DIM, min(MAX_DIM, int(v)))
    return (v // 32) * 32


def _sys_stats():
    """Best-effort Apple-Silicon GPU + unified-memory stats (no sudo needed)."""
    out = {"gpu_util": None, "gpu_mem_gb": None,
           "mem_used_gb": None, "mem_total_gb": None,
           "cpu_watts": None, "gpu_watts": None, "total_watts": None,
           "gpu_temp": None}
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
    # Power (watts) + GPU temp from the sudoless macmon streamer, if running.
    p = _power_latest()
    if p:
        out["cpu_watts"] = p.get("cpu_watts")
        out["gpu_watts"] = p.get("gpu_watts")
        out["total_watts"] = p.get("total_watts")
        out["gpu_temp"] = p.get("gpu_temp")
    return out


# --- Sudoless power monitor via `macmon pipe` (Apple Silicon) --------------
_MACMON_CANDIDATES = ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon",
                      "macmon")
_power_lock = threading.Lock()
_power_data = {}


def _power_latest():
    with _power_lock:
        return dict(_power_data)


def _macmon_bin():
    for c in _MACMON_CANDIDATES:
        if os.path.sep in c:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        else:
            try:
                subprocess.check_output([c, "--version"], text=True, timeout=3,
                                        stderr=subprocess.DEVNULL)
                return c
            except Exception:
                pass
    return None


def _power_worker():
    """Stream `macmon pipe` JSON lines; cache latest watts + GPU temp.

    macmon reports SoC power without root. Runs forever, auto-restarting the
    child if it dies. If macmon is missing, exits quietly (watts stay null).
    """
    binp = _macmon_bin()
    if not binp:
        return
    while True:
        try:
            proc = subprocess.Popen(
                [binp, "pipe", "-i", "2000"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                cpu = d.get("cpu_power")
                gpu = d.get("gpu_power")
                ane = d.get("ane_power") or 0.0
                total = d.get("all_power")
                if total is None and cpu is not None and gpu is not None:
                    total = cpu + gpu + ane
                temp = (d.get("temp") or {}).get("gpu_temp_avg")
                with _power_lock:
                    _power_data.clear()
                    _power_data.update({
                        "cpu_watts": round(cpu, 1) if cpu is not None else None,
                        "gpu_watts": round(gpu, 1) if gpu is not None else None,
                        "total_watts": round(total, 1) if total is not None else None,
                        "gpu_temp": round(temp) if temp is not None else None,
                    })
            proc.wait()
        except Exception:
            pass
        time.sleep(5)  # child died; back off then relaunch


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


def _guide_arg(g):
    """Wan2.2-A14B is a dual model — its sampler indexes guide_scale[0]/[1], so a
    single float crashes. Emit a "low,high" pair (same value for both boundaries)."""
    try:
        gv = float(g)
    except (TypeError, ValueError):
        gv = DEF_GUIDE
    if DUAL_MODEL:
        return f"{gv},{gv}"
    return str(gv)


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
        "--guide-scale", _guide_arg(opts["guide_scale"]),
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


# LongCat prints "segment i/n (K new frames) — total elapsed ...s" per segment.
_LC_SEG_RE = re.compile(r"segment\s+(\d+)\s*/\s*(\d+)")


def _run_longcat(jid, opts):
    """Generate a long clip with LongCat-Video (its own repo + venv).

    Runs scripts/run_long_video.py with cwd=LC_REPO_DIR so its `_common`
    imports resolve. Streams stdout for per-segment progress. LongCat is
    text-to-video; no reference image is used."""
    out_name = "lc_" + uuid.uuid4().hex[:12] + ".mp4"
    out_path = os.path.join(OUT_DIR, out_name)
    segments = int(opts["segments"])
    cmd = [
        LC_VENV_PY, os.path.join("scripts", "run_long_video.py"),
        "--weights", LC_WEIGHTS_DIR,
        "--variant", LC_VARIANT,
        "--cfg-step-lora",
        "--prompt", opts["prompt"],
        "--num-segments", str(segments),
        "--num-frames-per-segment", str(LC_SEG_FRAMES),
        "--num-cond-frames", str(LC_COND_FRAMES),
        "--height", str(opts["height"]),
        "--width", str(opts["width"]),
        "--seed", str(opts["seed"]),
        "--out", out_path,
    ]
    if opts.get("negative_prompt"):
        cmd += ["--negative-prompt", opts["negative_prompt"]]

    t0 = time.time()
    # Cap this run's MLX memory so an overshoot raises (catchable) instead of
    # exhausting unified RAM and panicking the machine. Budget = the memory we
    # measured free at launch, minus an OS/other-process headroom. Read by the
    # venv's sitecustomize.py at interpreter startup.
    avail = _avail_mem_gb()
    mem_limit_gb = max(16.0, round(avail - LC_MLX_HEADROOM_GB, 1))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "LONGCAT_MLX_MEM_LIMIT_GB": str(mem_limit_gb),
        "LONGCAT_MLX_CACHE_LIMIT_GB": str(LC_MLX_CACHE_GB),
    }
    _set(jid, mem_limit_gb=mem_limit_gb, avail_gb_at_launch=avail)
    proc = subprocess.Popen(cmd, cwd=LC_REPO_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 60:
            tail.pop(0)
        low = line.lower()
        if "building pipeline" in low or "loading from" in low:
            _set(jid, stage="loading model + merging fast-mode LoRA", progress=4)
        elif "merged" in low and "modules" in low:
            _set(jid, stage="encoding prompt", progress=6)
        else:
            m = _LC_SEG_RE.search(line)
            if m:
                done, tot = int(m.group(1)), int(m.group(2))
                # A "segment i/n" line prints when segment i FINISHES.
                _set(jid, stage=f"segment {done}/{tot} rendered",
                     progress=int(8 + 88 * done / max(1, tot)))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("longcat generate failed: " + " | ".join(tail[-8:]))
    if not os.path.isfile(out_path):
        raise RuntimeError("longcat generate failed: no output produced — "
                           + " | ".join(tail[-6:]))
    elapsed = round(time.time() - t0, 1)
    return out_name, elapsed


def _run_job(jid):
    job = _get(jid)
    if not job:
        return
    opts = job["opts"]
    try:
        if opts.get("engine") == "longcat":
            if not os.path.isfile(LC_VENV_PY):
                raise RuntimeError("LongCat runtime is not installed "
                                   f"({LC_VENV_PY} missing)")
            if not _lc_weights_ready():
                raise RuntimeError("LongCat-Video weights are missing — expected "
                                   f"{LC_WEIGHTS_DIR}/LongCat-Video-{LC_VARIANT}/")
            # Memory guardrail: LongCat needs a large chunk of unified memory.
            # Running it on top of the loaded LLM has hard-crashed the Mac, so
            # refuse to start unless enough RAM is genuinely free right now.
            avail = _avail_mem_gb()
            if avail < LC_REQUIRED_FREE_GB:
                raise RuntimeError(
                    f"Not enough free memory for LongCat: {avail:.0f}GB free, "
                    f"needs ~{LC_REQUIRED_FREE_GB:.0f}GB. Free up unified memory "
                    f"first — unload the chat LLM (or other large models) and "
                    f"close heavy apps, then try again. This guard prevents the "
                    f"out-of-memory crash that requires a reboot.")
            _set(jid, stage="loading model + merging fast-mode LoRA", progress=3)
            name, elapsed = _run_longcat(jid, opts)
            _set(jid, stage="saving", progress=96)
            num_frames = _lc_frames(opts["segments"])
            result = {
                "filename": name,
                "url": f"/files/{name}",
                "engine": "longcat",
                "model": LC_MODEL_LABEL,
                "width": opts["width"], "height": opts["height"],
                "num_frames": num_frames, "fps": LC_FPS,
                "duration": round(num_frames / LC_FPS, 2),
                "segments": opts["segments"], "steps": LC_STEPS,
                "seed": opts["seed"], "mode": "t2v",
                "prompt": opts["prompt"],
                "seconds": elapsed,
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            _set(jid, status="done", stage="done", progress=100, result=result)
            return
        if not _weights_ready():
            raise RuntimeError("Wan2.2-I2V-A14B weights are missing or still "
                               "downloading — check the model directory")
        if I2V_ONLY and not opts.get("image"):
            raise RuntimeError("This model is image-to-video only — a start "
                               "image is required")
        _set(jid, stage="loading model + encoders (first run is slow)", progress=3)
        name, elapsed = _run_generate(jid, opts)
        _set(jid, stage="saving", progress=96)
        result = {
            "filename": name,
            "url": f"/files/{name}",
            "engine": "wan",
            "width": opts["width"], "height": opts["height"],
            "num_frames": opts["num_frames"], "fps": FPS,
            "duration": round(opts["num_frames"] / FPS, 2),
            "steps": opts["steps"], "seed": opts["seed"],
            "mode": "i2v" if opts.get("image") else "t2v",
            "prompt": opts["prompt"],
            "seconds": elapsed,
            "tiling": opts.get("tiling"),
            "tiling_forced": opts.get("tiling_forced", False),
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
            lc_avail_gb = _avail_mem_gb()
            return self._json(200, {
                "ok": True,
                "scripts_available": _scripts_available(),
                "weights_ready": _weights_ready(),
                "model": MODEL_LABEL,
                "model_dir": MODEL_DIR,
                "i2v_only": I2V_ONLY,
                "engines": {
                    "wan": {"available": _scripts_available() and _weights_ready(),
                            "label": MODEL_LABEL, "modes": ["i2v"]},
                    "longcat": {
                        "available": _lc_available(),
                        "label": LC_MODEL_LABEL, "modes": ["t2v"],
                        "free_gb": lc_avail_gb,
                        "required_free_gb": LC_REQUIRED_FREE_GB,
                        "mem_ok": lc_avail_gb >= LC_REQUIRED_FREE_GB,
                    },
                },
                "queue": len(_work_q),
            })
        if path == "/info":
            return self._json(200, {
                "model": MODEL_LABEL,
                "modes": ["i2v"],
                "i2v_only": I2V_ONLY,
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
                "engines": {
                    "wan": {
                        "available": _scripts_available() and _weights_ready(),
                        "label": MODEL_LABEL, "modes": ["i2v"], "fps": FPS,
                        "res_presets": RES_PRESETS, "res_default": DEF_RES,
                        "needs_image": I2V_ONLY,
                    },
                    "longcat": {
                        "available": _lc_available(),
                        "label": LC_MODEL_LABEL, "modes": ["t2v"], "fps": LC_FPS,
                        "res_presets": LC_RES_PRESETS, "res_default": LC_DEF_RES,
                        "aspect_default": LC_DEF_ASPECT,
                        "length_presets": LC_LENGTH_PRESETS,
                        "needs_image": False,
                        "defaults": {"duration_seconds": LC_DEF_SECONDS,
                                     "res": LC_DEF_RES, "aspect": LC_DEF_ASPECT},
                        "limits": {"min_seconds": LC_MIN_SECONDS,
                                   "max_seconds": LC_MAX_SECONDS,
                                   "max_segments": LC_MAX_SEGMENTS},
                        "note": ("~10 min render per segment @ 480p; "
                                 "15s≈3 segments (~30 min), 30s≈6 segments (~60 min)"),
                    },
                },
                "engine_default": "wan",
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
            engine = str(d.get("engine") or "wan").lower()
            if engine not in ("wan", "longcat"):
                engine = "wan"

            # --- LongCat text-to-video (long clips) -------------------------
            if engine == "longcat":
                if not os.path.isfile(LC_VENV_PY):
                    return self._json(400, {"error": "LongCat runtime is not "
                                            "installed on this machine"})
                # Reject up front if memory is too low to run safely, so the
                # user gets an immediate, actionable message instead of a job
                # that queues then fails (and avoids the OOM crash entirely).
                avail = _avail_mem_gb()
                if avail < LC_REQUIRED_FREE_GB:
                    return self._json(400, {"error":
                        f"Not enough free memory for LongCat right now: "
                        f"{avail:.0f}GB free, needs ~{LC_REQUIRED_FREE_GB:.0f}GB. "
                        f"Unload the chat LLM / large models to free unified "
                        f"memory, then try again.",
                        "mem": {"free_gb": avail,
                                "required_free_gb": LC_REQUIRED_FREE_GB}})
                res = str(d.get("res") or LC_DEF_RES)
                if res not in LC_RES_PRESETS:
                    res = LC_DEF_RES
                aspect = d.get("aspect") or d.get("preset")
                if aspect in LC_RES_PRESETS[res]:
                    w, h = LC_RES_PRESETS[res][aspect]
                else:
                    w = _even16(d.get("width", LC_RES_PRESETS[res][LC_DEF_ASPECT][0]))
                    h = _even16(d.get("height", LC_RES_PRESETS[res][LC_DEF_ASPECT][1]))
                if d.get("segments"):
                    segments = max(1, min(LC_MAX_SEGMENTS, int(d.get("segments"))))
                else:
                    segments = _lc_segments_for_seconds(
                        d.get("duration_seconds", LC_DEF_SECONDS))
                opts = {
                    "engine": "longcat",
                    "prompt": prompt[:2000],
                    "negative_prompt": (d.get("negative_prompt") or "").strip(),
                    "width": w, "height": h,
                    "segments": segments,
                    "seed": int(d.get("seed", 42)),
                }
                jid = "lc_" + uuid.uuid4().hex[:12]
                _set(jid, status="running", stage="queued", progress=0,
                     result=None, error=None, opts=opts)
                with _work_cv:
                    _work_q.append(jid)
                    _work_cv.notify()
                return self._json(200, {"ok": True, "job_id": jid,
                                        "engine": "longcat", "segments": segments,
                                        "est_frames": _lc_frames(segments),
                                        "est_seconds": round(_lc_frames(segments) / LC_FPS, 1)})

            # --- Wan2.2 image-to-video (default) ----------------------------
            if I2V_ONLY and not (d.get("image") or "").strip():
                return self._json(400, {"error": "This model is image-to-video "
                                        "only — a reference / start image is required"})
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
            # Memory guardrail: VAE decode with tiling="none" holds the whole
            # frame stack in unified memory at once. On large clips that peak can
            # exceed physical RAM and hang the whole machine. If the requested
            # pixel*frame budget is heavy, force a safe tiling mode.
            tiling_forced = False
            px_frames = w * h * num_frames
            if tiling == "none" and px_frames > NONE_TILING_PX_BUDGET:
                tiling = "auto"
                tiling_forced = True
            shift = d.get("shift", None)
            try:
                shift = float(shift) if shift not in (None, "") else None
            except (TypeError, ValueError):
                shift = None
            opts = {
                "engine": "wan",
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
                "tiling_forced": tiling_forced,
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
        if path == "/save":
            d = self._read_body()
            name = os.path.basename(str(d.get("filename") or ""))
            if not (name.endswith(".mp4") and re.fullmatch(r"[\w.\-]+", name)):
                return self._json(400, {"error": "invalid filename"})
            src = os.path.join(OUT_DIR, name)
            if not os.path.isfile(src):
                return self._json(404, {"error": "clip not found (may have been cleaned up)"})
            dest_dir = str(d.get("dest") or SAVE_DIR)
            try:
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, name)
                # Avoid clobbering an existing file with the same name.
                if os.path.exists(dest):
                    stem, ext = os.path.splitext(name)
                    dest = os.path.join(
                        dest_dir,
                        f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}")
                shutil.copy2(src, dest)
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return self._json(200, {"ok": True, "path": dest})
        return self._json(404, {"error": "not found"})


def main():
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_power_worker, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-video ready at http://{HOST}:{PORT} "
          f"(weights_ready={_weights_ready()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
