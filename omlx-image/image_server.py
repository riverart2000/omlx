#!/usr/bin/env python3
"""omlx-image — shared local and cloud image-generation service (port 8400).

Mirrors the pattern of the TTS Studio (:8200) and memory (:8300) services:
a tiny stdlib HTTP server with an async job API and a single serialized
worker (image generation is compute-bound, so we run one at a time).

Endpoints:
  GET  /health             -> service + model status
  GET  /info               -> defaults, presets, sizes
  GET  /library            -> recent generated/overlay images
  POST /generate           -> {prompt,width,height,steps,seed,...} => {job_id}
  POST /overlay            -> apply ad text/sign overlay to saved/uploaded image
  GET  /status?id=JOB      -> job progress / result
  GET  /files/<name>.png   -> the rendered image
  POST /load               -> warm-load the model now (optional)
"""
import json
import os
import base64
import io
import re
import ssl
import shutil
import mimetypes
import select
import subprocess
import threading
import time
import uuid
import sys
import traceback
import numpy as np
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import hidream_engine as eng

COMMON_DIR = os.environ.get("OMLX_COMMON_DIR", "/Users/joebains/omlx-common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from model_memory_coordinator import (
    acquire_lease, available_memory_gb, lease_status, release_lease,
    pressure_available_memory_gb, reload_omlx_models, unload_omlx_models,
)
from studio_logging import EventLogger
from model_registry import REGISTRY_PATH, get_model, public_models

EVENT_LOG = EventLogger("image-service", "image-events.jsonl")

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
MFLUX_FLUX2_BIN = os.environ.get(
    "MFLUX_FLUX2_BIN",
    os.path.join(BASE_DIR, ".venv-mflux", "bin", "mflux-generate-flux2"),
)
MFLUX_FLUX2_EDIT_BIN = os.environ.get(
    "MFLUX_FLUX2_EDIT_BIN",
    os.path.join(BASE_DIR, ".venv-mflux", "bin", "mflux-generate-flux2-edit"),
)
FLUX2_LOCAL_MODEL = os.environ.get("FLUX2_LOCAL_MODEL", "flux2-klein-4b")
_flux2_quantize_setting = os.environ.get(
    "FLUX2_LOCAL_QUANTIZE", "full").strip().lower()
FLUX2_LOCAL_QUANTIZE = (
    None if _flux2_quantize_setting in {"", "none", "off", "false", "full", "bf16"}
    else max(3, min(8, int(_flux2_quantize_setting)))
)
FLUX2_LOCAL_PRECISION = (
    "full precision" if FLUX2_LOCAL_QUANTIZE is None
    else f"{FLUX2_LOCAL_QUANTIZE}-bit"
)
FLUX2_LOCAL_STEPS = 4
FLUX2_LOCAL_MAX_TOKENS = max(512, min(2048, int(os.environ.get(
    "FLUX2_LOCAL_MAX_TOKENS", "1024"))))
FLUX2_LOCAL_CACHE_GB = float(os.environ.get("FLUX2_LOCAL_MLX_CACHE_GB", "4"))
FLUX2_LOCAL_MIN_FREE_GB = float(os.environ.get("FLUX2_LOCAL_MIN_FREE_GB", "32"))
FLUX2_LOCAL_TIMEOUT_SECONDS = float(os.environ.get(
    "FLUX2_LOCAL_TIMEOUT_SECONDS", "3600"))
KONTEXT_MODEL = os.environ.get(
    "KONTEXT_MODEL",
    "akx/FLUX.1-Kontext-dev-mflux-4bit",
)
KONTEXT_BASE_MODEL = os.environ.get("KONTEXT_BASE_MODEL", "dev")
KONTEXT_LOW_RAM_MODE = os.environ.get("KONTEXT_LOW_RAM", "auto").strip().lower()
KONTEXT_CACHE_GB = float(os.environ.get("KONTEXT_MLX_CACHE_GB", "4"))
KONTEXT_LOW_RAM_CACHE_GB = float(
    os.environ.get("KONTEXT_LOW_RAM_CACHE_GB", "1"))
KONTEXT_PERFORMANCE_MIN_FREE_GB = float(
    os.environ.get("KONTEXT_PERFORMANCE_MIN_FREE_GB", "28"))
KONTEXT_TIMEOUT_SECONDS = float(os.environ.get("KONTEXT_TIMEOUT_SECONDS", "1800"))
IMAGE_MIN_FREE_GB = float(os.environ.get("IMAGE_MIN_FREE_GB", "18"))
IMAGE_WARM_MIN_FREE_GB = float(os.environ.get("IMAGE_WARM_MIN_FREE_GB", "12"))
IMAGE_MEMORY_WAIT_SECONDS = float(os.environ.get("IMAGE_MEMORY_WAIT_SECONDS", "10"))
IMAGE_RESTORE_OMLX_MODELS = os.environ.get(
    "IMAGE_RESTORE_OMLX_MODELS", "0").strip().lower() in ("1", "true", "yes")


def _load_env_file(path):
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return vals


NB_ENV_FILE = os.environ.get("OMLX_ENV_FILE", "/Users/joebains/.omlx/.env")
_nb_env = _load_env_file(NB_ENV_FILE)
NB_TOKEN = (os.environ.get("REPLICATE_API_TOKEN")
            or _nb_env.get("REPLICATE_API_TOKEN") or "").strip()
XAI_TOKEN = (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
             or _nb_env.get("XAI_API_KEY") or _nb_env.get("GROK_API_KEY") or "").strip()
XAI_API = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
_grok_registry = get_model("grok") or {}
GROK_IMAGE_MODEL = _grok_registry.get("model", "grok-imagine-image")
NB_API = "https://api.replicate.com/v1"
NB_MODEL = os.environ.get("REPLICATE_NANO_BANANA_MODEL", "google/nano-banana-2")
NB_VERSION = os.environ.get(
    "REPLICATE_NANO_BANANA_VERSION",
    # Latest version id on the model page at integration time.
    "402c3da8f79427a67715983c7e97ba0a1f909fd0ae445b1041357c7718016946",
)
NB_LABEL = "Nano Banana 2 (Replicate cloud, Gemini 3.1 Flash Image)"
NB_RESOLUTION = "1K"
NB_ASPECTS = [
    "match_input_image", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
    "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
]
NB_OUTPUT_FORMATS = ["png", "jpg"]

# FLUX.2 klein 4B is an official Replicate model and can be called without
# pinning a version. Keep its schema choices explicit so provider-side default
# changes cannot silently alter the Studio's print artwork.
FLUX2_MODEL = os.environ.get(
    "REPLICATE_FLUX2_KLEIN_MODEL", "black-forest-labs/flux-2-klein-4b")
FLUX2_LABEL = "FLUX 2 Klein 4B (Replicate cloud)"
FLUX2_RESOLUTIONS = ["0.25", "0.5", "1", "2", "4"]
FLUX2_ASPECTS = [
    "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "5:4",
    "4:5", "21:9", "9:21", "match_input_image",
]
FLUX2_OUTPUT_FORMATS = ["webp", "jpg", "png"]
FLUX2_MAX_REFS = 5
FLUX2_LOCAL_MAX_REFS = 3
# A conservative upload bandwidth budget; Klein performs its own input resizing.
FLUX2_REFERENCE_BUDGET_PIXELS = 9_000_000

MAX_DIM = 3104
MIN_DIM = 256
# Quality-first, trained/native-ish resolutions for HiDream-O1.
# The model snaps to its trained list internally; these are chosen to land on
# high-quality targets out of the box for marketing/sales creative.
PRESETS = {
    "1:1":  (2048, 2048),
    "9:16": (1440, 2560),
    "16:9": (2560, 1440),
    "4:5":  (1792, 2304),
    "3:2":  (2496, 1664),
}

_jobs = {}
_jobs_lock = threading.Lock()
_work_q = []
_work_cv = threading.Condition()
_active_lock = threading.Lock()
_active_jid = ""
_active_proc = None


class JobCanceled(RuntimeError):
    pass


def _is_canceled(jid: str) -> bool:
    with _jobs_lock:
        return bool((_jobs.get(jid) or {}).get("cancel_requested"))


def _check_canceled(jid: str) -> None:
    if _is_canceled(jid):
        raise JobCanceled("Image generation stopped")


def _set_active(jid: str, proc=None) -> None:
    global _active_jid, _active_proc
    with _active_lock:
        _active_jid = jid
        _active_proc = proc


def _set_active_proc(jid: str, proc) -> None:
    global _active_proc
    with _active_lock:
        if _active_jid == jid:
            _active_proc = proc


def _clear_active(jid: str) -> None:
    global _active_jid, _active_proc
    with _active_lock:
        if _active_jid == jid:
            _active_jid = ""
            _active_proc = None


def _cancel_all_images() -> dict:
    """Cancel the active generator and empty the serialized service queue."""
    with _work_cv:
        queued = list(_work_q)
        _work_q.clear()
    with _active_lock:
        active_jid = _active_jid
        proc = _active_proc
    with _jobs_lock:
        for jid in queued:
            job = _jobs.get(jid)
            if job:
                job.update(status="canceled", stage="stopped", progress=0,
                           error=None, cancel_requested=True)
        if active_jid and active_jid in _jobs:
            _jobs[active_jid]["cancel_requested"] = True
            _jobs[active_jid]["stage"] = "stopping image generator…"
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass
    result = {"ok": True, "active_canceled": bool(active_jid),
              "queued_canceled": len(queued)}
    EVENT_LOG.event("image.queue_canceled", level="warning",
                    active_job_id=active_jid, **result)
    return result


def _memory_headroom(required_gb: float, wait_seconds: float = None) -> dict:
    """Wait briefly for model unloads and return the best OS memory estimate.

    The raw vm_stat figure intentionally excludes compressed pages.  macOS's
    memory-pressure figure includes memory the kernel can safely reclaim, so we
    use the larger of the two while retaining both values for useful errors.
    """
    timeout = IMAGE_MEMORY_WAIT_SECONDS if wait_seconds is None else wait_seconds
    deadline = time.time() + max(0.0, timeout)
    snapshot = {"reclaimable_gb": 0.0, "pressure_available_gb": 0.0,
                "effective_gb": 0.0}
    while True:
        reclaimable = available_memory_gb()
        pressure_available = pressure_available_memory_gb()
        effective = max(reclaimable, pressure_available)
        snapshot = {
            "reclaimable_gb": reclaimable,
            "pressure_available_gb": pressure_available,
            "effective_gb": effective,
        }
        if effective <= 0 or effective >= required_gb or time.time() >= deadline:
            return snapshot
        time.sleep(1)


def _decode_data_url_image(raw: str) -> bytes:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty reference image payload")
    if "," in s and "base64" in s[:80].lower():
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64 reference image: {e}") from e


def _nb_ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _nb_headers(extra=None):
    h = {
        "Authorization": "Bearer " + NB_TOKEN,
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _nb_json(url, method="GET", payload=None, headers=None, timeout=60):
    import urllib.request
    import urllib.error

    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers=(headers or _nb_headers()),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_nb_ssl_ctx()) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = (e.read() or b"").decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        detail = body
        try:
            j = json.loads(body or "{}")
            detail = j.get("detail") or j.get("error") or body
        except Exception:
            pass
        raise RuntimeError(f"Replicate API HTTP {e.code}: {detail}") from e


def _nb_upload_file(path):
    """Upload a local file to Replicate Files API and return served URL."""
    import urllib.request

    fname = os.path.basename(path)
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    boundary = "----omlximg" + uuid.uuid4().hex
    pre = (f"--{boundary}\r\n"
           f'Content-Disposition: form-data; name="content"; filename="{fname}"\r\n'
           f"Content-Type: {ctype}\r\n\r\n").encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    body = pre + data + post
    req = urllib.request.Request(
        NB_API + "/files", data=body, method="POST",
        headers={
            "Authorization": "Bearer " + NB_TOKEN,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=180, context=_nb_ssl_ctx()) as r:
        resp = json.loads(r.read() or b"{}")
    url = (resp.get("urls") or {}).get("get") or resp.get("url")
    if not url:
        raise RuntimeError("Replicate file upload returned no URL")
    return url


def _nb_download(url, dest):
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300, context=_nb_ssl_ctx()) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _xai_json(path, payload, timeout=600):
    import urllib.error
    import urllib.request

    if not XAI_TOKEN:
        raise RuntimeError("XAI_API_KEY/GROK_API_KEY is not set (checked "
                           + NB_ENV_FILE + ")")
    req = urllib.request.Request(
        XAI_API + path, data=json.dumps(payload).encode(), method="POST",
        headers={
            "Authorization": "Bearer " + XAI_TOKEN,
            "Content-Type": "application/json",
            "User-Agent": "oMLX-Image-Studio/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = (exc.read() or b"").decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"xAI image API HTTP {exc.code}: {detail}") from exc


def _xai_upload_file(path):
    import urllib.error
    import urllib.request

    boundary = "----omlximage" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        content = handle.read()
    body = b"".join([
        (f"--{boundary}\r\nContent-Disposition: form-data; "
         "name=\"purpose\"\r\n\r\nassistants\r\n").encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: {ctype}\r\n\r\n").encode(),
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        XAI_API + "/files", data=body, method="POST",
        headers={
            "Authorization": "Bearer " + XAI_TOKEN,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = (exc.read() or b"").decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"xAI file upload HTTP {exc.code}: {detail}") from exc
    if not result.get("id"):
        raise RuntimeError("xAI file upload returned no file id")
    return result["id"]


def _run_grok_image(jid, opts):
    """Generate or edit an image through xAI using up to three references."""
    import urllib.request
    from PIL import Image

    if not XAI_TOKEN:
        raise RuntimeError("XAI_API_KEY/GROK_API_KEY is not set (checked "
                           + NB_ENV_FILE + ")")
    started = time.time()
    refs = [x for x in (opts.get("reference_images") or []) if x][:3]
    temp_refs = []
    try:
        file_ids = []
        for index, raw in enumerate(refs, start=1):
            _check_canceled(jid)
            path = os.path.join(TMP_DIR, f"{jid}_grok_ref_{index:02d}.png")
            try:
                image = Image.open(io.BytesIO(_decode_data_url_image(raw))).convert("RGB")
                image.save(path, format="PNG", optimize=True)
            except Exception as exc:
                raise ValueError(f"invalid reference image #{index}: {exc}") from exc
            temp_refs.append(path)
            _set(jid, stage=f"uploading Grok reference {index}/{len(refs)}",
                 progress=5 + int(7 * index / max(1, len(refs))))
            file_ids.append(_xai_upload_file(path))

        width = int(opts.get("width") or 1024)
        height = int(opts.get("height") or 1024)
        ratio = "1:1" if abs(width - height) / max(width, height) < .12 else (
            "3:4" if height > width else "4:3")
        payload = {
            "model": GROK_IMAGE_MODEL,
            "prompt": str(opts.get("prompt") or "")[:6000],
            "n": 1,
            "resolution": "1k",
            "aspect_ratio": ratio,
        }
        endpoint = "/images/generations"
        if file_ids:
            endpoint = "/images/edits"
            payload["images"] = [{"file_id": value} for value in file_ids]
        _set(jid, stage="waiting for Grok Imagine…", progress=25)
        result = _xai_json(endpoint, payload, timeout=600)
        items = result.get("data") or []
        if not items:
            raise RuntimeError("xAI returned no generated image")
        item = items[0] if isinstance(items[0], dict) else {}
        output = b""
        if item.get("b64_json"):
            output = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            _set(jid, stage="downloading Grok image", progress=90)
            with urllib.request.urlopen(item["url"], timeout=300) as response:
                output = response.read()
        if not output:
            raise RuntimeError("xAI returned an image without downloadable data")
        name = "img_" + uuid.uuid4().hex[:12] + ".png"
        path = os.path.join(OUT_DIR, name)
        with Image.open(io.BytesIO(output)) as image:
            image = image.convert("RGB")
            width, height = image.size
            image.save(path, format="PNG", optimize=True)
        return name, int(width), int(height), round(time.time() - started, 1)
    finally:
        for path in temp_refs:
            try:
                os.remove(path)
            except OSError:
                pass


def _run_nano_banana(jid, opts):
    from PIL import Image

    if not NB_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is not set (checked " + NB_ENV_FILE + ")")

    t0 = time.time()
    refs = [x for x in (opts.get("reference_images") or []) if x][:14]
    uploaded = []
    local_tmp = []
    try:
        if refs:
            _set(jid, stage="uploading reference images", progress=5)
            for i, raw in enumerate(refs, start=1):
                b = _decode_data_url_image(raw)
                lp = os.path.join(TMP_DIR, f"{jid}_nb_ref_{i:02d}.png")
                try:
                    img = Image.open(io.BytesIO(b)).convert("RGB")
                    img.save(lp, format="PNG")
                except Exception as e:
                    raise ValueError(f"invalid reference image #{i}: {e}") from e
                local_tmp.append(lp)
                uploaded.append(_nb_upload_file(lp))

        aspect = opts.get("nano_aspect_ratio") or "1:1"
        if aspect == "match_input_image" and not uploaded:
            aspect = "1:1"
        if aspect not in NB_ASPECTS:
            aspect = "1:1"

        out_fmt = (opts.get("nano_output_format") or "png").lower()
        if out_fmt not in NB_OUTPUT_FORMATS:
            out_fmt = "png"

        labels = [str(x).strip() for x in
                  (opts.get("reference_labels") or [])[:len(uploaded)]]
        mapped_prompt = _flux2_character_mapped_prompt(
            opts["prompt"], labels, len(uploaded))
        inp = {
            "prompt": mapped_prompt,
            "resolution": NB_RESOLUTION,
            "aspect_ratio": aspect,
            "output_format": out_fmt,
            "google_search": bool(opts.get("nano_google_search", False)),
            "image_search": bool(opts.get("nano_image_search", False)),
        }
        if uploaded:
            inp["image_input"] = uploaded

        payload = {"version": NB_VERSION, "input": inp}
        _set(jid, stage="submitting to Nano Banana 2", progress=12)
        pred = _nb_json(NB_API + "/predictions", method="POST", payload=payload, timeout=90)
        pid = pred.get("id")
        if not pid:
            raise RuntimeError("Replicate did not return a prediction id")
        _set(jid, remote_prediction_id=pid)

        status = pred.get("status")
        deadline = time.time() + 900
        while status not in ("succeeded", "failed", "canceled"):
            if _is_canceled(jid):
                try:
                    _nb_json(NB_API + "/predictions/" + pid + "/cancel",
                             method="POST", payload={}, timeout=30)
                except Exception:
                    pass
                raise JobCanceled("Image generation stopped")
            if time.time() > deadline:
                raise RuntimeError("Replicate prediction timed out (>15 min)")
            time.sleep(2.5)
            pred = _nb_json(
                NB_API + "/predictions/" + pid,
                method="GET",
                headers=_nb_headers(),
                timeout=60,
            )
            status = pred.get("status")
            _set(jid, stage="rendering on Nano Banana 2…", progress=58)
        if status != "succeeded":
            raise RuntimeError(f"Replicate prediction {status}: {pred.get('error')}")

        out = pred.get("output")
        out_url = out[0] if isinstance(out, list) else out
        if not out_url:
            raise RuntimeError("Replicate succeeded but returned no output URL")

        _set(jid, stage="downloading result", progress=92)
        ext = ".png" if out_fmt == "png" else ".jpg"
        out_name = "img_" + uuid.uuid4().hex[:12] + ext
        out_path = os.path.join(OUT_DIR, out_name)
        _nb_download(out_url, out_path)
        elapsed = round(time.time() - t0, 1)
        im = Image.open(out_path)
        w, h = im.size
        return out_name, int(w), int(h), elapsed
    finally:
        for p in local_tmp:
            try:
                os.remove(p)
            except OSError:
                pass


def _flux2_character_mapped_prompt(prompt, labels, reference_count,
                                   compact=False):
    """Put a deterministic natural-language identity map first in the prompt."""
    assignments = []
    for index in range(reference_count):
        label = labels[index] if index < len(labels) and labels[index] else (
            f"recurring character {index + 1}")
        assignments.append(
            f"Use the subject shown in image {index + 1} as {label}"
        )
    if not assignments:
        return prompt.rstrip()
    if compact:
        return (
            "MANDATORY REFERENCE MAP: " + "; ".join(assignments) + ". "
            "Each image belongs only to its named character. Preserve that "
            "character's face, age, body, colouring and permanent features. "
            "Apply every action, costume and position only to the name assigned "
            "to it; never swap them. Render each requested mapped character once "
            "and keep supporting figures visually distinct.\n"
            + prompt.rstrip()
        )
    return (
        "Character identity mapping (highest priority): "
        + ". ".join(assignments) + ". "
        "Each numbered image belongs only to its assigned named character. "
        "For every mapped character, preserve the assigned image's face, hair, "
        "apparent age, skin or fur colour, body proportions, permanent physical "
        "features, and signature accessories. Preserve its clothing by default. "
        "When a PER-CHARACTER SCENE CONTRACT assigns remains, portrait, statue, "
        "reflection or vision instead of a living character, use that numbered "
        "image only for recognisable likeness and render the identity solely in "
        "the assigned non-living form; do not create a separate living or full-body "
        "copy of that identity. "
        "When a PER-CHARACTER SCENE CONTRACT explicitly gives one named character "
        "a temporary wardrobe override, apply that replacement wardrobe to that "
        "assigned identity alone and remove the listed default items from that "
        "character alone. Keep every other character in its own assigned wardrobe; "
        "never transfer a costume, wig, armour, prop, action or position between "
        "cast records. Use the scene description for pose, action, expression, camera, "
        "setting, and relationships between characters. Keep every mapped name as "
        "one separate individual. Any supporting people requested by the scene are "
        "additional distinct individuals and do not inherit a mapped identity.\n\n"
        + prompt.rstrip()
    )


def _run_flux_2_klein_4b(jid, opts):
    """Run Replicate FLUX.2 klein 4B with ordered character references."""
    from PIL import Image, ImageOps

    if not NB_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is not set (checked " + NB_ENV_FILE + ")")

    t0 = time.time()
    refs = [x for x in (opts.get("reference_images") or []) if x][:FLUX2_MAX_REFS]
    labels = [str(x).strip() for x in
              (opts.get("reference_labels") or [])[:len(refs)]]
    local_tmp = []
    uploaded = []
    try:
        prepared = []
        total_pixels = 0
        for i, raw in enumerate(refs, start=1):
            try:
                image = Image.open(io.BytesIO(_decode_data_url_image(raw)))
                image = ImageOps.exif_transpose(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"invalid reference image #{i}: {e}") from e
            prepared.append(image)
            total_pixels += image.width * image.height

        # Keep multi-reference uploads compact. Scale every reference equally
        # so none is privileged and the numbered character order remains intact.
        scale = 1.0
        if total_pixels > FLUX2_REFERENCE_BUDGET_PIXELS:
            scale = (FLUX2_REFERENCE_BUDGET_PIXELS / total_pixels) ** 0.5

        if prepared:
            _set(jid, stage="uploading FLUX 2 Klein references", progress=5)
        for i, image in enumerate(prepared, start=1):
            if scale < 1.0:
                size = (max(1, int(image.width * scale)),
                        max(1, int(image.height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
            path = os.path.join(TMP_DIR, f"{jid}_flux2_ref_{i:02d}.png")
            image.save(path, format="PNG", optimize=True)
            local_tmp.append(path)
            uploaded.append(_nb_upload_file(path))
            _set(jid, stage=f"uploading reference {i}/{len(prepared)}",
                 progress=5 + int(5 * i / max(1, len(prepared))))

        aspect = str(opts.get("flux2_aspect_ratio") or "1:1")
        if aspect == "match_input_image" and not uploaded:
            aspect = "1:1"
        if aspect not in FLUX2_ASPECTS:
            aspect = "1:1"
        resolution = str(opts.get("flux2_resolution") or "1")
        if resolution not in FLUX2_RESOLUTIONS:
            resolution = "1"
        out_fmt = str(opts.get("flux2_output_format") or "png").lower()
        if out_fmt not in FLUX2_OUTPUT_FORMATS:
            out_fmt = "png"

        prompt = opts["prompt"].rstrip()
        prompt += (
            "\n\nFill the frame with continuous natural scenery, character action "
            "and organic material textures. Keep every area pictorial and immersive."
        )
        if uploaded:
            prompt = _flux2_character_mapped_prompt(
                prompt, labels, len(uploaded), compact=True)

        inp = {
            "prompt": prompt,
            "images": uploaded,
            "output_megapixels": resolution,
            "aspect_ratio": aspect,
            "seed": int(opts.get("seed", 32)),
            "go_fast": bool(opts.get("flux2_go_fast", False)),
            "output_format": out_fmt,
            "output_quality": max(0, min(100, int(
                opts.get("flux2_output_quality", 100)))),
            "disable_safety_checker": bool(
                opts.get("flux2_disable_safety_checker", False)),
        }

        endpoint = NB_API + "/models/" + FLUX2_MODEL + "/predictions"
        _set(jid, stage="submitting to FLUX 2 Klein", progress=12)
        pred = _nb_json(endpoint, method="POST", payload={"input": inp}, timeout=90)
        pid = pred.get("id")
        if not pid:
            raise RuntimeError("Replicate did not return a FLUX 2 Klein prediction id")
        _set(jid, remote_prediction_id=pid)
        status = pred.get("status")
        poll_url = (pred.get("urls") or {}).get("get") or (
            NB_API + "/predictions/" + pid)
        cancel_url = (pred.get("urls") or {}).get("cancel") or (
            NB_API + "/predictions/" + pid + "/cancel")
        deadline = time.time() + 900
        while status not in ("succeeded", "failed", "canceled"):
            if _is_canceled(jid):
                try:
                    _nb_json(cancel_url, method="POST", payload={}, timeout=30)
                except Exception:
                    pass
                raise JobCanceled("Image generation stopped")
            if time.time() > deadline:
                raise RuntimeError("FLUX 2 Klein prediction timed out (>15 min)")
            time.sleep(2.5)
            pred = _nb_json(poll_url, method="GET", headers=_nb_headers(), timeout=60)
            status = pred.get("status")
            elapsed_wait = time.time() - t0
            progress = min(88, 25 + int(elapsed_wait / 5))
            _set(jid, stage="rendering on FLUX 2 Klein…", progress=progress)
        if status != "succeeded":
            raise RuntimeError(f"FLUX 2 Klein prediction {status}: {pred.get('error')}")

        out = pred.get("output")
        out_url = out[0] if isinstance(out, list) else out
        if not out_url:
            raise RuntimeError("FLUX 2 Klein succeeded but returned no output URL")
        _set(jid, stage="downloading FLUX 2 Klein result", progress=92)
        ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}[out_fmt]
        out_name = "img_" + uuid.uuid4().hex[:12] + ext
        out_path = os.path.join(OUT_DIR, out_name)
        _nb_download(out_url, out_path)
        elapsed = round(time.time() - t0, 1)
        with Image.open(out_path) as image:
            width, height = image.size
        return out_name, int(width), int(height), elapsed
    finally:
        for path in local_tmp:
            try:
                os.remove(path)
            except OSError:
                pass


def _kontext_profile(opts):
    mode = KONTEXT_LOW_RAM_MODE
    if mode in ("1", "true", "yes", "on"):
        low_ram = True
    elif mode in ("0", "false", "no", "off", "performance"):
        low_ram = False
    else:
        available = float(opts.get("_available_memory_gb") or 0)
        low_ram = available <= 0 or available < KONTEXT_PERFORMANCE_MIN_FREE_GB
    cache_gb = KONTEXT_LOW_RAM_CACHE_GB if low_ram else KONTEXT_CACHE_GB
    return low_ram, cache_gb


def _run_kontext(jid, opts):
    from PIL import Image

    if not os.path.isfile(MFLUX_KONTEXT_BIN):
        raise RuntimeError(f"missing kontext binary: {MFLUX_KONTEXT_BIN}")

    ref_bytes = _decode_data_url_image(opts.get("reference_image", ""))
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
    low_ram, cache_gb = _kontext_profile(opts)
    if low_ram:
        cmd.append("--low-ram")
    cmd.extend(["--mlx-cache-limit-gb", str(cache_gb)])
    profile_name = "safe low-memory" if low_ram else "performance"
    _set(jid, stage=f"loading FLUX model ({profile_name} mode)", progress=8,
         flux_profile=profile_name, flux_cache_gb=cache_gb)
    _check_canceled(jid)
    proc = subprocess.Popen(
        cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0,
    )
    _set_active_proc(jid, proc)
    captured = bytearray()
    scan_text = ""
    denoising = False
    last_loading_progress = 8
    while proc.poll() is None:
        if _is_canceled(jid):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        elapsed_now = time.time() - t0
        if elapsed_now > KONTEXT_TIMEOUT_SECONDS:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            try:
                os.remove(ref_path)
            except OSError:
                pass
            raise RuntimeError(
                f"kontext timed out after {KONTEXT_TIMEOUT_SECONDS / 60:.0f} minutes")
        readable, _, _ = select.select([proc.stdout], [], [], 1.0)
        if readable:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if chunk:
                captured.extend(chunk)
                if len(captured) > 65536:
                    del captured[:-65536]
                scan_text = (scan_text + chunk.decode(
                    "utf-8", "replace"))[-8192:]
                matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)", scan_text))
                if matches:
                    done = int(matches[-1].group(1))
                    total = max(1, int(matches[-1].group(2)))
                    if total == int(opts["steps"]) and done <= total:
                        denoising = True
                        progress = int(18 + 77 * done / total)
                        _set(jid, stage=f"FLUX denoising {done}/{total}",
                             progress=min(95, progress))
        elif not denoising:
            # Model loading has no native step counter. Advance slowly as a
            # heartbeat so the UI distinguishes a long load from a dead job.
            loading_progress = min(17, 8 + int(elapsed_now / 30))
            if loading_progress != last_loading_progress:
                last_loading_progress = loading_progress
                _set(jid, stage=f"loading FLUX model ({profile_name} mode)",
                     progress=loading_progress)
    if proc.stdout:
        remainder = proc.stdout.read() or b""
        captured.extend(remainder)
    proc_output = captured.decode("utf-8", "replace")
    try:
        os.remove(ref_path)
    except OSError:
        pass
    _set_active_proc(jid, None)
    _check_canceled(jid)
    if proc.returncode != 0:
        tail = proc_output.strip().splitlines()[-6:]
        raise RuntimeError("kontext failed: " + " | ".join(tail))
    if not os.path.isfile(out_path):
        raise RuntimeError("kontext failed: output image not found")
    img = Image.open(out_path)
    w, h = img.size
    elapsed = round(time.time() - t0, 1)
    return out_name, int(w), int(h), elapsed


def _run_flux2_klein_local(jid, opts):
    """Run distilled FLUX.2 Klein 4B locally through native mflux/MLX."""
    from PIL import Image

    refs_raw = [
        x for x in (opts.get("reference_images") or []) if x
    ][:FLUX2_LOCAL_MAX_REFS]
    labels = [str(x).strip() for x in
              (opts.get("reference_labels") or [])[:len(refs_raw)]]
    binary = MFLUX_FLUX2_EDIT_BIN if refs_raw else MFLUX_FLUX2_BIN
    if not os.path.isfile(binary):
        raise RuntimeError(f"missing local FLUX 2 Klein binary: {binary}")

    ref_paths = []
    out_name = f"img_{uuid.uuid4().hex[:12]}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    t0 = time.time()
    try:
        if refs_raw:
            _set(jid, stage="preparing local Klein references", progress=4)
            ref_paths = _materialize_reference_images(
                jid, refs_raw, max_refs=FLUX2_LOCAL_MAX_REFS)

        prompt = opts["prompt"].rstrip()
        if ref_paths:
            prompt = _flux2_character_mapped_prompt(
                prompt, labels, len(ref_paths), compact=True)
        prompt += (
            "\n\nFill the frame with continuous natural scenery, character action "
            "and organic material textures. Keep every area pictorial and immersive."
        )

        cmd = [
            binary,
            "--model", FLUX2_LOCAL_MODEL,
            "--prompt", prompt,
            "--steps", str(FLUX2_LOCAL_STEPS),
            "--guidance", "1.0",
            "--seed", str(opts["seed"]),
            "--height", str(opts["height"]),
            "--width", str(opts["width"]),
            "--mlx-cache-limit-gb", str(FLUX2_LOCAL_CACHE_GB),
            "--output", out_path,
        ]
        if FLUX2_LOCAL_QUANTIZE is not None:
            cmd.extend(["--quantize", str(FLUX2_LOCAL_QUANTIZE)])
        if ref_paths:
            cmd.extend(["--image-paths", *ref_paths])

        _set(jid, stage="loading local FLUX 2 Klein (first use downloads weights)",
             progress=6, flux2_precision=FLUX2_LOCAL_PRECISION,
             flux2_cache_gb=FLUX2_LOCAL_CACHE_GB)
        _check_canceled(jid)
        max_tokens = max(512, min(2048, int(
            opts.get("flux2_max_tokens") or FLUX2_LOCAL_MAX_TOKENS)))
        flux2_env = os.environ.copy()
        flux2_env["MFLUX_FLUX2_MAX_SEQUENCE_LENGTH"] = str(
            max_tokens)
        _set(jid, flux2_max_tokens=max_tokens)
        proc = subprocess.Popen(
            cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, env=flux2_env,
        )
        _set_active_proc(jid, proc)
        captured = bytearray()
        scan_text = ""
        denoising = False
        last_loading_progress = 6
        while proc.poll() is None:
            if _is_canceled(jid):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            elapsed_now = time.time() - t0
            if elapsed_now > FLUX2_LOCAL_TIMEOUT_SECONDS:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise RuntimeError(
                    "local FLUX 2 Klein timed out after "
                    f"{FLUX2_LOCAL_TIMEOUT_SECONDS / 60:.0f} minutes")
            readable, _, _ = select.select([proc.stdout], [], [], 1.0)
            if readable:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if chunk:
                    captured.extend(chunk)
                    if len(captured) > 131072:
                        del captured[:-131072]
                    scan_text = (scan_text + chunk.decode(
                        "utf-8", "replace"))[-16384:]
                    matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)", scan_text))
                    if matches:
                        done = int(matches[-1].group(1))
                        total = max(1, int(matches[-1].group(2)))
                        if total == FLUX2_LOCAL_STEPS and done <= total:
                            denoising = True
                            progress = int(20 + 75 * done / total)
                            _set(jid, stage=f"local Klein rendering {done}/{total}",
                                 progress=min(95, progress))
            elif not denoising:
                # Includes the one-time ~15GB model download and MLX weight load.
                loading_progress = min(19, 6 + int(elapsed_now / 20))
                if loading_progress != last_loading_progress:
                    last_loading_progress = loading_progress
                    _set(jid, stage=(
                        "loading local FLUX 2 Klein (first use may be downloading)"),
                        progress=loading_progress)
        if proc.stdout:
            captured.extend(proc.stdout.read() or b"")
        proc_output = captured.decode("utf-8", "replace")
        _set_active_proc(jid, None)
        _check_canceled(jid)
        if proc.returncode != 0:
            tail = proc_output.strip().splitlines()[-8:]
            raise RuntimeError("local FLUX 2 Klein failed: " + " | ".join(tail))
        if not os.path.isfile(out_path):
            raise RuntimeError("local FLUX 2 Klein failed: output image not found")
        with Image.open(out_path) as image:
            width, height = image.size
        elapsed = round(time.time() - t0, 1)
        return out_name, int(width), int(height), elapsed
    finally:
        _set_active_proc(jid, None)
        for path in ref_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def _materialize_reference_images(jid, refs_raw, max_refs=3):
    """Decode base64 data URLs and persist normalized PNG references."""
    from PIL import Image

    out = []
    for i, raw in enumerate((refs_raw or [])[:max_refs], start=1):
        if not raw:
            continue
        b = _decode_data_url_image(raw)
        p = os.path.join(TMP_DIR, f"{jid}_href{i}.png")
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            img.save(p, format="PNG")
        except Exception as e:
            raise ValueError(f"invalid reference image #{i}: {e}") from e
        out.append(p)
    return out


def _combined_reference_guide(jid, ref_paths):
    """Place multiple identities on one neutral canvas for stable K=1 editing."""
    from PIL import Image, ImageOps

    count = len(ref_paths)
    if count < 2:
        return ref_paths[0] if ref_paths else ""
    canvas = Image.new("RGB", (1024, 1024), (242, 240, 236))
    panel_width = 1024 // count
    for index, path in enumerate(ref_paths):
        with Image.open(path) as source:
            subject = ImageOps.contain(
                source.convert("RGB"), (panel_width - 28, 968),
                method=Image.Resampling.LANCZOS,
            )
        x = index * panel_width + (panel_width - subject.width) // 2
        y = (1024 - subject.height) // 2
        canvas.paste(subject, (x, y))
    path = os.path.join(TMP_DIR, f"{jid}_combined_identity.png")
    canvas.save(path, format="PNG")
    return path


def _hidream_output_is_blank(arr) -> bool:
    """Detect HiDream's flat or coloured-noise generation failures.

    Failed outputs can contain lots of high-frequency texture, so ordinary
    pixel variance alone makes them look nonblank. Real illustrations retain
    strong structure after averaging into a 32x32 grid; collapsed outputs do
    not.
    """
    pixels = np.asarray(arr, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return True
    grey = pixels[:, :, :3].mean(axis=2)
    low, high = np.percentile(grey, [1, 99])
    height = grey.shape[0] - grey.shape[0] % 32
    width = grey.shape[1] - grey.shape[1] % 32
    if height < 32 or width < 32:
        structure_std = float(grey.std())
    else:
        grid = grey[:height, :width].reshape(
            32, height // 32, 32, width // 32).mean(axis=(1, 3))
        structure_std = float(grid.std())
    return (
        structure_std < 10.0
        and float(grey.std()) < 25.0
        and float(high - low) < 100.0
    )


def _load_overlay_source(source_filename="", source_data_url=""):
    from PIL import Image

    if source_filename:
        name = os.path.basename(str(source_filename))
        fp = os.path.join(OUT_DIR, name)
        ok_ext = (".png", ".jpg", ".jpeg", ".webp")
        if not (name.lower().endswith(ok_ext) and os.path.isfile(fp)):
            raise ValueError("source_filename must be an existing image in /files")
        return Image.open(fp).convert("RGBA")
    if source_data_url:
        raw = _decode_data_url_image(str(source_data_url))
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    raise ValueError("source image is required (source_filename or source_data_url)")


def _font(size, weight="regular"):
    from PIL import ImageFont
    
    # Avenir Next indices: 7=Regular, 5=Medium, 2=Demi Bold, 0=Bold, 8=Heavy
    idx_map = {
        "regular": 7,
        "medium": 5,
        "demi": 2,
        "bold": 0,
        "heavy": 8
    }
    
    try:
        idx = idx_map.get(weight, 7)
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", max(10, int(size)), index=idx)
    except Exception:
        pass
        
    try:
        # Fallback to Helvetica Neue if Avenir fails
        h_idx_map = {"regular": 0, "medium": 10, "demi": 10, "bold": 1, "heavy": 4}
        h_idx = h_idx_map.get(weight, 0)
        return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", max(10, int(size)), index=h_idx)
    except Exception:
        pass

    # Basic fallback
    is_bold = weight in ("bold", "heavy", "demi")
    bold_candidates = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
    reg_candidates = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    
    for p in (bold_candidates if is_bold else reg_candidates):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, max(10, int(size)))
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_lines(draw, text, font, max_w, max_lines=6):
    raw = " ".join(str(text or "").split())
    if not raw:
        return []
    words = raw.split(" ")
    lines, cur = [], words[0]
    for w in words[1:]:
        trial = cur + " " + w
        box = draw.textbbox((0, 0), trial, font=font)
        if (box[2] - box[0]) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    # No more '...' cutoff. Just let it break normally or show what fits.
    return lines


def _draw_block(draw, rect, fill=(0, 0, 0, 160), radius=26):
    draw.rounded_rectangle(rect, radius=radius, fill=fill)


def _draw_button(draw, x, y, text, min_w, h, align="left"):
    bf = _font(h * 0.46, weight="heavy")
    tb = draw.textbbox((0, 0), text, font=bf)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    w = max(min_w, tw + int(h * 0.9))
    if align == "right":
        x = x - (w - min_w)
    elif align == "center":
        x = x - (w - min_w) // 2
    _draw_block(draw, (x, y, x + w, y + h), fill=(255, 255, 255, 230), radius=max(8, h // 2))
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 1), text, font=bf, fill=(20, 24, 31, 255))


def _overlay_types():
    return [
        "cinematic_banner",
        "luxury_left_panel",
        "flash_sale_sign",
        "split_offer_card",
        "testimonial_quote",
        "feature_callouts",
    ]


def _apply_ad_overlay(img, overlay_type, data):
    from PIL import ImageDraw

    im = img.convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    pad = int(min(w, h) * 0.04)
    headline = str(data.get("headline") or "").strip()
    sub = str(data.get("subheadline") or "").strip()
    cta = str(data.get("cta") or "Shop now").strip()
    badge = str(data.get("badge") or "").strip()
    price = str(data.get("price") or "").strip()
    signs_raw = str(data.get("signs") or "").strip()
    signs = [s.strip() for s in re.split(r"[,\n|;]+", signs_raw) if s.strip()][:3]
    if not headline:
        raise ValueError("headline is required for overlay")

    title_font = _font(int(min(w, h) * 0.07), weight="heavy")
    sub_font = _font(int(min(w, h) * 0.034), weight="medium")
    chip_font = _font(int(min(w, h) * 0.028), weight="bold")

    if overlay_type == "cinematic_banner":
        block_h = int(h * 0.34)
        x1, y1, x2, y2 = pad, h - block_h - pad, w - pad, h - pad
        _draw_block(draw, (x1, y1, x2, y2), fill=(5, 8, 12, 182), radius=28)
        lines = _wrap_lines(draw, headline, title_font, int((x2 - x1) * 0.62), max_lines=6)
        draw.multiline_text((x1 + pad, y1 + pad), "\n".join(lines), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if sub:
            draw.multiline_text((x1 + pad, y1 + int(block_h * 0.58)), "\n".join(_wrap_lines(draw, sub, sub_font, int((x2 - x1) * 0.62), 5)), font=sub_font, fill=(226, 232, 240, 240), spacing=6)
        btn_w, btn_h = int((x2 - x1) * 0.22), int(block_h * 0.22)
        _draw_button(draw, x2 - btn_w - pad, y2 - btn_h - pad, cta or "Learn more", btn_w, btn_h, align="right")
        if badge:
            tb = draw.textbbox((0, 0), badge, font=chip_font)
            bw, bh = tb[2] - tb[0] + 26, tb[3] - tb[1] + 12
            _draw_block(draw, (x1 + pad, y1 - bh - 10, x1 + pad + bw, y1 - 10), fill=(239, 68, 68, 235), radius=12)
            draw.text((x1 + pad + 13, y1 - bh - 4), badge, font=chip_font, fill=(255, 255, 255, 255))

    elif overlay_type == "luxury_left_panel":
        x1, y1, x2, y2 = pad, pad, int(w * 0.46), h - pad
        _draw_block(draw, (x1, y1, x2, y2), fill=(8, 10, 14, 194), radius=30)
        lines = _wrap_lines(draw, headline, title_font, int((x2 - x1) * 0.84), 6)
        draw.multiline_text((x1 + pad, y1 + pad), "\n".join(lines), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if sub:
            draw.multiline_text((x1 + pad, y1 + int((y2 - y1) * 0.46)), "\n".join(_wrap_lines(draw, sub, sub_font, int((x2 - x1) * 0.84), 5)), font=sub_font, fill=(220, 226, 237, 240), spacing=6)
        _draw_button(draw, x1 + pad, y2 - int(h * 0.11), cta or "Shop now", int((x2 - x1) * 0.56), int(h * 0.075))
        if price:
            draw.text((x1 + pad, y2 - int(h * 0.18)), price, font=chip_font, fill=(251, 191, 36, 255))

    elif overlay_type == "flash_sale_sign":
        lines = _wrap_lines(draw, headline, title_font, int(w * 0.64), 6)
        _draw_block(draw, (pad, int(h * 0.62), w - pad, h - pad), fill=(0, 0, 0, 168), radius=24)
        draw.multiline_text((pad * 2, int(h * 0.65)), "\n".join(lines), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if sub:
            draw.multiline_text((pad * 2, int(h * 0.82)), "\n".join(_wrap_lines(draw, sub, sub_font, int(w * 0.64), 5)), font=sub_font, fill=(230, 236, 246, 245), spacing=6)
        sale_text = badge or "LIMITED OFFER"
        tb = draw.textbbox((0, 0), sale_text, font=chip_font)
        sw, sh = tb[2] - tb[0] + 34, tb[3] - tb[1] + 18
        sx2, sy1 = w - pad, pad
        sx1, sy2 = sx2 - sw, sy1 + sh
        _draw_block(draw, (sx1, sy1, sx2, sy2), fill=(220, 38, 38, 236), radius=14)
        draw.text((sx1 + 16, sy1 + 8), sale_text, font=chip_font, fill=(255, 255, 255, 255))
        _draw_button(draw, w - int(w * 0.22) - pad, h - int(h * 0.12), cta or "Buy now", int(w * 0.22), int(h * 0.075), align="right")

    elif overlay_type == "split_offer_card":
        panel_w = int(w * 0.42)
        x1, y1, x2, y2 = w - panel_w - pad, pad, w - pad, h - pad
        _draw_block(draw, (x1, y1, x2, y2), fill=(7, 9, 12, 198), radius=28)
        draw.multiline_text((x1 + pad, y1 + pad), "\n".join(_wrap_lines(draw, headline, title_font, int(panel_w * 0.8), 6)), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if price:
            draw.text((x1 + pad, y1 + int((y2 - y1) * 0.52)), price, font=_font(int(min(w, h) * 0.06), weight="heavy"), fill=(251, 191, 36, 255))
        if sub:
            draw.multiline_text((x1 + pad, y1 + int((y2 - y1) * 0.62)), "\n".join(_wrap_lines(draw, sub, sub_font, int(panel_w * 0.8), 5)), font=sub_font, fill=(228, 234, 243, 240), spacing=6)
        _draw_button(draw, x1 + pad, y2 - int(h * 0.11), cta or "Get deal", int(panel_w * 0.72), int(h * 0.075))
        if badge:
            draw.text((x1 + pad, y1 + int((y2 - y1) * 0.47)), badge, font=chip_font, fill=(129, 140, 248, 255))

    elif overlay_type == "testimonial_quote":
        bx1, by1, bx2, by2 = int(w * 0.08), int(h * 0.56), int(w * 0.92), h - pad
        _draw_block(draw, (bx1, by1, bx2, by2), fill=(15, 23, 42, 186), radius=30)
        quote = '“' + headline.strip().strip('"') + '”'
        draw.multiline_text((bx1 + pad, by1 + pad), "\n".join(_wrap_lines(draw, quote, title_font, int((bx2 - bx1) * 0.84), 3)), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if sub:
            draw.text((bx1 + pad, by2 - int(h * 0.14)), "— " + sub, font=sub_font, fill=(203, 213, 225, 245))
        _draw_button(draw, bx2 - int((bx2 - bx1) * 0.3) - pad, by2 - int(h * 0.1), cta or "Try now", int((bx2 - bx1) * 0.3), int(h * 0.07), align="right")
        if badge:
            draw.text((bx1 + pad, by1 - int(h * 0.05)), badge, font=chip_font, fill=(45, 212, 191, 255))

    else:  # feature_callouts
        _draw_block(draw, (pad, pad, w - pad, int(h * 0.28)), fill=(8, 10, 16, 186), radius=24)
        draw.multiline_text((pad * 2, pad * 2), "\n".join(_wrap_lines(draw, headline, title_font, int(w * 0.84), 6)), font=title_font, fill=(255, 255, 255, 255), spacing=8)
        if sub:
            draw.multiline_text((pad * 2, int(h * 0.19)), "\n".join(_wrap_lines(draw, sub, sub_font, int(w * 0.84), 5)), font=sub_font, fill=(230, 236, 246, 245), spacing=6)
        chip_items = signs or [badge or "Premium quality", price or "Fast shipping", cta or "Shop now"]
        chip_items = [x for x in chip_items if x][:3]
        cx = pad
        cy = h - int(h * 0.14)
        for text in chip_items:
            tbox = draw.textbbox((0, 0), text, font=chip_font)
            tw = (tbox[2] - tbox[0]) + 28
            th = (tbox[3] - tbox[1]) + 16
            if cx + tw > (w - pad):
                break
            _draw_block(draw, (cx, cy, cx + tw, cy + th), fill=(30, 64, 175, 220), radius=12)
            draw.text((cx + 14, cy + 8), text, font=chip_font, fill=(255, 255, 255, 255))
            cx += tw + 10

    return im


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
    """Stream `macmon pipe` JSON lines; cache latest watts + GPU temp."""
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
        time.sleep(5)


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
    if job.get("cancel_requested") or job.get("status") == "canceled":
        return
    opts = job["opts"]
    lease = None
    freed = []
    started_clock = time.monotonic()
    _set_active(jid)
    EVENT_LOG.event(
        "image.job_started", job_id=jid, engine=opts.get("engine"),
        width=opts.get("width"), height=opts.get("height"),
        steps=opts.get("steps"), seed=opts.get("seed"),
        reference_count=len(opts.get("reference_images") or []),
        reference_labels=opts.get("reference_labels") or [],
        reference_mapping_version=(
            "character-map-v2"
            if opts.get("engine") in (
                "nano_banana_2", "flux_2_klein_4b", "flux_2_klein_4b_local")
            else ""
        ),
        prompt_chars=len(opts.get("prompt") or ""),
    )
    try:
        _check_canceled(jid)
        # Cloud generation consumes no local model memory. Local engines share
        # one exclusive lease with video/song and use SSD as a cold model cache.
        if opts["engine"] not in ("grok", "nano_banana_2", "flux_2_klein_4b"):
            lease = acquire_lease(
                "image:" + opts["engine"], jid,
                waiting=lambda: _set(jid, stage="waiting for another local model job…", progress=1),
            )
            hidream_warm = opts["engine"] == "hidream" and eng.is_loaded()
            freed = unload_omlx_models(
                stage=lambda message, progress: _set(jid, stage=message, progress=progress))
            if opts["engine"] in ("kontext", "flux_2_klein_4b_local"):
                eng.unload()
            required_gb = (
                FLUX2_LOCAL_MIN_FREE_GB
                if opts["engine"] == "flux_2_klein_4b_local"
                else IMAGE_WARM_MIN_FREE_GB if hidream_warm
                else IMAGE_MIN_FREE_GB
            )
            memory = _memory_headroom(required_gb)
            opts["_available_memory_gb"] = memory["effective_gb"]
            EVENT_LOG.event(
                "image.memory_checked", job_id=jid, engine=opts.get("engine"),
                required_gb=required_gb, model_warm=hidream_warm,
                displaced_models=freed, **memory,
            )

            # If the warm model itself is leaving too little headroom, move it
            # back to the SSD cache and retry this as a clean cold start.
            if (hidream_warm and memory["effective_gb"]
                    and memory["effective_gb"] < required_gb):
                _set(jid, stage="freeing cached HiDream memory…", progress=2)
                eng.unload()
                hidream_warm = False
                required_gb = IMAGE_MIN_FREE_GB
                memory = _memory_headroom(required_gb)
                opts["_available_memory_gb"] = memory["effective_gb"]

            if (memory["effective_gb"]
                    and memory["effective_gb"] < required_gb):
                raise RuntimeError(
                    f"macOS can currently make about {memory['effective_gb']:.1f}GB "
                    f"available ({memory['reclaimable_gb']:.1f}GB immediately reclaimable); "
                    f"{required_gb:.0f}GB is required for a safe image job")
        if opts["engine"] == "grok":
            _set(jid, stage="preparing Grok Imagine", progress=4)
            name, w, h, elapsed = _run_grok_image(jid, opts)
            _set(jid, stage="saving", progress=97)
        elif opts["engine"] == "kontext":
            _set(jid, stage="kontext generating", progress=8)
            name, w, h, elapsed = _run_kontext(jid, opts)
            _set(jid, stage="saving", progress=97)
        elif opts["engine"] == "nano_banana_2":
            _set(jid, stage="preparing Nano Banana 2", progress=4)
            name, w, h, elapsed = _run_nano_banana(jid, opts)
            _set(jid, stage="saving", progress=97)
        elif opts["engine"] == "flux_2_klein_4b":
            _set(jid, stage="preparing FLUX 2 Klein", progress=4)
            name, w, h, elapsed = _run_flux_2_klein_4b(jid, opts)
            _set(jid, stage="saving", progress=97)
        elif opts["engine"] == "flux_2_klein_4b_local":
            _set(jid, stage="preparing local FLUX 2 Klein", progress=4)
            name, w, h, elapsed = _run_flux2_klein_local(jid, opts)
            _set(jid, stage="saving", progress=97)
        else:
            if not eng.weights_ready():
                raise RuntimeError("model weights are still downloading — "
                                   "try again once the download completes")
            if not eng.is_loaded():
                _set(jid, stage="loading model (first run, ~30s)", progress=2)
            _set(jid, stage="generating", progress=5)

            steps = opts["steps"]
            ref_paths = []
            combined_guide_path = ""
            try:
                if opts["engine"] == "hidream":
                    refs_raw = [x for x in (opts.get("reference_images") or []) if x]
                    if refs_raw:
                        _set(jid, stage="preparing reference images", progress=3)
                        ref_paths = _materialize_reference_images(jid, refs_raw)

                def prog(done, total):
                    _check_canceled(jid)
                    _set(jid, stage=f"denoising {done}/{total}",
                         progress=int(5 + 90 * done / max(1, total)))

                t0 = time.time()
                arr, w, h = eng.generate(
                    prompt=opts["prompt"], width=opts["width"], height=opts["height"],
                    steps=steps, seed=opts["seed"], snap=opts["snap"],
                    blend_seams=opts["blend_seams"], ref_images=ref_paths, progress=prog)
                reference_fallback = ""
                if _hidream_output_is_blank(arr):
                    if not ref_paths:
                        EVENT_LOG.event(
                            "hidream.collapsed_output", level="error", job_id=jid,
                            reference_count=0, action="reject",
                        )
                        raise RuntimeError(
                            "HiDream produced a blank image; the previous page image was kept")
                    if len(ref_paths) > 1:
                        _set(jid, stage="combining character references safely",
                             progress=4)
                        combined_guide_path = _combined_reference_guide(jid, ref_paths)
                        labels = [str(x) for x in
                                  (opts.get("reference_labels") or [])[:len(ref_paths)]]
                        positions = ["left", "right", "centre"]
                        identity_parts = [
                            f"{labels[index] if index < len(labels) and labels[index] else 'subject ' + str(index + 1)} on the {positions[index]}"
                            for index in range(len(ref_paths))
                        ]
                        guide_prompt = (
                            "The single visual reference is a side-by-side identity guide "
                            "containing distinct subjects: " + "; ".join(identity_parts)
                            + ". Depict every listed subject as a separate complete character. "
                              "Keep each subject's species, face, colours, clothing and permanent "
                              "accessories distinct; never merge or swap their features.\n"
                            + opts["prompt"]
                        )
                        _set(jid, stage="retrying with combined identity guide", progress=5)
                        EVENT_LOG.event(
                            "hidream.reference_collapse", level="warning", job_id=jid,
                            reference_count=len(ref_paths),
                            action="retry_combined_identity_guide", labels=labels,
                        )
                        arr, w, h = eng.generate(
                            prompt=guide_prompt, width=opts["width"],
                            height=opts["height"], steps=steps,
                            seed=opts["seed"] + 1, snap=opts["snap"],
                            blend_seams=opts["blend_seams"],
                            ref_images=[combined_guide_path], progress=prog)
                        reference_fallback = "combined_identity_guide"
                    if _hidream_output_is_blank(arr):
                        EVENT_LOG.event(
                            "hidream.reference_guide_collapse", level="warning",
                            job_id=jid, action="retry_without_references",
                        )
                        _set(jid, stage="reference guide was blank; retrying from descriptions",
                             progress=5)
                        arr, w, h = eng.generate(
                            prompt=opts["prompt"], width=opts["width"],
                            height=opts["height"], steps=steps,
                            seed=opts["seed"] + 2, snap=opts["snap"],
                            blend_seams=opts["blend_seams"], ref_images=[],
                            progress=prog)
                        reference_fallback = "description_only"
                        if _hidream_output_is_blank(arr):
                            EVENT_LOG.event(
                                "hidream.fallback_collapsed", level="error", job_id=jid,
                                action="reject",
                            )
                            raise RuntimeError(
                                "HiDream produced a blank image after safe retries; "
                                "the previous page image was kept")
                from PIL import Image
                _check_canceled(jid)
                _set(jid, stage="saving", progress=97)
                name = "img_" + uuid.uuid4().hex[:12] + ".png"
                path = os.path.join(OUT_DIR, name)
                Image.fromarray(arr).save(path)
                elapsed = round(time.time() - t0, 1)
            finally:
                for rp in ref_paths + ([combined_guide_path] if combined_guide_path else []):
                    try:
                        os.remove(rp)
                    except OSError:
                        pass

        _check_canceled(jid)
        result = {
            "filename": name,
            "url": f"/files/{name}",
            "width": int(w), "height": int(h),
            "steps": opts["steps"], "seed": opts["seed"],
            "prompt": opts["prompt"],
            "engine": opts.get("registry_id") or opts["engine"],
            "reference_images": len(opts.get("reference_images") or []),
            "reference_fallback": locals().get("reference_fallback", ""),
            "resolution": (opts.get("flux2_resolution")
                           if opts["engine"] in (
                               "flux_2_klein_4b", "flux_2_klein_4b_local")
                           else opts.get("nano_resolution")) or None,
            "aspect_ratio": (opts.get("flux2_aspect_ratio")
                             if opts["engine"] in (
                                 "flux_2_klein_4b", "flux_2_klein_4b_local")
                             else opts.get("nano_aspect_ratio")) or None,
            "output_format": (opts.get("flux2_output_format")
                              if opts["engine"] in (
                                  "flux_2_klein_4b", "flux_2_klein_4b_local")
                              else opts.get("nano_output_format")) or None,
            "seconds": elapsed,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        if IMAGE_RESTORE_OMLX_MODELS:
            reload_omlx_models(
                freed, stage=lambda message, progress: _set(
                    jid, stage=message, progress=progress))
        # By default, leave displaced chat models in their SSD cold cache.
        # oMLX loads them automatically when the next chat needs one. This
        # avoids holding a chat model and HiDream in unified memory together.
        freed = []
        _check_canceled(jid)
        _set(jid, status="done", stage="done", progress=100, result=result)
        EVENT_LOG.event(
            "image.job_completed", job_id=jid, engine=opts.get("engine"),
            duration_ms=round((time.monotonic() - started_clock) * 1000),
            filename=name, width=int(w), height=int(h), output_seconds=elapsed,
            reference_count=len(opts.get("reference_images") or []),
            reference_fallback=locals().get("reference_fallback", ""),
        )
    except JobCanceled:
        eng.unload()
        _set(jid, status="canceled", stage="stopped", progress=0,
             error=None, result=None)
        EVENT_LOG.event(
            "image.job_canceled", level="warning", job_id=jid,
            engine=opts.get("engine"),
            duration_ms=round((time.monotonic() - started_clock) * 1000),
        )
    except Exception as e:
        _set(jid, status="error", stage="error",
             error=f"{type(e).__name__}: {e}")
        EVENT_LOG.exception(
            "image.job_failed", e, job_id=jid, engine=opts.get("engine"),
            duration_ms=round((time.monotonic() - started_clock) * 1000),
        )
    finally:
        if freed:
            reload_omlx_models(freed)
        release_lease(lease)
        _clear_active(jid)


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
            return self._json(200, {
                "ok": True,
                "flux_2_klein_4b_local_ready": (
                    os.path.isfile(MFLUX_FLUX2_BIN)
                    and os.path.isfile(MFLUX_FLUX2_EDIT_BIN)),
                "flux_2_klein_4b_local_model": FLUX2_LOCAL_MODEL,
                "flux_2_klein_4b_local_quantize": FLUX2_LOCAL_QUANTIZE,
                "flux_2_klein_4b_local_precision": FLUX2_LOCAL_PRECISION,
                "flux_2_klein_4b_local_cache_gb": FLUX2_LOCAL_CACHE_GB,
                "flux_2_klein_4b_local_max_tokens": FLUX2_LOCAL_MAX_TOKENS,
                "memory_guard": {
                    "cold_start_gb": IMAGE_MIN_FREE_GB,
                    "warm_model_gb": IMAGE_WARM_MIN_FREE_GB,
                    "flux_2_klein_local_gb": FLUX2_LOCAL_MIN_FREE_GB,
                    "restore_chat_models": IMAGE_RESTORE_OMLX_MODELS,
                },
                "model_memory_lease": lease_status(),
                "grok_ready": bool(XAI_TOKEN),
                "grok_model": GROK_IMAGE_MODEL,
                "nano_banana_ready": bool(NB_TOKEN),
                "nano_banana_model": NB_MODEL,
                "flux_2_klein_4b_ready": bool(NB_TOKEN),
                "flux_2_klein_4b_model": FLUX2_MODEL,
                "queue": len(_work_q),
            })
        if path == "/info":
            registry_models = public_models("image", consumer="image_studio")
            return self._json(200, {
                "nano_banana_model": NB_MODEL,
                "flux_2_klein_4b_model": FLUX2_MODEL,
                "flux_2_klein_4b_local_model": FLUX2_LOCAL_MODEL,
                "flux_2_klein_4b_local_max_tokens": FLUX2_LOCAL_MAX_TOKENS,
                "registry": str(REGISTRY_PATH),
                "models": registry_models,
                "engines": [model["id"] for model in registry_models],
                "capabilities": {
                    model["id"]: model.get("capabilities", {})
                    for model in registry_models
                },
                "overlay": {
                    "enabled": True,
                    "types": _overlay_types(),
                },
                "presets": PRESETS, "default_steps": 28,
                "min_dim": MIN_DIM, "max_dim": MAX_DIM,
            })
        if path == "/library":
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = int((qs.get("limit") or ["80"])[0])
            except Exception:
                limit = 80
            limit = max(1, min(300, limit))
            rows = []
            for name in os.listdir(OUT_DIR):
                fp = os.path.join(OUT_DIR, name)
                if not (name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and os.path.isfile(fp)):
                    continue
                st = os.stat(fp)
                rows.append((st.st_mtime, name))
            rows.sort(reverse=True)
            items = []
            for ts, name in rows[:limit]:
                items.append({
                    "filename": name,
                    "url": f"/files/{name}",
                    "kind": "overlay" if name.startswith("ad_") else "image",
                    "created": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
                })
            return self._json(200, {"items": items})
        if path == "/status":
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            job = _get(jid)
            if not job:
                return self._json(404, {"error": "unknown job"})
            ret = {k: job.get(k) for k in ("status", "stage", "progress", "result", "error")}
            ret["prompt"] = job.get("opts", {}).get("prompt", "")
            return self._json(200, ret)
        if path == "/stats":
            return self._json(200, _sys_stats())
        if path.startswith("/files/"):
            return self._serve_file(os.path.basename(path))
        return self._json(404, {"error": "not found"})

    def _serve_file(self, name):
        fp = os.path.join(OUT_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if not (ext in (".png", ".jpg", ".jpeg", ".webp") and os.path.isfile(fp)):
            return self._json(404, {"error": "not found"})
        data = open(fp, "rb").read()
        ctype = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/cancel":
            return self._json(200, _cancel_all_images())
        if path == "/load":
            return self._json(410, {
                "error": "The retired HiDream model has been removed."
            })
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
            requested_engine = str(
                d.get("engine", "flux_2_klein_4b_local")
            ).strip().lower()
            registry_model = get_model(requested_engine)
            if not registry_model:
                return self._json(400, {"error": (
                    "unknown or disabled image model: " + requested_engine)})
            registry_id = registry_model["id"]
            engine = registry_model.get("service_engine") or registry_id
            ref_images = []
            raw_refs = d.get("reference_images")
            max_refs = max(0, int((registry_model.get("capabilities") or {}).get(
                "max_reference_images", 0)))
            if isinstance(raw_refs, list):
                ref_images = [str(x) for x in raw_refs if x]
            elif isinstance(raw_refs, str) and raw_refs.strip():
                ref_images = [raw_refs.strip()]
            if not ref_images:
                for key in (
                        "reference_image", "reference_image_2",
                        "reference_image_3", "reference_image_4"):
                    rv = d.get(key)
                    if isinstance(rv, str) and rv.strip():
                        ref_images.append(rv.strip())
            if len(ref_images) > max_refs:
                return self._json(400, {"error": (
                    f"{registry_model.get('label') or registry_id} accepts no more "
                    f"than {max_refs} reference images. Received {len(ref_images)}. "
                    "No image job or cloud request was started.")})
            ref_images = ref_images[:max_refs]
            raw_labels = d.get("reference_labels")
            reference_labels = ([str(x)[:120] for x in raw_labels[:max_refs]]
                                if isinstance(raw_labels, list) else [])
            nano_aspect = str(d.get("nano_aspect_ratio") or "1:1")
            if nano_aspect not in NB_ASPECTS:
                nano_aspect = "1:1"
            nano_fmt = str(d.get("nano_output_format") or "png").lower()
            if nano_fmt not in NB_OUTPUT_FORMATS:
                nano_fmt = "png"
            flux2_aspect = str(d.get("flux2_aspect_ratio") or nano_aspect)
            if flux2_aspect not in FLUX2_ASPECTS:
                flux2_aspect = "1:1"
            flux2_resolution = str(d.get("flux2_resolution") or "1")
            if flux2_resolution not in FLUX2_RESOLUTIONS:
                flux2_resolution = "1"
            flux2_fmt = str(d.get("flux2_output_format") or "png").lower()
            if flux2_fmt not in FLUX2_OUTPUT_FORMATS:
                flux2_fmt = "png"
            opts = {
                "engine": engine,
                "registry_id": registry_id,
                "prompt": (prompt[:6000] if engine in (
                    "grok", "flux_2_klein_4b", "flux_2_klein_4b_local")
                    else prompt[:2000]),
                "width": w, "height": h,
                "steps": (FLUX2_LOCAL_STEPS if engine == "flux_2_klein_4b_local"
                          else max(4, min(50, int(d.get("steps", 28))))),
                "seed": int(d.get("seed", 32)),
                "snap": bool(d.get("snap", True)),
                "blend_seams": max(0, min(4, int(d.get("blend_seams", 0)))),
                "guidance": float(d.get("guidance", 2.8)),
                "reference_image": d.get("reference_image", ""),
                "reference_images": ref_images,
                "reference_labels": reference_labels,
                "nano_resolution": NB_RESOLUTION,
                "nano_aspect_ratio": nano_aspect,
                "nano_output_format": nano_fmt,
                "nano_google_search": bool(d.get("nano_google_search", False)),
                "nano_image_search": bool(d.get("nano_image_search", False)),
                "flux2_resolution": flux2_resolution,
                "flux2_aspect_ratio": flux2_aspect,
                "flux2_output_format": flux2_fmt,
                "flux2_output_quality": max(0, min(100, int(
                    d.get("flux2_output_quality", 100)))),
                "flux2_go_fast": bool(d.get("flux2_go_fast", False)),
                "flux2_disable_safety_checker": bool(
                    d.get("flux2_disable_safety_checker", False)),
                "flux2_max_tokens": max(512, min(2048, int(
                    d.get("flux2_max_tokens") or FLUX2_LOCAL_MAX_TOKENS))),
            }
            if engine == "kontext" and not opts["reference_image"]:
                return self._json(400, {"error": "reference_image is required for kontext"})
            if engine == "grok" and not XAI_TOKEN:
                return self._json(400, {"error": "XAI_API_KEY/GROK_API_KEY is not set — add it to " + NB_ENV_FILE})
            if engine in ("nano_banana_2", "flux_2_klein_4b") and not NB_TOKEN:
                return self._json(400, {"error": "REPLICATE_API_TOKEN is not set — add it to " + NB_ENV_FILE})
            jid = "img_" + uuid.uuid4().hex[:12]
            _set(jid, status="running", stage="queued", progress=0,
                 result=None, error=None, cancel_requested=False, opts=opts)
            with _work_cv:
                _work_q.append(jid)
                queue_depth = len(_work_q)
                _work_cv.notify()
            EVENT_LOG.event(
                "image.job_queued", job_id=jid, engine=registry_id,
                service_engine=engine, width=w, height=h,
                steps=opts["steps"], seed=opts["seed"],
                reference_count=len(ref_images), queue_depth=queue_depth,
                reference_labels=reference_labels,
                reference_mapping_version=(
                    "character-map-v2"
                    if engine in ("flux_2_klein_4b", "flux_2_klein_4b_local")
                    else ""
                ),
                prompt_chars=len(prompt),
            )
            return self._json(200, {"ok": True, "job_id": jid})
        if path == "/overlay":
            d = self._body()
            overlay_type = str(d.get("overlay_type") or "cinematic_banner").strip().lower()
            if overlay_type not in _overlay_types():
                return self._json(400, {"error": "unsupported overlay_type"})
            try:
                src = _load_overlay_source(
                    source_filename=d.get("source_filename", ""),
                    source_data_url=d.get("source_data_url", ""),
                )
                composed = _apply_ad_overlay(src, overlay_type, {
                    "headline": str(d.get("headline") or "")[:180],
                    "subheadline": str(d.get("subheadline") or "")[:260],
                    "cta": str(d.get("cta") or "")[:60],
                    "badge": str(d.get("badge") or "")[:80],
                    "price": str(d.get("price") or "")[:60],
                    "signs": str(d.get("signs") or "")[:260],
                })
                out_name = "ad_" + uuid.uuid4().hex[:12] + ".png"
                out_path = os.path.join(OUT_DIR, out_name)
                composed.save(out_path, format="PNG")
                w, h = composed.size
                return self._json(200, {
                    "ok": True,
                    "result": {
                        "filename": out_name,
                        "url": f"/files/{out_name}",
                        "width": int(w),
                        "height": int(h),
                        "overlay_type": overlay_type,
                        "created": datetime.now().isoformat(timespec="seconds"),
                    },
                })
            except Exception as e:
                return self._json(400, {"error": f"{type(e).__name__}: {e}"})
        return self._json(404, {"error": "not found"})


def main():
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_power_worker, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-image ready at http://{HOST}:{PORT}", flush=True)
    EVENT_LOG.event("service.started", host=HOST, port=PORT)
    srv.serve_forever()


if __name__ == "__main__":
    main()
