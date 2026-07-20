#!/usr/bin/env python3
"""omlx-image — local HiDream-O1 text-to-image microservice (port 8400).

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
import subprocess
import threading
import time
import uuid
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import hidream_engine as eng

COMMON_DIR = os.environ.get("OMLX_COMMON_DIR", "/Users/joebains/omlx-common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from model_memory_coordinator import (
    acquire_lease, available_memory_gb, lease_status, release_lease,
    reload_omlx_models, unload_omlx_models,
)

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
KONTEXT_LOW_RAM = os.environ.get("KONTEXT_LOW_RAM", "1") not in ("0", "false", "no")
KONTEXT_CACHE_GB = float(os.environ.get("KONTEXT_MLX_CACHE_GB", "1"))
IMAGE_MIN_FREE_GB = float(os.environ.get("IMAGE_MIN_FREE_GB", "18"))


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

        inp = {
            "prompt": opts["prompt"],
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

        status = pred.get("status")
        deadline = time.time() + 900
        while status not in ("succeeded", "failed", "canceled"):
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
    if KONTEXT_LOW_RAM:
        cmd.append("--low-ram")
    cmd.extend(["--mlx-cache-limit-gb", str(KONTEXT_CACHE_GB)])
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


def _materialize_reference_images(jid, refs_raw):
    """Decode up to 3 base64 data URLs and persist normalized PNG refs."""
    from PIL import Image

    out = []
    for i, raw in enumerate((refs_raw or [])[:3], start=1):
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
    opts = job["opts"]
    lease = None
    freed = []
    try:
        # Cloud generation consumes no local model memory. Local engines share
        # one exclusive lease with video/song and use SSD as a cold model cache.
        if opts["engine"] != "nano_banana_2":
            lease = acquire_lease(
                "image:" + opts["engine"], jid,
                waiting=lambda: _set(jid, stage="waiting for another local model job…", progress=1),
            )
            freed = unload_omlx_models(
                stage=lambda message, progress: _set(jid, stage=message, progress=progress))
            if opts["engine"] == "kontext":
                eng.unload()
            free_gb = available_memory_gb()
            if free_gb and free_gb < IMAGE_MIN_FREE_GB:
                raise RuntimeError(
                    f"only {free_gb:.1f}GB reclaimable memory is available after unloading "
                    f"other models; {IMAGE_MIN_FREE_GB:.0f}GB is required for a safe image job")
        if opts["engine"] == "kontext":
            _set(jid, stage="kontext generating", progress=8)
            name, w, h, elapsed = _run_kontext(jid, opts)
            _set(jid, stage="saving", progress=97)
        elif opts["engine"] == "nano_banana_2":
            _set(jid, stage="preparing Nano Banana 2", progress=4)
            name, w, h, elapsed = _run_nano_banana(jid, opts)
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
            try:
                if opts["engine"] == "hidream":
                    refs_raw = [x for x in (opts.get("reference_images") or []) if x]
                    if refs_raw:
                        _set(jid, stage="preparing reference images", progress=3)
                        ref_paths = _materialize_reference_images(jid, refs_raw)

                def prog(done, total):
                    _set(jid, stage=f"denoising {done}/{total}",
                         progress=int(5 + 90 * done / max(1, total)))

                t0 = time.time()
                arr, w, h = eng.generate(
                    prompt=opts["prompt"], width=opts["width"], height=opts["height"],
                    steps=steps, seed=opts["seed"], snap=opts["snap"],
                    blend_seams=opts["blend_seams"], ref_images=ref_paths, progress=prog)
                from PIL import Image
                _set(jid, stage="saving", progress=97)
                name = "img_" + uuid.uuid4().hex[:12] + ".png"
                path = os.path.join(OUT_DIR, name)
                Image.fromarray(arr).save(path)
                elapsed = round(time.time() - t0, 1)
            finally:
                for rp in ref_paths:
                    try:
                        os.remove(rp)
                    except OSError:
                        pass

        result = {
            "filename": name,
            "url": f"/files/{name}",
            "width": int(w), "height": int(h),
            "steps": opts["steps"], "seed": opts["seed"],
            "prompt": opts["prompt"],
            "engine": opts["engine"],
            "reference_images": len(opts.get("reference_images") or []),
            "resolution": opts.get("nano_resolution") or None,
            "aspect_ratio": opts.get("nano_aspect_ratio") or None,
            "output_format": opts.get("nano_output_format") or None,
            "seconds": elapsed,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        reload_omlx_models(
            freed, stage=lambda message, progress: _set(jid, stage=message, progress=progress))
        freed = []
        _set(jid, status="done", stage="done", progress=100, result=result)
    except Exception as e:
        _set(jid, status="error", stage="error",
             error=f"{type(e).__name__}: {e}")
    finally:
        if freed:
            reload_omlx_models(freed)
        release_lease(lease)


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
                "kontext_low_ram": KONTEXT_LOW_RAM,
                "kontext_mlx_cache_gb": KONTEXT_CACHE_GB,
                "model_memory_lease": lease_status(),
                "nano_banana_ready": bool(NB_TOKEN),
                "nano_banana_model": NB_MODEL,
                "queue": len(_work_q),
            })
        if path == "/info":
            return self._json(200, {
                "model": "HiDream-O1-Image-Dev (MLX bf16)",
                "kontext_model": KONTEXT_MODEL,
                "nano_banana_model": NB_MODEL,
                "engines": ["hidream", "kontext", "nano_banana_2"],
                "capabilities": {
                    "hidream": {
                        "text_to_image": True,
                        "instruction_edit": True,
                        "multi_reference": True,
                        "max_reference_images": 3,
                    },
                    "kontext": {
                        "reference_edit": True,
                        "max_reference_images": 1,
                    },
                    "nano_banana_2": {
                        "text_to_image": True,
                        "multi_reference": True,
                        "max_reference_images": 14,
                        "native_text_rendering": True,
                        "resolutions": [NB_RESOLUTION],
                        "aspect_ratios": NB_ASPECTS,
                        "output_formats": NB_OUTPUT_FORMATS,
                        "google_search": True,
                        "image_search": True,
                    },
                },
                "overlay": {
                    "enabled": True,
                    "types": _overlay_types(),
                },
                "presets": PRESETS, "default_steps": 28,
                "min_dim": MIN_DIM, "max_dim": MAX_DIM,
                "weights_ready": eng.weights_ready(),
                "model_loaded": eng.is_loaded(),
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
            if engine not in ("hidream", "kontext", "nano_banana_2"):
                return self._json(400, {"error": "engine must be hidream, kontext, or nano_banana_2"})
            ref_images = []
            raw_refs = d.get("reference_images")
            max_refs = 14 if engine == "nano_banana_2" else 3
            if isinstance(raw_refs, list):
                ref_images = [str(x) for x in raw_refs if x][:max_refs]
            elif isinstance(raw_refs, str) and raw_refs.strip():
                ref_images = [raw_refs.strip()]
            if not ref_images:
                for key in ("reference_image", "reference_image_2", "reference_image_3"):
                    rv = d.get(key)
                    if isinstance(rv, str) and rv.strip():
                        ref_images.append(rv.strip())
                ref_images = ref_images[:max_refs]
            nano_aspect = str(d.get("nano_aspect_ratio") or "1:1")
            if nano_aspect not in NB_ASPECTS:
                nano_aspect = "1:1"
            nano_fmt = str(d.get("nano_output_format") or "png").lower()
            if nano_fmt not in NB_OUTPUT_FORMATS:
                nano_fmt = "png"
            opts = {
                "engine": engine,
                "prompt": prompt[:2000],
                "width": w, "height": h,
                "steps": max(4, min(50, int(d.get("steps", 28)))),
                "seed": int(d.get("seed", 32)),
                "snap": bool(d.get("snap", True)),
                "blend_seams": max(0, min(4, int(d.get("blend_seams", 0)))),
                "guidance": float(d.get("guidance", 2.8)),
                "reference_image": d.get("reference_image", ""),
                "reference_images": ref_images,
                "nano_resolution": NB_RESOLUTION,
                "nano_aspect_ratio": nano_aspect,
                "nano_output_format": nano_fmt,
                "nano_google_search": bool(d.get("nano_google_search", False)),
                "nano_image_search": bool(d.get("nano_image_search", False)),
            }
            if engine == "kontext" and not opts["reference_image"]:
                return self._json(400, {"error": "reference_image is required for kontext"})
            if engine == "nano_banana_2" and not NB_TOKEN:
                return self._json(400, {"error": "REPLICATE_API_TOKEN is not set — add it to " + NB_ENV_FILE})
            jid = "img_" + uuid.uuid4().hex[:12]
            _set(jid, status="running", stage="queued", progress=0,
                 result=None, error=None, opts=opts)
            with _work_cv:
                _work_q.append(jid)
                _work_cv.notify()
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
    print(f"omlx-image ready at http://{HOST}:{PORT} "
          f"(weights_ready={eng.weights_ready()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
