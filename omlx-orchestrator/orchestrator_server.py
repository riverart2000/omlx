#!/usr/bin/env python3
# ============================================================================
# omlx-orchestrator  —  local workflow coordinator on :8700
#
# Loads mlx-community/Orchestrator-8B-6bit (NVIDIA Nemotron ToolOrchestra) and
# uses it as a tool-calling "brain" that coordinates the rest of the local
# media stack:
#
#     memory (:8300)   image (:8400)   tts (:8200)   video (:8500)
#
# You give it a high-level GOAL (e.g. "make a 15s vertical talking-head promo
# for my app with a voiceover") and it decomposes the goal, calls the right
# sidecars in the right order, feeds each result into the next, and returns the
# finished artifacts.
#
# Design mirrors the other sidecars: stdlib BaseHTTPRequestHandler, /health,
# async job model (POST /orchestrate -> job_id, poll GET /status?id=), and a
# launchd KeepAlive agent. The Orchestrator model is loaded lazily and unloaded
# before heavy media steps (reload is ~0.4s from cache) to protect RAM.
#
#   Endpoints:
#     GET  /health              service + model status
#     GET  /info                available tools + engines
#     POST /orchestrate         {"goal": "..."} -> {"job_id": "orch_xxx"}
#     GET  /status?id=<job_id>  live steps + final result
#     GET  /files/<name>        proxy fetch an artifact from a sidecar
#     GET  /                     minimal standalone debug UI
# ============================================================================
import base64
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMON_DIR = os.environ.get("OMLX_COMMON_DIR", "/Users/joebains/omlx-common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from model_registry import get_model, public_models

IMAGE_MODELS = public_models("image", consumer="orchestrator")
IMAGE_MODEL_IDS = [model["id"] for model in IMAGE_MODELS]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = int(os.environ.get("ORCH_PORT", "8700"))
MODEL_PATH = os.environ.get(
    "ORCH_MODEL",
    "/Users/joebains/.omlx/models/mlx-community/Orchestrator-8B-6bit",
)

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8200")
MEMORY_URL = os.environ.get("MEMORY_URL", "http://127.0.0.1:8300")
IMAGE_URL = os.environ.get("IMAGE_URL", "http://127.0.0.1:8400")
VIDEO_URL = os.environ.get("VIDEO_URL", "http://127.0.0.1:8500")
OMLX_URL = os.environ.get("OMLX_URL", "http://127.0.0.1:8000")
QWEN_ANIMATION_MODEL = os.environ.get(
    "QWEN_ANIMATION_MODEL",
    "porschefreak--Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-mlx-6Bit",
)
FLOWAGENT_URL = os.environ.get("FLOWAGENT_URL", "http://127.0.0.1:3000")
FLOWAGENT_SH = os.environ.get("FLOWAGENT_SH", "/Users/joebains/FlowAgent/flowagent.sh")
# Only these socials are supervised here; the rest are handled elsewhere.
FA_PLATFORMS = [p.strip() for p in os.environ.get(
    "FA_PLATFORMS", "medium,quora,flipboard,blogger,substack").split(",") if p.strip()]

MAX_ITERS = int(os.environ.get("ORCH_MAX_ITERS", "16"))
MAX_TOKENS = int(os.environ.get("ORCH_MAX_TOKENS", "1024"))
UNLOAD_BEFORE_MEDIA = os.environ.get("ORCH_UNLOAD_BEFORE_MEDIA", "1") == "1"
# how long a single async media job may run before we give up polling
MEDIA_POLL_TIMEOUT = int(os.environ.get("ORCH_MEDIA_TIMEOUT", "1800"))

# ---------------------------------------------------------------------------
# Lazy MLX model manager
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_model_lock = threading.Lock()
_load_error = None


def _load_model():
    global _model, _tokenizer, _load_error
    with _model_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            from mlx_lm import load
            _model, _tokenizer = load(MODEL_PATH)
            _load_error = None
        except Exception as e:  # pragma: no cover
            _load_error = f"{type(e).__name__}: {e}"
            raise
    return _model, _tokenizer


def _unload_model():
    """Free the orchestrator model so heavy media gen has RAM headroom.
    Reload from cache is ~0.4s so this is cheap between planning turns."""
    global _model, _tokenizer
    with _model_lock:
        if _model is None:
            return
        _model = None
        _tokenizer = None
    try:
        import mlx.core as mx
        try:
            mx.clear_cache()
        except Exception:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def _generate(messages, tools, max_tokens=None, temperature=0.6):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = _load_model()
    prompt = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temperature, top_p=0.95)
    text = generate(
        model, tok, prompt=prompt, max_tokens=max_tokens or MAX_TOKENS,
        sampler=sampler, verbose=False,
    )
    return text


# ---------------------------------------------------------------------------
# Sidecar HTTP helpers
# ---------------------------------------------------------------------------
def _http_json(method, url, payload=None, timeout=120):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"_raw": raw.decode("utf-8", "replace")}


def _get_json(url, timeout=30):
    return _http_json("GET", url, None, timeout)


def _post_json(url, payload, timeout=120):
    return _http_json("POST", url, payload, timeout)


_CAMERA_MOVES = {
    "push_in", "pull_out", "pan_left", "pan_right", "tilt_up",
    "tilt_down", "arc_left", "arc_right", "bounce_soft", "hold",
}
_CAPTION_MOTIONS = {"rise", "glide_left", "glide_right", "pop_soft"}
_SHOT_TRANSITIONS = {"dissolve", "fade", "smoothleft", "circleopen", "fadeblack"}
_SOUND_KINDS = {"ambience", "foley", "impact", "nature", "comedy", "transition"}


def _number(value, default, low, high):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _qwen_animation_plan(payload):
    """Use the local Qwen3.6 model as a constrained Blender shot director."""
    raw_scenes = payload.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("animation-plan requires scenes")
    scenes = []
    for row in raw_scenes[:140]:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip():
            continue
        scenes.append({
            "id": str(row["id"])[:80],
            "heading": str(row.get("heading") or "")[:300],
            "text": str(row.get("text") or "")[:3000],
            "image_prompt": str(row.get("image_prompt") or "")[:2500],
            "seconds": _number(row.get("seconds"), 10, 1, 180),
        })
    if not scenes:
        raise ValueError("animation-plan did not contain valid scenes")
    # Keep long books reliable: one enormous JSON answer is more likely to be
    # truncated or malformed.  Qwen stays resident across these small batches
    # and is released once the complete plan has been assembled.
    if len(scenes) > 6:
        combined = []
        release_model = bool(payload.get("release_model", True))
        try:
            for start in range(0, len(scenes), 6):
                batch_payload = dict(payload)
                batch_payload["scenes"] = scenes[start:start + 6]
                batch_payload["release_model"] = False
                combined.extend(_qwen_animation_plan(
                    batch_payload)["scenes"])
        finally:
            if release_model:
                try:
                    _post_json(
                        f"{OMLX_URL}/v1/models/{urllib.parse.quote(QWEN_ANIMATION_MODEL, safe='')}/unload",
                        {}, timeout=120)
                except Exception:
                    pass
        return {"model": QWEN_ANIMATION_MODEL, "scenes": combined}
    context = {
        "title": str(payload.get("title") or "")[:300],
        "audience": str(payload.get("audience") or "")[:120],
        "tone": str(payload.get("tone") or "")[:300],
        "energy": str(payload.get("energy") or "balanced")[:40],
        "aspect": str(payload.get("aspect") or "16:9")[:20],
        "scenes": scenes,
    }
    prompt = """Act as the animation director for a premium, lively children's
storybook film made from full-screen flat illustrations. Design purposeful
camera choreography for every supplied scene. Avoid repeating the same move on
adjacent scenes. Match each story action and emotion. Keep important subjects
inside safe crop limits and captions readable. Treat every visible character's
head and face as protected: the main head should normally remain visible in
every beat. Prefer starting wide enough to establish the complete subject,
then push toward a face or pan gently from one named character's face to
another. Never direct a lingering body-only crop unless the story explicitly
requires a close-up of an object, hands or feet. Use 2 to 4 camera beats per
scene; 'at' is a fraction from 0.0 to 0.92 of that scene. focus_x/focus_y are
normalised image positions and should describe the likely head or face position
from the supplied composition. zoom must be 1.02 to 1.12 when a person or
character is present, or at most 1.16 for scenery. rotation is degrees from
-1.2 to 1.2. The first beat must start at 0.0. Prefer energetic changes for
action, gentle holds for emotion, and reveal/pull-out moves for discoveries.

Return exactly one JSON object with this shape and no extra keys or prose:
{"scenes":[{"id":"exact input id","pace":"gentle|balanced|lively",
"caption_motion":"rise|glide_left|glide_right|pop_soft",
"transition":"dissolve|fade|smoothleft|circleopen|fadeblack",
"beats":[{"at":0.0,"move":"push_in|pull_out|pan_left|pan_right|tilt_up|tilt_down|arc_left|arc_right|bounce_soft|hold","focus_x":0.5,"focus_y":0.5,"zoom":1.04,"rotation":0.0}],
"reason":"brief scene-specific direction"}]}

Book context:
""" + json.dumps(context, ensure_ascii=False)
    request = {
        "model": QWEN_ANIMATION_MODEL,
        "messages": [
            {"role": "system", "content": (
                "Return valid JSON only. You direct using the allowed values; "
                "never emit Python, bpy calls, shell commands, or file paths.")},
            {"role": "user", "content": prompt},
        ],
        "temperature": .62,
        "max_tokens": min(10000, max(1400, len(scenes) * 330)),
        "response_format": {"type": "json_object"},
    }
    release_model = bool(payload.get("release_model", True))
    generation_error = ""
    try:
        # The director is a much larger oMLX model.  Do not keep this
        # service's general-purpose model resident beside it.
        _unload_model()
        try:
            response = _post_json(
                f"{OMLX_URL}/v1/chat/completions", request, timeout=900)
            content = (((response.get("choices") or [{}])[0].get("message") or {})
                       .get("content"))
            generated = (content if isinstance(content, dict)
                         else json.loads(str(content or "")))
        except Exception as exc:
            # Direction is an enhancement, not a reason to lose an otherwise
            # completed narrated video. The normaliser below supplies gentle
            # deterministic moves for this batch while later batches continue.
            generation_error = str(exc)[:400]
            generated = {"scenes": []}
    finally:
        if release_model:
            try:
                _post_json(
                    f"{OMLX_URL}/v1/models/{urllib.parse.quote(QWEN_ANIMATION_MODEL, safe='')}/unload",
                    {}, timeout=120)
            except Exception:
                pass
    rows = generated.get("scenes") if isinstance(generated, dict) else None
    if not isinstance(rows, list):
        generation_error = generation_error or "Qwen omitted the scenes list"
        rows = []
    by_id = {str(row.get("id") or ""): row for row in rows if isinstance(row, dict)}
    result = []
    fallback_moves = ("push_in", "pan_right", "pull_out", "pan_left")
    for scene_index, scene in enumerate(scenes):
        # Preserve every usable part of Qwen's direction. A missing row or one
        # short beat should not discard the complete soundtrack or abort a
        # long video render; fill only the missing camera direction with a
        # conservative storybook move.
        row = by_id.get(scene["id"]) or {}
        beats = []
        for beat in (row.get("beats") or [])[:4]:
            if not isinstance(beat, dict):
                continue
            move = str(beat.get("move") or "hold")
            beats.append({
                "at": _number(beat.get("at"), 0, 0, .92),
                "move": move if move in _CAMERA_MOVES else "hold",
                "focus_x": _number(beat.get("focus_x"), .5, .16, .84),
                "focus_y": _number(beat.get("focus_y"), .5, .16, .84),
                "zoom": _number(beat.get("zoom"), 1.05, 1.02, 1.16),
                "rotation": _number(beat.get("rotation"), 0, -1.2, 1.2),
            })
        used_fallback = len(beats) < 2
        if not beats:
            beats.append({
                "at": 0.0, "move": "hold", "focus_x": .5,
                "focus_y": .5, "zoom": 1.03, "rotation": 0.0,
            })
        if len(beats) == 1:
            first = beats[0]
            fallback_move = fallback_moves[scene_index % len(fallback_moves)]
            if fallback_move == first.get("move"):
                fallback_move = "pull_out" if fallback_move == "push_in" else "push_in"
            beats.append({
                "at": .68, "move": fallback_move,
                "focus_x": _number(first.get("focus_x"), .5, .16, .84),
                "focus_y": _number(first.get("focus_y"), .5, .16, .84),
                "zoom": max(1.04, min(1.12,
                    _number(first.get("zoom"), 1.04, 1.02, 1.16) + .035)),
                "rotation": 0.0,
            })
        beats.sort(key=lambda item: item["at"])
        beats[0]["at"] = 0.0
        caption_motion = str(row.get("caption_motion") or "rise")
        transition = str(row.get("transition") or "dissolve")
        pace = str(row.get("pace") or "balanced")
        result.append({
            "id": scene["id"],
            "pace": pace if pace in {"gentle", "balanced", "lively"} else "balanced",
            "caption_motion": (caption_motion if caption_motion in _CAPTION_MOTIONS
                               else "rise"),
            "transition": transition if transition in _SHOT_TRANSITIONS else "dissolve",
            "beats": beats,
            "reason": str(row.get("reason") or (
                "A restrained fallback camera move completed incomplete Qwen direction"
                + (f" ({generation_error})" if generation_error else "")
                if used_fallback else ""))[:500],
        })
    return {"model": QWEN_ANIMATION_MODEL, "scenes": result}


def _qwen_sound_plan(payload):
    """Use local Qwen as a restrained children's-story sound designer."""
    raw_scenes = payload.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("sound-plan requires scenes")
    scenes = []
    for row in raw_scenes[:140]:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip():
            continue
        scenes.append({
            "id": str(row["id"])[:80],
            "heading": str(row.get("heading") or "")[:300],
            "text": str(row.get("text") or "")[:3500],
            "image_prompt": str(row.get("image_prompt") or "")[:2200],
            "seconds": _number(row.get("seconds"), 10, 1, 180),
        })
    if not scenes:
        raise ValueError("sound-plan did not contain valid scenes")
    # This particular Qwen release is excellent at an individual sound brief
    # but sometimes returns only the first row from a multi-scene JSON request.
    # Keep it resident and direct one short scene at a time for completeness.
    if len(scenes) > 1:
        combined = []
        warnings = []
        release_model = bool(payload.get("release_model", True))
        try:
            for scene in scenes:
                batch_payload = dict(payload)
                batch_payload["scenes"] = [scene]
                batch_payload["release_model"] = False
                try:
                    combined.extend(_qwen_sound_plan(batch_payload)["scenes"])
                except Exception as first_exc:
                    # A long book should not lose every completed sound cue
                    # because one model response was malformed. Give that page
                    # one lower-temperature retry, then preserve it as a quiet
                    # scene that the editor can fill manually if needed.
                    batch_payload["_sound_retry"] = True
                    try:
                        combined.extend(_qwen_sound_plan(batch_payload)["scenes"])
                    except Exception as retry_exc:
                        combined.append({
                            "id": scene["id"], "heading": scene["heading"],
                            "seconds": round(scene["seconds"], 2), "cues": [],
                        })
                        warnings.append({
                            "scene_id": scene["id"],
                            "message": (f"Qwen returned an unusable sound plan twice; "
                                        f"the scene was left quiet: {retry_exc}"),
                            "first_error": str(first_exc)[:300],
                        })
        finally:
            if release_model:
                try:
                    _post_json(
                        f"{OMLX_URL}/v1/models/{urllib.parse.quote(QWEN_ANIMATION_MODEL, safe='')}/unload",
                        {}, timeout=120)
                except Exception:
                    pass
        return {"model": QWEN_ANIMATION_MODEL, "scenes": combined,
                "warnings": warnings}
    intensity = str(payload.get("intensity") or "balanced")
    if intensity not in {"none", "gentle", "balanced", "lively", "cinematic"}:
        intensity = "balanced"
    context = {
        "title": str(payload.get("title") or "")[:300],
        "audience": str(payload.get("audience") or "")[:120],
        "tone": str(payload.get("tone") or "")[:300],
        "intensity": intensity,
        "scenes": scenes,
    }
    prompt = """Act as the supervising sound editor for a professionally produced
children's story video. Plan a restrained, clear, child-friendly soundtrack
that supports the final narration instead of competing with it. Use only sounds
that are literally justified by the page words or visible action. Do not invent
off-screen events. Normally choose 0-1 cue for gentle or balanced, and no more
than 2 for lively or cinematic. The cover needs at most one very subtle cue.
Quiet scenes should have no cue. Prefer silence over a weak decorative sound,
and prefer one meaningful action sound over several layers.
Every non-cover scene containing an explicit audible physical event—footsteps,
knocking, bouncing, impact, doors, water, wind, animals, machinery, breaking,
rustling, applause or a similar concrete action—must receive a matching event
cue unless intensity is none. Do not replace that literal event with generic
ambience. Do not amplify silent or nearly silent micro-actions such as a blink,
a smile, a tiny paw touching one speck of snow, or a small hand movement into
unnatural foley. Reproduce every input scene id once and in the original order.

Each prompt is sent to a sound-effects-only generator. Describe one isolated,
clean, gentle sound or ambience in concrete acoustic terms. Every prompt must
say child-friendly, non-startling, soft onset, no sudden loud peak and natural
real-world scale. Never request narration, dialogue, spoken words, singing,
melody, a musical score or copyrighted media.
Use loop=true only for steady ambience. 'at' is the fraction 0.0-0.96 through
the spoken scene when the sound begins. Keep event effects brief; ambience may
last longer. Volume should normally be 0.10-0.18 and never exceed 0.28, so
speech remains dominant. The trigger quotes a
short phrase or describes the exact story moment used for timing.

Return exactly one JSON object and no prose:
{"scenes":[{"id":"exact input id","cues":[{"kind":"ambience|foley|impact|nature|comedy|transition","prompt":"sound-only generation prompt","trigger":"exact story moment","at":0.25,"duration":2.0,"volume":0.22,"pan":0.0,"loop":false,"reason":"brief reason"}]}]}

Book context:
""" + json.dumps(context, ensure_ascii=False)
    request = {
        "model": QWEN_ANIMATION_MODEL,
        "messages": [
            {"role": "system", "content": (
                "Return valid JSON only. Create safe sound-design data; never "
                "emit code, shell commands, file paths or URLs.")},
            {"role": "user", "content": prompt},
        ],
        "temperature": .22 if payload.get("_sound_retry") else .45,
        "max_tokens": min(9000, max(1000, len(scenes) * 270)),
        "response_format": {"type": "json_object"},
    }
    release_model = bool(payload.get("release_model", True))
    try:
        _unload_model()
        response = _post_json(
            f"{OMLX_URL}/v1/chat/completions", request, timeout=900)
        content = (((response.get("choices") or [{}])[0].get("message") or {})
                   .get("content"))
        generated = content if isinstance(content, dict) else json.loads(str(content or ""))
    finally:
        if release_model:
            try:
                _post_json(
                    f"{OMLX_URL}/v1/models/{urllib.parse.quote(QWEN_ANIMATION_MODEL, safe='')}/unload",
                    {}, timeout=120)
            except Exception:
                pass
    rows = generated.get("scenes") if isinstance(generated, dict) else None
    if not isinstance(rows, list) and isinstance(generated, dict):
        # Qwen occasionally follows the cue schema but drops the outer
        # {"scenes": [...]} wrapper for a single scene. Accept the common
        # equivalent shapes instead of discarding an otherwise usable plan.
        candidate = generated.get("scene")
        if isinstance(candidate, dict):
            rows = [candidate]
        elif isinstance(generated.get("cues"), list):
            rows = [{"id": scenes[0]["id"], "cues": generated["cues"]}]
        else:
            for key in ("sound_plan", "plan", "result"):
                nested = generated.get(key)
                if not isinstance(nested, dict):
                    continue
                if isinstance(nested.get("scenes"), list):
                    rows = nested["scenes"]
                    break
                if isinstance(nested.get("cues"), list):
                    rows = [{"id": scenes[0]["id"],
                             "cues": nested["cues"]}]
                    break
    if not isinstance(rows, list):
        raise RuntimeError("Qwen sound director did not return a usable scene plan")
    by_id = {str(row.get("id") or ""): row for row in rows if isinstance(row, dict)}
    normalise_id = lambda value: re.sub(
        r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    by_normalised_id = {
        normalise_id(row_id): row for row_id, row in by_id.items()
        if normalise_id(row_id)
    }
    max_cues = {"none": 0, "gentle": 1, "balanced": 1,
                "lively": 2, "cinematic": 2}[intensity]
    result = []
    for scene in scenes:
        # Omitting a scene is a valid editorial choice here: quiet pages often
        # sound more professional with narration alone. Restore the row so the
        # saved plan still mirrors every book scene in order.
        row = (by_id.get(scene["id"])
               or by_normalised_id.get(normalise_id(scene["id"]))
               or {"cues": []})
        cues = []
        for index, cue in enumerate((row.get("cues") or [])[:max_cues]):
            if not isinstance(cue, dict):
                continue
            sound_prompt = " ".join(str(cue.get("prompt") or "").split())[:450]
            if not sound_prompt:
                continue
            kind = str(cue.get("kind") or "foley").lower()
            cues.append({
                "id": f"{scene['id']}-cue-{index + 1}",
                "kind": kind if kind in _SOUND_KINDS else "foley",
                "prompt": sound_prompt,
                "trigger": " ".join(str(cue.get("trigger") or "").split())[:180],
                "at": round(_number(cue.get("at"), .2, 0, .96), 3),
                "duration": round(_number(cue.get("duration"), 2, .4, 30), 2),
                "volume": round(_number(cue.get("volume"), .16, .04, .28), 3),
                "pan": round(_number(cue.get("pan"), 0, -1, 1), 2),
                "loop": bool(cue.get("loop")), "enabled": True,
                "reason": str(cue.get("reason") or "")[:500],
            })
        result.append({
            "id": scene["id"], "heading": scene["heading"],
            "seconds": round(scene["seconds"], 2), "cues": cues,
        })
    return {"model": QWEN_ANIMATION_MODEL, "scenes": result}


def _fetch_bytes(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _poll_job(status_url_base, job_id, on_stage=None, timeout=MEDIA_POLL_TIMEOUT):
    """Poll a sidecar /status?id= endpoint until done/error/failed."""
    url = f"{status_url_base}?id={urllib.parse.quote(job_id)}"
    t0 = time.time()
    last_stage = None
    while True:
        if time.time() - t0 > timeout:
            raise RuntimeError(f"job {job_id} timed out after {timeout}s")
        try:
            st = _get_json(url, timeout=30)
        except Exception as e:
            time.sleep(2)
            continue
        status = st.get("status")
        stage = st.get("stage")
        if on_stage and stage and stage != last_stage:
            on_stage(stage, st.get("progress"))
            last_stage = stage
        if status in ("done",):
            return st.get("result") or {}
        if status in ("error", "failed"):
            raise RuntimeError(st.get("error") or f"job {job_id} {status}")
        time.sleep(2)


def _resolve_image_ref(ref):
    """Turn an image reference (http url, /files/x.png, or bare img_x.png)
    into base64 bytes suitable for the video server's `image` field."""
    if not ref:
        return None
    if ref.startswith("data:") or (len(ref) > 200 and "/" not in ref[:40]):
        return ref  # already base64-ish / data-url
    if ref.startswith("http://") or ref.startswith("https://"):
        url = ref
    elif ref.startswith("/files/"):
        url = IMAGE_URL + ref
    else:
        url = f"{IMAGE_URL}/files/{ref}"
    raw = _fetch_bytes(url)
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Tool implementations  (each returns a small JSON-able dict for the model)
# ---------------------------------------------------------------------------
def tool_search_memory(args, step_cb):
    q = str(args.get("query", "")).strip()
    k = int(args.get("k", 5))
    out = _post_json(f"{MEMORY_URL}/memory/search", {"query": q, "k": k})
    res = out.get("results", [])
    # also search RAG docs for richer context
    try:
        rag = _post_json(f"{MEMORY_URL}/rag/search", {"query": q, "k": k})
        for r in rag.get("results", []):
            res.append({"text": r.get("text"), "source": r.get("source"),
                        "score": r.get("score")})
    except Exception:
        pass
    return {"results": res[: max(k, 5)]}


def tool_save_memory(args, step_cb):
    text = str(args.get("text", "")).strip()
    kind = str(args.get("kind", "note"))
    return _post_json(f"{MEMORY_URL}/memory/add", {"text": text, "kind": kind})


def tool_generate_speech(args, step_cb):
    text = str(args.get("text", "")).strip()
    if not text:
        raise ValueError("generate_speech requires 'text'")
    engine = str(args.get("engine", "kokoro"))
    voice = str(args.get("voice", "af_heart"))
    payload = {
        "text": text[:20000],
        "engine": engine,
        "voice": voice,
        "format": "wav",
        "loudness": "youtube",
        "sample_rate": 48000,
    }
    if "speed" in args:
        payload["speed"] = float(args["speed"])
    if UNLOAD_BEFORE_MEDIA:
        _unload_model()
    out = _post_json(f"{TTS_URL}/generate", payload, timeout=600)
    return {
        "audio_file": out.get("filename"),
        "url": TTS_URL + (out.get("url") or ""),
        "seconds": out.get("seconds"),
        "engine": out.get("engine"),
        "voice": out.get("voice"),
    }


def tool_generate_image(args, step_cb):
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("generate_image requires 'prompt'")
    requested_engine = str(args.get("engine") or "hidream").strip().lower()
    registry_model = get_model(requested_engine)
    if not registry_model or "orchestrator" not in registry_model.get("consumers", []):
        raise ValueError("unknown or disabled image model: " + requested_engine)
    payload = {"prompt": prompt[:6000], "engine": registry_model["id"]}
    if args.get("aspect"):
        payload["preset"] = str(args["aspect"])
    for k in ("width", "height", "steps", "seed", "guidance"):
        if k in args:
            payload[k] = args[k]
    refs = []
    if isinstance(args.get("reference_images"), list):
        refs.extend([str(x) for x in args.get("reference_images") if x])
    elif args.get("reference_images"):
        refs.append(str(args.get("reference_images")))
    if args.get("reference_image"):
        refs.append(str(args.get("reference_image")))
    max_refs = int((registry_model.get("capabilities") or {}).get(
        "max_reference_images", 0))
    refs = refs[:max_refs]
    if refs:
        resolved = []
        for r in refs:
            b64 = _resolve_image_ref(r)
            if b64:
                resolved.append(b64)
        if resolved:
            payload["reference_images"] = resolved[:max_refs]
    if "steps" not in payload:
        payload["steps"] = 28
    if UNLOAD_BEFORE_MEDIA:
        _unload_model()
    resp = _post_json(f"{IMAGE_URL}/generate", payload, timeout=60)
    job_id = resp.get("job_id")
    if not job_id:
        raise RuntimeError(resp.get("error") or "image generate failed")

    def stage(s, p):
        step_cb(f"image: {s}" + (f" ({p}%)" if p is not None else ""))

    result = _poll_job(f"{IMAGE_URL}/status", job_id, on_stage=stage)
    return {
        "image_file": result.get("filename"),
        "url": IMAGE_URL + (result.get("url") or ""),
        "width": result.get("width"),
        "height": result.get("height"),
        "seed": result.get("seed"),
        "engine": result.get("engine") or registry_model["id"],
    }


def tool_generate_video(args, step_cb):
    engine = str(args.get("engine", "longcat")).strip() or "longcat"
    prompt = str(args.get("prompt", "")).strip()
    payload = {"engine": engine}
    if prompt:
        payload["prompt"] = prompt[:2000]
    if args.get("video_type"):
        payload["video_type"] = str(args["video_type"])
    for k in ("negative_prompt", "aspect", "res", "resolution",
              "duration_seconds", "seed", "caption_style", "steps",
              "guide_scale", "voice_script", "voice", "voice_style",
              "video_style", "voice_language"):
        if k in args:
            payload[k] = args[k]
    if "voice_script" not in payload and args.get("script"):
        payload["voice_script"] = str(args["script"])
    # image input (wan i2v + avatar need a reference image)
    img_ref = args.get("image") or args.get("image_ref")
    if img_ref:
        b64 = _resolve_image_ref(img_ref)
        if b64:
            payload["image"] = b64
    # avatar needs an audio file produced by generate_speech
    if args.get("audio_file"):
        payload["audio_file"] = str(args["audio_file"])
    if UNLOAD_BEFORE_MEDIA:
        _unload_model()
    resp = _post_json(f"{VIDEO_URL}/generate", payload, timeout=120)
    job_id = resp.get("job_id")
    if not job_id:
        raise RuntimeError(resp.get("error") or "video generate failed")

    def stage(s, p):
        step_cb(f"video: {s}" + (f" ({p}%)" if p is not None else ""))

    result = _poll_job(f"{VIDEO_URL}/status", job_id, on_stage=stage)
    return {
        "video_file": result.get("video"),
        "url": f"{VIDEO_URL}/files/{result.get('video')}" if result.get("video") else None,
        "duration_seconds": result.get("duration_seconds"),
        "frames": result.get("frames"),
        "engine": engine,
    }


def tool_list_media_library(args, step_cb):
    out = _get_json(f"{VIDEO_URL}/library")
    return {
        "audio": [a.get("filename") for a in out.get("audio", [])][:25],
        "images": [i.get("filename") for i in out.get("images", [])][:25],
    }


# ---------------------------------------------------------------------------
# FlowAgent (social publishing agent on :3000) tools
# ---------------------------------------------------------------------------
def _fa_up():
    try:
        _get_json(f"{FLOWAGENT_URL}/api/status", timeout=5)
        return True
    except Exception:
        return False


def tool_flowagent_status(args, step_cb):
    h = _get_json(f"{FLOWAGENT_URL}/api/health", timeout=10)
    out = {
        "agent_ok": h.get("ok"),
        "uptime_s": h.get("uptime_s"),
        "queue_counts": h.get("queue"),
        "paused": h.get("kill_switch"),
        "manual_login_mode": h.get("manual_login_mode"),
        "paused_platforms": [p for p in (h.get("paused_platforms") or []) if p in FA_PLATFORMS],
        "llm": h.get("llm"),
        "last_diagnostic": h.get("last_diagnostic"),
        "supervised_platforms": FA_PLATFORMS,
    }
    # per-platform login/readiness from stored preflight results
    try:
        pf = _get_json(f"{FLOWAGENT_URL}/api/preflight", timeout=10)
        plats = pf.get("platforms") or {}
        pre = {}
        for name, r in plats.items():
            if name not in FA_PLATFORMS:
                continue
            r = r or {}
            status = r.get("status")
            reason = str(r.get("reason") or "")
            stale = bool(r.get("stale"))
            # legacy junk rows written before the inconclusive fix
            if "E_QUEUE_PAUSED" in reason or "Manual login mode" in reason:
                status = "unknown"
                reason = "artifact: preflight ran while queue was paused — NOT a login failure"
                stale = True
            pre[name] = {"status": status,
                         "logged_in": (None if stale else status == "ok"),
                         "age_minutes": r.get("ageMinutes"),
                         "stale": stale,
                         "reason": (reason[:200] or None)}
        out["preflight"] = pre
        out["preflight_note"] = ("Entries with stale=true say nothing about CURRENT login state. "
                                 "If login state matters, call flowagent_login_check for a fresh check.")
    except Exception:
        out["preflight"] = None
    return out


def tool_flowagent_login_check(args, step_cb):
    """Fresh, authoritative login check: visits each supervised platform headlessly."""
    step_cb(f"running fresh login preflight on {', '.join(FA_PLATFORMS)} (may take 1-3 min)")
    r = _post_json(f"{FLOWAGENT_URL}/api/preflight", {"platforms": FA_PLATFORMS}, timeout=420)
    out = {}
    for p in (r.get("results") or []):
        name = (p or {}).get("platform")
        if name not in FA_PLATFORMS:
            continue
        out[name] = {"status": p.get("status"),
                     "logged_in": p.get("status") == "ok",
                     "inconclusive": bool(p.get("inconclusive")),
                     "reason": (str(p.get("reason") or "")[:200]) or None}
    return {"checked_now": True,
            "summary": {k: r.get(k) for k in ("total", "ready", "needsAttention", "failed")},
            "platforms": out}


def tool_flowagent_failures(args, step_cb):
    n = max(1, min(20, int(args.get("limit", 8))))
    items = _get_json(f"{FLOWAGENT_URL}/api/tasks?status=failed&limit=50", timeout=10)
    if not isinstance(items, list):
        items = items.get("tasks", [])
    out = []
    now = time.time()
    for t in items:
        if t.get("platform") not in FA_PLATFORMS:
            continue
        age_days = None
        try:
            ts = time.mktime(time.strptime(str(t.get("updated_at")), "%Y-%m-%d %H:%M:%S"))
            age_days = round((now - ts) / 86400, 1)
        except Exception:
            pass
        out.append({
            "id": t.get("id"),
            "platform": t.get("platform"),
            "title": (str(t.get("title") or "")[:80]) or None,
            "error": (str(t.get("last_error") or t.get("result") or "")[:300]) or None,
            "retries": t.get("retries"),
            "updated_at": t.get("updated_at"),
            "age_days": age_days,
            "historical": bool(age_days is not None and age_days > 2),
        })
        if len(out) >= n:
            break
    return {"failed_tasks": out, "count": len(out),
            "note": "Tasks marked historical=true failed more than 2 days ago — treat as history, "
                    "NOT evidence of a current problem or login issue."}


def tool_flowagent_control(args, step_cb):
    action = str(args.get("action", "")).strip()
    if action == "resume":
        try:
            return _post_json(f"{FLOWAGENT_URL}/api/resume", {}, timeout=15)
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}
    if action == "pause":
        return _post_json(f"{FLOWAGENT_URL}/api/stop", {}, timeout=15)
    if action == "publish_next":
        return _post_json(f"{FLOWAGENT_URL}/api/publish-next", {}, timeout=15)
    if action == "restart_agent":
        step_cb("flowagent: restarting via flowagent.sh (may take ~30s)")
        try:
            r = subprocess.run(["bash", FLOWAGENT_SH, "restart"], capture_output=True,
                               text=True, timeout=120)
            ok = _fa_up()
            return {"ok": ok, "output": (r.stdout + r.stderr)[-600:]}
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}
    if action == "retry_task":
        tid = str(args.get("task_id", "")).strip()
        if not tid:
            return {"ok": False, "error": "task_id required for retry_task"}
        return _http_json("PATCH", f"{FLOWAGENT_URL}/api/tasks/{urllib.parse.quote(tid)}",
                          {"status": "pending"}, timeout=15)
    return {"ok": False, "error": f"unknown action '{action}'"}


# ---------------------------------------------------------------------------
# FlowAgent watchdog (deterministic — no LLM involved)
# ---------------------------------------------------------------------------
WATCHDOG_ON = os.environ.get("ORCH_WATCHDOG", "1") != "0"
WATCHDOG_INTERVAL = int(os.environ.get("ORCH_WATCHDOG_INTERVAL", "300"))
STALE_LOGIN_MIN = int(os.environ.get("ORCH_STALE_LOGIN_MIN", "30"))

_wd_lock = threading.Lock()
_wd_log = []          # ring buffer of {"ts","event","detail"}
_wd_login_since = None  # first time we saw manual_login_mode=true


def _wd_note(event, detail=""):
    with _wd_lock:
        _wd_log.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, "detail": detail})
        del _wd_log[:-50]
    print(f"[watchdog] {event} {detail}", flush=True)


def _watchdog_tick():
    global _wd_login_since
    try:
        h = _get_json(f"{FLOWAGENT_URL}/api/health", timeout=8)
    except Exception as e:
        _wd_note("flowagent_down", f"health unreachable ({str(e)[:120]}) -> flowagent.sh start")
        try:
            subprocess.Popen(["bash", FLOWAGENT_SH, "start"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e2:
            _wd_note("restart_failed", str(e2)[:200])
        return

    # stale manual-login pause: user logged in ages ago but never resumed
    if h.get("manual_login_mode"):
        now = time.time()
        if _wd_login_since is None:
            _wd_login_since = now
        elif now - _wd_login_since >= STALE_LOGIN_MIN * 60:
            try:
                _post_json(f"{FLOWAGENT_URL}/api/resume", {}, timeout=15)
                _wd_note("auto_resume", f"manual_login_mode stale >{STALE_LOGIN_MIN}min -> resumed")
                _wd_login_since = None
            except Exception as e:
                _wd_note("auto_resume_blocked", str(e)[:200])
                _wd_login_since = now  # re-arm, don't spam
    else:
        _wd_login_since = None


def _watchdog_loop():
    _wd_note("started", f"interval={WATCHDOG_INTERVAL}s stale_login={STALE_LOGIN_MIN}min "
                        f"platforms={','.join(FA_PLATFORMS)}")
    while True:
        time.sleep(WATCHDOG_INTERVAL)
        try:
            _watchdog_tick()
        except Exception as e:
            _wd_note("tick_error", str(e)[:200])


TOOL_IMPL = {
    "search_memory": tool_search_memory,
    "save_memory": tool_save_memory,
    "generate_speech": tool_generate_speech,
    "generate_image": tool_generate_image,
    "generate_video": tool_generate_video,
    "list_media_library": tool_list_media_library,
    "flowagent_status": tool_flowagent_status,
    "flowagent_failures": tool_flowagent_failures,
    "flowagent_login_check": tool_flowagent_login_check,
    "flowagent_control": tool_flowagent_control,
}

# ---------------------------------------------------------------------------
# Tool schemas advertised to the model (OpenAI/Qwen function-calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search the local memory/RAG store for facts, notes, brand voice, or prior context relevant to the goal. Call this early if the goal references the user's project, product, or preferences.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look up"},
            "k": {"type": "integer", "description": "max results (1-20)", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "Persist a useful fact or decision to memory for future workflows.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "kind": {"type": "string", "default": "note"},
        }, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "generate_speech",
        "description": "Synthesize a voiceover/narration to an audio file (returns audio_file usable by generate_video avatar engine). Use for narration, scripts, or talking-head voice.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the words to speak"},
            "voice": {"type": "string", "description": "voice id, e.g. af_heart, am_michael", "default": "af_heart"},
            "engine": {"type": "string", "enum": ["kokoro", "higgs", "voxtral"], "default": "kokoro"},
            "speed": {"type": "number", "default": 1.0},
        }, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "generate_image",
        "description": "Generate a still image with any enabled model from the shared oMLX image registry. Returns image_file + url. Use for thumbnails, portraits, product shots, book art, or a video reference frame.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "engine": {"type": "string", "enum": IMAGE_MODEL_IDS, "default": "hidream"},
            "aspect": {"type": "string", "enum": ["1:1", "9:16", "16:9", "4:5", "3:2"], "default": "9:16"},
            "steps": {"type": "integer", "default": 28},
            "seed": {"type": "integer"},
            "reference_image": {"type": "string", "description": "single reference image (data URL, image filename, /files path, or URL)"},
            "reference_images": {"type": "array", "items": {"type": "string"},
                                 "description": "reference images (the selected registry model's limit is enforced)"},
        }, "required": ["prompt"]},
    }},
    {"type": "function", "function": {
        "name": "generate_video",
        "description": "Generate a video. engine='longcat' = text-to-video (no image needed). engine='wan' = image-to-video (needs image). engine='avatar' = local audio-driven talking head (needs portrait + audio_file). engine='replicate_avatar' = cloud talking avatar (needs portrait + voice_script).",
        "parameters": {"type": "object", "properties": {
            "engine": {"type": "string", "enum": ["longcat", "wan", "avatar", "replicate_avatar"], "default": "longcat"},
            "prompt": {"type": "string", "description": "scene description (longcat/wan)"},
            "image": {"type": "string", "description": "image_file or url from generate_image (wan/avatar)"},
            "audio_file": {"type": "string", "description": "audio_file from generate_speech (avatar only)"},
            "voice_script": {"type": "string", "description": "script text (replicate_avatar)"},
            "voice": {"type": "string", "description": "cloud voice id (replicate_avatar)"},
            "voice_style": {"type": "string", "description": "cloud voice style (replicate_avatar)"},
            "video_style": {"type": "string", "description": "cloud visual style (replicate_avatar)"},
            "voice_language": {"type": "string", "description": "cloud language code (replicate_avatar)"},
            "aspect": {"type": "string", "enum": ["9:16", "16:9", "1:1"], "default": "9:16"},
            "duration_seconds": {"type": "number", "default": 10},
            "caption_style": {"type": "string", "enum": ["karaoke", "subtitle", "none"], "default": "karaoke"},
            "seed": {"type": "integer"},
        }, "required": ["engine"]},
    }},
    {"type": "function", "function": {
        "name": "list_media_library",
        "description": "List existing generated audio and image files available on disk, so you can reuse an asset instead of regenerating it.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "flowagent_status",
        "description": "Get the health of FlowAgent, the social blog-publishing agent (supervised platforms: medium, quora, flipboard, blogger, substack). Returns queue counts, paused state, manual-login mode, per-platform login/readiness (preflight), paused platforms, and the last diagnostic. Call this FIRST for any publishing/diagnosis goal.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "flowagent_failures",
        "description": "List recent FAILED publishing tasks on the supervised social platforms, with their error messages. Use to understand what is going wrong before fixing.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "max failures to return (1-20)", "default": 8},
        }},
    }},
    {"type": "function", "function": {
        "name": "flowagent_login_check",
        "description": "Run a FRESH, authoritative login check on the supervised platforms (slow: 1-3 min, visits each site headlessly). Use ONLY when login state is in question AND flowagent_status preflight data is stale/unknown. Result status 'ready' means logged in.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "flowagent_control",
        "description": "Control FlowAgent. action='resume' un-pauses the publishing queue; 'pause' stops it; 'publish_next' force-publishes the next queued blog now; 'restart_agent' fully restarts a stuck/dead agent process; 'retry_task' re-queues one failed task (requires task_id from flowagent_failures).",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["resume", "pause", "publish_next", "restart_agent", "retry_task"]},
            "task_id": {"type": "string", "description": "only for retry_task"},
        }, "required": ["action"]},
    }},
]

SYSTEM_PROMPT = """You are the Orchestrator for a local AI media studio running on the user's Mac. \
Your job is to accomplish the user's GOAL by planning and calling the available tools in the right order, \
feeding the output of one tool into the next.

The studio can: search/save memory, synthesize speech (generate_speech), create images (generate_image with optional reference images for consistency), \
and create video (generate_video: longcat=text-to-video, wan=image-to-video, avatar=local talking head from a \
portrait image + a speech audio_file, replicate_avatar=cloud talking head from a portrait + voice_script). \
It also supervises FlowAgent, the social blog-publishing agent \
(platforms: medium, quora, flipboard, blogger, substack) via flowagent_status / flowagent_failures / flowagent_control.

Guidelines:
- Think briefly, then act. Prefer the fewest steps that fully satisfy the goal.
- For FlowAgent health/diagnosis goals: call flowagent_status first, then flowagent_failures if anything \
looks wrong. Only use flowagent_control actions the goal permits. Never call restart_agent unless the agent \
is unreachable or the goal explicitly allows restarts. Errors mentioning captcha, verification, or login \
require a human - report them clearly instead of retrying.
- For a "talking head"/"spokesperson"/"presenter" video:
  - local avatar path: write script -> generate_speech -> generate_image -> generate_video(engine="avatar", image=<portrait>, audio_file=<audio_file>)
  - cloud avatar path: write script -> generate_image -> generate_video(engine="replicate_avatar", image=<portrait>, voice_script=<script>)
- For a general b-roll/scene video with narration: generate_speech for the voiceover AND generate_video(engine="longcat") \
for the visuals; tell the user both artifacts in your final summary.
- Pass files between tools using the exact image_file/audio_file/url values returned by earlier tool calls.
- When the goal is fully done, STOP calling tools and reply with a concise final summary that lists every \
produced artifact (its url) and how they fit together. Do not call any tool in that final message.
- If a tool errors, adapt (simpler params or a different engine) rather than repeating the same call."""


# ---------------------------------------------------------------------------
# Workflow templates
# Each workflow exposes a form (fields) in the UI and, from the filled inputs,
# builds an explicit GOAL/plan the Orchestrator model then executes with tools.
# A field with "source" pulls its dropdown options live from /options.
# ---------------------------------------------------------------------------
WORKFLOWS = [
    {
        "id": "sales_video",
        "label": "🎯 Sales video (talking head)",
        "description": "A spokesperson/avatar video that sells your product. Writes a script, "
                       "voices it, generates a presenter portrait, then an audio-driven talking video.",
        "note": "Video engine is configurable: local avatar or cloud replicate avatar.",
        "fields": [
            {"name": "offer", "label": "Product / offer", "type": "text", "required": True,
             "placeholder": "NoteZen — a minimalist markdown note-taking app"},
            {"name": "talking_points", "label": "Key talking points", "type": "textarea",
             "placeholder": "calm distraction-free UI; markdown; syncs everywhere; free to try"},
            {"name": "script", "label": "Exact voiceover script (optional)", "type": "textarea",
             "placeholder": "Leave blank to have the orchestrator write it from your talking points."},
            {"name": "presenter", "label": "Presenter / face", "type": "text",
             "default": "friendly professional presenter, warm smile, soft studio lighting, plain neutral background",
             "help": "Describe the person who appears and talks (a portrait is generated from this)."},
            {"name": "video_engine", "label": "Video engine", "type": "select", "source": "video_engines_talking"},
            {"name": "voice", "label": "Voice", "type": "select", "source": "voices"},
            {"name": "aspect", "label": "Aspect", "type": "select", "options": ["9:16", "1:1", "16:9"], "default": "9:16"},
            {"name": "duration", "label": "Duration (s)", "type": "number", "default": 15, "min": 5, "max": 30},
            {"name": "captions", "label": "Captions", "type": "select",
             "options": ["karaoke", "subtitle", "none"], "default": "karaoke"},
        ],
    },
    {
        "id": "narrated_promo",
        "label": "🎬 Narrated promo / b-roll clip",
        "description": "A cinematic promo scene with a separate voiceover track. "
                       "Returns both artifacts so you can combine them in the editor.",
        "note": "Video engine is configurable: LongCat text-to-video or Wan image-to-video.",
        "fields": [
            {"name": "topic", "label": "Topic / product", "type": "text", "required": True,
             "placeholder": "a productivity app launch"},
            {"name": "scene", "label": "Visual scene", "type": "textarea", "required": True,
             "placeholder": "slow cinematic push-in over a sunrise city skyline, warm golden light, drone shot"},
            {"name": "narration", "label": "Narration script (optional)", "type": "textarea",
             "placeholder": "Leave blank to auto-write a short narration about the topic."},
            {"name": "voice", "label": "Voice", "type": "select", "source": "voices"},
            {"name": "video_engine", "label": "Video engine", "type": "select", "source": "video_engines_scene"},
            {"name": "aspect", "label": "Aspect", "type": "select", "options": ["9:16", "16:9", "1:1"], "default": "9:16"},
            {"name": "duration", "label": "Duration (s)", "type": "number", "default": 10, "min": 5, "max": 30},
        ],
    },
    {
        "id": "animate_image",
        "label": "🖼️ Animate an image (image-to-video)",
        "description": "Generate a first-frame image, then animate it into a short motion clip (Wan2.2 i2v).",
        "note": "Wan i2v clips are short (≤8s).",
        "fields": [
            {"name": "subject", "label": "First-frame image", "type": "textarea", "required": True,
             "placeholder": "a red sports car parked on a coastal road at sunset, cinematic, 85mm"},
            {"name": "motion", "label": "Motion / action", "type": "text", "required": True,
             "placeholder": "camera slowly orbits the car, gentle wind, clouds drifting"},
            {"name": "aspect", "label": "Aspect", "type": "select", "options": ["9:16", "16:9", "1:1"], "default": "16:9"},
            {"name": "duration", "label": "Duration (s)", "type": "number", "default": 4, "min": 1, "max": 8},
        ],
    },
    {
        "id": "image_set",
        "label": "📸 Marketing image set",
        "description": "Generate a set of on-brand marketing images of your product or subject (optionally guided by attached reference images).",
        "fields": [
            {"name": "product", "label": "Subject / product", "type": "textarea", "required": True,
             "placeholder": "a matte-black wireless headphone on a marble surface"},
            {"name": "reference_images", "label": "Reference images (optional)", "type": "images",
             "help": "Attach product/brand images. The selected shared model's reference limit is applied automatically."},
            {"name": "image_engine", "label": "Image model", "type": "select",
             "source": "image_models"},
            {"name": "style", "label": "Style", "type": "select",
             "options": ["photorealistic studio", "lifestyle", "minimalist", "vibrant marketing", "cinematic"],
             "default": "photorealistic studio"},
            {"name": "aspect", "label": "Aspect", "type": "select",
             "options": ["1:1", "9:16", "16:9", "4:5", "3:2"], "default": "1:1"},
            {"name": "count", "label": "How many", "type": "number", "default": 3, "min": 1, "max": 4},
        ],
    },
    {
        "id": "voiceover",
        "label": "🎙️ Voiceover / narration",
        "description": "Turn a script into a mastered voiceover audio file.",
        "fields": [
            {"name": "script", "label": "Script", "type": "textarea", "required": True,
             "placeholder": "Welcome to NoteZen — calm notes for a busy mind."},
            {"name": "voice", "label": "Voice", "type": "select", "source": "voices"},
            {"name": "engine", "label": "Engine", "type": "select",
             "options": ["kokoro", "higgs", "voxtral"], "default": "kokoro"},
            {"name": "speed", "label": "Speed", "type": "number", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
        ],
    },
    {
        "id": "thumbnail",
        "label": "🔥 YouTube thumbnail",
        "description": "Generate a bold, high-contrast thumbnail image for a video.",
        "fields": [
            {"name": "topic", "label": "Video topic", "type": "text", "required": True,
             "placeholder": "how I automated my whole content pipeline locally"},
            {"name": "style", "label": "Style", "type": "select",
             "options": ["bold high-contrast", "clean minimal", "dramatic cinematic", "playful colorful"],
             "default": "bold high-contrast"},
            {"name": "aspect", "label": "Aspect", "type": "select", "options": ["16:9", "1:1", "9:16"], "default": "16:9"},
        ],
    },
    {
        "id": "social_doctor",
        "label": "🩺 Social publishing doctor",
        "description": "Check FlowAgent (medium, quora, flipboard, blogger, substack): diagnose stuck queues and failures, optionally fix what is safe to fix, and report.",
        "fields": [
            {"name": "focus", "label": "Focus (optional)", "type": "text",
             "placeholder": "e.g. why are quora posts failing?"},
            {"name": "fix", "label": "Auto-fix safe issues", "type": "select",
             "options": ["yes", "no"], "default": "yes"},
        ],
    },
    {
        "id": "custom",
        "label": "✨ Custom goal (free-form)",
        "description": "Describe any goal in your own words and let the orchestrator plan it.",
        "fields": [
            {"name": "goal", "label": "Goal", "type": "textarea", "required": True,
             "placeholder": "Make a 15s vertical talking-head promo for my note app with an upbeat voiceover."},
        ],
    },
]

_WF_BY_ID = {w["id"]: w for w in WORKFLOWS}


def _g(inp, key, default=""):
    v = inp.get(key, default)
    if v is None:
        return default
    return v


def _build_goal(wid, inp):
    """Compose an explicit goal/plan string from a workflow's filled inputs."""
    if wid not in _WF_BY_ID:
        raise ValueError(f"unknown workflow '{wid}'")
    # required-field validation
    for f in _WF_BY_ID[wid]["fields"]:
        if f.get("required") and not str(_g(inp, f["name"])).strip():
            raise ValueError(f"'{f['label']}' is required")

    if wid == "custom":
        return str(_g(inp, "goal")).strip()

    if wid == "sales_video":
        offer = _g(inp, "offer")
        pts = _g(inp, "talking_points")
        script = str(_g(inp, "script")).strip()
        presenter = _g(inp, "presenter") or "friendly professional presenter, soft studio lighting, plain background"
        vengine = str(_g(inp, "video_engine") or "avatar").strip().lower()
        if vengine not in ("avatar", "replicate_avatar"):
            vengine = "avatar"
        voice = _g(inp, "voice") or "af_heart"
        aspect = _g(inp, "aspect") or "9:16"
        dur = int(float(_g(inp, "duration", 15)))
        caps = _g(inp, "captions") or "karaoke"
        words = max(20, int(dur * 2.5))
        if script:
            script_step = f'Use EXACTLY this voiceover script (do not rewrite it):\n"""{script}"""'
        else:
            script_step = (f"Write a punchy, persuasive ~{dur}s spokesperson script (about {words} words) "
                           f"that sells: {offer}." + (f" Base it on these talking points: {pts}." if pts else ""))
        if vengine == "replicate_avatar":
            return (
                f"GOAL: Produce a {dur}-second {aspect} talking-head SALES VIDEO for: {offer}.\n"
                f"Use video engine: replicate_avatar (cloud).\n"
                f"Execute these steps in order, passing each output to the next:\n"
                f"1. {script_step}\n"
                f"2. Call generate_image to create a photorealistic PORTRAIT of the presenter "
                f"(this is the face that will talk): {presenter}. Use aspect \"{aspect}\".\n"
                f"3. Call generate_video with engine=\"replicate_avatar\", video_type=\"talking_head\", "
                f"image=<the portrait image_file from step 2>, voice_script=<the exact script from step 1>, "
                f"aspect=\"{aspect}\", caption_style=\"{caps}\", duration_seconds={dur}.\n"
                f"4. Finish with a summary containing the final video URL and the exact script used."
            )
        return (
            f"GOAL: Produce a {dur}-second {aspect} talking-head SALES VIDEO for: {offer}.\n"
            f"Use video engine: avatar (local).\n"
            f"Execute these steps in order, passing each output to the next:\n"
            f"1. {script_step}\n"
            f"2. Call generate_speech with that script and voice=\"{voice}\" to get an audio_file.\n"
            f"3. Call generate_image to create a photorealistic PORTRAIT of the presenter "
            f"(this is the face that will talk): {presenter}. Use aspect \"{aspect}\".\n"
            f"4. Call generate_video with engine=\"avatar\", image=<the portrait image_file from step 3>, "
            f"audio_file=<the audio_file from step 2>, aspect=\"{aspect}\", caption_style=\"{caps}\", "
            f"duration_seconds={dur}.\n"
            f"5. Finish with a summary containing the final video URL and the exact script used."
        )

    if wid == "narrated_promo":
        topic = _g(inp, "topic")
        scene = _g(inp, "scene")
        narration = str(_g(inp, "narration")).strip()
        voice = _g(inp, "voice") or "af_heart"
        vengine = str(_g(inp, "video_engine") or "longcat").strip().lower()
        if vengine not in ("longcat", "wan"):
            vengine = "longcat"
        aspect = _g(inp, "aspect") or "9:16"
        dur = int(float(_g(inp, "duration", 10)))
        words = max(18, int(dur * 2.4))
        if narration:
            narr_step = f'Use EXACTLY this narration:\n"""{narration}"""'
        else:
            narr_step = f"Write a short ~{dur}s narration (about {words} words) about: {topic}."
        if vengine == "wan":
            return (
                f"GOAL: Produce a {dur}-second {aspect} narrated promo/b-roll clip about: {topic}.\n"
                f"Use video engine: wan (image-to-video).\n"
                f"1. {narr_step} Then call generate_speech(voice=\"{voice}\") to get the voiceover audio_file.\n"
                f"2. Call generate_image with prompt=\"{scene}\" and aspect=\"{aspect}\" to create the first frame.\n"
                f"3. Call generate_video with engine=\"wan\", image=<the image_file from step 2>, "
                f"prompt=\"{scene}\", aspect=\"{aspect}\", duration_seconds={dur} to create the visuals.\n"
                f"4. Finish with a summary listing BOTH the voiceover audio URL and the video URL, and note "
                f"they are separate tracks to combine in the editor."
            )
        return (
            f"GOAL: Produce a {dur}-second {aspect} narrated promo/b-roll clip about: {topic}.\n"
            f"Use video engine: longcat (text-to-video).\n"
            f"1. {narr_step} Then call generate_speech(voice=\"{voice}\") to get the voiceover audio_file.\n"
            f"2. Call generate_video with engine=\"longcat\", prompt=\"{scene}\", aspect=\"{aspect}\", "
            f"duration_seconds={dur} to create the visuals.\n"
            f"3. Finish with a summary listing BOTH the voiceover audio URL and the video URL, and note "
            f"they are separate tracks to combine in the editor."
        )

    if wid == "animate_image":
        subject = _g(inp, "subject")
        motion = _g(inp, "motion")
        aspect = _g(inp, "aspect") or "16:9"
        dur = int(float(_g(inp, "duration", 4)))
        return (
            f"GOAL: Create a {dur}-second {aspect} image-to-video clip.\n"
            f"1. Call generate_image with prompt=\"{subject}\" and aspect=\"{aspect}\" to make the first frame.\n"
            f"2. Call generate_video with engine=\"wan\", image=<the image_file from step 1>, "
            f"prompt=\"{motion}\", aspect=\"{aspect}\", duration_seconds={dur}.\n"
            f"3. Finish with the final video URL."
        )

    if wid == "image_set":
        product = _g(inp, "product")
        style = _g(inp, "style") or "photorealistic studio"
        aspect = _g(inp, "aspect") or "1:1"
        count = max(1, min(4, int(float(_g(inp, "count", 3)))))
        refs = _g(inp, "reference_images", [])
        image_engine = str(_g(inp, "image_engine") or "hidream").strip().lower()
        if image_engine not in IMAGE_MODEL_IDS:
            image_engine = "hidream"
        ref_count = len(refs) if isinstance(refs, list) else 0
        ref_note = ("Reference images are attached (" + str(ref_count) + "). "
                    "Treat them as the source-of-truth for product identity/branding consistency."
                    if ref_count > 0 else "")
        return (
            f"GOAL: Generate {count} distinct {style} marketing images of: {product}.\n"
            + (ref_note + "\n" if ref_note else "")
            +
            f"Call generate_image {count} separate times with engine=\"{image_engine}\", "
            f"aspect=\"{aspect}\", a DIFFERENT seed, "
            f"and a slightly varied prompt/angle/lighting so the images differ. "
            f"Finish with a summary listing every image URL."
        )

    if wid == "voiceover":
        script = _g(inp, "script")
        voice = _g(inp, "voice") or "af_heart"
        engine = _g(inp, "engine") or "kokoro"
        speed = float(_g(inp, "speed", 1.0))
        return (
            f"GOAL: Produce a voiceover audio file.\n"
            f"Call generate_speech with text=\"{script}\", voice=\"{voice}\", engine=\"{engine}\", "
            f"speed={speed}. Finish with the audio URL."
        )

    if wid == "thumbnail":
        topic = _g(inp, "topic")
        style = _g(inp, "style") or "bold high-contrast"
        aspect = _g(inp, "aspect") or "16:9"
        return (
            f"GOAL: Generate a {style} YouTube thumbnail image for a video about: {topic}.\n"
            f"Call generate_image with a vivid, {style} prompt (strong focal subject, punchy colors, "
            f"leaves room for a short text overlay) and aspect=\"{aspect}\". Finish with the image URL."
        )

    if wid == "social_doctor":
        focus = str(_g(inp, "focus")).strip()
        fix = str(_g(inp, "fix", "yes")).strip().lower() != "no"
        if fix:
            fix_step = (
                "3. FIX what is safe to fix:\n"
                "   - Queue paused (paused=true) but manual_login_mode=false and no captcha/verification "
                "errors: call flowagent_control action=\"resume\".\n"
                "   - Failed tasks that are RECENT (historical=false) with a TRANSIENT error (timeout, "
                "navigation, network, element-not-found): retry AT MOST 2 with flowagent_control "
                "action=\"retry_task\". Never retry historical=true tasks.\n"
                "   - Do NOT retry tasks with captcha / verification / login / account-restricted errors — "
                "those need a human.\n"
                "   - Only use restart_agent if flowagent_status itself failed or agent_ok is false."
            )
        else:
            fix_step = "3. Do NOT change anything (no flowagent_control calls) — diagnose and report only."
        return (
            "GOAL: Health-check the social blog-publishing agent (FlowAgent) and report clearly."
            + (f" Focus especially on: {focus}." if focus else "") + "\n"
            "Only these platforms matter: medium, quora, flipboard, blogger, substack. Ignore all others.\n"
            "HOW TO INTERPRET DATA (important):\n"
            "- Failed tasks with historical=true (or age_days > 2) are OLD history from before recent fixes. "
            "Report them as history only. They are NOT current problems and NOT evidence of login issues.\n"
            "- Preflight entries marked 'stale artifact' or 'unknown' say NOTHING about login state. "
            "Never claim a platform needs login based on stale data.\n"
            "- Only a FRESH flowagent_login_check result of E_LOGIN_REQUIRED/E_CAPTCHA_REQUIRED means a "
            "platform actually needs human login.\n"
            "1. Call flowagent_status.\n"
            "2. If failed count > 0 or something looks off, call flowagent_failures.\n"
            "2b. If login state is unclear (stale/unknown preflight) AND it matters for your verdict, call "
            "flowagent_login_check once to get the truth.\n"
            f"{fix_step}\n"
            "4. Finish with a short report: overall state (healthy / degraded / blocked) judged on CURRENT "
            "data only, queue counts, current per-platform issues (clearly separated from historical "
            "failures), actions you took, and what (if anything) truly needs the human."
        )

    # fallback
    return str(_g(inp, "goal")).strip() or f"Workflow {wid}"


def _wf_label(wid):
    w = _WF_BY_ID.get(wid)
    return w["label"] if w else wid


def _workflows_public():
    return [{k: v for k, v in w.items()} for w in WORKFLOWS]


def _options():
    voices = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]
    vdefault = "af_heart"
    vavail = {}
    try:
        v = _get_json(f"{TTS_URL}/say/voices", timeout=8)
        if v.get("voices"):
            voices = v["voices"]
        vdefault = v.get("default", vdefault)
    except Exception:
        pass
    try:
        vinf = _get_json(f"{VIDEO_URL}/info", timeout=8)
        vavail = {k: bool((vinf.get("engines", {}).get(k, {}) or {}).get("available", True))
                  for k in ("avatar", "replicate_avatar", "longcat", "wan")}
    except Exception:
        vavail = {}
    talking_opts = [e for e in ("avatar", "replicate_avatar") if vavail.get(e, True)]
    if not talking_opts:
        talking_opts = ["avatar", "replicate_avatar"]
    scene_opts = [e for e in ("longcat", "wan") if vavail.get(e, True)]
    if not scene_opts:
        scene_opts = ["longcat", "wan"]
    return {
        "voices": {"options": voices, "default": vdefault},
        "video_engines_talking": {
            "options": talking_opts,
            "default": "avatar" if "avatar" in talking_opts else talking_opts[0],
        },
        "video_engines_scene": {
            "options": scene_opts,
            "default": "longcat" if "longcat" in scene_opts else scene_opts[0],
        },
        "image_models": {
            "options": [model["id"] for model in IMAGE_MODELS],
            "labels": {model["id"]: model["label"] for model in IMAGE_MODELS},
            "default": "hidream" if "hidream" in IMAGE_MODEL_IDS else IMAGE_MODEL_IDS[0],
        },
        "aspects": ["9:16", "16:9", "1:1", "4:5", "3:2"],
    }


# ---------------------------------------------------------------------------
# Agent loop  (runs in a background thread per job)
# ---------------------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()


def _new_job(goal, workflow=None):
    jid = "orch_" + uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[jid] = {
            "id": jid, "goal": goal, "status": "running",
            "workflow": workflow, "workflow_label": _wf_label(workflow) if workflow else None,
            "steps": [], "artifacts": [], "result": None,
            "error": None, "created": time.time(),
        }
    return jid


def _job(jid):
    with _jobs_lock:
        return _jobs.get(jid)


def _add_step(jid, kind, text, data=None):
    with _jobs_lock:
        j = _jobs.get(jid)
        if not j:
            return
        j["steps"].append({
            "t": round(time.time() - j["created"], 1),
            "kind": kind, "text": text, "data": data,
        })


def _add_artifact(jid, kind, url, label=None):
    if not url:
        return
    with _jobs_lock:
        j = _jobs.get(jid)
        if j is not None:
            j["artifacts"].append({"kind": kind, "url": url, "label": label})


_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _parse_output(text):
    think = ""
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
    body = _THINK_RE.sub("", text).strip()
    calls = []
    for cm in _TOOLCALL_RE.finditer(body):
        try:
            obj = json.loads(cm.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                calls.append(obj)
        except Exception:
            pass
    # assistant content minus think, kept for context (includes tool_call tags)
    return think, body, calls


def _wf_policy(wid, inp):
    """Deterministic per-job tool restrictions (don't trust the 8B to self-police)."""
    deny = set()
    default_tool_args = {}
    if wid == "social_doctor" and str(inp.get("fix", "yes")).strip().lower() == "no":
        deny.add("flowagent_control")
    if wid == "image_set":
        refs = inp.get("reference_images")
        if isinstance(refs, list):
            refs = [str(x) for x in refs if isinstance(x, str) and x.strip()][:3]
            if refs:
                default_tool_args["generate_image"] = {"reference_images": refs}
    return {"deny_tools": deny, "default_tool_args": default_tool_args}


def _scrub_tool_args(args):
    """Redact huge/base64 payloads from UI logs and loop-detection keys."""
    if not isinstance(args, dict):
        return args
    out = {}
    for k, v in args.items():
        if k == "reference_images" and isinstance(v, list):
            out[k] = [f"<image#{i + 1}:{len(str(x))} chars>" for i, x in enumerate(v)]
            continue
        if k == "reference_image" and isinstance(v, str):
            out[k] = f"<image:{len(v)} chars>"
            continue
        if isinstance(v, str) and (v.startswith("data:") or len(v) > 240):
            out[k] = v[:120] + "…"
        else:
            out[k] = v
    return out


def _run_job(jid, goal, policy=None):
    deny = (policy or {}).get("deny_tools") or set()
    default_tool_args = (policy or {}).get("default_tool_args") or {}
    seen_calls = {}

    def _force_final(msgs):
        msgs.append({"role": "user", "content":
                     "STOP. Do not call any more tools. Based on the observations above, write your "
                     "final report now as plain text (no <tool_call> tags)."})
        text2 = _generate(msgs, None)
        t2, b2, _ = _parse_output(text2)
        final = _TOOLCALL_RE.sub("", b2).strip() or t2 or "(no report)"
        with _jobs_lock:
            j = _jobs[jid]
            j["status"] = "done"
            j["result"] = final
        _add_step(jid, "final", final[:2000])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    try:
        for it in range(MAX_ITERS):
            _add_step(jid, "thinking", f"planning (turn {it + 1})")
            text = _generate(messages, TOOLS)
            think, body, calls = _parse_output(text)
            if think:
                _add_step(jid, "reason", think[:1200])
            messages.append({"role": "assistant", "content": body})

            if not calls:
                final = body or think or "(done)"
                with _jobs_lock:
                    j = _jobs[jid]
                    j["status"] = "done"
                    j["result"] = final
                _add_step(jid, "final", final[:2000])
                return

            for call in calls:
                name = call.get("name")
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict) and isinstance(default_tool_args.get(name), dict):
                    for k, v in default_tool_args[name].items():
                        args.setdefault(k, v)
                log_args = _scrub_tool_args(args)
                _add_step(jid, "tool_call", f"{name}", data=log_args)
                key = f"{name}|{json.dumps(log_args, sort_keys=True)}"
                seen_calls[key] = seen_calls.get(key, 0) + 1
                impl = TOOL_IMPL.get(name)
                if name in deny:
                    result = {"error": f"Tool '{name}' is DISABLED for this job (report-only mode). "
                                       "Do not call tools to change anything; write your final report."}
                elif seen_calls[key] >= 3:
                    _add_step(jid, "progress", "loop detected — forcing final report")
                    _force_final(messages)
                    return
                elif seen_calls[key] == 2:
                    result = {"error": "Duplicate call ignored — you already have this result above. "
                                       "Stop calling tools and write your final plain-text report."}
                elif impl is None:
                    result = {"error": f"unknown tool '{name}'"}
                else:
                    try:
                        result = impl(args, lambda s: _add_step(jid, "progress", s))
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                # collect artifacts for the UI
                if isinstance(result, dict):
                    if result.get("url") and name == "generate_speech":
                        _add_artifact(jid, "audio", result["url"], result.get("audio_file"))
                    if result.get("url") and name == "generate_image":
                        _add_artifact(jid, "image", result["url"], result.get("image_file"))
                    if result.get("url") and name == "generate_video":
                        _add_artifact(jid, "video", result["url"], result.get("video_file"))
                _add_step(jid, "observation", _short(result), data=result)
                messages.append({"role": "tool", "content": json.dumps(result)[:4000]})

        # ran out of iterations
        _add_step(jid, "progress", "step limit reached — forcing final report")
        _force_final(messages)
    except Exception as e:
        tb = traceback.format_exc()
        with _jobs_lock:
            j = _jobs.get(jid)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
        _add_step(jid, "error", f"{type(e).__name__}: {e}\n{tb[-500:]}")


def _short(obj, n=400):
    try:
        s = json.dumps(obj)
    except Exception:
        s = str(obj)
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# Compact chat gateway: orchestrator -> selected tools -> Qwen
# ---------------------------------------------------------------------------
_CHAT_ACTION_HINTS = (
    "search", "find", "look up", "browse", "fetch", "open", "read", "list",
    "show", "check", "inspect", "create", "write", "edit", "update", "change",
    "delete", "remove", "send", "sync", "upload", "download", "shopify",
    "order", "product", "inventory", "file", "folder", "drive", "gmail",
    "email", "calendar", "sheet", "document", "web", "latest", "current",
)
_CHAT_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "can", "could", "does",
    "for", "from", "have", "how", "into", "just", "more", "need", "only",
    "please", "that", "the", "their", "then", "there", "they", "this", "use",
    "using", "want", "what", "when", "where", "which", "with", "would", "you",
}
_CHAT_SERVER_TERMS = {
    "shopify": ("shopify", "product", "products", "order", "orders", "inventory",
                "collection", "collections", "customer", "discount", "store", "tag"),
    "brave-search": ("search", "web", "latest", "current", "news", "research",
                     "online", "internet", "source", "sources"),
    "fetch": ("fetch", "url", "website", "page", "link", "article"),
    "filesystem": ("file", "folder", "directory", "code", "project", "python",
                   "json", "config", "log", "local", "disk"),
    "google-workspace": ("google", "gmail", "email", "calendar", "drive", "sheet",
                         "sheets", "doc", "docs", "document", "workspace"),
}


def _chat_text(message):
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def _compact_chat_messages(messages, max_messages=14, max_chars=26000):
    """Keep the durable system instruction plus the recent working exchange."""
    rows = [m for m in (messages or []) if isinstance(m, dict) and m.get("role")]
    if not rows:
        return []
    systems = [m for m in rows if m.get("role") == "system"]
    conversation = [m for m in rows if m.get("role") != "system"][-max_messages:]
    result = []
    if systems:
        system_text = "\n\n".join(_chat_text(m) for m in systems if _chat_text(m))[-6000:]
        if system_text:
            result.append({"role": "system", "content": system_text})
    budget = max_chars - sum(len(_chat_text(m)) for m in result)
    kept = []
    for message in reversed(conversation):
        text_len = len(_chat_text(message))
        if kept and text_len > budget:
            break
        kept.append(message)
        budget -= min(text_len, budget)
        if budget <= 0:
            break
    result.extend(reversed(kept))
    return result


def _latest_chat_query(messages):
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _chat_text(message).strip()
    return ""


def _tool_score(tool, query, tokens):
    name = str(tool.get("name") or "").lower()
    desc = str(tool.get("description") or "").lower()
    server = str(tool.get("server") or "").lower()
    haystack = f"{name} {desc} {server}".replace("_", " ").replace("-", " ")
    score = sum(3 for token in tokens if token in haystack)
    for server_name, terms in _CHAT_SERVER_TERMS.items():
        if server == server_name and any(term in query for term in terms):
            score += 12
    if name in query:
        score += 20
    return score


def _model_tool_names(query, tools):
    """Ask ToolOrchestra for names only; never send full schemas to the router."""
    catalog = "\n".join(
            f"{t.get('name')} | {t.get('server')} | {str(t.get('description') or '')[:120]}"
                    for t in tools
                        )
    prompt = (
            "Choose at most 6 tools that are strictly necessary for the user request. "
                    "Return only a JSON array of exact tool names. Return [] when no external "
                            "tool is required. Do not choose speculative tools.\n\nUSER:\n" + query[:4000] +
                                    "\n\nAVAILABLE TOOLS:\n" + catalog[:30000]
                                        )
    text = _generate([
        {"role": "system", "content": "You are a fast, conservative tool router. JSON only."},
        {"role": "user", "content": prompt},
    ], None, max_tokens=180, temperature=0.0)
    match = re.search(r"\[[\s\S]*?\]", text or "")
    if not match:
        return []
    try:
        names = json.loads(match.group(0))
    except Exception:
        return []
    return [str(name) for name in names if isinstance(name, str)][:6]


def _route_chat(payload):
    started = time.time()
    compact = _compact_chat_messages(payload.get("messages") or [])
    query = _latest_chat_query(compact).lower()
    all_tools = (_get_json(f"{OMLX_URL}/v1/mcp/tools", timeout=20).get("tools") or [])
    tokens = {
        token for token in re.findall(r"[a-z0-9]{3,}", query)
        if token not in _CHAT_STOPWORDS
    }
    ranked = sorted(
        ((_tool_score(tool, query, tokens), tool) for tool in all_tools),
        key=lambda item: item[0], reverse=True,
    )
    deterministic = [tool for score, tool in ranked if score >= 6][:6]
    chosen_names = []
    used_model = False
    needs_tools = any(hint in query for hint in _CHAT_ACTION_HINTS)
    if needs_tools:
        try:
            chosen_names = _model_tool_names(query, all_tools)
            used_model = True
        except Exception:
            chosen_names = []
        finally:
            # Qwen3.8 gets the RAM; ToolOrchestra reloads quickly for the next route.
            _unload_model()
    by_name = {str(tool.get("name") or ""): tool for tool in all_tools}
    selected = []
    seen = set()
    for name in chosen_names + [str(tool.get("name") or "") for tool in deterministic]:
        tool = by_name.get(name)
        if not tool or name in seen:
            continue
        seen.add(name)
        selected.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or f"{tool.get('server')} tool"),
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            },
        })
        if len(selected) >= 6:
            break
    return {
        "messages": compact,
        "tools": selected,
        "tool_choice": "auto" if selected else None,
        "routing": {
            "orchestrator": True,
            "model_router_used": used_model,
            "input_messages": len(payload.get("messages") or []),
            "output_messages": len(compact),
            "available_tools": len(all_tools),
            "selected_tools": [item["function"]["name"] for item in selected],
            "elapsed_ms": round((time.time() - started) * 1000),
        },
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        if ctype == "application/json":
            body = json.dumps(obj).encode("utf-8")
        else:
            body = obj if isinstance(obj, bytes) else str(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if path == "/health":
            self._send(200, {
                "ok": True,
                "model": os.path.basename(MODEL_PATH),
                "model_loaded": _model is not None,
                "load_error": _load_error,
                "tools": list(TOOL_IMPL.keys()),
                "queue": sum(1 for j in _jobs.values() if j["status"] == "running"),
                "watchdog": {"enabled": WATCHDOG_ON,
                             "last": (_wd_log[-1] if _wd_log else None)},
            })
            return
        if path == "/watchdog":
            with _wd_lock:
                self._send(200, {"enabled": WATCHDOG_ON, "interval_s": WATCHDOG_INTERVAL,
                                 "stale_login_min": STALE_LOGIN_MIN,
                                 "platforms": FA_PLATFORMS, "log": list(_wd_log)})
            return
        if path == "/info":
            self._send(200, {
                "model": os.path.basename(MODEL_PATH),
                "tools": TOOLS,
                "sidecars": {"tts": TTS_URL, "memory": MEMORY_URL,
                             "image": IMAGE_URL, "video": VIDEO_URL},
                "max_iters": MAX_ITERS,
            })
            return
        if path == "/workflows":
            self._send(200, {"workflows": _workflows_public()})
            return
        if path == "/options":
            self._send(200, _options())
            return
        if path == "/status":
            jid = (qs.get("id") or [""])[0]
            j = _job(jid)
            if not j:
                self._send(404, {"error": "unknown job"})
                return
            with _jobs_lock:
                self._send(200, {
                    "status": j["status"], "goal": j["goal"],
                    "workflow": j.get("workflow"), "workflow_label": j.get("workflow_label"),
                    "steps": j["steps"], "artifacts": j["artifacts"],
                    "result": j["result"], "error": j["error"],
                })
            return
        if path == "/files":
            # proxy: /files?src=image&name=img_x.png
            src = (qs.get("src") or ["image"])[0]
            name = (qs.get("name") or [""])[0]
            base = {"image": IMAGE_URL, "video": VIDEO_URL,
                    "tts": TTS_URL}.get(src, IMAGE_URL)
            try:
                raw = _fetch_bytes(f"{base}/files/{urllib.parse.quote(name)}")
                ctype = "application/octet-stream"
                if name.endswith(".png"):
                    ctype = "image/png"
                elif name.endswith(".mp4"):
                    ctype = "video/mp4"
                elif name.endswith(".wav"):
                    ctype = "audio/wav"
                self._send(200, raw, ctype)
            except Exception as e:
                self._send(404, {"error": str(e)})
            return
        if path in ("/", "/index.html"):
            self._send(200, _UI_HTML, "text/html")
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            data = {}
        if path == "/route-chat":
            try:
                routed = _route_chat(data)
                self._send(200, {"ok": True, **routed})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        if path == "/animation-plan":
            try:
                plan = _qwen_animation_plan(data)
                self._send(200, {"ok": True, **plan})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {
                    "error": f"{type(exc).__name__}: {exc}",
                    "director_model": QWEN_ANIMATION_MODEL,
                })
            return
        if path == "/sound-plan":
            try:
                plan = _qwen_sound_plan(data)
                self._send(200, {"ok": True, **plan})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {
                    "error": f"{type(exc).__name__}: {exc}",
                    "director_model": QWEN_ANIMATION_MODEL,
                })
            return
        if path == "/orchestrate":
            wid = str(data.get("workflow", "")).strip()
            if wid:
                inputs = data.get("inputs") or {}
                try:
                    goal = _build_goal(wid, inputs)
                except ValueError as e:
                    self._send(400, {"error": str(e)})
                    return
                if not goal:
                    self._send(400, {"error": "workflow produced an empty goal"})
                    return
                jid = _new_job(goal, workflow=wid)
                policy = _wf_policy(wid, inputs)
            else:
                goal = str(data.get("goal", "")).strip()
                if not goal:
                    self._send(400, {"error": "goal is required"})
                    return
                jid = _new_job(goal)
                policy = None
            t = threading.Thread(target=_run_job, args=(jid, goal, policy), daemon=True)
            t.start()
            self._send(200, {"ok": True, "job_id": jid, "goal": goal})
            return
        self._send(404, {"error": "not found"})


_UI_HTML = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>omlx-orchestrator</title><style>
body{font:14px -apple-system,system-ui,sans-serif;max-width:860px;margin:20px auto;padding:0 16px;background:#0d1117;color:#e6edf3}
h1{font-size:18px;margin:0 0 4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:14px}
label{display:block;font-size:12px;color:#adbac7;margin:12px 0 4px;font-weight:600}
select,input,textarea{width:100%;box-sizing:border-box;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:9px;font:inherit}
textarea{min-height:64px;resize:vertical}
input[type=number]{max-width:160px}
button{background:#238636;color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer;margin-top:14px}
button:disabled{opacity:.5;cursor:default}
.wfdesc{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 12px;margin:8px 0;color:#adbac7;font-size:13px}
.wfnote{color:#e3b341;font-size:12px;margin-top:6px}
.help{color:#6e7681;font-size:11px;margin-top:3px}
#form{margin-top:4px}
.step{border-left:3px solid #30363d;padding:4px 10px;margin:6px 0;white-space:pre-wrap}
.reason{border-color:#8957e5;color:#c9a0ff}.tool_call{border-color:#1f6feb;color:#79c0ff}
.observation{border-color:#3fb950;color:#7ee787}.error{border-color:#f85149;color:#ff7b72}
.final{border-color:#d29922;color:#e3b341;font-weight:600}.progress{border-color:#484f58;color:#8b949e;font-size:12px}
a{color:#79c0ff}img,video{max-width:340px;border-radius:8px;display:block;margin:6px 0}
hr{border:0;border-top:1px solid #21262d;margin:18px 0}
</style></head><body>
<h1>🧭 omlx-orchestrator</h1>
<div class=sub>Pick a workflow, fill in the details, and the orchestrator builds &amp; runs the whole pipeline for you.</div>

<label for=wf>Workflow</label>
<select id=wf onchange=renderForm()></select>
<div id=wfdesc class=wfdesc></div>
<div id=form></div>
<button id=runbtn onclick=go()>Run workflow</button>

<div id=out></div>
<script>
let WORKFLOWS=[], OPTS={}, timer;

async function boot(){
 try{
  const [w,o]=await Promise.all([fetch('/workflows').then(r=>r.json()),fetch('/options').then(r=>r.json())]);
  WORKFLOWS=w.workflows||[];OPTS=o||{};
 }catch(e){document.getElementById('out').innerHTML='<div class="step error">Could not load workflows: '+esc(e.message)+'</div>';return;}
 const sel=document.getElementById('wf');
 sel.innerHTML=WORKFLOWS.map(w=>'<option value="'+w.id+'">'+esc(w.label)+'</option>').join('');
 renderForm();
}

function curWf(){return WORKFLOWS.find(w=>w.id===document.getElementById('wf').value);}

function renderForm(){
 const w=curWf();if(!w)return;
 let d='<div>'+esc(w.description||'')+'</div>';
 if(w.note)d+='<div class=wfnote>⏱ '+esc(w.note)+'</div>';
 document.getElementById('wfdesc').innerHTML=d;
 const f=document.getElementById('form');
 f.innerHTML=(w.fields||[]).map(fieldHtml).join('');
 for(const fl of (w.fields||[])){
  if(fl.type==='images'){
   const id='f_'+fl.name;
   const el=document.getElementById(id);
   const meta=document.getElementById(id+'_meta');
   if(el&&meta){
    el.onchange=()=>{const n=(el.files||[]).length;meta.textContent=n?(n+' image(s) selected'):'No images selected';};
   }
  }
 }
}

function fieldHtml(fl){
 const id='f_'+fl.name;
 let ctl='';
 let opts=fl.options;
 let def=fl.default;
 if(fl.source&&OPTS[fl.source]){const s=OPTS[fl.source];opts=s.options||s;if(def==null)def=s.default;}
 if(fl.type==='textarea'){
  ctl='<textarea id="'+id+'" placeholder="'+esc(fl.placeholder||'')+'">'+esc(def||'')+'</textarea>';
 }else if(fl.type==='images'){
  ctl='<input id="'+id+'" type=file accept="image/*" multiple><div id="'+id+'_meta" class=help>No images selected</div>';
 }else if(fl.type==='select'){
  ctl='<select id="'+id+'">'+(opts||[]).map(o=>'<option'+(o===def?' selected':'')+'>'+esc(o)+'</option>').join('')+'</select>';
 }else if(fl.type==='number'){
  ctl='<input id="'+id+'" type=number value="'+(def!=null?def:'')+'"'+
      (fl.min!=null?' min="'+fl.min+'"':'')+(fl.max!=null?' max="'+fl.max+'"':'')+
      (fl.step!=null?' step="'+fl.step+'"':'')+'>';
 }else{
  ctl='<input id="'+id+'" type=text value="'+esc(def||'')+'" placeholder="'+esc(fl.placeholder||'')+'">';
 }
 let h='<label for="'+id+'">'+esc(fl.label)+(fl.required?' *':'')+'</label>'+ctl;
 if(fl.help)h+='<div class=help>'+esc(fl.help)+'</div>';
 return h;
}

function fileToDataURL(file){
 return new Promise((resolve,reject)=>{
  const fr=new FileReader();
  fr.onload=()=>resolve(String(fr.result||''));
  fr.onerror=()=>reject(new Error('failed to read image'));
  fr.readAsDataURL(file);
 });
}

async function collect(){
 const w=curWf();const inputs={};
 for(const fl of (w.fields||[])){
  const el=document.getElementById('f_'+fl.name);if(!el)continue;
  let v=el.value;
  if(fl.type==='images'){
   const files=Array.from(el.files||[]).slice(0,3);
   v=await Promise.all(files.map(fileToDataURL));
  }else if(fl.type==='number'){
   v=v===''?null:Number(v);
  }
  inputs[fl.name]=v;
 }
 return {workflow:w.id,inputs};
}

async function go(){
 const w=curWf();if(!w)return;
 const body=await collect();
 const btn=document.getElementById('runbtn');btn.disabled=true;
 document.getElementById('out').innerHTML='<hr><div class=step>starting…</div>';
 let j;
 try{
  const r=await fetch('/orchestrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  j=await r.json();
 }catch(e){j={error:e.message};}
 btn.disabled=false;
 if(j.error){document.getElementById('out').innerHTML='<hr><div class="step error">'+esc(j.error)+'</div>';return;}
 clearInterval(timer);timer=setInterval(()=>poll(j.job_id),1400);poll(j.job_id);
}

async function poll(id){
 let j;try{const r=await fetch('/status?id='+id);j=await r.json();}catch(e){return;}
 let h='<hr>';
 if(j.workflow_label)h+='<div class=sub>'+esc(j.workflow_label)+' — '+esc(j.status)+'</div>';
 for(const s of (j.steps||[])){h+='<div class="step '+s.kind+'">['+s.t+'s] '+s.kind+': '+esc(s.text||'')+
   (s.kind==='tool_call'&&s.data?'  '+esc(JSON.stringify(s.data)):'')+'</div>';}
 if((j.artifacts||[]).length){h+='<h3>Artifacts</h3>';for(const a of j.artifacts){
   if(a.kind==='image')h+='<img src="'+a.url+'">';
   else if(a.kind==='video')h+='<video src="'+a.url+'" controls></video>';
   else if(a.kind==='audio')h+='<audio src="'+a.url+'" controls></audio>';
   h+='<div><a href="'+a.url+'" target=_blank>'+esc(a.label||a.url)+'</a></div>';}}
 document.getElementById('out').innerHTML=h;
 if(j.status!=='running')clearInterval(timer);
}
function esc(s){return (s+'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
boot();
</script></body></html>"""


def main():
    if WATCHDOG_ON:
        threading.Thread(target=_watchdog_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-orchestrator listening on http://{HOST}:{PORT}  model={os.path.basename(MODEL_PATH)}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
