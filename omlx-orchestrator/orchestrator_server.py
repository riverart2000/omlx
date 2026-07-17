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
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


def _generate(messages, tools):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = _load_model()
    prompt = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.6, top_p=0.95)
    text = generate(
        model, tok, prompt=prompt, max_tokens=MAX_TOKENS,
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
    payload = {"prompt": prompt[:2000]}
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
    refs = refs[:3]
    if refs:
        resolved = []
        for r in refs:
            b64 = _resolve_image_ref(r)
            if b64:
                resolved.append(b64)
        if resolved:
            payload["reference_images"] = resolved[:3]
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
        "description": "Generate a still image with HiDream. Returns image_file + url. Use for thumbnails, portraits (for avatar video), product shots, or the reference frame for image-to-video. Supports 1-3 reference images for identity/style consistency.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "aspect": {"type": "string", "enum": ["1:1", "9:16", "16:9", "4:5", "3:2"], "default": "9:16"},
            "steps": {"type": "integer", "default": 28},
            "seed": {"type": "integer"},
            "reference_image": {"type": "string", "description": "single reference image (data URL, image filename, /files path, or URL)"},
            "reference_images": {"type": "array", "items": {"type": "string"},
                                 "description": "up to 3 reference images (data URLs, image filenames, /files paths, or URLs)"},
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
            {"name": "reference_images", "label": "Reference images (optional, up to 3)", "type": "images",
             "help": "Attach product/brand images. The workflow will pass them to HiDream so outputs stay consistent."},
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
        ref_count = len(refs) if isinstance(refs, list) else 0
        ref_note = ("Reference images are attached (" + str(ref_count) + "). "
                    "Treat them as the source-of-truth for product identity/branding consistency."
                    if ref_count > 0 else "")
        return (
            f"GOAL: Generate {count} distinct {style} marketing images of: {product}.\n"
            + (ref_note + "\n" if ref_note else "")
            +
            f"Call generate_image {count} separate times, each with aspect=\"{aspect}\", a DIFFERENT seed, "
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
