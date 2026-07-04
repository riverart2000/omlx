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
              "guide_scale"):
        if k in args:
            payload[k] = args[k]
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


TOOL_IMPL = {
    "search_memory": tool_search_memory,
    "save_memory": tool_save_memory,
    "generate_speech": tool_generate_speech,
    "generate_image": tool_generate_image,
    "generate_video": tool_generate_video,
    "list_media_library": tool_list_media_library,
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
        "description": "Generate a still image with HiDream. Returns image_file + url. Use for thumbnails, portraits (for avatar video), product shots, or the reference frame for image-to-video.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "aspect": {"type": "string", "enum": ["1:1", "9:16", "16:9", "4:5", "3:2"], "default": "9:16"},
            "steps": {"type": "integer", "default": 28},
            "seed": {"type": "integer"},
        }, "required": ["prompt"]},
    }},
    {"type": "function", "function": {
        "name": "generate_video",
        "description": "Generate a video. engine='longcat' = text-to-video (no image needed). engine='wan' = image-to-video (needs image). engine='avatar' = audio-driven talking head (needs image portrait AND audio_file from generate_speech). Pass image as an image_file/url returned by generate_image.",
        "parameters": {"type": "object", "properties": {
            "engine": {"type": "string", "enum": ["longcat", "wan", "avatar"], "default": "longcat"},
            "prompt": {"type": "string", "description": "scene description (longcat/wan)"},
            "image": {"type": "string", "description": "image_file or url from generate_image (wan/avatar)"},
            "audio_file": {"type": "string", "description": "audio_file from generate_speech (avatar only)"},
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
]

SYSTEM_PROMPT = """You are the Orchestrator for a local AI media studio running on the user's Mac. \
Your job is to accomplish the user's GOAL by planning and calling the available tools in the right order, \
feeding the output of one tool into the next.

The studio can: search/save memory, synthesize speech (generate_speech), create images (generate_image), \
and create video (generate_video: longcat=text-to-video, wan=image-to-video, avatar=talking head from a \
portrait image + a speech audio_file).

Guidelines:
- Think briefly, then act. Prefer the fewest steps that fully satisfy the goal.
- For a "talking head"/"spokesperson"/"presenter" video: write a short script, call generate_speech to get an \
audio_file, call generate_image to get a portrait, then generate_video(engine="avatar", image=<portrait>, audio_file=<audio_file>).
- For a general b-roll/scene video with narration: generate_speech for the voiceover AND generate_video(engine="longcat") \
for the visuals; tell the user both artifacts in your final summary.
- Pass files between tools using the exact image_file/audio_file/url values returned by earlier tool calls.
- When the goal is fully done, STOP calling tools and reply with a concise final summary that lists every \
produced artifact (its url) and how they fit together. Do not call any tool in that final message.
- If a tool errors, adapt (simpler params or a different engine) rather than repeating the same call."""


# ---------------------------------------------------------------------------
# Agent loop  (runs in a background thread per job)
# ---------------------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()


def _new_job(goal):
    jid = "orch_" + uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[jid] = {
            "id": jid, "goal": goal, "status": "running",
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


def _run_job(jid, goal):
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
                _add_step(jid, "tool_call", f"{name}", data=args)
                impl = TOOL_IMPL.get(name)
                if impl is None:
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
        with _jobs_lock:
            j = _jobs[jid]
            j["status"] = "done"
            j["result"] = "Reached step limit. Partial results are in artifacts."
        _add_step(jid, "final", "Reached step limit.")
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
            })
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
        if path == "/status":
            jid = (qs.get("id") or [""])[0]
            j = _job(jid)
            if not j:
                self._send(404, {"error": "unknown job"})
                return
            with _jobs_lock:
                self._send(200, {
                    "status": j["status"], "goal": j["goal"],
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
            goal = str(data.get("goal", "")).strip()
            if not goal:
                self._send(400, {"error": "goal is required"})
                return
            jid = _new_job(goal)
            t = threading.Thread(target=_run_job, args=(jid, goal), daemon=True)
            t.start()
            self._send(200, {"ok": True, "job_id": jid})
            return
        self._send(404, {"error": "not found"})


_UI_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>omlx-orchestrator</title><style>
body{font:14px -apple-system,system-ui,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;background:#0d1117;color:#e6edf3}
h1{font-size:18px}textarea{width:100%;height:70px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:10px;font:inherit}
button{background:#238636;color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;cursor:pointer;margin-top:8px}
.step{border-left:3px solid #30363d;padding:4px 10px;margin:6px 0;white-space:pre-wrap}
.reason{border-color:#8957e5;color:#c9a0ff}.tool_call{border-color:#1f6feb;color:#79c0ff}
.observation{border-color:#3fb950;color:#7ee787}.error{border-color:#f85149;color:#ff7b72}
.final{border-color:#d29922;color:#e3b341;font-weight:600}.progress{border-color:#484f58;color:#8b949e;font-size:12px}
a{color:#79c0ff}img,video{max-width:320px;border-radius:8px;display:block;margin:6px 0}
</style></head><body>
<h1>🧭 omlx-orchestrator</h1>
<textarea id=g placeholder="Describe a goal, e.g. Make a 15s vertical talking-head promo for my note-taking app with an upbeat voiceover"></textarea>
<button onclick=go()>Run workflow</button>
<div id=out></div>
<script>
let timer;
async function go(){
 const goal=document.getElementById('g').value.trim();if(!goal)return;
 document.getElementById('out').innerHTML='<div class=step>starting…</div>';
 const r=await fetch('/orchestrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal})});
 const j=await r.json();if(j.error){document.getElementById('out').innerHTML='<div class="step error">'+j.error+'</div>';return;}
 clearInterval(timer);timer=setInterval(()=>poll(j.job_id),1200);poll(j.job_id);
}
async function poll(id){
 const r=await fetch('/status?id='+id);const j=await r.json();
 let h='';
 for(const s of j.steps){h+='<div class="step '+s.kind+'">['+s.t+'s] '+(s.kind)+': '+esc(s.text||'')+
   (s.kind==='tool_call'&&s.data?'  '+esc(JSON.stringify(s.data)):'')+'</div>';}
 if(j.artifacts.length){h+='<h3>Artifacts</h3>';for(const a of j.artifacts){
   if(a.kind==='image')h+='<img src="'+a.url+'">';
   else if(a.kind==='video')h+='<video src="'+a.url+'" controls></video>';
   else if(a.kind==='audio')h+='<audio src="'+a.url+'" controls></audio>';
   h+='<div><a href="'+a.url+'" target=_blank>'+(a.label||a.url)+'</a></div>';}}
 document.getElementById('out').innerHTML=h;
 if(j.status!=='running')clearInterval(timer);
}
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-orchestrator listening on http://{HOST}:{PORT}  model={os.path.basename(MODEL_PATH)}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
