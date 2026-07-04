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
        "note": "Local avatar engine — a 15s clip takes several minutes to render.",
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
        "description": "A cinematic text-to-video scene (LongCat) with a separate voiceover track. "
                       "Returns both artifacts so you can combine them in the editor.",
        "note": "LongCat text-to-video is slow: ~10 min per 5s segment.",
        "fields": [
            {"name": "topic", "label": "Topic / product", "type": "text", "required": True,
             "placeholder": "a productivity app launch"},
            {"name": "scene", "label": "Visual scene", "type": "textarea", "required": True,
             "placeholder": "slow cinematic push-in over a sunrise city skyline, warm golden light, drone shot"},
            {"name": "narration", "label": "Narration script (optional)", "type": "textarea",
             "placeholder": "Leave blank to auto-write a short narration about the topic."},
            {"name": "voice", "label": "Voice", "type": "select", "source": "voices"},
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
        "description": "Generate a set of on-brand marketing images of your product or subject.",
        "fields": [
            {"name": "product", "label": "Subject / product", "type": "textarea", "required": True,
             "placeholder": "a matte-black wireless headphone on a marble surface"},
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
        return (
            f"GOAL: Produce a {dur}-second {aspect} talking-head SALES VIDEO for: {offer}.\n"
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
        aspect = _g(inp, "aspect") or "9:16"
        dur = int(float(_g(inp, "duration", 10)))
        words = max(18, int(dur * 2.4))
        if narration:
            narr_step = f'Use EXACTLY this narration:\n"""{narration}"""'
        else:
            narr_step = f"Write a short ~{dur}s narration (about {words} words) about: {topic}."
        return (
            f"GOAL: Produce a {dur}-second {aspect} narrated promo/b-roll clip about: {topic}.\n"
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
        return (
            f"GOAL: Generate {count} distinct {style} marketing images of: {product}.\n"
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
    try:
        v = _get_json(f"{TTS_URL}/say/voices", timeout=8)
        if v.get("voices"):
            voices = v["voices"]
        vdefault = v.get("default", vdefault)
    except Exception:
        pass
    return {
        "voices": {"options": voices, "default": vdefault},
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
            else:
                goal = str(data.get("goal", "")).strip()
                if not goal:
                    self._send(400, {"error": "goal is required"})
                    return
                jid = _new_job(goal)
            t = threading.Thread(target=_run_job, args=(jid, goal), daemon=True)
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
}

function fieldHtml(fl){
 const id='f_'+fl.name;
 let ctl='';
 let opts=fl.options;
 let def=fl.default;
 if(fl.source&&OPTS[fl.source]){const s=OPTS[fl.source];opts=s.options||s;if(def==null)def=s.default;}
 if(fl.type==='textarea'){
  ctl='<textarea id="'+id+'" placeholder="'+esc(fl.placeholder||'')+'">'+esc(def||'')+'</textarea>';
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

function collect(){
 const w=curWf();const inputs={};
 for(const fl of (w.fields||[])){
  const el=document.getElementById('f_'+fl.name);if(!el)continue;
  let v=el.value;
  if(fl.type==='number')v=v===''?null:Number(v);
  inputs[fl.name]=v;
 }
 return {workflow:w.id,inputs};
}

async function go(){
 const w=curWf();if(!w)return;
 const body=collect();
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
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-orchestrator listening on http://{HOST}:{PORT}  model={os.path.basename(MODEL_PATH)}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
