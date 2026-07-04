#!/usr/bin/env python3
"""omlx-song — local ACE-Step1.5 music/song generation microservice (port 8600).

Mirrors the omlx-video (:8500) / omlx-image (:8400) pattern: a tiny stdlib HTTP
server with an async job API and a single serialized worker (music generation is
compute-bound, so exactly one runs at a time). Each job is generated in a fresh
subprocess (gen_song.py) so the ~10GB of MLX weights are returned to the OS the
moment it finishes — important on a 64GB box that also hosts the 35B chat LLM.

Model: mlx-community/ACE-Step1.5-MLX (Apache-2.0, commercial-safe). Text->music
plus optional lyrics for full sung songs. Native output 48kHz stereo.

Before each generation the oMLX chat model (:8000) is unloaded to free memory,
then reloaded afterwards (same auto-unload pattern as omlx-video).

Endpoints:
  GET  /health              -> service + weights status
  GET  /info                -> defaults, presets, limits, style/language options
  GET  /stats               -> GPU / memory / watts
  POST /generate            -> {prompt, lyrics?, duration, num_steps, seed, ...} => {job_id}
  GET  /status?id=JOB       -> job progress / result
  GET  /library?limit=N     -> recent generated songs
  GET  /files/<name>        -> a rendered clip (.flac/.mp3/.wav)
"""
import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOST = os.environ.get("SONG_HOST", "127.0.0.1")
PORT = int(os.environ.get("SONG_PORT", "8600"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

MODEL_DIR = os.environ.get(
    "SONG_MODEL_DIR",
    "/Users/joebains/.omlx/models/mlx-community/ACE-Step1.5-MLX")
VENV_PY = os.environ.get("SONG_PY", os.path.join(BASE_DIR, ".venv", "bin", "python"))
GEN_SCRIPT = os.path.join(BASE_DIR, "gen_song.py")
MODEL_LABEL = "ACE-Step1.5 (MLX) — text-to-music + songs, Apache-2.0"

# --- generation defaults / limits -----------------------------------------
SAMPLE_RATE = 48000
DEF_SECONDS = 30.0
MIN_SECONDS = 4.0
MAX_SECONDS = 240.0            # 4 min ceiling keeps render time + memory sane
DEF_STEPS_INSTRUMENTAL = 8    # turbo (CFG-distilled) — fast, good for instrumentals
DEF_STEPS_VOCAL = 20          # more steps for cleaner sung vocals
MIN_STEPS = 4
MAX_STEPS = 60
DEF_SHIFT = 3.0
DEF_GUIDANCE = 1.0            # turbo model default
DEF_LM_SIZE = "0.6B"
LM_SIZES = ["0.6B", "4B"]

# Keep both a lossless master and a shareable mp3; the user's rule is "always
# keep the originals" so nothing here is auto-deleted.
MAKE_MP3 = os.environ.get("SONG_MAKE_MP3", "1") not in ("0", "false", "no", "")
KEEP_WAV = os.environ.get("SONG_KEEP_WAV", "0") not in ("0", "false", "no", "")

# --- style presets (the "genre/vibe" dropdown) ----------------------------
# Each fills the ACE-Step text prompt; the user's own words are appended.
STYLE_PRESETS = [
    {"id": "lofi", "label": "Lo-fi chill",
     "prompt": "warm lo-fi hip hop, mellow piano, soft vinyl crackle, relaxed, "
               "chill beats, cozy, downtempo"},
    {"id": "cinematic", "label": "Cinematic score",
     "prompt": "epic cinematic orchestral score, sweeping strings, powerful brass, "
               "emotional, film trailer, dramatic build"},
    {"id": "pop", "label": "Modern pop",
     "prompt": "upbeat modern pop, catchy hook, bright synths, punchy drums, "
               "radio-ready, energetic, polished production"},
    {"id": "edm", "label": "EDM / dance",
     "prompt": "energetic EDM dance track, four-on-the-floor, big synth lead, "
               "festival drop, driving bassline, euphoric"},
    {"id": "acoustic", "label": "Acoustic / folk",
     "prompt": "warm acoustic folk, fingerpicked guitar, gentle, organic, "
               "intimate, singer-songwriter, heartfelt"},
    {"id": "hiphop", "label": "Hip-hop / trap",
     "prompt": "modern hip hop trap beat, hard 808 bass, crisp hi-hats, "
               "confident, moody, hard-hitting"},
    {"id": "ambient", "label": "Ambient / calm",
     "prompt": "calm ambient soundscape, soft pads, ethereal, spacious, "
               "meditative, slow, atmospheric"},
    {"id": "rock", "label": "Rock / band",
     "prompt": "energetic rock band, driving electric guitars, live drums, "
               "powerful, anthemic, distorted, punchy"},
    {"id": "corporate", "label": "Corporate / upbeat",
     "prompt": "bright uplifting corporate background, clean guitars, motivational, "
               "positive, light percussion, optimistic"},
    {"id": "custom", "label": "Custom (my words only)", "prompt": ""},
]
STYLE_BY_ID = {s["id"]: s for s in STYLE_PRESETS}
DEF_STYLE = "lofi"

# --- vocal language dropdown ----------------------------------------------
VOCAL_LANGUAGES = [
    {"id": "en", "label": "English"},
    {"id": "es", "label": "Spanish"},
    {"id": "fr", "label": "French"},
    {"id": "de", "label": "German"},
    {"id": "it", "label": "Italian"},
    {"id": "pt", "label": "Portuguese"},
    {"id": "ja", "label": "Japanese"},
    {"id": "ko", "label": "Korean"},
    {"id": "zh", "label": "Chinese"},
    {"id": "unknown", "label": "Auto / unknown"},
]
VOCAL_LANG_IDS = {v["id"] for v in VOCAL_LANGUAGES}
DEF_VOCAL_LANG = "en"

CONTENT_TYPES = {
    ".flac": "audio/flac", ".mp3": "audio/mpeg", ".wav": "audio/wav",
}

# --- job state ------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()
_work_q = []
_work_cv = threading.Condition()

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _weights_ready() -> bool:
    return (os.path.isdir(MODEL_DIR)
            and os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors"))
            and os.path.isfile(os.path.join(MODEL_DIR, "config.json")))


def _available() -> bool:
    return _weights_ready() and os.path.isfile(GEN_SCRIPT) and os.path.isfile(VENV_PY)


# --- oMLX chat-LLM auto-unload (same pattern as omlx-video) ----------------
OMLX_URL = os.environ.get("SONG_OMLX_URL", "http://127.0.0.1:8000").rstrip("/")
AUTO_UNLOAD_LLM = os.environ.get("SONG_AUTO_UNLOAD_LLM", "1") not in ("0", "false", "no", "")


def _omlx_get(path, timeout=8):
    import urllib.request
    req = urllib.request.Request(OMLX_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _omlx_post(path, timeout=600):
    import urllib.request
    req = urllib.request.Request(OMLX_URL + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _omlx_loaded_models():
    try:
        st = _omlx_get("/v1/models/status")
    except Exception:
        return []
    out = []
    for m in st.get("models", []):
        if m.get("loaded") or m.get("is_loading"):
            out.append(m["id"])
    return out


def _omlx_model_memory_gb():
    try:
        st = _omlx_get("/v1/models/status")
        return round(float(st.get("current_model_memory", 0)) / 1e9, 1)
    except Exception:
        return None


def _free_chat_llm(jid):
    if not AUTO_UNLOAD_LLM:
        return []
    ids = _omlx_loaded_models()
    if not ids:
        return []
    _set(jid, stage="freeing memory: unloading chat model…", progress=1)
    import urllib.parse
    for mid in ids:
        try:
            _omlx_post("/v1/models/" + urllib.parse.quote(mid, safe="") + "/unload")
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 30:
        mem = _omlx_model_memory_gb()
        if mem is not None and mem < 2.0:
            break
        if not _omlx_loaded_models():
            break
        time.sleep(1)
    time.sleep(1.5)
    return ids


def _restore_chat_llm(jid, ids):
    if not ids:
        return
    _set(jid, stage="reloading chat model…", progress=98)
    import urllib.parse
    for mid in ids:
        try:
            _omlx_post("/v1/models/" + urllib.parse.quote(mid, safe="") + "/load")
        except Exception:
            pass


# --- sudoless power + memory stats (same as omlx-video) --------------------
_MACMON_CANDIDATES = ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon", "macmon")
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


def _sys_stats():
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
        out.update({k: p.get(k) for k in
                    ("cpu_watts", "gpu_watts", "total_watts", "gpu_temp")})
    return out


# --- job helpers ----------------------------------------------------------
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


def _slug(text, n=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:n].rstrip("-") or "song")


def _stem_for(opts):
    kind = "inst" if not (opts.get("lyrics") or "").strip() else "vocal"
    dur = int(round(float(opts.get("duration", DEF_SECONDS))))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = _slug(opts.get("title") or opts.get("prompt") or "song")
    return f"song_{label}_{dur}s_{kind}_seed{opts.get('seed', 0)}_{ts}"


def _ffmpeg_bin():
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"):
        if os.path.sep in c:
            if os.path.isfile(c):
                return c
        else:
            return c
    return "ffmpeg"


def _run_song(jid, opts):
    stem = _stem_for(opts)
    spec = {
        "model_dir": MODEL_DIR,
        "out_dir": OUT_DIR,
        "stem": stem,
        "text": opts["prompt"],
        "lyrics": opts.get("lyrics", ""),
        "duration": float(opts["duration"]),
        "num_steps": int(opts["num_steps"]),
        "seed": int(opts["seed"]),
        "shift": float(opts.get("shift", DEF_SHIFT)),
        "guidance_scale": float(opts.get("guidance_scale", DEF_GUIDANCE)),
        "vocal_language": opts.get("vocal_language", DEF_VOCAL_LANG),
        "lm_model_size": opts.get("lm_model_size", DEF_LM_SIZE),
        "use_lm": True,
        "make_mp3": MAKE_MP3,
        "keep_wav": KEEP_WAV,
        "ffmpeg": _ffmpeg_bin(),
    }
    spec_path = os.path.join(TMP_DIR, f"{jid}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    cmd = [VENV_PY, GEN_SCRIPT, spec_path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    result = None
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
        if line.startswith("::P:: "):
            try:
                p = json.loads(line[6:])
                _set(jid, stage=p.get("stage", ""), progress=int(p.get("progress", 0)))
            except Exception:
                pass
        elif line.startswith("::R:: "):
            try:
                result = json.loads(line[6:])
            except Exception:
                pass
    proc.wait()
    try:
        os.remove(spec_path)
    except OSError:
        pass
    if result is None or not result.get("ok"):
        err = (result or {}).get("error") if result else None
        raise RuntimeError("song generation failed: "
                           + (err or " | ".join(tail[-6:])))
    result["elapsed"] = round(time.time() - t0, 1)
    return result


def _run_job(jid):
    job = _get(jid)
    if not job:
        return
    opts = job["opts"]
    freed = []
    try:
        freed = _free_chat_llm(jid)
        _set(jid, stage="starting", progress=2)
        result = _run_song(jid, opts)
        files = result.get("files", [])
        primary = result.get("flac") or (files[0] if files else None)
        _set(jid, status="done", stage="done", progress=100, result={
            "engine": "ace-step",
            "filename": primary,
            "url": f"/files/{primary}" if primary else None,
            "files": [{"name": f, "url": f"/files/{f}"} for f in files],
            "sample_rate": result.get("sample_rate", SAMPLE_RATE),
            "duration": result.get("duration"),
            "peak": result.get("peak"),
            "load_s": result.get("load_s"),
            "gen_s": result.get("gen_s"),
            "elapsed": result.get("elapsed"),
            "metadata": result.get("metadata", {}),
            "prompt": opts.get("prompt"),
            "lyrics": opts.get("lyrics", ""),
            "seed": opts.get("seed"),
        })
    except Exception as e:
        _set(jid, status="error", stage="error", error=str(e))
    finally:
        _restore_chat_llm(jid, freed)
        if _get(jid).get("status") not in ("error",):
            _set(jid, progress=100)


# --- HTTP -----------------------------------------------------------------
SONG_UI = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Song Studio — ACE-Step</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--dim:#8b949e;--accent:#238636;--accent2:#2ea043}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
  .wrap{max-width:820px;margin:0 auto;padding:18px 18px 60px}
  h1{font-size:18px;margin:0 0 2px;display:flex;gap:8px;align-items:center}
  .sub{color:var(--dim);font-size:12px;margin:0 0 16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
  label{display:block;font-size:12px;color:var(--dim);margin:10px 0 4px;font-weight:600}
  input,select,textarea,button{font:inherit;color:var(--ink)}
  input[type=text],select,textarea{width:100%;background:#010409;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
  textarea{resize:vertical;min-height:96px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .row>div{flex:1;min-width:150px}
  .tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  .tag{background:#21262d;border:1px solid var(--line);border-radius:6px;color:#c9d1d9;padding:3px 8px;font-size:11px;cursor:pointer}
  .tag:hover{border-color:var(--accent2)}
  .toggle{display:flex;align-items:center;gap:8px;margin-top:12px}
  .toggle input{width:auto}
  .btn{background:var(--accent);border:none;border-radius:8px;color:#fff;padding:10px 18px;font-weight:600;cursor:pointer}
  .btn:hover{background:var(--accent2)}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn.ghost{background:#21262d;border:1px solid var(--line)}
  .dice{background:#21262d;border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:8px 12px;cursor:pointer}
  .prog{height:8px;background:#010409;border-radius:6px;overflow:hidden;margin-top:10px;border:1px solid var(--line)}
  .bar{height:100%;width:0;background:linear-gradient(90deg,#2ea043,#3fb950);transition:width .3s}
  .muted{color:var(--dim);font-size:12px}
  .result{margin-top:12px}
  audio{width:100%;margin-top:8px}
  .lib-item{display:flex;align-items:center;gap:10px;padding:8px 0;border-top:1px solid var(--line)}
  .lib-item .name{flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  a.dl{color:#58a6ff;font-size:12px;text-decoration:none}
  .err{color:#f85149;font-size:12px;margin-top:8px;white-space:pre-wrap}
  .hint{font-size:11px;color:var(--dim);margin-top:4px}
</style></head><body>
<div class="wrap">
  <h1>🎵 Song Studio</h1>
  <p class="sub" id="modelLine">ACE-Step1.5 — local music &amp; song generation (Apache-2.0, safe for your videos)</p>

  <div class="card">
    <label>Style</label>
    <select id="style"></select>
    <label>Describe the music <span class="muted">(mood, instruments, tempo — added to the style)</span></label>
    <input type="text" id="prompt" placeholder="e.g. rainy night, warm piano, slow and dreamy">

    <div class="toggle">
      <input type="checkbox" id="instrumental">
      <label for="instrumental" style="margin:0">Instrumental only (no vocals)</label>
    </div>

    <div id="lyricsBox">
      <label>Lyrics <span class="muted">(leave blank for instrumental; use section tags)</span></label>
      <div class="tags" id="tagRow">
        <span class="tag" data-t="[Verse]">[Verse]</span>
        <span class="tag" data-t="[Chorus]">[Chorus]</span>
        <span class="tag" data-t="[Pre-Chorus]">[Pre-Chorus]</span>
        <span class="tag" data-t="[Bridge]">[Bridge]</span>
        <span class="tag" data-t="[Hook]">[Hook]</span>
        <span class="tag" data-t="[Outro]">[Outro]</span>
      </div>
      <textarea id="lyrics" placeholder="[Verse]
Your words here
[Chorus]
The catchy part"></textarea>
    </div>

    <div class="row">
      <div>
        <label>Vocal language</label>
        <select id="lang"></select>
      </div>
      <div>
        <label>Duration: <span id="durVal">30</span>s</label>
        <input type="range" id="duration" min="4" max="240" value="30" step="1" style="width:100%">
      </div>
      <div>
        <label>Quality (steps): <span id="stepVal">auto</span></label>
        <input type="range" id="steps" min="4" max="60" value="8" step="1" style="width:100%">
        <div class="hint">More steps = cleaner (slower). 8 for instrumentals, ~20 for vocals.</div>
      </div>
    </div>

    <div class="row" style="margin-top:6px">
      <div style="flex:2">
        <label>Seed</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="seed" value="" placeholder="random" style="flex:1">
          <button class="dice" id="dice" title="Randomize">🎲</button>
        </div>
      </div>
      <div>
        <label>Planner</label>
        <select id="lm"></select>
        <div class="hint">4B = higher quality, slower.</div>
      </div>
    </div>

    <div style="margin-top:16px;display:flex;gap:10px;align-items:center">
      <button class="btn" id="go">Generate</button>
      <span class="muted" id="statusText"></span>
    </div>
    <div class="prog" id="progWrap" style="display:none"><div class="bar" id="bar"></div></div>
    <div class="err" id="err"></div>
    <div class="result" id="result"></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <strong style="font-size:13px">Library</strong>
      <button class="btn ghost" id="refresh" style="padding:5px 12px;font-size:12px">Refresh</button>
    </div>
    <div id="library"><p class="muted">Loading…</p></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const BASE=location.origin;
let INFO=null, polling=null;

function opt(sel,items,valKey,labKey,def){
  sel.innerHTML='';
  items.forEach(it=>{const o=document.createElement('option');o.value=it[valKey];o.textContent=it[labKey];if(it[valKey]===def)o.selected=true;sel.appendChild(o);});
}

async function loadInfo(){
  try{ INFO=await (await fetch(BASE+'/info')).json(); }catch(e){ $('err').textContent='Service not reachable on :8600'; return; }
  opt($('style'),INFO.styles,'id','label',INFO.style_default);
  opt($('lang'),INFO.vocal_languages,'id','label',INFO.vocal_language_default);
  opt($('lm'),INFO.lm_sizes.map(s=>({id:s,label:s})),'id','label',INFO.defaults.lm_model_size);
  $('modelLine').textContent=INFO.model;
  $('duration').max=INFO.limits.max_seconds; $('duration').min=INFO.limits.min_seconds;
  $('steps').max=INFO.limits.max_steps; $('steps').min=INFO.limits.min_steps;
  syncSteps();
}

function instrumental(){ return $('instrumental').checked || !$('lyrics').value.trim(); }
function syncSteps(){
  // If the user hasn't dragged steps, show/apply the sensible default for the mode.
  if(!$('steps').dataset.touched){
    const d = instrumental()? (INFO?INFO.defaults.num_steps_instrumental:8) : (INFO?INFO.defaults.num_steps_vocal:20);
    $('steps').value=d; $('stepVal').textContent=d+' (auto)';
  } else { $('stepVal').textContent=$('steps').value; }
}
function updLyricsBox(){ $('lyricsBox').style.opacity=$('instrumental').checked?0.4:1; $('lyrics').disabled=$('instrumental').checked; syncSteps(); }

$('duration').oninput=()=>$('durVal').textContent=$('duration').value;
$('steps').oninput=()=>{$('steps').dataset.touched='1';$('stepVal').textContent=$('steps').value;};
$('instrumental').onchange=updLyricsBox;
$('lyrics').oninput=syncSteps;
$('dice').onclick=()=>{$('seed').value=Math.floor(Math.random()*16777215);};
document.querySelectorAll('.tag').forEach(t=>t.onclick=()=>{
  const ta=$('lyrics'); const ins=(ta.value && !ta.value.endsWith('\n')?'\n':'')+t.dataset.t+'\n';
  ta.value+=ins; ta.focus(); syncSteps();
});

async function generate(){
  $('err').textContent=''; $('result').innerHTML='';
  const body={
    style:$('style').value,
    prompt:$('prompt').value.trim(),
    instrumental:$('instrumental').checked,
    lyrics:$('instrumental').checked?'':$('lyrics').value,
    vocal_language:$('lang').value,
    duration:parseFloat($('duration').value),
    num_steps:parseInt($('steps').value),
    lm_model_size:$('lm').value,
  };
  const s=$('seed').value.trim(); if(s!=='')body.seed=parseInt(s);
  $('go').disabled=true; $('statusText').textContent='Submitting…';
  $('progWrap').style.display='block'; $('bar').style.width='2%';
  let jid;
  try{
    const r=await (await fetch(BASE+'/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
    if(r.error){throw new Error(r.error);} jid=r.job_id;
    if(r.seed!=null && s===''){$('seed').value=r.seed;}
  }catch(e){ fail(e.message); return; }
  poll(jid);
}
function fail(msg){ $('go').disabled=false; $('statusText').textContent=''; $('progWrap').style.display='none'; $('err').textContent='Error: '+msg; }

function poll(jid){
  clearInterval(polling);
  polling=setInterval(async()=>{
    let d; try{ d=await (await fetch(BASE+'/status?id='+encodeURIComponent(jid))).json(); }catch(e){return;}
    if(d.error){clearInterval(polling);fail(d.error);return;}
    $('bar').style.width=(d.progress||0)+'%';
    $('statusText').textContent=(d.stage||'')+' · '+(d.progress||0)+'%';
    if(d.status==='done'&&d.result){ clearInterval(polling); done(d.result); }
  },1500);
}

function done(res){
  $('go').disabled=false; $('progWrap').style.display='none';
  $('statusText').textContent='Done in '+(res.elapsed||'?')+'s (gen '+(res.gen_s||'?')+'s)';
  const flac=(res.files||[]).find(f=>f.name.endsWith('.flac'));
  const mp3=(res.files||[]).find(f=>f.name.endsWith('.mp3'));
  const play=mp3||flac;
  let html='<audio controls autoplay src="'+BASE+play.url+'"></audio>';
  html+='<div class="muted" style="margin-top:6px">'+ (play.name) +'</div>';
  html+='<div style="margin-top:6px;display:flex;gap:14px">';
  if(flac)html+='<a class="dl" href="'+BASE+flac.url+'" download>⬇ FLAC (lossless)</a>';
  if(mp3)html+='<a class="dl" href="'+BASE+mp3.url+'" download>⬇ MP3</a>';
  html+='</div>';
  if(res.metadata&&res.metadata.bpm){html+='<div class="muted" style="margin-top:6px">~'+res.metadata.bpm+' BPM · '+(res.metadata.keyscale||'')+' · '+(res.metadata.genres||'')+'</div>';}
  $('result').innerHTML=html;
  loadLibrary();
}

async function loadLibrary(){
  let d; try{ d=await (await fetch(BASE+'/library?limit=40')).json(); }catch(e){ $('library').innerHTML='<p class="muted">—</p>'; return; }
  const songs=d.songs||[];
  if(!songs.length){$('library').innerHTML='<p class="muted">No songs yet.</p>';return;}
  $('library').innerHTML=songs.map(s=>{
    const f=s.files||{}; const play=(f.mp3||f.flac||f.wav);
    let links=''; ['flac','mp3','wav'].forEach(k=>{if(f[k])links+='<a class="dl" href="'+BASE+f[k].url+'" download>'+k.toUpperCase()+'</a> ';});
    return '<div class="lib-item"><button class="tag" onclick="playLib(\''+(play?BASE+play.url:'')+'\')">▶</button>'+
      '<span class="name" title="'+s.stem+'">'+s.stem+'</span><span class="muted">'+(s.when||'')+'</span>'+
      '<span style="display:flex;gap:8px">'+links+'</span></div>';
  }).join('');
}
window.playLib=url=>{ if(!url)return; let a=$('libAudio'); if(!a){a=document.createElement('audio');a.id='libAudio';a.controls=true;a.style.width='100%';a.style.marginTop='8px';$('library').prepend(a);} a.src=url;a.play(); };

$('go').onclick=generate;
$('refresh').onclick=loadLibrary;
loadInfo(); loadLibrary();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "omlx-song/1.0"

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
        if path in ("/", "/ui", "/index.html"):
            body = SONG_UI.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            return self._json(200, {
                "ok": True,
                "available": _available(),
                "weights_ready": _weights_ready(),
                "model": MODEL_LABEL,
                "model_dir": MODEL_DIR,
                "queue": len(_work_q),
                "auto_unload_llm": AUTO_UNLOAD_LLM,
                "loaded_llms": _omlx_loaded_models() if AUTO_UNLOAD_LLM else [],
            })
        if path == "/info":
            return self._json(200, {
                "model": MODEL_LABEL,
                "available": _available(),
                "weights_ready": _weights_ready(),
                "sample_rate": SAMPLE_RATE,
                "styles": STYLE_PRESETS,
                "style_default": DEF_STYLE,
                "vocal_languages": VOCAL_LANGUAGES,
                "vocal_language_default": DEF_VOCAL_LANG,
                "lm_sizes": LM_SIZES,
                "defaults": {
                    "duration_seconds": DEF_SECONDS,
                    "num_steps_instrumental": DEF_STEPS_INSTRUMENTAL,
                    "num_steps_vocal": DEF_STEPS_VOCAL,
                    "shift": DEF_SHIFT,
                    "guidance_scale": DEF_GUIDANCE,
                    "lm_model_size": DEF_LM_SIZE,
                    "style": DEF_STYLE,
                    "vocal_language": DEF_VOCAL_LANG,
                },
                "limits": {
                    "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS,
                    "min_steps": MIN_STEPS, "max_steps": MAX_STEPS,
                },
                "formats": ["flac"] + (["mp3"] if MAKE_MP3 else []),
            })
        if path == "/stats":
            return self._json(200, _sys_stats())
        if path == "/status":
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            job = _get(jid)
            if not job:
                return self._json(404, {"error": "unknown job"})
            return self._json(200, {k: job.get(k) for k in
                                    ("status", "stage", "progress", "result", "error")})
        if path == "/library":
            qs = parse_qs(urlparse(self.path).query)
            limit = int((qs.get("limit") or ["60"])[0])
            return self._json(200, {"songs": _list_library(limit)})
        if path.startswith("/files/"):
            return self._serve_file(os.path.basename(path))
        return self._json(404, {"error": "not found"})

    def _serve_file(self, name):
        fp = os.path.join(OUT_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in CONTENT_TYPES or not os.path.isfile(fp):
            return self._json(404, {"error": "not found"})
        data = open(fp, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[ext])
        self._cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/generate":
            if not _available():
                return self._json(400, {"error": "ACE-Step model/venv not ready"})
            d = self._read_body()
            user_prompt = (d.get("prompt") or "").strip()
            style_id = str(d.get("style") or DEF_STYLE)
            style = STYLE_BY_ID.get(style_id, STYLE_BY_ID[DEF_STYLE])
            # Compose the ACE-Step text prompt: style preset + the user's own words.
            parts = [p for p in (style.get("prompt", ""), user_prompt) if p]
            prompt = ", ".join(parts).strip()
            if not prompt:
                return self._json(400, {"error": "Describe the music, or pick a style"})

            lyrics = (d.get("lyrics") or "").strip()
            instrumental = bool(d.get("instrumental")) or not lyrics
            if instrumental:
                lyrics = ""

            try:
                duration = float(d.get("duration", DEF_SECONDS))
            except (TypeError, ValueError):
                duration = DEF_SECONDS
            duration = max(MIN_SECONDS, min(MAX_SECONDS, duration))

            default_steps = DEF_STEPS_INSTRUMENTAL if instrumental else DEF_STEPS_VOCAL
            try:
                num_steps = int(d.get("num_steps") or default_steps)
            except (TypeError, ValueError):
                num_steps = default_steps
            num_steps = max(MIN_STEPS, min(MAX_STEPS, num_steps))

            vocal_language = str(d.get("vocal_language") or DEF_VOCAL_LANG)
            if vocal_language not in VOCAL_LANG_IDS:
                vocal_language = DEF_VOCAL_LANG

            lm_size = str(d.get("lm_model_size") or DEF_LM_SIZE)
            if lm_size not in LM_SIZES:
                lm_size = DEF_LM_SIZE

            seed = d.get("seed")
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = int.from_bytes(os.urandom(3), "big")

            opts = {
                "engine": "ace-step",
                "prompt": prompt[:1500],
                "title": (d.get("title") or user_prompt or style.get("label", "song")),
                "style": style_id,
                "lyrics": lyrics[:6000],
                "instrumental": instrumental,
                "duration": duration,
                "num_steps": num_steps,
                "seed": seed,
                "shift": float(d.get("shift", DEF_SHIFT)),
                "guidance_scale": float(d.get("guidance_scale", DEF_GUIDANCE)),
                "vocal_language": vocal_language,
                "lm_model_size": lm_size,
            }
            jid = "song_" + uuid.uuid4().hex[:12]
            _set(jid, status="running", stage="queued", progress=0,
                 result=None, error=None, opts=opts)
            with _work_cv:
                _work_q.append(jid)
                _work_cv.notify()
            reclaim = _omlx_model_memory_gb() if AUTO_UNLOAD_LLM else 0
            return self._json(200, {"ok": True, "job_id": jid, "engine": "ace-step",
                                    "instrumental": instrumental, "seed": seed,
                                    "duration": duration, "num_steps": num_steps,
                                    "reclaim_gb": reclaim})
        return self._json(404, {"error": "not found"})


def _list_library(limit=60):
    try:
        names = [n for n in os.listdir(OUT_DIR)
                 if os.path.splitext(n)[1].lower() in CONTENT_TYPES]
    except OSError:
        return []
    # Group by stem so flac+mp3 of one song show as a single entry.
    stems = {}
    for n in names:
        stem, ext = os.path.splitext(n)
        fp = os.path.join(OUT_DIR, n)
        try:
            mt = os.path.getmtime(fp)
        except OSError:
            continue
        e = stems.setdefault(stem, {"stem": stem, "mtime": mt, "files": {}})
        e["mtime"] = max(e["mtime"], mt)
        e["files"][ext.lstrip(".")] = {"name": n, "url": f"/files/{n}"}
    out = sorted(stems.values(), key=lambda e: e["mtime"], reverse=True)[:limit]
    for e in out:
        e["when"] = datetime.fromtimestamp(e["mtime"]).strftime("%Y-%m-%d %H:%M")
    return out


def main():
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_power_worker, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"omlx-song ready at http://{HOST}:{PORT} "
          f"(weights_ready={_weights_ready()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
