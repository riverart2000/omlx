#!/usr/bin/env python3
"""
Persistent high-quality TTS Studio server (mlx-audio).

Two engines, lazy-loaded and cached on a single dedicated worker thread
(which avoids the MLX "no Stream(gpu, 0)" cross-thread error):

  * voxtral  - fast, 20 built-in language/style presets
  * higgs    - Higgs Audio v2 (Apache-2.0): very realistic + voice cloning

Production features for YouTube voiceovers:
  * Voice cloning (Higgs) from a reference clip (auto-transcribed once)
  * Mastering chain via ffmpeg: loudness normalize, sample rate, channels,
    fades, silence trim, mp3 bitrate
  * Multi-take: render N seeded variations, audition, keep the best
  * History library: every render saved with text + params + seed
  * Long-script chunking: split, render, stitch with pauses

Endpoints:
  GET  /                 -> Studio web page
  GET  /health           -> {ready, model, loaded}
  GET  /engines          -> {engines, formats, lufs, refs}
  GET  /history          -> [...]
  GET  /refs             -> [...]
  GET  /files/<name>     -> serves a generated/reference audio file
  POST /generate         -> render (engine, clone, mastering, takes, script)
  POST /history/delete   -> {id}
  POST /history/favorite -> {id, value}
  POST /refs/upload      -> {name, data(base64), ref_text?}
  POST /refs/delete      -> {id}
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "8200"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
REF_DIR = os.path.join(BASE_DIR, "refs")
HISTORY_FILE = os.path.join(OUT_DIR, "history.json")
REFS_FILE = os.path.join(REF_DIR, "refs.json")

STT_REPO = os.environ.get("TTS_STT", "mlx-community/whisper-large-v3-turbo-asr-fp16")

VOXTRAL_VOICES = [
    "neutral_female", "neutral_male", "casual_female", "casual_male",
    "cheerful_female",
    "fr_female", "fr_male", "es_female", "es_male", "de_female", "de_male",
    "it_female", "it_male", "pt_female", "pt_male", "nl_female", "nl_male",
    "hi_female", "hi_male", "ar_male",
]
_VOICE_LANG = {"fr": "fr", "es": "es", "de": "de", "it": "it",
               "pt": "pt", "nl": "nl", "hi": "hi", "ar": "ar"}

ENGINES = {
    "higgs": {
        "label": "Higgs v2 — realistic + voice cloning",
        "repo": os.environ.get("TTS_HIGGS", "mlx-community/higgs-audio-v2-3B-mlx-q8"),
        "clone": True,
        "presets": False,
    },
    "voxtral": {
        "label": "Voxtral — fast, 20 presets",
        "repo": os.environ.get("TTS_VOXTRAL", "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"),
        "clone": False,
        "presets": True,
    },
    "kokoro": {
        "label": "Kokoro 82M — fast, natural (chat voice)",
        "repo": os.environ.get("TTS_KOKORO", "mlx-community/Kokoro-82M-bf16"),
        "clone": False,
        "presets": False,
        "kokoro": True,
    },
}
DEFAULT_ENGINE = "higgs"

# Default Kokoro voice for the in-chat reader. Overridable per request.
KOKORO_DEFAULT_VOICE = os.environ.get("TTS_KOKORO_VOICE", "af_heart")
# Which Kokoro language prefixes to expose. English only by default
# (a=American, b=British); override with e.g. TTS_KOKORO_LANGS="abj".
KOKORO_LANGS = (os.environ.get("TTS_KOKORO_LANGS", "ab") or "ab").lower()

# Curated fallback if the model's voice folder can't be enumerated. The live
# list is discovered from the downloaded model at runtime (see _kokoro_voices).
KOKORO_VOICES_FALLBACK = [
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
    "af_nova", "af_sky", "am_michael", "am_fenrir", "am_puck", "am_adam",
    "am_echo", "am_eric", "am_liam", "am_onyx",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
]
_kokoro_voices_cache = None


def _kokoro_voices() -> list:
    """Discover installed Kokoro voice names from the model's voices folder
    (HF cache snapshot), filtered to the languages in KOKORO_LANGS (English by
    default). Cached after first lookup; falls back to a curated list if the
    folder isn't found."""
    global _kokoro_voices_cache
    if _kokoro_voices_cache is not None:
        return _kokoro_voices_cache
    voices = []
    try:
        from glob import glob
        repo = ENGINES["kokoro"]["repo"].replace("/", "--")
        roots = [
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}/snapshots"),
            os.path.expanduser(f"~/.omlx/models/{ENGINES['kokoro']['repo']}"),
        ]
        for root in roots:
            hits = glob(os.path.join(root, "**", "voices", "*.safetensors"),
                        recursive=True)
            if hits:
                voices = sorted({os.path.splitext(os.path.basename(h))[0]
                                 for h in hits})
                break
    except Exception:
        voices = []
    voices = voices or list(KOKORO_VOICES_FALLBACK)
    # Keep only the enabled languages (by voice-name prefix letter).
    filtered = [v for v in voices if v[:1].lower() in KOKORO_LANGS]
    _kokoro_voices_cache = filtered or voices
    return _kokoro_voices_cache


def _kokoro_lang(voice: str) -> str:
    """Kokoro derives its G2P language from the voice-name prefix letter
    (a=American, b=British, j=Japanese, z=Mandarin, e=Spanish, f=French,
    h=Hindi, i=Italian, p=Brazilian Portuguese). Fall back to American."""
    c = (voice or "a")[:1].lower()
    return c if c in "abjzefhip" else "a"


_kokoro_patched = False


def _patch_kokoro_sinegen() -> None:
    """mlx_audio's Kokoro vocoder (istftnet.SineGen) crashes on certain inputs
    with `[broadcast_shapes] Shapes (1,N,1) and (1,N+hop,9) cannot be broadcast`.

    Root cause: SineGen._f02sine downsamples the F0 by 1/upsample_scale then
    re-upsamples by upsample_scale; when the sequence length isn't divisible by
    upsample_scale the round-trip returns a length that's off by one hop, so the
    resulting `sine_waves` no longer matches `uv` (which stays at F0 resolution).
    ~20% of normal chat sentences hit this and fail outright.

    Fix: force `sine_waves` back to `uv`'s (true F0) length before combining —
    exactly the length the non-crashing path already produces, so audio is
    unchanged. Patched at the class level once, before any Kokoro generation."""
    global _kokoro_patched
    if _kokoro_patched:
        return
    try:
        import mlx.core as mx
        from mlx_audio.tts.models.kokoro import istftnet as _istft
    except Exception as e:  # module not available yet / import error — non-fatal
        print(f"[kokoro] SineGen patch skipped: {e}", flush=True)
        return

    def _sinegen_call(self, f0):
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        L = uv.shape[1]
        t = sine_waves.shape[1]
        if t > L:
            sine_waves = sine_waves[:, :L, :]
        elif t < L:
            sine_waves = mx.pad(sine_waves, ((0, 0), (0, L - t), (0, 0)))
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    _istft.SineGen.__call__ = _sinegen_call
    _kokoro_patched = True
    print("[kokoro] SineGen length-mismatch patch applied", flush=True)

FORMATS = ["wav", "flac", "mp3"]
MAX_CHARS = 20000
SAMPLE_RATES = [24000, 44100, 48000]
LUFS_TARGETS = {"off": None, "youtube": -14.0, "podcast": -16.0, "broadcast": -23.0}
VIRAL_LENGTH_TARGETS = {
    "auto": "",
    "original": "Preserve the original video length and flow. Do not force a shorter cut; keep pacing natural while still improving clarity/retention.",
    "15_30": "Target a tight short-form cut around 15-30 seconds. Prioritize strongest hook and payoff only.",
    "30_45": "Target a concise short-form cut around 30-45 seconds. Keep one core idea and one strong payoff.",
    "45_60": "Target around 45-60 seconds. Maintain momentum while preserving a clear mini-story arc.",
    "60_90": "Target around 60-90 seconds. Allow a fuller story while keeping high retention pacing.",
    "90_120": "Target around 90-120 seconds. Keep sections structured, but preserve depth and context.",
}

# Default disfluency / filler tokens to strip in media cleanup. Kept tight on
# purpose (classic vocal fillers only) so real words like "a"/"so"/"like" are
# never removed unless the user adds them explicitly.
DEFAULT_FILLERS = [
    "um", "umm", "uhm", "uh", "uhh", "er", "err", "erm",
    "ah", "ahh", "eh", "hmm", "mhm", "mm-hmm",
]
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CI_ENHANCE_BIN = os.path.join(BASE_DIR, "tools", "ci_enhance")
CAPTION_BIN = os.path.join(BASE_DIR, "tools", "caption_render.py")
# Bump when caption/card layout or styling in caption_render.py changes so the
# overlay (ovl_*) cache is invalidated and frames are re-rendered.
OVERLAY_STYLE_VERSION = 3
ASPECTS = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1),
           "4:5": (4, 5), "original": None}
VOCAB_PATH = os.path.join(BASE_DIR, "voice_vocab.json")
CAPTION_LLM = ("porschefreak--Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-"
               "abliterated-mlx-6Bit")
# Selectable director/caption LLMs served by omlx-server (:8000). The user can
# switch at runtime from the Viral Studio UI to compare model behaviour on the
# same edit (plan decisions, highlight selection, captions, revise).
LLM_CHOICES = [
    {"id": CAPTION_LLM,
     "label": "Qwen3.6 35B · Claude-Opus abliterated (default)"},
    {"id": "Ornith-1.0-35B-8bit", "label": "Ornith 1.0 · 35B 8bit"},
]
_llm_lock = threading.Lock()
_current_llm = CAPTION_LLM


def _get_llm() -> str:
    with _llm_lock:
        return _current_llm


def _set_llm(model_id: str) -> str:
    """Switch the active director/caption LLM. Only ids in LLM_CHOICES allowed."""
    global _current_llm
    if model_id not in {c["id"] for c in LLM_CHOICES}:
        raise ValueError(f"unknown model: {model_id}")
    with _llm_lock:
        _current_llm = model_id
    return model_id


OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MEMORY_URL = os.environ.get("MEM_URL", "http://127.0.0.1:8300")
TRANSCRIPT_COLLECTION = "video_transcripts"


def _load_vocab() -> dict:
    try:
        with open(VOCAB_PATH) as f:
            d = json.load(f)
            return {"vocabulary": d.get("vocabulary") or [],
                    "corrections": d.get("corrections") or {}}
    except Exception:
        return {"vocabulary": [], "corrections": {}}


def _vocab_prompt() -> str:
    """Names/terms to bias Whisper toward correct spelling."""
    names = _load_vocab()["vocabulary"]
    return (" ".join(names) + ".") if names else ""


def _apply_corrections(text: str) -> str:
    """Fix known mistranscriptions (e.g. 'Joe Baines' -> 'Joe Bains')."""
    if not text:
        return text
    for wrong, right in _load_vocab()["corrections"].items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)


def _lang_for(voice: str) -> str:
    return _VOICE_LANG.get((voice or "").split("_", 1)[0], "en")


# --------------------------------------------------------------------------
# Worker thread: all model loading + generation happens here, serialized.
# --------------------------------------------------------------------------
_jobs: "queue.Queue" = queue.Queue()
_models: dict = {}          # engine -> loaded model module
_stt = None                 # cached STT model for reference transcription
_worker_started = threading.Event()
_load_errors: dict = {}     # engine -> last load error string


def _load_engine(engine: str):
    if engine in _models:
        return _models[engine]
    from mlx_audio.tts.utils import load
    repo = ENGINES[engine]["repo"]
    if ENGINES[engine].get("kokoro"):
        _patch_kokoro_sinegen()   # harden the vocoder before first synthesis
    model = load(repo)
    _models[engine] = model
    return model


WARM_KOKORO = os.environ.get("TTS_WARM_KOKORO", "1") not in ("0", "false", "no")


def _warm_kokoro() -> None:
    """Preload Kokoro and JIT-compile the vocoder for each enabled English
    language (US/UK) at startup, so the first in-chat /say is fast instead of
    paying a ~15-30s cold model load + per-language pipeline build."""
    if not WARM_KOKORO or "kokoro" not in ENGINES:
        return
    try:
        from mlx_audio.tts.generate import generate_audio
        model = _load_engine("kokoro")
        voices = _kokoro_voices()
        seen = set()
        for v in voices:
            lang = _kokoro_lang(v)
            if lang in seen:
                continue
            seen.add(lang)
            prefix = "warm_" + uuid.uuid4().hex[:8]
            try:
                generate_audio(
                    text="Ready.", model=model, voice=v, lang_code=lang,
                    speed=1.0, audio_format="wav", output_path=OUT_DIR,
                    file_prefix=prefix, join_audio=True, verbose=False,
                )
            finally:
                try:
                    os.remove(os.path.join(OUT_DIR, prefix + ".wav"))
                except OSError:
                    pass
        print(f"[kokoro] warmed ({', '.join(sorted(seen))})", flush=True)
    except Exception as e:  # non-fatal: falls back to lazy load on first call
        print(f"[kokoro] warm-up skipped: {type(e).__name__}: {e}", flush=True)


def _load_stt():
    global _stt
    if _stt is None:
        from mlx_audio.stt import load as load_stt
        _stt = load_stt(STT_REPO)
    return _stt


_denoiser = None            # cached DeepFilterNet speech-enhancement model
DENOISE_REPO = os.environ.get("TTS_DENOISE", "mlx-community/DeepFilterNet-mlx")
DENOISE_VERSION = os.environ.get("TTS_DENOISE_VERSION", "v3")


def _denoise_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(
            "mlx_audio.sts.models.deepfilternet") is not None
    except Exception:
        return False


def _load_denoiser():
    global _denoiser
    if _denoiser is None:
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel
        _denoiser = DeepFilterNetModel.from_pretrained(
            DENOISE_REPO, subfolder=DENOISE_VERSION)
    return _denoiser


def _worker() -> None:
    _worker_started.set()
    # Warm the default engine so the first request is fast.
    try:
        _load_engine(DEFAULT_ENGINE)
    except Exception as e:  # pragma: no cover
        _load_errors[DEFAULT_ENGINE] = f"{type(e).__name__}: {e}"
    # Warm Kokoro (in-chat voice) so the first /say isn't a cold load.
    try:
        _warm_kokoro()
    except Exception as e:  # pragma: no cover
        _load_errors["kokoro"] = f"{type(e).__name__}: {e}"

    while True:
        job = _jobs.get()
        if job is None:
            break
        kind, params, result, done = job
        try:
            if kind == "gen":
                out = _do_generate(params)
                result["raw"] = out["raw"]
                result["call"] = out["call"]
            elif kind == "transcribe":
                result["text"] = _do_transcribe(params["path"])
            elif kind == "denoise":
                result["out"] = _do_denoise(
                    params["in"], params["out"], params.get("sr", 48000))
            elif kind == "transcribe_words":
                result["data"] = _do_transcribe_words(
                    params["path"], params.get("initial_prompt"))
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()


def _do_generate(p: dict) -> dict:
    """Generate one raw 24 kHz wav segment; returns {raw, call}."""
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio

    engine = p["engine"]
    model = _load_engine(engine)
    if p.get("seed") is not None:
        try:
            mx.random.seed(int(p["seed"]))
        except Exception:
            pass

    prefix = "seg_" + uuid.uuid4().hex[:12]
    kw = dict(
        text=p["text"],
        model=model,
        audio_format="wav",
        output_path=OUT_DIR,
        file_prefix=prefix,
        join_audio=True,
        verbose=False,
        temperature=p["temperature"],
        top_p=p["top_p"],
        top_k=p["top_k"],
    )
    if engine == "voxtral":
        kw["voice"] = p["voice"]
        kw["lang_code"] = _lang_for(p["voice"])
        kw["max_tokens"] = p["max_len"]
    elif engine == "kokoro":
        kw["voice"] = p["voice"]
        kw["lang_code"] = _kokoro_lang(p["voice"])
        kw["speed"] = float(p.get("speed") or 1.0)
        kw["max_tokens"] = None
    else:  # higgs
        kw["max_tokens"] = None
        kw["max_new_frames"] = p["max_len"]
        if p.get("ref_audio"):
            kw["ref_audio"] = p["ref_audio"]
            kw["ref_text"] = p.get("ref_text") or None
            kw["stt_model"] = STT_REPO

    call = _call_repr(engine, kw, p)
    generate_audio(**kw)
    raw = os.path.join(OUT_DIR, prefix + ".wav")
    if not os.path.exists(raw):
        raise RuntimeError("generation produced no output file")
    return {"raw": raw, "call": call}


def _call_repr(engine: str, kw: dict, p: dict) -> dict:
    """Build a JSON-safe snapshot of the exact model call for the UI."""
    repo = ENGINES[engine]["repo"]
    args = {}
    for k, v in kw.items():
        if k == "model":
            continue
        if k == "ref_audio" and v:
            args[k] = os.path.basename(v)
        elif k == "text":
            args[k] = v if len(v) <= 600 else v[:600] + " …"
        elif k == "ref_text" and v and len(v) > 200:
            args[k] = v[:200] + " …"
        else:
            args[k] = v
    mode = ("voice_clone" if (engine == "higgs" and kw.get("ref_audio"))
            else "smart_voice" if engine == "higgs" else "preset")
    order = ["text", "voice", "lang_code", "ref_audio", "ref_text",
             "temperature", "top_p", "top_k", "max_tokens", "max_new_frames",
             "stt_model", "audio_format", "join_audio"]
    parts = []
    for k in order:
        if k in args and args[k] is not None:
            parts.append(f"{k}={args[k]!r}")
    pretty = f"generate_audio(model=<{repo}>,\n    " + ",\n    ".join(parts) + ")"
    return {"fn": "generate_audio", "engine": engine, "model": repo,
            "mode": mode, "seed": p.get("seed"), "args": args, "pretty": pretty}


def _do_transcribe(path: str) -> str:
    import soundfile as sf
    import numpy as np
    stt = _load_stt()
    audio, sr = sf.read(path)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return _apply_corrections((stt.generate(audio).text or "").strip())


def _do_transcribe_words(path: str, initial_prompt: str | None = None) -> dict:
    """Transcribe with word-level timestamps. Whisper assumes 16 kHz, so the
    caller MUST pass a 16 kHz mono wav for the timestamps to be real-time
    accurate. Returns {text, duration, words:[{word,start,end,prob}]}."""
    import soundfile as sf
    import numpy as np
    stt = _load_stt()
    audio, sr = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    kw = {"word_timestamps": True}
    vocab = _vocab_prompt()
    prompt = " ".join(p for p in [vocab, initial_prompt] if p).strip()
    if prompt:
        kw["initial_prompt"] = prompt
    res = stt.generate(audio, **kw)
    words = []
    for seg in (getattr(res, "segments", None) or []):
        for w in (seg.get("words") or []):
            words.append({
                "word": _apply_corrections(w.get("word", "")),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "prob": float(w.get("probability", 0.0)),
            })
    return {"text": _apply_corrections((getattr(res, "text", "") or "").strip()),
            "duration": (len(audio) / float(sr) if sr else 0.0),
            "words": words}


def _do_denoise(in_path: str, out_path: str, sr_out: int = 48000) -> str:
    """Suppress background noise with DeepFilterNet (operates at 48 kHz mono).
    Resamples in/out via ffmpeg for quality; writes out_path at sr_out."""
    import soundfile as sf
    import numpy as np
    model = _load_denoiser()
    tmp48 = in_path + ".df48.wav"
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
          "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", tmp48])
    clean48 = in_path + ".dfclean48.wav"
    try:
        audio, _sr = sf.read(tmp48, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        enhanced = np.asarray(
            model.enhance_array(np.asarray(audio, dtype=np.float32)),
            dtype=np.float32)
        sf.write(clean48, enhanced, 48000)
    finally:
        _safe_rm(tmp48)
    try:
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", clean48,
              "-ar", str(sr_out), "-ac", "1", "-c:a", "pcm_s16le", out_path])
    finally:
        _safe_rm(clean48)
    return out_path


def _enqueue(kind: str, params: dict, timeout: float = 900) -> dict:
    result: dict = {}
    done = threading.Event()
    _jobs.put((kind, params, result, done))
    if not done.wait(timeout=timeout):
        return {"error": f"timed out after {int(timeout)}s"}
    return result


def _denoise_file(in_path: str, out_path: str, sr_out: int = 48000) -> str:
    """Run a denoise job on the model worker (MLX work stays serialized)."""
    res = _enqueue("denoise", {"in": in_path, "out": out_path, "sr": sr_out},
                   timeout=600)
    if "error" in res:
        raise RuntimeError(res["error"])
    return res["out"]


# --------------------------------------------------------------------------
# Audio helpers (ffmpeg) - run off the worker thread.
# --------------------------------------------------------------------------
def _run(cmd: list) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        detail = (p.stderr or p.stdout or "").strip()
        tail = "\n".join(detail.splitlines()[-20:]) if detail else "(no stderr)"
        raise RuntimeError(f"command failed (exit {p.returncode}):\n{tail}")


def _fmt_cmd(cmd) -> str:
    """Render an argv list as a copy-pasteable shell command string."""
    if not cmd:
        return ""
    import shlex
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _curl_for(payload: dict) -> str:
    """Build a copy-pasteable curl that reproduces this /generate request."""
    body = json.dumps(payload, ensure_ascii=False)
    squoted = "'" + body.replace("'", "'\\''") + "'"
    return (f"curl -s http://127.0.0.1:{PORT}/generate \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d {squoted}")


def _mean_volume_db(path: str) -> float:
    """Return mean volume in dBFS via ffmpeg volumedetect. Near-silent
    degenerate takes come back around -70 dB or lower; normal speech is
    roughly -35 to -15 dB. Returns -120.0 on failure / pure silence."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        ).stderr
        for line in out.splitlines():
            if "mean_volume:" in line:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
    except Exception:
        pass
    return -120.0


def _audio_seconds(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if out:
            return round(float(out), 2)
    except Exception:
        pass
    try:
        with wave.open(path, "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return 0.0


def _make_silence(ms: int, path: str, sr: int = 24000) -> None:
    secs = max(0, ms) / 1000.0
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"anullsrc=r={sr}:cl=mono", "-t", f"{secs:.3f}",
          "-ar", str(sr), "-ac", "1", path])


def _concat_with_pauses(seg_paths: list, gaps_ms: list, out_path: str,
                        sr: int = 24000) -> None:
    """Concatenate raw wavs, inserting a silence of gaps_ms[i] before seg i+1."""
    parts = []
    tmp_sil = []
    for i, seg in enumerate(seg_paths):
        if i > 0:
            gap = gaps_ms[i - 1] if i - 1 < len(gaps_ms) else 0
            if gap > 0:
                sil = os.path.join(OUT_DIR, f"_sil_{uuid.uuid4().hex[:8]}.wav")
                _make_silence(gap, sil, sr)
                tmp_sil.append(sil)
                parts.append(sil)
        parts.append(seg)
    listfile = os.path.join(OUT_DIR, f"_concat_{uuid.uuid4().hex[:8]}.txt")
    with open(listfile, "w") as f:
        for pth in parts:
            f.write(f"file '{pth}'\n")
    try:
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
              "-safe", "0", "-i", listfile, "-ar", str(sr), "-ac", "1",
              out_path])
    finally:
        for pth in tmp_sil + [listfile]:
            try:
                os.remove(pth)
            except Exception:
                pass


def _master(raw_path: str, out_path: str, opts: dict) -> None:
    """Apply the mastering chain and encode to the final format."""
    fmt = opts["format"]
    filters = []
    speed = opts.get("speed", 1.0)
    if abs(speed - 1.0) > 1e-3:
        filters.append(f"atempo={max(0.5, min(2.0, speed)):.4f}")
    if opts.get("trim"):
        sr_thresh = "start_periods=1:start_silence=0.08:start_threshold=-45dB:detection=peak"
        filters.append(f"silenceremove={sr_thresh}")
        filters.append("areverse")
        filters.append(f"silenceremove={sr_thresh}")
        filters.append("areverse")
    lufs = opts.get("lufs")
    if lufs is not None:
        filters.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
    fin = opts.get("fade_in", 0)
    if fin and fin > 0:
        filters.append(f"afade=t=in:st=0:d={fin/1000.0:.3f}")
    fout = opts.get("fade_out", 0)
    if fout and fout > 0:
        d = fout / 1000.0
        filters.append(f"areverse,afade=t=in:st=0:d={d:.3f},areverse")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ar", str(opts.get("sample_rate", 48000)),
            "-ac", str(opts.get("channels", 1))]
    if fmt == "mp3":
        cmd += ["-b:a", opts.get("mp3_bitrate", "192k")]
    elif fmt == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    cmd += [out_path]
    _run(cmd)
    return cmd


# --------------------------------------------------------------------------
# History + reference library (JSON-backed)
# --------------------------------------------------------------------------
_store_lock = threading.Lock()


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _history_add(entry: dict) -> None:
    with _store_lock:
        hist = _read_json(HISTORY_FILE, [])
        hist.insert(0, entry)
        hist = hist[:500]
        _write_json(HISTORY_FILE, hist)


def _history_list() -> list:
    return _read_json(HISTORY_FILE, [])


def _history_delete(eid: str) -> bool:
    with _store_lock:
        hist = _read_json(HISTORY_FILE, [])
        keep, removed = [], None
        for h in hist:
            if h.get("id") == eid:
                removed = h
            else:
                keep.append(h)
        if removed:
            _write_json(HISTORY_FILE, keep)
            f = os.path.join(OUT_DIR, removed.get("filename", ""))
            if removed.get("filename") and os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
            return True
        return False


def _history_favorite(eid: str, value: bool) -> bool:
    with _store_lock:
        hist = _read_json(HISTORY_FILE, [])
        ok = False
        for h in hist:
            if h.get("id") == eid:
                h["favorite"] = bool(value)
                ok = True
        if ok:
            _write_json(HISTORY_FILE, hist)
        return ok


def _refs_list() -> list:
    return _read_json(REFS_FILE, [])


def _ref_get(ref_id: str):
    for r in _refs_list():
        if r.get("id") == ref_id:
            return r
    return None


def _ref_add(name: str, b64: str, ref_text: str | None,
             denoise: bool = False) -> dict:
    rid = "ref_" + uuid.uuid4().hex[:10]
    raw_tmp = os.path.join(REF_DIR, rid + ".src")
    with open(raw_tmp, "wb") as f:
        f.write(base64.b64decode(b64.split(",")[-1]))
    wav = os.path.join(REF_DIR, rid + ".wav")
    # normalize to mono 24 kHz wav
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_tmp,
          "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav])
    try:
        os.remove(raw_tmp)
    except Exception:
        pass
    if denoise:
        cleaned = os.path.join(REF_DIR, rid + ".dn.wav")
        try:
            _denoise_file(wav, cleaned, 24000)
            os.replace(cleaned, wav)
        except Exception:
            _safe_rm(cleaned)
    secs = _audio_seconds(wav)
    if not ref_text:
        res = _enqueue("transcribe", {"path": wav}, timeout=600)
        ref_text = res.get("text", "") if "error" not in res else ""
    entry = {
        "id": rid, "name": name or rid, "filename": rid + ".wav",
        "seconds": secs, "ref_text": ref_text or "",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with _store_lock:
        refs = _read_json(REFS_FILE, [])
        refs.insert(0, entry)
        _write_json(REFS_FILE, refs)
    return entry


def _ref_delete(ref_id: str) -> bool:
    with _store_lock:
        refs = _read_json(REFS_FILE, [])
        keep, removed = [], None
        for r in refs:
            if r.get("id") == ref_id:
                removed = r
            else:
                keep.append(r)
        if removed:
            _write_json(REFS_FILE, keep)
            f = os.path.join(REF_DIR, removed.get("filename", ""))
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
            return True
        return False


# --------------------------------------------------------------------------
# Script chunking
# --------------------------------------------------------------------------
def _split_script(text: str, max_chars: int = 350):
    """Split into (segment_text, gap_after_ms_kind) honoring paragraphs.

    Returns list of segments and list of gap kinds between them:
    'para' (blank line) or 'sent' (sentence within paragraph).
    """
    segments = []
    gaps = []  # kind between seg i and i+1
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for pi, para in enumerate(paragraphs):
        sents = re.split(r"(?<=[.!?])\s+", para)
        chunk = ""
        para_chunks = []
        for s in sents:
            if not s:
                continue
            if chunk and len(chunk) + len(s) + 1 > max_chars:
                para_chunks.append(chunk.strip())
                chunk = s
            else:
                chunk = (chunk + " " + s).strip()
        if chunk:
            para_chunks.append(chunk.strip())
        for ci, c in enumerate(para_chunks):
            if segments:
                gaps.append("para" if ci == 0 else "sent")
            segments.append(c)
    return segments, gaps


# --------------------------------------------------------------------------
# Media cleanup: denoise + remove filler words ("um"/"uh") + dead air, for a
# local audio OR video file. Runs as an async job (long media can take a while).
# --------------------------------------------------------------------------
_clean_jobs: dict = {}
_clean_lock = threading.Lock()


def _clean_set(jid: str, **kw) -> None:
    with _clean_lock:
        _clean_jobs.setdefault(jid, {}).update(kw)


def _clean_get(jid: str) -> dict:
    with _clean_lock:
        return dict(_clean_jobs.get(jid, {}))


def _probe_media(path: str) -> dict:
    info = {"duration": 0.0, "has_video": False, "has_audio": False,
            "width": 0, "height": 0}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height", "-of", "json", path],
            capture_output=True, text=True).stdout
        d = json.loads(out or "{}")
        info["duration"] = float(d.get("format", {}).get("duration") or 0.0)
        for s in d.get("streams", []):
            if s.get("codec_type") == "video":
                info["has_video"] = True
                info["width"] = int(s.get("width") or 0)
                info["height"] = int(s.get("height") or 0)
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
    except Exception:
        pass
    if info["duration"] <= 0:
        info["duration"] = _audio_seconds(path)
    return info


def _collapse_token(w: str) -> str:
    """Normalize a transcribed token for filler matching: lowercase, strip
    punctuation, and collapse repeated letters (ummm->um, uhh->uh, hmm->hm)."""
    w = re.sub(r"[^a-z\-]", "", (w or "").strip().lower())
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def _filler_cuts(words: list, fillers: list, pad: float = 0.04):
    fset = {_collapse_token(f) for f in fillers}
    fset.discard("")
    cuts, hits = [], []
    for w in words:
        if _collapse_token(w.get("word", "")) in fset:
            a = max(0.0, float(w["start"]) - pad)
            b = float(w["end"]) + pad
            cuts.append((a, b))
            hits.append({"word": (w.get("word") or "").strip(),
                         "start": round(float(w["start"]), 2),
                         "end": round(float(w["end"]), 2)})
    return cuts, hits


def _silence_cuts(path: str, min_silence: float, keep: float,
                  noise_db: float):
    """Detect silences longer than min_silence and return intervals to drop,
    leaving `keep` seconds of padding on each side of every gap."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    cuts, cur = [], None
    for line in out.splitlines():
        if "silence_start:" in line:
            try:
                cur = float(line.split("silence_start:")[1].strip())
            except Exception:
                cur = None
        elif "silence_end:" in line and cur is not None:
            try:
                end = float(line.split("silence_end:")[1].split("|")[0].strip())
            except Exception:
                cur = None
                continue
            a, b = cur + keep, end - keep
            if b > a:
                cuts.append((a, b))
            cur = None
    return cuts


def _merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1] + 0.01:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _keep_intervals(cuts, duration):
    cuts = _merge_intervals(cuts)
    keeps, pos = [], 0.0
    for a, b in cuts:
        a = max(0.0, min(a, duration))
        b = max(0.0, min(b, duration))
        if a > pos + 0.02:
            keeps.append((pos, a))
        pos = max(pos, b)
    if pos < duration - 0.02:
        keeps.append((pos, duration))
    return keeps


def _sel_expr(keeps):
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keeps)


def _render_clean_audio(src48, keeps, out_path, fmt, loudness):
    af = []
    if keeps is not None:
        af.append(f"aselect='{_sel_expr(keeps)}'")
        af.append("asetpts=N/SR/TB")
    lufs = LUFS_TARGETS.get(loudness)
    if lufs is not None:
        af.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src48]
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-ar", "48000", "-ac", "1"]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif fmt == "flac":
        cmd += ["-c:a", "flac"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    cmd += [out_path]
    _run(cmd)
    return cmd


def _render_clean_video(src_video, src48, keeps, out_path, loudness):
    expr = _sel_expr(keeps)
    fc = (f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[v];"
          f"[1:a]aselect='{expr}',asetpts=N/SR/TB")
    lufs = LUFS_TARGETS.get(loudness)
    if lufs is not None:
        fc += f",loudnorm=I={lufs}:TP=-1.5:LRA=11"
    fc += "[a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src_video, "-i", src48,
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    _run(cmd)
    return cmd


def _remux_video(src_video, src48, out_path, loudness):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src_video, "-i", src48,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    lufs = LUFS_TARGETS.get(loudness)
    if lufs is not None:
        cmd += ["-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    _run(cmd)
    return cmd


def _resolve_media_path(raw: str) -> str:
    p = (raw or "").strip().strip('"').strip("'")
    if p.startswith("file://"):
        p = p[len("file://"):]
    return os.path.abspath(os.path.expanduser(p))


def _ci_enhance_available() -> bool:
    return sys.platform == "darwin" and os.path.isfile(CI_ENHANCE_BIN) \
        and os.access(CI_ENHANCE_BIN, os.X_OK)


def _captions_available() -> bool:
    """Smart captions need the Pillow renderer present and the local oMLX LLM
    reachable."""
    if not os.path.isfile(CAPTION_BIN):
        return False
    try:
        import urllib.request
        base = OMLX_URL.rsplit("/v1/", 1)[0]
        with urllib.request.urlopen(base + "/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _memory_available() -> bool:
    """The omlx-memory sqlite-vec service (RAG/storage) is reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen(MEMORY_URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _mem_post(route: str, payload: dict, timeout: int = 120) -> dict:
    import urllib.request
    body = json.dumps(payload).encode()
    req = urllib.request.Request(MEMORY_URL + route, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _mem_get(route: str, timeout: int = 30) -> dict:
    import urllib.request
    with urllib.request.urlopen(MEMORY_URL + route, timeout=timeout) as r:
        return json.loads(r.read())


def _index_transcript(source: str, text: str, meta: dict) -> dict:
    """Store a cleanup transcript in the shared sqlite-vec RAG so the whole
    video library becomes semantically searchable. Re-indexing the same source
    replaces the previous copy. Best-effort: callers ignore failures."""
    text = (text or "").strip()
    if not text:
        return {"skipped": "empty transcript"}
    # Drop any earlier doc for this same source in our collection (dedupe).
    try:
        docs = _mem_get("/rag/docs").get("docs", [])
        for d in docs:
            if (d.get("collection") == TRANSCRIPT_COLLECTION
                    and d.get("source") == source):
                _mem_post("/rag/delete", {"doc_id": d.get("id")}, timeout=30)
    except Exception:
        pass
    header = (f"Video transcript: {source}\n"
             f"Cleaned {datetime.now().isoformat(timespec='seconds')}")
    extras = []
    if meta.get("saved_to"):
        extras.append(f"Saved to: {meta['saved_to']}")
    if meta.get("aspect"):
        extras.append(f"Aspect: {meta['aspect']}")
    if extras:
        header += "\n" + " | ".join(extras)
    payload = {"collection": TRANSCRIPT_COLLECTION, "source": source,
               "text": header + "\n\n" + text}
    return _mem_post("/rag/ingest", payload, timeout=180)


def _passthrough_media(src: str, out_path: str) -> None:
    """Copy a media file into an mp4 container (video copied, audio to aac)."""
    info = _probe_media(src)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", out_path]
    try:
        _run(cmd)
    except Exception:
        cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0"]
        if info["has_audio"]:
            cmd2 += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd2 += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt",
                 "yuv420p", "-movflags", "+faststart", out_path]
        _run(cmd2)


def _apply_enhance(video_path: str, base: str, opts: dict) -> str:
    """Run the macOS Core Image auto light/colour enhancement on a video, then
    mux the original audio back. Returns the new file path."""
    enh_vid = base + ".enh.mp4"
    cmd = [CI_ENHANCE_BIN, video_path, enh_vid,
           "--level", str(opts.get("enhance_level", 1.0))]
    if opts.get("enhance_face"):
        cmd.append("--face")
    subprocess.run(cmd, check=True, capture_output=True)
    info = _probe_media(video_path)
    out = base + ".enhmux.mp4"
    mux = ["ffmpeg", "-y", "-loglevel", "error", "-i", enh_vid]
    if info["has_audio"]:
        mux += ["-i", video_path, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "copy"]
    else:
        mux += ["-map", "0:v:0", "-c:v", "copy"]
    mux += ["-movflags", "+faststart", out]
    _run(mux)
    _safe_rm(enh_vid)
    _safe_rm(video_path)
    return out


def _aspect_dims(sw: int, sh: int, ratio, fit: str):
    """Return (W,H) output canvas for a target aspect ratio (tw,th), chosen from
    the source dimensions so we never upscale. fit='fill' crops; 'pad'* fits."""
    tw, th = ratio
    a = tw / th          # target aspect (w/h)
    s = (sw / sh) if sh else a
    if fit == "fill":
        if a < s:        # target narrower -> keep full height, crop width
            H, W = sh, round(sh * a)
        else:            # target wider -> keep full width, crop height
            W, H = sw, round(sw / a)
    else:                # pad: contain the whole frame, add bars
        if a < s:        # target narrower -> keep full width, bars top/bottom
            W, H = sw, round(sw / a)
        else:            # target wider -> keep full height, bars left/right
            H, W = sh, round(sh * a)
    W -= W % 2
    H -= H % 2
    return max(2, W), max(2, H)


def _apply_aspect(video_path: str, base: str, opts: dict) -> str:
    """Reframe a video to a target aspect ratio. Returns the new file path."""
    ratio = ASPECTS.get(opts.get("aspect"))
    if not ratio:
        return video_path
    fit = opts.get("aspect_fit", "fill")
    info = _probe_media(video_path)
    sw, sh = info["width"], info["height"]
    if not (sw and sh):
        return video_path
    W, H = _aspect_dims(sw, sh, ratio, fit)
    out = base + ".aspect.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path]
    if fit == "fill":
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H}")
        cmd += ["-vf", vf, "-map", "0:v:0"]
    elif fit == "pad_blur":
        fc = (f"[0:v]split=2[bg][fg];"
              f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},gblur=sigma=24[bgb];"
              f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
              f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2[v]")
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    else:                # pad_black
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")
        cmd += ["-vf", vf, "-map", "0:v:0"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out]
    _run(cmd)
    _safe_rm(video_path)
    return out


def _remap_time(t: float, keeps: list) -> float | None:
    """Map an original-timeline timestamp to the output timeline defined by the
    kept intervals. Returns None if t falls inside a removed (cut) region."""
    acc = 0.0
    for a, b in keeps:
        if t < a:
            return None
        if t <= b:
            return acc + (t - a)
        acc += (b - a)
    return acc


def _segment_lines(words: list, max_line: float = 2.4, max_words: int = 6):
    """Group word-timestamps into short caption lines for timing + LLM rewrite."""
    lines, cur = [], []
    for w in words:
        if not cur:
            cur = [w]
            continue
        span = w["end"] - cur[0]["start"]
        if span > max_line or len(cur) >= max_words or \
                (w["start"] - cur[-1]["end"]) > 0.7:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        out.append({
            "start": round(ln[0]["start"], 2),
            "end": round(ln[-1]["end"], 2),
            "t": " ".join((w.get("word") or "").strip() for w in ln).strip(),
        })
    return out


def _repair_json(txt: str):
    """Best-effort parse of a JSON object from an LLM reply that may be wrapped
    in markdown fences, prefixed with reasoning, or truncated mid-output."""
    if not txt:
        raise RuntimeError("empty reply")
    # Strip ```json ... ``` fences.
    txt = re.sub(r"```(?:json)?", "", txt)
    start = txt.find("{")
    if start < 0:
        raise RuntimeError("no JSON object in reply")
    frag = txt[start:]
    # Fast path.
    try:
        return json.loads(frag)
    except Exception:
        pass
    # Trim to the last complete object/array, then close open brackets.
    last = max(frag.rfind("}"), frag.rfind("]"))
    if last > 0:
        candidate = frag[:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Truncation repair: cut at the last comma, then balance brackets/quotes.
    cut = frag.rfind(",")
    body = frag[:cut] if cut > 0 else frag
    # Balance quotes.
    if body.count('"') % 2:
        body += '"'
    # Close any open braces/brackets in stack order.
    stack = []
    instr = False
    esc = False
    for ch in body:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            instr = not instr
            continue
        if instr:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    for ch in reversed(stack):
        body += "}" if ch == "{" else "]"
    return json.loads(body)


def _llm_json(messages: list, max_tokens: int = 6000, temp: float = 0.7):
    """Call the local oMLX LLM and parse a JSON object from the reply. Tolerant
    of markdown fences, leading reasoning, and truncated output."""
    import urllib.request
    body = json.dumps({"model": _get_llm(), "messages": messages,
                       "temperature": temp, "max_tokens": max_tokens,
                       "stream": False}).encode()
    req = urllib.request.Request(OMLX_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    txt = data["choices"][0]["message"]["content"]
    return _repair_json(txt)


def _kept_line_indices(lines: list, keeps: list | None) -> set:
    """Indices of caption lines that actually survive the kept windows — the same
    remap test the renderer uses in _build_overlays. keeps=None → every line.
    Lets us generate & display captions ONLY for content that ends up in the
    video (saves LLM compute and avoids showing captions for cut footage)."""
    if not keeps:
        return set(range(len(lines)))
    kept = set()
    for i, ln in enumerate(lines):
        a = _remap_time(ln["start"], keeps)
        b = _remap_time(ln["end"], keeps)
        if a is None:
            a = _remap_time(ln["start"] + 0.05, keeps)
        if b is None:
            b = _remap_time(ln["end"] - 0.05, keeps)
        if a is not None and b is not None and b - a >= 0.25:
            kept.add(i)
    return kept


def _gen_captions(lines: list, want_captions: bool = True, want_cards: bool = True,
                  topic: str = "", length_target: str = "auto",
                  keep_idx: set | None = None) -> dict:
    """Use the LLM to SELECT the strongest moments and write a few high-quality
    on-screen captions for them (not a line-by-line rewrite), plus optional
    concept/hook cards. Returns {captions:[...], cards:[...]} keyed by line index
    'i'. When keep_idx is given, only those lines are candidates — so we never
    spend compute on footage the edit cuts out."""
    sys_msg = (
        "You are an elite short-form (TikTok/Reels/YouTube Shorts) video editor "
        "and copywriter. You write a SMALL number of bold, high-impact on-screen "
        "captions that hook viewers and drive retention. Output ONLY one valid "
        "JSON object, no prose, no markdown.")
    lt = VIRAL_LENGTH_TARGETS.get(length_target, "")
    idxs = (sorted(keep_idx) if keep_idx is not None else list(range(len(lines))))
    payload = [{"i": i, "t": lines[i]["t"], "s": round(lines[i]["start"], 1)}
               for i in idxs if 0 <= i < len(lines)]

    # Few, high-quality captions: ~1 per 7s of kept content, capped low so they
    # stay sparse and meaningful.
    span = (payload[-1]["s"] - payload[0]["s"]) if len(payload) >= 2 else 7.0
    target_count = max(2, min(10, round(span / 7.0))) if payload else 0

    def _select(chunk, n):
        ask = (
            f"From these transcript lines, choose only the {n} MOST powerful "
            "moments to put a caption on screen — a sharp hook, a bold claim, a "
            "punchline, or a surprising/emotional beat. SKIP setup, filler, "
            "throat-clearing and repetition. For each chosen line write ONE "
            "punchy caption of 3 to 6 words that lands the IDEA in the speaker's "
            "voice (NOT a transcription), and pick 1 to 2 words to EMPHASISE. "
            "Spread the picks across the timeline; never caption adjacent "
            "lines.\n")
        if lt:
            ask += "Edit vibe: " + lt + "\n"
        if topic:
            ask += f"Topic/voice: {topic}\n"
        ask += ('Return JSON exactly: {"captions":[{"i":<int>,"text":"...",'
                '"emphasis":["..."]}]} using each line\'s given i.\nLines: '
                + json.dumps(chunk))
        return _llm_json([{"role": "system", "content": sys_msg},
                          {"role": "user", "content": ask}], temp=0.6)

    caps = {}
    if want_captions and payload:
        # One selective call when the candidate set is reasonable; otherwise
        # chunk it and give each chunk a proportional quota.
        if len(payload) <= 120:
            chunks = [(payload, target_count)]
        else:
            csz = 80
            parts = [payload[s:s + csz] for s in range(0, len(payload), csz)]
            chunks = [(p, max(1, round(target_count * len(p) / len(payload))))
                      for p in parts]
        for chunk, n in chunks:
            if n <= 0:
                continue
            try:
                out = _select(chunk, n)
            except Exception:
                try:
                    out = _select(chunk, n)
                except Exception:
                    continue
            for c in out.get("captions", []):
                if "i" in c:
                    try:
                        caps[int(c["i"])] = c
                    except (TypeError, ValueError):
                        pass

    cards = []
    if want_cards:
        # Cards are few; one short call over a condensed view of the transcript.
        try:
            card_ask = (
                "From this transcript, choose 3 to 6 standout moments for big "
                "concept 'hook cards' (a 2-5 word title + a short punchy "
                "subtitle that teases the idea). Use the line 'i' index where "
                "each moment occurs.\n")
            if topic:
                card_ask += f"Video topic/voice: {topic}\n"
            card_ask += ('Return JSON exactly: {"cards":[{"i":<int>,'
                         '"title":"...","subtitle":"..."}]}\nLines: '
                         + json.dumps(payload))
            cout = _llm_json([{"role": "system", "content": sys_msg},
                              {"role": "user", "content": card_ask}])
            cards = cout.get("cards", []) or []
        except Exception:
            cards = []
    return {"captions": caps, "cards": cards}


def _build_overlays(lines: list, gen: dict, keeps: list,
                    include_captions: bool = True,
                    include_cards: bool = True) -> list:
    """Combine LLM text with ASR timing, remap to the output timeline, and emit
    overlay items: {id,type,text/title/subtitle,emphasis,a,b}. Line captions and
    hook cards can be included independently."""
    items = []
    caps = gen.get("captions", {})

    def _txt(v):
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, (list, tuple)):
            return " ".join(_txt(x) for x in v).strip()
        return ("" if v is None else str(v)).strip()

    def _emph(v):
        if isinstance(v, str):
            v = [v]
        return [_txt(e) for e in (v or []) if _txt(e)]

    if include_captions:
        cap_items = []
        for i, ln in enumerate(lines):
            c = caps.get(i)
            if not c:
                continue  # selective: only the LLM-chosen moments get a caption
            text = _txt(c.get("text"))
            if not text:
                continue
            a = _remap_time(ln["start"], keeps)
            b = _remap_time(ln["end"], keeps)
            if a is None:
                a = _remap_time(ln["start"] + 0.05, keeps)
            if b is None:
                b = _remap_time(ln["end"] - 0.05, keeps)
            if a is None or b is None or b - a < 0.05:
                continue
            cap_items.append({"id": f"cap{i}", "type": "caption", "text": text,
                              "emphasis": _emph(c.get("emphasis")),
                              "a": round(a, 2), "b": round(b, 2)})
        # Hold each caption on screen longer (min ~3.2s), but never overlap the
        # next one and never run past the end of the output.
        cap_items.sort(key=lambda it: it["a"])
        out_dur = sum(max(0.0, float(b) - float(a)) for a, b in keeps) if keeps else None
        MIN_HOLD = 3.2
        for k, it in enumerate(cap_items):
            end = max(it["b"], it["a"] + MIN_HOLD)
            if k + 1 < len(cap_items):
                end = min(end, cap_items[k + 1]["a"] - 0.2)
            if out_dur:
                end = min(end, out_dur - 0.05)
            if end - it["a"] < 0.8:
                end = it["a"] + 0.8
            it["b"] = round(end, 2)
        items.extend(cap_items)
    if include_cards:
        card_items = []
        for j, card in enumerate(gen.get("cards", [])):
            try:
                i = int(card.get("i", 0))
            except (TypeError, ValueError):
                continue
            if i >= len(lines):
                continue
            ln = lines[i]
            a = _remap_time(ln["start"], keeps)
            if a is None:
                a = _remap_time(ln["start"] + 0.05, keeps)
            if a is None:
                continue
            card_items.append({"id": f"card{j}", "type": "card",
                               "title": _txt(card.get("title", "")),
                               "subtitle": _txt(card.get("subtitle", "")),
                               "a": round(a, 2), "b": round(a + 2.2, 2)})
        # Hold each card on screen longer (min ~4.5s) so they don't flash, but
        # never overlap the next card or run past the end of the output.
        card_items.sort(key=lambda it: it["a"])
        out_dur = sum(max(0.0, float(b) - float(a)) for a, b in keeps) if keeps else None
        CARD_HOLD = 4.5
        for k, it in enumerate(card_items):
            end = it["a"] + CARD_HOLD
            if k + 1 < len(card_items):
                end = min(end, card_items[k + 1]["a"] - 0.2)
            if out_dur:
                end = min(end, out_dur - 0.05)
            if end - it["a"] < 1.5:
                end = it["a"] + 1.5
            it["b"] = round(end, 2)
        items.extend(card_items)
    return items


def _render_captions_pngs(items: list, W: int, H: int, outdir: str) -> dict:
    spec = {"width": W, "height": H, "outdir": outdir,
            "items": [{k: it[k] for k in it if k not in ("a", "b")}
                      for it in items]}
    res = subprocess.run([sys.executable, CAPTION_BIN], input=json.dumps(spec),
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("caption render failed: " + res.stderr[-400:])
    return json.loads(res.stdout or "{}")


def _composite_captions(video_path: str, items: list, base: str) -> str:
    """Overlay caption/card PNGs onto the video at their output-timeline times
    with a quick pop-in. Returns the new file path.

    Overlays are applied in batches so the filter graph never gets so large that
    ffmpeg stalls (long videos can have dozens of caption lines). Each batch
    writes a near-lossless intermediate that feeds the next batch; only the final
    pass encodes at delivery quality."""
    if not items:
        return video_path
    info = _probe_media(video_path)
    W, H = info["width"], info["height"]
    if not (W and H):
        return video_path
    pngs = _render_captions_pngs(items, W, H, base + "_caps")
    valid = [it for it in items
             if pngs.get(it["id"]) and os.path.exists(pngs[it["id"]])]
    if not valid:
        return video_path

    BATCH = 12
    batches = [valid[i:i + BATCH] for i in range(0, len(valid), BATCH)]
    src = video_path
    tmps = []
    last = len(batches) - 1
    out = base + ".caps.mp4"
    for bi, group in enumerate(batches):
        final_pass = (bi == last)
        inputs, fc, prev = ["-i", src], "", "0:v"
        total = len(group)
        for n, it in enumerate(group, start=1):
            inputs += ["-i", pngs[it["id"]]]
            a, b = it["a"], it["b"]
            pop = 0.12
            scl = (f"[{n}:v]scale=iw*"
                   f"'min(1,0.82+0.18*(t-{a})/{pop})':-1:eval=frame[s{n}]")
            outl = "[vout]" if n == total else f"[v{n}]"
            fc += (scl + f";[{prev}][s{n}]overlay=(W-w)/2:(H-h)/2:"
                   f"enable='between(t,{a},{b})'{outl};")
            prev = f"v{n}"
        fc = fc.rstrip(";")
        dst = out if final_pass else (base + f".capb{bi}.mkv")
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + \
            ["-filter_complex", fc, "-map", "[vout]"]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0", "-c:a", ("copy" if not final_pass else "aac")]
        if final_pass:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
        else:
            # Near-lossless, fast intermediate to avoid generational quality loss.
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                    "-pix_fmt", "yuv420p", dst]
        _run(cmd)
        if src != video_path:
            tmps.append(src)
        src = dst
    for t in tmps:
        _safe_rm(t)
    _safe_rm(video_path)
    return out


def _render_punchin_video(src_video, src48, keeps, out_path, loudness, zoom=1.10,
                          ramp_sec=0.22):
    """Join the kept segments and apply a single, continuous, gentle zoom motion
    over the whole clip (a slow "breathing" Ken Burns push) instead of toggling
    the zoom level on every cut. This keeps the energy of a punch-in while
    eliminating the hard in/out jumps that read as jittery, and it scales to
    long videos because the zoom is one filter, not one-per-segment."""
    info = _probe_media(src_video)
    W, H = info["width"], info["height"]
    if not (W and H):
        # Fall back to the plain select-based render.
        return _render_clean_video(src_video, src48, keeps, out_path, loudness)

    # Source frame rate (for a smooth, non-resampled zoom pass).
    fps = 30.0
    try:
        rate = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of",
             "default=nw=1:nk=1", src_video],
            capture_output=True, text=True).stdout.strip()
        if "/" in rate:
            num, den = rate.split("/")
            if float(den) > 0:
                fps = float(num) / float(den)
        elif rate:
            fps = float(rate)
    except Exception:
        pass
    if not (1.0 <= fps <= 240.0):
        fps = 30.0

    # 1) Concatenate the kept segments (scaled to WxH, square pixels). No
    #    per-segment zoom — the cuts are joined cleanly.
    vparts, aparts, labels = [], [], []
    for idx, (a, b) in enumerate(keeps):
        v = (f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS,"
             f"scale={W}:{H}:flags=lanczos,setsar=1[v{idx}]")
        a_ = f"[1:a]atrim={a}:{b},asetpts=PTS-STARTPTS[a{idx}]"
        vparts.append(v)
        aparts.append(a_)
        labels.append(f"[v{idx}][a{idx}]")
    fc = ";".join(vparts + aparts)
    fc += ";" + "".join(labels) + f"concat=n={len(keeps)}:v=1:a=1[vc][a]"

    # 2) One continuous, gentle zoom oscillation over the joined video. zoompan
    #    evaluates its own expressions per frame (no crop eval=frame needed, which
    #    this ffmpeg build lacks). Amplitude stays subtle for a pro feel.
    amp = max(0.0, min(0.12, float(zoom) - 1.0)) * 0.6  # e.g. 1.10 -> ~0.06
    if amp < 0.005:
        amp = 0.04
    period = 9.0
    z = f"1+{amp:.4f}+{amp:.4f}*sin(2*PI*in_time/{period})"
    fc += (f";[vc]zoompan=z='{z}':d=1:"
           f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
           f"s={W}x{H}:fps={fps:.4f}[v]")

    lufs = LUFS_TARGETS.get(loudness)
    if lufs is not None:
        fc += f";[a]loudnorm=I={lufs}:TP=-1.5:LRA=11[aout]"
        amap = "[aout]"
    else:
        amap = "[a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src_video, "-i", src48,
           "-filter_complex", fc, "-map", "[v]", "-map", amap,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", out_path]
    _run(cmd)
    return cmd


# ==========================================================================
# LLM EDIT DIRECTOR — collaborative, stage-cached viral editing
# --------------------------------------------------------------------------
# Turns a source video + transcript into a structured, EDITABLE JSON edit plan
# proposed by the oMLX LLM. The user tweaks fields (mostly dropdowns) and/or
# gives freeform feedback and iterates with the model (/director/revise). When
# happy, /director/render runs a STAGE-CACHED executor: each stage's output mp4
# is content-addressed, so changing only (say) captions reuses every earlier
# stage and re-runs just the overlay pass. Re-renders are therefore fast.
# ==========================================================================
DIR_CACHE = os.path.join(OUT_DIR, "_dir_cache")
DIR_STYLES = ["high-energy", "clean & modern", "cinematic", "vlog",
              "educational", "hype / meme", "calm / aesthetic"]
DIR_SILENCE = {"max_silence": 0.6, "keep_silence": 0.15, "silence_db": -30.0}
_dir_jobs: dict = {}
_dir_lock = threading.Lock()
_dir_plans: dict = {}   # session_id -> {plan, file_sig, history:[...]}


def _dir_set(jid: str, **kw) -> None:
    with _dir_lock:
        _dir_jobs.setdefault(jid, {}).update(kw)


def _dir_get(jid: str) -> dict:
    with _dir_lock:
        return dict(_dir_jobs.get(jid, {}))


def _sig(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _file_sig(path: str) -> str:
    st = os.stat(path)
    return _sig(os.path.abspath(path), st.st_size, int(st.st_mtime))


def _cache_path(stage: str, key: str, ext: str = "mp4") -> str:
    os.makedirs(DIR_CACHE, exist_ok=True)
    return os.path.join(DIR_CACHE, f"{stage}_{key}.{ext}")


def _stage_mutate(in_path: str, out_path: str, fn) -> str:
    """Copy a cached stage input to a scratch file, run fn(scratch) -> result
    (fn may delete scratch), then move result to out_path. This preserves the
    cached input so earlier stages stay reusable across re-renders."""
    work = out_path + ".src.mp4"
    shutil.copy2(in_path, work)
    res = fn(work)
    if os.path.abspath(res) != os.path.abspath(out_path):
        os.replace(res, out_path)
    return out_path


# ---- Plan schema -----------------------------------------------------------
def _plan_defaults() -> dict:
    return {
        "version": 1,
        "meta": {"platform": "youtube", "aspect": "9:16", "aspect_fit": "fill",
                 "length": "auto", "loudness": "youtube", "style": "high-energy"},
        "stages": {"denoise": True, "remove_fillers": True,
                   "remove_silences": True, "punch_in": True,
                   "enhance": True, "enhance_face": False,
                   "enhance_level": 1.0, "captions": True, "cards": True},
        "gen": {"captions": {}, "cards": []},
        "highlights": [],
        "highlights_for": "",
        "notes": "",
    }


def _plan_normalize(plan: dict) -> dict:
    d = _plan_defaults()
    plan = plan if isinstance(plan, dict) else {}
    meta = {**d["meta"], **(plan.get("meta") or {})}
    if meta.get("aspect") not in ASPECTS:
        meta["aspect"] = "9:16"
    if meta.get("aspect_fit") not in ("fill", "pad_blur", "pad_black"):
        meta["aspect_fit"] = "fill"
    if meta.get("length") not in VIRAL_LENGTH_TARGETS:
        meta["length"] = "auto"
    if meta.get("loudness") not in LUFS_TARGETS:
        meta["loudness"] = "youtube"
    meta["style"] = str(meta.get("style") or "high-energy")[:60]
    meta["platform"] = str(meta.get("platform") or "youtube")[:30]
    st = {**d["stages"], **(plan.get("stages") or {})}
    for k in ("denoise", "remove_fillers", "remove_silences", "punch_in",
              "enhance", "enhance_face", "captions", "cards"):
        st[k] = bool(st.get(k))
    try:
        st["enhance_level"] = max(0.0, min(1.0, float(st.get("enhance_level", 1.0))))
    except (TypeError, ValueError):
        st["enhance_level"] = 1.0
    gen = plan.get("gen") if isinstance(plan.get("gen"), dict) else {}
    raw_caps = gen.get("captions") if isinstance(gen.get("captions"), dict) else {}
    caps = {}
    for k, v in raw_caps.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            caps[ik] = v
    cards = gen.get("cards") if isinstance(gen.get("cards"), list) else []
    hl = plan.get("highlights")
    hlw = []
    if isinstance(hl, list):
        for pair in hl:
            try:
                a, b = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            if b > a:
                hlw.append([round(a, 2), round(b, 2)])
    return {"version": 1, "meta": meta, "stages": st,
            "gen": {"captions": caps, "cards": cards},
            "highlights": hlw,
            "highlights_for": str(plan.get("highlights_for") or ""),
            "notes": str(plan.get("notes") or "")}


def _plan_public(plan: dict, lines: list | None = None) -> dict:
    """Editor-friendly view: caption/card dicts flattened to sorted lists, with
    the original transcript line text alongside each caption for context. Only
    captions/cards on lines that survive the highlight cut are shown — the editor
    should reflect what actually ends up in the video, not the whole transcript."""
    plan = _plan_normalize(plan)
    caps = plan["gen"]["captions"]
    keep = None
    if lines and plan.get("highlights"):
        keep = _kept_line_indices(lines, plan["highlights"])
    cap_list = []
    for i in sorted(caps):
        if keep is not None and i not in keep:
            continue
        c = caps[i]
        emph = c.get("emphasis")
        if isinstance(emph, list):
            emph = " ".join(str(e) for e in emph)
        orig = ""
        if lines and 0 <= i < len(lines):
            orig = lines[i].get("t", "")
        cap_list.append({"i": i, "text": str(c.get("text") or ""),
                         "emphasis": str(emph or ""), "orig": orig})
    cards = []
    for c in plan["gen"]["cards"]:
        try:
            i = int(c.get("i", 0))
        except (TypeError, ValueError):
            i = 0
        if keep is not None and i not in keep:
            continue
        cards.append({"i": i, "title": str(c.get("title") or ""),
                      "subtitle": str(c.get("subtitle") or "")})
    out = {"version": 1, "meta": plan["meta"], "stages": plan["stages"],
           "notes": plan["notes"], "captions": cap_list, "cards": cards,
           "highlights": plan["highlights"],
           "highlights_for": plan["highlights_for"]}
    return out


def _plan_from_public(raw: dict) -> dict:
    """Inverse of _plan_public: fold the editable lists back into canonical gen."""
    raw = raw if isinstance(raw, dict) else {}
    caps = {}
    for c in (raw.get("captions") or []):
        try:
            i = int(c.get("i"))
        except (TypeError, ValueError):
            continue
        emph = c.get("emphasis")
        if isinstance(emph, str):
            emph = [w for w in emph.split() if w]
        caps[i] = {"text": str(c.get("text") or ""), "emphasis": emph or []}
    cards = []
    for c in (raw.get("cards") or []):
        try:
            i = int(c.get("i", 0))
        except (TypeError, ValueError):
            i = 0
        cards.append({"i": i, "title": str(c.get("title") or ""),
                      "subtitle": str(c.get("subtitle") or "")})
    return _plan_normalize({"meta": raw.get("meta"), "stages": raw.get("stages"),
                            "notes": raw.get("notes"),
                            "highlights": raw.get("highlights"),
                            "highlights_for": raw.get("highlights_for"),
                            "gen": {"captions": caps, "cards": cards}})


# ---- Prep (cached transcription + clean audio) -----------------------------
def _director_prep(jid: str, path: str, denoise_on: bool) -> dict:
    """Extract 48k audio (denoise optional), transcribe words. Both the audio
    and the ASR result are content-addressed and cached, so re-planning/
    re-rendering the same source never re-transcribes."""
    info = _probe_media(path)
    if not info["has_video"]:
        raise RuntimeError("Edit Director needs a video file (use Media Cleanup "
                           "for audio-only).")
    if not info["has_audio"]:
        raise RuntimeError("no audio stream found in this video")
    fsig = _file_sig(path)
    denoise = bool(denoise_on) and _denoise_available()
    prep_key = _sig(fsig, denoise)
    aud = _cache_path("aud", prep_key, "wav")
    if not os.path.exists(aud):
        _dir_set(jid, stage="extracting audio", progress=12)
        raw48 = aud + ".raw.wav"
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn",
              "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", raw48])
        if denoise:
            _dir_set(jid, stage="denoising", progress=20)
            _denoise_file(raw48, aud, 48000)
            _safe_rm(raw48)
        else:
            os.replace(raw48, aud)
    asr = _cache_path("asr", prep_key, "json")
    if os.path.exists(asr):
        with open(asr) as f:
            data = json.load(f)
    else:
        _dir_set(jid, stage="transcribing", progress=40)
        wav16 = aud + ".16k.wav"
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", aud,
              "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16])
        res = _enqueue("transcribe_words", {"path": wav16}, timeout=3600)
        _safe_rm(wav16)
        if "error" in res:
            raise RuntimeError(res["error"])
        data = {"words": res["data"]["words"], "text": res["data"]["text"]}
        with open(asr, "w") as f:
            json.dump(data, f)
    return {"info": info, "file_sig": fsig, "prep_key": prep_key,
            "audio": aud, "words": data["words"], "text": data["text"],
            "denoised": denoise}


def _store_prep(sess: str | None, path: str, denoise: bool, prep: dict) -> None:
    """Persist the FULL first-pass payload on the session so any later edit
    (revise, render, re-render) reuses it without re-opening the video:
      • probe/info (duration, w/h, fps, has_video/has_audio)
      • the cleaned 48k audio path + cache keys
      • the word-level transcription WITH timings (prep['words'])
      • the segmented caption lines (with per-line start/end)
      • the plain transcript text."""
    if not sess:
        return
    lines = _segment_lines(prep["words"])
    with _dir_lock:
        entry = _dir_plans.get(sess) or {}
        entry["prep"] = {"denoise": bool(denoise), "path": path, "data": prep}
        entry.update({
            "info": prep["info"],
            "file_sig": prep["file_sig"],
            "prep_key": prep["prep_key"],
            "audio": prep["audio"],
            "duration": prep["info"]["duration"],
            "words": prep["words"],
            "lines": lines,
            "text": prep["text"],
            "denoised": prep["denoised"],
        })
        _dir_plans[sess] = entry


def _session_prep(jid: str, sess: str | None, path: str, denoise: bool) -> dict:
    """Return the prep payload for (path, denoise), reusing the session-cached
    one when present. Only re-runs _director_prep (itself disk-cached) when the
    source or the denoise choice actually changes."""
    if sess:
        with _dir_lock:
            c = (_dir_plans.get(sess) or {}).get("prep")
        if c and c.get("path") == path and c.get("denoise") == bool(denoise):
            return c["data"]
    prep = _director_prep(jid, path, denoise)
    _store_prep(sess, path, denoise, prep)
    return prep


def _director_foundation(jid: str, path: str, prep: dict, st: dict) -> dict:
    """Build STAGE 1: the cached 'foundation' the whole edit is built on —
    denoised audio (already in prep) + a light/colour-enhanced full-length
    picture. Cached by source + enhance params ONLY (independent of any editing
    decision), so re-planning and re-rendering never re-enhances. Returns the
    video to draw the picture from (enhanced file, or the original source)."""
    enhance = bool(st.get("enhance")) and prep["info"]["has_video"] \
        and _ci_enhance_available()
    if not enhance:
        return {"video": path, "enhanced": False, "key": prep["file_sig"]}
    level = round(float(st.get("enhance_level", 1.0)), 3)
    face = bool(st.get("enhance_face"))
    found_key = _sig(prep["file_sig"], "enh", face, level)
    found_path = _cache_path("found", found_key)
    if not os.path.exists(found_path):
        _dir_set(jid, stage="enhancing light & colour (foundation)", progress=46)
        opts = {"enhance_level": level, "enhance_face": face}
        _stage_mutate(path, found_path,
                      lambda w: _apply_enhance(w, found_path + ".t", opts))
    return {"video": found_path, "enhanced": True, "key": found_key}


def _length_target_seconds(length: str):
    """Midpoint seconds for a viral length bucket, or None for auto/original."""
    return {"15_30": 24, "30_45": 38, "45_60": 52,
            "60_90": 75, "90_120": 105}.get(length)


def _cuts_from_keeps(keeps: list, dur: float) -> list:
    """Return the complement (the gaps to CUT) of a set of keep windows."""
    keeps = sorted((max(0.0, float(a)), min(dur, float(b)))
                   for a, b in keeps if float(b) > float(a))
    cuts, prev = [], 0.0
    for a, b in keeps:
        if a > prev:
            cuts.append((prev, a))
        prev = max(prev, b)
    if prev < dur:
        cuts.append((prev, dur))
    return cuts


def _lines_to_intervals(lines: list, idx: list, dur: float, pad: float = 0.15):
    """Merge the kept line indices into continuous time windows."""
    iv = []
    for i in sorted(set(idx)):
        if not (0 <= i < len(lines)):
            continue
        a = max(0.0, lines[i]["start"] - pad)
        b = min(dur, lines[i]["end"] + pad)
        if iv and a <= iv[-1][1] + 0.30:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    return [(a, b) for a, b in iv if b - a > 0.05]


def _director_highlights(lines: list, target_sec: int, topic: str = "") -> list:
    """Assemble a highlight cut from the BEST moments spread across the WHOLE
    video. The transcript is divided into sequential sections spanning the entire
    duration and the LLM picks the single strongest self-contained moment from
    EACH section — this forces coverage across the timeline instead of the model
    front-loading its picks from the opening. Returns sorted kept line indices
    whose total duration is close to target_sec."""
    n = len(lines)
    if n == 0:
        return []

    # How many moments to stitch together (≈1 every 8s of target), and the
    # section boundaries that span the entire video.
    n_clips = max(3, min(12, round(target_sec / 8.0)))
    n_clips = min(n_clips, n)
    buckets = []
    for k in range(n_clips):
        lo = (k * n) // n_clips
        hi = ((k + 1) * n) // n_clips
        hi = max(hi, lo + 1)
        buckets.append((lo, min(hi, n)))

    sections = [{"section": k,
                 "lines": [{"i": i, "t": lines[i]["t"]} for i in range(lo, hi)]}
                for k, (lo, hi) in enumerate(buckets)]

    sys_msg = (
        "You are an elite short-form video editor assembling a HIGHLIGHT REEL "
        "from a longer video. You stitch together the most exciting, "
        "self-contained moments from ACROSS THE WHOLE video — a strong hook, the "
        "punchiest insights, the best payoff — into one tight viral cut. Output "
        "ONLY one valid JSON object.")
    per = max(3, round(target_sec / max(1, n_clips)))
    ask = (
        f"This transcript is split into {n_clips} sequential SECTIONS that span "
        f"the entire video from start to end. From EACH section, choose the ONE "
        f"strongest, most engaging self-contained moment as a CONTIGUOUS line "
        f"range [a..b] (about {per} seconds, using the line 'i' indices). Return "
        f"exactly one segment per section so the final reel pulls the best bits "
        f"from throughout — NOT just the beginning. Rate each moment 1-10 "
        f"(10 = must-keep hook/payoff).\n")
    if topic:
        ask += f"Topic/voice: {topic}\n"
    ask += ('Return JSON exactly: {"segments":[{"section":<int>,"a":<int>,'
            '"b":<int>,"score":<1-10>}]}\nSections: ' + json.dumps(sections))
    out = _llm_json([{"role": "system", "content": sys_msg},
                     {"role": "user", "content": ask}], temp=0.4)

    # Map each section to its pick, clamped to that section's line range.
    def seg_dur(a, b):
        return max(0.0, lines[b]["end"] - lines[a]["start"])

    # Cap any single moment so the model can't hand back a giant range that
    # blows the length budget (keeps the reel tight and the picks usable).
    cap = max(4.0, per * 1.8)

    def _cap_b(a, b):
        while b > a and seg_dur(a, b) > cap:
            b -= 1
        return b

    by_section = {}
    for s in (out.get("segments") or []):
        try:
            a = int(s["a"]); b = int(s["b"]); sc = float(s.get("score", 5))
            sec = int(s.get("section", -1))
        except (TypeError, ValueError, KeyError):
            continue
        a, b = min(a, b), max(a, b)
        # Snap into the bucket this segment belongs to (by its own index if the
        # section label is missing/wrong).
        bk = None
        if 0 <= sec < len(buckets):
            bk = buckets[sec]
        else:
            for lo, hi in buckets:
                if lo <= a < hi:
                    bk = (lo, hi); break
        if not bk:
            continue
        lo, hi = bk
        a = max(lo, min(a, hi - 1))
        b = max(a, min(b, hi - 1))
        b = _cap_b(a, b)
        prev = by_section.get((lo, hi))
        if prev is None or sc > prev[0]:
            by_section[(lo, hi)] = (sc, a, b)

    # Ensure EVERY section contributes something: synthesize a midpoint pick for
    # any section the model skipped, so coverage spans the whole video.
    picks = []
    for lo, hi in buckets:
        if (lo, hi) in by_section:
            sc, a, b = by_section[(lo, hi)]
        else:
            mid = (lo + hi) // 2
            a = mid
            b = mid
            # grow to ~per seconds within the bucket
            while b + 1 < hi and seg_dur(a, b + 1) < per:
                b += 1
            sc = 4.0
        picks.append((sc, a, b))

    # picks already span the timeline (one per section). If their combined
    # duration overshoots the target, drop the weakest sections (keeps spread);
    # never drop below 3 clips.
    total = sum(seg_dur(a, b) for _, a, b in picks)
    if total > target_sec * 1.15 and len(picks) > 3:
        order = sorted(range(len(picks)), key=lambda j: picks[j][0])  # weakest first
        keep_flags = [True] * len(picks)
        for j in order:
            if total <= target_sec * 1.05 or sum(keep_flags) <= 3:
                break
            keep_flags[j] = False
            total -= seg_dur(picks[j][1], picks[j][2])
        picks = [p for p, f in zip(picks, keep_flags) if f]

    chosen = set()
    for _, a, b in picks:
        for i in range(a, b + 1):
            chosen.add(i)
    return sorted(chosen)


def _fallback_highlights(lines: list, target: float, dur: float) -> list:
    """Deterministic last-resort selection that STILL spreads across the whole
    video (not just the front): sample evenly-spaced short runs across the full
    duration until we reach the target length."""
    n = len(lines)
    if n == 0:
        return []
    n_clips = max(3, min(10, round(target / 8.0)))
    n_clips = min(n_clips, n)
    per = max(2.0, target / n_clips)
    chosen = set()
    for k in range(n_clips):
        center_t = dur * (k + 0.5) / n_clips
        # nearest line to this evenly-spaced point in the timeline
        ci = min(range(n), key=lambda i: abs((lines[i]["start"] + lines[i]["end"]) / 2 - center_t))
        a = b = ci
        while b + 1 < n and (lines[b]["end"] - lines[a]["start"]) < per:
            b += 1
        for i in range(a, b + 1):
            chosen.add(i)
    return sorted(chosen)


def _ensure_highlights(plan: dict, lines: list, dur: float, topic: str = "") -> dict:
    """Compute (or reuse) the highlight windows for the plan's length target.
    Stored on the plan so renders stay deterministic and cacheable; only
    recomputed when the length target changes."""
    length = plan["meta"]["length"]
    target = _length_target_seconds(length)
    if not target:
        plan["highlights"] = []
        plan["highlights_for"] = length
        return plan
    if plan.get("highlights") and plan.get("highlights_for") == length:
        return plan
    # If the source is already within ~15% of the target there is nothing to
    # trim — keep everything (no highlight windows).
    if dur <= target * 1.15:
        plan["highlights"] = []
        plan["highlights_for"] = length
        return plan
    idx = []
    try:
        idx = _director_highlights(lines, target, topic)
    except Exception as e:
        print(f"[highlights] LLM error: {e}", flush=True)
        idx = []
    iv = _lines_to_intervals(lines, idx, dur) if idx else []
    kept = sum(b - a for a, b in iv) if iv else 0.0
    llm_ok = bool(iv) and kept <= target * 1.4
    # Guarantee the length is obeyed: if the model gave nothing, or its pick is
    # still far longer than the target, fall back to a deterministic trim.
    if not llm_ok:
        idx = _fallback_highlights(lines, target, dur)
        iv = _lines_to_intervals(lines, idx, dur)
    span = (max(b for a, b in iv) - min(a for a, b in iv)) if iv else 0.0
    print(f"[highlights] target={target}s dur={dur:.0f}s lines={len(lines)} "
          f"llm_idx={len(idx)} llm_ok={llm_ok} windows={len(iv)} "
          f"kept={sum(b-a for a,b in iv):.1f}s span={span:.0f}s "
          f"first={iv[0] if iv else None}", flush=True)
    plan["highlights"] = [[round(a, 2), round(b, 2)] for a, b in (iv or [])]
    plan["highlights_for"] = length
    return plan


def _director_cuts(prep: dict, plan: dict):
    """Compute keep-intervals from the plan's length target + cut stages. Returns
    (keeps_or_None, out_keeps, n_fillers, n_silences)."""
    st = plan["stages"]
    words = prep["words"]
    dur = prep["info"]["duration"]
    cuts = []
    n_fillers = n_silences = 0
    # Length target -> keep only the selected highlight windows (cut the rest).
    hl = plan.get("highlights") or []
    if hl:
        keeps_hl = [(float(a), float(b)) for a, b in hl if float(b) > float(a)]
        if keeps_hl:
            cuts += _cuts_from_keeps(keeps_hl, dur)
    if st["remove_fillers"]:
        fcuts, hits = _filler_cuts(words, DEFAULT_FILLERS, 0.04)
        cuts += fcuts
        n_fillers = len(hits)
    if st["remove_silences"]:
        scuts = _silence_cuts(prep["audio"], DIR_SILENCE["max_silence"],
                              DIR_SILENCE["keep_silence"], DIR_SILENCE["silence_db"])
        cuts += scuts
        n_silences = len(scuts)
    has_cuts = bool(cuts)
    keeps = _keep_intervals(cuts, dur) if has_cuts else None
    if has_cuts and not keeps:
        raise RuntimeError("everything would be cut — loosen the settings")
    out_keeps = keeps if has_cuts else [(0.0, dur)]
    return keeps, out_keeps, n_fillers, n_silences


# ---- Staged, content-addressed executor ------------------------------------
def _director_execute(jid: str, path: str, plan: dict, sess: str | None = None) -> dict:
    plan = _plan_normalize(plan)
    st = plan["stages"]
    meta = plan["meta"]
    loudness = meta["loudness"]
    prep = _session_prep(jid, sess, path, st["denoise"])
    info = prep["info"]
    dur = info["duration"]
    aud = prep["audio"]
    # STAGE 1 — foundation: denoise (in prep) + enhance light/colour, cached and
    # reused across every plan/render iteration.
    found = _director_foundation(jid, path, prep, st)
    fvideo = found["video"]
    # Honour the length target even if it changed since the last revise.
    if _length_target_seconds(meta["length"]):
        _ensure_highlights(plan, _segment_lines(prep["words"]), dur)
    keeps, out_keeps, n_fillers, n_silences = _director_cuts(prep, plan)
    has_cuts = keeps is not None
    punch = st["punch_in"]

    # STAGE 3 — editing. Base video: cuts + optional continuous breathing zoom,
    # drawn from the enhanced foundation picture + denoised audio.
    base_key = _sig(found["key"], out_keeps, has_cuts, punch, loudness)
    base_path = _cache_path("base", base_key)
    if not os.path.exists(base_path):
        _dir_set(jid, stage="rendering base cut", progress=58)
        if punch:
            _render_punchin_video(fvideo, aud, out_keeps, base_path, loudness)
        elif has_cuts:
            _render_clean_video(fvideo, aud, keeps, base_path, loudness)
        else:
            _remux_video(fvideo, aud, base_path, loudness)
    cur = base_path

    # Aspect reframe.
    if ASPECTS.get(meta["aspect"]):
        asp_key = _sig(_file_sig(cur), meta["aspect"], meta["aspect_fit"])
        asp_path = _cache_path("asp", asp_key)
        if not os.path.exists(asp_path):
            _dir_set(jid, stage="reframing aspect ratio", progress=80)
            opts = {"aspect": meta["aspect"], "aspect_fit": meta["aspect_fit"]}
            _stage_mutate(cur, asp_path,
                          lambda w: _apply_aspect(w, asp_path + ".t", opts))
        cur = asp_path

    # Stage 4: caption / hook-card overlays.
    captioned = False
    cap_err = None
    if st["captions"] or st["cards"]:
        lines = _segment_lines(prep["words"])
        gen = plan["gen"]
        ovl_key = _sig(_file_sig(cur), st["captions"], st["cards"],
                       gen["captions"], gen["cards"], out_keeps,
                       OVERLAY_STYLE_VERSION)
        ovl_path = _cache_path("ovl", ovl_key)
        if os.path.exists(ovl_path):
            cur = ovl_path
            captioned = True
        else:
            items = _build_overlays(lines, gen, out_keeps,
                                    include_captions=st["captions"],
                                    include_cards=st["cards"])
            if items:
                _dir_set(jid, stage="compositing captions", progress=90)
                try:
                    _stage_mutate(cur, ovl_path,
                                  lambda w: _composite_captions(w, items, ovl_path + ".t"))
                    cur = ovl_path
                    captioned = True
                except Exception as ce:
                    cap_err = f"{type(ce).__name__}: {ce}"

    # Deliver: served copy + a copy next to the original.
    _dir_set(jid, stage="finalizing", progress=96)
    eid = "director_" + uuid.uuid4().hex[:12]
    final = os.path.join(OUT_DIR, eid + ".mp4")
    shutil.copy2(cur, final)
    # Clean stray per-stage scratch (caption PNG dirs, copy-in sources).
    try:
        for n in os.listdir(DIR_CACHE):
            if n.endswith(("_caps", ".src.mp4")) or ".t." in n or n.endswith(".t"):
                fp = os.path.join(DIR_CACHE, n)
                shutil.rmtree(fp, ignore_errors=True) if os.path.isdir(fp) else _safe_rm(fp)
    except Exception:
        pass
    new_dur = _audio_seconds(final)
    saved_to, save_error = None, None
    try:
        src_dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        cand = os.path.join(src_dir, f"{stem} viral.mp4")
        i = 2
        while os.path.exists(cand):
            cand = os.path.join(src_dir, f"{stem} viral ({i}).mp4")
            i += 1
        shutil.copy2(final, cand)
        saved_to = cand
    except Exception as e:
        save_error = f"{type(e).__name__}: {e}"
    return {
        "filename": os.path.basename(final),
        "url": f"/files/{os.path.basename(final)}",
        "kind": "video",
        "source": os.path.basename(path),
        "saved_to": saved_to,
        "save_error": save_error,
        "orig_seconds": round(dur, 2),
        "new_seconds": round(new_dur, 2),
        "saved_seconds": round(max(0.0, dur - new_dur), 2),
        "denoised": prep["denoised"],
        "enhanced": bool(st["enhance"] and _ci_enhance_available()),
        "punch_in": punch,
        "captioned": captioned,
        "caption_error": cap_err,
        "aspect": meta["aspect"] if ASPECTS.get(meta["aspect"]) else None,
        "fillers_removed": n_fillers,
        "silences_removed": n_silences,
        "length_target": meta["length"],
        "highlights": len(plan.get("highlights") or []),
        "plan": _plan_public(plan, _segment_lines(prep["words"])),
    }


# ---- LLM: propose & revise the edit plan -----------------------------------
def _director_decisions(text: str, dur: float, meta_hint: dict,
                        feedback: str = "", cur: dict | None = None) -> dict:
    """Ask the oMLX LLM for high-level edit DECISIONS (stages, style, aspect,
    length, rationale, hook-card ideas). Cheap single call; captions handled
    separately by _gen_captions."""
    sys_msg = (
        "You are an award-winning short-form video editor and creative director "
        "for viral YouTube/TikTok/Reels content. Decide how to edit a talking "
        "video for maximum retention and a professional, high-quality feel. "
        "Output ONLY one valid JSON object, no prose, no markdown.")
    schema = (
        '{"stages":{"denoise":bool,"remove_fillers":bool,"remove_silences":bool,'
        '"punch_in":bool,"enhance":bool,"enhance_face":bool,"captions":bool,'
        '"cards":bool},"meta":{"aspect":"9:16|16:9|1:1|4:5|original",'
        '"aspect_fit":"fill|pad_blur|pad_black","length":'
        '"auto|original|15_30|30_45|45_60|60_90|90_120","loudness":'
        '"youtube|podcast|broadcast|off","style":"short label","platform":'
        '"youtube|tiktok|reels"},"cards":[{"i":<line index int>,"title":"2-5 '
        'words","subtitle":"short tease"}],"notes":"2-4 sentence rationale of '
        'your creative choices"}')
    ask = (f"Source video is {dur:.0f}s of talking-to-camera. Target platform "
           f"hint: {meta_hint.get('platform','youtube')}. Decide the edit. "
           f"Favour punchy retention pacing, remove fillers/dead-air, add a "
           f"gentle continuous push-in, smart captions and a few hook cards. "
           f"Pick 3-6 hook cards at the strongest transcript moments.\n")
    if feedback:
        ask += ("The creator gave this feedback on your current plan — honour "
                f"it precisely: \"{feedback}\"\n")
    if cur:
        ask += ("Current plan to revise (keep what works, change per feedback): "
                + json.dumps({"stages": cur.get("stages"),
                              "meta": cur.get("meta")}) + "\n")
    ask += ("Transcript:\n" + text[:6000] + "\n\nReturn JSON exactly: " + schema)
    return _llm_json([{"role": "system", "content": sys_msg},
                      {"role": "user", "content": ask}], temp=0.6)


def _merge_decisions(plan: dict, dec: dict) -> dict:
    p = _plan_normalize(plan)
    dec = dec if isinstance(dec, dict) else {}
    if isinstance(dec.get("meta"), dict):
        p["meta"].update({k: v for k, v in dec["meta"].items() if v is not None})
    if isinstance(dec.get("stages"), dict):
        for k, v in dec["stages"].items():
            if k in p["stages"] and isinstance(v, bool):
                p["stages"][k] = v
    if isinstance(dec.get("cards"), list) and dec["cards"]:
        p["gen"]["cards"] = dec["cards"]
    if dec.get("notes"):
        p["notes"] = str(dec["notes"])
    return _plan_normalize(p)


def _run_director_prep(jid: str, opts: dict) -> None:
    """STAGE 1 — build & cache the foundation (denoise audio + enhance light/
    colour), return a picture-preview URL + transcript so the user can lock the
    look before directing the edit."""
    try:
        path = opts["path"]
        sess = opts.get("session", jid)
        _dir_set(jid, status="running", stage="probing", progress=5)
        denoise = bool(opts.get("denoise", True))
        prep = _session_prep(jid, sess, path, denoise)
        found_st = {
            "denoise": denoise,
            "enhance": bool(opts.get("enhance", True)),
            "enhance_face": bool(opts.get("enhance_face", False)),
            "enhance_level": float(opts.get("enhance_level", 1.0)),
        }
        found = _director_foundation(jid, path, prep, found_st)
        url = None
        if found["enhanced"]:
            _dir_set(jid, stage="preparing preview", progress=92)
            pv = "found_" + uuid.uuid4().hex[:12] + ".mp4"
            shutil.copy2(found["video"], os.path.join(OUT_DIR, pv))
            url = "/files/" + pv
        # Seed (or update) the session with the locked foundation settings.
        with _dir_lock:
            entry = _dir_plans.get(sess) or {}
            plan = entry.get("plan") or _plan_defaults()
            plan["meta"]["platform"] = opts.get("platform", plan["meta"]["platform"])
            plan["stages"].update(found_st)
            entry.update({"plan": plan, "path": path,
                          "file_sig": prep["file_sig"],
                          "history": entry.get("history", []),
                          "prepped": True,
                          "words": prep["words"], "text": prep["text"],
                          "duration": prep["info"]["duration"]})
            _dir_plans[sess] = entry
        result = {"kind": "prep", "session": sess, "url": url,
                  "duration": round(prep["info"]["duration"], 2),
                  "has_video": bool(prep["info"]["has_video"]),
                  "denoised": prep["denoised"], "enhanced": found["enhanced"],
                  "transcript": prep["text"][:8000]}
        _dir_set(jid, stage="done", progress=100, status="done", result=result)
    except Exception as e:
        _dir_set(jid, status="error", stage="error", error=str(e))


def _run_director_plan(jid: str, opts: dict) -> None:
    try:
        path = opts["path"]
        sess = opts.get("session", jid)
        _dir_set(jid, status="running", stage="probing", progress=5)
        # Reuse a prepped foundation (denoise/enhance settings) if present.
        with _dir_lock:
            prev_entry = _dir_plans.get(sess)
        found_st = None
        if prev_entry and prev_entry.get("path") == path:
            found_st = {k: prev_entry["plan"]["stages"][k] for k in
                        ("denoise", "enhance", "enhance_face", "enhance_level")}
        denoise = found_st["denoise"] if found_st else True
        prep = _session_prep(jid, sess, path, denoise)
        lines = _segment_lines(prep["words"])
        _dir_set(jid, stage="directing the edit", progress=55)
        plan = _plan_defaults()
        plan["meta"]["platform"] = opts.get("platform", "youtube")
        try:
            dec = _director_decisions(prep["text"], prep["info"]["duration"],
                                      plan["meta"])
            plan = _merge_decisions(plan, dec)
        except Exception as e:
            plan["notes"] = f"(auto defaults — director call failed: {e})"
        # Honour an explicit user length/aspect choice from the launch form.
        if opts.get("length"):
            plan["meta"]["length"] = opts["length"]
        if opts.get("aspect"):
            plan["meta"]["aspect"] = opts["aspect"]
        # Keep the locked foundation settings from Stage 1.
        if found_st:
            plan["stages"].update(found_st)
        plan = _plan_normalize(plan)
        if _length_target_seconds(plan["meta"]["length"]):
            _dir_set(jid, stage="selecting highlights for length", progress=64)
            _ensure_highlights(plan, lines, prep["info"]["duration"],
                               topic=opts.get("topic", ""))
        if plan["stages"]["captions"] or plan["stages"]["cards"]:
            _dir_set(jid, stage="writing smart captions", progress=72)
            keep_idx = _kept_line_indices(lines, plan.get("highlights"))
            try:
                gen = _gen_captions(lines,
                                    want_captions=plan["stages"]["captions"],
                                    want_cards=plan["stages"]["cards"],
                                    topic=opts.get("topic", ""),
                                    length_target=plan["meta"]["length"],
                                    keep_idx=keep_idx)
                plan["gen"]["captions"] = {int(k): v for k, v in
                                           gen.get("captions", {}).items()}
                if gen.get("cards"):
                    plan["gen"]["cards"] = gen["cards"]
            except Exception as e:
                _dir_set(jid, stage=f"captions skipped: {e}")
        plan = _plan_normalize(plan)
        with _dir_lock:
            entry = _dir_plans.get(sess) or {}
            entry.update({"plan": plan, "path": path,
                          "file_sig": prep["file_sig"],
                          "history": entry.get("history", []),
                          "words": prep["words"], "text": prep["text"],
                          "duration": prep["info"]["duration"]})
            _dir_plans[sess] = entry
        pub = _plan_public(plan, lines)
        pub["duration"] = round(prep["info"]["duration"], 2)
        pub["transcript"] = prep["text"][:8000]
        _dir_set(jid, stage="done", progress=100, status="done",
                 result={"kind": "plan", "plan": pub, "session": sess})
    except Exception as e:
        _dir_set(jid, status="error", stage="error",
                 error=f"{type(e).__name__}: {e}")


def _run_director_revise(jid: str, opts: dict) -> None:
    try:
        sess = opts["session"]
        with _dir_lock:
            entry = _dir_plans.get(sess)
        if not entry:
            raise RuntimeError("no active plan for this session — run Plan first")
        path = entry["path"]
        _dir_set(jid, status="running", stage="applying your edits", progress=20)
        # Start from the user's edited plan (their explicit dropdown/text edits win).
        base_plan = _plan_from_public(opts.get("plan") or {})
        feedback = (opts.get("feedback") or "").strip()
        regen_caps = bool(opts.get("regen_captions"))
        # Revise is a pure LLM + metadata step — reuse the transcript & word
        # timeline captured during prep/plan; never re-open or re-watch the video.
        words = entry.get("words")
        text = entry.get("text")
        dur = entry.get("duration")
        lines = entry.get("lines")
        if words is None or text is None or dur is None:
            prep = _session_prep(jid, sess, path, base_plan["stages"]["denoise"])
            words, text, dur = prep["words"], prep["text"], prep["info"]["duration"]
            lines = _segment_lines(words)
        if lines is None:
            lines = _segment_lines(words)
        if feedback:
            _dir_set(jid, stage="consulting the director", progress=45)
            try:
                dec = _director_decisions(text, dur,
                                          base_plan["meta"], feedback=feedback,
                                          cur=base_plan)
                base_plan = _merge_decisions(base_plan, dec)
            except Exception as e:
                base_plan["notes"] = (base_plan.get("notes", "") +
                                      f"\n(director revise failed: {e})")
        base_plan = _plan_normalize(base_plan)
        # Recompute the highlight cut FIRST, so any caption rewrite below only
        # spends LLM compute on the lines that survive the new cut.
        _dir_set(jid, stage="updating length cut", progress=68)
        _ensure_highlights(base_plan, lines, dur, topic=feedback)
        if feedback and regen_caps and (base_plan["stages"]["captions"]
                                        or base_plan["stages"]["cards"]):
            _dir_set(jid, stage="rewriting captions", progress=78)
            keep_idx = _kept_line_indices(lines, base_plan.get("highlights"))
            try:
                gen = _gen_captions(
                    lines, want_captions=base_plan["stages"]["captions"],
                    want_cards=base_plan["stages"]["cards"],
                    topic=feedback, length_target=base_plan["meta"]["length"],
                    keep_idx=keep_idx)
                base_plan["gen"]["captions"] = {int(k): v for k, v in
                                                gen.get("captions", {}).items()}
                if gen.get("cards"):
                    base_plan["gen"]["cards"] = gen["cards"]
            except Exception:
                pass
        base_plan = _plan_normalize(base_plan)
        with _dir_lock:
            entry["history"].append(entry["plan"])
            entry["plan"] = base_plan
        pub = _plan_public(base_plan, lines)
        _dir_set(jid, stage="done", progress=100, status="done",
                 result={"kind": "plan", "plan": pub, "session": sess})
    except Exception as e:
        _dir_set(jid, status="error", stage="error",
                 error=f"{type(e).__name__}: {e}")


def _run_director_render(jid: str, opts: dict) -> None:
    try:
        sess = opts["session"]
        with _dir_lock:
            entry = _dir_plans.get(sess)
        if not entry:
            raise RuntimeError("no active plan for this session — run Plan first")
        path = entry["path"]
        plan = _plan_from_public(opts.get("plan") or {})
        with _dir_lock:
            entry["plan"] = plan
        _dir_set(jid, status="running", stage="starting render", progress=8)
        report = _director_execute(jid, path, plan, sess)
        _dir_set(jid, stage="done", progress=100, status="done",
                 result={"kind": "render", **report})
    except Exception as e:
        _dir_set(jid, status="error", stage="error",
                 error=f"{type(e).__name__}: {e}")


def _run_cleanup(jid: str, opts: dict) -> None:
    tmps = []
    try:
        path = opts["path"]
        mode = opts.get("mode", "render")
        _clean_set(jid, stage="probing", progress=5)
        info = _probe_media(path)
        dur = info["duration"]
        if dur <= 0:
            raise RuntimeError("could not read media duration")
        base = os.path.join(OUT_DIR, "_clean_" + jid)
        explicit = opts.get("cuts")
        enhance = (bool(opts.get("enhance_video")) and info["has_video"]
                   and _ci_enhance_available())
        want_audio = (opts["denoise"] or opts["remove_fillers"]
                      or opts["remove_silences"] or explicit is not None)
        if want_audio and not info["has_audio"]:
            raise RuntimeError("no audio stream found in this file")

        # ---------------- ANALYZE (preview, no render) ----------------
        if mode == "analyze":
            if not info["has_audio"]:
                raise RuntimeError("no audio stream to analyze")
            _clean_set(jid, stage="extracting audio", progress=20)
            audio48 = base + ".48k.wav"
            tmps.append(audio48)
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn",
                  "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", audio48])
            hits, sil, text = [], [], ""
            if opts["remove_fillers"]:
                _clean_set(jid, stage="transcribing", progress=55)
                wav16 = base + ".16k.wav"
                tmps.append(wav16)
                _run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio48,
                      "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16])
                res = _enqueue("transcribe_words",
                               {"path": wav16, "initial_prompt": opts.get("prompt")},
                               timeout=3600)
                if "error" in res:
                    raise RuntimeError(res["error"])
                data = res["data"]
                text = data["text"]
                fcuts, hits = _filler_cuts(data["words"], opts["fillers"], opts["pad"])
                for idx, h in enumerate(hits):
                    h["id"] = "f%d" % idx
                    h["cut"] = [round(fcuts[idx][0], 3), round(fcuts[idx][1], 3)]
            if opts["remove_silences"]:
                _clean_set(jid, stage="detecting silence", progress=80)
                sraw = _silence_cuts(audio48, opts["max_silence"],
                                     opts["keep_silence"], opts["silence_db"])
                sil = [[round(a, 3), round(b, 3)] for a, b in sraw]
            report = {
                "kind": "analysis",
                "source": os.path.basename(path),
                "has_video": info["has_video"],
                "duration": round(dur, 2),
                "filler_hits": hits,
                "silence_cuts": sil,
                "transcript": text[:8000],
            }
            _clean_set(jid, stage="done", progress=100, status="done", result=report)
            return

        # ---------------- RENDER ----------------
        render48 = None
        denoised = False
        if want_audio:  # implies has_audio
            _clean_set(jid, stage="extracting audio", progress=15)
            audio48 = base + ".48k.wav"
            tmps.append(audio48)
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn",
                  "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", audio48])
            render48 = audio48
            denoised = bool(opts["denoise"]) and _denoise_available()
            if denoised:
                _clean_set(jid, stage="denoising", progress=28)
                den = base + ".den48.wav"
                tmps.append(den)
                _denoise_file(audio48, den, 48000)
                render48 = den

        cuts, hits, sil, text = [], [], [], ""
        words = None
        n_fillers, n_silences = 0, 0
        if explicit is not None:
            cuts = [(float(a), float(b)) for a, b in explicit if float(b) > float(a)]
            n_fillers = int(opts.get("cuts_fillers", 0))
            n_silences = int(opts.get("cuts_silences", 0))
        elif render48 is not None:
            if opts["remove_fillers"]:
                _clean_set(jid, stage="transcribing", progress=48)
                wav16 = base + ".16k.wav"
                tmps.append(wav16)
                _run(["ffmpeg", "-y", "-loglevel", "error", "-i", render48,
                      "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16])
                res = _enqueue("transcribe_words",
                               {"path": wav16, "initial_prompt": opts.get("prompt")},
                               timeout=3600)
                if "error" in res:
                    raise RuntimeError(res["error"])
                data = res["data"]
                text = data["text"]
                words = data["words"]
                fcuts, hits = _filler_cuts(data["words"], opts["fillers"], opts["pad"])
                cuts += fcuts
                n_fillers = len(hits)
            if opts["remove_silences"]:
                _clean_set(jid, stage="trimming silence", progress=64)
                sil = _silence_cuts(render48, opts["max_silence"],
                                    opts["keep_silence"], opts["silence_db"])
                cuts += sil
                n_silences = len(sil)

        # Viral length target: keep only the LLM-selected highlight windows
        # (video only; skipped when the user is applying manual review cuts).
        vlen = (_length_target_seconds(opts.get("viral_length"))
                if explicit is None else None)
        if vlen and info["has_video"] and render48 is not None:
            if words is None:
                _clean_set(jid, stage="transcribing for length cut", progress=50)
                wav16l = base + ".len16k.wav"
                tmps.append(wav16l)
                _run(["ffmpeg", "-y", "-loglevel", "error", "-i", render48,
                      "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16l])
                res = _enqueue("transcribe_words",
                               {"path": wav16l, "initial_prompt": opts.get("prompt")},
                               timeout=3600)
                if "error" not in res:
                    words = res["data"]["words"]
                    if not text:
                        text = res["data"]["text"]
            if words:
                _clean_set(jid, stage="selecting highlights for length", progress=58)
                try:
                    hl_lines = _segment_lines(words)
                    idx = _director_highlights(hl_lines, vlen,
                                               opts.get("caption_topic", ""))
                    iv = _lines_to_intervals(hl_lines, idx, dur)
                    if iv:
                        cuts += _cuts_from_keeps(iv, dur)
                except Exception:
                    pass

        has_cuts = bool(cuts)
        keeps = _keep_intervals(cuts, dur) if has_cuts else None
        if has_cuts and not keeps:
            raise RuntimeError("everything would be cut — loosen the settings")

        # Viral-edit options (video only).
        want_punchin = bool(opts.get("dynamic_edit")) and info["has_video"]
        want_caps = bool(opts.get("captions")) and info["has_video"]
        want_cards = bool(opts.get("cards")) and info["has_video"]
        want_overlays = want_caps or want_cards
        # The keep-intervals that define the OUTPUT timeline (whole clip if no cuts).
        out_keeps = keeps if has_cuts else [(0.0, dur)]

        # Captions need word timestamps even when fillers aren't being removed.
        if want_overlays and words is None and render48 is not None:
            _clean_set(jid, stage="transcribing for captions", progress=52)
            wav16c = base + ".caps16k.wav"
            tmps.append(wav16c)
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", render48,
                  "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav16c])
            res = _enqueue("transcribe_words",
                           {"path": wav16c, "initial_prompt": opts.get("prompt")},
                           timeout=3600)
            if "error" in res:
                raise RuntimeError(res["error"])
            words = res["data"]["words"]
            if not text:
                text = res["data"]["text"]

        _clean_set(jid, stage="rendering", progress=82)
        eid = "clean_" + uuid.uuid4().hex[:12]
        captioned = False
        if info["has_video"]:
            fmt = "mp4"
            out = base + ".mp4"
            if render48 is not None:
                if has_cuts and want_punchin:
                    _render_punchin_video(path, render48, keeps, out,
                                          opts["loudness"])
                elif has_cuts:
                    _render_clean_video(path, render48, keeps, out, opts["loudness"])
                else:
                    _remux_video(path, render48, out, opts["loudness"])
            else:
                _passthrough_media(path, out)
            if enhance:
                _clean_set(jid, stage="enhancing light & colour", progress=92)
                out = _apply_enhance(out, base, opts)
            if ASPECTS.get(opts.get("aspect")):
                _clean_set(jid, stage="reframing aspect ratio", progress=95)
                out = _apply_aspect(out, base, opts)
            if want_overlays and words:
                lines = _segment_lines(words)
                try:
                    stage_msg = ("writing smart captions" if want_caps
                                 else "writing hook cards")
                    _clean_set(jid, stage=stage_msg, progress=97)
                    gen = _gen_captions(lines, want_captions=want_caps,
                                        want_cards=want_cards,
                                        topic=opts.get("caption_topic", ""),
                                        length_target=opts.get("viral_length", "auto"),
                                        keep_idx=_kept_line_indices(lines, out_keeps))
                except Exception as ce:
                    _clean_set(jid, stage="smart captions failed, using fallback", progress=98)
                    opts["_caption_error"] = f"{type(ce).__name__}: {ce}"
                    gen = {"captions": {}, "cards": []}
                items = _build_overlays(lines, gen, out_keeps,
                                        include_captions=want_caps,
                                        include_cards=want_cards)
                if items:
                    try:
                        out = _composite_captions(out, items, base)
                        captioned = True
                    except Exception as ce2:
                        captioned = False
                        opts["_caption_error"] = f"{type(ce2).__name__}: {ce2}"
        else:
            fmt = opts["format"]
            out = base + "." + fmt
            _render_clean_audio(render48, keeps, out, fmt, opts["loudness"])

        final = os.path.join(OUT_DIR, eid + "." + fmt)
        os.replace(out, final)
        new_dur = _audio_seconds(final)

        # Also save a copy next to the original file, named "<original> cleaned.<ext>".
        saved_to, save_error = None, None
        try:
            src_dir = os.path.dirname(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            cand = os.path.join(src_dir, f"{stem} cleaned.{fmt}")
            i = 2
            while os.path.exists(cand):
                cand = os.path.join(src_dir, f"{stem} cleaned ({i}).{fmt}")
                i += 1
            shutil.copy2(final, cand)
            saved_to = cand
        except Exception as e:
            save_error = f"{type(e).__name__}: {e}"

        report = {
            "filename": os.path.basename(final),
            "url": f"/files/{os.path.basename(final)}",
            "kind": "video" if info["has_video"] else "audio",
            "source": os.path.basename(path),
            "saved_to": saved_to,
            "save_error": save_error,
            "orig_seconds": round(dur, 2),
            "new_seconds": round(new_dur, 2),
            "saved_seconds": round(max(0.0, dur - new_dur), 2),
            "denoised": denoised,
            "enhanced": enhance,
            "punch_in": bool(want_punchin and has_cuts),
            "captioned": captioned,
            "caption_error": opts.get("_caption_error"),
            "viral_length": opts.get("viral_length", "auto"),
            "aspect": (opts.get("aspect") if ASPECTS.get(opts.get("aspect"))
                       else None),
            "fillers_removed": n_fillers,
            "filler_hits": hits[:300],
            "silences_removed": n_silences,
            "transcript": text[:8000],
        }

        # Index the transcript into the shared sqlite-vec RAG so the whole
        # processed-video library becomes searchable. Best-effort.
        if opts.get("index_transcript", True) and text.strip():
            try:
                res = _index_transcript(os.path.basename(path), text,
                                        {"saved_to": saved_to,
                                         "aspect": report["aspect"]})
                report["indexed"] = bool(res.get("doc_id"))
                report["index_doc_id"] = res.get("doc_id")
                report["index_chunks"] = res.get("n_chunks")
            except Exception as ie:
                report["indexed"] = False
                report["index_error"] = f"{type(ie).__name__}: {ie}"

        _clean_set(jid, stage="done", progress=100, status="done", result=report)
    except Exception as e:
        _clean_set(jid, stage="error", status="error",
                   error=f"{type(e).__name__}: {e}")
    finally:
        for t in tmps:
            _safe_rm(t)


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "TTSStudio/2.0"

    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
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

    # ---- GET ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = STUDIO_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            self._json(200, {
                "ready": _worker_started.is_set(),
                "loaded": sorted(_models.keys()),
                "errors": _load_errors,
                "default_engine": DEFAULT_ENGINE,
            })
            return
        if path == "/engines":
            self._json(200, {
                "engines": [{
                    "id": k, "label": v["label"], "clone": v["clone"],
                    "presets": v["presets"],
                    "voices": (VOXTRAL_VOICES if v["presets"]
                               else _kokoro_voices() if v.get("kokoro") else []),
                } for k, v in ENGINES.items()],
                "formats": FORMATS,
                "sample_rates": SAMPLE_RATES,
                "lufs": LUFS_TARGETS,
                "refs": _refs_list(),
                "default_engine": DEFAULT_ENGINE,
                "denoise_available": _denoise_available(),
                "enhance_available": _ci_enhance_available(),
                "captions_available": _captions_available(),
                "director_available": _captions_available(),
                "director_styles": DIR_STYLES,
                "viral_lengths": list(VIRAL_LENGTH_TARGETS.keys()),
                "aspects": list(ASPECTS.keys()),
                "memory_available": _memory_available(),
                "memory_url": MEMORY_URL,
                "transcript_collection": TRANSCRIPT_COLLECTION,
            })
            return
        if path == "/say/voices":
            self._json(200, {"voices": _kokoro_voices(),
                             "default": KOKORO_DEFAULT_VOICE})
            return
        if path == "/history":
            self._json(200, {"history": _history_list()})
            return
        if path == "/llm":
            self._json(200, {"choices": LLM_CHOICES, "current": _get_llm()})
            return
        if path == "/cleanup/status":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            job = _clean_get(jid)
            if not job:
                self._json(404, {"error": "job not found"})
                return
            self._json(200, job)
            return
        if path == "/director/status":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            jid = (qs.get("id") or [""])[0]
            job = _dir_get(jid)
            if not job:
                self._json(404, {"error": "job not found"})
                return
            self._json(200, job)
            return
        if path == "/refs":
            self._json(200, {"refs": _refs_list()})
            return
        if path.startswith("/files/"):
            self._serve(OUT_DIR, path[len("/files/"):])
            return
        if path.startswith("/reffiles/"):
            self._serve(REF_DIR, path[len("/reffiles/"):])
            return
        self._json(404, {"error": "not found"})

    def _serve(self, base: str, name: str):
        name = os.path.basename(name)
        fpath = os.path.join(base, name)
        if not os.path.isfile(fpath):
            self._json(404, {"error": "file not found"})
            return
        ext = name.rsplit(".", 1)[-1].lower()
        ctype = {"wav": "audio/wav", "mp3": "audio/mpeg",
                 "flac": "audio/flac", "mp4": "video/mp4",
                 "m4a": "audio/mp4", "mov": "video/quicktime",
                 "mkv": "video/x-matroska", "webm": "video/webm",
                 "m4v": "video/x-m4v"}.get(ext, "application/octet-stream")
        size = os.path.getsize(fpath)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---- POST ----
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._json(400, {"error": f"invalid JSON: {e}"})
            return

        if path == "/generate":
            return self._generate(data)
        if path == "/say":
            return self._say(data)
        if path == "/history/delete":
            return self._json(200, {"ok": _history_delete(data.get("id", ""))})
        if path == "/history/favorite":
            return self._json(200, {"ok": _history_favorite(
                data.get("id", ""), data.get("value", True))})
        if path == "/refs/upload":
            try:
                entry = _ref_add(data.get("name", ""), data.get("data", ""),
                                 data.get("ref_text"), bool(data.get("denoise")))
                return self._json(200, {"ok": True, "ref": entry})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        if path == "/refs/delete":
            return self._json(200, {"ok": _ref_delete(data.get("id", ""))})
        if path == "/cleanup":
            return self._cleanup(data)
        if path == "/llm":
            try:
                cur = _set_llm(data.get("model", ""))
                return self._json(200, {"ok": True, "current": cur})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if path in ("/director/prep", "/director/plan", "/director/revise", "/director/render"):
            return self._director(path, data)
        self._json(404, {"error": "not found"})

    # ---- LLM edit director orchestration ----
    def _director(self, path: str, data: dict):
        if not _worker_started.is_set():
            return self._json(503, {"error": "starting up, try again shortly"})
        jid = "dir_" + uuid.uuid4().hex[:12]
        if path == "/director/prep":
            mpath = _resolve_media_path(data.get("path", ""))
            if not mpath or not os.path.isfile(mpath):
                return self._json(400, {"error": f"file not found: {mpath or '(empty)'}"})
            opts = {
                "path": mpath,
                "session": data.get("session") or ("sess_" + uuid.uuid4().hex[:8]),
                "platform": data.get("platform", "youtube"),
                "denoise": bool(data.get("denoise", True)),
                "enhance": bool(data.get("enhance", True)),
                "enhance_face": bool(data.get("enhance_face", False)),
                "enhance_level": data.get("enhance_level", 1.0),
            }
            target = _run_director_prep
        elif path == "/director/plan":
            mpath = _resolve_media_path(data.get("path", ""))
            if not mpath or not os.path.isfile(mpath):
                return self._json(400, {"error": f"file not found: {mpath or '(empty)'}"})
            opts = {
                "path": mpath,
                "session": data.get("session") or ("sess_" + uuid.uuid4().hex[:8]),
                "platform": data.get("platform", "youtube"),
                "topic": (data.get("topic") or "")[:400],
                "length": (data.get("length")
                           if data.get("length") in VIRAL_LENGTH_TARGETS else None),
                "aspect": (data.get("aspect") if data.get("aspect") in ASPECTS else None),
            }
            target = _run_director_plan
        elif path == "/director/revise":
            if not data.get("session"):
                return self._json(400, {"error": "session required"})
            opts = {"session": data["session"], "plan": data.get("plan") or {},
                    "feedback": (data.get("feedback") or "")[:2000],
                    "regen_captions": bool(data.get("regen_captions"))}
            target = _run_director_revise
        else:  # /director/render
            if not data.get("session"):
                return self._json(400, {"error": "session required"})
            opts = {"session": data["session"], "plan": data.get("plan") or {}}
            target = _run_director_render
        _dir_set(jid, status="running", stage="queued", progress=0,
                 result=None, error=None)
        threading.Thread(target=target, args=(jid, opts), daemon=True).start()
        return self._json(200, {"ok": True, "job_id": jid,
                                "session": opts.get("session")})

    # ---- media cleanup orchestration ----
    def _cleanup(self, data: dict):
        if not _worker_started.is_set():
            return self._json(503, {"error": "starting up, try again shortly"})
        path = _resolve_media_path(data.get("path", ""))
        if not path or not os.path.isfile(path):
            return self._json(400, {"error": f"file not found: {path or '(empty)'}"})

        def fnum(key, d, lo, hi):
            try:
                return max(lo, min(hi, float(data.get(key, d))))
            except Exception:
                return d

        fillers = data.get("fillers")
        if isinstance(fillers, str):
            fillers = [t for t in re.split(r"[\s,]+", fillers) if t]
        if not fillers:
            fillers = list(DEFAULT_FILLERS)

        remove_fillers = bool(data.get("remove_fillers", True))
        mode = "analyze" if data.get("mode") == "analyze" else "render"
        cuts = data.get("cuts")
        if cuts is not None:
            try:
                cuts = [[float(a), float(b)] for a, b in cuts]
            except Exception:
                return self._json(400, {"error": "invalid cuts list"})
        opts = {
            "path": path,
            "mode": mode,
            "cuts": cuts,
            "cuts_fillers": data.get("cuts_fillers", 0),
            "cuts_silences": data.get("cuts_silences", 0),
            "denoise": bool(data.get("denoise", True)),
            "remove_fillers": remove_fillers,
            "remove_silences": bool(data.get("remove_silences", False)),
            "enhance_video": bool(data.get("enhance_video", False)),
            "enhance_face": bool(data.get("enhance_face", False)),
            "enhance_level": fnum("enhance_level", 1.0, 0.0, 1.0),
            "aspect": (data.get("aspect") if data.get("aspect") in ASPECTS
                       else "original"),
            "aspect_fit": (data.get("aspect_fit")
                           if data.get("aspect_fit") in
                           ("fill", "pad_black", "pad_blur") else "fill"),
            "dynamic_edit": bool(data.get("dynamic_edit", False)),
            "captions": bool(data.get("captions", False)),
            "cards": bool(data.get("cards", False)),
            "caption_topic": (data.get("caption_topic") or "")[:400],
            "viral_length": (data.get("viral_length")
                             if data.get("viral_length") in VIRAL_LENGTH_TARGETS
                             else "auto"),
            "index_transcript": bool(data.get("index_transcript", True)),
            "fillers": fillers,
            "pad": fnum("pad_ms", 60, 0, 500) / 1000.0,
            "max_silence": fnum("max_silence_ms", 700, 150, 10000) / 1000.0,
            "keep_silence": fnum("keep_silence_ms", 250, 0, 2000) / 1000.0,
            "silence_db": fnum("silence_db", -32, -90, -10),
            "loudness": data.get("loudness", "youtube"),
            "format": (data.get("format") or "wav").lower(),
            "prompt": (data.get("prompt")
                       or ("Um, uh, er, ah, hmm, you know, like, so, I mean."
                           if remove_fillers else None)),
        }
        if opts["format"] not in FORMATS:
            opts["format"] = "wav"
        if mode == "analyze":
            if not (opts["remove_fillers"] or opts["remove_silences"]):
                return self._json(400, {"error": "enable fillers or silences to preview"})
        else:
            any_step = (opts["denoise"] or opts["remove_fillers"]
                        or opts["remove_silences"] or opts["enhance_video"]
                        or bool(ASPECTS.get(opts["aspect"]))
                        or opts["dynamic_edit"] or opts["captions"]
                        or opts["cards"]
                        or cuts is not None)
            if not any_step:
                return self._json(400, {"error": "enable at least one cleanup step"})

        jid = "clj_" + uuid.uuid4().hex[:12]
        _clean_set(jid, status="running", stage="queued", progress=0,
                   result=None, error=None)
        threading.Thread(target=_run_cleanup, args=(jid, opts),
                         daemon=True).start()
        return self._json(200, {"ok": True, "job_id": jid})

    # ---- generate orchestration ----
    def _say(self, data: dict):
        """Low-latency chat reader: synthesize one short utterance with Kokoro
        and stream the raw WAV back directly. No mastering/takes/history — kept
        deliberately lean so the in-chat voice stays responsive."""
        if not _worker_started.is_set():
            return self._json(503, {"error": "starting up, try again shortly"})
        text = (data.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": "text is required"})
        text = text[:1500]  # chat utterances are short; guard runaway inputs
        voice = data.get("voice") or KOKORO_DEFAULT_VOICE
        if voice not in _kokoro_voices():
            voice = KOKORO_DEFAULT_VOICE
        try:
            speed = max(0.5, min(2.0, float(data.get("speed", 1.0))))
        except Exception:
            speed = 1.0
        params = {
            "engine": "kokoro",
            "text": text,
            "voice": voice,
            "speed": speed,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_len": None,
            "seed": None,
        }
        res = _enqueue("gen", params, timeout=120)
        if "error" in res:
            return self._json(500, {"error": res["error"]})
        raw = res.get("raw")
        if not raw or not os.path.isfile(raw):
            return self._json(500, {"error": "no audio produced"})
        try:
            with open(raw, "rb") as f:
                audio = f.read()
        finally:
            _safe_rm(raw)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self._cors()
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def _generate(self, data: dict):
        if not _worker_started.is_set():
            return self._json(503, {"error": "starting up, try again shortly"})

        text = (data.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": "text is required"})
        if len(text) > MAX_CHARS:
            return self._json(400, {"error": f"text too long (>{MAX_CHARS})"})

        engine = data.get("engine") or DEFAULT_ENGINE
        if engine not in ENGINES:
            return self._json(400, {"error": f"unknown engine '{engine}'"})

        fmt = (data.get("format") or "wav").lower()
        if fmt not in FORMATS:
            return self._json(400, {"error": f"unknown format '{fmt}'"})

        def num(key, d, lo, hi, cast=float):
            try:
                return max(lo, min(hi, cast(data.get(key, d))))
            except Exception:
                return d

        temperature = num("temperature", 0.7, 0.1, 1.5)
        top_p = num("top_p", 0.95, 0.1, 1.0)
        top_k = num("top_k", 50, 0, 200, int)
        speed = num("speed", 1.0, 0.5, 2.0)
        takes = num("takes", 1, 1, 8, int)

        # voice / cloning
        voice = data.get("voice") or "neutral_female"
        ref_audio = None
        ref_text = None
        if ENGINES[engine]["presets"]:
            if voice not in VOXTRAL_VOICES:
                return self._json(400, {"error": f"unknown voice '{voice}'"})
        elif ENGINES[engine].get("kokoro"):
            if voice == "neutral_female":
                voice = KOKORO_DEFAULT_VOICE
            if voice not in _kokoro_voices():
                return self._json(400, {"error": f"unknown kokoro voice '{voice}'"})
        if ENGINES[engine]["clone"] and data.get("ref_id"):
            ref = _ref_get(data["ref_id"])
            if not ref:
                return self._json(400, {"error": "reference not found"})
            ref_audio = os.path.join(REF_DIR, ref["filename"])
            ref_text = ref.get("ref_text") or None

        master = {
            "format": fmt,
            "speed": speed,
            "lufs": LUFS_TARGETS.get(data.get("loudness", "youtube"), -14.0),
            "sample_rate": int(data.get("sample_rate", 48000)),
            "channels": 2 if data.get("stereo") else 1,
            "fade_in": num("fade_in", 0, 0, 10000, int),
            "fade_out": num("fade_out", 0, 0, 10000, int),
            "trim": bool(data.get("trim", True)),
            "mp3_bitrate": data.get("mp3_bitrate", "192k"),
        }

        script_mode = bool(data.get("script"))
        pause_ms = num("pause_ms", 350, 0, 3000, int)
        denoise = bool(data.get("denoise"))
        base_seed = data.get("seed")
        try:
            base_seed = int(base_seed) if base_seed not in (None, "") else None
        except Exception:
            base_seed = None

        t0 = time.time()
        try:
            if script_mode:
                results = [self._render_script(
                    text, engine, voice, temperature, top_p, top_k,
                    ref_audio, ref_text, base_seed, pause_ms, master, denoise)]
            else:
                results = []
                for i in range(takes):
                    seed = (base_seed + i) if base_seed is not None else None
                    results.append(self._render_one(
                        text, engine, voice, temperature, top_p, top_k,
                        ref_audio, ref_text, seed, master, denoise))
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        for r in results:
            _history_add({
                "id": r["id"], "filename": r["filename"],
                "text": text[:5000], "snippet": text[:120],
                "engine": engine, "voice": (None if ENGINES[engine]["clone"] else voice),
                "ref_id": data.get("ref_id"), "ref_name": (_ref_get(data["ref_id"]) or {}).get("name") if data.get("ref_id") else None,
                "seconds": r["seconds"], "seed": r["seed"],
                "temperature": temperature, "top_p": top_p, "top_k": top_k,
                "speed": speed, "format": fmt, "loudness": data.get("loudness", "youtube"),
                "sample_rate": master["sample_rate"], "favorite": False,
                "created": datetime.now().isoformat(timespec="seconds"),
            })

        req_echo = {"text": text, "engine": engine, "format": fmt,
                    "temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "speed": speed, "loudness": data.get("loudness", "youtube"),
                    "sample_rate": master["sample_rate"], "trim": master["trim"]}
        if ENGINES[engine]["presets"]:
            req_echo["voice"] = voice
        elif data.get("ref_id"):
            req_echo["ref_id"] = data["ref_id"]
        if master["channels"] == 2:
            req_echo["stereo"] = True
        if master["fade_in"]:
            req_echo["fade_in"] = master["fade_in"]
        if master["fade_out"]:
            req_echo["fade_out"] = master["fade_out"]
        if fmt == "mp3":
            req_echo["mp3_bitrate"] = master["mp3_bitrate"]
        if script_mode:
            req_echo["script"] = True
            req_echo["pause_ms"] = pause_ms
        elif takes > 1:
            req_echo["takes"] = takes
        if denoise:
            req_echo["denoise"] = True
        if base_seed is not None:
            req_echo["seed"] = base_seed
        curl_cmd = _curl_for(req_echo)

        first = results[0]
        self._json(200, {
            "ok": True,
            "engine": engine,
            "voice": (None if ENGINES[engine]["clone"] else voice),
            "format": fmt,
            "gen_seconds": round(time.time() - t0, 2),
            "curl": curl_cmd,
            "request": req_echo,
            "takes": [{
                "id": r["id"], "filename": r["filename"],
                "url": f"/files/{r['filename']}", "seconds": r["seconds"],
                "seed": r["seed"],
                "calls": r.get("calls", []),
                "master_cmd": r.get("master_cmd", ""),
            } for r in results],
            **{k: first[k] for k in ("filename", "seconds", "seed")},
            "url": f"/files/{first['filename']}",
        })

    def _render_one(self, text, engine, voice, temperature, top_p, top_k,
                    ref_audio, ref_text, seed, master, denoise=False):
        max_len = _auto_len(engine, text, has_ref=bool(ref_audio))
        res = _gen_guarded({
            "engine": engine, "text": text, "voice": voice,
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "max_len": max_len, "seed": seed,
            "ref_audio": ref_audio, "ref_text": ref_text,
        }, engine, text)
        if "error" in res:
            raise RuntimeError(res["error"])
        raw = res["raw"]
        cleaned = None
        eid = "tts_" + uuid.uuid4().hex[:12]
        out = os.path.join(OUT_DIR, f"{eid}.{master['format']}")
        try:
            src = raw
            if denoise:
                cleaned = os.path.join(OUT_DIR, "_dn_" + uuid.uuid4().hex[:8] + ".wav")
                _denoise_file(raw, cleaned, 48000)
                src = cleaned
            mcmd = _master(src, out, master)
        finally:
            _safe_rm(raw)
            if cleaned:
                _safe_rm(cleaned)
        return {"id": eid, "filename": os.path.basename(out),
                "seconds": _audio_seconds(out), "seed": seed,
                "calls": [res.get("call")] if res.get("call") else [],
                "master_cmd": _fmt_cmd(mcmd)}

    def _render_script(self, text, engine, voice, temperature, top_p, top_k,
                       ref_audio, ref_text, base_seed, pause_ms, master, denoise=False):
        segs, gap_kinds = _split_script(text)
        if not segs:
            raise RuntimeError("nothing to render")
        raws = []
        calls = []
        try:
            for i, seg in enumerate(segs):
                seed = (base_seed + i) if base_seed is not None else None
                res = _gen_guarded({
                    "engine": engine, "text": seg, "voice": voice,
                    "temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "max_len": _auto_len(engine, seg, has_ref=bool(ref_audio)), "seed": seed,
                    "ref_audio": ref_audio, "ref_text": ref_text,
                }, engine, seg)
                if "error" in res:
                    raise RuntimeError(res["error"])
                raws.append(res["raw"])
                if res.get("call"):
                    calls.append(res["call"])
            gaps = [pause_ms * 2 if k == "para" else pause_ms for k in gap_kinds]
            joined = os.path.join(OUT_DIR, f"_join_{uuid.uuid4().hex[:8]}.wav")
            _concat_with_pauses(raws, gaps, joined)
            cleaned = None
            eid = "tts_" + uuid.uuid4().hex[:12]
            out = os.path.join(OUT_DIR, f"{eid}.{master['format']}")
            try:
                src = joined
                if denoise:
                    cleaned = os.path.join(OUT_DIR, "_dn_" + uuid.uuid4().hex[:8] + ".wav")
                    _denoise_file(joined, cleaned, 48000)
                    src = cleaned
                mcmd = _master(src, out, master)
            finally:
                _safe_rm(joined)
                if cleaned:
                    _safe_rm(cleaned)
            return {"id": eid, "filename": os.path.basename(out),
                    "seconds": _audio_seconds(out), "seed": base_seed,
                    "calls": calls, "master_cmd": _fmt_cmd(mcmd)}
        finally:
            for r in raws:
                _safe_rm(r)


def _expected_seconds(text: str) -> float:
    """Rough expected speech duration (~2.8 words/sec)."""
    return max(1, len(text.split())) / 2.8


def _gen_guarded(params: dict, engine: str, text: str, retries: int = 4) -> dict:
    """Run a generation job; for Higgs, detect degenerate (too-short / empty)
    takes and retry with a fresh random seed. Higgs occasionally collapses to
    a near-silent or truncated take regardless of voice; a short retry loop
    makes every built-in voice reliable for production use."""
    res = _enqueue("gen", params)
    if engine != "higgs" or "error" in res:
        return res
    floor = max(0.4, _expected_seconds(text) * 0.45)
    for _ in range(retries):
        raw = res.get("raw")
        secs = _audio_seconds(raw) if raw and os.path.exists(raw) else 0.0
        vol = _mean_volume_db(raw) if secs > 0 else -120.0
        # accept only takes that are both long enough AND not near-silent
        if secs >= floor and vol > -55.0:
            return res
        if raw:
            _safe_rm(raw)
        params = {**params, "seed": random.randint(1, 2_147_483_646)}
        res = _enqueue("gen", params)
        if "error" in res:
            return res
    return res


def _auto_len(engine: str, text: str, has_ref: bool = False) -> int:
    words = max(1, len(text.split()))
    if engine == "voxtral":
        return int(max(400, min(8192, words * 55)))
    # higgs frames (~40 ms each). Cloning stops at EOS so allow generous
    # headroom; smart-voice can ramble, so bound it tighter.
    if has_ref:
        return int(max(300, min(6000, words * 28 + 200)))
    return int(max(150, min(1400, words * 18 + 60)))


def _safe_rm(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Studio - Pro</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:#0d1117; color:#e6edf3; display:flex; justify-content:center; padding:28px 16px 80px; }
  .wrap { width:100%; max-width:860px; }
  h1 { font-size:20px; margin:0 0 4px; display:flex; align-items:center; gap:10px; }
  .sub { color:#8b949e; font-size:13px; margin:0 0 20px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:14px; padding:20px; margin-bottom:18px; }
  textarea { width:100%; min-height:150px; resize:vertical; background:#0d1117; color:#e6edf3;
             border:1px solid #30363d; border-radius:10px; padding:14px; font-size:15px; line-height:1.55; outline:none; }
  textarea:focus { border-color:#388bfd; }
  .row { display:flex; gap:14px; flex-wrap:wrap; margin-top:16px; align-items:flex-end; }
  .field { display:flex; flex-direction:column; gap:6px; }
  label { font-size:12px; color:#8b949e; text-transform:uppercase; letter-spacing:.4px; }
  select, input[type=number], input[type=text] { background:#0d1117; color:#e6edf3; border:1px solid #30363d;
             border-radius:8px; padding:9px 10px; font-size:14px; outline:none; min-width:140px; }
  select:focus, input:focus { border-color:#388bfd; }
  .spacer { flex:1; }
  button.primary { background:#238636; color:#fff; border:none; border-radius:9px; padding:11px 22px;
           font-size:15px; font-weight:600; cursor:pointer; transition:.15s; }
  button.primary:hover { background:#2ea043; }
  button.primary:disabled { background:#30363d; color:#8b949e; cursor:not-allowed; }
  .mini { background:#21262d; color:#c9d1d9; padding:9px 11px; font-size:14px; border:1px solid #30363d;
          border-radius:8px; cursor:pointer; }
  .mini:hover { background:#30363d; }
  .status { margin-top:14px; font-size:13px; color:#8b949e; min-height:18px; }
  .status.err { color:#f85149; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#3fb950; }
  .dot.load { background:#d29922; animation:pulse 1s infinite; }
  .dot.off { background:#f85149; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  details { border-top:1px solid #21262d; padding-top:14px; margin-top:16px; }
  summary { cursor:pointer; color:#8b949e; font-size:12px; text-transform:uppercase; letter-spacing:.4px;
            list-style:none; user-select:none; }
  summary::-webkit-details-marker { display:none; }
  summary::before { content:'> '; }
  details[open] summary::before { content:'v '; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px 18px; margin-top:14px; }
  .slider { display:flex; flex-direction:column; gap:6px; }
  .slider .lab { display:flex; justify-content:space-between; align-items:baseline; }
  .slider .lab b { color:#e6edf3; font-weight:600; font-size:13px; }
  input[type=range] { width:100%; accent-color:#388bfd; }
  .hint { font-size:11px; color:#6e7681; line-height:1.4; }
  .seedrow { display:flex; gap:8px; align-items:center; }
  .seedrow input { min-width:0; flex:1; }
  .take { display:flex; align-items:center; gap:10px; flex-wrap:wrap; background:#0d1117; border:1px solid #30363d;
          border-radius:10px; padding:10px 12px; margin-top:10px; }
  .take audio { flex:1; height:34px; }
  .take .tag { font-size:11px; color:#8b949e; white-space:nowrap; }
  .take a, .take button { color:#58a6ff; text-decoration:none; font-size:13px; background:none; border:none; cursor:pointer; }
  .take .cmd { flex-basis:100%; width:100%; margin-top:4px; }
  .take .cmd summary { font-size:12px; color:#8b949e; cursor:pointer; list-style:none; user-select:none; }
  .take .cmd summary:hover { color:#58a6ff; }
  .take .cmd pre { margin:8px 0 2px; padding:10px 12px; background:#010409; border:1px solid #21262d;
          border-radius:8px; font-size:11.5px; line-height:1.5; color:#c9d1d9; white-space:pre-wrap;
          word-break:break-word; overflow-x:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  details.curlbox { background:#0d1117; border:1px solid #30363d; border-radius:10px; padding:10px 12px; margin-top:10px; }
  details.curlbox summary { font-size:12px; color:#8b949e; cursor:pointer; list-style:none; user-select:none; }
  details.curlbox summary:hover { color:#58a6ff; }
  details.curlbox pre { margin:8px 0; padding:10px 12px; background:#010409; border:1px solid #21262d;
          border-radius:8px; font-size:11.5px; line-height:1.5; color:#c9d1d9; white-space:pre-wrap;
          word-break:break-word; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  button.copy { font-size:12px; color:#c9d1d9; background:#21262d; border:1px solid #30363d;
          border-radius:6px; padding:5px 10px; cursor:pointer; }
  button.copy:hover { border-color:#58a6ff; color:#58a6ff; }
  .pill { display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px; background:#21262d; color:#8b949e; }
  .hrow { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid #21262d; font-size:13px; }
  .hrow .snip { flex:1; color:#c9d1d9; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .hrow audio { height:30px; width:200px; }
  .hrow button, .hrow a { background:none; border:none; color:#8b949e; cursor:pointer; font-size:14px; text-decoration:none; }
  .hrow button:hover, .hrow a:hover { color:#e6edf3; }
  .fav { color:#e3b341 !important; }
  .refchip { display:flex; align-items:center; gap:8px; background:#0d1117; border:1px solid #30363d;
             border-radius:8px; padding:6px 10px; margin-top:8px; font-size:13px; }
  .toggle { display:flex; align-items:center; gap:8px; font-size:13px; color:#c9d1d9; text-transform:none; letter-spacing:0; }
  .seg { display:inline-flex; border:1px solid #30363d; border-radius:8px; overflow:hidden; }
  .seg button { background:#0d1117; color:#8b949e; border:none; padding:9px 14px; font-size:13px; cursor:pointer; }
  .seg button.on { background:#1f6feb; color:#fff; }
</style>
</head>
<body>
<div class="wrap">
  <h1><span class="dot off" id="dot"></span> TTS Studio <span class="pill" id="ver">Pro</span></h1>
  <p class="sub">Realistic local voiceovers for YouTube. <span id="modelInfo"></span></p>

  <div class="card">
    <div class="row" style="margin-top:0">
      <div class="field">
        <label>Engine</label>
        <div class="seg" id="engineSeg"></div>
      </div>
      <div class="field" id="voiceField">
        <label>Voice</label>
        <select id="voice"></select>
      </div>
      <div class="field" id="cloneField" style="display:none">
        <label>Voice</label>
        <select id="ref"></select>
      </div>
      <div class="field" id="cloneAddField" style="display:none">
        <label>&nbsp;</label>
        <button class="mini" id="addRef">+ Add voice</button>
      </div>
      <label class="toggle" id="refDenoiseField" style="display:none"><input type="checkbox" id="refDenoise"> Denoise new voice clips</label>
      <div class="field">
        <label>Delivery style</label>
        <select id="style">
          <option value="clean">Clean / Consistent (VO)</option>
          <option value="natural" selected>Natural</option>
          <option value="expressive">Expressive</option>
          <option value="custom">Custom...</option>
        </select>
      </div>
    </div>

    <textarea id="text" placeholder="Type or paste your script..." style="margin-top:16px"></textarea>

    <div class="row">
      <div class="field">
        <label>Takes</label>
        <input id="takes" type="number" min="1" max="8" step="1" value="1" style="min-width:80px">
      </div>
      <label class="toggle"><input type="checkbox" id="script"> Long script (chunk &amp; stitch)</label>
      <div class="field" id="pauseField" style="display:none">
        <label>Pause (ms)</label>
        <input id="pause" type="number" min="0" max="3000" step="50" value="350" style="min-width:90px">
      </div>
      <div class="spacer"></div>
      <button class="primary" id="gen">Generate</button>
    </div>

    <details id="advTune">
      <summary>Voice tuning</summary>
      <div class="grid">
        <div class="slider"><div class="lab"><label>Temperature</label><b id="tempV">0.70</b></div>
          <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value="0.7">
          <div class="hint">Lower = stable, consistent reads. Higher = expressive but variable.</div></div>
        <div class="slider"><div class="lab"><label>Speed</label><b id="speedV">1.00x</b></div>
          <input id="speed" type="range" min="0.5" max="2" step="0.05" value="1.0">
          <div class="hint">Tempo only, pitch preserved. ~0.95 reads relaxed.</div></div>
        <div class="slider"><div class="lab"><label>Top-p</label><b id="toppV">0.95</b></div>
          <input id="topp" type="range" min="0.1" max="1" step="0.01" value="0.95">
          <div class="hint">Nucleus sampling. ~0.9 tightens delivery.</div></div>
        <div class="slider"><div class="lab"><label>Top-k</label><b id="topkV">50</b></div>
          <input id="topk" type="range" min="0" max="200" step="1" value="50">
          <div class="hint">Candidate cap. Lower = safer.</div></div>
        <div class="slider"><label>Seed</label>
          <div class="seedrow"><input id="seed" type="number" placeholder="random">
            <button class="mini" id="dice" title="Random">&#127922;</button>
            <button class="mini" id="lockSeed" title="Reuse last seed">&#128274;</button></div>
          <div class="hint">Set a seed for repeatable takes. Generate, then lock the one you like.</div></div>
      </div>
    </details>

    <details id="advMaster" open>
      <summary>Mastering &amp; export (YouTube)</summary>
      <div class="row">
        <div class="field"><label>Loudness</label>
          <select id="loudness">
            <option value="youtube" selected>YouTube (-14 LUFS)</option>
            <option value="podcast">Podcast (-16 LUFS)</option>
            <option value="broadcast">Broadcast (-23 LUFS)</option>
            <option value="off">Off (raw)</option>
          </select></div>
        <div class="field"><label>Sample rate</label>
          <select id="sr">
            <option value="48000" selected>48 kHz (video)</option>
            <option value="44100">44.1 kHz</option>
            <option value="24000">24 kHz (native)</option>
          </select></div>
        <div class="field"><label>Format</label><select id="format"></select></div>
        <div class="field" id="brField" style="display:none"><label>MP3 bitrate</label>
          <select id="mp3br"><option>320k</option><option selected>192k</option><option>128k</option></select></div>
        <div class="field"><label>Fade in (ms)</label>
          <input id="fadein" type="number" min="0" max="5000" step="50" value="0" style="min-width:100px"></div>
        <div class="field"><label>Fade out (ms)</label>
          <input id="fadeout" type="number" min="0" max="5000" step="50" value="0" style="min-width:100px"></div>
        <label class="toggle"><input type="checkbox" id="trim" checked> Trim silence</label>
        <label class="toggle"><input type="checkbox" id="stereo"> Stereo</label>
        <label class="toggle" id="denoiseField" style="display:none"><input type="checkbox" id="denoise"> Denoise (DeepFilterNet)</label>
      </div>
    </details>

    <div class="status" id="status"></div>
    <div id="results"></div>
  </div>

  <div class="card">
    <details id="cleanPanel">
      <summary>🔊 Audio cleanup &mdash; denoise &amp; remove &ldquo;um / uh&rdquo; fillers (audio files)</summary>
      <div style="margin-top:12px">
        <div class="hint">For audio-only files (voiceovers, podcasts, music beds). Denoise, strip fillers &amp; dead air, preview, then export clean audio. For video editing use <b>Viral Studio</b> above.</div>
        <div class="row" style="margin-top:10px">
          <div class="field" style="flex:1">
            <label>Audio file path</label>
            <input id="clPath" type="text" placeholder="/Users/you/Audio/voiceover.wav  (or .mp3 .m4a .flac ...)" style="width:100%">
            <div class="hint">Tip: in Finder, right-click the file, hold &#8997; Option, then &ldquo;Copy as Pathname&rdquo; and paste here.</div>
          </div>
        </div>
        <div class="row" style="margin-top:6px">
          <label class="toggle"><input type="checkbox" id="clDenoise" checked> Denoise (DeepFilterNet)</label>
          <label class="toggle"><input type="checkbox" id="clFillers" checked> Remove fillers (um/uh/er…)</label>
          <label class="toggle"><input type="checkbox" id="clSilence" checked> Trim long silences</label>
        </div>
        <div class="row" id="clEnhanceRow" style="margin-top:6px;display:none">
          <label class="toggle"><input type="checkbox" id="clEnhance" checked> Enhance light &amp; colour (Core Image, video only)</label>
          <label class="toggle" id="clEnhanceFaceWrap" style="display:none"><input type="checkbox" id="clEnhanceFace" checked> Face-aware</label>
          <div class="field" id="clEnhanceLevelWrap" style="display:none"><label>Strength</label>
            <input id="clEnhanceLevel" type="number" min="0" max="100" step="5" value="100" style="min-width:80px"> <span class="hint">%</span></div>
        </div>
        <div class="row" id="clAspectRow" style="margin-top:6px">
          <div class="field"><label>Aspect ratio (video)</label>
            <select id="clAspect">
              <option value="original" selected>Original</option>
              <option value="9:16">9:16 — TikTok / Reels / Shorts</option>
              <option value="16:9">16:9 — YouTube / landscape</option>
              <option value="1:1">1:1 — Instagram square</option>
              <option value="4:5">4:5 — Instagram portrait</option>
            </select></div>
          <div class="field" id="clAspectFitWrap" style="display:none"><label>Reframe</label>
            <select id="clAspectFit">
              <option value="fill" selected>Fill (crop to fit)</option>
              <option value="pad_blur">Fit (blurred background)</option>
              <option value="pad_black">Fit (black bars)</option>
            </select></div>
        </div>
        <div class="row" id="clViralRow" style="margin-top:6px">
          <label class="toggle"><input type="checkbox" id="clDynamic" checked> Dynamic punch-in (smooth eased zoom on cuts, video only)</label>
          <label class="toggle" id="clCaptionsWrap" style="display:none"><input type="checkbox" id="clCaptions" checked> Smart captions (LLM, video only)</label>
          <label class="toggle" id="clCardsWrap" style="display:none"><input type="checkbox" id="clCards" checked> Hook cards (LLM, video only)</label>
          <div class="field" id="clLenWrap" style="display:none"><label>Viral length target</label>
            <select id="clViralLength">
              <option value="auto" selected>Auto (from source)</option>
              <option value="original">Use original video length</option>
              <option value="15_30">15-30s (aggressive short-form)</option>
              <option value="30_45">30-45s (tight)</option>
              <option value="45_60">45-60s (balanced)</option>
              <option value="60_90">60-90s (story-first)</option>
              <option value="90_120">90-120s (deeper)</option>
            </select></div>
          <div class="field" id="clTopicWrap" style="display:none;flex:1"><label>Topic / voice hint (optional)</label>
            <input id="clTopic" type="text" placeholder="e.g. punchy founder talking about AI startups" style="width:100%"></div>
        </div>
        <div class="row" id="clMemRow" style="margin-top:6px;display:none">
          <label class="toggle"><input type="checkbox" id="clIndex" checked> Save transcript to searchable library (sqlite-vec RAG)</label>
          <div class="field" style="flex:1"><label>Search your transcript library</label>
            <div style="display:flex;gap:6px">
              <input id="clMemQ" type="text" placeholder="e.g. consistency beats intensity" style="flex:1" onkeydown="if(event.key==='Enter')clMemSearch()">
              <button class="ghost" type="button" onclick="clMemSearch()">Search</button>
            </div>
          </div>
        </div>
        <div id="clMemResults" style="margin-top:6px"></div>
        <details style="margin-top:8px"><summary>Advanced</summary>
          <div class="row" style="margin-top:8px">
            <div class="field" style="flex:1"><label>Filler words (comma/space separated)</label>
              <input id="clFillerList" type="text" style="width:100%"></div>
          </div>
          <div class="row" style="margin-top:6px">
            <div class="field"><label>Cut padding (ms)</label>
              <input id="clPad" type="number" min="0" max="500" step="10" value="60" style="min-width:90px"></div>
            <div class="field"><label>Min silence (ms)</label>
              <input id="clMaxSil" type="number" min="150" max="10000" step="50" value="700" style="min-width:100px"></div>
            <div class="field"><label>Keep silence (ms)</label>
              <input id="clKeepSil" type="number" min="0" max="2000" step="50" value="250" style="min-width:100px"></div>
            <div class="field"><label>Silence floor (dB)</label>
              <input id="clSilDb" type="number" min="-90" max="-10" step="1" value="-32" style="min-width:90px"></div>
            <div class="field"><label>Loudness</label>
              <select id="clLoud">
                <option value="youtube" selected>YouTube (-14)</option>
                <option value="podcast">Podcast (-16)</option>
                <option value="broadcast">Broadcast (-23)</option>
                <option value="off">Off (raw)</option>
              </select></div>
            <div class="field" id="clFmtField"><label>Audio format</label>
              <select id="clFmt"><option value="wav" selected>WAV</option><option value="mp3">MP3</option><option value="flac">FLAC</option></select></div>
          </div>
        </details>
        <div class="row" style="margin-top:10px">
          <div class="spacer"></div>
          <button id="clAnalyze">Analyze (preview)</button>
          <button class="primary" id="clRun">Clean up</button>
        </div>
        <div class="status" id="clStatus" style="margin-top:10px"></div>
        <div id="clReview" style="margin-top:10px"></div>
        <div id="clResult" style="margin-top:10px"></div>
      </div>
    </details>
  </div>

  <div class="card">
    <details id="dirPanel" open>
      <summary>🎬 Viral Studio &mdash; AI-directed viral edit (3 stages, collaborative)</summary>
      <div style="margin-top:12px">
        <div class="hint">Turn a plain video into a viral clip in three cached stages. <b>1)</b> Lock the look (denoise + light/colour). <b>2)</b> Let the AI direct the edit &mdash; refine it together. <b>3)</b> Pick your editing options and render. Each stage is cached, so iterating is fast.</div>
        <div class="row" style="margin-top:10px;align-items:center;gap:8px">
          <div class="field" style="flex:1"><label>🧠 Director AI model <span class="hint">switch to compare how different models edit</span></label>
            <select id="dLlm" style="width:100%"></select></div>
          <div id="dLlmMsg" class="hint" style="min-width:120px"></div>
        </div>

        <!-- STAGE 1 — FOUNDATION -->
        <div class="card" style="margin:12px 0 0;background:#161a22;border-left:3px solid #4ea1ff">
          <div style="font-weight:600;margin-bottom:8px">① Foundation &mdash; clean the source <span class="hint">denoise + light/colour, built once &amp; reused</span></div>
          <div class="field" style="flex:1">
            <label>Video path</label>
            <input id="dPath" type="text" placeholder="/Users/you/Movies/clip.mov" style="width:100%">
            <div class="hint">Finder &rarr; right-click &rarr; hold &#8997; Option &rarr; &ldquo;Copy as Pathname&rdquo;.</div>
          </div>
          <div class="row" style="margin-top:8px;flex-wrap:wrap;align-items:center">
            <label class="toggle"><input type="checkbox" id="fDenoise" checked> Denoise audio</label>
            <label class="toggle"><input type="checkbox" id="fEnhance" checked> Enhance light &amp; colour</label>
            <label class="toggle"><input type="checkbox" id="fFace"> Face-aware</label>
            <div class="field"><label>Strength</label>
              <input id="fLevel" type="number" min="0" max="100" step="5" value="100" style="min-width:80px"> <span class="hint">%</span></div>
            <div class="spacer"></div>
            <button class="primary" id="dPrepBtn">Prep foundation &rarr;</button>
          </div>
          <div id="fResult" style="margin-top:10px"></div>
        </div>

        <!-- STAGE 2 — DIRECTOR'S PLAN -->
        <div class="card" id="dStage2" style="margin:12px 0 0;background:#161a22;border-left:3px solid #f5c518;display:none">
          <div style="font-weight:600;margin-bottom:8px">② Director's plan &mdash; the AI decides the edit</div>
          <div class="row">
            <div class="field"><label>Platform</label>
              <select id="dPlatform">
                <option value="youtube">YouTube</option>
                <option value="tiktok">TikTok</option>
                <option value="reels">Instagram Reels</option>
              </select></div>
            <div class="field"><label>Length</label><select id="dLength"></select></div>
            <div class="field"><label>Aspect</label><select id="dAspect"></select></div>
            <div class="field" style="flex:1"><label>Topic / voice hint (optional)</label>
              <input id="dTopic" type="text" placeholder="e.g. founder talking about AI startups" style="width:100%"></div>
          </div>
          <div class="row" style="margin-top:10px">
            <div class="spacer"></div>
            <button class="primary" id="dPlanBtn">Direct the edit &rarr;</button>
          </div>
          <div id="dPlanWrap" style="margin-top:12px;display:none">
            <div class="card" style="margin:0;background:#11151c">
              <div style="font-weight:600;margin-bottom:8px">The director's plan <span class="hint" id="dDur"></span></div>
              <div id="dNotesBox" class="hint" style="white-space:pre-wrap;border-left:3px solid #f5c518;padding-left:8px;margin-bottom:10px"></div>
              <div id="dHl" class="hint" style="margin-bottom:10px"></div>
              <div class="row">
                <div class="field"><label>Aspect</label><select id="pAspect"></select></div>
                <div class="field"><label>Reframe</label><select id="pFit">
                  <option value="fill">Fill (crop)</option><option value="pad_blur">Fit (blur bg)</option><option value="pad_black">Fit (black bars)</option></select></div>
                <div class="field"><label>Length</label><select id="pLength"></select></div>
                <div class="field"><label>Loudness</label><select id="pLoud">
                  <option value="youtube">YouTube (-14)</option><option value="podcast">Podcast (-16)</option><option value="broadcast">Broadcast (-23)</option><option value="off">Off</option></select></div>
                <div class="field"><label>Style</label><select id="pStyle"></select></div>
              </div>
              <details id="dCapDetails" style="margin-top:10px"><summary>Captions (<span id="dCapCount">0</span>) &mdash; edit any line</summary>
                <div id="dCaptions" style="margin-top:8px;max-height:260px;overflow:auto"></div></details>
              <details id="dCardDetails" style="margin-top:8px"><summary>Hook cards (<span id="dCardCount">0</span>)</summary>
                <div id="dCards" style="margin-top:8px"></div>
                <button class="ghost" type="button" id="dAddCard" style="margin-top:6px">+ Add card</button></details>
              <div class="field" style="margin-top:12px"><label>Tell the director what to change (freeform)</label>
                <textarea id="dFeedback" rows="2" placeholder="e.g. punchier hook, keep it under 30s, warmer colours, drop the cards" style="width:100%"></textarea></div>
              <div class="row" style="margin-top:8px">
                <label class="toggle"><input type="checkbox" id="dRegenCaps"> Rewrite captions too</label>
                <div class="spacer"></div>
                <button class="ghost" id="dReviseBtn">Revise with AI &#8635;</button>
              </div>
            </div>
          </div>
        </div>

        <!-- STAGE 3 — RENDER -->
        <div class="card" id="dStage3" style="margin:12px 0 0;background:#161a22;border-left:3px solid #2ecc71;display:none">
          <div style="font-weight:600;margin-bottom:8px">③ Render &mdash; pick your editing options</div>
          <div class="hint">Tick the touches you want. Length, aspect &amp; style come from the plan above. Re-render anytime &mdash; unchanged stages are reused, so it's fast.</div>
          <div class="row" style="margin-top:8px;flex-wrap:wrap">
            <label class="toggle"><input type="checkbox" id="sFillers"> Cut fillers (um/uh)</label>
            <label class="toggle"><input type="checkbox" id="sSilence"> Trim silences</label>
            <label class="toggle"><input type="checkbox" id="sPunch"> Push-in motion</label>
            <label class="toggle"><input type="checkbox" id="sCaptions" name="overlay"> Smart captions</label>
            <label class="toggle"><input type="checkbox" id="sCards" name="overlay"> Hook cards</label>
          </div>
          <div class="row" style="margin-top:10px">
            <div class="spacer"></div>
            <button class="primary" id="dRenderBtn">Make it happen &#9889;</button>
          </div>
          <div id="dResult" style="margin-top:12px"></div>
        </div>

        <div class="status" id="dStatus" style="margin-top:10px"></div>
      </div>
    </details>
  </div>

  <div class="card">
    <details id="histPanel">
      <summary>History</summary>
      <div id="history" style="margin-top:10px"></div>
    </details>
  </div>
</div>

<input type="file" id="refFile" accept="audio/*" style="display:none">

<script>
const $ = id => document.getElementById(id);
let ready=false, lastSeed=null, ENG=null, ENGINES=[], REFS=[], curEngine='higgs', DENOISE_AVAILABLE=false, ENHANCE_AVAILABLE=false, CAPTIONS_AVAILABLE=false, MEMORY_AVAILABLE=false, MEMORY_URL='', TRANSCRIPT_COLLECTION='video_transcripts';

const STYLES = { clean:{temp:0.50,topp:0.90,topk:40}, natural:{temp:0.70,topp:0.95,topk:50}, expressive:{temp:0.95,topp:0.97,topk:80} };

function setDot(s){ $('dot').className='dot'+(s==='on'?'':s==='load'?' load':' off'); }
function syncLabels(){
  $('tempV').textContent=parseFloat($('temp').value).toFixed(2);
  $('speedV').textContent=parseFloat($('speed').value).toFixed(2)+'x';
  $('toppV').textContent=parseFloat($('topp').value).toFixed(2);
  $('topkV').textContent=$('topk').value;
}
function applyStyle(n){ const s=STYLES[n]; if(!s)return; $('temp').value=s.temp;$('topp').value=s.topp;$('topk').value=s.topk; syncLabels(); }
['temp','topp','topk','speed'].forEach(id=>$(id).addEventListener('input',()=>{ syncLabels(); if(id!=='speed')$('style').value='custom'; }));
$('style').addEventListener('change',()=>{ applyStyle($('style').value); $('advTune').open=true; });
$('dice').onclick=()=>{ $('seed').value=Math.floor(Math.random()*1e9); };
$('lockSeed').onclick=()=>{ if(lastSeed!=null){ $('seed').value=lastSeed; $('advTune').open=true; flash('Locked seed '+lastSeed); } };
$('script').addEventListener('change',()=>{ $('pauseField').style.display=$('script').checked?'':'none'; if($('script').checked)$('takes').value=1; });
$('format').addEventListener('change',()=>{ $('brField').style.display=$('format').value==='mp3'?'':'none'; });

function flash(m,err){ $('status').className='status'+(err?' err':''); $('status').textContent=m; }

async function loadEngines(){
  const d = await (await fetch('/engines')).json();
  ENGINES=d.engines; REFS=d.refs||[];
  DENOISE_AVAILABLE=!!d.denoise_available;
  $('denoiseField').style.display=DENOISE_AVAILABLE?'':'none';
  ENHANCE_AVAILABLE=!!d.enhance_available;
  // Audio cleanup card is audio-only — keep all video controls hidden here.
  $('clEnhanceRow').style.display='none';
  $('clAspectRow').style.display='none';
  $('clViralRow').style.display='none';
  CAPTIONS_AVAILABLE=!!d.captions_available;
  // ---- Edit Director dropdowns ----
  const LEN_LABELS={auto:'Auto (from source)',original:'Original length','15_30':'15-30s (aggressive)','30_45':'30-45s (tight)','45_60':'45-60s (balanced)','60_90':'60-90s (story)','90_120':'90-120s (deeper)'};
  const ASP_LABELS={'9:16':'9:16 — Reels/Shorts','16:9':'16:9 — YouTube','1:1':'1:1 — square','4:5':'4:5 — IG portrait','original':'Original'};
  const lens=(d.viral_lengths||['auto']), asps=(d.aspects||['9:16']);
  const lenOpts=lens.map(v=>`<option value="${v}">${LEN_LABELS[v]||v}</option>`).join('');
  const aspOpts=asps.map(v=>`<option value="${v}">${ASP_LABELS[v]||v}</option>`).join('');
  const styOpts=(d.director_styles||['high-energy']).map(v=>`<option value="${v}">${v}</option>`).join('');
  $('dLength').innerHTML=lenOpts; $('pLength').innerHTML=lenOpts;
  $('dAspect').innerHTML=aspOpts; $('pAspect').innerHTML=aspOpts;
  $('pStyle').innerHTML=styOpts;
  $('dAspect').value='9:16';
  MEMORY_AVAILABLE=!!d.memory_available;
  MEMORY_URL=d.memory_url||'';
  TRANSCRIPT_COLLECTION=d.transcript_collection||'video_transcripts';
  $('clMemRow').style.display=MEMORY_AVAILABLE?'':'none';
  $('format').innerHTML=d.formats.map(f=>`<option value="${f}">${f.toUpperCase()}</option>`).join('');
  const seg=$('engineSeg'); seg.innerHTML='';
  ENGINES.forEach(e=>{ const b=document.createElement('button'); b.textContent=e.label.split(' ')[0];
    b.title=e.label; b.dataset.id=e.id; b.onclick=()=>selectEngine(e.id); seg.appendChild(b); });
  curEngine=d.default_engine||'higgs'; selectEngine(curEngine);
}
function selectEngine(id){
  curEngine=id; ENG=ENGINES.find(e=>e.id===id);
  [...$('engineSeg').children].forEach(b=>b.classList.toggle('on',b.dataset.id===id));
  if(ENG.presets){
    $('voiceField').style.display=''; $('cloneField').style.display='none'; $('cloneAddField').style.display='none';
    $('refDenoiseField').style.display='none';
    $('voice').innerHTML=ENG.voices.map(v=>`<option value="${v}">${v.replace('_',' ')}</option>`).join('');
    $('voice').value='neutral_female';
  } else {
    $('voiceField').style.display='none'; $('cloneField').style.display=''; $('cloneAddField').style.display='';
    $('refDenoiseField').style.display=DENOISE_AVAILABLE?'':'none';
    renderRefs();
  }
}
function renderRefs(){
  const esc=s=>String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  const builtin=REFS.filter(r=>r.builtin), mine=REFS.filter(r=>!r.builtin);
  const opt=r=>`<option value="${r.id}">${esc(r.name)} (${Math.round(r.seconds)}s)</option>`;
  let html='';
  if(builtin.length) html+=`<optgroup label="Built-in voices">${builtin.map(opt).join('')}</optgroup>`;
  if(mine.length)    html+=`<optgroup label="My cloned voices">${mine.map(opt).join('')}</optgroup>`;
  html+=`<optgroup label="Other"><option value="">Smart voice (no clone)</option></optgroup>`;
  $('ref').innerHTML=html;
  const first=(builtin[0]||mine[0]);
  if(first) $('ref').value=first.id;
}
async function refreshRefs(){ const d=await(await fetch('/refs')).json(); REFS=d.refs||[]; if(!ENG.presets)renderRefs(); }

$('addRef').onclick=()=>$('refFile').click();
$('refFile').onchange=async()=>{
  const f=$('refFile').files[0]; if(!f)return;
  const name=prompt('Name this voice:', f.name.replace(/\.[^.]+$/,'')); if(name===null)return;
  flash('Uploading & analyzing reference (first time loads Whisper)...');
  const b64=await new Promise(res=>{ const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(f); });
  try{
    const d=await(await fetch('/refs/upload',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,data:b64,denoise:$('refDenoise').checked})})).json();
    if(d.error)throw new Error(d.error);
    await refreshRefs(); $('ref').value=d.ref.id; flash('Added voice "'+d.ref.name+'".');
  }catch(e){ flash('Upload failed: '+e.message,true); }
  $('refFile').value='';
};

async function poll(){
  try{ const d=await(await fetch('/health')).json();
    ready=d.ready; setDot(ready?'on':'load');
    $('modelInfo').textContent=d.loaded&&d.loaded.length?('Loaded: '+d.loaded.join(', ')):'warming up...';
    if(d.errors&&Object.keys(d.errors).length) flash('Engine error: '+JSON.stringify(d.errors),true);
  }catch(e){ setDot('off'); ready=false; }
  $('gen').disabled=!ready;
}

$('gen').onclick=async()=>{
  const text=$('text').value.trim();
  if(!text){ flash('Enter some text first.',true); return; }
  $('gen').disabled=true; flash('Generating'+($('script').checked?' script (chunking)...':'...'));
  $('results').innerHTML='';
  const seedV=$('seed').value.trim();
  const body={
    text, engine:curEngine, format:$('format').value,
    voice: ENG.presets?$('voice').value:undefined,
    ref_id: ENG.clone?($('ref').value||undefined):undefined,
    temperature:parseFloat($('temp').value), top_p:parseFloat($('topp').value),
    top_k:parseInt($('topk').value,10), speed:parseFloat($('speed').value),
    takes:parseInt($('takes').value,10)||1,
    seed: seedV===''?null:parseInt(seedV,10),
    script:$('script').checked, pause_ms:parseInt($('pause').value,10)||350,
    loudness:$('loudness').value, sample_rate:parseInt($('sr').value,10),
    stereo:$('stereo').checked, trim:$('trim').checked,
    denoise:$('denoise').checked,
    fade_in:parseInt($('fadein').value,10)||0, fade_out:parseInt($('fadeout').value,10)||0,
    mp3_bitrate:$('mp3br').value,
  };
  try{
    const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok||d.error)throw new Error(d.error||('HTTP '+r.status));
    lastSeed=d.seed; renderTakes(d); loadHistory();
    flash(`Done in ${d.gen_seconds}s.`+(d.seed!=null?' Lock to reuse seed '+d.seed+'.':''));
  }catch(e){ flash('Error: '+e.message,true); }
  finally{ $('gen').disabled=!ready; }
};

function renderTakes(d){
  const box=$('results'); box.innerHTML='';
  const esc=s=>String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  if(d.curl){
    const c=document.createElement('details'); c.className='cmd curlbox'; c.open=false;
    c.innerHTML=`<summary>&#9654; Reproduce on the command line (curl)</summary>
      <pre id="curlPre">${esc(d.curl)}</pre>
      <button class="copy" id="copyCurl">Copy curl</button>`;
    box.appendChild(c);
    c.querySelector('#copyCurl').onclick=()=>{ navigator.clipboard.writeText(d.curl).then(()=>flash('curl copied')); };
  }
  d.takes.forEach((t,i)=>{
    const url=t.url+'?t='+Date.now();
    const el=document.createElement('div'); el.className='take';
    el.innerHTML=`<span class="tag">${d.takes.length>1?('Take '+(i+1)):''} ${t.seconds||'?'}s${t.seed!=null?' &middot; seed '+t.seed:''}</span>
      <audio controls src="${url}"></audio>
      <a href="${url}" download="${t.filename}">&#8595;</a>`;
    const calls=t.calls||[];
    if(calls.length||t.master_cmd){
      let body='';
      calls.forEach((c,ci)=>{
        const hdr=calls.length>1?`# segment ${ci+1} — ${c.engine} (${c.mode})`:`# ${c.engine} (${c.mode})`;
        body+=`${hdr}\n${c.pretty}\n\n`;
      });
      if(t.master_cmd) body+=`# ffmpeg mastering\n${t.master_cmd}\n`;
      const det=document.createElement('details'); det.className='cmd';
      det.innerHTML=`<summary>&#8984; Show model command</summary><pre>${esc(body.trim())}</pre>`;
      el.appendChild(det);
    }
    box.appendChild(el);
  });
  const a=box.querySelector('audio'); if(a)a.play().catch(()=>{});
}

async function loadHistory(){
  try{ const d=await(await fetch('/history')).json();
    const box=$('history'); box.innerHTML='';
    (d.history||[]).slice(0,60).forEach(h=>{
      const url='/files/'+h.filename;
      const el=document.createElement('div'); el.className='hrow';
      const who=h.engine==='higgs'?(h.ref_name?('clone:'+h.ref_name):'higgs'):(h.voice||'voxtral');
      el.innerHTML=`<button class="fv ${h.favorite?'fav':''}" title="Favorite">${h.favorite?'\u2605':'\u2606'}</button>
        <span class="snip" title="${(h.text||'').replace(/"/g,'&quot;')}">${h.snippet||''}</span>
        <span class="pill">${who} &middot; ${h.seconds||'?'}s</span>
        <audio controls src="${url}"></audio>
        <button class="re" title="Load settings">&#8634;</button>
        <a href="${url}" download="${h.filename}">&#8595;</a>
        <button class="del" title="Delete">&#10005;</button>`;
      el.querySelector('.fv').onclick=async()=>{ await fetch('/history/favorite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:h.id,value:!h.favorite})}); loadHistory(); };
      el.querySelector('.del').onclick=async()=>{ if(confirm('Delete this render?')){ await fetch('/history/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:h.id})}); loadHistory(); } };
      el.querySelector('.re').onclick=()=>{ reloadSettings(h); };
      box.appendChild(el);
    });
    if(!box.children.length) box.innerHTML='<div class="hint">No renders yet.</div>';
  }catch(e){}
}
function reloadSettings(h){
  $('text').value=h.text||''; selectEngine(h.engine);
  if(h.engine==='higgs'&&h.ref_id){ $('ref').value=h.ref_id; } else if(h.voice){ $('voice').value=h.voice; }
  $('temp').value=h.temperature??0.7; $('topp').value=h.top_p??0.95; $('topk').value=h.top_k??50;
  $('speed').value=h.speed??1.0; $('seed').value=h.seed??''; $('loudness').value=h.loudness||'youtube';
  $('sr').value=h.sample_rate||48000; $('style').value='custom'; syncLabels(); $('advTune').open=true;
  window.scrollTo({top:0,behavior:'smooth'}); flash('Loaded settings from history.');
}

$('text').addEventListener('keydown',e=>{ if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){ e.preventDefault(); $('gen').click(); }});

// ---- Media cleanup ----
const DEFAULT_FILLERS='um, umm, uhm, uh, uhh, er, err, erm, ah, ahh, eh, hmm, mhm, mm-hmm';
$('clFillerList').value=DEFAULT_FILLERS;
let clTimer=null, clAnalysis=null;
function clFlash(m,err){ $('clStatus').className='status'+(err?' err':''); $('clStatus').textContent=m; }
function fmtDur(s){ s=Math.max(0,Math.round(s||0)); const m=Math.floor(s/60); return (m?m+'m ':'')+(s%60)+'s'; }
function clEsc(s){ return String(s==null?'':s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }
$('clEnhance').onchange=()=>{ const on=$('clEnhance').checked;
  $('clEnhanceFaceWrap').style.display=on?'':'none'; $('clEnhanceLevelWrap').style.display=on?'':'none'; };
$('clAspect').onchange=()=>{ $('clAspectFitWrap').style.display=$('clAspect').value!=='original'?'':'none'; };
$('clCaptions').onchange=()=>{ const on=$('clCaptions').checked||$('clCards').checked;
  $('clTopicWrap').style.display=on?'':'none'; $('clLenWrap').style.display=on?'':'none'; };
$('clCards').onchange=$('clCaptions').onchange;
async function clMemSearch(){
  const q=$('clMemQ').value.trim();
  const box=$('clMemResults');
  if(!q){ box.innerHTML=''; return; }
  box.innerHTML='<div class="hint">Searching…</div>';
  try{
    const r=await fetch(MEMORY_URL+'/rag/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,k:6,collection:TRANSCRIPT_COLLECTION})});
    const d=await r.json();
    const res=(d.results||[]);
    if(!res.length){ box.innerHTML='<div class="hint">No matches in your transcript library yet.</div>'; return; }
    box.innerHTML='<div class="hint" style="margin-bottom:4px">Top matches:</div>'+res.map(x=>{
      const src=clEsc(x.source||'clip'); const sc=Math.round((x.score||0)*100);
      const snip=clEsc((x.text||'').replace(/\n+/g,' ').slice(0,220));
      return `<div style="border:1px solid var(--bd,#333);border-radius:8px;padding:8px;margin-bottom:6px"><b>${src}</b> <span class="tag">${sc}%</span><div class="hint" style="margin-top:4px">${snip}…</div></div>`;
    }).join('');
  }catch(e){ box.innerHTML='<div class="hint err">Search failed: '+clEsc(e.message)+'</div>'; }
}
function clBusy(b){ $('clRun').disabled=b; $('clAnalyze').disabled=b; }
function clBaseBody(){
  return {
    path:$('clPath').value.trim(),
    denoise:$('clDenoise').checked,
    remove_fillers:$('clFillers').checked,
    remove_silences:$('clSilence').checked,
    enhance_video:ENHANCE_AVAILABLE&&$('clEnhance').checked,
    enhance_face:$('clEnhanceFace').checked,
    enhance_level:(parseInt($('clEnhanceLevel').value,10)||100)/100,
    aspect:$('clAspect').value,
    aspect_fit:$('clAspectFit').value,
    dynamic_edit:$('clDynamic').checked,
    captions:$('clCaptions').checked,
    cards:$('clCards').checked,
    caption_topic:$('clTopic').value.trim(),
    viral_length:$('clViralLength').value,
    index_transcript:$('clIndex').checked,
    fillers:$('clFillerList').value,
    pad_ms:parseInt($('clPad').value,10)||60,
    max_silence_ms:parseInt($('clMaxSil').value,10)||700,
    keep_silence_ms:parseInt($('clKeepSil').value,10)||250,
    silence_db:parseFloat($('clSilDb').value)||-32,
    loudness:$('clLoud').value, format:$('clFmt').value,
  };
}

$('clAnalyze').onclick=async()=>{
  const body=clBaseBody();
  if(!body.path){ clFlash('Paste a file path first.',true); return; }
  if(!body.remove_fillers&&!body.remove_silences){ clFlash('Enable fillers or silences to preview.',true); return; }
  body.mode='analyze';
  clBusy(true); $('clResult').innerHTML=''; $('clReview').innerHTML=''; clFlash('Analyzing…');
  try{
    const r=await fetch('/cleanup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    pollCleanup(d.job_id);
  }catch(e){ clFlash('Error: '+e.message,true); clBusy(false); }
};

$('clRun').onclick=async()=>{
  const body=clBaseBody();
  if(!body.path){ clFlash('Paste a file path first.',true); return; }
  if(!body.denoise&&!body.remove_fillers&&!body.remove_silences&&!body.enhance_video&&body.aspect==='original'&&!body.dynamic_edit&&!body.captions&&!body.cards){ clFlash('Enable at least one cleanup step.',true); return; }
  body.mode='render';
  clBusy(true); $('clResult').innerHTML=''; $('clReview').innerHTML=''; clFlash('Starting…');
  try{
    const r=await fetch('/cleanup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    pollCleanup(d.job_id);
  }catch(e){ clFlash('Error: '+e.message,true); clBusy(false); }
};

function pollCleanup(jid){
  if(clTimer) clearInterval(clTimer);
  clTimer=setInterval(async()=>{
    try{
      const d=await(await fetch('/cleanup/status?id='+encodeURIComponent(jid))).json();
      if(d.error&&d.status!=='error'){ /* transient */ }
      if(d.status==='error'){ clearInterval(clTimer); clBusy(false); clFlash('Failed: '+(d.error||'unknown'),true); return; }
      if(d.status==='done'){ clearInterval(clTimer); clBusy(false);
        if(d.result&&d.result.kind==='analysis'){ clFlash('Preview ready — choose what to remove.'); renderReview(d.result); }
        else { clFlash('Done.'); renderCleanResult(d.result); }
        return; }
      clFlash((d.stage||'working')+'… '+(d.progress||0)+'%');
    }catch(e){}
  },1200);
}

function clProjected(){
  if(!clAnalysis) return;
  let cut=0;
  document.querySelectorAll('.clHit:checked').forEach(c=>{ cut+=parseFloat(c.dataset.len)||0; });
  if($('clSilApply')&&$('clSilApply').checked) cut+=clAnalysis._silTotal||0;
  const nd=Math.max(0,(clAnalysis.duration||0)-cut);
  $('clProj').textContent=fmtDur(clAnalysis.duration)+' → '+fmtDur(nd)+'  (cut '+fmtDur(cut)+')';
}

function renderReview(res){
  clAnalysis=res;
  const sil=res.silence_cuts||[];
  res._silTotal=sil.reduce((a,c)=>a+(c[1]-c[0]),0);
  const hits=res.filler_hits||[];
  let rows=hits.map(h=>`<label class="toggle" style="display:block;margin:2px 0">
      <input type="checkbox" class="clHit" checked data-a="${h.cut[0]}" data-b="${h.cut[1]}" data-len="${(h.cut[1]-h.cut[0]).toFixed(3)}">
      <code>${clEsc(h.word)}</code> <span class="hint">@ ${h.start}s</span></label>`).join('');
  if(!hits.length) rows='<div class="hint">No filler words detected.</div>';
  let silBlock='';
  if(sil.length){
    silBlock=`<label class="toggle" style="display:block;margin-top:8px">
      <input type="checkbox" id="clSilApply" ${$('clSilence').checked?'checked':''}> Trim ${sil.length} long silence${sil.length>1?'s':''} (${fmtDur(res._silTotal)})</label>`;
  }
  let tx='';
  if(res.transcript){ tx='<details style="margin-top:8px"><summary>Transcript</summary><pre style="white-space:pre-wrap;margin-top:6px">'+clEsc(res.transcript)+'</pre></details>'; }
  $('clReview').innerHTML=`<div class="card" style="margin:0">
    <b>Preview</b> <span class="hint">— ${res.has_video?'video':'audio'}, ${fmtDur(res.duration)}</span>
    <div style="margin-top:6px;max-height:220px;overflow:auto">${rows}</div>
    ${silBlock}
    <div class="status" id="clProj" style="margin-top:8px"></div>
    <div class="row" style="margin-top:8px"><div class="spacer"></div>
      <button class="mini" id="clHitAll">All</button><button class="mini" id="clHitNone">None</button>
      <button class="primary" id="clApply">Apply &amp; export</button></div>
    ${tx}</div>`;
  $('clHitAll').onclick=()=>{ document.querySelectorAll('.clHit').forEach(c=>c.checked=true); clProjected(); };
  $('clHitNone').onclick=()=>{ document.querySelectorAll('.clHit').forEach(c=>c.checked=false); clProjected(); };
  document.querySelectorAll('.clHit').forEach(c=>c.onchange=clProjected);
  if($('clSilApply')) $('clSilApply').onchange=clProjected;
  $('clApply').onclick=applyReview;
  clProjected();
}

async function applyReview(){
  if(!clAnalysis) return;
  const cuts=[]; let nf=0;
  document.querySelectorAll('.clHit:checked').forEach(c=>{ cuts.push([parseFloat(c.dataset.a),parseFloat(c.dataset.b)]); nf++; });
  let ns=0;
  if($('clSilApply')&&$('clSilApply').checked){ (clAnalysis.silence_cuts||[]).forEach(s=>{ cuts.push(s); ns++; }); }
  const body=clBaseBody();
  body.mode='render';
  body.remove_fillers=false; body.remove_silences=false;
  body.cuts=cuts; body.cuts_fillers=nf; body.cuts_silences=ns;
  clBusy(true); $('clResult').innerHTML=''; clFlash('Exporting…');
  try{
    const r=await fetch('/cleanup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    pollCleanup(d.job_id);
  }catch(e){ clFlash('Error: '+e.message,true); clBusy(false); }
}

function renderCleanResult(res){
  if(!res){ return; }
  $('clReview').innerHTML='';
  const esc=clEsc;
  const isVid=res.kind==='video';
  const media=isVid
    ? `<video src="${res.url}" controls style="width:100%;max-height:340px;border-radius:8px;background:#000"></video>`
    : `<audio src="${res.url}" controls style="width:100%"></audio>`;
  let chips=`<span class="tag">${fmtDur(res.orig_seconds)} &rarr; ${fmtDur(res.new_seconds)}</span>`;
  if(res.saved_seconds>0) chips+=` <span class="tag">saved ${fmtDur(res.saved_seconds)}</span>`;
  if(res.denoised) chips+=' <span class="tag">denoised</span>';
  if(res.enhanced) chips+=' <span class="tag">enhanced</span>';
  if(res.punch_in) chips+=' <span class="tag">punch-in</span>';
  if(res.captioned) chips+=' <span class="tag">captions</span>';
  if(res.viral_length&&res.viral_length!=='auto'){
    const lbl=res.viral_length==='original'?'original length':res.viral_length.replace('_','-');
    chips+=` <span class="tag">${esc(lbl)} target</span>`;
  }
  if(res.caption_error) chips+=' <span class="tag" style="opacity:.7">captions skipped</span>';
  if(res.indexed) chips+=' <span class="tag">indexed \u2192 library</span>';
  if(res.aspect) chips+=` <span class="tag">${esc(res.aspect)}</span>`;
  chips+=` <span class="tag">${res.fillers_removed} fillers</span>`;
  if(res.silences_removed) chips+=` <span class="tag">${res.silences_removed} silences</span>`;
  let hits='';
  if(res.filler_hits&&res.filler_hits.length){
    hits='<details style="margin-top:8px"><summary>Removed fillers ('+res.filler_hits.length+')</summary><div class="hint" style="margin-top:6px;max-height:160px;overflow:auto">'
      +res.filler_hits.map(h=>`${esc(h.word)} @ ${h.start}s`).join(' &middot; ')+'</div></details>';
  }
  let tx='';
  if(res.transcript){ tx='<details style="margin-top:8px"><summary>Transcript</summary><pre style="white-space:pre-wrap;margin-top:6px">'+esc(res.transcript)+'</pre></details>'; }
  if(res.caption_error){ tx += '<div class="hint" style="margin-top:8px">Caption note: '+esc(res.caption_error)+'</div>'; }
  let saved='';
  if(res.saved_to){ saved=`<div class="hint" style="margin-top:8px">Saved next to original: <code>${esc(res.saved_to)}</code></div>`; }
  else if(res.save_error){ saved=`<div class="status err" style="margin-top:8px">Couldn't save next to original: ${esc(res.save_error)}</div>`; }
  const ext=(res.filename||res.url||'').split('.').pop();
  const stem=(res.source||'cleaned').replace(/\.[^.]+$/,'');
  const dlName=`${stem} cleaned.${ext}`;
  $('clResult').innerHTML=`<div class="card" style="margin:0">${media}<div style="margin-top:8px">${chips}</div>
    <div class="row" style="margin-top:8px"><a class="mini" href="${res.url}" download="${esc(dlName)}">Download cleaned ${isVid?'video':'audio'}</a></div>${saved}${hits}${tx}</div>`;
}

// ================= Viral Studio (3 stages) =================
let DSESS=null, DPLAN=null, DBUSY=false, DPREPPED=false;
function dMsg(t,err){ const s=$('dStatus'); s.className='status'+(err?' err':''); s.textContent=t||''; }
function dBusy(b){ DBUSY=b; ['dPrepBtn','dPlanBtn','dReviseBtn','dRenderBtn'].forEach(id=>{const e=$(id); if(e)e.disabled=b;}); }
async function dPoll(jobId){
  while(true){
    await new Promise(r=>setTimeout(r,1500));
    let j; try{ j=await (await fetch('/director/status?id='+encodeURIComponent(jobId))).json(); }catch(e){ continue; }
    if(j.stage) dMsg((j.stage||'')+(j.progress?(' — '+j.progress+'%'):''));
    if(j.status==='done') return j.result;
    if(j.status==='error') throw new Error(j.error||'failed');
  }
}
// ---- Stage 1: foundation ----
async function dPrep(){
  const path=$('dPath').value.trim();
  if(!path){ dMsg('Paste a video path first.',true); return; }
  dBusy(true); $('fResult').innerHTML=''; $('dResult').innerHTML='';
  $('dStage2').style.display='none'; $('dStage3').style.display='none'; $('dPlanWrap').style.display='none';
  DPREPPED=false; DPLAN=null;
  dMsg('Building the foundation (denoise + light/colour)…');
  try{
    const body={path,platform:$('dPlatform').value,
      denoise:$('fDenoise').checked,enhance:$('fEnhance').checked,
      enhance_face:$('fFace').checked,enhance_level:(parseFloat($('fLevel').value)||100)/100};
    const r=await fetch('/director/prep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json(); if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    DSESS=d.session;
    const res=await dPoll(d.job_id);
    DPREPPED=true; dShowPrep(res);
    $('dStage2').style.display='';
    dMsg('Foundation ready — now let the AI direct the edit (stage 2).');
  }catch(e){ dMsg('Error: '+e.message,true); }
  dBusy(false);
}
function dShowPrep(res){
  let chips='';
  if(res.duration) chips+=`<span class="tag">${fmtDur(res.duration)}</span>`;
  if(res.denoised) chips+=' <span class="tag">denoised</span>';
  if(res.enhanced) chips+=' <span class="tag">enhanced</span>';
  if(!res.has_video) chips+=' <span class="tag" style="opacity:.7">audio only</span>';
  const vid=res.url?`<video src="${res.url}" controls style="width:100%;max-height:300px;border-radius:8px;background:#000"></video>`:'';
  const tr=res.transcript?`<details style="margin-top:8px"><summary>Transcript</summary><div class="hint" style="white-space:pre-wrap;margin-top:6px">${clEsc(res.transcript)}</div></details>`:'';
  $('fResult').innerHTML=`<div class="card" style="margin:0;background:#11151c">${vid}
    <div style="margin-top:8px">${chips}</div>
    <div class="hint" style="margin-top:6px">${res.enhanced?'This cleaned look is cached — every edit below builds on it without re-processing.':'Foundation cached.'}</div>${tr}</div>`;
}
// ---- Stage 2: director's plan ----
async function dPlanRun(){
  if(!DPREPPED||!DSESS){ dMsg('Prep the foundation first (stage 1).',true); return; }
  dBusy(true); $('dResult').innerHTML=''; $('dStage3').style.display='none';
  dMsg('Sending to the director…');
  try{
    const body={path:$('dPath').value.trim(),session:DSESS,platform:$('dPlatform').value,
      length:$('dLength').value,aspect:$('dAspect').value,topic:$('dTopic').value.trim()};
    const r=await fetch('/director/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json(); if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    DSESS=d.session;
    const res=await dPoll(d.job_id);
    dShowPlan(res.plan); dMsg('Plan ready — tweak it, revise with the AI, or jump to stage 3 to render.');
  }catch(e){ dMsg('Error: '+e.message,true); }
  dBusy(false);
}
function dShowPlan(plan){
  DPLAN=plan; const m=plan.meta||{}, s=plan.stages||{};
  $('dPlanWrap').style.display=''; $('dStage3').style.display='';
  if(plan.duration) $('dDur').textContent='— source '+fmtDur(plan.duration);
  $('dNotesBox').textContent=plan.notes||'';
  $('pAspect').value=m.aspect||'9:16'; $('pFit').value=m.aspect_fit||'fill';
  $('pLength').value=m.length||'auto'; $('pLoud').value=m.loudness||'youtube';
  if([...$('pStyle').options].some(o=>o.value===m.style)) $('pStyle').value=m.style;
  else { const o=document.createElement('option'); o.value=m.style; o.textContent=m.style; $('pStyle').appendChild(o); $('pStyle').value=m.style; }
  // Foundation toggles reflect what the AI chose (kept in sync with stage 1).
  if(typeof s.denoise==='boolean') $('fDenoise').checked=s.denoise;
  if(typeof s.enhance==='boolean') $('fEnhance').checked=s.enhance;
  if(typeof s.enhance_face==='boolean') $('fFace').checked=s.enhance_face;
  // Stage 3 editing toggles.
  $('sFillers').checked=!!s.remove_fillers; $('sSilence').checked=!!s.remove_silences;
  $('sPunch').checked=!!s.punch_in;
  // Captions and hook cards are mutually exclusive; captions win if both set.
  $('sCaptions').checked=!!s.captions; $('sCards').checked=!!s.cards && !s.captions;
  const hl=plan.highlights||[];
  if(hl.length){ const secs=hl.reduce((t,p)=>t+(p[1]-p[0]),0);
    $('dHl').innerHTML='✂ <b>Highlight cut active</b> — '+hl.length+' segment'+(hl.length>1?'s':'')+' kept (~'+Math.round(secs)+'s) to hit the length target. Change <b>Length</b> and Revise (or just Make it happen) to re-pick.'; }
  else { $('dHl').textContent=''; }
  dRenderCaps(plan.captions||[]); dRenderCards(plan.cards||[]);
}
function dRenderCaps(caps){
  $('dCapCount').textContent=caps.length;
  $('dCaptions').innerHTML=caps.map(c=>`<div class="dcap" data-i="${c.i}" style="margin-bottom:8px">
    <div class="hint" title="original line">${clEsc(c.orig||'')}</div>
    <div style="display:flex;gap:6px;margin-top:2px">
      <input class="dcap-text" value="${clEsc(c.text||'')}" style="flex:1">
      <input class="dcap-emph" value="${clEsc(c.emphasis||'')}" placeholder="emphasis" style="width:130px">
    </div></div>`).join('')||'<div class="hint">No captions.</div>';
}
function dRenderCards(cards){
  $('dCardCount').textContent=cards.length;
  $('dCards').innerHTML=cards.map(c=>`<div class="dcard" data-i="${c.i}" style="display:flex;gap:6px;margin-bottom:6px;align-items:center">
    <input class="dcard-i" type="number" value="${c.i}" style="width:64px" title="line index">
    <input class="dcard-title" value="${clEsc(c.title||'')}" placeholder="title" style="flex:1">
    <input class="dcard-sub" value="${clEsc(c.subtitle||'')}" placeholder="subtitle" style="flex:2">
    <button class="mini dcard-del" type="button">&times;</button></div>`).join('');
  document.querySelectorAll('.dcard-del').forEach(b=>b.onclick=()=>{ b.closest('.dcard').remove(); $('dCardCount').textContent=document.querySelectorAll('.dcard').length; });
}
function dCollect(){
  const caps=[...document.querySelectorAll('.dcap')].map(d=>({i:parseInt(d.dataset.i,10),
    text:d.querySelector('.dcap-text').value, emphasis:d.querySelector('.dcap-emph').value}));
  const cards=[...document.querySelectorAll('.dcard')].map(d=>({i:parseInt(d.querySelector('.dcard-i').value,10)||0,
    title:d.querySelector('.dcard-title').value, subtitle:d.querySelector('.dcard-sub').value}));
  return {version:1,
    meta:{platform:$('dPlatform').value,aspect:$('pAspect').value,aspect_fit:$('pFit').value,
      length:$('pLength').value,loudness:$('pLoud').value,style:$('pStyle').value},
    stages:{denoise:$('fDenoise').checked,remove_fillers:$('sFillers').checked,remove_silences:$('sSilence').checked,
      punch_in:$('sPunch').checked,enhance:$('fEnhance').checked,enhance_face:$('fFace').checked,
      enhance_level:(parseFloat($('fLevel').value)||100)/100,
      captions:$('sCaptions').checked,cards:$('sCards').checked},
    notes:$('dNotesBox').textContent, captions:caps, cards:cards,
    highlights:(DPLAN&&DPLAN.highlights)||[], highlights_for:(DPLAN&&DPLAN.highlights_for)||''};
}
async function dRevise(){
  if(!DSESS){ dMsg('Run the director first.',true); return; }
  dBusy(true); dMsg('Revising with the director…');
  try{
    const body={session:DSESS,plan:dCollect(),feedback:$('dFeedback').value.trim(),regen_captions:$('dRegenCaps').checked};
    const r=await fetch('/director/revise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json(); if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    const res=await dPoll(d.job_id); dShowPlan(res.plan); $('dFeedback').value='';
    dMsg('Updated — review the changes, revise again, or make it happen.');
  }catch(e){ dMsg('Error: '+e.message,true); }
  dBusy(false);
}
// ---- Stage 3: render ----
async function dRender(){
  if(!DSESS){ dMsg('Run the director first.',true); return; }
  dBusy(true); $('dResult').innerHTML=''; dMsg('Rendering…');
  try{
    const body={session:DSESS,plan:dCollect()};
    const r=await fetch('/director/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json(); if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    const res=await dPoll(d.job_id); dShowResult(res); dMsg('Done.');
  }catch(e){ dMsg('Error: '+e.message,true); }
  dBusy(false);
}
function dShowResult(res){
  if(res.plan) dShowPlan(res.plan);
  let chips=`<span class="tag">${fmtDur(res.orig_seconds)} &rarr; ${fmtDur(res.new_seconds)}</span>`;
  if(res.saved_seconds>0) chips+=` <span class="tag">saved ${fmtDur(res.saved_seconds)}</span>`;
  if(res.denoised) chips+=' <span class="tag">denoised</span>';
  if(res.enhanced) chips+=' <span class="tag">enhanced</span>';
  if(res.punch_in) chips+=' <span class="tag">push-in</span>';
  if(res.captioned) chips+=' <span class="tag">captions</span>';
  if(res.aspect) chips+=` <span class="tag">${clEsc(res.aspect)}</span>`;
  if(res.fillers_removed) chips+=` <span class="tag">${res.fillers_removed} fillers</span>`;
  if(res.silences_removed) chips+=` <span class="tag">${res.silences_removed} silences</span>`;
  if(res.length_target&&res.length_target!=='auto'){
    const lbl=res.length_target==='original'?'original length':res.length_target.replace('_','-')+'s';
    chips+=` <span class="tag">${clEsc(lbl)} target</span>`;
    if(res.highlights) chips+=` <span class="tag">${res.highlights} highlights</span>`;
  }
  if(res.caption_error) chips+=' <span class="tag" style="opacity:.7">captions skipped</span>';
  let saved=''; if(res.saved_to) saved=`<div class="hint" style="margin-top:8px">Saved next to original: <code>${clEsc(res.saved_to)}</code></div>`;
  $('dResult').innerHTML=`<div class="card" style="margin:0;background:#11151c">
    <video src="${res.url}" controls style="width:100%;max-height:380px;border-radius:8px;background:#000"></video>
    <div style="margin-top:8px">${chips}</div>
    <div class="row" style="margin-top:8px"><a class="mini" href="${res.url}" download>Download</a></div>
    ${saved}<div class="hint" style="margin-top:6px">Toggle editing options above and hit <b>Make it happen</b> again — unchanged stages are reused, so it's fast.</div></div>`;
}
$('dPrepBtn').onclick=dPrep;
$('dPlanBtn').onclick=dPlanRun;
$('dReviseBtn').onclick=dRevise;
$('dRenderBtn').onclick=dRender;
// ---- Director AI model switcher ----
async function dLoadLlm(){
  try{
    const d=await (await fetch('/llm')).json();
    const sel=$('dLlm'); sel.innerHTML='';
    (d.choices||[]).forEach(c=>{ const o=document.createElement('option');
      o.value=c.id; o.textContent=c.label; if(c.id===d.current) o.selected=true;
      sel.appendChild(o); });
  }catch(e){}
}
$('dLlm').addEventListener('change',async()=>{
  const model=$('dLlm').value; const m=$('dLlmMsg'); m.textContent='switching…';
  try{
    const r=await fetch('/llm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model})});
    const d=await r.json(); if(!r.ok||d.error) throw new Error(d.error||('HTTP '+r.status));
    const lbl=$('dLlm').options[$('dLlm').selectedIndex].text;
    m.textContent='✓ using '+lbl;
    dMsg('Director AI switched to '+lbl+' — re-run stage 2 to compare.');
  }catch(e){ m.textContent='switch failed'; }
});
// Captions and hook cards are mutually exclusive: either one, or neither.
$('sCaptions').addEventListener('change',()=>{ if($('sCaptions').checked) $('sCards').checked=false; });
$('sCards').addEventListener('change',()=>{ if($('sCards').checked) $('sCaptions').checked=false; });
$('dAddCard').onclick=()=>{ const wrap=$('dCards'); const div=document.createElement('div');
  div.className='dcard'; div.dataset.i=0; div.style.cssText='display:flex;gap:6px;margin-bottom:6px;align-items:center';
  div.innerHTML=`<input class="dcard-i" type="number" value="0" style="width:64px"><input class="dcard-title" placeholder="title" style="flex:1"><input class="dcard-sub" placeholder="subtitle" style="flex:2"><button class="mini dcard-del" type="button">&times;</button>`;
  wrap.appendChild(div); div.querySelector('.dcard-del').onclick=()=>{ div.remove(); $('dCardCount').textContent=document.querySelectorAll('.dcard').length; };
  $('dCardCount').textContent=document.querySelectorAll('.dcard').length; };

syncLabels(); applyStyle('natural'); loadEngines(); loadHistory(); dLoadLlm(); poll(); setInterval(poll,2500);
$('clEnhance').onchange();
$('clAspect').onchange();
$('clCaptions').onchange();
</script>
</body>
</html>
"""


def main():
    threading.Thread(target=_worker, daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"TTS Studio v2 on http://{HOST}:{PORT}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
