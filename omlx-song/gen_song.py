#!/usr/bin/env python3
"""ACE-Step1.5 music generator with automatic best-take selection.

song_server.py invokes this as a subprocess with a JSON spec file path.
Progress and final result are emitted on stdout:

  ::P:: {"progress": 40, "stage": "..."}
  ::R:: {"ok": true, ...}
"""
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np


def emit(kind, payload):
    sys.stdout.write(f"::{kind}:: " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def progress(pct, stage):
    emit("P", {"progress": int(pct), "stage": stage})


def _norm_audio(result_audio):
    import mlx.core as mx

    audio = np.array(result_audio.astype(mx.float32))
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    if audio.ndim == 1:
        audio = audio[:, None]
    return np.clip(audio, -1.0, 1.0)


def _clean_tokens(text):
    text = re.sub(r"\[[^\]]+\]", " ", text or "")
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    toks = [t for t in text.split() if len(t) > 2]
    return toks


def _acoustic_score(audio):
    """Generic quality proxy for selecting stronger takes."""
    if audio.size == 0:
        return 0.0, {"rms": 0.0, "peak": 0.0, "stereo": 0.0, "hf": 0.0}
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(audio)))
    if audio.shape[1] >= 2:
        stereo = float(np.mean(np.abs(audio[:, 0] - audio[:, 1])))
    else:
        stereo = 0.0
    # High-frequency energy ratio (helps reject muddy outputs).
    spec = np.abs(np.fft.rfft(mono))
    n = spec.shape[0]
    if n > 10:
        split = int(n * 0.35)
        hf = float(spec[split:].sum() / (spec.sum() + 1e-9))
    else:
        hf = 0.0
    score = (rms * 2.2) + (stereo * 0.9) + (hf * 1.1)
    return score, {"rms": round(rms, 4), "peak": round(peak, 3),
                   "stereo": round(stereo, 4), "hf": round(hf, 4)}


def _vocal_score(audio_path, lyric_tokens, whisper_repo):
    """Score vocal presence/word adherence via Whisper transcript."""
    try:
        import mlx_whisper
    except Exception:
        return 0.0, "", {"whisper": "unavailable"}
    try:
        r = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=whisper_repo, word_timestamps=False
        )
        txt = (r.get("text") or "").strip()
    except Exception as e:
        return 0.0, "", {"whisper_error": str(e)[:160]}

    toks = _clean_tokens(txt)
    if not toks:
        return -0.6, txt, {"words": 0, "coverage": 0.0, "repeat": 1.0}

    tset = set(toks)
    lset = set(lyric_tokens)
    coverage = len(tset & lset) / max(1, len(lset))
    freq = Counter(toks)
    top_ratio = max(freq.values()) / max(1, len(toks))
    repeat_penalty = max(0.0, top_ratio - 0.34) * 1.4
    word_bonus = min(len(toks), 36) / 36.0
    score = (coverage * 3.2) + (word_bonus * 1.1) - repeat_penalty
    return score, txt, {"words": len(toks), "coverage": round(coverage, 3),
                        "repeat": round(top_ratio, 3)}


def _write_outputs(audio, sr, out_dir, stem, keep_wav, make_mp3, ffmpeg):
    import soundfile as sf

    files = []
    flac_path = os.path.join(out_dir, stem + ".flac")
    sf.write(flac_path, audio, sr, format="FLAC")
    files.append(os.path.basename(flac_path))

    wav_path = None
    if keep_wav or make_mp3:
        wav_path = os.path.join(out_dir, stem + ".wav")
        sf.write(wav_path, audio, sr, subtype="PCM_16")

    if make_mp3 and wav_path:
        progress(92, "encoding mp3")
        mp3_path = os.path.join(out_dir, stem + ".mp3")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "320k", mp3_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            files.append(os.path.basename(mp3_path))
        except Exception as e:
            emit("P", {"progress": 93, "stage": f"mp3 skipped ({e})"})

    if wav_path and not keep_wav:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    elif keep_wav and wav_path:
        files.append(os.path.basename(wav_path))

    return files, os.path.basename(flac_path)


def main():
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)

    model_dir = spec["model_dir"]
    out_dir = spec["out_dir"]
    stem = spec["stem"]
    keep_wav = bool(spec.get("keep_wav", False))
    make_mp3 = bool(spec.get("make_mp3", True))
    ffmpeg = spec.get("ffmpeg", "ffmpeg")
    whisper_repo = spec.get("whisper_repo", "mlx-community/whisper-large-v3-turbo")
    attempts = max(1, min(5, int(spec.get("attempts", 1))))
    max_attempts = max(attempts, min(10, int(spec.get("max_attempts", attempts))))
    min_vocal_words = max(1, min(20, int(spec.get("min_vocal_words", 6))))

    progress(3, "loading model")
    from mlx_audio.tts.models.ace_step import Model

    t_load = time.time()
    model = Model.from_pretrained(model_dir)
    load_s = round(time.time() - t_load, 1)
    progress(15, "model ready")

    base_seed = int(spec["seed"]) if spec.get("seed") is not None else int.from_bytes(os.urandom(3), "big")
    lyrics = spec.get("lyrics", "") or ""
    lyric_tokens = _clean_tokens(lyrics)
    vocal_mode = bool(lyric_tokens)

    gen_kwargs = dict(
        text=spec["text"],
        lyrics=lyrics,
        duration=float(spec.get("duration", 30.0)),
        num_steps=int(spec.get("num_steps", 8)),
        shift=float(spec.get("shift", 3.0)),
        guidance_scale=float(spec.get("guidance_scale", 1.0)),
        guidance_interval=float(spec.get("guidance_interval", 0.5)),
        cfg_type=spec.get("cfg_type", "apg"),
        vocal_language=spec.get("vocal_language", "unknown"),
        use_lm=bool(spec.get("use_lm", True)),
        lm_model_size=spec.get("lm_model_size", "0.6B"),
        verbose=True,
    )

    os.makedirs(out_dir, exist_ok=True)

    best = None
    picks = []
    total_gen_s = 0.0

    total_tries = max_attempts if vocal_mode else attempts
    for i in range(total_tries):
        seed_i = base_seed + (i * 9973)
        gen_kwargs["seed"] = seed_i
        p0 = 20 + int((58 * i) / max(1, total_tries))
        progress(p0, f"attempt {i+1}/{total_tries}: planning (5Hz LM)")
        t_gen = time.time()
        result = None
        for r in model.generate(**gen_kwargs):
            result = r
            break
        gen_s = round(time.time() - t_gen, 1)
        total_gen_s += gen_s
        if result is None:
            continue
        audio = _norm_audio(result.audio)
        sr = int(result.sample_rate)
        duration_s = round(audio.shape[0] / sr, 2) if sr > 0 else 0.0

        aq_score, aq = _acoustic_score(audio)
        vocal_bonus = 0.0
        transcript = ""
        vdbg = {}
        if vocal_mode:
            # score vocals using transcript from a temp wav per attempt
            tmp_wav = os.path.join(out_dir, "_tmp", f"{stem}_take{i+1}.wav")
            os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
            import soundfile as sf
            sf.write(tmp_wav, audio, sr, subtype="PCM_16")
            progress(p0 + 8, f"attempt {i+1}/{total_tries}: checking vocals")
            vocal_bonus, transcript, vdbg = _vocal_score(tmp_wav, lyric_tokens, whisper_repo)
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        score = aq_score + vocal_bonus
        pick = {
            "take": i + 1,
            "seed": seed_i,
            "score": round(score, 3),
            "acoustic": aq,
            "vocal": vdbg,
            "duration": duration_s,
            "gen_s": gen_s,
            "audio": audio,
            "sr": sr,
            "meta": getattr(result, "metadata", None) or {},
            "transcript": transcript[:220],
        }
        picks.append({k: v for k, v in pick.items() if k not in ("audio",)})
        if best is None or pick["score"] > best["score"]:
            best = pick
        if vocal_mode and (i + 1) >= attempts:
            words = int(vdbg.get("words", 0) or 0)
            coverage = float(vdbg.get("coverage", 0.0) or 0.0)
            if words >= min_vocal_words and coverage >= 0.08:
                break

    if best is None:
        emit("R", {"ok": False, "error": "generation produced no audio"})
        return
    if vocal_mode:
        bwords = int((best.get("vocal") or {}).get("words", 0) or 0)
        if bwords < min_vocal_words:
            emit("R", {
                "ok": False,
                "error": ("Could not produce clear sung vocals from this prompt/lyrics "
                          f"after {len(picks)} takes (best detected words={bwords}). "
                          "Try simpler chorus-style lyrics, longer duration (20-30s), "
                          "or switch model."),
                "selection_debug": picks[-5:],
            })
            return

    progress(85, "encoding best take")
    files, flac_name = _write_outputs(
        best["audio"], best["sr"], out_dir, stem, keep_wav, make_mp3, ffmpeg
    )

    peak = float(np.max(np.abs(best["audio"]))) if best["audio"].size else 0.0
    emit("R", {
        "ok": True,
        "files": files,
        "flac": flac_name,
        "sample_rate": best["sr"],
        "duration": best["duration"],
        "peak": round(peak, 3),
        "load_s": load_s,
        "gen_s": round(total_gen_s, 1),
        "metadata": best["meta"] if isinstance(best["meta"], dict) else {},
        "selected_seed": best["seed"],
        "selected_take": best["take"],
        "attempts": len(picks),
        "selection_score": best["score"],
        "selection_debug": picks,
        "transcript_preview": best["transcript"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit("R", {"ok": False, "error": str(e), "trace": traceback.format_exc()[-2000:]})
