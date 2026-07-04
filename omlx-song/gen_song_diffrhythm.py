#!/usr/bin/env python3
"""DiffRhythm generator subprocess for song_server.py.

This wrapper makes the UI "quality" control meaningful by mapping it to:
1) batch candidates per run, 2) multi-seed reruns, 3) chunked/non-chunked decode,
and then selecting the best output automatically.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime


def emit(kind, payload):
    sys.stdout.write(f"::{kind}:: " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def progress(pct, stage):
    emit("P", {"progress": int(pct), "stage": stage})


def _clean_tokens(text):
    s = re.sub(r"\[[^\]]+\]", " ", text or "")
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return [t for t in s.split() if len(t) > 2]


def _lyrics_lines(raw):
    lines = []
    for ln in (raw or "").splitlines():
        ln = re.sub(r"\[[^\]]+\]", " ", ln).strip()
        ln = re.sub(r"\s+", " ", ln)
        if len(ln) >= 2:
            lines.append(ln)
    return lines


def _mk_lrc(lines, total_seconds, out_path):
    if not lines:
        return None
    span = max(8.0, float(total_seconds) - 6.0)
    step = span / max(1, len(lines))
    cur = 0.0
    out = []
    for ln in lines:
        mm = int(cur // 60)
        ss = cur - mm * 60
        out.append(f"[{mm:02d}:{ss:05.2f}]{ln}")
        cur += step
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return out_path


def _read_wav(path):
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return np.clip(data, -1.0, 1.0), int(sr)


def _acoustic_score(audio):
    import numpy as np

    if audio.size == 0:
        return 0.0, {"rms": 0.0, "stereo": 0.0, "hf": 0.0}
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    stereo = float(np.mean(np.abs(audio[:, 0] - audio[:, 1]))) if audio.shape[1] >= 2 else 0.0
    spec = np.abs(np.fft.rfft(mono))
    if spec.size > 8:
        split = int(spec.size * 0.35)
        hf = float(spec[split:].sum() / (spec.sum() + 1e-9))
    else:
        hf = 0.0
    score = (rms * 2.4) + (stereo * 0.8) + (hf * 1.3)
    return score, {"rms": round(rms, 4), "stereo": round(stereo, 4), "hf": round(hf, 4)}


def _vocal_score(wav_path, lyric_tokens):
    try:
        import mlx_whisper
    except Exception:
        return 0.0, "", {"words": 0, "coverage": 0.0, "repeat": 1.0}
    try:
        r = mlx_whisper.transcribe(
            wav_path, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", word_timestamps=False
        )
        txt = (r.get("text") or "").strip()
    except Exception:
        return 0.0, "", {"words": 0, "coverage": 0.0, "repeat": 1.0}

    toks = _clean_tokens(txt)
    if not toks:
        return -0.8, txt, {"words": 0, "coverage": 0.0, "repeat": 1.0}
    tset = set(toks)
    lset = set(lyric_tokens)
    coverage = len(tset & lset) / max(1, len(lset))
    from collections import Counter

    freq = Counter(toks)
    top_ratio = max(freq.values()) / max(1, len(toks))
    repeat_penalty = max(0.0, top_ratio - 0.34) * 1.6
    word_bonus = min(len(toks), 36) / 36.0
    score = (coverage * 3.4) + (word_bonus * 1.0) - repeat_penalty
    return score, txt, {"words": len(toks), "coverage": round(coverage, 3), "repeat": round(top_ratio, 3)}


def _encode_outputs(src_wav, out_dir, stem, keep_wav, make_mp3, ffmpeg):
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(src_wav, dtype="float32", always_2d=True)
    data = np.clip(data, -1.0, 1.0)

    files = []
    flac_path = os.path.join(out_dir, stem + ".flac")
    sf.write(flac_path, data, sr, format="FLAC")
    files.append(os.path.basename(flac_path))

    wav_path = None
    if keep_wav or make_mp3:
        wav_path = os.path.join(out_dir, stem + ".wav")
        sf.write(wav_path, data, sr, subtype="PCM_16")

    if make_mp3 and wav_path:
        progress(92, "encoding mp3")
        mp3_path = os.path.join(out_dir, stem + ".mp3")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "320k", mp3_path]
        subprocess.run(cmd, check=True)
        files.append(os.path.basename(mp3_path))

    if wav_path and not keep_wav:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    elif keep_wav and wav_path:
        files.append(os.path.basename(wav_path))

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    dur = round(data.shape[0] / float(sr), 2)
    return files, os.path.basename(flac_path), sr, dur, peak


def _quality_profile(duration, quality_value):
    q = max(4, min(60, int(quality_value)))
    if q >= 52:
        profile = "max"
        runs = 3
        batch = 4
        chunked = False
        base_len = 140
    elif q >= 40:
        profile = "high"
        runs = 2
        batch = 3
        chunked = False
        base_len = 120
    elif q >= 24:
        profile = "balanced"
        runs = 1
        batch = 2
        chunked = True
        base_len = 95
    else:
        profile = "fast"
        runs = 1
        batch = 1
        chunked = True
        base_len = 95
    if duration <= 95:
        audio_len = max(95, base_len)
    else:
        audio_len = int(round(duration))
    audio_len = max(95, min(285, audio_len))
    if audio_len > 140:
        chunked = True
    return {
        "profile": profile,
        "runs": runs,
        "batch": batch,
        "chunked": chunked,
        "audio_len": audio_len,
    }


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        spec = json.load(f)

    dr_repo = spec["dr_repo"]
    dr_py = spec["dr_py"]
    espeak_lib = spec.get("espeak_lib") or ""
    out_dir = spec["out_dir"]
    stem = spec["stem"]
    prompt = (spec.get("text") or "").strip()
    duration = max(8.0, float(spec.get("duration", 30.0)))
    quality_value = int(spec.get("num_steps", 28))
    keep_wav = bool(spec.get("keep_wav", False))
    make_mp3 = bool(spec.get("make_mp3", True))
    ffmpeg = spec.get("ffmpeg", "ffmpeg")
    seed = int(spec.get("seed", int.from_bytes(os.urandom(3), "big")))

    if not os.path.isdir(dr_repo):
        raise RuntimeError("DiffRhythm repo not found")
    if not os.path.isfile(dr_py):
        raise RuntimeError("DiffRhythm python runtime not found")

    profile = _quality_profile(duration, quality_value)
    lyrics_raw = spec.get("lyrics", "") or ""
    lyric_tokens = _clean_tokens(lyrics_raw)
    vocal_mode = bool(lyric_tokens)
    lyrics = _lyrics_lines(lyrics_raw) if vocal_mode else []

    os.makedirs(out_dir, exist_ok=True)
    tmp_root = os.path.join(out_dir, "_tmp", "diffrhythm_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(tmp_root, exist_ok=True)

    env = os.environ.copy()
    if espeak_lib:
        env["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib

    runs = profile["runs"]
    candidates = []
    t0 = time.time()

    for i in range(runs):
        run_seed = seed + (i * 7919)
        run_dir = os.path.join(tmp_root, f"run_{i+1}")
        os.makedirs(run_dir, exist_ok=True)
        lrc_path = None
        if lyrics:
            lrc_path = _mk_lrc(lyrics, profile["audio_len"], os.path.join(run_dir, "input.lrc"))

        cmd = [
            dr_py, "infer/infer.py",
            "--ref-prompt", prompt[:400],
            "--audio-length", str(profile["audio_len"]),
            "--output-dir", run_dir,
            "--batch-infer-num", str(profile["batch"]),
            "--pick", "best",
            "--seed", str(run_seed),
        ]
        if profile["chunked"]:
            cmd.append("--chunked")
        if lrc_path:
            cmd += ["--lrc-path", lrc_path]

        p0 = 20 + int((58 * i) / max(1, runs))
        progress(p0, f"DiffRhythm {profile['profile']} run {i+1}/{runs}")

        proc = subprocess.Popen(
            cmd,
            cwd=dr_repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail = []
        for line in proc.stdout:
            s = line.rstrip()
            if not s:
                continue
            tail.append(s)
            if len(tail) > 60:
                tail.pop(0)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("DiffRhythm failed: " + " | ".join(tail[-8:]))

        wav = os.path.join(run_dir, "output.wav")
        if not os.path.isfile(wav):
            raise RuntimeError("DiffRhythm produced no output.wav")

        # Optional trim to requested duration before scoring.
        scored_wav = wav
        if duration < (profile["audio_len"] - 0.5):
            trimmed = os.path.join(run_dir, "trimmed.wav")
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", wav, "-t", f"{duration:.3f}", trimmed],
                check=True,
            )
            scored_wav = trimmed

        audio, sr = _read_wav(scored_wav)
        aq_score, aq = _acoustic_score(audio)
        v_score = 0.0
        transcript = ""
        vdbg = {"words": 0, "coverage": 0.0, "repeat": 1.0}
        if vocal_mode:
            progress(p0 + 8, f"checking vocals run {i+1}/{runs}")
            v_score, transcript, vdbg = _vocal_score(scored_wav, lyric_tokens)
        total = aq_score + (v_score * 2.4)
        candidates.append({
            "run": i + 1,
            "seed": run_seed,
            "wav": scored_wav,
            "score": total,
            "acoustic": aq,
            "vocal": vdbg,
            "transcript": transcript[:220],
            "sr": sr,
        })

    if not candidates:
        raise RuntimeError("DiffRhythm produced no candidates")

    best = max(candidates, key=lambda x: x["score"])
    warning = ""
    if vocal_mode and int(best["vocal"].get("words", 0)) < 3:
        warning = ("Low vocal confidence: generated audio may be mostly instrumental or unclear "
                   f"(detected words={best['vocal'].get('words', 0)}).")

    progress(88, "encoding outputs")
    files, flac_name, sr, dur, peak = _encode_outputs(
        best["wav"], out_dir, stem, keep_wav, make_mp3, ffmpeg
    )
    elapsed = round(time.time() - t0, 1)

    debug = []
    for c in candidates:
        debug.append({
            "run": c["run"],
            "seed": c["seed"],
            "score": round(float(c["score"]), 3),
            "acoustic": c["acoustic"],
            "vocal": c["vocal"],
            "transcript": c["transcript"],
        })

    try:
        shutil.rmtree(tmp_root)
    except OSError:
        pass

    emit("R", {
        "ok": True,
        "engine": "diffrhythm",
        "files": files,
        "flac": flac_name,
        "sample_rate": sr,
        "duration": dur,
        "peak": round(peak, 3),
        "load_s": 0.0,
        "gen_s": elapsed,
        "metadata": {
            "audio_length_model_s": profile["audio_len"],
            "quality_profile": profile["profile"],
            "batch_infer_num": profile["batch"],
            "chunked": profile["chunked"],
        },
        "selected_seed": best["seed"],
        "selected_take": best["run"],
        "attempts": len(candidates),
        "selection_score": round(float(best["score"]), 3),
        "selection_debug": debug,
        "transcript_preview": best["transcript"],
        "warning": warning,
        "num_steps": quality_value,
        "lm_model_size": None,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit("R", {"ok": False, "error": str(e), "trace": traceback.format_exc()[-1800:]})
