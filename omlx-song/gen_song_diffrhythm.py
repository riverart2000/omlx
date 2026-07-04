#!/usr/bin/env python3
"""DiffRhythm generator subprocess for song_server.py."""
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

    peak = float(abs(data).max()) if data.size else 0.0
    dur = round(data.shape[0] / float(sr), 2)
    return files, os.path.basename(flac_path), sr, dur, peak


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
    keep_wav = bool(spec.get("keep_wav", False))
    make_mp3 = bool(spec.get("make_mp3", True))
    ffmpeg = spec.get("ffmpeg", "ffmpeg")
    seed = int(spec.get("seed", int.from_bytes(os.urandom(3), "big")))

    if not os.path.isdir(dr_repo):
        raise RuntimeError("DiffRhythm repo not found")
    if not os.path.isfile(dr_py):
        raise RuntimeError("DiffRhythm python runtime not found")

    os.makedirs(out_dir, exist_ok=True)
    tmp_root = os.path.join(out_dir, "_tmp", "diffrhythm_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(tmp_root, exist_ok=True)
    dr_out = os.path.join(tmp_root, "out")
    os.makedirs(dr_out, exist_ok=True)

    dr_len = 95 if duration <= 95 else min(285, int(round(duration)))
    lyrics = _lyrics_lines(spec.get("lyrics", ""))
    lrc_path = None
    if lyrics:
        lrc_path = _mk_lrc(lyrics, dr_len, os.path.join(tmp_root, "input.lrc"))

    cmd = [
        dr_py, "infer/infer.py",
        "--ref-prompt", prompt[:400],
        "--audio-length", str(dr_len),
        "--output-dir", dr_out,
        "--chunked",
        "--batch-infer-num", "1",
    ]
    if lrc_path:
        cmd += ["--lrc-path", lrc_path]

    env = os.environ.copy()
    if espeak_lib:
        env["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
    env["PYTHONHASHSEED"] = str(seed)

    progress(20, "DiffRhythm generating")
    t0 = time.time()
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

    raw_wav = os.path.join(dr_out, "output.wav")
    if not os.path.isfile(raw_wav):
        raise RuntimeError("DiffRhythm produced no output.wav")

    final_wav = raw_wav
    if duration < (dr_len - 0.5):
        progress(84, "trimming output")
        trimmed = os.path.join(tmp_root, "trimmed.wav")
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", raw_wav, "-t", f"{duration:.3f}", trimmed],
            check=True,
        )
        final_wav = trimmed

    progress(88, "encoding outputs")
    files, flac_name, sr, dur, peak = _encode_outputs(
        final_wav, out_dir, stem, keep_wav, make_mp3, ffmpeg
    )
    elapsed = round(time.time() - t0, 1)

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
        "metadata": {"audio_length_model_s": dr_len},
        "selected_seed": seed,
        "selected_take": 1,
        "attempts": 1,
        "num_steps": None,
        "lm_model_size": None,
        "transcript_preview": "",
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit("R", {"ok": False, "error": str(e), "trace": traceback.format_exc()[-1800:]})
