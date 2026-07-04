#!/usr/bin/env python3
"""ACE-Step1.5 music generator — one clip per subprocess (clean memory release).

song_server.py invokes this in the service venv, one job at a time. Args come in
as a JSON spec file (argv[1]); progress + the final result are emitted on stdout
as line-prefixed markers the server parses:

    ::P:: {"progress": 40, "stage": "diffusing"}      # progress update
    ::R:: {"ok": true, "files": [...], ...}            # final result (last line)

Running each generation in its own process guarantees the ~10GB of MLX weights are
returned to the OS the moment it exits — important on a 64GB box that also hosts a
35B chat LLM. (The oMLX chat model is unloaded by the server before we start.)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np


def emit(kind, payload):
    sys.stdout.write(f"::{kind}:: " + json.dumps(payload) + "\n")
    sys.stdout.flush()


def progress(pct, stage):
    emit("P", {"progress": int(pct), "stage": stage})


def main():
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)

    model_dir = spec["model_dir"]
    out_dir = spec["out_dir"]
    stem = spec["stem"]                     # output filename stem (no extension)
    keep_wav = bool(spec.get("keep_wav", False))
    make_mp3 = bool(spec.get("make_mp3", True))
    ffmpeg = spec.get("ffmpeg", "ffmpeg")

    progress(3, "loading model")
    import mlx.core as mx
    from mlx_audio.tts.models.ace_step import Model

    t_load = time.time()
    model = Model.from_pretrained(model_dir)
    load_s = round(time.time() - t_load, 1)
    progress(15, "model ready")

    gen_kwargs = dict(
        text=spec["text"],
        lyrics=spec.get("lyrics", "") or "",
        duration=float(spec.get("duration", 30.0)),
        num_steps=int(spec.get("num_steps", 8)),
        seed=int(spec["seed"]) if spec.get("seed") is not None else None,
        shift=float(spec.get("shift", 3.0)),
        guidance_scale=float(spec.get("guidance_scale", 1.0)),
        guidance_interval=float(spec.get("guidance_interval", 0.5)),
        cfg_type=spec.get("cfg_type", "apg"),
        vocal_language=spec.get("vocal_language", "unknown"),
        use_lm=bool(spec.get("use_lm", True)),
        lm_model_size=spec.get("lm_model_size", "0.6B"),
        verbose=True,
    )

    progress(20, "planning (5Hz LM)")
    t_gen = time.time()
    result = None
    for r in model.generate(**gen_kwargs):
        result = r
        break
    gen_s = round(time.time() - t_gen, 1)
    if result is None:
        emit("R", {"ok": False, "error": "generation produced no audio"})
        return

    progress(85, "encoding audio")
    audio = np.array(result.audio.astype(mx.float32))
    sr = int(result.sample_rate)
    # Normalize to [samples, channels] float32.
    if audio.ndim == 3:            # (1, C, N) or (1, N, C)
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[0] < audio.shape[1]:
        audio = audio.T            # (C, N) -> (N, C)
    if audio.ndim == 1:
        audio = audio[:, None]
    peak = float(np.abs(audio).max()) if audio.size else 0.0

    os.makedirs(out_dir, exist_ok=True)
    files = []

    import soundfile as sf
    flac_path = os.path.join(out_dir, stem + ".flac")
    sf.write(flac_path, np.clip(audio, -1.0, 1.0), sr, format="FLAC")
    files.append(os.path.basename(flac_path))

    wav_path = None
    if keep_wav or make_mp3:
        wav_path = os.path.join(out_dir, stem + ".wav")
        sf.write(wav_path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")

    if make_mp3:
        progress(92, "encoding mp3")
        mp3_path = os.path.join(out_dir, stem + ".mp3")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "320k", mp3_path]
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

    duration_s = round(audio.shape[0] / sr, 2)
    meta = getattr(result, "metadata", None) or {}
    emit("R", {
        "ok": True,
        "files": files,
        "flac": os.path.basename(flac_path),
        "sample_rate": sr,
        "duration": duration_s,
        "peak": round(peak, 3),
        "load_s": load_s,
        "gen_s": gen_s,
        "metadata": meta if isinstance(meta, dict) else {},
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit("R", {"ok": False, "error": str(e),
                   "trace": traceback.format_exc()[-1500:]})
