#!/usr/bin/env python3
"""Custom LongCat-Video-Avatar driver for the oMLX video service.

Unlike run_inference.py (which hardcodes Meituan's shipped demo), this accepts
arbitrary --image / --audio / --prompt and generates an audio-synced avatar
video. For clips longer than one pass it chunks the audio into fixed-length
segments and stitches them, re-anchoring each segment on the reference image
(identity stays consistent) and carrying the audio forward so lip-sync tracks
the whole voiceover.

Frame rate: the model's audio path is interpolated to `target_fps` (25), so the
output MUST be written at 25 fps for correct lip-sync. Do not "retime" to 15.

Emits progress as JSON lines on stdout: {"stage":..., "progress":0-100}. The
final line is {"ok":true, "out":..., "frames":..., "fps":..., "seconds":...}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

# Reuse the vetted helpers from run_inference.py
from run_inference import (  # noqa: E402
    VARIANT_DIRNAMES,
    build_pipeline,
    preprocess_image,
    preprocess_audio_mel,
    tokenize_prompt,
)

TARGET_FPS = 25          # native audio-sync rate — do not change
VAE_T = 4                # vae temporal scale (num_frames must be 1 + 4k)


def _emit(stage: str, progress: float, **extra):
    rec = {"stage": stage, "progress": round(float(progress), 1)}
    rec.update(extra)
    print(json.dumps(rec), flush=True)


def _round_frames(n: int) -> int:
    """Snap to the VAE-legal 1 + 4k frame grid (>=5)."""
    n = max(5, int(n))
    k = round((n - 1) / VAE_T)
    return int(1 + VAE_T * max(1, k))


def _audio_duration(path: pathlib.Path, sr: int = 16000) -> float:
    import librosa
    return float(librosa.get_duration(path=str(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=pathlib.Path, required=True,
                    help="dir CONTAINING the per-variant subdir")
    ap.add_argument("--variant", default="q8-merged", choices=list(VARIANT_DIRNAMES))
    ap.add_argument("--image", type=pathlib.Path, required=True)
    ap.add_argument("--audio", type=pathlib.Path, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--height", type=int, default=810,   # 9:16 portrait output
                    help="final OUTPUT height (video is upscaled to this)")
    ap.add_argument("--width", type=int, default=480,
                    help="final OUTPUT width (video is upscaled to this)")
    ap.add_argument("--gen-short", type=int, default=256,
                    help="native generation SHORT side (256 is the documented "
                         "stable size; 480p direct produces NaN on q8). The "
                         "clip is generated at this short side preserving the "
                         "output aspect, then Lanczos-upscaled to --width/height.")
    ap.add_argument("--num-frames", type=int, default=0,
                    help="0 = derive from audio length at 25fps")
    ap.add_argument("--seg-frames", type=int, default=93,
                    help="frames per generation pass (memory-bounded)")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    variant_dir = args.weights / VARIANT_DIRNAMES[args.variant]

    # Native generation size: scale the OUTPUT aspect down to the stable short
    # side (256). q8 produces NaN when denoised directly at 480p, so we always
    # generate small and upscale.
    out_w, out_h = int(args.width), int(args.height)
    short = max(64, int(args.gen_short))
    if out_w <= out_h:
        gen_w = short
        gen_h = int(round(short * out_h / out_w))
    else:
        gen_h = short
        gen_w = int(round(short * out_w / out_h))
    # snap to /16 (patch+VAE friendly)
    gen_w = max(64, (gen_w // 16) * 16)
    gen_h = max(64, (gen_h // 16) * 16)
    do_upscale = (out_w, out_h) != (gen_w, gen_h)

    dur = _audio_duration(args.audio)
    dur = min(dur, args.max_seconds)
    if args.num_frames and args.num_frames > 0:
        total_frames = _round_frames(args.num_frames)
    else:
        total_frames = _round_frames(round(dur * TARGET_FPS))
    seg_frames = _round_frames(args.seg_frames)
    num_segments = max(1, -(-total_frames // seg_frames))  # ceil

    _emit("loading model", 2, total_frames=total_frames,
          segments=num_segments, duration=round(dur, 2),
          gen_size=[gen_w, gen_h], out_size=[out_w, out_h])

    pipeline = build_pipeline(args.weights, variant=args.variant)

    # Full audio mel once; the pipeline trims/pads audio to the latent extent
    # per pass, so we pass a per-segment mel slice by time.
    _emit("preprocessing audio", 6)
    full_mel = preprocess_audio_mel(args.audio)   # [1, 128, T_mel]
    T_mel = full_mel.shape[-1]

    _emit("encoding prompt", 8)
    ids, mask = tokenize_prompt(args.prompt, variant_dir)
    text_hidden = pipeline.text_encoder(ids, mask=mask)
    text_embeds = text_hidden[:, None, :, :]
    text_mask = mask[:, None, None, :]
    empty_ids = mx.zeros_like(ids)
    empty_mask = mx.zeros_like(mask)
    uncond_hidden = pipeline.text_encoder(empty_ids, mask=empty_mask)
    uncond_embeds = uncond_hidden[:, None, :, :]
    uncond_mask = empty_mask[:, None, None, :]

    ref_image = preprocess_image(args.image, height=gen_h, width=gen_w)

    t0 = time.time()
    all_frames: list[np.ndarray] = []
    frames_done = 0
    for si in range(num_segments):
        seg_n = min(seg_frames, total_frames - frames_done)
        seg_n = _round_frames(seg_n)
        # Slice the mel window for this segment (proportional by frame position)
        a0 = int(round(frames_done / max(1, total_frames) * T_mel))
        a1 = int(round((frames_done + seg_n) / max(1, total_frames) * T_mel))
        a1 = max(a1, a0 + 1)
        seg_mel = full_mel[:, :, a0:a1]

        base = 10 + int(85 * si / num_segments)
        _emit(f"generating segment {si + 1}/{num_segments}", base,
              seg_frames=seg_n)

        video = pipeline(
            image=ref_image,
            audio_mel=seg_mel,
            text_embeds=text_embeds,
            text_mask=text_mask,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
            height=gen_h,
            width=gen_w,
            num_frames=seg_n,
            seed=args.seed + si,
        )
        mx.eval(video)
        raw = np.asarray(video).transpose(0, 2, 3, 4, 1)[0]
        # Defensive: any NaN/inf -> 0 so a numerical blip can't crash the encode
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        arr = (raw * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        all_frames.append(arr)
        frames_done += seg_n
        # release between segments
        mx.clear_cache()
        if frames_done >= total_frames:
            break

    frames = np.concatenate(all_frames, axis=0)
    elapsed = time.time() - t0

    # Upscale native frames to the requested output size (Lanczos). This is an
    # interim enhancer; a model-based upscaler can replace it later.
    if do_upscale:
        _emit("upscaling", 94, frames=int(frames.shape[0]))
        from PIL import Image
        ups = np.empty((frames.shape[0], out_h, out_w, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            ups[i] = np.asarray(
                Image.fromarray(f).resize((out_w, out_h), Image.LANCZOS))
        frames = ups

    _emit("encoding mp4", 96, frames=int(frames.shape[0]))
    import imageio
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=TARGET_FPS,
                                codec="libx264", quality=8,
                                macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(json.dumps({
        "ok": True,
        "out": str(args.out),
        "frames": int(frames.shape[0]),
        "fps": TARGET_FPS,
        "seconds": round(elapsed, 1),
        "segments": num_segments,
    }), flush=True)


if __name__ == "__main__":
    main()
