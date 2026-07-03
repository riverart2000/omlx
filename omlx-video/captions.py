#!/usr/bin/env python3
"""Caption + audio finishing for avatar/viral videos.

The raw model output is a silent mp4. This module:
  1. transcribes the voiceover with word-level timestamps (mlx_whisper),
  2. builds an .ass subtitle file in one of two viral styles, and
  3. runs a single ffmpeg pass that burns the captions AND muxes the voice
     audio track into the final mp4.

Styles:
  karaoke   big bold centred words that highlight (fill bright) as spoken —
            the classic TikTok / Reels animated-caption look.
  subtitle  clean phrase-by-phrase captions, white with a heavy outline.
  none      no captions — just mux the audio onto the video.

Runs as a CLI so the video service can call it in this venv (which has
mlx_whisper + imageio-ffmpeg):

  python captions.py --audio v.wav --video-in raw.mp4 --video-out final.mp4 \
      --style karaoke --width 480 --height 810
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

WHISPER_REPO = os.environ.get("CAPTION_WHISPER_REPO",
                              "mlx-community/whisper-large-v3-turbo")


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def transcribe_words(audio_path: str) -> list[dict]:
    """Return [{start, end, word}] with word-level timings."""
    import mlx_whisper
    r = mlx_whisper.transcribe(audio_path, path_or_hf_repo=WHISPER_REPO,
                               word_timestamps=True)
    words: list[dict] = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            tok = (w.get("word") or "").strip()
            if not tok:
                continue
            words.append({"start": float(w["start"]),
                          "end": float(w["end"]), "word": tok})
    return words


def _group_lines(words: list[dict], max_words: int, max_chars: int,
                 max_dur: float, max_gap: float) -> list[list[dict]]:
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            dur = w["end"] - cur[0]["start"]
            if (len(cur) >= max_words or chars > max_chars
                    or dur > max_dur or gap > max_gap):
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _cs(t: float) -> str:
    """seconds -> ASS H:MM:SS.cs"""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_header(w: int, h: int, style: str) -> str:
    # Font size + margin scale with height.
    if style == "karaoke":
        fs = max(28, int(h * 0.075))
        outline, shadow, bold = max(3, int(h * 0.006)), 1, -1
        primary = "&H0000FFFF"    # yellow fill (BGR) — "spoken"
        secondary = "&H00FFFFFF"  # white — "upcoming"
    else:  # subtitle
        fs = max(22, int(h * 0.052))
        outline, shadow, bold = max(2, int(h * 0.004)), 1, -1
        primary = "&H00FFFFFF"
        secondary = "&H00FFFFFF"
    mv = int(h * 0.16)  # bottom margin (above platform UI chrome)
    font = os.environ.get("CAPTION_FONT", "Arial")
    outc = "&H00000000"  # black outline
    backc = "&H64000000"  # semi-transparent shadow
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Main,{font},{fs},{primary},{secondary},{outc},{backc},"
        f"{bold},0,0,0,100,100,0,0,1,{outline},{shadow},2,"
        f"{int(w*0.06)},{int(w*0.06)},{mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "Effect, Text\n"
    )


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(words: list[dict], w: int, h: int, style: str) -> str:
    out = [_ass_header(w, h, style)]
    if not words:
        return out[0]
    if style == "karaoke":
        lines = _group_lines(words, max_words=4, max_chars=20,
                             max_dur=2.4, max_gap=0.7)
        for ln in lines:
            start = ln[0]["start"]
            end = ln[-1]["end"] + 0.12
            parts = []
            for i, wd in enumerate(ln):
                nxt = ln[i + 1]["start"] if i + 1 < len(ln) else wd["end"]
                k = max(1, int(round((nxt - wd["start"]) * 100)))
                parts.append(f"{{\\kf{k}}}{_esc(wd['word'].upper())} ")
            text = "".join(parts).rstrip()
            # subtle pop-in
            text = "{\\fad(60,60)}" + text
            out.append(f"Dialogue: 0,{_cs(start)},{_cs(end)},Main,,0,0,0,,{text}\n")
    else:  # subtitle
        lines = _group_lines(words, max_words=7, max_chars=42,
                             max_dur=3.2, max_gap=0.8)
        for ln in lines:
            start = ln[0]["start"]
            end = ln[-1]["end"] + 0.2
            text = _esc(" ".join(x["word"] for x in ln))
            text = "{\\fad(80,80)}" + text
            out.append(f"Dialogue: 0,{_cs(start)},{_cs(end)},Main,,0,0,0,,{text}\n")
    return "".join(out)


def finish(video_in: str, audio: str, video_out: str, w: int, h: int,
           style: str) -> dict:
    ff = _ffmpeg()
    info = {"style": style}
    vf = None
    if style and style != "none":
        words = transcribe_words(audio)
        info["words"] = len(words)
        ass = build_ass(words, w, h, style)
        ass_path = os.path.splitext(video_out)[0] + ".ass"
        with open(ass_path, "w") as f:
            f.write(ass)
        info["ass"] = ass_path
        # Escape path for ffmpeg filter
        esc = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf = f"ass='{esc}'"

    cmd = [ff, "-y", "-i", video_in, "-i", audio]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest", video_out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg finish failed: {p.stderr[-800:]}")
    info["out"] = video_out
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--video-in", required=True)
    ap.add_argument("--video-out", required=True)
    ap.add_argument("--style", default="karaoke",
                    choices=["karaoke", "subtitle", "none"])
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    args = ap.parse_args()
    info = finish(args.video_in, args.audio, args.video_out,
                  args.width, args.height, args.style)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
