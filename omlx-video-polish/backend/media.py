"""FFmpeg/ffprobe utilities and media filter construction."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from .config import FFMPEG, FFPROBE
from .models import ExportSettings, VideoSettings


LogFn = Callable[[str], None]


def safe_filename(value: str, fallback: str = "video") -> str:
    value = Path(value).name.strip().replace("\x00", "")
    value = re.sub(r"[^A-Za-z0-9._() -]+", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return (value or fallback)[:180]


def command_exists(path: str) -> bool:
    return Path(path).exists()


def run_command(
    command: list[str],
    *,
    logger: LogFn | None = None,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if logger:
        logger("Running: " + " ".join(_display_arg(part) for part in command))
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        tail = "\n".join((completed.stderr or "").splitlines()[-30:])
        if logger and tail:
            logger(tail)
        raise RuntimeError(f"Media command failed ({completed.returncode}): {tail[-1800:]}")
    return completed


def _display_arg(value: str) -> str:
    if len(value) > 220:
        return value[:217] + "..."
    return value


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_media(path: Path) -> dict:
    result = run_command(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    raw = json.loads(result.stdout or "{}")
    streams = raw.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not video:
        raise ValueError("The selected file does not contain a video stream")

    duration_values = [
        raw.get("format", {}).get("duration"),
        video.get("duration"),
        audio.get("duration") if audio else None,
    ]
    duration = next(
        (float(value) for value in duration_values if value not in (None, "N/A") and float(value) > 0),
        0.0,
    )
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = 0
    for item in video.get("side_data_list") or []:
        if "rotation" in item:
            rotation = int(item.get("rotation") or 0) % 360
    if rotation in (90, 270):
        width, height = height, width
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")) or 30.0
    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "video_codec": video.get("codec_name") or "unknown",
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_audio": bool(audio),
        "audio_channels": int(audio.get("channels") or 0) if audio else 0,
        "rotation": rotation,
    }


def even(value: float) -> int:
    return max(2, int(round(value / 2.0) * 2))


def target_dimensions(meta: dict, resolution: str) -> tuple[int, int]:
    width = int(meta.get("width") or 1920)
    height = int(meta.get("height") or 1080)
    if resolution == "source":
        return even(width), even(height)
    short_or_height = int(resolution)
    if width >= height:
        target_height = min(height, short_or_height) if resolution != "2160" else short_or_height
        target_width = target_height * width / max(height, 1)
    else:
        target_width = min(width, short_or_height) if resolution != "2160" else short_or_height
        target_height = target_width * height / max(width, 1)
    return even(target_width), even(target_height)


def target_fps(meta: dict) -> float:
    fps = float(meta.get("fps") or 30)
    if fps < 12:
        fps = 30
    return round(min(fps, 60), 3)


def build_video_filters(settings: VideoSettings, width: int, height: int, fps: float) -> list[str]:
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
        f"fps={fps}",
    ]
    if settings.auto_balance:
        # Temporal smoothing avoids the frame-to-frame pumping common in naive
        # auto-level filters while retaining natural skin tones.
        filters.append(
            "normalize=blackpt=black:whitept=white:smoothing=200:independence=0.12:strength=0.22"
        )

    brightness = max(-0.35, min(0.35, settings.exposure * 0.065 + settings.light * 0.0012))
    contrast = max(0.55, min(1.55, 1.0 + settings.contrast * 0.0045))
    saturation = max(0.0, min(2.0, 1.0 + settings.saturation * 0.009))
    gamma = max(0.55, min(1.55, 1.0 + settings.shadows * 0.003 - settings.highlights * 0.0015))
    gamma_weight = max(0.55, min(1.0, 1.0 - max(0, settings.shadows) * 0.0035))
    filters.append(
        "eq="
        f"brightness={brightness:.4f}:contrast={contrast:.4f}:"
        f"saturation={saturation:.4f}:gamma={gamma:.4f}:gamma_weight={gamma_weight:.4f}"
    )
    if settings.warmth:
        shift = max(-0.25, min(0.25, settings.warmth * 0.0022))
        filters.append(f"colorbalance=rs={shift:.4f}:bs={-shift:.4f}:pl=1")
    filters.append("format=yuv420p")
    return filters


def codec_args(export: ExportSettings, width: int, height: int) -> list[str]:
    codec = "hevc_videotoolbox" if export.codec == "hevc" else "h264_videotoolbox"
    pixels = width * height
    base_mbps = 8 if pixels <= 1280 * 720 else 14 if pixels <= 1920 * 1080 else 40
    multiplier = {"compact": 0.62, "balanced": 0.82, "high": 1.0}[export.quality]
    bitrate = max(4, int(math.ceil(base_mbps * multiplier)))
    return [
        "-c:v",
        codec,
        "-b:v",
        f"{bitrate}M",
        "-maxrate",
        f"{int(bitrate * 1.5)}M",
        "-bufsize",
        f"{bitrate * 2}M",
        "-allow_sw",
        "1",
    ]


def merge_ranges(ranges: Iterable[tuple[float, float]], *, gap: float = 0.08) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in ranges if end > start)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(start, 4), round(end, 4)) for start, end in merged]


def keep_intervals(
    trim_start: float,
    trim_end: float,
    cuts: Iterable[tuple[float, float]],
    *,
    minimum: float = 0.06,
) -> list[tuple[float, float]]:
    start = max(0.0, trim_start)
    end = max(start, trim_end)
    clipped = merge_ranges(
        (max(start, cut_start), min(end, cut_end))
        for cut_start, cut_end in cuts
        if cut_end > start and cut_start < end
    )
    keeps: list[tuple[float, float]] = []
    cursor = start
    for cut_start, cut_end in clipped:
        if cut_start - cursor >= minimum:
            keeps.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if end - cursor >= minimum:
        keeps.append((cursor, end))
    if not keeps:
        raise ValueError("The selected trims and speech cuts remove the entire clip")
    return keeps

