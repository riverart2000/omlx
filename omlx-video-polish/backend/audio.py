"""Dialogue cleanup pipeline: MLX voice isolation plus deterministic mastering."""

from __future__ import annotations

from pathlib import Path

from .config import DEEPFILTER_MODEL, FFMPEG, MLX_PYTHON
from .media import LogFn, run_command
from .models import AudioSettings


def _audio_filters(settings: AudioSettings) -> list[str]:
    filters = ["highpass=f=65", "lowpass=f=17000"]
    if settings.noise_removal:
        reduction = 5 + settings.noise_removal * 0.17
        noise_floor = -38 - settings.noise_removal * 0.12
        filters.append(f"afftdn=nr={reduction:.2f}:nf={noise_floor:.2f}:tn=1:gs=6")
    if settings.dialogue_enhance:
        presence = settings.dialogue_enhance * 0.045
        warmth = settings.dialogue_enhance * 0.018
        filters.extend(
            [
                f"equalizer=f=180:t=q:w=0.9:g={warmth:.2f}",
                f"equalizer=f=2800:t=q:w=1.1:g={presence:.2f}",
            ]
        )
    if settings.compression:
        ratio = 1.4 + settings.compression * 0.026
        threshold = max(0.06, 0.22 - settings.compression * 0.0013)
        makeup = 1.0 + settings.compression * 0.008
        filters.append(
            "acompressor="
            f"threshold={threshold:.4f}:ratio={ratio:.2f}:attack=18:release=220:"
            f"makeup={makeup:.2f}:knee=2.5:link=average:detection=rms"
        )
    filters.extend(["alimiter=limit=0.97:attack=5:release=80", "aresample=48000"])
    return filters


def prepare_audio(
    source: Path,
    *,
    has_audio: bool,
    duration: float,
    settings: AudioSettings,
    work_dir: Path,
    logger: LogFn | None = None,
) -> tuple[Path, dict]:
    """Create a timeline-aligned 48 kHz mono WAV for one source clip."""
    original = work_dir / "audio_original.wav"
    if has_audio:
        run_command(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "pcm_f32le",
                str(original),
            ],
            logger=logger,
        )
    else:
        run_command(
            [
                FFMPEG,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-t",
                f"{duration:.4f}",
                "-c:a",
                "pcm_f32le",
                str(original),
            ],
            logger=logger,
        )

    current = original
    isolation_used = False
    isolation_error = None
    if has_audio and settings.voice_isolation > 0 and Path(MLX_PYTHON).exists() and Path(DEEPFILTER_MODEL).exists():
        cleaned = work_dir / "audio_deepfiltered.wav"
        worker = Path(__file__).with_name("deepfilter_worker.py")
        try:
            run_command(
                [MLX_PYTHON, str(worker), str(original), str(cleaned), DEEPFILTER_MODEL],
                logger=logger,
            )
            strength = settings.voice_isolation / 100.0
            if strength >= 0.995:
                current = cleaned
            else:
                blended = work_dir / "audio_isolated_mix.wav"
                run_command(
                    [
                        FFMPEG,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(original),
                        "-i",
                        str(cleaned),
                        "-filter_complex",
                        f"[0:a][1:a]amix=inputs=2:weights='{1-strength:.4f} {strength:.4f}':normalize=0,alimiter=limit=0.98",
                        "-c:a",
                        "pcm_f32le",
                        str(blended),
                    ],
                    logger=logger,
                )
                current = blended
            isolation_used = True
        except Exception as exc:
            # The FFmpeg denoiser below remains available, so a Metal/model issue
            # should degrade gracefully instead of losing the whole export.
            isolation_error = str(exc)
            if logger:
                logger("MLX voice isolation unavailable; continuing with spectral cleanup")

    mastered = work_dir / "audio_prepared.wav"
    filters = _audio_filters(settings)
    run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(current),
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_f32le",
            str(mastered),
        ],
        logger=logger,
    )
    return mastered, {
        "voice_isolation_used": isolation_used,
        "voice_isolation_error": isolation_error,
        "filters": filters,
    }

