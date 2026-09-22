"""Modular render pipeline for trim, clean-up, stitching, transitions and export."""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .audio import prepare_audio
from .config import FFMPEG, OUTPUT_DIR, WORK_DIR
from .jobs import JOBS
from .media import (
    build_video_filters,
    codec_args,
    keep_intervals,
    probe_media,
    run_command,
    safe_filename,
    target_dimensions,
    target_fps,
)
from .models import Clip, ExportSettings, RenderRequest, VideoSettings
from .storage import get_upload


def _software_codec_args(export: ExportSettings) -> list[str]:
    if export.codec == "hevc":
        crf = {"compact": "28", "balanced": "24", "high": "20"}[export.quality]
        return ["-c:v", "libx265", "-preset", "medium", "-crf", crf]
    crf = {"compact": "25", "balanced": "21", "high": "18"}[export.quality]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", crf]


def _render_clip(
    source: Path,
    prepared_audio: Path,
    output: Path,
    clip: Clip,
    source_duration: float,
    video: VideoSettings,
    export: ExportSettings,
    width: int,
    height: int,
    fps: float,
    logger,
) -> dict:
    trim_end = min(source_duration, clip.trim_end if clip.trim_end is not None else source_duration)
    trim_start = min(max(0.0, clip.trim_start), trim_end)
    cuts = [(item.start, item.end) for item in clip.cuts if item.enabled]
    keeps = keep_intervals(trim_start, trim_end, cuts)

    graph: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []
    for index, (start, end) in enumerate(keeps):
        graph.append(f"[0:v:0]trim=start={start:.4f}:end={end:.4f},setpts=PTS-STARTPTS[v{index}]")
        graph.append(f"[1:a:0]atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS[a{index}]")
        video_labels.append(f"[v{index}]")
        audio_labels.append(f"[a{index}]")
    if len(keeps) == 1:
        video_base = "v0"
        audio_base = "a0"
    else:
        inputs = "".join(
            video_labels[index] + audio_labels[index] for index in range(len(keeps))
        )
        graph.append(f"{inputs}concat=n={len(keeps)}:v=1:a=1[vbase][abase]")
        video_base = "vbase"
        audio_base = "abase"
    graph.append(f"[{video_base}]" + ",".join(build_video_filters(video, width, height, fps)) + "[vout]")
    graph.append(f"[{audio_base}]pan=stereo|c0=c0|c1=c0,aresample=48000[aout]")

    common = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-i",
        str(prepared_audio),
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
    ]
    ending = [
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-metadata:s:v:0",
        "rotate=0",
        "-shortest",
        str(output),
    ]
    try:
        run_command(common + codec_args(export, width, height) + ending, logger=logger)
    except RuntimeError:
        if logger:
            logger("VideoToolbox encoding failed; retrying with the software encoder")
        output.unlink(missing_ok=True)
        run_command(common + _software_codec_args(export) + ending, logger=logger)
    rendered_meta = probe_media(output)
    return {
        "kept_ranges": keeps,
        "removed_seconds": round((trim_end - trim_start) - rendered_meta["duration"], 3),
        "duration": rendered_meta["duration"],
    }


def _stitch_without_transition(inputs: list[Path], output: Path, work_dir: Path, logger) -> None:
    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in inputs),
        encoding="utf-8",
    )
    run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        logger=logger,
    )


def _stitch_with_transitions(
    inputs: list[Path],
    output: Path,
    style: str,
    requested_duration: float,
    export: ExportSettings,
    width: int,
    height: int,
    logger,
) -> None:
    durations = [probe_media(path)["duration"] for path in inputs]
    command = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for path in inputs:
        command.extend(["-i", str(path)])

    graph: list[str] = []
    for index in range(len(inputs)):
        graph.append(f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[vn{index}]")
        graph.append(f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[an{index}]")
    video_label = "vn0"
    audio_label = "an0"
    timeline = durations[0]
    for index in range(1, len(inputs)):
        transition_duration = min(requested_duration, durations[index - 1] / 3, durations[index] / 3)
        transition_duration = max(0.05, transition_duration)
        offset = max(0.0, timeline - transition_duration)
        next_video = f"vx{index}"
        next_audio = f"ax{index}"
        graph.append(
            f"[{video_label}][vn{index}]xfade=transition={style}:duration={transition_duration:.4f}:offset={offset:.4f}[{next_video}]"
        )
        graph.append(
            f"[{audio_label}][an{index}]acrossfade=d={transition_duration:.4f}:c1=tri:c2=tri[{next_audio}]"
        )
        timeline += durations[index] - transition_duration
        video_label, audio_label = next_video, next_audio

    common = command + [
        "-filter_complex",
        ";".join(graph),
        "-map",
        f"[{video_label}]",
        "-map",
        f"[{audio_label}]",
    ]
    ending = [
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        run_command(common + codec_args(export, width, height) + ending, logger=logger)
    except RuntimeError:
        if logger:
            logger("VideoToolbox transition encode failed; retrying with the software encoder")
        output.unlink(missing_ok=True)
        run_command(common + _software_codec_args(export) + ending, logger=logger)


def _loudness_measurement(source: Path, target_lufs: float, true_peak: float, logger) -> dict | None:
    completed = run_command(
        [
            FFMPEG,
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            f"loudnorm=I={target_lufs}:LRA=11:TP={true_peak}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        logger=logger,
        capture=True,
    )
    blocks = re.findall(r"\{\s*\"input_i\".*?\}", completed.stderr or "", re.DOTALL)
    if not blocks:
        return None
    try:
        measured = json.loads(blocks[-1])
        required = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
        if any(not math.isfinite(float(measured.get(key, "nan"))) for key in required):
            return None
        return measured
    except json.JSONDecodeError:
        return None
    except (TypeError, ValueError):
        return None


def _normalise_loudness(source: Path, output: Path, target_lufs: float, true_peak: float, logger) -> dict:
    measured = _loudness_measurement(source, target_lufs, true_peak, logger)
    base = f"loudnorm=I={target_lufs}:LRA=11:TP={true_peak}"
    if measured:
        audio_filter = (
            base
            + f":measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}"
            + f":measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}"
            + f":offset={measured['target_offset']}:linear=true:print_format=summary"
        )
    else:
        audio_filter = base + ":print_format=summary"
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
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-af",
            audio_filter,
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output),
        ],
        logger=logger,
    )
    return measured or {}


def _output_path(request: RenderRequest, first_name: str) -> Path:
    requested = safe_filename(request.export.filename, "")
    if requested:
        stem = Path(requested).stem
    else:
        stem = Path(first_name).stem + " polished"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{safe_filename(stem)} {timestamp}"
    candidate = OUTPUT_DIR / f"{base}.mp4"
    suffix = 2
    while candidate.exists():
        candidate = OUTPUT_DIR / f"{base}-{suffix}.mp4"
        suffix += 1
    return candidate


def run_render(job_id: str, request_data: dict) -> dict:
    request = RenderRequest.from_dict(request_data)
    work_dir = WORK_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    records = [get_upload(clip.upload_id) for clip in request.clips]
    first_meta = records[0]["media"]
    width, height = target_dimensions(first_meta, request.export.resolution)
    fps = target_fps(first_meta)
    messages: list[str] = []

    def log(message: str) -> None:
        messages.append(message)
        state = JOBS.get(job_id) or {}
        JOBS.progress(job_id, int(state.get("progress") or 1), str(state.get("stage") or "Rendering"), message)

    intermediates: list[Path] = []
    clip_reports: list[dict] = []
    total = len(request.clips)
    for index, (clip, record) in enumerate(zip(request.clips, records)):
        clip_dir = work_dir / f"clip-{index + 1:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        progress = 5 + int(index / total * 65)
        JOBS.progress(job_id, progress, f"Cleaning audio {index + 1} of {total}", record["filename"])
        audio_path, audio_report = prepare_audio(
            Path(record["path"]),
            has_audio=bool(record["media"].get("has_audio")),
            duration=float(record["media"]["duration"]),
            settings=request.audio,
            work_dir=clip_dir,
            logger=log,
        )
        JOBS.progress(
            job_id,
            progress + max(3, int(35 / total)),
            f"Polishing clip {index + 1} of {total}",
            "Applying trims, approved speech edits, light and colour",
        )
        output = clip_dir / "polished.mp4"
        clip_report = _render_clip(
            Path(record["path"]),
            audio_path,
            output,
            clip,
            float(record["media"]["duration"]),
            request.video,
            request.export,
            width,
            height,
            fps,
            log,
        )
        clip_report.update(
            {
                "filename": record["filename"],
                "audio": audio_report,
                "speech_cuts": len([item for item in clip.cuts if item.enabled]),
            }
        )
        clip_reports.append(clip_report)
        intermediates.append(output)

    assembled = work_dir / "assembled.mp4"
    JOBS.progress(job_id, 76, "Joining clips", "Building the final timeline")
    if len(intermediates) == 1:
        shutil.copy2(intermediates[0], assembled)
    elif request.transition.style == "none" or request.transition.duration <= 0:
        _stitch_without_transition(intermediates, assembled, work_dir, log)
    else:
        _stitch_with_transitions(
            intermediates,
            assembled,
            request.transition.style,
            request.transition.duration,
            request.export,
            width,
            height,
            log,
        )

    output = _output_path(request, records[0]["filename"])
    JOBS.progress(job_id, 88, "Correcting loudness", "Measuring the complete programme")
    loudness = {}
    if request.audio.normalize:
        loudness = _normalise_loudness(
            assembled,
            output,
            request.audio.target_lufs,
            request.audio.true_peak,
            log,
        )
    else:
        shutil.copy2(assembled, output)

    final_meta = probe_media(output)
    report = {
        "job_id": job_id,
        "output": str(output),
        "filename": output.name,
        "url": "/media/outputs/" + quote(output.name),
        "duration": final_meta["duration"],
        "width": final_meta["width"],
        "height": final_meta["height"],
        "fps": final_meta["fps"],
        "clips": clip_reports,
        "loudness": loudness,
        "settings": request.to_dict(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    JOBS.progress(job_id, 98, "Finishing export", f"Saved to {output}")
    return report
