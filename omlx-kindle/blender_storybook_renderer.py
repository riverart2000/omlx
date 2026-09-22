#!/usr/bin/env python3
"""Blender Python renderer for Kindle Studio's professional storybook videos.

The Kindle service prepares full-bleed plates plus a JSON timeline.  This
script uses Blender's Video Sequence Editor to apply smooth camera motion and
animated lower captions, then writes silent scene clips for final audio
mastering by the service.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bpy


FPS = 30


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    return parser.parse_args(values)


def _set_duration(strip, frames: int) -> None:
    # frame_final_duration remains the cross-version VSE duration setter in
    # Blender 4.x/5.x, though 5.x marks it for a future API replacement.
    strip.frame_final_duration = max(1, int(frames))


def _image(editor, name: str, path: str, channel: int,
           frames: int, fit: str = "FILL"):
    strip = editor.strips.new_image(
        name=name, filepath=path, channel=channel, frame_start=1,
        fit_method=fit)
    _set_duration(strip, frames)
    strip.blend_type = "ALPHA_OVER"
    return strip


def _keyframe(value, data_path: str, frame: int) -> None:
    value.keyframe_insert(data_path=data_path, frame=max(1, int(frame)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _animate_art(strip, frames: int, movement: float, pattern: int,
                 width: int, height: int, beats: list[dict],
                 focus: dict | None = None) -> None:
    focus = focus if isinstance(focus, dict) else {}
    faces = [row for row in (focus.get("faces") or []) if isinstance(row, dict)]
    points = [row for row in (focus.get("points") or []) if isinstance(row, dict)]
    focus_kind = str(focus.get("kind") or "")

    def number(row: dict, key: str, default: float) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    def target_for(index: int) -> tuple[float, float] | None:
        if not points:
            return None
        # Establish the complete face group first, then pan gently between
        # individual heads. Single-subject scenes remain anchored on the face.
        if len(points) > 1 and index:
            point = sorted(points, key=lambda row: number(row, "x", .5))[
                (index - 1) % len(points)]
            return number(point, "x", .5), number(point, "y", .38)
        total = sum(max(.05, number(row, "confidence", .5)) for row in points)
        return (
            sum(number(row, "x", .5) * max(.05, number(row, "confidence", .5))
                for row in points) / total,
            sum(number(row, "y", .4) * max(.05, number(row, "confidence", .5))
                for row in points) / total,
        )

    def protect_faces(zoom: float, offset_x: float,
                      offset_y: float) -> tuple[float, float]:
        if not faces:
            return offset_x, offset_y
        left = min(number(row, "x", .5) - number(row, "width", .1) / 2
                   for row in faces)
        right = max(number(row, "x", .5) + number(row, "width", .1) / 2
                    for row in faces)
        top = min(number(row, "y", .35) - number(row, "height", .12) / 2
                  for row in faces)
        bottom = max(number(row, "y", .35) + number(row, "height", .12) / 2
                     for row in faces)
        # Blender X is positive to the right. Y is positive upward while our
        # detector uses top-origin coordinates. These limits keep every head
        # inside a generous title/action-safe box for the entire interpolation.
        minimum_x = (.035 - .5 - (left - .5) * zoom) * width
        maximum_x = (.965 - .5 - (right - .5) * zoom) * width
        minimum_y = (.5 + (bottom - .5) * zoom - .82) * height
        maximum_y = (.5 + (top - .5) * zoom - .035) * height
        if minimum_x <= maximum_x:
            offset_x = _clamp(offset_x, minimum_x, maximum_x)
        if minimum_y <= maximum_y:
            offset_y = _clamp(offset_y, minimum_y, maximum_y)
        return offset_x, offset_y

    if beats:
        previous_frame = 0
        last = None
        for index, beat in enumerate(sorted(
                beats[:4], key=lambda row: float(row.get("at") or 0))):
            frame = 1 + round(_clamp(beat.get("at", 0), 0, .92) * (frames - 1))
            frame = min(frames, max(previous_frame + 1, frame))
            maximum_zoom = 1.105 if faces else (1.11 if points else 1.16)
            zoom = _clamp(beat.get("zoom", 1.05), 1.018, maximum_zoom)
            focus_x = _clamp(beat.get("focus_x", .5), .16, .84)
            focus_y = _clamp(beat.get("focus_y", .5), .16, .84)
            subject = target_for(index)
            if subject:
                # The director still contributes a small compositional bias,
                # but the measured subject position has authority.
                strength = .90 if faces or focus_kind == "estimated_head" else .76
                focus_x = subject[0] * strength + focus_x * (1.0 - strength)
                focus_y = subject[1] * strength + focus_y * (1.0 - strength)
            max_x = width * (zoom - 1.0) * .46
            max_y = height * (zoom - 1.0) * .46
            offset_x = _clamp((.5 - focus_x) * width, -max_x, max_x)
            offset_y = _clamp((focus_y - .5) * height, -max_y, max_y)
            move = str(beat.get("move") or "hold")
            # When the artwork has a measured focal subject, movement comes
            # from interpolating the successive face anchors. An extra blind
            # shove risks pushing the head out of frame.
            if move == "pan_left" and not points:
                offset_x = _clamp(offset_x + max_x * .42, -max_x, max_x)
            elif move == "pan_right" and not points:
                offset_x = _clamp(offset_x - max_x * .42, -max_x, max_x)
            elif move == "tilt_up" and not points:
                offset_y = _clamp(offset_y - max_y * .35, -max_y, max_y)
            elif move == "tilt_down" and not points:
                offset_y = _clamp(offset_y + max_y * .35, -max_y, max_y)
            elif move == "bounce_soft" and not points:
                offset_y = _clamp(
                    offset_y + (-1 if index % 2 else 1) * max_y * .24,
                    -max_y, max_y)
            offset_x, offset_y = protect_faces(zoom, offset_x, offset_y)
            rotation = _clamp(beat.get("rotation", 0), -1.2, 1.2)
            if move == "arc_left":
                rotation = max(rotation, .35)
            elif move == "arc_right":
                rotation = min(rotation, -.35)
            if faces:
                rotation = _clamp(rotation, -.25, .25)
            strip.transform.scale_x = zoom
            strip.transform.scale_y = zoom
            strip.transform.offset_x = offset_x
            strip.transform.offset_y = offset_y
            strip.transform.rotation = math.radians(rotation)
            for data_path in (
                    "scale_x", "scale_y", "offset_x", "offset_y", "rotation"):
                _keyframe(strip.transform, data_path, frame)
            previous_frame = frame
            last = (zoom, offset_x, offset_y, rotation)
        if last and previous_frame < frames:
            zoom, offset_x, offset_y, rotation = last
            strip.transform.scale_x = zoom
            strip.transform.scale_y = zoom
            strip.transform.offset_x = offset_x
            strip.transform.offset_y = offset_y
            strip.transform.rotation = math.radians(rotation)
            for data_path in (
                    "scale_x", "scale_y", "offset_x", "offset_y", "rotation"):
                _keyframe(strip.transform, data_path, frames)
        return

    scale_start = 1.018
    scale_end = 1.018 + movement
    if pattern == 2:
        scale_start, scale_end = scale_end, scale_start
    strip.transform.scale_x = scale_start
    strip.transform.scale_y = scale_start
    _keyframe(strip.transform, "scale_x", 1)
    _keyframe(strip.transform, "scale_y", 1)
    strip.transform.scale_x = scale_end
    strip.transform.scale_y = scale_end
    _keyframe(strip.transform, "scale_x", frames)
    _keyframe(strip.transform, "scale_y", frames)

    travel = width * min(.022, movement * .35)
    if pattern in (1, 3):
        strip.transform.offset_x = -travel if pattern == 1 else travel
        _keyframe(strip.transform, "offset_x", 1)
        strip.transform.offset_x = travel if pattern == 1 else -travel
        _keyframe(strip.transform, "offset_x", frames)


def _font(path: str):
    try:
        if path and Path(path).is_file():
            return bpy.data.fonts.load(path, check_existing=True)
    except Exception:
        pass
    return None


def _caption(editor, spec: dict, index: int, fps: int, height: int,
             selected_font) -> None:
    start = max(1, round(float(spec["start"]) * fps) + 1)
    end = max(start + 6, round(float(spec["end"]) * fps) + 1)
    strip = editor.strips.new_effect(
        name=f"caption-{index:03d}", type="TEXT",
        channel=4 + (index % 5), frame_start=start, length=end - start)
    # The server premeasures every rolling phrase as one line. Flatten any
    # manually supplied newline and retain more wrap width than the planner used
    # so Blender cannot silently create a second line that overlaps its neighbour.
    strip.text = " ".join(str(spec.get("text") or "").split())
    strip.font_size = max(24, int(spec.get("font_size") or round(height * (
        .052 if spec.get("kind") != "heading" else .045))))
    strip.wrap_width = .90
    strip.anchor_x = "CENTER"
    strip.anchor_y = "BOTTOM"
    motion = str(spec.get("motion") or "rise")
    paths = {
        "rise": ((.5, .055), (.5, .155)),
        "glide_left": ((.64, .09), (.44, .135)),
        "glide_right": ((.36, .09), (.56, .135)),
        "pop_soft": ((.5, .102), (.5, .13)),
    }
    location_start, location_end = paths.get(motion, paths["rise"])
    strip.location = location_start
    strip.color = ((1.0, .96, .82, 1.0) if spec.get("kind") == "heading"
                   else (1.0, 1.0, 1.0, 1.0))
    strip.use_shadow = True
    strip.shadow_color = (0.0, 0.0, 0.0, .88)
    if selected_font is not None:
        strip.font = selected_font

    fade_frames = max(3, min(round(fps * .22), (end - start) // 4))
    if motion == "scroll_up":
        base_y = .055
        line_gap = .067
        transition_frames = max(6, round(fps * .34))
        half_transition = transition_frames // 2
        strip.location = (.5, base_y - .025)
        strip.blend_alpha = 0.0
        _keyframe(strip, "location", start)
        _keyframe(strip, "blend_alpha", start)
        strip.location = (.5, base_y)
        strip.blend_alpha = 1.0
        _keyframe(strip, "location", start + fade_frames)
        _keyframe(strip, "blend_alpha", start + fade_frames)

        steps = [float(value) for value in (spec.get("scroll_steps") or [])[:3]]
        current_y = base_y
        last_frame = start + fade_frames
        faded_as_fourth = False
        for step_index, step_seconds in enumerate(steps):
            centre = round(step_seconds * fps) + 1
            before = max(last_frame + 1, centre - half_transition)
            after = min(end - 1, max(before + 2, centre + half_transition))
            strip.location = (.5, current_y)
            strip.blend_alpha = 1.0
            _keyframe(strip, "location", before)
            _keyframe(strip, "blend_alpha", before)
            current_y = base_y + line_gap * (step_index + 1)
            strip.location = (.5, current_y)
            if step_index == 2:
                strip.blend_alpha = 0.0
                faded_as_fourth = True
            else:
                strip.blend_alpha = 1.0
            _keyframe(strip, "location", after)
            _keyframe(strip, "blend_alpha", after)
            last_frame = after

        if not faded_as_fourth:
            fade_out_start = max(last_frame + 1, end - fade_frames)
            strip.location = (.5, current_y)
            strip.blend_alpha = 1.0
            _keyframe(strip, "location", fade_out_start)
            _keyframe(strip, "blend_alpha", fade_out_start)
            strip.location = (.5, current_y + .018)
            strip.blend_alpha = 0.0
            _keyframe(strip, "location", end)
            _keyframe(strip, "blend_alpha", end)
        return

    strip.blend_alpha = 0.0
    _keyframe(strip, "blend_alpha", start)
    strip.blend_alpha = 1.0
    _keyframe(strip, "blend_alpha", start + fade_frames)
    _keyframe(strip, "location", start)

    # Qwen chooses one of four restrained motions per scene.  Phrases still
    # occupy the same safe lower-third lane and overlap without hard cuts.
    strip.location = location_end
    _keyframe(strip, "location", end)
    strip.blend_alpha = 1.0
    _keyframe(strip, "blend_alpha", end - fade_frames)
    strip.blend_alpha = 0.0
    _keyframe(strip, "blend_alpha", end)


def _configure(scene, job: dict, frames: int) -> None:
    scene.render.resolution_x = int(job["width"])
    scene.render.resolution_y = int(job["height"])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(job.get("fps") or FPS)
    scene.frame_start = 1
    scene.frame_end = frames
    # Blender 5.1's macOS background build exposes VSE rendering but not movie
    # formats in the writable file-format enum. Render high-quality JPEG frames
    # and hand them to the app's FFmpeg for a reliable H.264 result.
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 96
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(Path(job["frames_dir"]) / "frame-")
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


def _render(job: dict) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = int(job.get("fps") or FPS)
    frames = max(2, math.ceil(float(job["duration"]) * fps))
    _configure(scene, job, frames)
    editor = scene.sequence_editor_create()

    art = _image(editor, "full-bleed-art", job["plate"], 1, frames)
    _animate_art(art, frames, float(job.get("movement") or .045),
                 int(job.get("pattern") or 0), int(job["width"]),
                 int(job["height"]), list(job.get("beats") or []),
                 job.get("focus") or {})
    if job.get("backdrop"):
        _image(editor, "caption-gradient", job["backdrop"], 2, frames, "FILL")
    if job.get("cover_overlay"):
        _image(editor, "cover-typography", job["cover_overlay"], 3, frames, "FILL")

    selected_font = _font(str(job.get("font") or ""))
    for index, spec in enumerate(job.get("captions") or []):
        _caption(editor, spec, index, fps, int(job["height"]), selected_font)

    output = Path(job["output"])
    frames_dir = Path(job["frames_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    print(f"STORYBOOK_RENDER {job.get('id')} {frames} frames", flush=True)
    bpy.ops.render.render(animation=True)
    source = frames_dir / "frame-%04d.jpg"
    process = subprocess.run([
        str(job.get("ffmpeg") or "/opt/homebrew/bin/ffmpeg"), "-y",
        "-loglevel", "error", "-framerate", str(fps), "-start_number", "1",
        "-i", str(source), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ], capture_output=True, text=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if process.returncode:
        raise RuntimeError("FFmpeg could not assemble Blender frames: "
                           + (process.stderr or process.stdout)[-900:])
    if not output.exists():
        raise RuntimeError(f"Blender did not create {output}")


def main() -> None:
    with open(_args().job, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for job in payload.get("scenes") or []:
        _render(job)
    print("STORYBOOK_RENDER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
