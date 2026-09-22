"""Typed request models and validation kept independent from the HTTP layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    return int(round(_number(value, default, minimum, maximum)))


def _boolean(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Cut:
    start: float
    end: float
    enabled: bool = True
    reason: str = "manual"
    text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cut":
        start = _number(data.get("start"), 0, 0, 24 * 60 * 60)
        end = _number(data.get("end"), start, start, 24 * 60 * 60)
        return cls(
            start=start,
            end=end,
            enabled=_boolean(data.get("enabled"), True),
            reason=str(data.get("reason") or "manual")[:80],
            text=str(data.get("text") or "")[:500],
        )


@dataclass(slots=True)
class Clip:
    upload_id: str
    trim_start: float = 0.0
    trim_end: float | None = None
    cuts: list[Cut] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        upload_id = str(data.get("upload_id") or "").strip()
        if not upload_id:
            raise ValueError("Every clip needs an upload_id")
        trim_start = _number(data.get("trim_start"), 0, 0, 24 * 60 * 60)
        raw_end = data.get("trim_end")
        trim_end = None if raw_end in (None, "") else _number(raw_end, trim_start, trim_start, 24 * 60 * 60)
        return cls(
            upload_id=upload_id,
            trim_start=trim_start,
            trim_end=trim_end,
            cuts=[Cut.from_dict(item) for item in (data.get("cuts") or [])],
        )


@dataclass(slots=True)
class VideoSettings:
    auto_balance: bool = True
    light: int = 15
    exposure: float = 0.0
    contrast: int = 8
    highlights: int = 0
    shadows: int = 8
    saturation: int = 8
    warmth: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VideoSettings":
        data = data or {}
        return cls(
            auto_balance=_boolean(data.get("auto_balance"), True),
            light=_integer(data.get("light"), 15, -100, 100),
            exposure=_number(data.get("exposure"), 0, -2, 2),
            contrast=_integer(data.get("contrast"), 8, -100, 100),
            highlights=_integer(data.get("highlights"), 0, -100, 100),
            shadows=_integer(data.get("shadows"), 8, -100, 100),
            saturation=_integer(data.get("saturation"), 8, -100, 100),
            warmth=_integer(data.get("warmth"), 0, -100, 100),
        )


@dataclass(slots=True)
class AudioSettings:
    voice_isolation: int = 70
    noise_removal: int = 45
    dialogue_enhance: int = 40
    compression: int = 35
    normalize: bool = True
    target_lufs: float = -16.0
    true_peak: float = -1.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AudioSettings":
        data = data or {}
        return cls(
            voice_isolation=_integer(data.get("voice_isolation"), 70, 0, 100),
            noise_removal=_integer(data.get("noise_removal"), 45, 0, 100),
            dialogue_enhance=_integer(data.get("dialogue_enhance"), 40, 0, 100),
            compression=_integer(data.get("compression"), 35, 0, 100),
            normalize=_boolean(data.get("normalize"), True),
            target_lufs=_number(data.get("target_lufs"), -16, -24, -12),
            true_peak=_number(data.get("true_peak"), -1.5, -3, -1),
        )


@dataclass(slots=True)
class EditSettings:
    remove_fillers: bool = True
    remove_false_starts: bool = True
    use_llm: bool = True
    cut_padding_ms: int = 55

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EditSettings":
        data = data or {}
        return cls(
            remove_fillers=_boolean(data.get("remove_fillers"), True),
            remove_false_starts=_boolean(data.get("remove_false_starts"), True),
            use_llm=_boolean(data.get("use_llm"), True),
            cut_padding_ms=_integer(data.get("cut_padding_ms"), 55, 0, 250),
        )


@dataclass(slots=True)
class TransitionSettings:
    style: str = "fade"
    duration: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TransitionSettings":
        data = data or {}
        allowed = {"none", "fade", "dissolve", "wipeleft", "wiperight", "smoothleft", "smoothright"}
        style = str(data.get("style") or "fade")
        if style not in allowed:
            style = "fade"
        return cls(style=style, duration=_number(data.get("duration"), 0.35, 0, 2.0))


@dataclass(slots=True)
class ExportSettings:
    resolution: str = "source"
    codec: str = "h264"
    quality: str = "high"
    filename: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExportSettings":
        data = data or {}
        resolution = str(data.get("resolution") or "source")
        if resolution not in {"source", "2160", "1080", "720"}:
            resolution = "source"
        codec = str(data.get("codec") or "h264")
        if codec not in {"h264", "hevc"}:
            codec = "h264"
        quality = str(data.get("quality") or "high")
        if quality not in {"compact", "balanced", "high"}:
            quality = "high"
        return cls(
            resolution=resolution,
            codec=codec,
            quality=quality,
            filename=str(data.get("filename") or "")[:120],
        )


@dataclass(slots=True)
class RenderRequest:
    clips: list[Clip]
    video: VideoSettings
    audio: AudioSettings
    edit: EditSettings
    transition: TransitionSettings
    export: ExportSettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderRequest":
        clips = [Clip.from_dict(item) for item in (data.get("clips") or [])]
        if not clips:
            raise ValueError("Add at least one video")
        if len(clips) > 50:
            raise ValueError("A project can contain at most 50 clips")
        return cls(
            clips=clips,
            video=VideoSettings.from_dict(data.get("video")),
            audio=AudioSettings.from_dict(data.get("audio")),
            edit=EditSettings.from_dict(data.get("edit")),
            transition=TransitionSettings.from_dict(data.get("transition")),
            export=ExportSettings.from_dict(data.get("export")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

