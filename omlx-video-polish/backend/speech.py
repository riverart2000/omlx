"""MLX Whisper transcription and safe speech-edit suggestions."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .config import FFMPEG, MLX_PYTHON, WHISPER_MODEL
from .llm_editor import review_false_starts
from .media import LogFn, merge_ranges, run_command


FILLERS = {
    "um",
    "umm",
    "ummm",
    "uh",
    "uhh",
    "uhhh",
    "erm",
    "err",
    "er",
    "ah",
    "aah",
    "hmm",
    "hm",
    "mm",
}


def _normalise_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", value.lower()).strip("'")


def _word_list(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "token": _normalise_word(text),
                    "start": float(word.get("start") or 0),
                    "end": float(word.get("end") or word.get("start") or 0),
                }
            )
    return words


def _cut_around(words: list[dict], start_index: int, end_index: int, padding: float) -> tuple[float, float]:
    word_start = words[start_index]["start"]
    word_end = words[end_index]["end"]
    if start_index:
        left_gap_midpoint = (words[start_index - 1]["end"] + word_start) / 2
        start = max(left_gap_midpoint, word_start - padding)
    else:
        start = max(0, word_start - padding)
    if end_index + 1 < len(words):
        right_gap_midpoint = (word_end + words[end_index + 1]["start"]) / 2
        end = min(right_gap_midpoint, word_end + padding)
    else:
        end = word_end + padding
    return round(max(0, start), 3), round(max(start, end), 3)


def detect_fillers(words: list[dict], padding_ms: int = 55) -> list[dict]:
    suggestions = []
    padding = padding_ms / 1000.0
    for index, word in enumerate(words):
        if word["token"] not in FILLERS:
            continue
        start, end = _cut_around(words, index, index, padding)
        suggestions.append(
            {
                "id": uuid.uuid4().hex[:12],
                "start": start,
                "end": end,
                "text": word["word"],
                "reason": "filler word",
                "confidence": 0.99,
                "enabled": True,
                "source": "Whisper timing",
            }
        )
    return suggestions


def detect_repeated_starts(words: list[dict], padding_ms: int = 55) -> list[dict]:
    suggestions: list[dict] = []
    tokens = [word["token"] for word in words]
    padding = padding_ms / 1000.0
    used_until = -1
    for index in range(len(words)):
        if index <= used_until or not tokens[index]:
            continue
        found = None
        for length in range(min(6, len(words) - index), 0, -1):
            first = tokens[index : index + length]
            if any(not token or token in FILLERS for token in first):
                continue
            cursor = index + length
            while cursor < len(words) and cursor <= index + length + 3 and tokens[cursor] in FILLERS:
                cursor += 1
            second = tokens[cursor : cursor + length]
            if first != second:
                continue
            # Single repeated words such as "very very" are often intentional.
            # Leave these to the local LLM instead of making an unsafe automatic cut.
            if length == 1:
                continue
            found = (length, cursor)
            break
        if not found:
            raw = words[index]["word"].strip()
            if raw.endswith(("-", "—")) and index + 1 < len(words):
                start, end = _cut_around(words, index, index, padding)
                suggestions.append(
                    {
                        "id": uuid.uuid4().hex[:12],
                        "start": start,
                        "end": end,
                        "text": raw,
                        "reason": "abandoned word",
                        "confidence": 0.88,
                        "enabled": True,
                        "source": "speech pattern",
                    }
                )
            continue
        length, repeat_at = found
        remove_end = repeat_at - 1
        start, end = _cut_around(words, index, remove_end, padding)
        suggestions.append(
            {
                "id": uuid.uuid4().hex[:12],
                "start": start,
                "end": end,
                "text": " ".join(word["word"] for word in words[index : remove_end + 1]),
                "reason": "repeated false start",
                "confidence": 0.94 if length > 1 else 0.82,
                "enabled": True,
                "source": "speech pattern",
            }
        )
        used_until = remove_end
    return suggestions


def _snap_llm_suggestions(raw_cuts: list[dict], words: list[dict], duration: float) -> list[dict]:
    suggestions = []
    for raw in raw_cuts:
        try:
            start = max(0.0, float(raw.get("start")))
            end = min(duration, float(raw.get("end")))
            confidence = float(raw.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if confidence < 0.72 or end <= start or end - start > 18:
            continue
        covered = [word for word in words if word["end"] > start and word["start"] < end]
        if not covered or len(covered) > 35:
            continue
        # Snap model-supplied times to the word-timing data rather than trusting
        # generated decimal values verbatim.
        snapped_start = covered[0]["start"]
        snapped_end = covered[-1]["end"]
        suggestions.append(
            {
                "id": uuid.uuid4().hex[:12],
                "start": round(snapped_start, 3),
                "end": round(snapped_end, 3),
                "text": " ".join(word["word"] for word in covered),
                "reason": str(raw.get("reason") or "false start")[:80],
                "confidence": round(confidence, 2),
                "enabled": confidence >= 0.82,
                "source": "local oMLX editor",
            }
        )
    return suggestions


def _deduplicate(suggestions: list[dict]) -> list[dict]:
    suggestions.sort(key=lambda item: (item["start"], -item["confidence"]))
    result: list[dict] = []
    for item in suggestions:
        duplicate = next(
            (
                current
                for current in result
                if min(current["end"], item["end"]) - max(current["start"], item["start"]) > 0.08
            ),
            None,
        )
        if duplicate:
            if item["confidence"] > duplicate["confidence"]:
                result[result.index(duplicate)] = item
            continue
        result.append(item)
    return result


def transcribe_and_suggest(
    source: Path,
    work_dir: Path,
    *,
    remove_fillers: bool,
    remove_false_starts: bool,
    use_llm: bool,
    padding_ms: int,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    speech_wav = work_dir / "speech_for_transcription.wav"
    run_command(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-af",
            "highpass=f=65,lowpass=f=16000,afftdn=nr=8:nf=-45:tn=1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(speech_wav),
        ],
        logger=logger,
    )
    whisper_cli = str(Path(MLX_PYTHON).with_name("mlx_whisper"))
    run_command(
        [
            whisper_cli,
            str(speech_wav),
            "--model",
            WHISPER_MODEL,
            "--output-dir",
            str(work_dir),
            "--output-name",
            "transcript",
            "--output-format",
            "json",
            "--word-timestamps",
            "True",
            "--verbose",
            "False",
        ],
        logger=logger,
    )
    transcript_path = work_dir / "transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    words = _word_list(transcript)
    duration = max((word["end"] for word in words), default=0.0)
    suggestions: list[dict] = []
    if remove_fillers:
        suggestions.extend(detect_fillers(words, padding_ms))
    if remove_false_starts:
        suggestions.extend(detect_repeated_starts(words, padding_ms))
        if use_llm:
            raw_llm_cuts = review_false_starts(transcript.get("segments") or [])
            suggestions.extend(_snap_llm_suggestions(raw_llm_cuts, words, duration))
    suggestions = _deduplicate(suggestions)
    return {
        "text": str(transcript.get("text") or "").strip(),
        "language": transcript.get("language") or "unknown",
        "word_count": len(words),
        "suggestions": suggestions,
        "cut_seconds": round(sum(item["end"] - item["start"] for item in suggestions if item["enabled"]), 2),
    }
