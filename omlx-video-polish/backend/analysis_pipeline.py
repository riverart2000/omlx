"""Asynchronous speech-analysis job orchestration."""

from __future__ import annotations

from pathlib import Path

from .config import WORK_DIR
from .jobs import JOBS
from .models import EditSettings
from .speech import transcribe_and_suggest
from .storage import get_upload


def run_analysis(job_id: str, clip_ids: list[str], edit_data: dict) -> dict:
    edit = EditSettings.from_dict(edit_data)
    results = []
    total = len(clip_ids)

    def log(message: str) -> None:
        state = JOBS.get(job_id) or {}
        JOBS.progress(job_id, int(state.get("progress") or 1), str(state.get("stage") or "Analysing"), message)

    for index, upload_id in enumerate(clip_ids):
        record = get_upload(upload_id)
        media = record.get("media") or {}
        base_progress = 5 + int(index / max(total, 1) * 85)
        JOBS.progress(
            job_id,
            base_progress,
            f"Transcribing clip {index + 1} of {total}",
            f"Analysing {record['filename']}",
        )
        if not media.get("has_audio"):
            results.append(
                {
                    "upload_id": upload_id,
                    "filename": record["filename"],
                    "text": "",
                    "language": "unknown",
                    "word_count": 0,
                    "suggestions": [],
                    "cut_seconds": 0,
                    "warning": "This clip has no audio track",
                }
            )
            continue
        clip_work = WORK_DIR / job_id / upload_id
        result = transcribe_and_suggest(
            Path(record["path"]),
            clip_work,
            remove_fillers=edit.remove_fillers,
            remove_false_starts=edit.remove_false_starts,
            use_llm=edit.use_llm,
            padding_ms=edit.cut_padding_ms,
            logger=log,
        )
        result.update({"upload_id": upload_id, "filename": record["filename"]})
        results.append(result)
    JOBS.progress(job_id, 95, "Preparing review", "Speech edit suggestions are ready to review")
    return {"clips": results}

