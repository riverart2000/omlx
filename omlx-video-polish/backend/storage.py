"""Durable upload records so projects survive a service restart."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote

from .config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, UPLOAD_DIR
from .media import probe_media, safe_filename


UPLOAD_ID_RE = re.compile(r"^[a-fA-F0-9-]{16,64}$")


def _record_path(upload_id: str) -> Path:
    return UPLOAD_DIR / f"{upload_id}.json"


def validate_upload_id(upload_id: str) -> str:
    if not UPLOAD_ID_RE.fullmatch(upload_id):
        raise ValueError("Invalid upload id")
    return upload_id


def store_upload(
    upload_id: str,
    filename_header: str,
    stream: BinaryIO,
    content_length: int,
) -> dict[str, Any]:
    validate_upload_id(upload_id)
    if content_length <= 0:
        raise ValueError("The upload is empty")
    if content_length > MAX_UPLOAD_BYTES:
        raise ValueError("The video is larger than the configured upload limit")
    filename = safe_filename(unquote(filename_header or "video.mp4"))
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {suffix or 'unknown'}")

    final_path = UPLOAD_DIR / f"{upload_id}{suffix}"
    part_path = UPLOAD_DIR / f"{upload_id}.part"
    remaining = content_length
    with part_path.open("wb") as output:
        while remaining:
            chunk = stream.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Upload ended before all bytes arrived")
            output.write(chunk)
            remaining -= len(chunk)
    part_path.replace(final_path)

    try:
        media = probe_media(final_path)
    except Exception:
        final_path.unlink(missing_ok=True)
        raise
    record = {
        "id": upload_id,
        "filename": filename,
        "path": str(final_path),
        "size": content_length,
        "content_type": mimetypes.guess_type(filename)[0] or "video/mp4",
        "media": media,
        "url": f"/media/uploads/{upload_id}",
    }
    _record_path(upload_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def get_upload(upload_id: str) -> dict[str, Any]:
    validate_upload_id(upload_id)
    record_path = _record_path(upload_id)
    if not record_path.exists():
        raise FileNotFoundError("Upload not found")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not Path(record["path"]).exists():
        raise FileNotFoundError("Uploaded video file is missing")
    return record


def delete_upload(upload_id: str) -> None:
    try:
        record = get_upload(upload_id)
    except FileNotFoundError:
        _record_path(upload_id).unlink(missing_ok=True)
        return
    Path(record["path"]).unlink(missing_ok=True)
    _record_path(upload_id).unlink(missing_ok=True)

