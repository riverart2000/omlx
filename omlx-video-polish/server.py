#!/usr/bin/env python3
"""Local HTTP sidecar for oMLX Video Polish (default port 8950)."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.analysis_pipeline import run_analysis
from backend.config import (
    DEEPFILTER_MODEL,
    FFMPEG,
    FFPROBE,
    HOST,
    MLX_PYTHON,
    OUTPUT_DIR,
    PORT,
    STATIC_DIR,
    WHISPER_MODEL,
)
from backend.jobs import JOBS
from backend.models import RenderRequest
from backend.render_pipeline import run_render
from backend.storage import delete_upload, get_upload, store_upload


class VideoPolishHandler(BaseHTTPRequestHandler):
    server_version = "oMLXVideoPolish/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, DELETE, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._handle_get(head_only=True)

    def do_GET(self) -> None:
        self._handle_get(head_only=False)

    def _handle_get(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/health", "/api/health"}:
                self._json(
                    {
                        "ok": True,
                        "service": "oMLX Video Polish",
                        "version": "0.1.0",
                        "port": PORT,
                        "capabilities": {
                            "ffmpeg": Path(FFMPEG).exists(),
                            "ffprobe": Path(FFPROBE).exists(),
                            "mlx_python": Path(MLX_PYTHON).exists(),
                            "whisper": Path(WHISPER_MODEL).exists(),
                            "deepfilter": Path(DEEPFILTER_MODEL).exists(),
                        },
                        "output_dir": str(OUTPUT_DIR),
                    },
                    head_only=head_only,
                )
                return
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{16})", path)
            if match:
                job = JOBS.get(match.group(1))
                if not job:
                    self._error(HTTPStatus.NOT_FOUND, "Job not found")
                else:
                    self._json(job, head_only=head_only)
                return
            match = re.fullmatch(r"/api/uploads/([a-fA-F0-9-]{16,64})", path)
            if match:
                self._json(get_upload(match.group(1)), head_only=head_only)
                return
            match = re.fullmatch(r"/media/uploads/([a-fA-F0-9-]{16,64})", path)
            if match:
                record = get_upload(match.group(1))
                self._file(Path(record["path"]), record.get("content_type"), head_only=head_only)
                return
            if path.startswith("/media/outputs/"):
                name = Path(unquote(path.removeprefix("/media/outputs/"))).name
                candidate = (OUTPUT_DIR / name).resolve()
                if candidate.parent != OUTPUT_DIR.resolve() or not candidate.exists():
                    self._error(HTTPStatus.NOT_FOUND, "Output not found")
                else:
                    self._file(candidate, "video/mp4", head_only=head_only)
                return
            if path in {"", "/"}:
                self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8", head_only=head_only)
                return
            if path.startswith("/static/"):
                name = Path(unquote(path.removeprefix("/static/"))).name
                candidate = STATIC_DIR / name
                if not candidate.exists():
                    self._error(HTTPStatus.NOT_FOUND, "Asset not found")
                else:
                    self._file(candidate, mimetypes.guess_type(name)[0], head_only=head_only)
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/uploads/([a-fA-F0-9-]{16,64})", path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            record = store_upload(
                match.group(1),
                self.headers.get("X-Filename") or "video.mp4",
                self.rfile,
                length,
            )
            self._json(record, status=HTTPStatus.CREATED)
        except (ValueError, OSError, RuntimeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/uploads/([a-fA-F0-9-]{16,64})", path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            delete_upload(match.group(1))
            self._json({"ok": True})
        except (ValueError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/analyze":
                clips = payload.get("clips") or []
                clip_ids = [str(item.get("upload_id") or "") for item in clips]
                if not clip_ids:
                    raise ValueError("Add at least one video before analysing speech")
                for upload_id in clip_ids:
                    get_upload(upload_id)
                job_id = JOBS.create("analysis")
                JOBS.run(job_id, run_analysis, clip_ids, payload.get("edit") or {})
                self._json({"job_id": job_id}, status=HTTPStatus.ACCEPTED)
                return
            if path == "/api/render":
                request = RenderRequest.from_dict(payload)
                for clip in request.clips:
                    get_upload(clip.upload_id)
                job_id = JOBS.create("render")
                JOBS.run(job_id, run_render, request.to_dict())
                self._json({"job_id": job_id}, status=HTTPStatus.ACCEPTED)
                return
            if path == "/api/reveal":
                raw = str(payload.get("path") or "")
                candidate = Path(raw).resolve()
                if candidate.parent != OUTPUT_DIR.resolve() or not candidate.exists():
                    raise ValueError("Output path is not valid")
                subprocess.Popen(["/usr/bin/open", "-R", str(candidate)])
                self._json({"ok": True})
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")
        except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > 10 * 1024 * 1024:
            raise ValueError("Invalid JSON request size")
        parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def _json(self, value: dict, status: int = HTTPStatus.OK, head_only: bool = False) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _file(self, path: Path, content_type: str | None, *, head_only: bool) -> None:
        if not path.exists() or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "File not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2) or size - 1), size - 1)
            else:
                suffix = int(match.group(2) or 0)
                start = max(0, size - suffix)
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if path.parent == STATIC_DIR:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), VideoPolishHandler)
    server.daemon_threads = True
    print(f"oMLX Video Polish ready at http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
