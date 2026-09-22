"""Runtime configuration for the local Video Polish sidecar."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_DIR / "static"
DATA_DIR = Path(os.environ.get("VIDEO_POLISH_DATA_DIR", PROJECT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"
WORK_DIR = DATA_DIR / "work"
OUTPUT_DIR = Path(
    os.environ.get(
        "VIDEO_POLISH_OUTPUT_DIR",
        str(Path.home() / "Movies" / "oMLX Video Polish"),
    )
)

for directory in (DATA_DIR, UPLOAD_DIR, JOB_DIR, WORK_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("VIDEO_POLISH_HOST", "127.0.0.1")
PORT = int(os.environ.get("VIDEO_POLISH_PORT", "8950"))

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
FFPROBE = os.environ.get(
    "FFPROBE", shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
)

# Reuse the known-good MLX environment that already powers oMLX Video Studio.
MLX_PYTHON = os.environ.get(
    "VIDEO_POLISH_MLX_PYTHON",
    str(Path.home() / "omlx-video" / ".venv" / "bin" / "python"),
)
WHISPER_MODEL = os.environ.get(
    "VIDEO_POLISH_WHISPER_MODEL",
    str(Path.home() / ".omlx" / "models" / "mlx-community" / "whisper-large-v3-turbo"),
)
DEEPFILTER_MODEL = os.environ.get(
    "VIDEO_POLISH_DEEPFILTER_MODEL",
    str(Path.home() / ".omlx" / "models" / "mlx-community" / "DeepFilterNet-mlx"),
)

OMLX_API_URL = os.environ.get("VIDEO_POLISH_OMLX_URL", "http://127.0.0.1:8000")
OMLX_API_KEY = os.environ.get("VIDEO_POLISH_OMLX_API_KEY", "1234")
OMLX_LLM_MODEL = os.environ.get(
    "VIDEO_POLISH_LLM_MODEL", "Qwen3.8-27B-8bit"
)

MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO_POLISH_MAX_UPLOAD_BYTES", str(250 * 1024**3)))
ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
    ".webm",
    ".mts",
    ".m2ts",
}
