#!/usr/bin/env python3
"""Check installed Hugging Face models for newer repository revisions.

This is intentionally a read-only checker.  Model downloads are large and a
new revision can break a working pipeline, so this script reports updates but
never changes model files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOME = Path.home()
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", HOME / ".cache/huggingface/hub"))
OMLX_MODELS = Path(os.environ.get("OMLX_MODEL_DIR", HOME / ".omlx/models"))
STATUS_FILE = Path(
    os.environ.get("OMLX_MODEL_UPDATE_STATUS", HOME / ".omlx/model-update-status.json")
)
API_ROOT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def revision_from_metadata(model_dir: Path) -> tuple[str | None, list[str]]:
    """Return the dominant revision recorded by snapshot_download metadata."""
    download_dir = model_dir / ".cache/huggingface/download"
    revisions: list[str] = []
    if download_dir.is_dir():
        for metadata in download_dir.rglob("*.metadata"):
            try:
                first_line = metadata.read_text(errors="replace").splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if len(first_line) >= 7:
                revisions.append(first_line)
    if not revisions:
        return None, []
    counts = Counter(revisions)
    return counts.most_common(1)[0][0], sorted(counts)


def contains_weights(model_dir: Path) -> bool:
    """Exclude README-only and interrupted downloads from the daily report."""
    try:
        return any(
            path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
            for path in model_dir.rglob("*")
        )
    except OSError:
        return False


def installed_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    if HF_HUB.is_dir():
        for cache_dir in sorted(HF_HUB.glob("models--*--*")):
            encoded = cache_dir.name.removeprefix("models--")
            parts = encoded.split("--", 1)
            if len(parts) != 2:
                continue
            ref = cache_dir / "refs/main"
            try:
                revision = ref.read_text().strip()
            except OSError:
                continue
            snapshot_dir = cache_dir / "snapshots" / revision
            if not snapshot_dir.is_dir() or not contains_weights(snapshot_dir):
                continue
            models.append(
                {
                    "repo": f"{parts[0]}/{parts[1]}",
                    "location": str(cache_dir),
                    "installation": "huggingface_cache",
                    "local_revision": revision,
                    "local_revisions": [revision],
                }
            )

    if OMLX_MODELS.is_dir():
        for owner_dir in sorted(p for p in OMLX_MODELS.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in owner_dir.iterdir() if p.is_dir()):
                revision, revisions = revision_from_metadata(model_dir)
                if not revision or not contains_weights(model_dir):
                    continue
                models.append(
                    {
                        "repo": f"{owner_dir.name}/{model_dir.name}",
                        "location": str(model_dir),
                        "installation": "omlx_models",
                        "local_revision": revision,
                        "local_revisions": revisions,
                    }
                )
    return models


def latest_revision(repo: str, timeout: float = 20.0) -> tuple[str | None, str | None]:
    url = f"{API_ROOT}/api/models/{urllib.parse.quote(repo, safe='/')}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "oMLX-model-update-checker/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        revision = str(payload.get("sha") or "").strip()
        return (revision or None), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)


def load_previous() -> dict[str, Any]:
    try:
        return json.loads(STATUS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_report(report: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="model-update-", suffix=".json", dir=STATUS_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, STATUS_FILE)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def notify(title: str, message: str) -> None:
    # AppleScript receives arguments separately, so model names cannot become code.
    script = 'on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run'
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e", script, title, message],
            check=False,
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def run(send_notification: bool) -> int:
    previous = load_previous()
    checked_at = utc_now()
    installs = installed_models()
    remote_by_repo: dict[str, tuple[str | None, str | None]] = {}

    for model in installs:
        repo = model["repo"]
        if repo not in remote_by_repo:
            remote_by_repo[repo] = latest_revision(repo)
        remote, error = remote_by_repo[repo]
        model["remote_revision"] = remote
        if error:
            model["status"] = "check_failed"
            model["error"] = error
        elif remote == model["local_revision"]:
            model["status"] = "current"
        else:
            model["status"] = "update_available"

    updates = [m for m in installs if m["status"] == "update_available"]
    failures = [m for m in installs if m["status"] == "check_failed"]
    report = {
        "checked_at": checked_at,
        "checker": "read_only",
        "summary": {
            "installations": len(installs),
            "repositories": len(remote_by_repo),
            "current": sum(m["status"] == "current" for m in installs),
            "updates_available": len(updates),
            "checks_failed": len(failures),
        },
        "models": installs,
    }
    save_report(report)

    print(
        f"Checked {len(installs)} model installations: "
        f"{len(updates)} update(s), {len(failures)} check failure(s)."
    )
    for model in updates:
        print(f"UPDATE  {model['repo']}  ({model['installation']})")
    for model in failures:
        print(f"ERROR   {model['repo']}: {model['error']}")

    if send_notification and updates:
        previous_updates = {
            (m.get("repo"), m.get("location"), m.get("remote_revision"))
            for m in previous.get("models", [])
            if m.get("status") == "update_available"
        }
        current_updates = {
            (m["repo"], m["location"], m["remote_revision"]) for m in updates
        }
        if current_updates != previous_updates:
            names = sorted({m["repo"] for m in updates})
            preview = ", ".join(names[:3])
            if len(names) > 3:
                preview += f" and {len(names) - 3} more"
            notify("oMLX model updates available", preview)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notify",
        action="store_true",
        help="show one macOS notification when the available-update set changes",
    )
    args = parser.parse_args()
    return run(args.notify)


if __name__ == "__main__":
    sys.exit(main())
