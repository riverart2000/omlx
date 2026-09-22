"""Cross-service large-model lease and oMLX cold-cache helpers.

Large MLX models share the Mac's unified memory.  This module serializes local
image/video/song jobs across processes and treats model files on SSD as the
cold cache: unload oMLX models before a heavy job, then optionally reload them.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request

LOCK_PATH = os.environ.get("OMLX_MODEL_LEASE", "/Users/joebains/.omlx/model-memory.lock")
OMLX_URL = os.environ.get("OMLX_URL", "http://127.0.0.1:8000").rstrip("/")


def _write_owner(fd, owner: str, job_id: str) -> None:
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps({
        "owner": owner, "job_id": job_id, "pid": os.getpid(),
        "started": time.time(),
    }).encode())
    os.fsync(fd)


def acquire_lease(owner: str, job_id: str, timeout: float = 1800,
                  waiting=None):
    """Acquire the single cross-service heavy-model lease."""
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.time() + timeout
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_owner(fd, owner, job_id)
            return fd
        except BlockingIOError:
            if not announced and waiting:
                waiting()
                announced = True
            if time.time() >= deadline:
                os.close(fd)
                raise RuntimeError("timed out waiting for another local model job to finish")
            time.sleep(1)


def release_lease(fd) -> None:
    if fd is None:
        return
    try:
        os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def lease_status() -> dict:
    try:
        with open(LOCK_PATH) as f:
            status = json.load(f)
    except Exception:
        return {}
    pid = int(status.get("pid") or 0)
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # A hard stop releases the OS flock automatically, but can leave
            # the old descriptive JSON behind. Do not present that dead job as
            # active to the UI.
            return {}
        except PermissionError:
            pass
    return status


def available_memory_gb() -> float:
    """Best-effort genuinely reclaimable memory from vm_stat."""
    try:
        raw = subprocess.check_output(["vm_stat"], text=True, timeout=3)
        page = 4096
        if "page size of " in raw:
            page = int(raw.split("page size of ", 1)[1].split(" bytes", 1)[0])
        values = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                values[key] = int(val.strip().rstrip("."))
            except ValueError:
                pass
        pages = sum(values.get(k, 0) for k in (
            "Pages free", "Pages inactive", "Pages speculative",
            "Pages purgeable"))
        return pages * page / 1e9
    except Exception:
        return 0.0


def pressure_available_memory_gb() -> float:
    """Memory macOS says is available after accounting for compression.

    ``vm_stat`` alone is deliberately conservative, but on unified-memory Macs
    it can substantially under-count memory that the kernel can recover from
    the compressor.  ``memory_pressure -Q`` is the OS-level pressure signal and
    reports that additional headroom as a percentage of physical memory.
    """
    try:
        raw = subprocess.check_output(
            ["memory_pressure", "-Q"], text=True, timeout=4)
        match = re.search(r"memory free percentage:\s*(\d+)%", raw,
                          flags=re.IGNORECASE)
        if not match:
            return 0.0
        pct = max(0.0, min(100.0, float(match.group(1))))
        try:
            total = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True,
                timeout=3, stderr=subprocess.DEVNULL).strip())
        except Exception:
            total_match = re.search(r"system has\s+(\d+)", raw,
                                    flags=re.IGNORECASE)
            if not total_match:
                return 0.0
            total = int(total_match.group(1))
        return total / 1e9 * pct / 100.0
    except Exception:
        return 0.0


def _get(path: str, timeout=8) -> dict:
    req = urllib.request.Request(OMLX_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def _post(path: str, timeout=600) -> dict:
    req = urllib.request.Request(OMLX_URL + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def loaded_omlx_models() -> list[str]:
    try:
        status = _get("/v1/models/status")
    except Exception:
        return []
    return [m["id"] for m in status.get("models", [])
            if m.get("loaded") or m.get("is_loading")]


def unload_omlx_models(stage=None, timeout=45) -> list[str]:
    """Unload resident oMLX models and wait for their memory to be released."""
    ids = loaded_omlx_models()
    if not ids:
        return []
    if stage:
        stage("freeing memory: moving chat model to SSD cold cache…", 1)
    for model_id in ids:
        try:
            _post("/v1/models/" + urllib.parse.quote(model_id, safe="") + "/unload")
        except Exception:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not loaded_omlx_models():
            break
        time.sleep(1)
    time.sleep(1.5)
    return ids


def reload_omlx_models(ids: list[str], stage=None) -> None:
    if not ids:
        return
    if stage:
        stage("reloading chat model from SSD…", 98)
    for model_id in ids:
        try:
            _post("/v1/models/" + urllib.parse.quote(model_id, safe="") + "/load")
        except Exception:
            pass
