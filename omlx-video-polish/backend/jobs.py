"""Small persistent job registry for analysis and render work."""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import JOB_DIR


class JobRegistry:
    def __init__(self, job_dir: Path = JOB_DIR):
        self.job_dir = job_dir
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, kind: str) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        state = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "stage": "Waiting",
            "progress": 0,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = state
            self._persist(state)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                path = self.job_dir / f"{job_id}.json"
                if path.exists():
                    try:
                        state = json.loads(path.read_text(encoding="utf-8"))
                        self._jobs[job_id] = state
                    except (OSError, json.JSONDecodeError):
                        return None
            return json.loads(json.dumps(state)) if state else None

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            state = self._jobs[job_id]
            state.update(changes)
            state["updated_at"] = time.time()
            self._persist(state)

    def progress(self, job_id: str, progress: int, stage: str, message: str | None = None) -> None:
        with self._lock:
            state = self._jobs[job_id]
            state["progress"] = max(0, min(100, int(progress)))
            state["stage"] = stage
            state["updated_at"] = time.time()
            if message:
                state["messages"] = (state.get("messages") or [])[-39:] + [message]
            self._persist(state)

    def run(self, job_id: str, target: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        thread = threading.Thread(
            target=self._runner,
            args=(job_id, target, args, kwargs),
            name=f"video-polish-{job_id}",
            daemon=True,
        )
        thread.start()

    def _runner(
        self,
        job_id: str,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.update(job_id, status="running", stage="Starting", progress=1)
        try:
            result = target(job_id, *args, **kwargs)
            self.update(
                job_id,
                status="complete",
                stage="Complete",
                progress=100,
                result=result,
            )
        except Exception as exc:  # keep the service alive and expose useful diagnostics
            error_detail = traceback.format_exc()
            (self.job_dir / f"{job_id}.error.log").write_text(error_detail, encoding="utf-8")
            self.update(
                job_id,
                status="error",
                stage="Failed",
                error=str(exc),
            )

    def _persist(self, state: dict[str, Any]) -> None:
        path = self.job_dir / f"{state['id']}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


JOBS = JobRegistry()

