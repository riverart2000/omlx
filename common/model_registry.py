"""Shared, SQLite-backed registry for oMLX media models.

The registry contains public model metadata and capabilities only. API keys stay
in ``~/.omlx/.env`` and execution remains inside the owning service.  Every
consumer receives the same stable model ids, while ``service_engine`` and
``aliases`` preserve compatibility with older UI/request names.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_PATH = Path(os.environ.get(
    "OMLX_MODEL_REGISTRY",
    "/Users/joebains/.omlx/studio-models.sqlite3",
))
_LOCK = threading.RLock()
RETIRED_BUILTIN_IMAGE_MODELS = ("hidream", "flux")


def _default_image_models() -> list[dict]:
    return [
        {
            "id": "grok",
            "label": "Grok Imagine · xAI cloud",
            "driver": "xai_image",
            "model": os.environ.get("GROK_IMAGE_MODEL", "grok-imagine-image"),
            "description": (
                "Cloud image generation with strong prompt following and native "
                "signage/text rendering. Supports up to three references."
            ),
            "sort_order": 10,
            "consumers": ["image_studio", "kindle", "orchestrator"],
            "aliases": [],
            "capabilities": {
                "text_to_image": True, "multi_reference": True,
                "max_reference_images": 3, "native_text_rendering": True,
                "cloud": True, "provider": "xAI", "fixed_resolution": "1K",
                "aspect_ratios": ["1:1", "3:4", "4:3"],
                "output_formats": ["png"], "settings_profile": "standard",
            },
            "settings": {"resolution": "1k", "output_format": "png"},
        },
        {
            "id": "nano_banana_2",
            "label": "Nano Banana 2 · Replicate",
            "driver": "replicate_nano_banana",
            "model": os.environ.get(
                "REPLICATE_NANO_BANANA_MODEL", "google/nano-banana-2"),
            "description": (
                "Replicate cloud generation at 1K with up to fourteen references, "
                "native text/signage, and optional search grounding."
            ),
            "sort_order": 40,
            "consumers": ["image_studio", "kindle", "orchestrator"],
            "aliases": [],
            "capabilities": {
                "text_to_image": True, "multi_reference": True,
                "max_reference_images": 14, "native_text_rendering": True,
                "cloud": True, "provider": "Replicate",
                "fixed_resolution": "1K", "settings_profile": "nano_banana",
                "aspect_ratios": [
                    "match_input_image", "1:1", "1:4", "1:8", "2:3", "3:2",
                    "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16",
                    "16:9", "21:9",
                ],
                "output_formats": ["png", "jpg"],
                "google_search": True, "image_search": True,
            },
            "settings": {"resolution": "1K", "output_format": "png"},
        },
        {
            "id": "flux_2_klein_4b",
            "label": "FLUX 2 Klein 4B · Replicate",
            "driver": "replicate_flux2_klein",
            "model": os.environ.get(
                "REPLICATE_FLUX2_KLEIN_MODEL",
                "black-forest-labs/flux-2-klein-4b"),
            "description": (
                "Fast Replicate cloud generation with up to five ordered references. "
                "Text, labels and signage are strongly discouraged."
            ),
            "sort_order": 50,
            "consumers": ["image_studio", "kindle", "orchestrator"],
            "aliases": ["flux_2_pro"],
            "capabilities": {
                "text_to_image": True, "multi_reference": True,
                "max_reference_images": 5, "native_text_rendering": False,
                "cloud": True, "provider": "Replicate",
                "settings_profile": "flux2",
                "resolutions": ["0.25", "0.5", "1", "2", "4"],
                "aspect_ratios": [
                    "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4",
                    "5:4", "4:5", "21:9", "9:21", "match_input_image",
                ],
                "output_formats": ["webp", "jpg", "png"],
            },
            "settings": {
                "resolution": "1", "output_format": "png",
                "output_quality": 100, "go_fast": False,
            },
        },
        {
            "id": "flux_2_klein_4b_local",
            "label": "FLUX 2 Klein 4B · local MLX",
            "driver": "mflux_flux2_klein",
            "model": "black-forest-labs/FLUX.2-klein-4B · mflux full precision",
            "description": (
                "Maximum-quality local full-precision generation at up to 1K with "
                "up to three ordered references, a 1,024-token prompt window, four "
                "intended MLX steps and SSD safeguards."
            ),
            "sort_order": 60,
            "consumers": ["image_studio", "kindle", "orchestrator"],
            "aliases": [],
            "capabilities": {
                "text_to_image": True, "multi_reference": True,
                "max_reference_images": 3, "native_text_rendering": False,
                "negative_prompt": False, "cloud": False,
                "provider": "Local MLX", "settings_profile": "flux2_local",
                "resolutions": ["1"],
                "aspect_ratios": [
                    "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4",
                    "5:4", "4:5", "21:9", "9:21", "match_input_image",
                ],
                "output_formats": ["png"], "steps": 4,
            },
            "settings": {
                "resolution": "1", "output_format": "png",
                "output_quality": 100, "steps": 4,
            },
        },
    ]


def _connect() -> sqlite3.Connection:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(REGISTRY_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=10000")
    return con


def ensure_registry() -> Path:
    """Create the schema and add any new built-in models without overwriting edits."""
    with _LOCK, _connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS media_models (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                driver TEXT NOT NULL,
                service_engine TEXT NOT NULL,
                model TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                sort_order INTEGER NOT NULL DEFAULT 100,
                builtin INTEGER NOT NULL DEFAULT 0,
                consumers_json TEXT NOT NULL DEFAULT '[]',
                aliases_json TEXT NOT NULL DEFAULT '[]',
                capabilities_json TEXT NOT NULL DEFAULT '{}',
                settings_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS media_models_kind_enabled "
                    "ON media_models(kind, enabled, sort_order)")
        now = datetime.now(timezone.utc).isoformat()
        for item in _default_image_models():
            con.execute("""
                INSERT OR IGNORE INTO media_models (
                    id, kind, label, driver, service_engine, model, description,
                    enabled, sort_order, builtin, consumers_json, aliases_json,
                    capabilities_json, settings_json, created_at, updated_at
                ) VALUES (?, 'image', ?, ?, ?, ?, ?, 1, ?, 1, ?, ?, ?, ?, ?, ?)
            """, (
                item["id"], item["label"], item["driver"],
                item.get("service_engine") or item["id"], item["model"],
                item.get("description", ""), item.get("sort_order", 100),
                json.dumps(item.get("consumers", [])),
                json.dumps(item.get("aliases", [])),
                json.dumps(item.get("capabilities", {}), sort_keys=True),
                json.dumps(item.get("settings", {}), sort_keys=True), now, now,
            ))

        # Retired models must disappear from existing SQLite registries too;
        # removing them only from the defaults would leave older rows enabled.
        con.executemany(
            "DELETE FROM media_models WHERE id = ? AND kind = 'image'",
            ((model_id,) for model_id in RETIRED_BUILTIN_IMAGE_MODELS),
        )

        # Safety-critical built-in limits must also reach registries created by an
        # older oMLX release.  Keep the user's enabled/disabled choice and any
        # other registry customisation intact; only tighten the local FLUX
        # reference ceiling and refresh the matching explanatory text.
        local_flux = next(
            item for item in _default_image_models()
            if item["id"] == "flux_2_klein_4b_local"
        )
        row = con.execute(
            "SELECT capabilities_json, description FROM media_models "
            "WHERE id = ? AND builtin = 1",
            (local_flux["id"],),
        ).fetchone()
        if row:
            try:
                capabilities = json.loads(row["capabilities_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                capabilities = {}
            stale_description = "1,500-token prompt window" in str(
                row["description"] or "")
            if (capabilities.get("max_reference_images") != 3
                    or stale_description):
                capabilities["max_reference_images"] = 3
                con.execute(
                    "UPDATE media_models SET description = ?, "
                    "capabilities_json = ?, updated_at = ? WHERE id = ?",
                    (
                        local_flux["description"],
                        json.dumps(capabilities, sort_keys=True),
                        now,
                        local_flux["id"],
                    ),
                )
        con.commit()
    return REGISTRY_PATH


def _decode_row(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["enabled"] = bool(item["enabled"])
    item["builtin"] = bool(item["builtin"])
    for column, key, fallback in (
        ("consumers_json", "consumers", []),
        ("aliases_json", "aliases", []),
        ("capabilities_json", "capabilities", {}),
        ("settings_json", "settings", {}),
    ):
        try:
            item[key] = json.loads(item.pop(column) or json.dumps(fallback))
        except (TypeError, json.JSONDecodeError):
            item[key] = fallback
    return item


def list_models(kind: str = "image", consumer: str | None = None,
                enabled_only: bool = True) -> list[dict]:
    ensure_registry()
    query = "SELECT * FROM media_models WHERE kind = ?"
    args: list[object] = [kind]
    if enabled_only:
        query += " AND enabled = 1"
    query += " ORDER BY sort_order, label COLLATE NOCASE"
    with _LOCK, _connect() as con:
        models = [_decode_row(row) for row in con.execute(query, args)]
    if consumer:
        models = [m for m in models if not m["consumers"] or consumer in m["consumers"]]
    return models


def get_model(model_id: str, kind: str = "image",
              enabled_only: bool = True) -> dict | None:
    wanted = str(model_id or "").strip().lower()
    for model in list_models(kind=kind, enabled_only=enabled_only):
        names = [model["id"], model.get("service_engine", ""), *model["aliases"]]
        if wanted in {str(name).strip().lower() for name in names if name}:
            return model
    return None


def public_models(kind: str = "image", consumer: str | None = None) -> list[dict]:
    """JSON-safe registry rows for UIs and service discovery endpoints."""
    return list_models(kind=kind, consumer=consumer, enabled_only=True)


def upsert_model(model: dict) -> dict:
    """Add/update a registry row. Intended for future admin UI or migration tools."""
    model_id = str(model.get("id") or "").strip().lower()
    if not model_id:
        raise ValueError("model id is required")
    existing = get_model(model_id, kind=str(model.get("kind") or "image"),
                         enabled_only=False) or {}
    merged = {**existing, **model, "id": model_id}
    now = datetime.now(timezone.utc).isoformat()
    with _LOCK, _connect() as con:
        con.execute("""
            INSERT INTO media_models (
                id, kind, label, driver, service_engine, model, description,
                enabled, sort_order, builtin, consumers_json, aliases_json,
                capabilities_json, settings_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind, label=excluded.label, driver=excluded.driver,
                service_engine=excluded.service_engine, model=excluded.model,
                description=excluded.description, enabled=excluded.enabled,
                sort_order=excluded.sort_order, consumers_json=excluded.consumers_json,
                aliases_json=excluded.aliases_json,
                capabilities_json=excluded.capabilities_json,
                settings_json=excluded.settings_json, updated_at=excluded.updated_at
        """, (
            model_id, merged.get("kind", "image"), merged.get("label", model_id),
            merged.get("driver", model_id), merged.get("service_engine", model_id),
            merged.get("model", ""), merged.get("description", ""),
            int(bool(merged.get("enabled", True))), int(merged.get("sort_order", 100)),
            int(bool(merged.get("builtin", False))),
            json.dumps(merged.get("consumers", [])),
            json.dumps(merged.get("aliases", [])),
            json.dumps(merged.get("capabilities", {}), sort_keys=True),
            json.dumps(merged.get("settings", {}), sort_keys=True),
            existing.get("created_at", now), now,
        ))
        con.commit()
    return get_model(model_id, kind=merged.get("kind", "image"), enabled_only=False)


def set_enabled(model_id: str, enabled: bool) -> bool:
    ensure_registry()
    with _LOCK, _connect() as con:
        cur = con.execute(
            "UPDATE media_models SET enabled=?, updated_at=? WHERE id=?",
            (int(bool(enabled)), datetime.now(timezone.utc).isoformat(), model_id),
        )
        con.commit()
        return cur.rowcount > 0


ensure_registry()
