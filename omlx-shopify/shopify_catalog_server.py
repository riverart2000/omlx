#!/usr/bin/env python3
"""Shopify catalogue mirror and reviewed Grok rewrite studio for oMLX Chat."""
from __future__ import annotations

import json
import html
import math
import os
import re
import sqlite3
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from growth import GrowthService
from growth.changes import (
    PUBLISHABLE_FIELDS, expected_product_state, select_fields,
    verification_mismatches,
)
from providers import GrokModelResolver, GrokPromptCache


HOST = os.environ.get("OMLX_SHOPIFY_HOST", "127.0.0.1")
PORT = int(os.environ.get("OMLX_SHOPIFY_PORT", "8900"))
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.environ.get(
    "OMLX_SHOPIFY_DB", str(BASE_DIR / "data" / "shopify_catalog.sqlite3")))
ENV_PATHS = [
    Path(os.environ.get(
        "OMLX_SHOPIFY_ENV_FILE", "/Users/joebains/shopify-ai-blog-system/.env")),
    Path(os.environ.get("OMLX_ENV_FILE", "/Users/joebains/.omlx/.env")),
]
API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2026-07")
GROK_MODEL = os.environ.get("GROK_PRODUCT_MODEL", "latest")
XAI_BASE = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
GROK_RESOLVER = GrokModelResolver(
    XAI_BASE, GROK_MODEL,
    fallback_model=os.environ.get("GROK_FALLBACK_MODEL", "grok-4.6"),
)
GROK_CACHE = GrokPromptCache(os.environ.get(
    "GROK_PROMPT_CACHE_KEY", "omlx-shopify"))
GROK_REWRITE_WORKLOAD = "catalogue-rewrite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
BACKUP_DIR = DB_PATH.parent / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

SYNC_LOCK = threading.Lock()
SYNC_STATE_LOCK = threading.Lock()
SYNC_STATE = {
    "running": False,
    "phase": "idle",
    "message": "Ready to sync",
    "pages": 0,
    "products_seen": 0,
    "variants_seen": 0,
    "started_at": "",
    "finished_at": "",
    "elapsed_seconds": 0.0,
    "error": "",
}
REWRITE_LOCK = threading.Lock()
REWRITE_STATE_LOCK = threading.Lock()
REWRITE_STATE = {
    "running": False, "phase": "idle", "message": "Ready",
    "total": 0, "completed": 0, "current_product_id": "",
    "started_at": "", "finished_at": "", "elapsed_seconds": 0.0,
    "error": "", "errors": [],
}
PUBLISH_LOCK = threading.Lock()
PUBLISH_STATE_LOCK = threading.Lock()
PUBLISH_STATE = {
    "running": False, "phase": "idle", "message": "Ready",
    "total": 0, "completed": 0, "succeeded": 0, "failed": 0,
    "current_product_id": "", "started_at": "", "finished_at": "",
    "elapsed_seconds": 0.0, "error": "", "errors": [],
}
GROWTH_SERVICE: GrowthService | None = None
GROWTH_SERVICE_LOCK = threading.Lock()


VARIANTS_QUERY = """
query CatalogueVariants($cursor: String) {
  shop { name myshopifyDomain currencyCode }
  productVariants(first: 100, after: $cursor, sortKey: ID) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id legacyResourceId title displayName sku barcode price compareAtPrice
      inventoryQuantity availableForSale taxable inventoryPolicy createdAt updatedAt
      selectedOptions { name value }
      inventoryItem {
        id legacyResourceId sku tracked requiresShipping
        unitCost { amount currencyCode }
        measurement { weight { value unit } }
        countryCodeOfOrigin harmonizedSystemCode
      }
      product {
        id legacyResourceId handle title description descriptionHtml vendor
        productType status tags createdAt updatedAt publishedAt onlineStoreUrl
        totalInventory hasOnlyDefaultVariant
        featuredMedia { preview { image { url altText } } }
        collections(first: 50) { nodes { id title handle } }
      }
    }
  }
}
"""

COLLECTIONS_QUERY = """
query CatalogueCollections($cursor: String) {
  collections(first: 100, after: $cursor, sortKey: TITLE) {
    pageInfo { hasNextPage endCursor }
    nodes { id legacyResourceId title handle updatedAt ruleSet { appliedDisjunctively } }
  }
}
"""

REVIEWS_QUERY = """
query CatalogueReviews($cursor: String) {
  products(first: 100, after: $cursor, sortKey: ID) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      metafields(first: 10, namespace: "judgeme") {
        nodes { key value }
      }
    }
  }
}
"""

PRODUCT_UPDATE_MUTATION = """
mutation RewriteProduct($product: ProductUpdateInput!) {
  productUpdate(product: $product) {
    product { id title descriptionHtml productType tags updatedAt }
    userErrors { field message }
  }
}
"""

PRODUCT_CURRENT_QUERY = """
query CurrentProductForChange($id: ID!) {
  product(id: $id) {
    id title descriptionHtml productType tags updatedAt
    collections(first: 100) { nodes { id title } }
  }
}
"""

COLLECTION_ADD_MUTATION = """
mutation AddProductToCollection($id: ID!, $productIds: [ID!]!) {
  collectionAddProductsV2(id: $id, productIds: $productIds) {
    job { id done }
    userErrors { field message }
  }
}
"""

COLLECTION_REMOVE_MUTATION = """
mutation RemoveProductFromCollection($id: ID!, $productIds: [ID!]!) {
  collectionRemoveProducts(id: $id, productIds: $productIds) {
    job { id done }
    userErrors { field message }
  }
}
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_path in ENV_PATHS:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.strip()
            if not value or value.startswith("#") or "=" not in value:
                continue
            key, raw = value.split("=", 1)
            values.setdefault(key.strip(), raw.strip().strip('"').strip("'"))
    values.update(os.environ)
    return values


def shop_config() -> tuple[str, str, str, str]:
    env = load_env()
    domain = str(env.get("MYSHOPIFY_DOMAIN") or
                 env.get("SHOPIFY_STOREFRONT_DOMAIN") or "")
    domain = re.sub(r"^https?://", "", domain).rstrip("/")
    client_id = str(env.get("SHOPIFY_CLIENT_ID") or "")
    client_secret = str(env.get("SHOPIFY_CLIENT_SECRET") or
                        env.get("SHOPIFY_API_SECRET") or "")
    fallback = str(env.get("SHOPIFY_ADMIN_ACCESS_TOKEN") or
                   env.get("SHOPIFY_ACCESS_TOKEN") or
                   env.get("SHOPIFY_APP_AUTOMATION_TOKEN") or "")
    if not domain:
        raise RuntimeError("MYSHOPIFY_DOMAIN is not configured in ~/.omlx/.env")
    if not (client_id and client_secret) and not fallback:
        raise RuntimeError(
            "Shopify Admin credentials are not configured in ~/.omlx/.env")
    return domain, client_id, client_secret, fallback


def access_token(domain: str, client_id: str, client_secret: str,
                 fallback: str) -> str:
    if client_id and client_secret:
        payload = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode()
        request = urllib.request.Request(
            f"https://{domain}/admin/oauth/access_token", data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                token = str(json.load(response).get("access_token") or "")
            if token:
                return token
        except Exception:
            if not fallback:
                raise
    if fallback:
        return fallback
    raise RuntimeError("Shopify did not issue an Admin API access token")


def graphql(domain: str, token: str, query: str, variables: dict) -> dict:
    request = urllib.request.Request(
        f"https://{domain}/admin/api/{API_VERSION}/graphql.json",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": token,
            "User-Agent": "oMLX-Shopify-Catalogue/1.0",
        })
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        if exc.code in (401, 403):
            raise RuntimeError(
                "Shopify rejected the Admin API credentials. Reconnect the app "
                "and grant read_products and read_inventory access.") from exc
        raise RuntimeError(f"Shopify returned HTTP {exc.code}: {detail}") from exc
    if payload.get("errors"):
        details = "; ".join(str(row.get("message") or row)
                            for row in payload["errors"][:6])
        raise RuntimeError("Shopify GraphQL error: " + details)
    return payload


def connection() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS catalogue_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sync_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          started_at TEXT NOT NULL,
          finished_at TEXT,
          status TEXT NOT NULL,
          products_count INTEGER NOT NULL DEFAULT 0,
          variants_count INTEGER NOT NULL DEFAULT 0,
          error TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS products (
          shopify_id TEXT PRIMARY KEY,
          legacy_id TEXT NOT NULL DEFAULT '',
          handle TEXT NOT NULL DEFAULT '',
          title TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          description_html TEXT NOT NULL DEFAULT '',
          title_grok TEXT NOT NULL DEFAULT '',
          description_grok TEXT NOT NULL DEFAULT '',
          grok_updated_at TEXT NOT NULL DEFAULT '',
          review_count INTEGER NOT NULL DEFAULT 0,
          review_rating REAL NOT NULL DEFAULT 0,
          vendor TEXT NOT NULL DEFAULT '',
          product_type TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT '',
          tags_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL DEFAULT '',
          published_at TEXT NOT NULL DEFAULT '',
          online_store_url TEXT NOT NULL DEFAULT '',
          total_inventory INTEGER,
          has_only_default_variant INTEGER NOT NULL DEFAULT 0,
          featured_image_url TEXT NOT NULL DEFAULT '',
          featured_image_alt TEXT NOT NULL DEFAULT '',
          sync_marker TEXT NOT NULL,
          synced_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_products_title ON products(title);
        CREATE INDEX IF NOT EXISTS idx_products_status ON products(status);
        CREATE TABLE IF NOT EXISTS variants (
          shopify_id TEXT PRIMARY KEY,
          product_id TEXT NOT NULL REFERENCES products(shopify_id) ON DELETE CASCADE,
          inventory_item_id TEXT NOT NULL DEFAULT '',
          legacy_id TEXT NOT NULL DEFAULT '',
          title TEXT NOT NULL DEFAULT '',
          display_name TEXT NOT NULL DEFAULT '',
          sku TEXT NOT NULL DEFAULT '',
          barcode TEXT NOT NULL DEFAULT '',
          price REAL,
          compare_at_price REAL,
          cost REAL,
          currency TEXT NOT NULL DEFAULT '',
          inventory_quantity INTEGER,
          available_for_sale INTEGER NOT NULL DEFAULT 0,
          taxable INTEGER NOT NULL DEFAULT 0,
          inventory_policy TEXT NOT NULL DEFAULT '',
          tracked INTEGER NOT NULL DEFAULT 0,
          requires_shipping INTEGER NOT NULL DEFAULT 0,
          weight REAL,
          weight_unit TEXT NOT NULL DEFAULT '',
          country_of_origin TEXT NOT NULL DEFAULT '',
          harmonized_system_code TEXT NOT NULL DEFAULT '',
          selected_options_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL DEFAULT '',
          sync_marker TEXT NOT NULL,
          synced_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_variants_product ON variants(product_id);
        CREATE INDEX IF NOT EXISTS idx_variants_sku ON variants(sku);
        CREATE TABLE IF NOT EXISTS collections (
          shopify_id TEXT PRIMARY KEY,
          legacy_id TEXT NOT NULL DEFAULT '',
          title TEXT NOT NULL,
          handle TEXT NOT NULL DEFAULT '',
          is_smart INTEGER NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL DEFAULT '',
          sync_marker TEXT NOT NULL,
          synced_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS product_collections (
          product_id TEXT NOT NULL REFERENCES products(shopify_id) ON DELETE CASCADE,
          collection_id TEXT NOT NULL REFERENCES collections(shopify_id) ON DELETE CASCADE,
          PRIMARY KEY(product_id, collection_id)
        );
        CREATE TABLE IF NOT EXISTS rewrite_drafts (
          product_id TEXT PRIMARY KEY REFERENCES products(shopify_id) ON DELETE CASCADE,
          model TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'draft',
          original_json TEXT NOT NULL DEFAULT '{}',
          title TEXT NOT NULL DEFAULT '',
          hero_summary TEXT NOT NULL DEFAULT '',
          body_paragraphs_json TEXT NOT NULL DEFAULT '[]',
          lifestyle_close TEXT NOT NULL DEFAULT '',
          description_html TEXT NOT NULL DEFAULT '',
          product_type TEXT NOT NULL DEFAULT '',
          category TEXT NOT NULL DEFAULT '',
          collection_ids_json TEXT NOT NULL DEFAULT '[]',
          suggested_collections_json TEXT NOT NULL DEFAULT '[]',
          tags_json TEXT NOT NULL DEFAULT '[]',
          rationale TEXT NOT NULL DEFAULT '',
          warnings_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          published_at TEXT NOT NULL DEFAULT '',
          publish_error TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS rewrite_revisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          product_id TEXT NOT NULL,
          saved_at TEXT NOT NULL,
          reason TEXT NOT NULL,
          draft_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS grok_content_archive (
          product_id TEXT PRIMARY KEY,
          legacy_id TEXT NOT NULL DEFAULT '',
          handle TEXT NOT NULL DEFAULT '',
          source_title TEXT NOT NULL DEFAULT '',
          title_grok TEXT NOT NULL DEFAULT '',
          description_grok TEXT NOT NULL DEFAULT '',
          draft_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          published_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_grok_archive_handle
          ON grok_content_archive(handle);
        """)
        # Migrate existing catalogues without rebuilding or losing synced data.
        product_columns = {row[1] for row in db.execute("PRAGMA table_info(products)")}
        text_migrations = ("title_grok", "description_grok", "grok_updated_at")
        for column in text_migrations:
            if column not in product_columns:
                db.execute(
                    f"ALTER TABLE products ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
        if "review_count" not in product_columns:
            db.execute(
                "ALTER TABLE products ADD COLUMN review_count INTEGER NOT NULL DEFAULT 0")
        if "review_rating" not in product_columns:
            db.execute(
                "ALTER TABLE products ADD COLUMN review_rating REAL NOT NULL DEFAULT 0")
        # Seed the durable archive from drafts made before these dedicated fields
        # were introduced. The archive deliberately has no product foreign key,
        # so a Shopify deletion or DSers re-import cannot cascade-delete it.
        rows = db.execute("""
          SELECT r.*,p.legacy_id,p.handle,p.title AS source_title
          FROM rewrite_drafts r JOIN products p ON p.shopify_id=r.product_id
        """).fetchall()
        for row in rows:
            draft = draft_row_dict(row)
            db.execute("""
              INSERT INTO grok_content_archive(
                product_id,legacy_id,handle,source_title,title_grok,
                description_grok,draft_json,created_at,updated_at,published_at
              ) VALUES(?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(product_id) DO NOTHING
            """, (
                row["product_id"], row["legacy_id"], row["handle"],
                row["source_title"], row["title"], row["description_html"],
                json.dumps(draft, ensure_ascii=False), row["created_at"],
                row["updated_at"], row["published_at"],
            ))
            db.execute("""
              UPDATE products SET title_grok=?,description_grok=?,grok_updated_at=?
              WHERE shopify_id=?
            """, (row["title"], row["description_html"], row["updated_at"],
                  row["product_id"]))


def growth_service() -> GrowthService:
    """Return the lazily-created growth service.

    Lazy construction keeps imports and command-line diagnostics cheap while
    guaranteeing that the catalogue schema exists before growth migrations run.
    """
    global GROWTH_SERVICE
    if GROWTH_SERVICE is None:
        with GROWTH_SERVICE_LOCK:
            if GROWTH_SERVICE is None:
                GROWTH_SERVICE = GrowthService(
                    connection, graphql, shop_config, access_token)
    return GROWTH_SERVICE


def backup_database() -> Path | None:
    """Keep recoverable catalogue snapshots before remote synchronization."""
    if not DB_PATH.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = BACKUP_DIR / f"shopify-catalog-before-sync-{stamp}.sqlite3"
    source_db = connection()
    backup_db = sqlite3.connect(target)
    try:
        source_db.backup(backup_db)
    finally:
        backup_db.close()
        source_db.close()
    backups = sorted(BACKUP_DIR.glob("shopify-catalog-before-sync-*.sqlite3"))
    for old in backups[:-12]:
        old.unlink(missing_ok=True)
    return target


def number(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def set_state(**values) -> None:
    with SYNC_STATE_LOCK:
        SYNC_STATE.update(values)
        if SYNC_STATE.get("running") and SYNC_STATE.get("started_epoch"):
            SYNC_STATE["elapsed_seconds"] = round(
                time.time() - SYNC_STATE["started_epoch"], 1)


def state() -> dict:
    with SYNC_STATE_LOCK:
        result = dict(SYNC_STATE)
    result.pop("started_epoch", None)
    if result.get("running"):
        started = SYNC_STATE.get("started_epoch")
        if started:
            result["elapsed_seconds"] = round(time.time() - started, 1)
    return result


def set_rewrite_state(**values) -> None:
    with REWRITE_STATE_LOCK:
        REWRITE_STATE.update(values)
        started = REWRITE_STATE.get("started_epoch")
        if REWRITE_STATE.get("running") and started:
            REWRITE_STATE["elapsed_seconds"] = round(time.time() - started, 1)


def rewrite_state() -> dict:
    with REWRITE_STATE_LOCK:
        result = dict(REWRITE_STATE)
    started = result.pop("started_epoch", None)
    if result.get("running") and started:
        result["elapsed_seconds"] = round(time.time() - started, 1)
    return result


def set_publish_state(**values) -> None:
    with PUBLISH_STATE_LOCK:
        PUBLISH_STATE.update(values)
        started = PUBLISH_STATE.get("started_epoch")
        if PUBLISH_STATE.get("running") and started:
            PUBLISH_STATE["elapsed_seconds"] = round(time.time() - started, 1)


def publish_state() -> dict:
    with PUBLISH_STATE_LOCK:
        result = dict(PUBLISH_STATE)
    started = result.pop("started_epoch", None)
    if result.get("running") and started:
        result["elapsed_seconds"] = round(time.time() - started, 1)
    return result


def xai_keys() -> list[str]:
    keys = []
    for name in ("GROK_API_KEY", "XAI_API_KEY"):
        value = str(os.environ.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    for env_path in ENV_PATHS:
        if not env_path.exists():
            continue
        values = {}
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if raw and not raw.startswith("#") and "=" in raw:
                key, value = raw.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
        for name in ("GROK_API_KEY", "XAI_API_KEY"):
            value = str(values.get(name) or "").strip()
            if value and value not in keys:
                keys.append(value)
    if not keys:
        raise RuntimeError(
            "GROK_API_KEY is not configured in shopify-ai-blog-system/.env")
    return keys


def extract_grok_text(payload: dict) -> str:
    candidates = []
    if payload.get("output_text"):
        candidates.append(str(payload["output_text"]))
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("text"):
                candidates.append(str(content["text"]))
    choices = payload.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        if message.get("content"):
            candidates.append(str(message["content"]))
    return candidates[-1].strip() if candidates else ""


def parse_json_object(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(),
                     flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidates = [cleaned]
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Grok did not return one valid JSON object")


def grok_status() -> dict:
    """Return one UI-safe view of model selection and prompt-cache routing."""
    result = GROK_RESOLVER.status()
    result["prompt_cache"] = GROK_CACHE.status(GROK_REWRITE_WORKLOAD)
    return result


def grok_json(prompt: str, workload: str = GROK_REWRITE_WORKLOAD) -> dict:
    last_error = ""
    configured_keys = xai_keys()
    # Discovery supplies the newest concrete model ID. xAI currently advertises
    # a global `latest` alias that some Responses API accounts reject.
    GROK_RESOLVER.resolve(configured_keys)
    selected_model = GROK_RESOLVER.request_model
    payload = {
        "model": selected_model, "input": prompt, "store": False,
        "include": ["no_inline_citations"],
        "text": {"format": {"type": "json_object"}},
    }
    cache_key = GROK_CACHE.key(workload)
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    for key_index, api_key in enumerate(configured_keys):
        request = urllib.request.Request(
            XAI_BASE + "/responses", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key,
                     "User-Agent": "oMLX-Shopify-Catalogue/1.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    response_payload = json.load(response)
                    actual_model = str(response_payload.get("model") or selected_model)
                    GROK_RESOLVER.observe(actual_model)
                    result = parse_json_object(extract_grok_text(response_payload))
                    result["_grok_model_used"] = actual_model
                    return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = f"Grok returned HTTP {exc.code}: {detail}"
                # A different configured key may belong to a team with credit.
                if exc.code in (401, 403) and key_index < len(configured_keys) - 1:
                    break
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(last_error) from exc
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error or "Grok rewrite failed")


def clean_text(value, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def clean_list(value, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        text = clean_text(item, item_limit)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) >= limit:
            break
    return result


def description_html(draft: dict) -> str:
    pieces = []
    hero = clean_text(draft.get("hero_summary"), 420)
    if hero:
        pieces.append("<h2>" + html.escape(hero) + "</h2>")
    for paragraph in clean_list(draft.get("body_paragraphs"), 8, 1200):
        pieces.append("<p>" + html.escape(paragraph) + "</p>")
    close = clean_text(draft.get("lifestyle_close"), 400)
    if close:
        pieces.append("<p><strong>" + html.escape(close) + "</strong></p>")
    product_type = clean_text(draft.get("product_type"), 255)
    category = clean_text(draft.get("category"), 255)
    pieces.append("<p><strong>Product Type:</strong> " + html.escape(product_type) + "</p>")
    pieces.append("<p><strong>Category:</strong> " + html.escape(category) + "</p>")
    return "\n".join(pieces)


def product_for_rewrite(product_id: str) -> dict:
    with connection() as db:
        product = db.execute(
            "SELECT * FROM products WHERE shopify_id=?", (product_id,)).fetchone()
        if not product:
            raise ValueError("Product is not in the local catalogue")
        variants = [dict(row) for row in db.execute(
            "SELECT * FROM variants WHERE product_id=? ORDER BY title", (product_id,))]
        current_collections = [dict(row) for row in db.execute("""
          SELECT c.* FROM collections c JOIN product_collections pc
          ON pc.collection_id=c.shopify_id WHERE pc.product_id=? ORDER BY c.title
        """, (product_id,))]
        all_collections = [dict(row) for row in db.execute(
            "SELECT * FROM collections ORDER BY title")]
    value = dict(product)
    value["tags"] = json.loads(value.pop("tags_json") or "[]")
    value["variants"] = variants
    value["collections"] = current_collections
    value["all_collections"] = all_collections
    return value


def create_rewrite(product: dict) -> dict:
    current_collection_titles = [row["title"] for row in product["collections"]]
    available_collection_titles = [row["title"] for row in product["all_collections"]]
    variant_facts = [{
        "title": row.get("title"), "options": json.loads(
            row.get("selected_options_json") or "[]"),
        "requires_shipping": bool(row.get("requires_shipping")),
        "weight": row.get("weight"), "weight_unit": row.get("weight_unit"),
    } for row in product["variants"][:60]]
    source = {
        "title": product.get("title"), "description": product.get("description"),
        "vendor": product.get("vendor"), "product_type": product.get("product_type"),
        "tags": product.get("tags"), "current_collections": current_collection_titles,
        "variants": variant_facts,
    }
    result = grok_json("""You are the senior ecommerce copy director for Bio Luxe Lab,
a premium wellness brand. Rewrite this Shopify product accurately and elegantly.

STYLE AND STRUCTURE
- Create a compelling premium product title. The Shopify theme renders the title
  as the page H1, so return plain title text and never put an H1 in the description.
- Write a 2–3 line hero summary that leads with the realistic transformation and
  feeling. It will be rendered as H2.
- Then write about 200 words across normal paragraphs covering key benefits, how
  to use it, and what genuinely makes it different.
- Tone: spa-inspired, editorial, elevated, calm and specific—appropriate for a
  luxury wellness brand. Use natural British English, not generic AI copy.
- Close with one concise line placing the product in the customer's wellness lifestyle.
- Recommend one concise Shopify Product Type and one customer-facing Category.
- Return focused, commercially useful tags rather than keyword stuffing.

NON-NEGOTIABLE ACCURACY AND SAFETY
- Use only facts supported by the supplied Shopify product and variant data.
- Never invent materials, dimensions, certifications, provenance, ingredients,
  included accessories, guarantees, clinical evidence or capabilities.
- Do not make medical, diagnostic, treatment, cure, pain-relief, eyesight,
  weight-loss or guaranteed-result claims. Reframe unsupported health claims as
  neutral ritual, comfort, relaxation or general-wellbeing language.
- Do not preserve supplier spam, factory boilerplate, shipping promises, review
  requests, dropshipping language, repeated specifications or awkward model codes.
- If an essential fact is unclear, omit it and add a short warning for the editor.
- The body paragraphs must be prose, not headings, bullets, Markdown or HTML.

COLLECTION RULES
- Choose collection_titles only from the exact Available Shopify collections below.
- Keep a current collection when it remains relevant. Never invent an ID.
- Put useful collection ideas that do not already exist into suggested_new_collections.
  These are suggestions only and will not be created automatically.

Return ONLY one JSON object with exactly this structure:
{
  "title": "Plain premium product title",
  "hero_summary": "Two or three concise lines",
  "body_paragraphs": ["Paragraph", "Paragraph", "Paragraph"],
  "lifestyle_close": "One-line wellness lifestyle close",
  "product_type": "Concise Shopify product type",
  "category": "Customer-facing category",
  "collection_titles": ["Exact existing collection title"],
  "suggested_new_collections": ["Optional new collection idea"],
  "tags": ["focused tag"],
  "rationale": "Concise editorial reasoning",
  "warnings": ["Any unsupported or unclear source detail omitted"]
}

Available Shopify collections:
""" + json.dumps(available_collection_titles, ensure_ascii=False) +
        "\n\nSource product:\n" + json.dumps(source, ensure_ascii=False))

    title_map = {row["title"].casefold(): row["shopify_id"]
                 for row in product["all_collections"]}
    selected_ids = []
    for title in clean_list(result.get("collection_titles"), 20, 255):
        collection_id = title_map.get(title.casefold())
        if collection_id and collection_id not in selected_ids:
            selected_ids.append(collection_id)
    model_used = str(result.pop("_grok_model_used", "") or
                     grok_status().get("resolved") or GROK_MODEL)
    draft = {
        "product_id": product["shopify_id"], "model": model_used, "status": "draft",
        "original": source, "title": clean_text(result.get("title"), 255),
        "hero_summary": clean_text(result.get("hero_summary"), 420),
        "body_paragraphs": clean_list(result.get("body_paragraphs"), 8, 1200),
        "lifestyle_close": clean_text(result.get("lifestyle_close"), 400),
        "product_type": clean_text(result.get("product_type"), 255),
        "category": clean_text(result.get("category"), 255),
        "collection_ids": selected_ids,
        "suggested_collections": clean_list(
            result.get("suggested_new_collections"), 12, 255),
        "tags": clean_list(result.get("tags"), 30, 255),
        "rationale": clean_text(result.get("rationale"), 1200),
        "warnings": clean_list(result.get("warnings"), 12, 600),
    }
    if not draft["title"] or not draft["hero_summary"] or not draft["body_paragraphs"]:
        raise RuntimeError("Grok returned an incomplete product rewrite")
    draft["description_html"] = description_html(draft)
    return draft


def draft_row_dict(row: sqlite3.Row | None) -> dict | None:
    if not row:
        return None
    value = dict(row)
    for source, target, fallback in (
        ("body_paragraphs_json", "body_paragraphs", []),
        ("collection_ids_json", "collection_ids", []),
        ("suggested_collections_json", "suggested_collections", []),
        ("tags_json", "tags", []), ("warnings_json", "warnings", []),
        ("original_json", "original", {}),
    ):
        try:
            value[target] = json.loads(value.pop(source) or json.dumps(fallback))
        except (TypeError, json.JSONDecodeError):
            value[target] = fallback
    return value


def save_draft(draft: dict, reason="generated") -> dict:
    stamp = now()
    draft = dict(draft)
    draft["description_html"] = description_html(draft)
    with connection() as db:
        old = db.execute(
            "SELECT * FROM rewrite_drafts WHERE product_id=?",
            (draft["product_id"],)).fetchone()
        if old:
            db.execute("""
              INSERT INTO rewrite_revisions(product_id,saved_at,reason,draft_json)
              VALUES(?,?,?,?)
            """, (draft["product_id"], stamp, reason,
                  json.dumps(draft_row_dict(old), ensure_ascii=False)))
        db.execute("""
        INSERT INTO rewrite_drafts (
          product_id,model,status,original_json,title,hero_summary,
          body_paragraphs_json,lifestyle_close,description_html,product_type,
          category,collection_ids_json,suggested_collections_json,tags_json,
          rationale,warnings_json,created_at,updated_at,published_at,publish_error
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(product_id) DO UPDATE SET
          model=excluded.model,status=excluded.status,original_json=excluded.original_json,
          title=excluded.title,hero_summary=excluded.hero_summary,
          body_paragraphs_json=excluded.body_paragraphs_json,
          lifestyle_close=excluded.lifestyle_close,
          description_html=excluded.description_html,product_type=excluded.product_type,
          category=excluded.category,collection_ids_json=excluded.collection_ids_json,
          suggested_collections_json=excluded.suggested_collections_json,
          tags_json=excluded.tags_json,rationale=excluded.rationale,
          warnings_json=excluded.warnings_json,updated_at=excluded.updated_at,
          publish_error=''
        """, (
            draft["product_id"], str(draft.get("model") or GROK_MODEL),
            str(draft.get("status") or "draft"),
            json.dumps(draft.get("original") or {}, ensure_ascii=False),
            clean_text(draft.get("title"), 255),
            clean_text(draft.get("hero_summary"), 420),
            json.dumps(clean_list(draft.get("body_paragraphs"), 8, 1200),
                       ensure_ascii=False),
            clean_text(draft.get("lifestyle_close"), 400),
            draft["description_html"], clean_text(draft.get("product_type"), 255),
            clean_text(draft.get("category"), 255),
            json.dumps(draft.get("collection_ids") or []),
            json.dumps(clean_list(draft.get("suggested_collections"), 12, 255),
                       ensure_ascii=False),
            json.dumps(clean_list(draft.get("tags"), 30, 255), ensure_ascii=False),
            clean_text(draft.get("rationale"), 1200),
            json.dumps(clean_list(draft.get("warnings"), 12, 600),
                       ensure_ascii=False), stamp, stamp,
            str(draft.get("published_at") or ""), "",
        ))
        saved = draft_row_dict(db.execute(
            "SELECT * FROM rewrite_drafts WHERE product_id=?",
            (draft["product_id"],)).fetchone())
        product = db.execute("""
          SELECT legacy_id,handle,title FROM products WHERE shopify_id=?
        """, (draft["product_id"],)).fetchone()
        if product:
            db.execute("""
              UPDATE products SET title_grok=?,description_grok=?,grok_updated_at=?
              WHERE shopify_id=?
            """, (saved["title"], saved["description_html"], stamp,
                  draft["product_id"]))
            db.execute("""
              INSERT INTO grok_content_archive(
                product_id,legacy_id,handle,source_title,title_grok,
                description_grok,draft_json,created_at,updated_at,published_at
              ) VALUES(?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(product_id) DO UPDATE SET
                legacy_id=excluded.legacy_id,handle=excluded.handle,
                source_title=excluded.source_title,title_grok=excluded.title_grok,
                description_grok=excluded.description_grok,
                draft_json=excluded.draft_json,updated_at=excluded.updated_at,
                published_at=excluded.published_at
            """, (
                draft["product_id"], product["legacy_id"], product["handle"],
                product["title"], saved["title"], saved["description_html"],
                json.dumps(saved, ensure_ascii=False), saved["created_at"], stamp,
                str(saved.get("published_at") or ""),
            ))
        db.commit()
        return saved


def rewrite_worker(product_ids: list[str]) -> None:
    if not REWRITE_LOCK.acquire(blocking=False):
        return
    epoch = time.time()
    errors = []
    set_rewrite_state(
        running=True, phase="generating", total=len(product_ids), completed=0,
        current_product_id="", message="Starting Grok product rewrites",
        started_at=now(), started_epoch=epoch, finished_at="", error="", errors=[])
    try:
        for index, product_id in enumerate(product_ids):
            try:
                product = product_for_rewrite(product_id)
                set_rewrite_state(
                    current_product_id=product_id,
                    message=f"Grok is rewriting {index + 1} of {len(product_ids)}: {product['title']}")
                save_draft(create_rewrite(product), "before-regeneration")
            except Exception as exc:
                errors.append({"product_id": product_id,
                               "error": f"{type(exc).__name__}: {exc}"})
            set_rewrite_state(completed=index + 1, errors=errors)
        set_rewrite_state(
            running=False, phase="complete" if not errors else "complete_with_errors",
            message=(f"Created {len(product_ids)-len(errors)} of {len(product_ids)} drafts"),
            current_product_id="", finished_at=now(),
            elapsed_seconds=round(time.time()-epoch, 1), errors=errors,
            error=(errors[0]["error"] if errors else ""))
    except Exception as exc:
        set_rewrite_state(running=False, phase="failed", message="Rewrite failed",
                          current_product_id="", finished_at=now(),
                          elapsed_seconds=round(time.time()-epoch, 1),
                          error=f"{type(exc).__name__}: {exc}", errors=errors)
    finally:
        REWRITE_LOCK.release()


def load_draft(product_id: str) -> dict:
    with connection() as db:
        draft = draft_row_dict(db.execute(
            "SELECT * FROM rewrite_drafts WHERE product_id=?", (product_id,)).fetchone())
    if not draft:
        raise ValueError("Generate a Grok rewrite draft first")
    return draft


def update_draft_from_editor(product_id: str, data: dict) -> dict:
    old = load_draft(product_id)
    valid_collection_ids = set()
    with connection() as db:
        valid_collection_ids = {row[0] for row in db.execute(
            "SELECT shopify_id FROM collections")}
    draft = dict(old)
    draft.update({
        "title": clean_text(data.get("title"), 255),
        "hero_summary": clean_text(data.get("hero_summary"), 420),
        "body_paragraphs": clean_list(data.get("body_paragraphs"), 8, 1200),
        "lifestyle_close": clean_text(data.get("lifestyle_close"), 400),
        "product_type": clean_text(data.get("product_type"), 255),
        "category": clean_text(data.get("category"), 255),
        "tags": clean_list(data.get("tags"), 30, 255),
        "suggested_collections": clean_list(
            data.get("suggested_collections"), 12, 255),
        "collection_ids": [str(value) for value in data.get("collection_ids") or []
                           if str(value) in valid_collection_ids],
        "status": "draft",
    })
    if not draft["title"] or not draft["hero_summary"] or not draft["body_paragraphs"]:
        raise ValueError("Title, hero summary and body paragraphs cannot be empty")
    return save_draft(draft, "editor-save")


def current_product_state(domain: str, token: str, product_id: str) -> dict:
    response = graphql(
        domain, token, PRODUCT_CURRENT_QUERY, {"id": product_id})
    product = ((response.get("data") or {}).get("product") or {})
    if not product.get("id"):
        raise RuntimeError("Shopify product could not be loaded before the change")
    return {
        "id": product["id"], "title": str(product.get("title") or ""),
        "description_html": str(product.get("descriptionHtml") or ""),
        "product_type": str(product.get("productType") or ""),
        "tags": list(product.get("tags") or []),
        "collection_ids": [str(row.get("id")) for row in
                           ((product.get("collections") or {}).get("nodes") or [])
                           if row.get("id")],
        "updated_at": str(product.get("updatedAt") or ""),
    }


def update_local_product_row(db: sqlite3.Connection, product_id: str,
                             remote: dict, stamp: str) -> None:
    """Mirror a verified Shopify product state in one existing transaction."""
    plain_description = html.unescape(re.sub(
        r"<[^>]+>", " ", remote["description_html"]))
    plain_description = re.sub(r"\s+", " ", plain_description).strip()
    db.execute("""
      UPDATE products SET title=?,description=?,description_html=?,product_type=?,
        tags_json=?,updated_at=?,synced_at=? WHERE shopify_id=?
    """, (
        remote["title"], plain_description, remote["description_html"],
        remote["product_type"], json.dumps(remote["tags"], ensure_ascii=False),
        remote.get("updated_at") or stamp, stamp, product_id,
    ))
    db.execute("DELETE FROM product_collections WHERE product_id=?", (product_id,))
    for collection_id in remote.get("collection_ids") or []:
        db.execute("""
          INSERT OR IGNORE INTO product_collections(product_id,collection_id)
          VALUES(?,?)
        """, (product_id, collection_id))


def publish_draft(product_id: str, shop_auth: tuple[str, str] | None = None,
                  selected_fields: list[str] | None = None) -> dict:
    """Apply reviewed fields and persist an exact, rollback-ready change set."""
    draft = load_draft(product_id)
    fields = select_fields(selected_fields)
    if not fields:
        raise ValueError("Select at least one field to apply")
    if shop_auth:
        domain, token = shop_auth
    else:
        domain, client_id, client_secret, fallback = shop_config()
        token = access_token(domain, client_id, client_secret, fallback)

    before = current_product_state(domain, token, product_id)
    draft_values = {
        "title": draft["title"], "description_html": draft["description_html"],
        "product_type": draft["product_type"], "tags": list(draft["tags"]),
        "collection_ids": list(draft.get("collection_ids") or []),
    }
    expected = expected_product_state(before, draft_values, fields)
    change_id = growth_service().repository.create_change_set(
        product_id, "rewrite-draft", fields, before, expected)
    collection_notes: list[str] = []
    try:
        product_input = {"id": product_id}
        input_names = {
            "title": "title", "description_html": "descriptionHtml",
            "product_type": "productType", "tags": "tags",
        }
        for field, input_name in input_names.items():
            if field in fields:
                product_input[input_name] = draft_values[field]
        remote = {"id": product_id, "updatedAt": before.get("updated_at")}
        if len(product_input) > 1:
            result = graphql(domain, token, PRODUCT_UPDATE_MUTATION,
                             {"product": product_input})
            payload = ((result.get("data") or {}).get("productUpdate") or {})
            errors = payload.get("userErrors") or []
            if errors:
                raise RuntimeError("; ".join(
                    str(row.get("message") or row) for row in errors))
            remote = payload.get("product") or {}
            if not remote.get("id"):
                raise RuntimeError("Shopify did not confirm the product update")

        with connection() as db:
            collections = {row["shopify_id"]: dict(row) for row in db.execute(
                "SELECT * FROM collections")}
        current_ids = set(before["collection_ids"])
        if "collections" in fields:
            for collection_id in draft.get("collection_ids") or []:
                collection_id = str(collection_id)
                collection = collections.get(collection_id)
                if not collection or collection_id in current_ids:
                    continue
                if collection.get("is_smart"):
                    collection_notes.append(
                        f"{collection['title']} is automated; Shopify controls its membership")
                    continue
                added = graphql(domain, token, COLLECTION_ADD_MUTATION, {
                    "id": collection_id, "productIds": [product_id]})
                addition = ((added.get("data") or {}).get("collectionAddProductsV2") or {})
                add_errors = addition.get("userErrors") or []
                if add_errors:
                    collection_notes.extend(str(row.get("message") or row)
                                            for row in add_errors)
                else:
                    current_ids.add(collection_id)

        verification_error = ""
        try:
            after = current_product_state(domain, token, product_id)
            mismatches = verification_mismatches(expected, after, fields)
            if mismatches:
                verification_error = (
                    "Applied but Shopify verification did not match: "
                    + ", ".join(mismatches))
        except Exception as exc:
            after = expected
            verification_error = f"Applied but verification failed: {type(exc).__name__}: {exc}"
        change_status = "applied_unverified" if verification_error else "applied"
        growth_service().repository.update_change_set(
            change_id, change_status, after=after, error=verification_error)

        stamp = now()
        fully_published = all(field in fields for field in
                              ("title", "description_html", "product_type", "tags"))
        draft_status = "published" if fully_published else "partially_published"
        with connection() as db:
            update_local_product_row(db, product_id, after, stamp)
            db.execute("""
              INSERT INTO rewrite_revisions(product_id,saved_at,reason,draft_json)
              VALUES(?,?,?,?)
            """, (product_id, stamp, "before-publish",
                  json.dumps(draft, ensure_ascii=False)))
            db.execute("""
              UPDATE rewrite_drafts SET status=?,published_at=?,updated_at=?,
                publish_error=? WHERE product_id=?
            """, (draft_status, stamp, stamp,
                  "\n".join(collection_notes + ([verification_error]
                                                if verification_error else [])),
                  product_id))
            archived_draft = dict(draft)
            archived_draft.update({"status": draft_status, "published_at": stamp,
                                   "updated_at": stamp})
            db.execute("""
              UPDATE grok_content_archive SET draft_json=?,updated_at=?,published_at=?
              WHERE product_id=?
            """, (json.dumps(archived_draft, ensure_ascii=False), stamp, stamp,
                  product_id))
            db.commit()
        return {"ok": True, "product_id": product_id, "published_at": stamp,
                "change_id": change_id, "selected_fields": fields,
                "verification_warning": verification_error,
                "collection_notes": collection_notes,
                "product": {"title": after["title"],
                            "product_type": after["product_type"],
                            "tags": after["tags"]}}
    except Exception as exc:
        growth_service().repository.update_change_set(
            change_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def rollback_change(change_id: str) -> dict:
    """Restore the exact fields captured before an applied product change."""
    repository = growth_service().repository
    change = repository.change_set(change_id)
    if not change:
        raise ValueError("Change record was not found")
    if change["status"] not in ("applied", "applied_unverified", "rollback_failed"):
        raise ValueError("Only an applied or previously failed rollback can be retried")
    before, after = change["before"], change["after"]
    fields = change["selected_fields"]
    domain, client_id, client_secret, fallback = shop_config()
    token = access_token(domain, client_id, client_secret, fallback)
    product_input = {"id": change["product_id"]}
    input_names = {
        "title": "title", "description_html": "descriptionHtml",
        "product_type": "productType", "tags": "tags",
    }
    for field, input_name in input_names.items():
        if field in fields:
            product_input[input_name] = before.get(field)
    try:
        if len(product_input) > 1:
            response = graphql(domain, token, PRODUCT_UPDATE_MUTATION,
                               {"product": product_input})
            payload = ((response.get("data") or {}).get("productUpdate") or {})
            errors = payload.get("userErrors") or []
            if errors:
                raise RuntimeError("; ".join(
                    str(row.get("message") or row) for row in errors))
        collection_notes = []
        if "collections" in fields:
            added_ids = set(after.get("collection_ids") or []) - set(
                before.get("collection_ids") or [])
            for collection_id in added_ids:
                response = graphql(domain, token, COLLECTION_REMOVE_MUTATION, {
                    "id": collection_id, "productIds": [change["product_id"]]})
                payload = ((response.get("data") or {}).get(
                    "collectionRemoveProducts") or {})
                errors = payload.get("userErrors") or []
                collection_notes.extend(str(row.get("message") or row)
                                        for row in errors)
        if collection_notes:
            raise RuntimeError("; ".join(collection_notes))
        verified = current_product_state(domain, token, change["product_id"])
        scalar_fields = [field for field in fields if field != "collections"]
        mismatches = verification_mismatches(before, verified, scalar_fields)
        if "collections" in fields:
            remaining = added_ids & set(verified.get("collection_ids") or [])
            if remaining:
                mismatches.append("collections")
        if mismatches:
            raise RuntimeError(
                "Rollback verification did not match: " + ", ".join(mismatches))
        with connection() as db:
            update_local_product_row(db, change["product_id"], verified, now())
            db.commit()
        repository.update_change_set(change_id, "rolled_back")
        return {"ok": True, "change_id": change_id,
                "product_id": change["product_id"], "rolled_back_at": now()}
    except Exception as exc:
        repository.update_change_set(
            change_id, "rollback_failed", error=f"{type(exc).__name__}: {exc}")
        raise


def publish_worker(product_ids: list[str]) -> None:
    """Publish saved rewrites sequentially with one Shopify authentication."""
    if not PUBLISH_LOCK.acquire(blocking=False):
        return
    epoch = time.time()
    errors = []
    succeeded = 0
    set_publish_state(
        running=True, phase="authenticating", total=len(product_ids), completed=0,
        succeeded=0, failed=0, current_product_id="",
        message="Connecting securely to Shopify", started_at=now(),
        started_epoch=epoch, finished_at="", elapsed_seconds=0.0,
        error="", errors=[])
    try:
        domain, client_id, client_secret, fallback = shop_config()
        token = access_token(domain, client_id, client_secret, fallback)
        set_publish_state(phase="publishing", message="Starting bulk Shopify update")
        for index, product_id in enumerate(product_ids):
            try:
                with connection() as db:
                    row = db.execute(
                        "SELECT title FROM products WHERE shopify_id=?",
                        (product_id,)).fetchone()
                title = row["title"] if row else product_id
                set_publish_state(
                    current_product_id=product_id,
                    message=(f"Applying rewrite {index + 1} of "
                             f"{len(product_ids)}: {title}"))
                publish_draft(product_id, (domain, token))
                succeeded += 1
            except Exception as exc:
                errors.append({"product_id": product_id,
                               "error": f"{type(exc).__name__}: {exc}"})
            set_publish_state(
                completed=index + 1, succeeded=succeeded,
                failed=len(errors), errors=errors)
        set_publish_state(
            running=False,
            phase="complete" if not errors else "complete_with_errors",
            message=(f"Applied {succeeded} of {len(product_ids)} rewrites to Shopify"),
            current_product_id="", finished_at=now(),
            elapsed_seconds=round(time.time() - epoch, 1),
            error=(errors[0]["error"] if errors else ""), errors=errors)
    except Exception as exc:
        set_publish_state(
            running=False, phase="failed", message="Bulk Shopify update failed",
            current_product_id="", finished_at=now(),
            elapsed_seconds=round(time.time() - epoch, 1),
            error=f"{type(exc).__name__}: {exc}", errors=errors,
            failed=len(errors))
    finally:
        PUBLISH_LOCK.release()


def fetch_catalogue() -> tuple[dict, list[dict], list[dict], dict[str, dict]]:
    domain, client_id, client_secret, fallback = shop_config()
    set_state(phase="authenticating", message="Connecting securely to Shopify")
    token = access_token(domain, client_id, client_secret, fallback)
    cursor = None
    variants: list[dict] = []
    shop = {}
    pages = 0
    products_seen: set[str] = set()
    while True:
        payload = graphql(domain, token, VARIANTS_QUERY, {"cursor": cursor})
        data = payload.get("data") or {}
        shop = data.get("shop") or shop
        product_variants = data.get("productVariants") or {}
        batch = product_variants.get("nodes") or []
        variants.extend(batch)
        pages += 1
        products_seen.update(str((row.get("product") or {}).get("id") or "")
                             for row in batch)
        products_seen.discard("")
        set_state(
            phase="downloading",
            message=(f"Downloaded {len(products_seen):,} products and "
                     f"{len(variants):,} variants"),
            pages=pages, products_seen=len(products_seen),
            variants_seen=len(variants))
        page_info = product_variants.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            raise RuntimeError("Shopify pagination stopped without an end cursor")
    collections: list[dict] = []
    cursor = None
    set_state(phase="collections", message="Downloading Shopify collections")
    while True:
        payload = graphql(domain, token, COLLECTIONS_QUERY, {"cursor": cursor})
        connection_data = ((payload.get("data") or {}).get("collections") or {})
        collections.extend(connection_data.get("nodes") or [])
        page_info = connection_data.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            raise RuntimeError("Shopify collection pagination stopped without a cursor")
    reviews: dict[str, dict] = {}
    cursor = None
    set_state(phase="reviews", message="Downloading product ratings and reviews")
    while True:
        payload = graphql(domain, token, REVIEWS_QUERY, {"cursor": cursor})
        connection_data = ((payload.get("data") or {}).get("products") or {})
        for product in connection_data.get("nodes") or []:
            review_data = {}
            metafields = ((product.get("metafields") or {}).get("nodes") or [])
            raw = next((row.get("value") for row in metafields
                        if row.get("key") == "review_widget_data"), "")
            try:
                parsed = json.loads(raw or "{}")
                if isinstance(parsed, dict):
                    review_data = parsed
            except (TypeError, json.JSONDecodeError):
                review_data = {}
            reviews[str(product.get("id") or "")] = {
                "count": int(number(review_data.get("number_of_reviews")) or 0),
                "rating": number(review_data.get("average_rating")) or 0,
            }
        page_info = connection_data.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            raise RuntimeError("Shopify review pagination stopped without a cursor")
    return shop, variants, collections, reviews


def store_catalogue(shop: dict, remote_variants: list[dict],
                    remote_collections: list[dict], remote_reviews: dict[str, dict],
                    marker: str,
                    synced_at: str) -> tuple[int, int]:
    products: dict[str, dict] = {}
    for row in remote_variants:
        product = row.get("product") or {}
        if product.get("id"):
            products[str(product["id"])] = product
    with connection() as db:
        existing_count = int(db.execute("SELECT COUNT(*) FROM products").fetchone()[0])
    remote_count = len(products)
    if existing_count and remote_count == 0:
        raise RuntimeError(
            f"Shopify returned zero products; preserving the {existing_count} products "
            "already stored locally. Check Shopify and DSers before allowing a replacement.")
    if existing_count >= 10 and remote_count < math.ceil(existing_count * .5):
        raise RuntimeError(
            f"Shopify returned only {remote_count} products, down from {existing_count}; "
            "the safety lock preserved the existing local catalogue.")
    set_state(phase="saving", message="Saving the catalogue to the local database",
              products_seen=len(products), variants_seen=len(remote_variants))
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for collection in remote_collections:
            db.execute("""
            INSERT INTO collections (
              shopify_id,legacy_id,title,handle,is_smart,updated_at,sync_marker,synced_at
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(shopify_id) DO UPDATE SET
              legacy_id=excluded.legacy_id,title=excluded.title,handle=excluded.handle,
              is_smart=excluded.is_smart,updated_at=excluded.updated_at,
              sync_marker=excluded.sync_marker,synced_at=excluded.synced_at
            """, (
                str(collection.get("id") or ""),
                str(collection.get("legacyResourceId") or ""),
                str(collection.get("title") or "Untitled collection"),
                str(collection.get("handle") or ""),
                bool(collection.get("ruleSet")),
                str(collection.get("updatedAt") or ""), marker, synced_at,
            ))
        for product in products.values():
            preview = ((product.get("featuredMedia") or {}).get("preview") or {})
            image = preview.get("image") or {}
            review = remote_reviews.get(str(product.get("id") or ""), {})
            db.execute("""
            INSERT INTO products (
              shopify_id,legacy_id,handle,title,description,description_html,
              review_count,review_rating,vendor,
              product_type,status,tags_json,created_at,updated_at,published_at,
              online_store_url,total_inventory,has_only_default_variant,
              featured_image_url,featured_image_alt,sync_marker,synced_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(shopify_id) DO UPDATE SET
              legacy_id=excluded.legacy_id,handle=excluded.handle,title=excluded.title,
              description=excluded.description,description_html=excluded.description_html,
              review_count=excluded.review_count,review_rating=excluded.review_rating,
              vendor=excluded.vendor,product_type=excluded.product_type,status=excluded.status,
              tags_json=excluded.tags_json,created_at=excluded.created_at,
              updated_at=excluded.updated_at,published_at=excluded.published_at,
              online_store_url=excluded.online_store_url,
              total_inventory=excluded.total_inventory,
              has_only_default_variant=excluded.has_only_default_variant,
              featured_image_url=excluded.featured_image_url,
              featured_image_alt=excluded.featured_image_alt,
              sync_marker=excluded.sync_marker,synced_at=excluded.synced_at
            """, (
                str(product.get("id") or ""), str(product.get("legacyResourceId") or ""),
                str(product.get("handle") or ""), str(product.get("title") or "Untitled"),
                str(product.get("description") or ""), str(product.get("descriptionHtml") or ""),
                int(review.get("count") or 0), number(review.get("rating")) or 0,
                str(product.get("vendor") or ""), str(product.get("productType") or ""),
                str(product.get("status") or ""), json.dumps(product.get("tags") or []),
                str(product.get("createdAt") or ""), str(product.get("updatedAt") or ""),
                str(product.get("publishedAt") or ""), str(product.get("onlineStoreUrl") or ""),
                product.get("totalInventory"), bool(product.get("hasOnlyDefaultVariant")),
                str(image.get("url") or ""), str(image.get("altText") or ""), marker, synced_at,
            ))
            # A normal sync never writes the *_grok fields. If DSers has
            # re-created the same product with a new Shopify ID, recover its
            # separately archived Grok copy by stable Shopify handle.
            product_id = str(product.get("id") or "")
            handle = str(product.get("handle") or "")
            archived = db.execute("""
              SELECT title_grok,description_grok,updated_at
              FROM grok_content_archive
              WHERE product_id=? OR (?<>'' AND handle=?)
              ORDER BY CASE WHEN product_id=? THEN 0 ELSE 1 END,updated_at DESC
              LIMIT 1
            """, (product_id, handle, handle, product_id)).fetchone()
            if archived:
                db.execute("""
                  UPDATE products SET title_grok=?,description_grok=?,grok_updated_at=?
                  WHERE shopify_id=?
                """, (archived["title_grok"], archived["description_grok"],
                      archived["updated_at"], product_id))
            db.execute("DELETE FROM product_collections WHERE product_id=?",
                       (product_id,))
            for collection in (product.get("collections") or {}).get("nodes") or []:
                if collection.get("id"):
                    db.execute("""
                      INSERT OR IGNORE INTO product_collections(product_id,collection_id)
                      VALUES(?,?)
                    """, (str(product.get("id")), str(collection.get("id"))))
        for variant in remote_variants:
            product = variant.get("product") or {}
            item = variant.get("inventoryItem") or {}
            cost = item.get("unitCost") or {}
            weight = ((item.get("measurement") or {}).get("weight") or {})
            db.execute("""
            INSERT INTO variants (
              shopify_id,product_id,inventory_item_id,legacy_id,title,display_name,
              sku,barcode,price,compare_at_price,cost,currency,inventory_quantity,
              available_for_sale,taxable,inventory_policy,tracked,requires_shipping,
              weight,weight_unit,country_of_origin,harmonized_system_code,
              selected_options_json,created_at,updated_at,sync_marker,synced_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(shopify_id) DO UPDATE SET
              product_id=excluded.product_id,inventory_item_id=excluded.inventory_item_id,
              legacy_id=excluded.legacy_id,title=excluded.title,
              display_name=excluded.display_name,sku=excluded.sku,barcode=excluded.barcode,
              price=excluded.price,compare_at_price=excluded.compare_at_price,
              cost=excluded.cost,currency=excluded.currency,
              inventory_quantity=excluded.inventory_quantity,
              available_for_sale=excluded.available_for_sale,taxable=excluded.taxable,
              inventory_policy=excluded.inventory_policy,tracked=excluded.tracked,
              requires_shipping=excluded.requires_shipping,weight=excluded.weight,
              weight_unit=excluded.weight_unit,country_of_origin=excluded.country_of_origin,
              harmonized_system_code=excluded.harmonized_system_code,
              selected_options_json=excluded.selected_options_json,
              created_at=excluded.created_at,updated_at=excluded.updated_at,
              sync_marker=excluded.sync_marker,synced_at=excluded.synced_at
            """, (
                str(variant.get("id") or ""), str(product.get("id") or ""),
                str(item.get("id") or ""), str(variant.get("legacyResourceId") or ""),
                str(variant.get("title") or ""), str(variant.get("displayName") or ""),
                str(variant.get("sku") or item.get("sku") or ""),
                str(variant.get("barcode") or ""), number(variant.get("price")),
                number(variant.get("compareAtPrice")), number(cost.get("amount")),
                str(cost.get("currencyCode") or shop.get("currencyCode") or ""),
                variant.get("inventoryQuantity"), bool(variant.get("availableForSale")),
                bool(variant.get("taxable")), str(variant.get("inventoryPolicy") or ""),
                bool(item.get("tracked")), bool(item.get("requiresShipping")),
                number(weight.get("value")), str(weight.get("unit") or ""),
                str(item.get("countryCodeOfOrigin") or ""),
                str(item.get("harmonizedSystemCode") or ""),
                json.dumps(variant.get("selectedOptions") or []),
                str(variant.get("createdAt") or ""), str(variant.get("updatedAt") or ""),
                marker, synced_at,
            ))
        db.execute("DELETE FROM variants WHERE sync_marker <> ?", (marker,))
        db.execute("DELETE FROM products WHERE sync_marker <> ?", (marker,))
        db.execute("DELETE FROM collections WHERE sync_marker <> ?", (marker,))
        meta = {
            "shop_name": str(shop.get("name") or ""),
            "shop_domain": str(shop.get("myshopifyDomain") or ""),
            "currency": str(shop.get("currencyCode") or ""),
            "api_version": API_VERSION,
            "last_sync_at": synced_at,
            "products_count": str(len(products)),
            "variants_count": str(len(remote_variants)),
            "collections_count": str(len(remote_collections)),
        }
        db.executemany("""
          INSERT INTO catalogue_meta(key,value) VALUES(?,?)
          ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, meta.items())
        db.commit()
    return len(products), len(remote_variants)


def sync_worker() -> None:
    if not SYNC_LOCK.acquire(blocking=False):
        return
    started = now()
    epoch = time.time()
    set_state(running=True, phase="starting", message="Starting Shopify sync",
              pages=0, products_seen=0, variants_seen=0, started_at=started,
              started_epoch=epoch, finished_at="", elapsed_seconds=0.0, error="")
    run_id = None
    try:
        backup_database()
        with connection() as db:
            run_id = db.execute(
                "INSERT INTO sync_runs(started_at,status) VALUES(?,?)",
                (started, "running")).lastrowid
            db.commit()
        shop, remote_variants, remote_collections, remote_reviews = fetch_catalogue()
        marker = "sync_" + str(time.time_ns())
        synced_at = now()
        products_count, variants_count = store_catalogue(
            shop, remote_variants, remote_collections, remote_reviews, marker, synced_at)
        finished = now()
        with connection() as db:
            db.execute("""
              UPDATE sync_runs SET finished_at=?,status='complete',
                products_count=?,variants_count=? WHERE id=?
            """, (finished, products_count, variants_count, run_id))
            db.commit()
        set_state(running=False, phase="complete",
                  message=(f"Synced {products_count:,} products and "
                           f"{variants_count:,} variants"),
                  products_seen=products_count, variants_seen=variants_count,
                  finished_at=finished, elapsed_seconds=round(time.time()-epoch, 1))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
        traceback.print_exc()
        if run_id:
            with connection() as db:
                db.execute("""
                  UPDATE sync_runs SET finished_at=?,status='failed',error=? WHERE id=?
                """, (now(), error[:2000], run_id))
                db.commit()
        set_state(running=False, phase="failed", message="Shopify sync failed",
                  finished_at=now(), elapsed_seconds=round(time.time()-epoch, 1),
                  error=error)
    finally:
        SYNC_LOCK.release()


def catalog_payload() -> dict:
    with connection() as db:
        meta = {row["key"]: row["value"] for row in db.execute(
            "SELECT key,value FROM catalogue_meta")}
        product_rows = db.execute("""
          SELECT p.*,
            COUNT(v.shopify_id) AS variant_count,
            MIN(v.price) AS price_min, MAX(v.price) AS price_max,
            MIN(v.cost) AS cost_min, MAX(v.cost) AS cost_max,
            MIN(CASE WHEN v.cost IS NOT NULL AND v.price IS NOT NULL
                     THEN v.price-v.cost END) AS profit_min,
            MAX(CASE WHEN v.cost IS NOT NULL AND v.price IS NOT NULL
                     THEN v.price-v.cost END) AS profit_max,
            SUM(CASE WHEN v.cost IS NULL THEN 1 ELSE 0 END) AS missing_cost_count,
            MAX(v.requires_shipping) AS requires_shipping,
            MIN(v.weight) AS weight_min, MAX(v.weight) AS weight_max,
            GROUP_CONCAT(DISTINCT NULLIF(v.sku,'')) AS skus
          FROM products p LEFT JOIN variants v ON v.product_id=p.shopify_id
          GROUP BY p.shopify_id ORDER BY p.updated_at DESC, p.title COLLATE NOCASE
        """).fetchall()
        variant_rows = db.execute(
            "SELECT * FROM variants ORDER BY product_id,title COLLATE NOCASE").fetchall()
        collection_rows = db.execute(
            "SELECT * FROM collections ORDER BY title COLLATE NOCASE").fetchall()
        membership_rows = db.execute("""
          SELECT pc.product_id,c.* FROM product_collections pc
          JOIN collections c ON c.shopify_id=pc.collection_id
          ORDER BY c.title COLLATE NOCASE
        """).fetchall()
        draft_rows = db.execute("SELECT * FROM rewrite_drafts").fetchall()
    latest_changes = growth_service().repository.latest_change_sets_by_product()
    variants_by_product: dict[str, list[dict]] = {}
    for row in variant_rows:
        value = dict(row)
        value["selected_options"] = json.loads(
            value.pop("selected_options_json") or "[]")
        variants_by_product.setdefault(value["product_id"], []).append(value)
    collections = [dict(row) for row in collection_rows]
    collections_by_product: dict[str, list[dict]] = {}
    for row in membership_rows:
        value = dict(row)
        product_id = value.pop("product_id")
        collections_by_product.setdefault(product_id, []).append(value)
    drafts_by_product = {}
    for row in draft_rows:
        value = dict(row)
        for source, target, fallback in (
            ("body_paragraphs_json", "body_paragraphs", []),
            ("collection_ids_json", "collection_ids", []),
            ("suggested_collections_json", "suggested_collections", []),
            ("tags_json", "tags", []), ("warnings_json", "warnings", []),
            ("original_json", "original", {}),
        ):
            try:
                value[target] = json.loads(value.pop(source) or json.dumps(fallback))
            except (TypeError, json.JSONDecodeError):
                value[target] = fallback
        drafts_by_product[value["product_id"]] = value
    products = []
    for row in product_rows:
        value = dict(row)
        value["tags"] = json.loads(value.pop("tags_json") or "[]")
        value["variants"] = variants_by_product.get(value["shopify_id"], [])
        value["collections"] = collections_by_product.get(value["shopify_id"], [])
        value["rewrite_draft"] = drafts_by_product.get(value["shopify_id"])
        value["last_change"] = latest_changes.get(value["shopify_id"])
        draft = value["rewrite_draft"]
        value["rewrite_pending"] = bool(
            draft and (
                str(value.get("title") or "").strip() !=
                str(draft.get("title") or "").strip()
                or str(value.get("description_html") or "").strip() !=
                str(draft.get("description_html") or "").strip()
                or str(value.get("product_type") or "").strip() !=
                str(draft.get("product_type") or "").strip()
                or sorted(str(item).strip().casefold() for item in value.get("tags") or []) !=
                   sorted(str(item).strip().casefold() for item in draft.get("tags") or [])
            ))
        products.append(value)
    return {"meta": meta, "products": products, "collections": collections,
            "grok_model": grok_status(),
            "sync": state(), "rewrite": rewrite_state(),
            "publish": publish_state()}


class Handler(BaseHTTPRequestHandler):
    server_version = "oMLXShopify/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        try:
            length = min(2_000_000, max(0, int(self.headers.get("Content-Length") or 0)))
        except ValueError:
            length = 0
        if not length:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/health":
                return self.send_json(200, {
                    "ok": True, "service": "Shopify Growth Studio",
                    "database": str(DB_PATH), "api_version": API_VERSION,
                    "grok_model": grok_status(),
                })
            if path == "/api/catalog":
                return self.send_json(200, catalog_payload())
            if path == "/api/growth":
                days = (query.get("days") or ["30"])[0]
                return self.send_json(200, growth_service().payload(days))
            if path == "/api/growth/status":
                service = growth_service()
                return self.send_json(200, {
                    "sync": service.sync_state(),
                    "analysis": service.analysis_state(),
                })
            if path == "/api/growth/exclusions":
                return self.send_json(200, {
                    "exclusions": growth_service().repository.order_exclusions(False)})
            if path == "/api/growth/debug":
                days = (query.get("days") or ["30"])[0]
                return self.send_json(200, growth_service().debug_payload(days))
            if path == "/api/grok/model":
                if (query.get("refresh") or [""])[0] in ("1", "true"):
                    GROK_RESOLVER.resolve(xai_keys(), force=True)
                return self.send_json(200, grok_status())
            if path == "/api/changes":
                limit = (query.get("limit") or ["30"])[0]
                return self.send_json(200, {
                    "changes": growth_service().repository.change_sets(int(limit))})
            if path == "/api/sync/status":
                return self.send_json(200, state())
            if path == "/api/rewrite/status":
                return self.send_json(200, rewrite_state())
            if path == "/api/publish/status":
                return self.send_json(200, publish_state())
            if path == "/growth.js":
                return self.send_file(STATIC_DIR / "growth.js",
                                      "text/javascript; charset=utf-8")
            if path == "/growth.css":
                return self.send_file(STATIC_DIR / "growth.css",
                                      "text/css; charset=utf-8")
            if path in ("/", "/index.html"):
                return self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return self.send_json(404, {"error": "Not found"})
        except Exception as exc:
            traceback.print_exc()
            return self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/sync":
                if publish_state().get("running"):
                    return self.send_json(409, {
                        "error": "Wait for the Shopify update to finish before syncing"})
                if state().get("running"):
                    return self.send_json(202, state())
                threading.Thread(target=sync_worker, daemon=True,
                                 name="shopify-catalogue-sync").start()
                deadline = time.time() + .35
                while not state().get("running") and time.time() < deadline:
                    time.sleep(.01)
                return self.send_json(202, state())
            if path == "/api/growth/sync":
                body = self.read_json()
                service = growth_service()
                return self.send_json(202, service.start_sync(body.get("days", 30)))
            if path == "/api/growth/analyse":
                body = self.read_json()
                service = growth_service()
                return self.send_json(202, service.start_analysis(body.get("days", 30)))
            if path == "/api/growth/paid-media/import":
                body = self.read_json()
                service = growth_service()
                return self.send_json(200, service.refresh_paid_media(
                    body.get("days", 30)))
            if path == "/api/growth/exclusions":
                body = self.read_json()
                service = growth_service()
                return self.send_json(200, service.set_order_exclusion(
                    body.get("order_id"), body.get("active", True),
                    str(body.get("reason") or "Internal end-to-end test")))
            if path == "/api/rewrite":
                body = self.read_json()
                product_ids = []
                seen = set()
                for value in body.get("product_ids") or []:
                    product_id = str(value)
                    if product_id.startswith("gid://shopify/Product/") and product_id not in seen:
                        seen.add(product_id)
                        product_ids.append(product_id)
                if not product_ids:
                    raise ValueError("Select at least one product to rewrite")
                if len(product_ids) > 150:
                    raise ValueError("A rewrite batch can contain at most 150 products")
                if rewrite_state().get("running"):
                    return self.send_json(409, {
                        "error": "A Grok rewrite batch is already running",
                        "rewrite": rewrite_state()})
                if publish_state().get("running"):
                    return self.send_json(409, {
                        "error": "Wait for the Shopify update to finish before rewriting"})
                threading.Thread(target=rewrite_worker, args=(product_ids,), daemon=True,
                                 name="shopify-grok-rewrites").start()
                deadline = time.time() + .35
                while not rewrite_state().get("running") and time.time() < deadline:
                    time.sleep(.01)
                return self.send_json(202, rewrite_state())
            if path == "/api/draft":
                body = self.read_json()
                product_id = str(body.get("product_id") or "")
                if not product_id:
                    raise ValueError("Missing product_id")
                return self.send_json(200, {
                    "ok": True, "draft": update_draft_from_editor(product_id, body)})
            if path == "/api/publish":
                if publish_state().get("running"):
                    return self.send_json(409, {
                        "error": "A bulk Shopify update is already running"})
                if state().get("running"):
                    return self.send_json(409, {
                        "error": "Wait for the Shopify sync to finish first"})
                body = self.read_json()
                product_id = str(body.get("product_id") or "")
                if not product_id:
                    raise ValueError("Missing product_id")
                fields = body.get("fields")
                if fields is not None and not isinstance(fields, list):
                    raise ValueError("fields must be a list")
                return self.send_json(200, publish_draft(product_id,
                                                         selected_fields=fields))
            if path == "/api/changes/rollback":
                if publish_state().get("running") or state().get("running"):
                    return self.send_json(409, {
                        "error": "Wait for the current Shopify operation to finish"})
                body = self.read_json()
                change_id = str(body.get("change_id") or "")
                if not change_id:
                    raise ValueError("Missing change_id")
                return self.send_json(200, rollback_change(change_id))
            if path == "/api/publish/batch":
                return self.send_json(410, {
                    "error": "Bulk publishing is disabled. Review and apply each product with field-level controls."})
            return self.send_json(404, {"error": "Not found"})
        except ValueError as exc:
            return self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    init_db()
    growth_service()
    threading.Thread(
        target=lambda: GROK_RESOLVER.resolve(xai_keys()), daemon=True,
        name="grok-model-discovery").start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Shopify Growth Studio ready at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
