#!/usr/bin/env python3
"""oMLX Kindle Book Studio — Grok-powered story, illustration and KDP exporter."""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = os.environ.get("KINDLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("KINDLE_PORT", "8800"))
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ENV_FILE = Path(os.environ.get("OMLX_ENV_FILE", "/Users/joebains/.omlx/.env"))
LIBRARY_DIR = Path(os.environ.get(
    "KINDLE_LIBRARY_DIR", "/Users/joebains/Documents/Kindle Books/Projects"))
EXPORT_DIR = Path(os.environ.get(
    "KINDLE_EXPORT_DIR", "/Users/joebains/Documents/Kindle Books/Exports"))
TEXT_MODEL = os.environ.get("GROK_BOOK_MODEL", "grok-4.3")
IMAGE_MODEL = os.environ.get("GROK_IMAGE_MODEL", "grok-imagine-image")
XAI_BASE = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
LOCAL_IMAGE_BASE = os.environ.get(
    "OMLX_IMAGE_BASE_URL", "http://127.0.0.1:8400").rstrip("/")
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_ENGINES = [
    {
        "id": "grok", "label": "Grok Imagine · xAI cloud",
        "model": IMAGE_MODEL,
        "description": "Cloud generation with up to three character references.",
    },
    {
        "id": "hidream", "label": "HiDream O1 · local MLX",
        "model": "HiDream-O1-Image-Dev",
        "description": "Runs locally and uses the SSD-backed memory safeguards.",
    },
    {
        "id": "flux", "label": "FLUX.1 Kontext · local MLX",
        "model": "FLUX.1-Kontext-dev-mflux-4bit",
        "description": "Reference-led local generation; a starter reference is made automatically if needed.",
    },
    {
        "id": "nano_banana_2", "label": "Nano Banana 2 · Replicate",
        "model": "google/nano-banana-2",
        "description": "Replicate cloud generation with strong character consistency.",
    },
]
IMAGE_ENGINE_IDS = {x["id"] for x in IMAGE_ENGINES}
_project_create_lock = threading.Lock()

BOOK_TYPES = [
    ("picture_book", "Children’s picture book"),
    ("early_reader", "Early reader"),
    ("chapter_book", "Illustrated chapter book"),
    ("comic", "Comic / graphic novel"),
    ("activity", "Activity book"),
    ("colouring", "Colouring book"),
    ("educational", "Educational workbook"),
    ("fiction", "Fiction novel"),
    ("nonfiction", "Non-fiction guide"),
    ("cookbook", "Cookbook"),
    ("poetry", "Poetry collection"),
    ("journal", "Journal / planner"),
]
GENRES = [
    "Adventure", "Fantasy", "Mystery", "Comedy", "Bedtime", "Friendship",
    "Animals", "Science fiction", "Educational", "Historical", "Romance",
    "Thriller", "Self-help", "Business", "Health and wellness", "Custom",
]
IMAGE_STYLES = [
    ("cinematic_3d", "Cinematic 3D family animation"),
    ("claymation", "Claymation"),
    ("watercolour", "Watercolour storybook"),
    ("paper_cut", "Layered paper cut-out"),
    ("comic", "Bold comic-book art"),
    ("manga", "Manga / anime"),
    ("ink", "Pen-and-ink illustration"),
    ("pencil", "Coloured pencil"),
    ("retro", "Retro children’s cartoon"),
    ("photoreal", "Photorealistic"),
    ("colouring", "Clean colouring-book line art"),
    ("custom", "Custom visual style"),
]
TRIMS = [
    ("8x10", "8 × 10 in portrait", 8.0, 10.0),
    ("8.5x8.5", "8.5 × 8.5 in square", 8.5, 8.5),
    ("6x9", "6 × 9 in novel", 6.0, 9.0),
    ("7x10", "7 × 10 in", 7.0, 10.0),
    ("landscape", "10 × 8 in landscape", 10.0, 8.0),
]
READING_LEVELS = [
    "Ages 2–4", "Ages 4–6", "Ages 6–8", "Ages 8–12",
    "Young adult", "Adult general", "Professional / specialist",
]
PAGE_COUNTS = [4, 6, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]
SERIES_NAME = "The 28-Day Inner Transformation Series"
SERIES_BOOKS = [
    {
        "id": "introspection", "title": "28 Days of Introspection",
        "aliases": ["Daily Introspection"],
        "focus": "deeper self-awareness through reflection, gratitude, intentions and honest daily review",
        "weekly_arc": ["observe yourself", "understand your patterns", "make conscious changes", "integrate and continue"],
        "palette": "warm ivory, sunrise gold, charcoal and soft sage",
    },
    {
        "id": "gratitude", "title": "28 Days of Gratitude",
        "focus": "building a sustainable gratitude practice that notices ordinary, relational and personal gifts",
        "weekly_arc": ["notice everyday gifts", "appreciate people and support", "find learning in difficulty", "live and express gratitude"],
        "palette": "honey gold, blush, warm cream and gentle green",
    },
    {
        "id": "self-confidence", "title": "28 Days of Self-Confidence",
        "focus": "recognising strengths, quieting self-doubt and taking increasingly courageous action",
        "weekly_arc": ["recognise existing strengths", "challenge limiting stories", "practise visible courage", "embody steady confidence"],
        "palette": "deep blue, amber, warm white and confident coral",
    },
    {
        "id": "calm", "title": "28 Days of Calm",
        "focus": "creating practical moments of calm, steadiness and restoration in ordinary daily life",
        "weekly_arc": ["settle the body", "quiet mental noise", "respond calmly to pressure", "build a sustainable calm rhythm"],
        "palette": "mist blue, sea glass, pale sand and soft lavender",
    },
    {
        "id": "self-love", "title": "28 Days of Self-Love",
        "focus": "developing kinder self-talk, healthy boundaries, self-respect and compassionate daily care",
        "weekly_arc": ["meet yourself kindly", "accept the whole self", "protect your needs", "live from self-respect"],
        "palette": "rose, plum, warm cream and muted gold",
    },
    {
        "id": "emotional-resilience", "title": "28 Days of Emotional Resilience",
        "focus": "understanding emotions, recovering from setbacks and responding with flexibility and self-compassion",
        "weekly_arc": ["name and allow emotions", "discover coping strengths", "reframe setbacks", "create a personal resilience plan"],
        "palette": "storm blue, fresh green, copper and clear sky",
    },
    {
        "id": "purpose-clarity", "title": "28 Days of Purpose and Clarity",
        "focus": "clarifying values, priorities, meaningful direction and the next practical steps",
        "weekly_arc": ["clear the noise", "identify values and strengths", "shape a meaningful vision", "commit to aligned action"],
        "palette": "indigo, parchment, sunlit yellow and forest green",
    },
    {
        "id": "better-habits", "title": "28 Days of Better Habits",
        "focus": "designing realistic routines through small actions, useful cues, reflection and compassionate consistency",
        "weekly_arc": ["understand current patterns", "design tiny changes", "strengthen consistency", "make the habits sustainable"],
        "palette": "fresh teal, tangerine, clean white and graphite",
    },
    {
        "id": "focus-productivity", "title": "28 Days of Focus and Productivity",
        "focus": "choosing meaningful priorities, reducing distraction and completing important work without burnout",
        "weekly_arc": ["discover attention patterns", "simplify priorities", "practise deep focus", "build a balanced productivity system"],
        "palette": "navy, electric blue, citrus and cool white",
    },
    {
        "id": "mindful-living", "title": "28 Days of Mindful Living",
        "focus": "bringing non-judgemental awareness into the senses, routines, relationships and choices",
        "weekly_arc": ["arrive in the senses", "bring presence to routines", "relate with awareness", "carry mindfulness forward"],
        "palette": "moss, stone, water blue and natural linen",
    },
    {
        "id": "creativity", "title": "28 Days of Creativity",
        "focus": "reawakening curiosity, overcoming creative inhibition and establishing a playful creative practice",
        "weekly_arc": ["notice and collect inspiration", "play without judgement", "develop original ideas", "complete and share something"],
        "palette": "magenta, turquoise, sunshine yellow and ink",
    },
    {
        "id": "positive-thinking", "title": "28 Days of Positive Thinking",
        "focus": "building realistic optimism by noticing helpful possibilities without denying difficult feelings",
        "weekly_arc": ["notice thought patterns", "find balanced alternatives", "practise possibility and appreciation", "make optimism actionable"],
        "palette": "sunflower, sky blue, fresh white and warm orange",
    },
    {
        "id": "personal-growth", "title": "28 Days of Personal Growth",
        "focus": "reviewing identity, values, courage, relationships and goals to create an integrated growth plan",
        "weekly_arc": ["take an honest inventory", "stretch beyond old limits", "strengthen relationships and choices", "design the next chapter"],
        "palette": "emerald, midnight blue, warm gold and ivory",
    },
]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(value: str, fallback="book") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip()).strip("-").lower()
    return (s[:70].rstrip("-") or fallback)


def load_env() -> dict:
    vals = {}
    try:
        for line in ENV_FILE.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return vals


def api_key() -> str:
    vals = load_env()
    return (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
            or vals.get("XAI_API_KEY") or vals.get("GROK_API_KEY") or "").strip()


def api_key_tag() -> str:
    return hashlib.sha256(api_key().encode()).hexdigest()[:12]


def project_dir(project_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{4,80}", project_id or ""):
        raise ValueError("invalid project id")
    return LIBRARY_DIR / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def save_project(project: dict) -> dict:
    pid = project["id"]
    folder = project_dir(pid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    (folder / "revisions").mkdir(exist_ok=True)
    project["updated_at"] = now()
    tmp = folder / "project.json.tmp"
    tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2))
    tmp.replace(folder / "project.json")
    return project


def preserve_server_assets(current: dict, incoming: dict) -> dict:
    """Do not let a stale browser tab erase private references or generated art."""
    incoming["reference_images"] = current.get("reference_images", [])
    current_cover = current.get("cover") or {}
    incoming_cover = incoming.setdefault("cover", {})
    if current_cover.get("image") and not incoming_cover.get("image"):
        incoming_cover["image"] = current_cover["image"]
    current_pages = {int(p.get("number", 0)): p for p in current.get("pages", [])}
    for page in incoming.get("pages", []):
        old = current_pages.get(int(page.get("number", 0)))
        if old and old.get("image") and not page.get("image"):
            page["image"] = old["image"]
    current_characters = {
        str(c.get("name", "")).strip().casefold(): c
        for c in current.get("character_bible", [])
    }
    for character in incoming.get("character_bible", []):
        old = current_characters.get(
            str(character.get("name", "")).strip().casefold(), {})
        for key in ("reference_image", "reference_file_id"):
            if old.get(key) and not character.get(key):
                character[key] = old[key]
    return incoming


def load_project(project_id: str) -> dict:
    path = project_file(project_id)
    if not path.exists():
        raise FileNotFoundError(project_id)
    project = json.loads(path.read_text())
    references = project.get("reference_images") or []
    characters = project.setdefault("character_bible", [])
    by_name = {
        str(c.get("name", "")).strip().casefold(): c for c in characters
    }
    for slot, reference in enumerate(references[:3]):
        reference["slot"] = slot
        name = str(reference.get("character_name") or f"Character {slot + 1}").strip()
        character = by_name.get(name.casefold())
        if character is None and len(characters) < 3:
            character = {
                "name": name, "role": "", "appearance": "", "personality": "",
                "continuity_rules": "",
            }
            characters.append(character)
            by_name[name.casefold()] = character
        if character is not None:
            character["reference_image"] = reference.get("path", "")
            character["reference_file_id"] = reference.get("file_id", "")
            character["reference_slot"] = slot
    return project


def revision_snapshot(project: dict, label: str) -> None:
    folder = project_dir(project["id"]) / "revisions"
    folder.mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + slug(label) + ".json"
    (folder / name).write_text(json.dumps(project, ensure_ascii=False, indent=2))


def list_projects() -> list[dict]:
    out = []
    for path in LIBRARY_DIR.glob("*/project.json"):
        try:
            p = json.loads(path.read_text())
            out.append({
                "id": p["id"], "title": p.get("title") or "Untitled",
                "type": p.get("settings", {}).get("book_type"),
                "stage": p.get("stage", "setup"),
                "updated_at": p.get("updated_at", ""),
                "page_count": len(p.get("pages") or []),
            })
        except Exception:
            continue
    return sorted(out, key=lambda x: x["updated_at"], reverse=True)


def _recent_duplicate_project(data: dict, window_seconds: int = 45) -> dict | None:
    """Return a just-created matching project when a UI action was submitted twice."""
    if not data.get("prevent_duplicate"):
        return None
    title = str(data.get("title") or "Untitled Book").strip().casefold()
    series_name = str(data.get("series_name") or "").strip().casefold()
    book_number = str(data.get("book_number") or "").strip()
    template_id = str((data.get("series_template") or {}).get("id") or "")
    cutoff = time.time() - window_seconds
    for path in LIBRARY_DIR.glob("*/project.json"):
        try:
            if path.stat().st_mtime < cutoff:
                continue
            candidate = json.loads(path.read_text())
            settings = candidate.get("settings") or {}
            candidate_template = candidate.get("series_template") or {}
            if (
                str(candidate.get("title") or "").strip().casefold() == title
                and str(settings.get("series_name") or "").strip().casefold() == series_name
                and str(settings.get("book_number") or "").strip() == book_number
                and str(candidate_template.get("id") or "") == template_id
            ):
                return load_project(candidate["id"])
        except Exception:
            continue
    return None


def new_project(data: dict) -> dict:
    # The lock makes the duplicate check and project creation one atomic action.
    with _project_create_lock:
        duplicate = _recent_duplicate_project(data)
        if duplicate:
            return duplicate
        return _new_project(data)


def _new_project(data: dict) -> dict:
    pid = "book_" + uuid.uuid4().hex[:12]
    settings = {
        "book_type": data.get("book_type", "picture_book"),
        "genre": data.get("genre", "Adventure"),
        "audience": data.get("audience", "Ages 4–6"),
        "language": data.get("language", "English"),
        "tone": data.get("tone", "Warm, engaging and imaginative"),
        "page_count": int(data.get("page_count", 24)),
        "image_style": data.get("image_style", "cinematic_3d"),
        "image_engine": (
            data.get("image_engine", "grok")
            if data.get("image_engine", "grok") in IMAGE_ENGINE_IDS else "grok"
        ),
        "custom_style": data.get("custom_style", ""),
        "trim": data.get("trim", "8x10"),
        "layout": data.get("layout", "fixed"),
        "author": data.get("author", ""),
        "primary_marketplace": data.get("primary_marketplace", "Amazon.co.uk"),
        "print_interior": data.get("print_interior", "premium_colour"),
        "font_style": data.get("font_style", "Friendly storybook"),
        "dedication": data.get("dedication", ""),
        "series_name": data.get("series_name", ""),
        "book_number": data.get("book_number", ""),
        "personalisation": data.get("personalisation", ""),
    }
    p = {
        "id": pid, "title": data.get("title", "Untitled Book"),
        "subtitle": "", "prompt": data.get("prompt", ""),
        "stage": "setup", "created_at": now(), "updated_at": now(),
        "settings": settings, "story_summary": "",
        "series_template": data.get("series_template") or {},
        "character_bible": data.get("character_bible") or [],
        "world_bible": "", "cover": {"title": "", "subtitle": "", "image_prompt": "",
                                     "image": "", "approved": False},
        "pages": [], "metadata": {}, "notes": "", "history": [],
    }
    return save_project(p)


def xai_json(url: str, payload: dict, timeout=300) -> dict:
    key = api_key()
    if not key:
        raise RuntimeError(f"GROK_API_KEY/XAI_API_KEY is missing from {ENV_FILE}")
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "oMLX-Kindle-Studio/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:1000]
            if e.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"xAI API {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == 2:
                raise RuntimeError(
                    "The xAI image connection failed after three attempts: " + str(e)) from e
        time.sleep(2 ** attempt)
    raise RuntimeError("The xAI request could not be completed")


def xai_upload_file(path: Path) -> str:
    boundary = "----omlx" + uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n").encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        XAI_BASE + "/files", data=b"".join(parts), method="POST",
        headers={"Authorization": "Bearer " + api_key(),
                 "Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
        if not result.get("id"):
            raise RuntimeError("xAI did not return a file ID")
        return result["id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"xAI file upload {e.code}: "
                           + e.read().decode("utf-8", "replace")[:1000]) from e


def extract_response_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text") and content.get("text"):
                    return content["text"]
    choices = data.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def grok_structured(instruction: str, schema_hint: dict, timeout=600) -> dict:
    prompt = (
        instruction + "\n\nReturn ONLY one valid JSON object. Follow this shape exactly:\n"
        + json.dumps(schema_hint, ensure_ascii=False, indent=2)
        + "\nDo not use Markdown fences."
    )
    payload = {
        "model": TEXT_MODEL, "input": prompt, "store": False,
        "include": ["no_inline_citations"],
        "text": {"format": {"type": "json_object"}},
    }
    data = xai_json(XAI_BASE + "/responses", payload, timeout)
    raw = extract_response_text(data).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("Grok returned invalid JSON: " + raw[:700]) from e


def style_label(settings: dict) -> str:
    sid = settings.get("image_style", "cinematic_3d")
    label = dict(IMAGE_STYLES).get(sid, sid)
    if sid == "custom" and settings.get("custom_style"):
        label = settings["custom_style"]
    return label


def story_instruction(p: dict) -> str:
    s = p["settings"]
    kind = dict(BOOK_TYPES).get(s["book_type"], s["book_type"])
    count = int(s["page_count"])
    series = p.get("series_template") or {}
    series_rules = ""
    if series:
        weekly_arc = series.get("weekly_arc") or []
        series_rules = f"""
SERIES PRODUCTION BIBLE - follow this exactly:
- This is book {s.get('book_number') or '1'} in "{s.get('series_name') or SERIES_NAME}".
- Keep the main title exactly "{p.get('title')}"; do not rename it.
- This book's unique focus is: {series.get('focus')}.
- Its four weekly phases are: {'; '.join(weekly_arc)}.
- Create exactly 28 daily workbook pages, one complete day per page.
- Give every day 4 to 8 concise, useful exercises combining reflection,
  gratitude, intention, a practical action, journaling and an end-of-day review
  when relevant to the focus.
- In each page's text, place exactly two empty lines between exercises so the
  printed workbook provides intentional writing space. Preserve those blank lines.
- Progress gently across the four phases without repeating prompts, sentences or
  exercises from another day or another title in the series.
- Keep the shared series structure and polished adult tone, while making all
  content, headings, examples, metadata and illustrations original to this topic.
- This is a reflective wellbeing workbook, not diagnosis, treatment or a promise
  of medical results.
- Use this book's identifying visual palette: {series.get('palette')}.
"""
    return f"""You are an expert commercially published Kindle author, developmental
editor, book designer, and art director. Create a complete original {kind}.

Core idea: {p.get('prompt')}
Working title: {p.get('title')}
Genre: {s.get('genre')}
Audience/reading level: {s.get('audience')}
Language: {s.get('language')}
Tone: {s.get('tone')}
Exact story/content pages: {count}
Visual direction: {style_label(s)}
Layout: {s.get('layout')}
Author: {s.get('author')}
Primary Amazon marketplace: {s.get('primary_marketplace', 'Amazon.co.uk')}
Dedication or personal foreword: {s.get('dedication')}
Series: {s.get('series_name')} {s.get('book_number')}
Personalisation and character notes: {s.get('personalisation')}
Characters supplied by the creator: {json.dumps(p.get('character_bible'), ensure_ascii=False)}
{series_rules}

Build a coherent beginning, development, climax/payoff, and satisfying ending.
For children, keep age-appropriate vocabulary and page text length. For comics,
provide concise narration/dialogue and panel-aware image direction. For
non-fiction, build a useful factual progression and flag anything requiring
fact checking. Maintain exact character visual continuity. Image prompts must
describe the full scene, recurring character appearance, composition, camera,
lighting, palette, and leave safe negative space for text. Do not put lettering,
captions, logos, or watermarks inside generated images.

Also create cover direction and KDP metadata. The pages array MUST contain
exactly {count} objects numbered 1 through {count}.

Create a complete Amazon KDP listing worksheet. The description must accurately
sell this specific book without reviews, unverifiable claims, keyword stuffing,
URLs, prices, or promotional language, and must not exceed 4,000 characters.
Return exactly seven useful multi-word customer search phrases and exactly three
highly relevant Amazon category-path suggestions. Include the reading age and
responsible recommendations for marketplace, rights, territories, DRM, KDP
Select, royalty and price. Because Grok creates the book text and illustrations,
the AI disclosure must say that both text and images are AI-generated.

Required titling:
- Supply a compelling book subtitle that complements the main title.
- Supply a cover subtitle; it may match the book subtitle when appropriate.
- Every page must have a short, meaningful, unique heading suited to its content.
- Never leave title, subtitle, cover title, cover subtitle, or page heading blank.
- Do not use placeholders such as "Page 1", "Untitled", or "Chapter"."""


KDP_METADATA_SHAPE = {
    "description": "Compelling accurate Amazon description, maximum 4,000 characters",
    "keywords": [
        "Exactly 7 distinct multi-word customer search phrases"
    ],
    "categories": [
        "Exactly 3 relevant Amazon category paths for the selected marketplace"
    ],
    "age_range": "Reader-facing target age range",
    "reading_age_min": "Minimum age as a number",
    "reading_age_max": "Maximum age as a number",
    "grade_range": "Suggested grade range, or Not applicable",
    "primary_marketplace": "Recommended Amazon marketplace, e.g. Amazon.co.uk",
    "sexually_explicit": "Yes or No",
    "publishing_rights": "I own the copyright and hold the necessary publishing rights",
    "territories": "All territories (worldwide rights), if original content",
    "ai_generated_content": "Yes — AI-generated text and images; no AI translation",
    "drm_recommendation": "Apply DRM or DRM-free, with a short reason",
    "kdp_select_recommendation": "Enroll or do not enroll, with a short reason",
    "royalty_recommendation": "35% or 70%, with a short reason",
    "list_price": "Suggested numeric list price",
    "currency": "GBP, USD, EUR, etc.",
    "price_rationale": "Short editable pricing rationale",
    "contributors": "Other contributors or None",
    "publisher": "Publisher/imprint name or Independently published",
    "edition_number": "Edition number or 1",
    "release_timing": "Publish now or suggested release approach",
    "copyright_text": "Copyright page copy",
    "author_bio": "Editable author biography",
}


STORY_SHAPE = {
    "title": "Compelling book title",
    "subtitle": "Required compelling book subtitle; never blank",
    "story_summary": "Full synopsis",
    "character_bible": [{
        "name": "Name", "role": "Role", "appearance": "Exact reusable appearance",
        "personality": "Traits", "continuity_rules": "Never-changing details",
    }],
    "world_bible": "Locations, palette, era, props and continuity",
    "cover": {
        "title": "Required cover title",
        "subtitle": "Required compelling cover subtitle; never blank",
        "image_prompt": "Detailed cover art prompt",
    },
    "pages": [{
        "number": 1, "heading": "Required short unique heading; never blank",
        "text": "Final page text",
        "dialogue": [], "image_prompt": "Detailed consistent illustration prompt",
        "negative_prompt": "Unwanted elements", "layout_note": "Text and image placement",
    }],
    "metadata": KDP_METADATA_SHAPE,
}


def normalize_metadata(metadata: dict, project: dict) -> dict:
    out = dict(metadata or {})
    out["description"] = str(out.get("description") or "")[:4000]
    out["keywords"] = [str(x).strip() for x in out.get("keywords", [])
                       if str(x).strip()][:7]
    out["categories"] = [str(x).strip() for x in out.get("categories", [])
                         if str(x).strip()][:3]
    out["primary_marketplace"] = (
        project.get("settings", {}).get("primary_marketplace") or "Amazon.co.uk")
    out.setdefault("sexually_explicit", "No")
    out.setdefault(
        "publishing_rights",
        "I own the copyright and hold the necessary publishing rights")
    out.setdefault("territories", "All territories (worldwide rights)")
    out.setdefault(
        "ai_generated_content",
        "Yes — AI-generated text and images; no AI translation")
    out.setdefault("publisher", "Independently published")
    out.setdefault("edition_number", "1")
    out.setdefault("contributors", "None")
    out.setdefault("release_timing", "Publish now")
    return out


def normalize_story(project: dict, data: dict) -> dict:
    existing_pages = {
        int(page.get("number", 0)): page
        for page in project.get("pages", [])
        if int(page.get("number", 0)) > 0
    }
    if not (project.get("series_template") or {}).get("lock_title"):
        project["title"] = str(data.get("title") or project["title"])
    project["subtitle"] = str(data.get("subtitle") or "")
    project["story_summary"] = str(data.get("story_summary") or "")
    generated_characters = data.get("character_bible") or []
    existing_by_name = {
        str(c.get("name", "")).strip().casefold(): c
        for c in project.get("character_bible", [])
    }
    existing_characters = project.get("character_bible", [])
    for index, character in enumerate(generated_characters):
        existing = existing_by_name.get(
            str(character.get("name", "")).strip().casefold(),
            existing_characters[index] if index < len(existing_characters) else {})
        for key in ("reference_image", "reference_file_id", "reference_slot"):
            if existing.get(key):
                character[key] = existing[key]
    project["character_bible"] = generated_characters
    project["world_bible"] = str(data.get("world_bible") or "")
    project["metadata"] = normalize_metadata(data.get("metadata") or {}, project)
    cover = data.get("cover") or {}
    project["cover"].update({
        "title": cover.get("title") or project["title"],
        "subtitle": cover.get("subtitle") or project["subtitle"],
        "image_prompt": cover.get("image_prompt") or "",
    })
    pages = []
    expected = int(project["settings"]["page_count"])
    source = data.get("pages") or []
    for i in range(expected):
        raw = source[i] if i < len(source) else {}
        existing = existing_pages.get(i + 1, {})
        existing_image = str(existing.get("image") or "")
        pages.append({
            "number": i + 1, "heading": str(raw.get("heading") or ""),
            "text": str(raw.get("text") or ""),
            "dialogue": raw.get("dialogue") or [],
            "image_prompt": str(raw.get("image_prompt") or ""),
            "negative_prompt": str(raw.get("negative_prompt") or
                                   "text, letters, logo, watermark, distorted anatomy"),
            "layout_note": str(raw.get("layout_note") or ""),
            "image": existing_image,
            "approved": bool(existing.get("approved")) if existing_image else False,
            "text_approved": False,
        })
    project["pages"] = pages
    project["stage"] = "story"
    project["history"].append({"at": now(), "action": "Generated full book with Grok"})
    return project


def ensure_grok_titles(data: dict, expected_pages: int) -> dict:
    pages = data.get("pages") or []
    cover = data.get("cover") or {}
    missing = (
        not str(data.get("subtitle") or "").strip()
        or not str(cover.get("title") or "").strip()
        or not str(cover.get("subtitle") or "").strip()
        or any(not str((pages[i] if i < len(pages) else {}).get("heading") or "").strip()
               for i in range(expected_pages))
    )
    if not missing:
        return data
    context = {
        "title": data.get("title"), "subtitle": data.get("subtitle"),
        "summary": data.get("story_summary"),
        "cover": cover,
        "pages": [{"number": p.get("number"), "text": p.get("text"),
                   "heading": p.get("heading")} for p in pages],
    }
    completed = grok_structured(
        """Act as a senior publishing copywriter. Complete all required titling
for this book. Write a compelling book subtitle, a cover title and subtitle,
and one short, meaningful, unique heading for every page. No field may be
blank. Do not use generic placeholders such as Page 1, Untitled or Chapter.
Book context:\n""" + json.dumps(context, ensure_ascii=False),
        {
            "subtitle": "Required book subtitle",
            "cover_title": "Required cover title",
            "cover_subtitle": "Required cover subtitle",
            "page_headings": [
                {"number": i, "heading": "Required unique heading"}
                for i in range(1, expected_pages + 1)
            ],
        })
    data["subtitle"] = completed.get("subtitle") or data.get("subtitle")
    cover["title"] = completed.get("cover_title") or cover.get("title") or data.get("title")
    cover["subtitle"] = completed.get("cover_subtitle") or cover.get("subtitle")
    data["cover"] = cover
    headings = completed.get("page_headings") or []
    by_number = {int(x.get("number", 0)): x.get("heading") for x in headings}
    for i, page in enumerate(pages, 1):
        page["heading"] = page.get("heading") or by_number.get(i) or f"Part {i}"
    return data


def generate_story(project_id: str) -> dict:
    p = load_project(project_id)
    revision_snapshot(p, "before-full-generation")
    result = grok_structured(story_instruction(p), STORY_SHAPE, timeout=900)
    result = ensure_grok_titles(result, int(p["settings"]["page_count"]))
    return save_project(normalize_story(p, result))


def populate_titles(project_id: str) -> dict:
    p = load_project(project_id)
    revision_snapshot(p, "before-title-population")
    ensure_grok_titles(p, len(p.get("pages") or []))
    p["history"].append({"at": now(), "action": "Populated missing titles with Grok"})
    return save_project(p)


def generate_metadata(project_id: str) -> dict:
    p = load_project(project_id)
    revision_snapshot(p, "before-kdp-listing")
    context = {
        "title": p.get("title"), "subtitle": p.get("subtitle"),
        "author": p.get("settings", {}).get("author"),
        "language": p.get("settings", {}).get("language"),
        "book_type": p.get("settings", {}).get("book_type"),
        "genre": p.get("settings", {}).get("genre"),
        "audience": p.get("settings", {}).get("audience"),
        "series_name": p.get("settings", {}).get("series_name"),
        "book_number": p.get("settings", {}).get("book_number"),
        "primary_marketplace": p.get("settings", {}).get(
            "primary_marketplace", "Amazon.co.uk"),
        "summary": p.get("story_summary"),
        "characters": p.get("character_bible"),
        "page_text": [page.get("text") for page in p.get("pages", [])],
    }
    instruction = """Act as an ethical Amazon KDP metadata specialist. Create a
complete, accurate, conversion-focused listing worksheet for this exact book.
The description must be plain text, appealing to the intended buyer, contain
no reviews, unverifiable claims, prices, URLs, keyword stuffing or misleading
language, and be no more than 4,000 characters. Supply exactly seven distinct
multi-word search phrases and exactly three highly relevant Amazon category
path suggestions. Do not invent content that is not in the book. Recommend the
remaining KDP choices responsibly. Mark both text and images as AI-generated,
with no AI translation. Book context:\n""" + json.dumps(context, ensure_ascii=False)
    p["metadata"] = normalize_metadata(
        grok_structured(instruction, KDP_METADATA_SHAPE), p)
    p["history"].append({"at": now(), "action": "Generated Amazon KDP listing with Grok"})
    return save_project(p)


def regenerate_page(project_id: str, page_number: int, request: str) -> dict:
    p = load_project(project_id)
    idx = page_number - 1
    if idx < 0 or idx >= len(p["pages"]):
        raise ValueError("page out of range")
    revision_snapshot(p, f"before-page-{page_number}-revision")
    page = p["pages"][idx]
    context = {
        "title": p["title"], "summary": p["story_summary"],
        "characters": p["character_bible"], "world": p["world_bible"],
        "previous_page": p["pages"][idx - 1] if idx else None,
        "current_page": page,
        "next_page": p["pages"][idx + 1] if idx + 1 < len(p["pages"]) else None,
    }
    fallback_request = ("Improve clarity, pacing, age fit, continuity and "
                        "illustration specificity.")
    instruction = f"""Act as a meticulous book editor and art director. Revise page
{page_number} according to this request: {request or fallback_request}
Preserve continuity and do not rewrite adjacent pages. Book context:
{json.dumps(context, ensure_ascii=False)}"""
    shape = {
        "heading": "Required short meaningful page heading; never blank",
        "text": "Revised final page text", "dialogue": [],
        "image_prompt": "Revised detailed illustration prompt",
        "negative_prompt": "Unwanted elements", "layout_note": "Placement",
    }
    result = grok_structured(instruction, shape)
    for key in ("heading", "text", "dialogue", "image_prompt",
                "negative_prompt", "layout_note"):
        if key in result:
            page[key] = result[key]
    page["approved"] = False
    page["text_approved"] = False
    page["image"] = ""
    p["history"].append({"at": now(), "action": f"Regenerated page {page_number}"})
    return save_project(p)


def regenerate_cover(project_id: str, request: str) -> dict:
    p = load_project(project_id)
    instruction = f"""Create improved Kindle cover copy and a detailed image-generation
prompt for this book. User request: {request or 'Make it commercially compelling.'}
Title: {p['title']}; subtitle: {p.get('subtitle')}; summary: {p.get('story_summary')}
Characters: {json.dumps(p.get('character_bible'), ensure_ascii=False)}
Visual style: {style_label(p['settings'])}. Art must contain no generated text;
the application overlays typography separately."""
    result = grok_structured(instruction + """
The title and subtitle are both required and must not be blank. The subtitle
should be concise, commercially appealing, and complement rather than repeat
the title.""", {
        "title": p["title"], "subtitle": "Required compelling cover subtitle",
        "image_prompt": "Detailed portrait cover illustration prompt",
    })
    p["cover"].update(result)
    p["cover"]["image"] = ""
    p["cover"]["approved"] = False
    return save_project(p)


def download_image(item: dict) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = item.get("url")
    if not url:
        raise RuntimeError("xAI image response contained no URL or image data")
    req = urllib.request.Request(url, headers={"User-Agent": "oMLX-Kindle-Studio/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Image download failed with HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == 2:
                raise RuntimeError(
                    "The generated image download failed after three attempts: " + str(e)) from e
        time.sleep(2 ** attempt)
    raise RuntimeError("The generated image could not be downloaded")


def local_image_json(path: str, payload: dict | None = None, timeout=90) -> dict:
    url = LOCAL_IMAGE_BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json",
                 "User-Agent": "oMLX-Kindle-Studio/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"Local image service {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise RuntimeError(
            "The local image service is not available. Open oMLX and try again. " + str(e)) from e


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def local_reference_images(project: dict, target: str, engine: str) -> list[str]:
    """Collect character and existing-art references without exposing local paths."""
    folder = project_dir(project["id"])
    attached = []
    for ref in (project.get("reference_images") or [])[:3]:
        path = folder / str(ref.get("path") or "")
        if path.exists() and path.is_file():
            attached.append(path)

    current = None
    if target == "cover":
        rel = (project.get("cover") or {}).get("image")
    else:
        rel = project["pages"][int(target) - 1].get("image")
    if rel:
        candidate = folder / rel
        if candidate.exists() and candidate.is_file():
            current = candidate

    existing = []
    cover_rel = (project.get("cover") or {}).get("image")
    if cover_rel:
        existing.append(folder / cover_rel)
    for page in project.get("pages") or []:
        if page.get("image"):
            existing.append(folder / page["image"])

    ordered = ([current] if current and engine == "flux" else []) + attached
    if current and engine != "flux" and not attached:
        ordered.append(current)
    ordered.extend(existing)
    unique = []
    for path in ordered:
        if path and path.exists() and path.is_file() and path not in unique:
            unique.append(path)
    limit = 1 if engine == "flux" else 3
    return [image_data_url(path) for path in unique[:limit]]


def local_image_geometry(settings: dict) -> tuple[int, int, str, str]:
    w_in, h_in = trim_size(settings)
    ratio = w_in / h_in
    candidates = {
        "1:1": 1.0, "2:3": 2 / 3, "3:4": 3 / 4, "4:5": 4 / 5,
        "3:2": 3 / 2, "4:3": 4 / 3,
    }
    aspect = min(candidates, key=lambda key: abs(candidates[key] - ratio))
    if aspect == "1:1":
        return 2048, 2048, "1:1", "1:1"
    if aspect == "4:5":
        return 1792, 2304, "4:5", "4:5"
    if aspect == "2:3":
        return 1664, 2496, "", "2:3"
    if aspect == "3:4":
        return 1792, 2304, "", "3:4"
    if aspect == "3:2":
        return 2496, 1664, "3:2", "3:2"
    return 2304, 1792, "", "4:3"


def run_local_image(engine: str, prompt: str, settings: dict,
                    references: list[str]) -> bytes:
    service_engine = "kontext" if engine == "flux" else engine
    width, height, preset, aspect = local_image_geometry(settings)
    payload = {
        "engine": service_engine, "prompt": prompt[:2000],
        "width": width, "height": height, "steps": 28,
        "seed": int(uuid.uuid4().hex[:8], 16),
        "reference_images": references,
        "nano_aspect_ratio": aspect, "nano_output_format": "png",
    }
    if preset:
        payload["preset"] = preset
    if service_engine == "kontext":
        payload["reference_image"] = references[0] if references else ""
    response = local_image_json("/generate", payload, timeout=120)
    job_id = response.get("job_id")
    if not job_id:
        raise RuntimeError("The local image service did not start the image job")
    deadline = time.time() + 1200
    while time.time() < deadline:
        status = local_image_json(
            "/status?" + urllib.parse.urlencode({"id": job_id}), timeout=60)
        if status.get("status") == "done":
            result = status.get("result") or {}
            image_url = result.get("url")
            if not image_url:
                raise RuntimeError("The local image job finished without an image")
            return download_image({"url": urllib.parse.urljoin(
                LOCAL_IMAGE_BASE + "/", image_url.lstrip("/"))})
        if status.get("status") == "error" or status.get("error"):
            raise RuntimeError(status.get("error") or "The local image job failed")
        time.sleep(2)
    raise RuntimeError("The local image job timed out after 20 minutes")


def generate_image(project_id: str, target: str) -> dict:
    p = load_project(project_id)
    s = p["settings"]
    variation_id = uuid.uuid4().hex[:10]
    common = (
        f"Original book illustration. Visual style: {style_label(s)}. "
        f"Audience: {s.get('audience')}. Book continuity: {p.get('world_bible')}. "
        f"Character bible: {json.dumps(p.get('character_bible'), ensure_ascii=False)}. "
        "Maintain exact recurring character identity, clothing and palette. "
        "Professional publishable composition, no text, no letters, no logo, no watermark."
    )
    ref_notes = [r.get("label") for r in p.get("reference_images", []) if r.get("label")]
    if ref_notes:
        common += " Reference image roles: " + "; ".join(ref_notes) + "."
    if target == "cover":
        prompt = common + "\nCOVER ART: " + p["cover"]["image_prompt"]
        had_existing_image = bool(p["cover"].get("image"))
        filename = f"cover-{variation_id}.png" if had_existing_image else "cover.png"
    else:
        number = int(target)
        page = p["pages"][number - 1]
        prompt = common + f"\nPAGE {number}: " + page["image_prompt"]
        if page.get("negative_prompt"):
            prompt += "\nAvoid: " + page["negative_prompt"]
        had_existing_image = bool(page.get("image"))
        filename = (
            f"page-{number:03d}-{variation_id}.png"
            if had_existing_image else f"page-{number:03d}.png"
        )
    if had_existing_image:
        revision_snapshot(
            p, "before-cover-image-regeneration" if target == "cover"
            else f"before-page-{number}-image-regeneration")
        prompt += (
            f"\nREGENERATION VARIATION {variation_id}: Create a clearly new "
            "alternative illustration. Keep the established characters and art "
            "style, but substantially change the composition, poses, camera angle, "
            "background details and visual storytelling. Do not reproduce the "
            "previous image."
        )
    engine = s.get("image_engine", "grok")
    if engine not in IMAGE_ENGINE_IDS:
        engine = "grok"
    if engine == "grok":
        w, h = trim_size(s)
        aspect = "1:1" if abs(w - h) < .25 else ("3:4" if h > w else "4:3")
        payload = {
            "model": IMAGE_MODEL, "prompt": prompt[:6000],
            "n": 1, "resolution": "1k", "aspect_ratio": aspect,
        }
        for ref in p.get("reference_images", []):
            if ref.get("key_tag") == api_key_tag() and ref.get("file_id"):
                continue
            local_ref = project_dir(project_id) / str(ref.get("path") or "")
            if local_ref.exists():
                ref["file_id"] = xai_upload_file(local_ref)
                ref["key_tag"] = api_key_tag()
        save_project(p)
        refs = [r for r in p.get("reference_images", []) if r.get("file_id")][:3]
        endpoint = "/images/generations"
        if refs:
            endpoint = "/images/edits"
            payload["images"] = [{"file_id": r["file_id"]} for r in refs]
        data = xai_json(XAI_BASE + endpoint, payload, timeout=600)
        items = data.get("data") or []
        if not items:
            raise RuntimeError("xAI returned no generated image")
        raw = download_image(items[0])
    else:
        references = local_reference_images(p, target, engine)
        if engine == "flux" and not references:
            # Kontext is reference-led. Make a private HiDream starter, then let
            # FLUX produce the selected final image from that reference.
            starter = run_local_image("hidream", prompt, s, [])
            references = ["data:image/png;base64," + base64.b64encode(starter).decode()]
        raw = run_local_image(engine, prompt, s, references)
    path = project_dir(project_id) / "images" / filename
    path.write_bytes(raw)
    rel = "images/" + filename
    if target == "cover":
        p["cover"]["image"] = rel
        p["cover"]["approved"] = False
    else:
        p["pages"][int(target) - 1]["image"] = rel
        p["pages"][int(target) - 1]["approved"] = False
    p["stage"] = "illustrations"
    p["history"].append({
        "at": now(),
        "action": (
            f"Regenerated {'cover' if target == 'cover' else f'page {target}'} image"
            if had_existing_image
            else f"Generated {'cover' if target == 'cover' else f'page {target}'} image"
        ) + f" with {next(x['label'] for x in IMAGE_ENGINES if x['id'] == engine)}",
    })
    save_project(p)
    return {"project": p, "image": rel, "image_engine": engine}


def upload_reference(project_id: str, data: dict) -> dict:
    p = load_project(project_id)
    refs = p.setdefault("reference_images", [])
    character_name = str(data.get("character_name") or "").strip()
    character_index = int(data.get("character_index", -1))
    matching = [
        i for i, ref in enumerate(refs)
        if ref.get("slot") == character_index
        or (str(ref.get("character_name", "")).casefold()
            == character_name.casefold() and character_name)
    ]
    if not matching and len(refs) >= 3:
        raise ValueError("Grok supports up to three reference images per illustration")
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,(.+)",
                         data.get("data", ""), re.S)
    if not match:
        raise ValueError("Please choose a PNG, JPEG or WebP image")
    raw = base64.b64decode(match.group(2), validate=True)
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Reference image must be smaller than 15 MB")
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[match.group(1)]
    folder = project_dir(project_id) / "references"
    folder.mkdir(exist_ok=True)
    filename = f"reference-{len(refs) + 1}-{slug(data.get('name', 'image'))}{ext}"
    path = folder / filename
    path.write_bytes(raw)
    # Keep the local reference even if xAI is temporarily unavailable. Grok
    # uploads it lazily when that engine is actually selected.
    file_id = ""
    key_tag = ""
    if api_key():
        try:
            file_id = xai_upload_file(path)
            key_tag = api_key_tag()
        except Exception:
            pass
    entry = {"name": data.get("name") or filename,
             "label": data.get("label") or f"Identity reference for {character_name}",
             "path": "references/" + filename, "file_id": file_id,
             "character_name": character_name, "key_tag": key_tag,
             "slot": character_index}
    if matching:
        refs[matching[0]] = entry
        for duplicate in reversed(matching[1:]):
            refs.pop(duplicate)
    else:
        refs.append(entry)
    characters = p.setdefault("character_bible", [])
    if 0 <= character_index < len(characters):
        characters[character_index]["reference_image"] = entry["path"]
        characters[character_index]["reference_file_id"] = file_id
        characters[character_index]["reference_slot"] = character_index
    save_project(p)
    return p


def trim_size(settings: dict) -> tuple[float, float]:
    row = next((x for x in TRIMS if x[0] == settings.get("trim")), TRIMS[0])
    return row[2], row[3]


def image_abs(p: dict, rel: str) -> Path | None:
    if not rel:
        return None
    path = project_dir(p["id"]) / rel
    return path if path.exists() else None


def export_pdf(p: dict, out: Path, include_cover=True, bleed=False,
               pad_even=False) -> None:
    from PIL import Image, ImageOps
    from reportlab.lib.colors import Color, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    w_in, h_in = trim_size(p["settings"])
    render_w = w_in + (.125 if bleed else 0)
    render_h = h_in + (.25 if bleed else 0)
    page_w, page_h = render_w * 72, render_h * 72
    regular = "BookArial"
    bold = "BookArialBold"
    try:
        if regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(
                regular, "/System/Library/Fonts/Supplemental/Arial.ttf"))
        if bold not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(
                bold, "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
    except Exception:
        regular, bold = "Helvetica", "Helvetica-Bold"

    doc = canvas.Canvas(str(out), pagesize=(page_w, page_h),
                        pageCompression=1, pdfVersion=(1, 7))
    doc.setTitle(p.get("title") or "Kindle Book")
    doc.setAuthor(p.get("settings", {}).get("author") or "Author")

    def wrap(text: str, font_name: str, font_size: float,
             max_width: float, preserve_blank=False) -> list[str]:
        lines = []
        for paragraph in (text or "").split("\n") or [""]:
            if not paragraph.strip():
                if preserve_blank:
                    lines.append("")
                continue
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = (line + " " + word).strip()
                if line and pdfmetrics.stringWidth(
                        candidate, font_name, font_size) > max_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                lines.append(line)
        return lines

    def draw_background(path: Path | None, cover=False) -> None:
        if not path:
            doc.setFillColor(
                Color(.20, .14, .20) if cover else Color(.97, .96, .94))
            doc.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            return
        target = (max(1, int(render_w * 300)), max(1, int(render_h * 300)))
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            prepared = ImageOps.fit(
                source, target, method=Image.Resampling.LANCZOS,
                centering=(.5, .5))
            panel_h = int(target[1] * (.36 if cover else .50))
            mask = Image.linear_gradient("L").resize(
                (target[0], panel_h), Image.Resampling.BICUBIC)
            if cover:
                mask = mask.point(lambda value: int(18 + value * .80))
                overlay = Image.new("RGB", (target[0], panel_h), (18, 8, 15))
            else:
                mask = mask.point(lambda value: int(12 + value * .88))
                overlay = Image.new("RGB", (target[0], panel_h), "white")
            prepared.paste(
                overlay, (0, target[1] - panel_h), mask)
            buffer = io.BytesIO()
            prepared.save(buffer, "PNG", compress_level=1)
        buffer.seek(0)
        doc.drawImage(
            ImageReader(buffer), 0, 0, width=page_w, height=page_h,
            preserveAspectRatio=False, mask="auto")

    def alpha(value: float) -> None:
        try:
            doc.setFillAlpha(value)
        except Exception:
            pass

    def reset_alpha() -> None:
        try:
            doc.setFillAlpha(1)
        except Exception:
            pass

    def draw_heading(title: str, cover=False) -> None:
        size = 30 if cover else 24
        max_width = page_w * .82
        lines = wrap(title, bold, size, max_width)
        while len(lines) > 2 and size > 16:
            size -= 1
            lines = wrap(title, bold, size, max_width)
        line_h = size * 1.18
        box_h = max(line_h * len(lines) + 22, 50)
        box_x, box_w = page_w * .06, page_w * .88
        box_y = page_h - box_h - page_h * .035
        doc.setFillColor(Color(.08, .04, .07))
        alpha(.66)
        doc.roundRect(box_x, box_y, box_w, box_h, 12,
                      fill=1, stroke=0)
        reset_alpha()
        doc.setFillColor(white)
        doc.setFont(bold, size)
        y = box_y + box_h - 16 - size
        for line in lines:
            doc.drawCentredString(page_w / 2, y, line)
            y -= line_h

    def draw_body(body: str, cover=False) -> None:
        panel_h = page_h * (.36 if cover else .50)
        font_name = bold
        size = 24 if not cover else 20
        max_width = page_w * .84
        max_height = panel_h * .78
        lines = wrap(body, font_name, size, max_width, preserve_blank=True)
        while lines and len(lines) * size * 1.24 > max_height and size > 7:
            size -= .5
            lines = wrap(body, font_name, size, max_width, preserve_blank=True)
        line_h = size * 1.24
        total_h = len(lines) * line_h
        y = (panel_h + total_h) / 2 - size
        doc.setFillColor(white if cover else Color(.08, .06, .08))
        doc.setFont(font_name, size)
        for line in lines:
            if line:
                doc.drawCentredString(page_w / 2, y, line)
            y -= line_h

    def compose(path: Path | None, title: str, body: str,
                cover=False) -> None:
        draw_background(path, cover)
        draw_heading(title, cover)
        draw_body(body, cover)
        doc.showPage()

    if include_cover:
        compose(image_abs(p, p["cover"].get("image")),
                p["cover"].get("title") or p["title"],
                p["cover"].get("subtitle", ""), True)
    for page in p["pages"]:
        body = page.get("text", "")
        if page.get("dialogue"):
            body += "\n" + " ".join(map(str, page["dialogue"]))
        compose(image_abs(p, page.get("image")),
                page.get("heading", ""), body)
    if pad_even and len(p.get("pages", [])) % 2:
        doc.showPage()
    doc.save()


def export_print_cover(p: dict, out: Path, page_count: int) -> None:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from reportlab.lib.colors import Color, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    w_in, h_in = trim_size(p["settings"])
    interior = p.get("settings", {}).get("print_interior", "premium_colour")
    rate = .002347 if interior == "premium_colour" else .002252
    spine_in = page_count * rate
    bleed_in = .125
    full_w_in = bleed_in + w_in + spine_in + w_in + bleed_in
    full_h_in = bleed_in + h_in + bleed_in
    full_w, full_h = full_w_in * 72, full_h_in * 72
    bleed, trim_w, trim_h = bleed_in * 72, w_in * 72, h_in * 72
    back_left = bleed
    back_right = back_left + trim_w
    spine_left = back_right
    front_left = spine_left + spine_in * 72

    regular, bold = "CoverArial", "CoverArialBold"
    try:
        if regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(
                regular, "/System/Library/Fonts/Supplemental/Arial.ttf"))
        if bold not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(
                bold, "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
    except Exception:
        regular, bold = "Helvetica", "Helvetica-Bold"

    doc = canvas.Canvas(str(out), pagesize=(full_w, full_h),
                        pageCompression=1, pdfVersion=(1, 7))
    doc.setTitle((p.get("title") or "Book") + " - Print Cover")
    cover_path = image_abs(p, p.get("cover", {}).get("image"))
    target = (max(1, int(full_w_in * 300)), max(1, int(full_h_in * 300)))
    background = Image.new("RGB", target, (70, 46, 63))
    if cover_path:
        with Image.open(cover_path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            panel_size = (max(1, int((w_in + bleed_in) * 300)),
                          max(1, int(full_h_in * 300)))
            front = ImageOps.fit(
                source, panel_size, Image.Resampling.LANCZOS)
            front_x = int(front_left / 72 * 300)
            background.paste(front, (front_x, 0))
            back = ImageOps.fit(
                source, (int((w_in + bleed_in) * 300), target[1]),
                Image.Resampling.LANCZOS)
            back = ImageEnhance.Brightness(
                back.filter(ImageFilter.GaussianBlur(16))).enhance(.36)
            background.paste(back, (0, 0))
    buffer = io.BytesIO()
    background.save(buffer, "PNG", compress_level=1)
    buffer.seek(0)
    doc.drawImage(ImageReader(buffer), 0, 0, full_w, full_h,
                  preserveAspectRatio=False, mask="auto")

    def wrap(text: str, font: str, size: float,
             max_width: float) -> list[str]:
        lines, line = [], ""
        for word in (text or "").split():
            candidate = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(
                    candidate, font, size) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    def alpha(value: float) -> None:
        try:
            doc.setFillAlpha(value)
        except Exception:
            pass

    def reset_alpha() -> None:
        try:
            doc.setFillAlpha(1)
        except Exception:
            pass

    # Front-cover typography.
    title = p.get("cover", {}).get("title") or p.get("title") or ""
    subtitle = p.get("cover", {}).get("subtitle") or p.get("subtitle") or ""
    box_x = front_left + trim_w * .07
    box_w = trim_w * .86
    title_size = 28
    title_lines = wrap(title, bold, title_size, box_w - 24)
    while len(title_lines) > 3 and title_size > 18:
        title_size -= 1
        title_lines = wrap(title, bold, title_size, box_w - 24)
    title_h = len(title_lines) * title_size * 1.16 + 24
    title_y = full_h - bleed - title_h - 24
    doc.setFillColor(Color(.08, .04, .07))
    alpha(.68)
    doc.roundRect(box_x, title_y, box_w, title_h, 12, fill=1, stroke=0)
    reset_alpha()
    doc.setFillColor(white)
    doc.setFont(bold, title_size)
    y = title_y + title_h - title_size - 12
    for line in title_lines:
        doc.drawCentredString(front_left + trim_w / 2, y, line)
        y -= title_size * 1.16
    if subtitle:
        sub_size = 16
        sub_lines = wrap(subtitle, bold, sub_size, box_w - 20)
        sub_h = len(sub_lines) * sub_size * 1.18 + 20
        sub_y = bleed + 28
        doc.setFillColor(Color(.08, .04, .07))
        alpha(.72)
        doc.roundRect(box_x, sub_y, box_w, sub_h, 10, fill=1, stroke=0)
        reset_alpha()
        doc.setFillColor(white)
        doc.setFont(bold, sub_size)
        y = sub_y + sub_h - sub_size - 9
        for line in sub_lines:
            doc.drawCentredString(front_left + trim_w / 2, y, line)
            y -= sub_size * 1.18

    # Back-cover copy and reserved barcode area.
    safe_x = back_left + 28
    safe_w = trim_w - 56
    safe_top = full_h - bleed - 34
    doc.setFillColor(white)
    doc.setFont(bold, 21)
    doc.drawCentredString(back_left + trim_w / 2, safe_top,
                          p.get("title") or "")
    description = str(p.get("metadata", {}).get("description") or
                      p.get("story_summary") or "")
    body_size = 13
    body_lines = wrap(description, regular, body_size, safe_w)
    max_lines = max(1, int((trim_h - 190) / (body_size * 1.35)))
    while len(body_lines) > max_lines and body_size > 9:
        body_size -= .5
        body_lines = wrap(description, regular, body_size, safe_w)
        max_lines = max(1, int((trim_h - 190) / (body_size * 1.35)))
    doc.setFont(regular, body_size)
    y = safe_top - 42
    for line in body_lines[:max_lines]:
        doc.drawString(safe_x, y, line)
        y -= body_size * 1.35
    author = p.get("settings", {}).get("author") or ""
    if author:
        doc.setFont(bold, 13)
        doc.drawString(safe_x, bleed + 118, "By " + author)
    barcode_w, barcode_h = 2 * 72, 1.2 * 72
    barcode_x = back_right - barcode_w - 18
    barcode_y = bleed + 18
    doc.setFillColor(white)
    doc.roundRect(barcode_x, barcode_y, barcode_w, barcode_h,
                  4, fill=1, stroke=0)
    doc.setFillColor(Color(.38, .38, .38))
    doc.setFont(regular, 8)
    doc.drawCentredString(
        barcode_x + barcode_w / 2, barcode_y + barcode_h / 2,
        "Reserved for Amazon barcode")

    if page_count >= 79 and spine_in * 72 >= 18:
        doc.saveState()
        doc.translate(spine_left + spine_in * 36, full_h / 2)
        doc.rotate(90)
        spine_title = (p.get("title") or "")[:80]
        size = min(12, max(7, spine_in * 72 * .45))
        doc.setFillColor(white)
        doc.setFont(bold, size)
        doc.drawCentredString(0, -size / 3, spine_title)
        doc.restoreState()
    doc.save()


def export_docx(p: dict, out: Path) -> None:
    from docx import Document
    doc = Document()
    doc.add_heading(p["title"], 0)
    if p.get("subtitle"):
        doc.add_paragraph(p["subtitle"])
    doc.add_paragraph("By " + (p["settings"].get("author") or "Author"))
    doc.add_page_break()
    for page in p["pages"]:
        doc.add_heading(page.get("heading") or f"Page {page['number']}", level=1)
        if page.get("image"):
            img = image_abs(p, page["image"])
            if img:
                doc.add_picture(str(img), width=None)
        doc.add_paragraph(page.get("text", ""))
        for dialogue in page.get("dialogue") or []:
            doc.add_paragraph(str(dialogue))
    doc.save(out)


def xhtml_page(title: str, text: str, image_name: str | None,
               css_class="book-page", viewport: tuple[int, int] | None = None,
               body_size: int | None = None) -> str:
    image = (f'<img src="../images/{html.escape(image_name)}" alt="Illustration"/>'
             if image_name else "")
    paras = "".join(
        f"<p>{html.escape(line)}</p>" if line.strip()
        else '<p class="blank">&#160;</p>'
        for line in (text or "").split("\n")
    )
    viewport_meta = (
        f'<meta name="viewport" content="width={viewport[0]},height={viewport[1]}"/>'
        if viewport else "")
    body_style = f' style="--body-size:{body_size}px"' if body_size else ""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head>
<title>{html.escape(title)}</title>{viewport_meta}
<link rel="stylesheet" href="../styles/book.css" type="text/css"/>
</head><body class="{css_class}"><main>{image}<section class="copy">
<h1>{html.escape(title)}</h1><div class="body-copy"{body_style}>{paras}</div>
</section></main></body></html>"""


def fixed_epub_body_size(text: str, viewport: tuple[int, int],
                         cover=False) -> int:
    max_size = 34 if cover else 46
    min_size = 14
    available_width = viewport[0] * .86
    available_height = viewport[1] * (.30 if cover else .50) - 64
    logical_lines = (text or "").split("\n")
    for size in range(max_size, min_size - 1, -1):
        chars_per_line = max(8, int(available_width / (size * .54)))
        visual_lines = sum(
            1 if not line.strip() else
            max(1, math.ceil(len(line.expandtabs(4)) / chars_per_line))
            for line in logical_lines
        )
        if visual_lines * size * 1.18 <= available_height:
            return size
    return min_size


def export_epub(p: dict, out: Path) -> None:
    fixed = p["settings"].get("layout") == "fixed"
    w_in, h_in = trim_size(p["settings"])
    viewport = (int(w_in * 150), int(h_in * 150)) if fixed else None
    entries: dict[str, bytes] = {}
    entries["META-INF/container.xml"] = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    if fixed:
        entries["OEBPS/styles/book.css"] = b"""html,body{margin:0;width:100%;height:100%;overflow:hidden}
body{font-family:serif;color:#111;background:#332334}main{position:relative;width:100vw;height:100vh;
box-sizing:border-box;overflow:hidden}main>img{position:absolute;inset:0;display:block;width:100%;
height:100%;object-fit:cover}.copy{position:absolute;inset:0;display:grid;grid-template-rows:auto 1fr;
box-sizing:border-box}.copy h1{align-self:start;text-align:center;font-size:38px;line-height:1.12;
margin:28px 5% 0;padding:14px 24px;border-radius:16px;color:#fff;background:rgba(20,12,18,.62);
text-shadow:0 2px 8px #000}.body-copy{align-self:end;height:50%;display:flex;flex-direction:column;
align-items:center;justify-content:center;padding:32px 7%;box-sizing:border-box;
overflow:hidden;background:linear-gradient(transparent 0%,rgba(255,255,255,.74) 24%,rgba(255,255,255,.92) 100%)}
p{text-align:center;font-size:var(--body-size,46px);font-weight:600;line-height:1.18;margin:0;color:#171219;
text-shadow:0 1px 1px rgba(255,255,255,.8)}p.blank{min-height:1.18em}
.cover .copy h1{font-size:52px;margin-top:42px;
background:rgba(20,10,18,.68)}.cover .body-copy{height:36%;padding-bottom:44px;
background:linear-gradient(transparent,rgba(20,10,18,.86))}.cover p{font-size:var(--body-size,34px);color:#fff;
text-shadow:0 2px 8px #000}"""
    else:
        entries["OEBPS/styles/book.css"] = b"""body{margin:0;font-family:serif;color:#111;background:#fff}
main{padding:4%;box-sizing:border-box}img{display:block;width:100%;height:auto;margin:0 auto 1rem}
.copy{max-width:48em;margin:auto}h1{text-align:center;font-size:1.5em}
p{font-size:1.1em;line-height:1.45;margin:.5em 0}.cover main{padding:0}
.blank{min-height:1.45em}.cover .copy{text-align:center;padding:0 5% 5%}"""
    manifest = [
        '<item id="css" href="styles/book.css" media-type="text/css"/>',
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
    ]
    spine = []
    nav = []
    images_seen = {}

    def add_image(rel):
        if not rel:
            return None
        src = image_abs(p, rel)
        if not src:
            return None
        name = src.name
        if name not in images_seen:
            data = src.read_bytes()
            try:
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=92, optimize=True)
                    data = buf.getvalue()
                name = src.stem + ".jpg"
            except Exception:
                pass
            entries["OEBPS/images/" + name] = data
            iid = "img" + str(len(images_seen) + 1)
            images_seen[src.name] = (iid, name)
            manifest.append(f'<item id="{iid}" href="images/{name}" media-type="image/jpeg"/>')
        return images_seen[src.name][1]

    cover_img = add_image(p["cover"].get("image"))
    entries["OEBPS/text/cover.xhtml"] = xhtml_page(
        p["cover"].get("title") or p["title"],
        p["cover"].get("subtitle", ""), cover_img, "cover", viewport,
        fixed_epub_body_size(
            p["cover"].get("subtitle", ""), viewport, cover=True
        ) if fixed else None).encode()
    cover_image_id = ""
    if cover_img:
        for i, item in enumerate(manifest):
            if f'href="images/{cover_img}"' in item:
                cover_image_id = re.search(r'id="([^"]+)"', item).group(1)
                manifest[i] = item.replace("/>", ' properties="cover-image"/>')
                break
    manifest.append('<item id="cover" href="text/cover.xhtml" media-type="application/xhtml+xml"/>')
    spine.append('<itemref idref="cover"/>')
    nav.append('<li><a href="text/cover.xhtml">Cover</a></li>')
    for page in p["pages"]:
        pid = f"p{page['number']}"
        fname = f"page-{page['number']:03d}.xhtml"
        img = add_image(page.get("image"))
        body = page.get("text", "")
        if page.get("dialogue"):
            body += "\n" + "\n".join(map(str, page["dialogue"]))
        entries["OEBPS/text/" + fname] = xhtml_page(
            page.get("heading") or f"Page {page['number']}", body, img,
            "book-page", viewport,
            fixed_epub_body_size(body, viewport) if fixed else None).encode()
        manifest.append(f'<item id="{pid}" href="text/{fname}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{pid}"/>')
        nav.append(f'<li><a href="text/{fname}">Page {page["number"]}</a></li>')
    entries["OEBPS/nav.xhtml"] = ("""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Contents</title></head>
<body><nav epub:type="toc" xmlns:epub="http://www.idpf.org/2007/ops"><h1>Contents</h1><ol>"""
        + "".join(nav) + "</ol></nav></body></html>").encode()
    ident = "urn:uuid:" + uuid.uuid5(uuid.NAMESPACE_URL, p["id"]).hex
    if fixed:
        orientation = "landscape" if w_in > h_in else "portrait"
        rendition = (
            '<meta property="rendition:layout">pre-paginated</meta>'
            '<meta property="rendition:spread">none</meta>'
            f'<meta property="rendition:orientation">{orientation}</meta>'
        )
    else:
        rendition = '<meta property="rendition:layout">reflowable</meta>'
    legacy_cover = (f'<meta name="cover" content="{cover_image_id}"/>'
                    if cover_image_id else "")
    entries["OEBPS/content.opf"] = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="bookid">{ident}</dc:identifier><dc:title>{html.escape(p["title"])}</dc:title>
<dc:language>en</dc:language><dc:creator>{html.escape(p["settings"].get("author") or "Author")}</dc:creator>
<dc:date>{datetime.now().date().isoformat()}</dc:date>{legacy_cover}{rendition}</metadata>
<manifest>{''.join(manifest)}</manifest><spine page-progression-direction="ltr">{''.join(spine)}</spine>
</package>""".encode()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for name, data in entries.items():
            z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)


def export_project(project_id: str) -> dict:
    p = load_project(project_id)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = EXPORT_DIR / f"{slug(p['title'])}-{stamp}"
    epub = base.with_suffix(".epub")
    proof_pdf = base.parent / (base.name + "-review.pdf")
    manuscript_pdf = base.parent / (base.name + "-manuscript.pdf")
    cover_pdf = base.parent / (base.name + "-book-cover.pdf")
    docx = base.with_suffix(".docx")
    metadata = base.with_suffix(".metadata.json")
    bundle = base.with_suffix(".zip")
    export_epub(p, epub)
    export_pdf(p, proof_pdf)
    story_pages = len(p.get("pages") or [])
    print_page_count = story_pages + (story_pages % 2)
    print_eligible = print_page_count >= 24
    if print_eligible:
        export_pdf(p, manuscript_pdf, include_cover=False,
                   bleed=True, pad_even=True)
        export_print_cover(p, cover_pdf, print_page_count)
    export_docx(p, docx)
    w_in, h_in = trim_size(p["settings"])
    interior = p.get("settings", {}).get("print_interior", "premium_colour")
    spine_rate = .002347 if interior == "premium_colour" else .002252
    print_setup = {
        "binding": "Paperback",
        "trim_size": f"{w_in:g} × {h_in:g} inches",
        "interior": ("Premium colour" if interior == "premium_colour"
                     else "Standard colour"),
        "paper": "White",
        "bleed": "Yes",
        "page_count": print_page_count,
        "minimum_page_count": 24,
        "print_eligible": print_eligible,
        "spine_width_inches": round(print_page_count * spine_rate, 4),
        "warning": ("" if print_eligible else
                    "Amazon KDP paperback requires at least 24 interior pages. "
                    "Choose a longer book before submitting print files."),
    }
    metadata.write_text(json.dumps({
        "title": p["title"], "subtitle": p.get("subtitle"),
        "author": p["settings"].get("author"),
        "print_setup": print_setup, **(p.get("metadata") or {}),
    }, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        package_paths = [epub, proof_pdf, docx, metadata]
        if print_eligible:
            package_paths.extend([manuscript_pdf, cover_pdf])
        for path in package_paths:
            z.write(path, path.name)
        z.write(project_file(project_id), "project.json")
        for image in (project_dir(project_id) / "images").glob("*"):
            if image.is_file():
                z.write(image, "images/" + image.name)
        for reference in (project_dir(project_id) / "references").glob("*"):
            if reference.is_file():
                z.write(reference, "references/" + reference.name)
    p["stage"] = "exported"
    p["history"].append({"at": now(), "action": "Exported Kindle package"})
    save_project(p)
    return {
        "epub": str(epub), "pdf": str(proof_pdf),
        "manuscript_pdf": str(manuscript_pdf) if print_eligible else "",
        "cover_pdf": str(cover_pdf) if print_eligible else "",
        "print_setup": print_setup, "docx": str(docx),
        "metadata": str(metadata), "bundle": str(bundle),
    }


def run_job(job_id: str, fn, *args):
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "stage": "working", "progress": 5}
    try:
        result = fn(*args)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "stage": "done",
                             "progress": 100, "result": result}
    except Exception as e:
        raw_error = f"{type(e).__name__}: {e}"
        if "used all available credits" in raw_error or "monthly spending limit" in raw_error:
            shown_error = (
                "xAI could not start this request because the account has no "
                "available API credits or has reached its monthly spending limit. "
                "Add credits or raise the spending limit in the xAI Console, then try again."
            )
        else:
            shown_error = raw_error
        print(f"[{now()}] job {job_id} failed: {raw_error}", file=sys.stderr, flush=True)
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "stage": "error",
                             "progress": 100, "error": shown_error}


def start_job(fn, *args) -> str:
    jid = "job_" + uuid.uuid4().hex[:12]
    threading.Thread(target=run_job, args=(jid, fn, *args), daemon=True).start()
    return jid


class Handler(BaseHTTPRequestHandler):
    server_version = "omlx-kindle/1.0"

    def log_message(self, *args):
        pass

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def json(self, code: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204); self.cors(); self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self.serve_file(STATIC_DIR / "index.html")
            if path == "/health":
                return self.json(200, {
                    "ok": True, "service": "Kindle Book Studio",
                    "grok_key_configured": bool(api_key()), "text_model": TEXT_MODEL,
                    "image_model": IMAGE_MODEL, "projects": len(list_projects()),
                })
            if path == "/api/presets":
                return self.json(200, {
                    "book_types": [{"id": x, "label": y} for x, y in BOOK_TYPES],
                    "genres": GENRES,
                    "image_styles": [{"id": x, "label": y} for x, y in IMAGE_STYLES],
                    "trims": [{"id": x[0], "label": x[1]} for x in TRIMS],
                    "reading_levels": READING_LEVELS, "page_counts": PAGE_COUNTS,
                    "series_name": SERIES_NAME, "series_books": SERIES_BOOKS,
                    "text_model": TEXT_MODEL, "image_model": IMAGE_MODEL,
                    "image_engines": IMAGE_ENGINES,
                })
            if path == "/api/projects":
                return self.json(200, {"projects": list_projects()})
            if path.startswith("/api/projects/"):
                parts = path.strip("/").split("/")
                pid = parts[2]
                if len(parts) == 3:
                    return self.json(200, load_project(pid))
                if len(parts) == 5 and parts[3] == "images":
                    rel = "images/" + os.path.basename(parts[4])
                    target = image_abs(load_project(pid), rel)
                    if not target:
                        return self.json(404, {"error": "image not found"})
                    return self.serve_file(target)
                if len(parts) == 5 and parts[3] == "references":
                    rel = "references/" + os.path.basename(parts[4])
                    target = project_dir(pid) / rel
                    if not target.exists() or not target.is_file():
                        return self.json(404, {"error": "reference image not found"})
                    return self.serve_file(target)
            if path.startswith("/api/jobs/"):
                jid = path.rsplit("/", 1)[-1]
                with _jobs_lock:
                    job = dict(_jobs.get(jid) or {})
                return self.json(200 if job else 404, job or {"error": "unknown job"})
            return self.json(404, {"error": "not found"})
        except Exception as e:
            return self.json(500, {"error": f"{type(e).__name__}: {e}"})

    def serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            return self.json(404, {"error": "not found"})
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or
                         "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.cors()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        data = self.body()
        try:
            if path == "/api/projects":
                return self.json(201, new_project(data))
            if path.startswith("/api/projects/"):
                parts = path.strip("/").split("/")
                pid = parts[2]
                action = parts[3] if len(parts) > 3 else ""
                if action == "generate":
                    return self.json(202, {"job_id": start_job(generate_story, pid)})
                if action == "populate-titles":
                    return self.json(202, {"job_id": start_job(populate_titles, pid)})
                if action == "generate-metadata":
                    return self.json(202, {"job_id": start_job(generate_metadata, pid)})
                if action == "regenerate-cover":
                    return self.json(202, {"job_id": start_job(
                        regenerate_cover, pid, data.get("request", ""))})
                if action == "regenerate-page":
                    return self.json(202, {"job_id": start_job(
                        regenerate_page, pid, int(data["page"]), data.get("request", ""))})
                if action == "generate-image":
                    return self.json(202, {"job_id": start_job(
                        generate_image, pid, str(data["target"]))})
                if action == "upload-reference":
                    return self.json(200, upload_reference(pid, data))
                if action == "export":
                    return self.json(202, {"job_id": start_job(export_project, pid)})
            return self.json(404, {"error": "not found"})
        except Exception as e:
            return self.json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_PUT(self):
        path = urllib.parse.urlparse(self.path).path
        data = self.body()
        try:
            if path.startswith("/api/projects/"):
                pid = path.strip("/").split("/")[2]
                current = load_project(pid)
                revision_snapshot(current, "manual-save")
                data["id"] = pid
                data.setdefault("created_at", current.get("created_at", now()))
                data.setdefault("history", current.get("history", []))
                return self.json(200, save_project(
                    preserve_server_assets(current, data)))
            return self.json(404, {"error": "not found"})
        except Exception as e:
            return self.json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"Kindle Book Studio ready at http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
