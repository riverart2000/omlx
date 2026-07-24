#!/usr/bin/env python3
"""oMLX Kindle Book Studio — Grok-powered story, illustration and KDP exporter."""
from __future__ import annotations

import base64
import html
import io
import json
import mimetypes
import os
import re
import shutil
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
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

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
PAGE_COUNTS = [12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128]

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


def load_project(project_id: str) -> dict:
    path = project_file(project_id)
    if not path.exists():
        raise FileNotFoundError(project_id)
    return json.loads(path.read_text())


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


def new_project(data: dict) -> dict:
    pid = "book_" + uuid.uuid4().hex[:12]
    settings = {
        "book_type": data.get("book_type", "picture_book"),
        "genre": data.get("genre", "Adventure"),
        "audience": data.get("audience", "Ages 4–6"),
        "language": data.get("language", "English"),
        "tone": data.get("tone", "Warm, engaging and imaginative"),
        "page_count": int(data.get("page_count", 24)),
        "image_style": data.get("image_style", "cinematic_3d"),
        "custom_style": data.get("custom_style", ""),
        "trim": data.get("trim", "8x10"),
        "layout": data.get("layout", "fixed"),
        "author": data.get("author", ""),
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"xAI API {e.code}: {detail}") from e


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
Dedication or personal foreword: {s.get('dedication')}
Series: {s.get('series_name')} {s.get('book_number')}
Personalisation and character notes: {s.get('personalisation')}
Characters supplied by the creator: {json.dumps(p.get('character_bible'), ensure_ascii=False)}

Build a coherent beginning, development, climax/payoff, and satisfying ending.
For children, keep age-appropriate vocabulary and page text length. For comics,
provide concise narration/dialogue and panel-aware image direction. For
non-fiction, build a useful factual progression and flag anything requiring
fact checking. Maintain exact character visual continuity. Image prompts must
describe the full scene, recurring character appearance, composition, camera,
lighting, palette, and leave safe negative space for text. Do not put lettering,
captions, logos, or watermarks inside generated images.

Also create cover direction and KDP metadata. The pages array MUST contain
exactly {count} objects numbered 1 through {count}."""


STORY_SHAPE = {
    "title": "Book title", "subtitle": "Optional subtitle",
    "story_summary": "Full synopsis",
    "character_bible": [{
        "name": "Name", "role": "Role", "appearance": "Exact reusable appearance",
        "personality": "Traits", "continuity_rules": "Never-changing details",
    }],
    "world_bible": "Locations, palette, era, props and continuity",
    "cover": {"title": "Title", "subtitle": "", "image_prompt": "Detailed cover art prompt"},
    "pages": [{
        "number": 1, "heading": "Optional heading", "text": "Final page text",
        "dialogue": [], "image_prompt": "Detailed consistent illustration prompt",
        "negative_prompt": "Unwanted elements", "layout_note": "Text and image placement",
    }],
    "metadata": {
        "description": "KDP product description", "keywords": ["7 phrases"],
        "categories": ["Suggested categories"], "age_range": "Target ages",
        "copyright_text": "Copyright page copy", "author_bio": "Editable biography",
    },
}


def normalize_story(project: dict, data: dict) -> dict:
    project["title"] = str(data.get("title") or project["title"])
    project["subtitle"] = str(data.get("subtitle") or "")
    project["story_summary"] = str(data.get("story_summary") or "")
    project["character_bible"] = data.get("character_bible") or []
    project["world_bible"] = str(data.get("world_bible") or "")
    project["metadata"] = data.get("metadata") or {}
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
        pages.append({
            "number": i + 1, "heading": str(raw.get("heading") or ""),
            "text": str(raw.get("text") or ""),
            "dialogue": raw.get("dialogue") or [],
            "image_prompt": str(raw.get("image_prompt") or ""),
            "negative_prompt": str(raw.get("negative_prompt") or
                                   "text, letters, logo, watermark, distorted anatomy"),
            "layout_note": str(raw.get("layout_note") or ""),
            "image": "", "approved": False, "text_approved": False,
        })
    project["pages"] = pages
    project["stage"] = "story"
    project["history"].append({"at": now(), "action": "Generated full book with Grok"})
    return project


def generate_story(project_id: str) -> dict:
    p = load_project(project_id)
    revision_snapshot(p, "before-full-generation")
    result = grok_structured(story_instruction(p), STORY_SHAPE, timeout=900)
    return save_project(normalize_story(p, result))


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
        "heading": "", "text": "Revised final page text", "dialogue": [],
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
    result = grok_structured(instruction, {
        "title": p["title"], "subtitle": p.get("subtitle", ""),
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
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def generate_image(project_id: str, target: str) -> dict:
    p = load_project(project_id)
    s = p["settings"]
    common = (
        f"Original book illustration. Visual style: {style_label(s)}. "
        f"Audience: {s.get('audience')}. Book continuity: {p.get('world_bible')}. "
        f"Character bible: {json.dumps(p.get('character_bible'), ensure_ascii=False)}. "
        "Maintain exact recurring character identity, clothing and palette. "
        "Professional publishable composition, no text, no letters, no logo, no watermark."
    )
    if target == "cover":
        prompt = common + "\nCOVER ART: " + p["cover"]["image_prompt"]
        filename = "cover.png"
    else:
        number = int(target)
        page = p["pages"][number - 1]
        prompt = common + f"\nPAGE {number}: " + page["image_prompt"]
        if page.get("negative_prompt"):
            prompt += "\nAvoid: " + page["negative_prompt"]
        filename = f"page-{number:03d}.png"
    w, h = trim_size(s)
    aspect = "1:1" if abs(w - h) < .25 else ("4:5" if h > w else "5:4")
    payload = {
        "model": IMAGE_MODEL, "prompt": prompt[:6000],
        "n": 1, "resolution": "1k", "aspect_ratio": aspect,
    }
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
    save_project(p)
    return {"project": p, "image": rel}


def upload_reference(project_id: str, data: dict) -> dict:
    p = load_project(project_id)
    refs = p.setdefault("reference_images", [])
    if len(refs) >= 3:
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
    file_id = xai_upload_file(path)
    refs.append({"name": data.get("name") or filename,
                 "label": data.get("label") or "Character or style reference",
                 "path": "references/" + filename, "file_id": file_id})
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


def export_pdf(p: dict, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    w_in, h_in = trim_size(p["settings"])
    dpi = 150
    size = (int(w_in * dpi), int(h_in * dpi))
    pages = []

    def font(sz):
        for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/System/Library/Fonts/SFNS.ttf"):
            try:
                return ImageFont.truetype(path, sz)
            except Exception:
                pass
        return ImageFont.load_default()

    def compose(img_path, title, body):
        canvas = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(canvas)
        if img_path:
            with Image.open(img_path) as src:
                src = ImageOps.exif_transpose(src).convert("RGB")
                max_img_h = int(size[1] * .70)
                fitted = ImageOps.fit(src, (size[0], max_img_h))
                canvas.paste(fitted, (0, 0))
        y = int(size[1] * .73)
        if title:
            draw.text((70, y), title, fill="#111", font=font(34))
            y += 52
        words = (body or "").split()
        lines, line = [], ""
        for word in words:
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font(25)) > size[0] - 140:
                lines.append(line); line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        for ln in lines[:10]:
            draw.text((70, y), ln, fill="#222", font=font(25)); y += 36
        return canvas

    pages.append(compose(image_abs(p, p["cover"].get("image")),
                         p["cover"].get("title") or p["title"],
                         p["cover"].get("subtitle", "")))
    for page in p["pages"]:
        body = page.get("text", "")
        if page.get("dialogue"):
            body += "\n" + " ".join(map(str, page["dialogue"]))
        pages.append(compose(image_abs(p, page.get("image")),
                             page.get("heading", ""), body))
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:],
                  resolution=dpi, quality=95)


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
               css_class="book-page") -> str:
    image = (f'<img src="../images/{html.escape(image_name)}" alt="Illustration"/>'
             if image_name else "")
    paras = "".join(f"<p>{html.escape(x)}</p>" for x in (text or "").split("\n") if x.strip())
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head>
<title>{html.escape(title)}</title><link rel="stylesheet" href="../styles/book.css" type="text/css"/>
</head><body class="{css_class}"><main>{image}<h1>{html.escape(title)}</h1>{paras}</main></body></html>"""


def export_epub(p: dict, out: Path) -> None:
    fixed = p["settings"].get("layout") == "fixed"
    entries: dict[str, bytes] = {}
    entries["META-INF/container.xml"] = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    entries["OEBPS/styles/book.css"] = b"""body{margin:0;font-family:serif;color:#111;background:#fff}
main{padding:5%;box-sizing:border-box}img{display:block;max-width:100%;max-height:72vh;margin:0 auto 1rem}
h1{text-align:center;font-size:1.5em}p{font-size:1.1em;line-height:1.45;margin:.5em 0}
.cover main{padding:0}.cover h1,.cover p{text-align:center;padding:0 5%}"""
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
        p["cover"].get("subtitle", ""), cover_img, "cover").encode()
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
            page.get("heading") or f"Page {page['number']}", body, img).encode()
        manifest.append(f'<item id="{pid}" href="text/{fname}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{pid}"/>')
        nav.append(f'<li><a href="text/{fname}">Page {page["number"]}</a></li>')
    entries["OEBPS/nav.xhtml"] = ("""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Contents</title></head>
<body><nav epub:type="toc" xmlns:epub="http://www.idpf.org/2007/ops"><h1>Contents</h1><ol>"""
        + "".join(nav) + "</ol></nav></body></html>").encode()
    ident = "urn:uuid:" + uuid.uuid5(uuid.NAMESPACE_URL, p["id"]).hex
    rendition = ('<meta property="rendition:layout">pre-paginated</meta>'
                 if fixed else '<meta property="rendition:layout">reflowable</meta>')
    entries["OEBPS/content.opf"] = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="bookid">{ident}</dc:identifier><dc:title>{html.escape(p["title"])}</dc:title>
<dc:language>en</dc:language><dc:creator>{html.escape(p["settings"].get("author") or "Author")}</dc:creator>
<dc:date>{datetime.now().date().isoformat()}</dc:date>{rendition}</metadata>
<manifest>{''.join(manifest)}</manifest><spine>{''.join(spine)}</spine></package>""".encode()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for name, data in entries.items():
            z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)


def export_project(project_id: str) -> dict:
    p = load_project(project_id)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = EXPORT_DIR / f"{slug(p['title'])}-{stamp}"
    epub = base.with_suffix(".epub")
    pdf = base.with_suffix(".pdf")
    docx = base.with_suffix(".docx")
    metadata = base.with_suffix(".metadata.json")
    bundle = base.with_suffix(".zip")
    export_epub(p, epub)
    export_pdf(p, pdf)
    export_docx(p, docx)
    metadata.write_text(json.dumps({
        "title": p["title"], "subtitle": p.get("subtitle"),
        "author": p["settings"].get("author"), **(p.get("metadata") or {}),
    }, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for path in (epub, pdf, docx, metadata):
            z.write(path, path.name)
        z.write(project_file(project_id), "project.json")
        for image in (project_dir(project_id) / "images").glob("*"):
            if image.is_file():
                z.write(image, "images/" + image.name)
    p["stage"] = "exported"
    p["history"].append({"at": now(), "action": "Exported Kindle package"})
    save_project(p)
    return {"epub": str(epub), "pdf": str(pdf), "docx": str(docx),
            "metadata": str(metadata), "bundle": str(bundle)}


def run_job(job_id: str, fn, *args):
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "stage": "working", "progress": 5}
    try:
        result = fn(*args)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "stage": "done",
                             "progress": 100, "result": result}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "stage": "error",
                             "progress": 100, "error": f"{type(e).__name__}: {e}"}


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
                    "text_model": TEXT_MODEL, "image_model": IMAGE_MODEL,
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
                return self.json(200, save_project(data))
            return self.json(404, {"error": "not found"})
        except Exception as e:
            return self.json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"Kindle Book Studio ready at http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
