#!/usr/bin/env python3
"""Render styled caption + hook-card PNG overlays for the viral-edit pipeline.

Input (stdin or --json file): a spec describing the output canvas and a list of
overlay items. Each item produces one transparent PNG sized to the full canvas
(so ffmpeg can overlay at 0,0 with an enable=between(t,a,b) expression).

Spec:
{
  "width": 1080, "height": 1920,
  "outdir": "/tmp/caps",
  "items": [
    {"id": "c0", "type": "caption", "text": "THIS changes everything",
     "emphasis": ["THIS"]},
    {"id": "k0", "type": "card", "title": "The 3-second rule",
     "subtitle": "why hooks live or die here"}
  ]
}

Writes <outdir>/<id>.png for each item and prints JSON {"id": "path", ...}.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
FONT_AVENIR = "/System/Library/Fonts/Avenir Next.ttc"

ACCENT = (255, 214, 10, 255)        # punchy yellow for emphasis
WHITE = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 235)


def _font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.truetype(FONT_BOLD, size)


def _wrap(draw, words, font, max_w):
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
    if cur:
        lines.append(cur)
    return lines


def _draw_outlined_word(draw, x, y, word, font, fill, ow):
    for dx in range(-ow, ow + 1):
        for dy in range(-ow, ow + 1):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), word, font=font, fill=SHADOW)
    draw.text((x, y), word, font=font, fill=fill)


def render_caption(spec, item, path):
    W, H = spec["width"], spec["height"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    text = (item.get("text") or "").strip()
    if not text:
        img.save(path)
        return
    emph = {e.strip().lower() for e in (item.get("emphasis") or []) if e.strip()}

    # Caption sits in the lower third (safe area), scales with canvas height.
    size = max(34, int(H * 0.052))
    font = _font(FONT_IMPACT, size)
    max_w = int(W * 0.86)
    ow = max(2, size // 16)

    words = text.split()
    lines = _wrap(d, words, font, max_w)
    asc, desc = font.getmetrics()
    lh = asc + desc + int(size * 0.16)
    total_h = lh * len(lines)
    y0 = int(H * 0.78) - total_h // 2
    y0 = min(y0, H - total_h - int(H * 0.06))

    y = y0
    for line in lines:
        line_w = d.textlength(" ".join(line), font=font)
        x = (W - line_w) // 2
        for w in line:
            fill = ACCENT if w.lower().strip(".,!?") in emph else WHITE
            _draw_outlined_word(d, x, y, w, font, fill, ow)
            x += d.textlength(w + " ", font=font)
        y += lh
    img.save(path)


def render_card(spec, item, path):
    W, H = spec["width"], spec["height"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    title = (item.get("title") or "").strip()
    subtitle = (item.get("subtitle") or "").strip()
    if not title:
        img.save(path)
        return

    t_size = max(48, int(H * 0.075))
    s_size = max(28, int(H * 0.034))
    t_font = _font(FONT_IMPACT, t_size)
    s_font = _font(FONT_BOLD, s_size)
    max_w = int(W * 0.84)
    ow = max(3, t_size // 14)

    t_lines = _wrap(d, title.upper().split(), t_font, max_w)
    s_lines = _wrap(d, subtitle.split(), s_font, max_w) if subtitle else []

    t_asc, t_desc = t_font.getmetrics()
    s_asc, s_desc = s_font.getmetrics()
    t_lh = t_asc + t_desc + int(t_size * 0.12)
    s_lh = s_asc + s_desc + int(s_size * 0.2)
    block_h = t_lh * len(t_lines) + (int(t_size * 0.4) + s_lh * len(s_lines)
                                     if s_lines else 0)
    # Hook cards sit in the upper-middle for impact.
    y = int(H * 0.30) - block_h // 2

    for line in t_lines:
        txt = " ".join(line)
        x = (W - d.textlength(txt, font=t_font)) // 2
        _draw_outlined_word(d, x, y, txt, t_font, ACCENT, ow)
        y += t_lh

    if s_lines:
        y += int(t_size * 0.4)
        for line in s_lines:
            txt = " ".join(line)
            x = (W - d.textlength(txt, font=s_font)) // 2
            _draw_outlined_word(d, x, y, txt, s_font, WHITE, max(2, ow // 2))
            y += s_lh
    img.save(path)


def main():
    raw = (open(sys.argv[1]).read() if len(sys.argv) > 1
           else sys.stdin.read())
    spec = json.loads(raw)
    outdir = spec["outdir"]
    os.makedirs(outdir, exist_ok=True)
    result = {}
    for item in spec.get("items", []):
        path = os.path.join(outdir, item["id"] + ".png")
        if item.get("type") == "card":
            render_card(spec, item, path)
        else:
            render_caption(spec, item, path)
        result[item["id"]] = path
    print(json.dumps(result))


if __name__ == "__main__":
    main()
