"""Conservative local-LLM review of transcripts for abandoned false starts."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .config import OMLX_API_KEY, OMLX_API_URL, OMLX_LLM_MODEL


SYSTEM_PROMPT = """You are a conservative dialogue editor. Identify only words that are clearly an
abandoned attempt, repeated start, or self-correction and can be removed without changing meaning.
Do not remove pauses, stylistic repetition, emphasis, complete sentences, or useful content. Return
JSON only in this shape: {"cuts":[{"start":1.2,"end":2.1,"text":"...","reason":"false start","confidence":0.92}]}.
Times must come from the timestamped transcript. If uncertain, omit the cut."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {"cuts": []}


def review_false_starts(segments: list[dict], *, timeout: int = 180) -> list[dict]:
    if not segments:
        return []
    transcript_lines = []
    for segment in segments:
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        text = str(segment.get("text") or "").strip()
        timed_words = []
        for word in segment.get("words") or []:
            word_text = str(word.get("word") or "").strip()
            if not word_text:
                continue
            word_start = float(word.get("start") or 0)
            word_end = float(word.get("end") or word_start)
            timed_words.append(f"<{word_start:.2f}-{word_end:.2f}>{word_text}")
        if timed_words:
            transcript_lines.append(" ".join(timed_words))
        elif text:
            transcript_lines.append(f"[{start:.2f}-{end:.2f}] {text}")
    if not transcript_lines:
        return []
    # A bounded transcript keeps the local model responsive. Very long clips are
    # still covered by deterministic repeated-phrase detection.
    transcript = "\n".join(transcript_lines)
    if len(transcript) > 40_000:
        transcript = transcript[:40_000]
    payload = {
        "model": OMLX_LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Review this transcript and return only safe false-start cuts:\n\n" + transcript,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1800,
        "stream": False,
    }
    request = urllib.request.Request(
        OMLX_API_URL.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OMLX_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        cuts = _extract_json(content).get("cuts") or []
    except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        return []
    return cuts if isinstance(cuts, list) else []
