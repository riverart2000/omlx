#!/usr/bin/env python3
"""Redact secret-looking values from JSON config backups, and merge them back
without clobbering live secrets. Keeps real keys off the (public) repo."""
import json, re, sys

SENTINEL = "__SECRET_REDACTED__"
SENSITIVE = re.compile(r"(secret|password|passwd|token|bearer|api[_-]?key)", re.I)

def _is_sensitive(key):
    return bool(SENSITIVE.search(key)) and not key.lower().startswith("skip_")

def redact(o):
    if isinstance(o, dict):
        return {k: (SENTINEL if (_is_sensitive(k) and isinstance(v, str) and v)
                    else redact(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [redact(v) for v in o]
    return o

def merge(stored, live):
    """Return stored, but wherever stored == SENTINEL use the live value."""
    if isinstance(stored, dict):
        out = {}
        for k, v in stored.items():
            lv = live.get(k) if isinstance(live, dict) else None
            if v == SENTINEL:
                out[k] = lv if lv is not None else v
            else:
                out[k] = merge(v, lv)
        return out
    if isinstance(stored, list):
        return stored
    return stored

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "redact":
        p = sys.argv[2]
        data = json.load(open(p))
        json.dump(redact(data), open(p, "w"), indent=2)
        open(p, "a").write("\n")
    elif mode == "merge":
        repo_f, live_f, out_f = sys.argv[2], sys.argv[3], sys.argv[4]
        stored = json.load(open(repo_f))
        try:
            live = json.load(open(live_f))
        except Exception:
            live = {}
        json.dump(merge(stored, live), open(out_f, "w"), indent=2)
        open(out_f, "a").write("\n")
