#!/bin/bash
# Copy files FROM this repo back to their live locations (overwrites live files).
# Config secrets that were redacted in the repo are preserved from your live
# files (the redacted placeholder is never written over a real key).
# Revert workflow:
#   git log --oneline                  # find the version you want
#   git checkout <commit> -- .         # load that version into the working tree
#   ./restore.sh                       # write it back to the live locations
#   git checkout customizations -- .   # (optional) return working tree to latest
set -euo pipefail
cd "$(dirname "$0")"

echo "This will OVERWRITE your live files with the versions in this repo."
read -r -p "Continue? [y/N] " ans
case "$ans" in y|Y) ;; *) echo "Aborted."; exit 1;; esac

n=0
while IFS='|' read -r live repo; do
  [ -z "${live:-}" ] && continue
  case "$live" in \#*) continue;; esac
  if [ ! -e "$repo" ]; then
    echo "  ! not in repo (skipped): $repo"; continue
  fi
  mkdir -p "$(dirname "$live")"
  case "$repo" in
    config/*.json)
      # Merge: take repo version but keep live secret values for redacted fields.
      tmp="$(mktemp)"
      python3 _secrets.py merge "$repo" "$live" "$tmp"
      cp "$tmp" "$live"; rm -f "$tmp";;
    *)
      cp -p "$repo" "$live";;
  esac
  echo "  -> $live"
  n=$((n+1))
done < files.txt
echo "restored $n file(s)."
echo
echo "Apply changes by restarting the services:"
echo "  bash /Users/joebains/mlx-audio/tts-studio.sh restart   # TTS Studio"
echo "  (restart the oMLX app to reload chat.html)"
