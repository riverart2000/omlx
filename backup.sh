#!/bin/bash
# Copy live customized files into this repo, commit, and push to GitHub.
# Secret values in config/*.json are redacted before commit (repo is public).
# Usage: ./backup.sh ["commit message"]
set -euo pipefail
cd "$(dirname "$0")"
MSG="${1:-backup $(date '+%Y-%m-%d %H:%M:%S')}"

n=0; missing=0
while IFS='|' read -r live repo; do
  [ -z "${live:-}" ] && continue
  case "$live" in \#*) continue;; esac
  if [ ! -e "$live" ]; then
    echo "  ! missing (skipped): $live"; missing=$((missing+1)); continue
  fi
  mkdir -p "$(dirname "$repo")"
  cp -p "$live" "$repo"
  # Redact secrets from config JSON copies so they never reach the public repo.
  case "$repo" in
    config/*.json) python3 _secrets.py redact "$repo";;
  esac
  n=$((n+1))
done < files.txt
echo "copied $n file(s); $missing missing"

git add -A
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi
git commit -m "$MSG"
git push -u origin customizations
echo "Pushed to riverart2000/omlx (branch: customizations)."
