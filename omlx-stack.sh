#!/usr/bin/env bash
# ============================================================================
# omlx-stack — one control script for the whole local AI media stack.
#
#   ./omlx-stack.sh {start|stop|restart|status}
#
# Controls, in order:
#   * oMLX desktop app  — serves the GUI on :8000 (http://127.0.0.1:8000/admin/chat)
#   * omlx-memory       — :8300  (launchd: com.joebains.omlx-memory)
#   * tts-studio        — :8200  (launchd: com.joebains.tts-studio)
#   * omlx-image        — :8400  (launchd: com.joebains.omlx-image)
#   * omlx-video        — :8500  (launchd: com.joebains.omlx-video)
#   * omlx-orchestrator — :8700  (launchd: com.joebains.omlx-orchestrator)
#
# The sidecars are launchd user-agents with KeepAlive=true, so a plain
# `kill` just gets them respawned — this script drives them via `launchctl`
# (bootstrap / bootout / kickstart) so stop actually sticks. The GUI is a
# normal .app, launched with `open -a` and quit via AppleScript.
# ============================================================================
set -uo pipefail

APP_NAME="oMLX"
APP_PATH="/Applications/oMLX.app"
GUI_PORT=8000
GUI_HEALTH="http://127.0.0.1:${GUI_PORT}/admin/chat"

UID_NUM="$(id -u)"
DOMAIN="gui/${UID_NUM}"
AGENT_DIR="${HOME}/Library/LaunchAgents"

# label|port|friendly-name  (start order: memory first, then the rest)
SERVICES=(
  "com.joebains.omlx-memory|8300|memory (RAG/embeddings)"
  "com.joebains.tts-studio|8200|tts-studio (voice/viral)"
  "com.joebains.omlx-image|8400|image (HiDream/Kontext)"
  "com.joebains.omlx-video|8500|video (Wan2.2)"
  "com.joebains.omlx-orchestrator|8700|orchestrator (workflow coordinator)"
)

# --- colours (fall back to plain if not a tty) ------------------------------
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_BAD=$'\033[31m'; C_WARN=$'\033[33m'; C_DIM=$'\033[2m'; C_R=$'\033[0m'
else
  C_OK=""; C_BAD=""; C_WARN=""; C_DIM=""; C_R=""
fi

port_up() { curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
gui_up()  { local c; c="$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$GUI_HEALTH" 2>/dev/null)"; [ "$c" = "200" ] || [ "$c" = "302" ] || [ "$c" = "301" ]; }

# ---------------------------------------------------------------------------
# GUI (desktop app)
# ---------------------------------------------------------------------------
gui_start() {
  if gui_up; then echo "  ${C_OK}✓${C_R} GUI already up on :${GUI_PORT}"; return 0; fi
  echo "  → launching ${APP_NAME}.app…"
  open -a "$APP_PATH" >/dev/null 2>&1 || { echo "  ${C_BAD}✗${C_R} could not open ${APP_PATH}"; return 1; }
  for _ in $(seq 1 30); do
    if gui_up; then echo "  ${C_OK}✓${C_R} GUI ready — ${GUI_HEALTH}"; return 0; fi
    sleep 1
  done
  echo "  ${C_BAD}✗${C_R} GUI did not answer on :${GUI_PORT} within 30s"; return 1
}

gui_stop() {
  if ! pgrep -f "${APP_PATH}/Contents/MacOS/${APP_NAME}" >/dev/null 2>&1 && ! gui_up; then
    echo "  ${C_DIM}·${C_R} GUI already stopped"; return 0
  fi
  echo "  → quitting ${APP_NAME}.app…"
  osascript -e "quit app \"${APP_NAME}\"" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do gui_up || break; sleep 1; done
  # Fallback: kill the specific app process by its exact executable path.
  if gui_up || pgrep -f "${APP_PATH}/Contents/MacOS/${APP_NAME}" >/dev/null 2>&1; then
    for p in $(pgrep -f "${APP_PATH}/Contents/MacOS/${APP_NAME}" 2>/dev/null); do
      kill "$p" 2>/dev/null || true
    done
  fi
  echo "  ${C_OK}✓${C_R} GUI stopped"
}

# ---------------------------------------------------------------------------
# launchd sidecars
# ---------------------------------------------------------------------------
svc_start() {  # label
  local label="$1" plist="${AGENT_DIR}/$1.plist"
  if launchctl print "${DOMAIN}/${label}" >/dev/null 2>&1; then
    launchctl kickstart "${DOMAIN}/${label}" >/dev/null 2>&1 || true
  elif [ -f "$plist" ]; then
    launchctl bootstrap "$DOMAIN" "$plist" >/dev/null 2>&1 || true
  else
    echo "  ${C_BAD}✗${C_R} $label — no plist at $plist"; return 1
  fi
}

svc_stop() {  # label
  local label="$1"
  if launchctl print "${DOMAIN}/${label}" >/dev/null 2>&1; then
    launchctl bootout "${DOMAIN}/${label}" >/dev/null 2>&1 || true
  fi
}

# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
cmd_start() {
  echo "Starting omlx stack…"
  gui_start
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r label port name <<<"$entry"
    if port_up "$port"; then echo "  ${C_OK}✓${C_R} ${name} already up on :${port}"; continue; fi
    echo "  → starting ${name}…"
    svc_start "$label"
    ok=""
    for _ in $(seq 1 30); do if port_up "$port"; then ok=1; break; fi; sleep 1; done
    if [ -n "$ok" ]; then echo "  ${C_OK}✓${C_R} ${name} ready on :${port}";
    else echo "  ${C_WARN}⚠${C_R} ${name} not answering on :${port} yet (may still be loading)"; fi
  done
  echo
  cmd_status
}

cmd_stop() {
  echo "Stopping omlx stack…"
  # Stop sidecars in reverse so dependents go first.
  for (( i=${#SERVICES[@]}-1 ; i>=0 ; i-- )); do
    IFS='|' read -r label port name <<<"${SERVICES[$i]}"
    echo "  → stopping ${name}…"
    svc_stop "$label"
  done
  gui_stop
  echo "  ${C_OK}✓${C_R} all stop requests sent"
}

cmd_restart() {
  cmd_stop
  echo
  # Give launchd a moment to release the ports before re-bootstrapping.
  sleep 2
  cmd_start
}

cmd_status() {
  echo "omlx stack status:"
  local line state
  if gui_up; then state="${C_OK}UP  ${C_R}"; else state="${C_BAD}DOWN${C_R}"; fi
  printf "  %s  %-28s :%s\n" "$state" "GUI (oMLX.app)" "$GUI_PORT"
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r label port name <<<"$entry"
    if port_up "$port"; then state="${C_OK}UP  ${C_R}"; else state="${C_BAD}DOWN${C_R}"; fi
    printf "  %s  %-28s :%s\n" "$state" "$name" "$port"
  done
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 2 ;;
esac
