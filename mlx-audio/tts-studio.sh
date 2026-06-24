#!/bin/bash
# Control script for the TTS Studio background service (Voxtral HQ TTS).
#
#   ./tts-studio.sh start | stop | restart | status | logs
#
# The service itself is managed by launchd via:
#   ~/Library/LaunchAgents/com.joebains.tts-studio.plist
# so it auto-starts at login and auto-restarts on crash.

set -euo pipefail

LABEL="com.joebains.tts-studio"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PORT="${TTS_PORT:-8200}"
URL="http://127.0.0.1:${PORT}"
LOG="$HOME/mlx-audio/logs/tts_server.log"
DOMAIN="gui/$(id -u)"

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }
blue()  { printf '\033[34m%s\033[0m\n' "$1"; }

is_loaded() { launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; }

health() { curl -s --max-time 3 "${URL}/health" 2>/dev/null || true; }

wait_ready() {
  local tries="${1:-20}"
  for ((i=1; i<=tries; i++)); do
    local h; h="$(health)"
    if echo "$h" | grep -q '"ready": true'; then return 0; fi
    if echo "$h" | grep -q '"error": "[^n]'; then return 2; fi
    sleep 2
  done
  return 1
}

cmd_start() {
  if ! [ -f "$PLIST" ]; then red "Missing plist: $PLIST"; exit 1; fi
  if is_loaded; then
    blue "Service already loaded — ensuring it's started…"
    launchctl kickstart "${DOMAIN}/${LABEL}" 2>/dev/null || true
  else
    launchctl bootstrap "$DOMAIN" "$PLIST"
    launchctl enable "${DOMAIN}/${LABEL}" 2>/dev/null || true
  fi
  blue "Waiting for model to load…"
  if wait_ready 30; then green "TTS Studio ready at ${URL}"; else
    red "Service started but model not ready yet. Check: ./tts-studio.sh logs"; exit 1
  fi
}

cmd_stop() {
  if is_loaded; then
    launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
    sleep 1
    green "TTS Studio stopped."
  else
    blue "Service not loaded."
  fi
}

cmd_status() {
  echo "Service : ${LABEL}"
  if is_loaded; then
    local pid; pid="$(launchctl print "${DOMAIN}/${LABEL}" 2>/dev/null | awk -F'=' '/[[:space:]]pid[[:space:]]*=/{gsub(/[^0-9]/,"",$2); print $2; exit}')"
    green "launchd  : loaded (pid ${pid:-?})"
  else
    red   "launchd  : not loaded"
  fi
  local h; h="$(health)"
  if [ -z "$h" ]; then
    red   "HTTP     : not responding on ${URL}"
  elif echo "$h" | grep -q '"ready": true'; then
    green "HTTP     : ready  ${URL}"
  elif echo "$h" | grep -q '"loading": true'; then
    blue  "HTTP     : loading model…  ${URL}"
  else
    red   "HTTP     : $h"
  fi
}

cmd_restart() { cmd_stop; sleep 2; cmd_start; }

cmd_logs() { tail -n "${1:-40}" "$LOG" 2>/dev/null || red "No log at $LOG"; }

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    cmd_logs "${2:-40}" ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}"; exit 1 ;;
esac
