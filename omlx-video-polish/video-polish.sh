#!/usr/bin/env bash
# oMLX Video Polish service control: start | stop | restart | status | logs
set -euo pipefail

DIR="/Users/joebains/omlx-video-polish"
PORT="${VIDEO_POLISH_PORT:-8950}"
LOG="$DIR/service.log"
LABEL="com.joebains.omlx-video-polish"
PLIST="/Users/joebains/Library/LaunchAgents/com.joebains.omlx-video-polish.plist"
DOMAIN="gui/$(id -u)"

is_up() { curl -s -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; }

start() {
  if is_up; then echo "oMLX Video Polish already running on $PORT"; exit 0; fi
  echo "Starting oMLX Video Polish…"
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    launchctl kickstart "$DOMAIN/$LABEL" >/dev/null
  else
    launchctl bootstrap "$DOMAIN" "$PLIST"
  fi
  for _ in $(seq 1 30); do
    if is_up; then echo "Video Polish ready at http://127.0.0.1:$PORT"; exit 0; fi
    sleep 1
  done
  echo "Video Polish did not start; see $LOG"; tail -30 "$LOG"; exit 1
}

stop() {
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    launchctl bootout "$DOMAIN/$LABEL"
  else
    PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
    for process_id in $PIDS; do kill "$process_id" 2>/dev/null || true; done
  fi
  echo "oMLX Video Polish stopped."
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 1; start ;;
  status) if is_up; then curl -s "http://127.0.0.1:$PORT/api/health"; echo; else echo "oMLX Video Polish: DOWN"; exit 1; fi ;;
  logs) tail -f "$LOG" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}"; exit 2 ;;
esac
