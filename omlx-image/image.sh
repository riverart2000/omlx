#!/usr/bin/env bash
# omlx-image service control: start | stop | status | restart
set -euo pipefail

DIR="/Users/joebains/omlx-image"
PY="$DIR/.venv/bin/python"
SERVER="$DIR/image_server.py"
PORT="${IMAGE_PORT:-8400}"
PID_FILE="$DIR/.image.pid"
LOG="$DIR/image_server.log"

is_up() { curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

start() {
  if is_up; then echo "omlx-image already running on $PORT"; exit 0; fi
  echo "Starting omlx-image…"
  nohup "$PY" "$SERVER" >"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
  for i in $(seq 1 30); do
    if is_up; then echo "omlx-image ready at http://127.0.0.1:$PORT"; exit 0; fi
    sleep 1
  done
  echo "omlx-image did not come up; see $LOG"; tail -20 "$LOG"; exit 1
}

stop() {
  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then kill "$PID" 2>/dev/null || true; fi
    rm -f "$PID_FILE"
  fi
  # Fallback: find the listener by port.
  PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
  for p in $PIDS; do kill "$p" 2>/dev/null || true; done
  echo "omlx-image stopped."
}

status() {
  if is_up; then
    echo "omlx-image: UP"
    curl -s "http://127.0.0.1:$PORT/health"; echo
  else
    echo "omlx-image: DOWN"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *) echo "usage: $0 {start|stop|status|restart}"; exit 2 ;;
esac
