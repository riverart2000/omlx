#!/usr/bin/env bash
# omlx-orchestrator service control: start | stop | status | restart | logs
set -euo pipefail

DIR="/Users/joebains/omlx-orchestrator"
PY="${ORCH_PY:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}"
SERVER="$DIR/orchestrator_server.py"
PORT="${ORCH_PORT:-8700}"
PID_FILE="$DIR/.orchestrator.pid"
LOG="$DIR/orchestrator_server.log"

is_up() { curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

start() {
  if is_up; then
    echo "omlx-orchestrator already running on $PORT"
    exit 0
  fi
  echo "Starting omlx-orchestrator…"
  nohup "$PY" "$SERVER" >"$LOG" 2>&1 &
  echo "$!" >"$PID_FILE"
  for _ in $(seq 1 45); do
    if is_up; then
      echo "omlx-orchestrator ready at http://127.0.0.1:$PORT"
      exit 0
    fi
    sleep 1
  done
  echo "omlx-orchestrator did not come up; see $LOG"
  tail -40 "$LOG" || true
  exit 1
}

stop() {
  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
  PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
  if [ -n "$PIDS" ]; then
    for p in $PIDS; do
      kill "$p" 2>/dev/null || true
    done
  fi
  echo "omlx-orchestrator stopped."
}

status() {
  if is_up; then
    echo "omlx-orchestrator: UP"
    curl -s "http://127.0.0.1:$PORT/health"; echo
  else
    echo "omlx-orchestrator: DOWN"
  fi
}

logs() {
  if [ -f "$LOG" ]; then
    tail -n 120 "$LOG"
  else
    echo "no log at $LOG"
  fi
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    logs ;;
  *) echo "usage: $0 {start|stop|status|restart|logs}"; exit 2 ;;
esac
