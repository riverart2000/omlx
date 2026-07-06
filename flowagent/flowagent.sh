#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

_load_env_file() {
  local env_file="$SCRIPT_DIR/.env"
  [[ -f "$env_file" ]] || return 0

  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$ ]] || continue

    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$env_file"
}

_load_env_file

PID_FILE="$SCRIPT_DIR/.flowagent.pid"
SUPERVISOR_PID_FILE="$SCRIPT_DIR/.flowagent-supervisor.pid"
UI_PID_FILE="$SCRIPT_DIR/.flowagent-ui.pid"
LLM_PID_FILE="$SCRIPT_DIR/.flowagent-llm.pid"
LOG_FILE="$SCRIPT_DIR/logs/flowagent.log"
UI_LOG_FILE="$SCRIPT_DIR/logs/flowagent-ui.log"
LLM_LOG_FILE="$SCRIPT_DIR/logs/flowagent-llm.log"
NODE_ENTRY="$SCRIPT_DIR/agent/index.js"
UI_ENTRY="$SCRIPT_DIR/ui/server.js"

# LLM server config (override via env)
DEFAULT_LLM_MODEL="mlx-community/Qwen3-14B-6bit"
LLM_MODEL="${QWEN_MODEL:-$DEFAULT_LLM_MODEL}"
LLM_PORT="${LLM_PORT:-8080}"
UI_PORT="${UI_PORT:-3000}"
QWEN_URL="${QWEN_URL:-http://localhost:${LLM_PORT}/v1/chat/completions}"
CHROME_REMOTE_DEBUGGING_PORT="${CHROME_REMOTE_DEBUGGING_PORT:-9222}"
FA_HEADLESS="${FA_HEADLESS:-1}"

# When QWEN_URL points somewhere other than our own local LLM_PORT (e.g. the
# always-on oMLX model server on :8000), treat the LLM as external: do not
# launch, stop, or manage a local mlx_lm.server here.
LLM_EXTERNAL="${LLM_EXTERNAL:-0}"
if [[ "$QWEN_URL" != *"localhost:${LLM_PORT}/"* && "$QWEN_URL" != *"127.0.0.1:${LLM_PORT}/"* ]]; then
  LLM_EXTERNAL=1
fi

export LLM_PORT UI_PORT QWEN_MODEL="$LLM_MODEL" QWEN_URL CHROME_REMOTE_DEBUGGING_PORT FA_HEADLESS LLM_EXTERNAL

mkdir -p "$SCRIPT_DIR/logs"

_pid() {
  local pid live
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  live=$(_agent_pids_live | awk 'NF { print; exit }')
  echo "$live"
}

_supervisor_pid() {
  [[ -f "$SUPERVISOR_PID_FILE" ]] && cat "$SUPERVISOR_PID_FILE" || echo ""
}

_supervisor_is_running() {
  local pid
  pid=$(_supervisor_pid)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_agent_autorestart_enabled() {
  [[ "${FA_AGENT_AUTORESTART:-1}" == "1" || "${FA_AGENT_AUTORESTART:-}" == "true" ]]
}

_ui_pid() {
  [[ -f "$UI_PID_FILE" ]] && cat "$UI_PID_FILE" || echo ""
}

# Find every PID running a node process whose argv contains the given
# absolute script path. Resilient to stale / missing PID files — a node
# process started outside this script (manual launch, leftover from a
# crashed shell, etc.) will still be found and stopped.
_find_pids_by_entry() {
  local entry="$1"
  [[ -n "$entry" ]] || { echo ""; return 0; }
  # pgrep -f matches the full argv. Filter strictly to node + exact path.
  # `|| true` so a no-match (rc=1) never aborts the caller under set -e.
  pgrep -f "node .*${entry}" 2>/dev/null || true
}

_agent_pids_live() { _find_pids_by_entry "$NODE_ENTRY"; }
_ui_pids_live()    { _find_pids_by_entry "$UI_ENTRY"; }

_flowagent_chrome_pids() {
  local profile_dir
  profile_dir="$(_chrome_profile_dir 2>/dev/null || true)"
  [[ -n "$profile_dir" ]] || { echo ""; return 0; }

  ps -axo pid=,command= 2>/dev/null | awk -v profile_dir="$profile_dir" '
    /[G]oogle Chrome/ && index($0, "--user-data-dir=" profile_dir) { print $1 }
  ' || true
}

_stop_flowagent_chrome_profile() {
  local label="${1:-FlowAgent Chrome profile}"
  local chrome_pids
  chrome_pids="$(_flowagent_chrome_pids || true)"
  chrome_pids=$(echo "$chrome_pids" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' ')
  if [[ -z "${chrome_pids// }" ]]; then
    return 0
  fi

  echo "Stopping $label (PIDs: $chrome_pids)..."
  for pid in $chrome_pids; do kill -TERM "$pid" 2>/dev/null || true; done
  local waited=0
  while :; do
    local alive=""
    for pid in $chrome_pids; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
    [[ -z "${alive// }" ]] && break
    sleep 1; (( waited += 1 ))
    if (( waited >= 5 )); then
      for pid in $alive; do kill -KILL "$pid" 2>/dev/null || true; done
      break
    fi
  done
  echo "$label stopped."
}

# Remove stale Chrome singleton lock symlinks for a profile. When a prior
# (often headless) Chrome held the profile, these dangling links make a fresh
# `open` silently attach to nothing instead of showing a window.
_clear_profile_singleton_locks() {
  local profile_dir="$1"
  [[ -n "$profile_dir" ]] || return 0
  rm -f "$profile_dir/SingletonLock" "$profile_dir/SingletonSocket" "$profile_dir/SingletonCookie" 2>/dev/null || true
}

# Fully stop the agent + supervisor so Playwright releases the Chrome profile
# and nothing relaunches it headless. Required before a manual login: the
# running agent owns the profile in a headless Chrome, which prevents the
# visible login window from ever appearing.
_stop_agent_and_supervisor() {
  local sup_pid
  sup_pid="$(_supervisor_pid)"
  if [[ -n "$sup_pid" ]] && kill -0 "$sup_pid" 2>/dev/null; then
    echo "Stopping FlowAgent supervisor (PID $sup_pid)..."
    kill -TERM "$sup_pid" 2>/dev/null || true
    local w=0
    while kill -0 "$sup_pid" 2>/dev/null; do
      sleep 1; (( w += 1 ))
      (( w >= 5 )) && { kill -KILL "$sup_pid" 2>/dev/null || true; break; }
    done
  fi
  rm -f "$SUPERVISOR_PID_FILE"

  local agent_pids
  agent_pids="$(_pid) $(_agent_pids_live || true)"
  agent_pids=$(echo "$agent_pids" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' ')
  if [[ -n "${agent_pids// }" ]]; then
    echo "Stopping FlowAgent agent (PIDs: $agent_pids)..."
    for pid in $agent_pids; do kill -TERM "$pid" 2>/dev/null || true; done
    local waited=0
    while :; do
      local alive=""
      for pid in $agent_pids; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
      [[ -z "${alive// }" ]] && break
      sleep 1; (( waited += 1 ))
      (( waited >= 10 )) && { for pid in $alive; do kill -KILL "$pid" 2>/dev/null || true; done; break; }
    done
    local strag
    strag="$(_agent_pids_live || true)"
    [[ -n "${strag// }" ]] && for pid in $strag; do kill -KILL "$pid" 2>/dev/null || true; done
  fi
  rm -f "$PID_FILE"
}

_is_running() {
  local pid pids
  pid=$(_pid)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  pids=$(_agent_pids_live)
  [[ -n "${pids// }" ]]
}

_ui_is_running() {
  local pid pids
  pid=$(_ui_pid)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  pids=$(_ui_pids_live)
  [[ -n "${pids// }" ]]
}

_llm_pid() {
  [[ -f "$LLM_PID_FILE" ]] && cat "$LLM_PID_FILE" || echo ""
}

_llm_is_running() {
  local pid
  pid=$(_llm_pid)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_llm_cmd() {
  # Returns the command to launch mlx_lm.server, or empty string if not found
  if command -v mlx_lm.server >/dev/null 2>&1; then
    echo "mlx_lm.server"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "import mlx_lm" 2>/dev/null; then
    echo "python3 -m mlx_lm.server"
  else
    echo ""
  fi
}

# Check if a TCP port is already bound (regardless of PID file)
_port_in_use() {
  lsof -ti :"$1" >/dev/null 2>&1
}

_llm_http_ok() {
  command -v curl >/dev/null 2>&1 && \
    curl -fsS --max-time 2 "http://localhost:$LLM_PORT/v1/models" >/dev/null 2>&1
}

# Check an external LLM's /v1/models given its chat-completions URL.
_llm_http_ok_url() {
  local url="$1"
  local base="${url%%/v1/*}"
  command -v curl >/dev/null 2>&1 && \
    curl -fsS --max-time 3 "${base}/v1/models" >/dev/null 2>&1
}

_browser_mode() {
  node -e "const db=require('./db/sqlite'); process.stdout.write(db.getState('browser_mode') || process.env.FLOWAGENT_BROWSER_MODE || 'browsermcp')"
}

_chrome_profile_dir() {
  node <<'NODE'
const db = require('./db/sqlite');
const { getRuntimeConfig } = require('./config');

process.stdout.write(getRuntimeConfig(db.getSettings()).chromeProfileDir);
NODE
}

_browser_startup_context() {
  node <<'NODE'
const db = require('./db/sqlite');
const { ALL_PLATFORMS, getRuntimeConfig, getRetiredPlatforms, getSocialTargets } = require('./config');

const settings = db.getSettings();
const runtime = getRuntimeConfig(settings);
const targets = getSocialTargets(settings);
const retired = new Set(getRetiredPlatforms());

function normalizeUrl(url) {
  const clean = String(url || '').trim();
  if (!clean) return '';
  if (/^https?:\/\//i.test(clean)) return clean;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(clean)) return clean;
  return 'https://' + clean.replace(/^\/+/, '');
}

const urls = [];
const seen = new Set();
for (const platform of ALL_PLATFORMS) {
  if (retired.has(platform.name)) continue;
  const url = normalizeUrl(targets[platform.name]);
  if (!url || seen.has(url)) continue;
  seen.add(url);
  urls.push(url);
}

process.stdout.write([
  runtime.chromeProfileDir,
  ...urls,
].join('\t'));
NODE
}

_chrome_debug_ok() {
  command -v curl >/dev/null 2>&1 && \
    curl -fsS --max-time 2 "http://127.0.0.1:$CHROME_REMOTE_DEBUGGING_PORT/json/version" >/dev/null 2>&1
}

_foreground_mode() {
  [[ "${FA_FOREGROUND:-0}" == "1" || "${FA_FOREGROUND:-}" == "true" ]]
}

_ensure_flowagent_tabs_cdp() {
  local urls=("$@")
  [[ ${#urls[@]} -gt 0 ]] || return 0

  node - "$CHROME_REMOTE_DEBUGGING_PORT" "${urls[@]}" <<'NODE'
const http = require('http');

const [, , rawPort, ...urls] = process.argv;
const port = Number(rawPort || 9222);

function request(method, path) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, path, method }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', reject);
    req.setTimeout(2500, () => req.destroy(new Error('Chrome remote debugging request timed out')));
    req.end();
  });
}

(async () => {
  const list = await request('GET', '/json/list');
  const tabs = JSON.parse(list.body || '[]');
  const existing = new Set(tabs.map(tab => String(tab.url || '')));

  for (const url of urls.filter(Boolean)) {
    if (existing.has(url)) continue;
    const opened = await request('PUT', `/json/new?${encodeURIComponent(url)}`);
    if (opened.status < 200 || opened.status >= 300) {
      throw new Error(`Chrome remote debugging could not open ${url}: HTTP ${opened.status}`);
    }
  }
})().catch(err => {
  console.error(err.message || err);
  process.exit(1);
});
NODE
}

_ensure_flowagent_tabs() {
  local urls=("$@")
  [[ ${#urls[@]} -gt 0 ]] || return 0

  if ! _foreground_mode; then
    if ! _ensure_flowagent_tabs_cdp "${urls[@]}"; then
      echo "WARNING: Failed to open one or more startup tabs via Chrome remote debugging."
    fi
    return 0
  fi

  command -v osascript >/dev/null 2>&1 || return 0

  local should_activate="false"
  if _foreground_mode; then
    should_activate="true"
  fi

  if ! osascript - "$should_activate" "${urls[@]}" <<'APPLESCRIPT'
on run argv
  set shouldActivate to item 1 of argv
  set targetUrls to items 2 thru -1 of argv
  tell application "Google Chrome"
    if shouldActivate is "true" then activate
    if (count of windows) = 0 then
      make new window
    end if

    set existingUrls to {}
    repeat with w in windows
      repeat with t in tabs of w
        try
          set end of existingUrls to (URL of t as text)
        end try
      end repeat
    end repeat

    set targetWindow to front window
    repeat with targetUrl in targetUrls
      set targetText to targetUrl as text
      if existingUrls does not contain targetText then
        tell targetWindow to make new tab with properties {URL:targetText}
      end if
    end repeat
  end tell
end run
APPLESCRIPT
  then
    echo "WARNING: Failed to open one or more startup tabs in Google Chrome."
  fi
}

_ensure_login_tabs_osascript() {
  local urls=("$@")
  [[ ${#urls[@]} -gt 0 ]] || return 0
  command -v osascript >/dev/null 2>&1 || return 0

  if ! osascript - "${urls[@]}" <<'APPLESCRIPT'
on run argv
  set targetUrls to argv
  tell application "Google Chrome"
    activate
    if (count of windows) = 0 then
      make new window
    end if

    set existingUrls to {}
    repeat with w in windows
      repeat with t in tabs of w
        try
          set end of existingUrls to (URL of t as text)
        end try
      end repeat
    end repeat

    set targetWindow to front window
    repeat with targetUrl in targetUrls
      set targetText to targetUrl as text
      if existingUrls does not contain targetText then
        tell targetWindow to make new tab with properties {URL:targetText}
      end if
    end repeat
  end tell
end run
APPLESCRIPT
  then
    echo "WARNING: Failed to open one or more login tabs in Google Chrome."
  fi
}

_launch_login_chrome() {
  local profile_dir="${1:-}"
  shift || true
  local urls=("$@")

  [[ -n "$profile_dir" ]] || profile_dir="$(_chrome_profile_dir)"
  mkdir -p "$profile_dir"

  # Manual login must use normal Chrome. Playwright/remote-debugging-pipe and
  # mock-keychain launches trigger Google "browser or app may not be secure".
  _stop_flowagent_chrome_profile "FlowAgent Chrome profile before manual login"
  # Clear stale singleton locks left by the (headless) agent Chrome, otherwise
  # the visible login window silently attaches instead of opening.
  _clear_profile_singleton_locks "$profile_dir"

  echo "Starting login-friendly FlowAgent Chrome profile: $profile_dir"
  echo "  Automation flags: disabled for manual login"

  if ! command -v open >/dev/null 2>&1; then
    echo "open is not available on this system. Launch Chrome manually with:"
    echo "  Google Chrome --user-data-dir='$profile_dir' --new-window ${urls[*]:-about:blank}"
    return 1
  fi

  # Pass every platform URL to the launch command so Chrome opens them as tabs
  # in one visible window. This avoids AppleScript (which needs macOS Automation
  # permission and blocks on its consent prompt).
  local -a open_urls=("${urls[@]}")
  [[ ${#open_urls[@]} -gt 0 ]] || open_urls=("about:blank")
  open -na "Google Chrome" --args \
    --user-data-dir="$profile_dir" \
    --no-first-run --no-default-browser-check \
    --new-window "${open_urls[@]}"

  local waited=0
  until [[ -n "$(_flowagent_chrome_pids || true)" ]] || (( waited >= 15 )); do
    sleep 1
    (( waited += 1 ))
  done

  if [[ -z "$(_flowagent_chrome_pids || true)" ]]; then
    echo "WARNING: FlowAgent Chrome did not appear after ${waited}s."
    return 1
  fi
}

_launch_flowagent_chrome() {
  local profile_dir="${1:-}"
  shift || true
  local urls=("$@")

  [[ -n "$profile_dir" ]] || profile_dir="$(_chrome_profile_dir)"

  mkdir -p "$profile_dir"

  if _chrome_debug_ok; then
    echo "Chrome remote debugging already available on port $CHROME_REMOTE_DEBUGGING_PORT."
    _ensure_flowagent_tabs "${urls[@]}"
    return 0
  fi

  if _port_in_use "$CHROME_REMOTE_DEBUGGING_PORT"; then
    echo "WARNING: Port $CHROME_REMOTE_DEBUGGING_PORT is already in use, but Chrome remote debugging is not responding yet."
  fi

  echo "Starting FlowAgent Chrome profile: $profile_dir"
  echo "  Remote debugging: http://127.0.0.1:$CHROME_REMOTE_DEBUGGING_PORT"
  # Background-safe flags: keep tabs/renderers fully alive when the window is
  # hidden, minimized, or occluded so Playwright can drive them silently
  # without Chrome throttling timers, freezing renderers, or backgrounding
  # the page. Required for FA_FOREGROUND=0 (default) operation.
  local bg_flags=(
    --disable-background-timer-throttling
    --disable-renderer-backgrounding
    --disable-backgrounding-occluded-windows
    --disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling
  )
  if command -v open >/dev/null 2>&1; then
    if _foreground_mode; then
      open -na "Google Chrome" --args --user-data-dir="$profile_dir" "--remote-debugging-port=$CHROME_REMOTE_DEBUGGING_PORT" "${bg_flags[@]}" --new-window about:blank
    else
      open -g -na "Google Chrome" --args --user-data-dir="$profile_dir" "--remote-debugging-port=$CHROME_REMOTE_DEBUGGING_PORT" "${bg_flags[@]}" --new-window about:blank
    fi
  else
    echo "open is not available on this system. Launch Chrome manually with:"
    echo "  Google Chrome --user-data-dir='$profile_dir' --remote-debugging-port=$CHROME_REMOTE_DEBUGGING_PORT ${bg_flags[*]} --new-window about:blank"
    return 1
  fi

  local waited=0
  until _chrome_debug_ok || (( waited >= 15 )); do
    sleep 1
    (( waited += 1 ))
  done

  if _chrome_debug_ok; then
    echo "Chrome remote debugging ready → http://127.0.0.1:$CHROME_REMOTE_DEBUGGING_PORT"
    _ensure_flowagent_tabs "${urls[@]}"
  else
    echo "WARNING: Chrome remote debugging did not respond on port $CHROME_REMOTE_DEBUGGING_PORT after ${waited}s."
    echo "         If FlowAgent Chrome was already open without remote debugging, close that window and retry ./flowagent.sh browser-login"
  fi
}

_start_agent_supervisor() {
  nohup env NODE_ENTRY="$NODE_ENTRY" LOG_FILE="$LOG_FILE" bash -c '
    set +e
    while true; do
      node "$NODE_ENTRY" >> "$LOG_FILE" 2>&1
      exit_code=$?
      printf "%s [WARN] [supervisor] Agent exited with code %s; restarting in 2s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >> "$LOG_FILE"
      sleep 2
    done
  ' >/dev/null 2>&1 &
  echo $! > "$SUPERVISOR_PID_FILE"
}

cmd_start() {
  # ── LLM server ──────────────────────────────────────────────────────────────
  if [[ "$LLM_EXTERNAL" == "1" ]]; then
    echo "Using external LLM at $QWEN_URL"
    echo "  Model: $LLM_MODEL (not launching a local LLM server)."
    if _llm_http_ok_url "$QWEN_URL"; then
      echo "  External LLM is reachable."
    else
      echo "  WARNING: external LLM at $QWEN_URL is not responding — start it before running tasks."
    fi
  elif _llm_is_running; then
    echo "LLM server already running (PID $(_llm_pid)) on port $LLM_PORT."
  elif _port_in_use "$LLM_PORT"; then
    echo "LLM port $LLM_PORT already in use — assuming LLM server is running externally."
  else
    local llm_bin
    llm_bin=$(_llm_cmd)
    if [[ -z "$llm_bin" ]]; then
      echo "WARNING: mlx_lm not found — LLM server will not be started."
      echo "         Install with: pip install mlx-lm"
      echo "         Or start manually: mlx_lm.server --model $LLM_MODEL --port $LLM_PORT"
    else
      echo "Starting LLM server ($LLM_MODEL on port $LLM_PORT)..."
      # shellcheck disable=SC2086
      nohup $llm_bin --model "$LLM_MODEL" --port "$LLM_PORT" >> "$LLM_LOG_FILE" 2>&1 &
      echo $! > "$LLM_PID_FILE"
      # Give the model server time to load weights before the agent needs it
      echo "  Waiting for LLM server to become ready (this may take ~30s for large models)..."
      local llm_waited=0
      until _port_in_use "$LLM_PORT" || (( llm_waited >= 60 )); do
        sleep 2; (( llm_waited += 2 ))
        printf "."
      done
      echo ""
      if _port_in_use "$LLM_PORT"; then
        echo "LLM server ready (PID $(_llm_pid)) → http://localhost:$LLM_PORT"
      else
        echo "WARNING: LLM server did not become ready after ${llm_waited}s."
        echo "         Check $LLM_LOG_FILE for errors."
      fi
    fi
  fi

  local browser_mode
    browser_mode="$(_browser_mode)"
    if [[ "$browser_mode" == "playwright" || "$browser_mode" == "both" ]]; then
      echo "Playwright browser mode: agent will launch and own the configured Chrome profile."
    fi

  # ── Manual-login auto-recovery ──────────────────────────────────────────────
  # If manual-login mode was left on but no login Chrome window is open and the
  # profile already holds session cookies, the login is evidently done (or was
  # never needed). Clear the pause automatically instead of sitting forever
  # waiting for a human that already logged in.
  if [[ "$(node -e "process.stdout.write(String(require('./agent/control').isManualLoginMode()))" 2>/dev/null)" == "true" ]]; then
    if [[ -n "$(_flowagent_chrome_pids || true)" ]]; then
      echo "NOTE: manual-login Chrome is still open. Finish signing in, then run: $0 finish-login"
    elif node -e "process.exit(require('./playwright/profileHealth').getProfileCookieHealth().totalCookies > 0 ? 0 : 1)" 2>/dev/null; then
      echo "Manual-login mode was left on, but the profile already has saved sessions — auto-resuming queue."
      node -e "require('./agent/control').resumeQueue({ requireBrowserProfileReady: true }); console.log('Queue resumed (manual-login mode cleared).')" || \
        echo "WARNING: could not auto-resume; run $0 resume manually."
    else
      echo "WARNING: manual-login mode is on and the profile has no saved cookies."
      echo "         Run: $0 browser-login   then sign into the platform tabs."
    fi
  fi

  # ── Agent ───────────────────────────────────────────────────────────────────
  if _is_running; then
    echo "FlowAgent agent is already running (PID $(_pid))."
    if _supervisor_is_running; then
      echo "Agent supervisor: running (PID $(_supervisor_pid))."
    fi
  else
    if _agent_autorestart_enabled; then
      if _supervisor_is_running; then
        echo "FlowAgent supervisor is already running (PID $(_supervisor_pid))."
      else
        echo "Starting FlowAgent supervisor (auto-restart enabled)..."
        _start_agent_supervisor
      fi
    else
      echo "Starting FlowAgent (single-run mode; auto-restart disabled)..."
      nohup node "$NODE_ENTRY" >> "$LOG_FILE" 2>&1 &
      echo $! > "$PID_FILE"
    fi

    local waited=0
    until _is_running || (( waited >= 12 )); do
      sleep 1
      (( waited += 1 ))
    done

    if _is_running; then
      local live_pid
      live_pid=$(_agent_pids_live | awk 'NF { print; exit }')
      if [[ -n "$live_pid" ]]; then
        echo "$live_pid" > "$PID_FILE"
      fi
      echo "FlowAgent started (PID $(_pid)). Logs: $LOG_FILE"
      if _supervisor_is_running; then
        echo "FlowAgent supervisor running (PID $(_supervisor_pid))."
      fi
    else
      echo "FlowAgent failed to start. Check $LOG_FILE for details." >&2
      rm -f "$PID_FILE" "$SUPERVISOR_PID_FILE"
      exit 1
    fi
  fi

  # Start UI server
  local ui_port="$UI_PORT"

  if _ui_is_running; then
    echo "FlowAgent UI is already running (PID $(_ui_pid)) → http://localhost:$ui_port"
  elif _port_in_use "$ui_port"; then
    echo "WARNING: Port $ui_port is already in use by another process — skipping UI start."
    echo "         Stop that process first, or set UI_PORT=<other> and restart."
  else
    echo "Starting FlowAgent UI on port $ui_port..."
    UI_PORT="$ui_port" nohup node "$UI_ENTRY" >> "$UI_LOG_FILE" 2>&1 &
    echo $! > "$UI_PID_FILE"
    sleep 1

    if _ui_is_running; then
      echo "FlowAgent UI started (PID $(_ui_pid)) → http://localhost:$ui_port"
    else
      echo "UI failed to start. Check $UI_LOG_FILE for details."
      rm -f "$UI_PID_FILE"
    fi
  fi
}

cmd_stop() {
  # Stop supervisor first so it does not immediately respawn the agent while
  # we're trying to shut down.
  local supervisor_pid
  supervisor_pid=$(_supervisor_pid)
  if [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    echo "Stopping FlowAgent supervisor (PID $supervisor_pid)..."
    kill -TERM "$supervisor_pid" 2>/dev/null || true
    local s_waited=0
    while kill -0 "$supervisor_pid" 2>/dev/null; do
      sleep 1; (( s_waited += 1 ))
      if (( s_waited >= 5 )); then
        kill -KILL "$supervisor_pid" 2>/dev/null || true
        break
      fi
    done
    echo "FlowAgent supervisor stopped."
  fi
  rm -f "$SUPERVISOR_PID_FILE"

  # Stop agent — kill EVERY matching node process, not just the PID file.
  # The PID file goes stale whenever a process was started outside the
  # script (manual launch, crashed shell, prior daemon). Pattern-match
  # by absolute entry path so we always get the right process.
  local agent_pids
  agent_pids="$(_pid) $(_agent_pids_live || true)"
  # Dedupe + drop blanks.
  agent_pids=$(echo "$agent_pids" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' ')
  if [[ -n "${agent_pids// }" ]]; then
    echo "Stopping FlowAgent agent (PIDs: $agent_pids)..."
    for pid in $agent_pids; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    local waited=0
    while :; do
      local alive=""
      for pid in $agent_pids; do
        kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
      done
      [[ -z "${alive// }" ]] && break
      sleep 1; (( waited += 1 ))
      if (( waited >= 10 )); then
        echo "Agent did not stop after 10s — sending SIGKILL to:$alive"
        for pid in $alive; do kill -KILL "$pid" 2>/dev/null || true; done
        break
      fi
    done
    # Final sweep — anything still matching the pattern must die.
    local stragglers
    stragglers="$(_agent_pids_live || true)"
    if [[ -n "${stragglers// }" ]]; then
      echo "Killing straggler agent processes: $stragglers"
      for pid in $stragglers; do kill -KILL "$pid" 2>/dev/null || true; done
    fi
    echo "FlowAgent agent stopped."
  else
    echo "FlowAgent agent is not running."
  fi
  rm -f "$PID_FILE"
  # Playwright owns the dedicated FlowAgent Chrome profile now. SIGTERM should
  # let the agent close it cleanly; this profile-scoped fallback handles hard
  # exits without touching the user's normal Chrome windows.
  _stop_flowagent_chrome_profile "FlowAgent Chrome profile"

  # Stop UI server — kill by PID file + pattern + port for full coverage.
  local ui_pids
  ui_pids="$(_ui_pid) $(_ui_pids_live || true)"
  ui_pids=$(echo "$ui_pids" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' ')
  if [[ -n "${ui_pids// }" ]]; then
    echo "Stopping FlowAgent UI (PIDs: $ui_pids)..."
    for pid in $ui_pids; do kill -TERM "$pid" 2>/dev/null || true; done
    local w=0
    while :; do
      local alive=""
      for pid in $ui_pids; do
        kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
      done
      [[ -z "${alive// }" ]] && break
      sleep 1; (( w += 1 ))
      if (( w >= 5 )); then
        for pid in $alive; do kill -KILL "$pid" 2>/dev/null || true; done
        break
      fi
    done
    echo "FlowAgent UI stopped."
  fi
  # Port-level fallback regardless of how we got here.
  local ui_port="${UI_PORT:-3000}"
  if _port_in_use "$ui_port"; then
    echo "Killing process holding UI port $ui_port..."
    lsof -ti :"$ui_port" | xargs kill -KILL 2>/dev/null || true
  fi
  rm -f "$UI_PID_FILE"

  # Stop LLM server
  if [[ "$LLM_EXTERNAL" == "1" ]]; then
    echo "External LLM ($QWEN_URL) is not managed by this script — leaving it running."
  elif _llm_is_running; then
    local llm_pid
    llm_pid=$(_llm_pid)
    echo "Stopping LLM server (PID $llm_pid)..."
    kill -TERM "$llm_pid" 2>/dev/null || true
    local w=0
    while kill -0 "$llm_pid" 2>/dev/null; do
      sleep 1; (( w += 1 ))
      if (( w >= 10 )); then kill -KILL "$llm_pid" 2>/dev/null || true; break; fi
    done
    echo "LLM server stopped."
  else
    if _port_in_use "$LLM_PORT"; then
      echo "Killing stale process on LLM port $LLM_PORT..."
      lsof -ti :"$LLM_PORT" | xargs kill -KILL 2>/dev/null || true
      echo "LLM server stopped."
    else
      echo "LLM server is not running."
    fi
  fi
  rm -f "$LLM_PID_FILE"
}

cmd_status() {
  if _is_running; then
    echo "Agent  : running (PID $(_pid))"
  else
    echo "Agent  : not running"
    rm -f "$PID_FILE"
  fi
  if _supervisor_is_running; then
    echo "Superv.: running (PID $(_supervisor_pid)) auto-restart=on"
  else
    if _agent_autorestart_enabled; then
      echo "Superv.: not running (auto-restart is enabled for next start)"
    else
      echo "Superv.: disabled (FA_AGENT_AUTORESTART=0)"
    fi
    rm -f "$SUPERVISOR_PID_FILE"
  fi
  if _ui_is_running; then
    echo "UI     : running (PID $(_ui_pid)) → http://localhost:${UI_PORT:-3000}"
  else
    echo "UI     : not running"
    rm -f "$UI_PID_FILE"
  fi
  if [[ "$LLM_EXTERNAL" == "1" ]]; then
    if _llm_http_ok_url "$QWEN_URL"; then
      echo "LLM    : external, reachable → $QWEN_URL"
      echo "         model=$LLM_MODEL"
    else
      echo "LLM    : external, NOT responding → $QWEN_URL  ← start the oMLX server"
    fi
  elif _llm_is_running; then
    if _llm_http_ok; then
      echo "LLM    : running (PID $(_llm_pid)) → http://localhost:${LLM_PORT}"
    else
      echo "LLM    : process running (PID $(_llm_pid)), but /v1/models is not responding yet"
    fi
  elif _port_in_use "$LLM_PORT"; then
    if _llm_http_ok; then
      echo "LLM    : running externally on port $LLM_PORT (not managed by this script)"
    else
      echo "LLM    : port $LLM_PORT is in use, but /v1/models is not responding"
    fi
  else
    echo "LLM    : not running  ← chat and agent calls will fail until this is started"
    echo "         Run:  $0 start"
  fi
}

cmd_restart() {
  cmd_stop || true
  # Belt-and-braces: ensure no residual agent/ui process survived.
  local strag
  strag="$(_agent_pids_live || true)"
  if [[ -n "${strag// }" ]]; then
    echo "Killing residual agent processes after stop: $strag"
    for pid in $strag; do kill -KILL "$pid" 2>/dev/null || true; done
  fi
  strag="$(_ui_pids_live || true)"
  if [[ -n "${strag// }" ]]; then
    echo "Killing residual UI processes after stop: $strag"
    for pid in $strag; do kill -KILL "$pid" 2>/dev/null || true; done
  fi
  sleep 1
  cmd_start
  # Verify the new processes are actually OURS (matching PIDs in pid files)
  # and that they were started AFTER the restart began. Refuse to declare
  # success otherwise.
  local pid running_pids
  pid="$(_pid)"
  running_pids="$(_agent_pids_live || true)"
  if [[ -z "$pid" || -z "${running_pids// }" ]]; then
    echo "ERROR: restart did not produce a running agent process." >&2
    exit 1
  fi
  if ! echo " $running_pids " | grep -q " $pid "; then
    echo "ERROR: PID file ($pid) does not match running agent PIDs ($running_pids)." >&2
    exit 1
  fi
  echo "Restart verified — agent PID $pid is the active process."
}

cmd_restart_ui() {
  local ui_port="$UI_PORT"

  if _ui_is_running; then
    local ui_pid
    ui_pid=$(_ui_pid)
    echo "Stopping FlowAgent UI (PID $ui_pid)..."
    kill -TERM "$ui_pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$ui_pid" 2>/dev/null; do
      sleep 1; (( waited += 1 ))
      if (( waited >= 5 )); then
        kill -KILL "$ui_pid" 2>/dev/null || true
        break
      fi
    done
  elif _port_in_use "$ui_port"; then
    echo "Killing stale process on UI port $ui_port..."
    lsof -ti :"$ui_port" | xargs kill -KILL 2>/dev/null || true
  fi
  rm -f "$UI_PID_FILE"

  echo "Starting FlowAgent UI on port $ui_port..."
  UI_PORT="$ui_port" nohup node "$UI_ENTRY" >> "$UI_LOG_FILE" 2>&1 &
  echo $! > "$UI_PID_FILE"
  sleep 1

  if _ui_is_running; then
    echo "FlowAgent UI started (PID $(_ui_pid)) → http://localhost:$ui_port"
  else
    echo "UI failed to start. Check $UI_LOG_FILE for details."
    rm -f "$UI_PID_FILE"
    exit 1
  fi
}

cmd_logs() {
  tail -f "$LOG_FILE"
}

cmd_ui_logs() {
  tail -f "$UI_LOG_FILE"
}

cmd_llm_logs() {
  tail -f "$LLM_LOG_FILE"
}

cmd_diagnose() {
  # Pretty-print a failure dossier so an LLM (or human) can triage fast.
  #   diagnose                 → latest failure across all platforms
  #   diagnose <platform>      → latest failure on that platform
  #   diagnose <task-id>       → that specific task
  #   diagnose <plat>:<id>     → exact dossier
  #   diagnose --list          → list recent diagnostic files
  cd "$SCRIPT_DIR" >/dev/null 2>&1
  if [ "${2:-}" = "--list" ]; then
    find logs/diagnostics -maxdepth 2 -name '*.json' -type f 2>/dev/null \
      | xargs -I{} stat -f '%m %N' {} 2>/dev/null \
      | sort -rn | head -30 | awk '{ $1=""; sub(/^ /,""); print }'
    return
  fi
  node agent/diagnoseCli.js "${2:-latest}"
}

_cmd_debug_bundle_snapshot() {
  local requested_task_id="${1:-}"

  node - "$requested_task_id" "$CHROME_REMOTE_DEBUGGING_PORT" <<'NODE'
const fs = require('fs');
const path = require('path');
const http = require('http');
const Database = require('better-sqlite3');

const requestedTaskId = String(process.argv[2] || '').trim();
const chromePort = Number(process.argv[3] || 9222);
const configuredDbPath = process.env.FLOWAGENT_DB_PATH && process.env.FLOWAGENT_DB_PATH !== ':memory:'
  ? process.env.FLOWAGENT_DB_PATH
  : path.join('data', 'agent.db');
const dbPath = path.isAbsolute(configuredDbPath)
  ? configuredDbPath
  : path.join(process.cwd(), configuredDbPath);

let sqlite;
try {
  sqlite = new Database(dbPath, { readonly: true, fileMustExist: true });
} catch (err) {
  console.error(`Could not open DB in read-only mode at ${dbPath}: ${err.message}`);
  process.exit(1);
}

function getState(key) {
  const row = sqlite.prepare('SELECT value FROM state WHERE key = ?').get(key);
  return row ? row.value : null;
}

function getAllPlatformPauses() {
  const rows = sqlite.prepare("SELECT key, value FROM state WHERE key LIKE 'platform_paused_%' AND value != ''").all();
  const now = Date.now();
  const pauses = {};

  for (const row of rows) {
    try {
      const info = JSON.parse(row.value);
      if (info && info.until && Number(info.until) <= now) {
        continue;
      }
      const platform = String(row.key || '').replace(/^platform_paused_/, '');
      if (platform) {
        pauses[platform] = info;
      }
    } catch {
      continue;
    }
  }

  return pauses;
}

function chooseTask(taskId) {
  if (taskId) {
    return sqlite.prepare(`
      SELECT
        t.id,
        t.platform,
        t.type,
        t.generated_title,
        p.title AS post_title,
        t.status,
        t.retries,
        t.last_error,
        t.next_attempt_at,
        t.updated_at,
        t.run_token
      FROM tasks t
      LEFT JOIN posts p ON p.id = t.content_id
      WHERE t.id = ?
      LIMIT 1
    `).get(taskId);
  }

  return sqlite.prepare(`
    SELECT
      t.id,
      t.platform,
      t.type,
      t.generated_title,
      p.title AS post_title,
      t.status,
      t.retries,
      t.last_error,
      t.next_attempt_at,
      t.updated_at,
      t.run_token
    FROM tasks t
    LEFT JOIN posts p ON p.id = t.content_id
    ORDER BY
      CASE t.status
        WHEN 'running' THEN 0
        WHEN 'pending' THEN 1
        WHEN 'needs_attention' THEN 2
        WHEN 'failed' THEN 3
        WHEN 'done' THEN 4
        ELSE 5
      END,
      datetime(t.updated_at) DESC,
      datetime(t.created_at) DESC
    LIMIT 1
  `).get();
}

function summarizeState() {
  const keys = [
    'kill',
    'manual_login_mode',
    'force_next_blog',
    'single_platform_debug',
    'browser_mode',
    'active_task_id',
  ];
  const out = {};
  for (const key of keys) {
    const value = getState(key);
    out[key] = value === null || value === '' ? '(empty)' : value;
  }
  if (out.browser_mode === '(empty)') {
    out.browser_mode = process.env.FLOWAGENT_BROWSER_MODE || 'browsermcp';
  }
  return out;
}

function getTaskCounts() {
  return sqlite.prepare(`
    SELECT status, COUNT(*) AS count
    FROM tasks
    GROUP BY status
    ORDER BY status ASC
  `).all();
}

function readDebugTail(taskId) {
  const debugPath = path.join('logs', 'debug', taskId, '00-events.log');
  if (!fs.existsSync(debugPath)) {
    return { path: debugPath, lines: [] };
  }

  const text = fs.readFileSync(debugPath, 'utf8');
  const allLines = text.split(/\r?\n/).filter(Boolean);
  return {
    path: debugPath,
    lines: allLines.slice(-25),
  };
}

function readDiagnosticSummary(platform, taskId) {
  const diagPath = path.join('logs', 'diagnostics', platform, `${taskId}.json`);
  if (!fs.existsSync(diagPath)) {
    return { path: diagPath, summary: null };
  }

  try {
    const doc = JSON.parse(fs.readFileSync(diagPath, 'utf8'));
    const triage = doc?.error?.context?.triage || null;
    return {
      path: diagPath,
      summary: {
        written_at: doc.written_at || null,
        outcome: doc.outcome || null,
        error_code: doc?.error?.code || null,
        error_phase: doc?.error?.phase || null,
        error_message: doc?.error?.message || null,
        triage,
      },
    };
  } catch (err) {
    return {
      path: diagPath,
      summary: {
        parse_error: err.message,
      },
    };
  }
}

function fetchChromeTabs(port) {
  return new Promise((resolve) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: '/json/list',
        method: 'GET',
        timeout: 2500,
      },
      (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          body += chunk;
        });
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            resolve({ ok: false, reason: `HTTP ${res.statusCode}` });
            return;
          }
          try {
            const rows = JSON.parse(body);
            const pages = Array.isArray(rows)
              ? rows
                  .filter((row) => row && (row.type === 'page' || row.type === ''))
                  .map((row) => ({
                    title: String(row.title || '').trim(),
                    url: String(row.url || '').trim(),
                  }))
                  .filter((row) => row.url)
              : [];
            resolve({ ok: true, pages });
          } catch (err) {
            resolve({ ok: false, reason: `Invalid JSON: ${err.message}` });
          }
        });
      }
    );

    req.on('error', (err) => {
      resolve({ ok: false, reason: err.message });
    });
    req.on('timeout', () => {
      req.destroy(new Error('request timed out'));
    });
    req.end();
  });
}

function printHeader(label) {
  console.log('');
  console.log(label);
  console.log('-'.repeat(label.length));
}

function printTask(task) {
  if (!task) {
    console.log('Task: none found in tasks table.');
    return;
  }

  console.log(`Task ID      : ${task.id}`);
  console.log(`Platform/Type: ${task.platform}/${task.type}`);
  console.log(`Status       : ${task.status}`);
  console.log(`Retries      : ${task.retries}`);
  console.log(`Updated At   : ${task.updated_at || '(unknown)'}`);
  console.log(`Next Attempt : ${task.next_attempt_at || '(none)'}`);
  console.log(`Last Error   : ${task.last_error || '(none)'}`);
  const displayTitle = task.post_title || task.generated_title;
  if (displayTitle) {
    console.log(`Title        : ${displayTitle}`);
  }
}

async function main() {
  const task = chooseTask(requestedTaskId);
  if (requestedTaskId && !task) {
    console.error(`No task found for id: ${requestedTaskId}`);
    process.exit(1);
  }

  const state = summarizeState();
  const taskCounts = getTaskCounts();
  const platformPauses = getAllPlatformPauses();
  const debugTail = task ? readDebugTail(task.id) : null;
  const diag = task ? readDiagnosticSummary(task.platform, task.id) : null;
  const tabs = await fetchChromeTabs(chromePort);

  console.log('FlowAgent Debug Bundle');
  console.log('======================');
  console.log(`Generated At : ${new Date().toISOString()}`);
  console.log(`Chrome CDP   : http://127.0.0.1:${chromePort}`);

  printHeader('Task Status');
  printTask(task);

  printHeader('Task Counts');
  if (taskCounts.length === 0) {
    console.log('No tasks recorded.');
  } else {
    for (const row of taskCounts) {
      console.log(`${String(row.status || '').padEnd(14)} ${row.count}`);
    }
  }

  printHeader('State Flags');
  for (const [key, value] of Object.entries(state)) {
    console.log(`${key.padEnd(20)} ${value}`);
  }

  printHeader('Platform Pauses');
  const pausedEntries = Object.entries(platformPauses || {});
  if (pausedEntries.length === 0) {
    console.log('No platform pauses are active.');
  } else {
    for (const [name, info] of pausedEntries) {
      const code = info?.code || 'unknown';
      const until = info?.until ? new Date(info.until).toISOString() : '(manual/none)';
      const reason = info?.reason || '(no reason)';
      console.log(`${name}: ${code} until=${until}`);
      console.log(`  reason: ${reason}`);
    }
  }

  printHeader('Latest Debug Markers');
  if (!debugTail) {
    console.log('No task selected, so no debug markers are available.');
  } else if (!debugTail.lines.length) {
    console.log(`No events found at ${debugTail.path}`);
  } else {
    console.log(`Source: ${debugTail.path}`);
    for (const line of debugTail.lines) {
      console.log(line);
    }
  }

  printHeader('Latest Diagnostic Summary');
  if (!diag) {
    console.log('No task selected, so no diagnostic summary is available.');
  } else if (!diag.summary) {
    console.log(`No diagnostic file found at ${diag.path}`);
  } else {
    console.log(`Source      : ${diag.path}`);
    if (diag.summary.parse_error) {
      console.log(`Parse Error : ${diag.summary.parse_error}`);
    } else {
      console.log(`Written At  : ${diag.summary.written_at || '(unknown)'}`);
      console.log(`Outcome     : ${diag.summary.outcome || '(unknown)'}`);
      console.log(`Error Code  : ${diag.summary.error_code || '(none)'}`);
      console.log(`Error Phase : ${diag.summary.error_phase || '(none)'}`);
      console.log(`Message     : ${diag.summary.error_message || '(none)'}`);
      if (diag.summary.triage) {
        console.log(`Triage      : ${JSON.stringify(diag.summary.triage)}`);
      }
    }
  }

  printHeader('Active Browser Tabs');
  if (!tabs.ok) {
    console.log(`Could not query Chrome tabs: ${tabs.reason}`);
  } else if (!tabs.pages.length) {
    console.log('No page tabs reported by CDP.');
  } else {
    for (const [index, page] of tabs.pages.slice(0, 20).entries()) {
      const title = page.title || '(untitled)';
      console.log(`${String(index + 1).padStart(2, '0')}. ${title}`);
      console.log(`    ${page.url}`);
    }
    if (tabs.pages.length > 20) {
      console.log(`... ${tabs.pages.length - 20} additional tab(s) omitted`);
    }
  }
}

main().catch((err) => {
  console.error(err.stack || err.message || String(err));
  process.exit(1);
}).finally(() => {
  try {
    sqlite.close();
  } catch {
    // no-op
  }
});
NODE
}

cmd_debug_bundle() {
  local requested_task_id=""
  local watch_seconds=""
  local arg=""

  while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
      -h|--help)
        echo "Usage: $0 debug-bundle [task-id] [--watch <seconds>|--watch=<seconds>]"
        return 0
        ;;
      --watch)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --watch. Example: $0 debug-bundle --watch 5"
          return 1
        fi
        watch_seconds="$2"
        shift 2
        ;;
      --watch=*)
        watch_seconds="${arg#*=}"
        shift
        ;;
      --*)
        echo "Unknown option for debug-bundle: $arg"
        echo "Usage: $0 debug-bundle [task-id] [--watch <seconds>|--watch=<seconds>]"
        return 1
        ;;
      *)
        if [[ -z "$requested_task_id" ]]; then
          requested_task_id="$arg"
        else
          echo "Unexpected argument: $arg"
          echo "Usage: $0 debug-bundle [task-id] [--watch <seconds>|--watch=<seconds>]"
          return 1
        fi
        shift
        ;;
    esac
  done

  if [[ -n "$watch_seconds" ]]; then
    if ! [[ "$watch_seconds" =~ ^[0-9]+$ ]] || (( watch_seconds < 1 )); then
      echo "Invalid --watch value: $watch_seconds (must be an integer >= 1)"
      return 1
    fi
  fi

  if [[ -z "$watch_seconds" ]]; then
    _cmd_debug_bundle_snapshot "$requested_task_id"
    return $?
  fi

  while :; do
    if [[ -t 1 ]]; then
      command -v clear >/dev/null 2>&1 && clear
    fi
    echo "FlowAgent debug-bundle watch mode (${watch_seconds}s refresh). Press Ctrl+C to stop."
    if ! _cmd_debug_bundle_snapshot "$requested_task_id"; then
      if [[ -n "$requested_task_id" ]]; then
        return 1
      fi
    fi
    sleep "$watch_seconds"
  done
}

cmd_setup_db() {
  node db/sqlite.js
}

cmd_test() {
  env \
    -u SHOPIFY_RSS_URL \
    -u MYSHOPIFY_DOMAIN \
    -u SHOPIFY_STOREFRONT_DOMAIN \
    -u SHOPIFY_STOREFRONT_ACCESS_TOKEN \
    -u SHOPIFY_STOREFRONT_PRIVATE_TOKEN \
    FLOWAGENT_DB_PATH=':memory:' \
    node --test --test-reporter=spec \
    tests/browser.test.js \
    tests/config.test.js \
    tests/control.test.js \
    tests/llmRegistry.test.js \
    tests/preflight.test.js \
    tests/db.test.js \
    tests/rateLimiter.test.js \
    tests/postEnhancements.test.js \
    tests/blogger.test.js \
    tests/substack.test.js \
    tests/pinterestApi.test.js \
    tests/tumblr.test.js \
    tests/x.test.js \
    tests/linkedin.test.js \
    tests/shopify.test.js \
    tests/scheduler.test.js \
    tests/agentWatchdog.test.js \
    tests/agentLoop.test.js \
    tests/diagnostics.test.js \
    tests/platformContract.test.js \
    tests/tier2Resilience.test.js \
    tests/tier3Hardening.test.js \
    tests/challengeHandling.test.js
}

cmd_pause() {
  node -e "const c=require('./agent/control'); const r=c.pauseQueue('Paused from CLI'); console.log('Queue paused. Requeued running tasks: ' + r.requeuedRunning.length)"
}

cmd_resume() {
  node <<'NODE'
try {
  require('./agent/control').resumeQueue({ requireBrowserProfileReady: true });
  console.log('Queue resumed.');
} catch (err) {
  if (err && err.code === 'E_BROWSER_PROFILE_EMPTY') {
    console.error(err.message);
    if (err.profileHealth) {
      console.error(`Profile cookie count: ${err.profileHealth.totalCookies}`);
    }
    process.exit(1);
  }
  throw err;
}
NODE
}

cmd_open() {
  local url="http://localhost:$UI_PORT"
  if command -v open >/dev/null 2>&1; then
    open "$url"
  else
    echo "$url"
  fi
}

cmd_browser_help() {
  cat <<'EOF'
Browser automation help
=======================
BrowserMCP mode
---------------
Chrome can already be open and logged in. BrowserMCP still needs one tab to be connected:

1. In your existing Chrome window, select any authenticated platform tab.
2. Click the Browser MCP extension icon in the Chrome toolbar.
3. Click Connect.
4. Keep that Chrome window/tab open.
5. Resume FlowAgent:
     ./flowagent.sh resume
6. If the agent process has exited after pausing, start it again:
     ./flowagent.sh start

Playwright mode
---------------
Playwright is more reliable when it owns a dedicated FlowAgent Chrome profile instead of attaching to your live daily Chrome profile. The agent launches that configured profile itself and reuses the saved login cookies.

1. Switch modes:
     ./flowagent.sh browser-mode playwright
2. Open the dedicated FlowAgent Chrome profile once for manual login:
     ./flowagent.sh browser-login
3. Sign into every configured platform tab in that FlowAgent Chrome window.
   The login window is launched as normal Chrome without Playwright, remote
   debugging, mock-keychain, or sandbox-disabling flags so Google/social login
   pages treat it as a secure browser.
4. Close that login Chrome window when done so Playwright can reopen the profile.
5. Start or resume FlowAgent:
     ./flowagent.sh resume
     ./flowagent.sh start

If you still see browser launch failures, run:
  ./flowagent.sh doctor
EOF
}

cmd_browser_login() {
  local browser_context profile_dir
  local -a fields urls

  # Pre-check: the profile may already hold live sessions — tell the user
  # before making them re-login for nothing.
  local cookie_count
  cookie_count="$(node -e "process.stdout.write(String(require('./playwright/profileHealth').getProfileCookieHealth().totalCookies))" 2>/dev/null || echo 0)"
  if [[ "${cookie_count:-0}" -gt 0 ]]; then
    echo "Profile already has $cookie_count saved cookies — the platforms may already be logged in."
    echo "  Verify without manual login:  $0 login-check"
    echo "  Skip login and resume:        $0 finish-login"
    echo "Continuing with manual login anyway..."
    echo
  fi

  node -e "require('./agent/control').enterManualLoginMode('Manual browser login in progress')"
  # The running agent owns the Chrome profile in a headless instance (Playwright).
  # Stop it fully so the profile is released and nothing relaunches it headless —
  # otherwise the visible login window can't take the profile and never appears.
  _stop_agent_and_supervisor
  browser_context="$(_browser_startup_context)"
  IFS=$'\t' read -r -a fields <<< "$browser_context"
  profile_dir="${fields[0]:-}"

  mkdir -p "$profile_dir"

  urls=()
  for url in "${fields[@]:1}"; do
    [[ -n "$url" ]] && urls+=("$url")
  done

  _launch_login_chrome "$profile_dir" "${urls[@]}"

  cat <<EOF

A visible FlowAgent Chrome window should now be open.
Opened ${#urls[@]} configured platform tab(s). Sign into each platform, then keep the profile for future runs.
Current browser mode: $(node -e "const db=require('./db/sqlite'); process.stdout.write(db.getState('browser_mode') || 'browsermcp')")
Manual login Chrome is intentionally launched without Playwright automation or remote debugging flags.

When finished logging in:
  Run:  $0 finish-login     (closes this window, resumes the queue, restarts the agent)
Or manually:
  1. Close the login Chrome window.
  2. Restart the agent:  $0 start
  3. Resume posting:      $0 resume
EOF
}

# One command to leave manual-login mode: close the login Chrome, verify the
# profile has sessions, resume the queue, and restart the agent.
cmd_finish_login() {
  echo "Finishing manual login..."
  _stop_flowagent_chrome_profile "manual-login Chrome"
  _clear_profile_singleton_locks "$(_chrome_profile_dir)"

  if ! node <<'NODE'
try {
  require('./agent/control').resumeQueue({ requireBrowserProfileReady: true });
  console.log('Queue resumed (manual-login mode cleared).');
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
NODE
  then
    echo "Login sessions look missing — run $0 browser-login and sign in first."
    return 1
  fi

  cmd_start
}

# Deterministically verify each active platform's login state by loading its
# page in the automation profile and checking for login walls / captchas.
# No manual login and no LLM involved.
cmd_login_check() {
  if _is_running; then
    echo "Agent is running and owns the browser profile — showing its stored preflight results."
    echo "(Stop the agent first for a fresh check: $0 stop && $0 login-check)"
    node <<'NODE'
const db = require('./db/sqlite');
const all = db.getAllPlatformPreflights ? db.getAllPlatformPreflights() : {};
const names = Object.keys(all || {});
if (!names.length) { console.log('No stored preflight results yet.'); process.exit(0); }
for (const name of names.sort()) {
  const r = all[name] || {};
  const age = r.checkedAt ? Math.round((Date.now() - r.checkedAt) / 60000) + 'm ago' : 'unknown';
  console.log(`${(r.status || 'unknown').padEnd(16)} ${name.padEnd(12)} ${age}  ${r.reason || ''}`);
}
NODE
    return 0
  fi

  echo "Running deterministic login preflight across active platforms (headless, no manual login)..."
  node <<'NODE'
(async () => {
  const preflight = require('./agent/preflight');
  const { closeBrowser } = require('./playwright/browser');
  try {
    const summary = await preflight.runPreflight({ force: true });
    console.log('');
    for (const r of summary.results) {
      const mark = r.status === 'ok' ? 'OK  ' : (r.status === 'needs_attention' ? 'WARN' : 'FAIL');
      console.log(`${mark} ${r.platform.padEnd(12)} ${r.reason || ''}`);
    }
    console.log(`\nReady: ${summary.ready}/${summary.total}  needs attention: ${summary.needsAttention}  failed: ${summary.failed}`);
    if (summary.needsAttention === 0 && summary.failed === 0) {
      console.log('All platforms are logged in. If the queue is paused, run: ./flowagent.sh finish-login');
    }
  } finally {
    await closeBrowser().catch(() => {});
  }
  process.exit(0);
})().catch(err => { console.error(err.message); process.exit(1); });
NODE
}

cmd_browser_mode() {
  local mode="${1:-}"
  if [[ -z "$mode" ]]; then
    node -e "const db=require('./db/sqlite'); console.log(db.getState('browser_mode') || 'browsermcp')"
    return 0
  fi

  case "$mode" in
    browsermcp|playwright|both) ;;
    *)
      echo "Invalid browser mode: $mode"
      echo "Use one of: browsermcp, playwright, both"
      exit 1
      ;;
  esac

  node -e "const db=require('./db/sqlite'); db.setState('browser_mode', '$mode'); console.log('Browser mode set to: $mode')"
  if _is_running; then
    echo "Agent is running; restart it for this mode to take effect: ./flowagent.sh restart"
  else
    echo "Mode will take effect on next ./flowagent.sh start."
  fi
}

cmd_doctor() {
  echo "FlowAgent doctor"
  echo "================"
  cmd_status
  echo ""
  node <<'NODE'
const fs = require('fs');
const db = require('./db/sqlite');
const { getRuntimeConfig, getShopifyConfig, getSocialTargets } = require('./config');

const runtime = getRuntimeConfig(db.getSettings());
const shopify = getShopifyConfig(db.getSettings());
const targets = getSocialTargets(db.getSettings());
const browserMode = db.getState('browser_mode') || process.env.FLOWAGENT_BROWSER_MODE || 'browsermcp';
const platforms = ['quora', 'substack', 'facebook'];
const counts = {
  posts: db.db.prepare('SELECT COUNT(*) AS count FROM posts').get().count,
  postHistory: db.db.prepare('SELECT COUNT(*) AS count FROM post_history').get().count,
  pendingTasks: db.db.prepare("SELECT COUNT(*) AS count FROM tasks WHERE status='pending'").get().count,
  runningTasks: db.db.prepare("SELECT COUNT(*) AS count FROM tasks WHERE status='running'").get().count,
  failedTasks: db.db.prepare("SELECT COUNT(*) AS count FROM tasks WHERE status='failed'").get().count,
};
const oldest = db.getOldestPostNeedingPlatforms(platforms);
const logText = fs.existsSync('logs/flowagent.log')
  ? fs.readFileSync('logs/flowagent.log', 'utf8').slice(-30000)
  : '';
const browserDisconnected = /No connection to browser extension|BrowserMCP is not connected/i.test(logText);

console.log('Config:');
console.log(`  Qwen URL     : ${runtime.qwenUrl}`);
console.log(`  Qwen model   : ${runtime.qwenModel}`);
console.log(`  Browser mode : ${browserMode}`);
console.log(`  Chrome profile: ${runtime.chromeProfileDir}`);
console.log(`  Chrome debug : http://127.0.0.1:${runtime.chromeRemoteDebuggingPort}`);
console.log(`  Shopify source: ${shopify.rssUrl ? 'RSS' : (shopify.domain && shopify.storefrontToken ? 'Storefront API' : 'missing')}`);
if (browserMode === 'playwright' && /\/Google\/Chrome\/?$/i.test(runtime.chromeProfileDir)) {
  console.log('  Playwright note: live system Chrome profiles will lock if Chrome is already open. Prefer a dedicated profile such as ./chrome-profile');
}
console.log('');
console.log('Targets:');
for (const platform of platforms) {
  console.log(`  ${platform.padEnd(9)}: ${targets[platform] || '(missing)'}`);
}
console.log('');
console.log('Database:');
console.log(`  Kill switch  : ${db.isKilled() ? 'enabled' : 'disabled'}`);
console.log(`  Posts        : ${counts.posts}`);
console.log(`  Post history : ${counts.postHistory}`);
console.log(`  Pending tasks: ${counts.pendingTasks}`);
console.log(`  Running tasks: ${counts.runningTasks}`);
console.log(`  Failed tasks : ${counts.failedTasks}`);
if (oldest) {
  console.log(`  Next blog    : ${oldest.title}`);
  console.log(`  Missing      : ${db.getPlatformsNeedingPost(oldest.id, platforms).join(', ')}`);
} else {
  console.log('  Next blog    : none; all focused platforms are queued or posted');
}
console.log('');
console.log(`MCP config    : ${fs.existsSync('.vscode/mcp.json') ? '.vscode/mcp.json found' : 'missing'}`);
if (browserDisconnected) {
  console.log('BrowserMCP    : recent disconnect seen; run ./flowagent.sh browser-help');
}
NODE
}

cmd_queue_status() {
  node <<'NODE'
const db = require('./db/sqlite');
const rows = db.db.prepare(`
  SELECT status, platform, type, COUNT(*) AS count
  FROM tasks
  GROUP BY status, platform, type
  ORDER BY status, platform, type
`).all();
const singlePlatformDebug = db.getState('single_platform_debug') === 'true';
const delayed = db.db.prepare(`
  SELECT platform, type, COUNT(*) AS count, MIN(next_attempt_at) AS next_attempt_at
  FROM tasks
  WHERE status = 'pending'
    AND next_attempt_at IS NOT NULL
    AND datetime(next_attempt_at) > datetime('now')
  GROUP BY platform, type
  ORDER BY next_attempt_at ASC
`).all();

if (rows.length === 0) {
  console.log('No tasks found.');
  process.exit(0);
}

console.log('Task queue:');
if (singlePlatformDebug) {
  console.log('  scheduler paused: single-platform debug mode is enabled');
}
for (const row of rows) {
  console.log(`  ${row.status.padEnd(8)} ${String(row.platform || '').padEnd(10)} ${String(row.type || '').padEnd(10)} ${row.count}`);
}
if (delayed.length) {
  console.log('Delayed retries:');
  for (const row of delayed) {
    console.log(`  ${String(row.platform || '').padEnd(10)} ${String(row.type || '').padEnd(10)} ${row.count} next ${row.next_attempt_at}`);
  }
}
NODE
}

cmd_schedule() {
  node -e "require('./agent/scheduler').runScheduler().catch(err => { console.error(err.message); process.exit(1); })"
}

cmd_rebuild_focused_queue() {
  local confirm="${1:-}"
  if _is_running; then
    echo "Refusing to rebuild the queue while the agent is running."
    echo "Run: $0 stop"
    echo "Then: $0 rebuild-focused-queue --yes"
    exit 1
  fi

  if [[ "$confirm" != "--yes" ]]; then
    node <<'NODE'
const db = require('./db/sqlite');
const active = db.db.prepare("SELECT COUNT(*) AS count FROM tasks WHERE status IN ('pending','running')").get().count;
const queuedHistory = db.db.prepare("SELECT COUNT(*) AS count FROM post_history WHERE status IN ('queued','running','failed')").get().count;
const singlePlatformDebug = db.getState('single_platform_debug') === 'true';
console.log('Dry run only.');
console.log(`Would delete pending/running tasks: ${active}`);
console.log(`Would clear queued/running/failed post history rows: ${queuedHistory}`);
if (singlePlatformDebug) console.log('Would disable single-platform debug mode.');
console.log('Done post_history rows would be preserved.');
NODE
    echo "Run with --yes to rebuild the focused Quora/Substack/Facebook queue."
    return 0
  fi

  node <<'NODE'
const db = require('./db/sqlite');
const { runScheduler } = require('./agent/scheduler');

const deletedTasks = db.db.prepare("DELETE FROM tasks WHERE status IN ('pending','running')").run().changes;
const deletedHistory = db.db.prepare("DELETE FROM post_history WHERE status IN ('queued','running','failed')").run().changes;
db.db.prepare('UPDATE posts SET processed = 0').run();
db.setState('single_platform_debug', 'false');

runScheduler()
  .then(() => {
    const pending = db.db.prepare("SELECT COUNT(*) AS count FROM tasks WHERE status='pending'").get().count;
    console.log(`Deleted pending/running tasks: ${deletedTasks}`);
    console.log(`Cleared queued/running/failed post history rows: ${deletedHistory}`);
    console.log(`Pending tasks after focused scheduler run: ${pending}`);
  })
  .catch(err => {
    console.error(err.message);
    process.exit(1);
  });
NODE
}

cmd_reset_one_per_platform() {
  local confirm="${1:-}"
  if _is_running; then
    echo "Refusing to reset tasks while the agent is running."
    echo "Run: $0 pause"
    echo "Wait for the agent to stop, or run: $0 stop"
    exit 1
  fi

  if [[ "$confirm" != "--yes" ]]; then
    node <<'NODE'
const db = require('./db/sqlite');
const platforms = ['quora', 'substack', 'facebook'];
const counts = {
  tasks: db.db.prepare('SELECT COUNT(*) AS count FROM tasks').get().count,
  logs: db.db.prepare('SELECT COUNT(*) AS count FROM logs').get().count,
  history: db.db.prepare('SELECT COUNT(*) AS count FROM post_history').get().count,
};
const post = db.db.prepare(`
  SELECT id, title, source_url
  FROM posts
  WHERE COALESCE(TRIM(body), '') <> ''
  ORDER BY rowid ASC
  LIMIT 1
`).get();

console.log('Dry run only.');
console.log(`Would delete tasks: ${counts.tasks}`);
console.log(`Would delete logs: ${counts.logs}`);
console.log(`Would clear post_history rows: ${counts.history}`);
console.log(`Would create ${platforms.length} tasks: ${platforms.join(', ')}`);
console.log(`Selected post: ${post ? post.title : '(none with body found)'}`);
NODE
    echo "Run with --yes to reset to exactly one task per focused platform."
    return 0
  fi

  node <<'NODE'
const db = require('./db/sqlite');
const { getSocialTargets } = require('./config');

const platforms = ['quora', 'substack', 'facebook'];
const post = db.db.prepare(`
  SELECT *
  FROM posts
  WHERE COALESCE(TRIM(body), '') <> ''
  ORDER BY rowid ASC
  LIMIT 1
`).get();

if (!post) {
  console.error('No Shopify post with a body was found. Sync Shopify content first.');
  process.exit(1);
}

const targets = getSocialTargets(db.getSettings());
const tx = db.db.transaction(() => {
  const deletedLogs = db.db.prepare('DELETE FROM logs').run().changes;
  const deletedHistory = db.db.prepare('DELETE FROM post_history').run().changes;
  const deletedTasks = db.db.prepare('DELETE FROM tasks').run().changes;
  db.db.prepare('UPDATE posts SET processed = 0').run();
  db.setState('kill', 'true');
  db.setState('single_platform_debug', 'true');

  const taskIds = [];
  for (const platform of platforms) {
    const taskId = db.insertTask({ type: 'post', platform, content_id: post.id, priority: 1 });
    taskIds.push({ platform, taskId });
    db.recordPostHistory({
      post_id: post.id,
      platform,
      task_id: taskId,
      target_url: targets[platform] || '',
      status: 'queued',
      result: 'Clean one-per-platform reset',
    });
    db.insertLog({
      task_id: taskId,
      step: 'queued_clean_reset',
      result: `Queued clean ${platform} test task for "${post.title}"`,
    });
  }

  db.markPostProcessed(post.id);
  return { deletedLogs, deletedTasks, deletedHistory, taskIds };
});

const result = tx();
console.log(`Deleted tasks: ${result.deletedTasks}`);
console.log(`Deleted logs: ${result.deletedLogs}`);
console.log(`Cleared post_history rows: ${result.deletedHistory}`);
console.log(`Selected post: ${post.title}`);
console.log('Created tasks:');
for (const row of result.taskIds) {
  console.log(`  ${row.platform.padEnd(9)} ${row.taskId}`);
}
console.log(`Kill switch enabled. Review the ${result.taskIds.length} tasks, then run ./flowagent.sh resume && ./flowagent.sh start when ready.`);
console.log('Scheduler paused in single-platform debug mode. Run ./flowagent.sh rebuild-focused-queue --yes later to return to normal queueing.');
NODE
}

usage() {
  echo "Usage: $0 {start|stop|restart|restart-ui|status|doctor|browser-help|browser-login|finish-login|login-check|browser-mode|queue-status|schedule|rebuild-focused-queue|reset-one-per-platform|setup-db|test|pause|resume|open|logs|ui-logs|llm-logs|debug-bundle}"
  echo ""
  echo "  start     Start the LLM server, agent, and UI dashboard"
  echo "  stop      Stop all three"
  echo "  restart   Restart all three"
  echo "  restart-ui Restart only the dashboard UI"
  echo "  status    Show running status of all three"
  echo "  doctor    Check runtime config, targets, DB counts, and MCP config"
  echo "  browser-help  Show how to connect the BrowserMCP Chrome extension"
  echo "  browser-login Open the dedicated FlowAgent Chrome profile for Playwright login"
  echo "  finish-login  Close the login Chrome, resume the queue, restart the agent"
  echo "  login-check   Verify each platform's login state deterministically (no manual login)"
  echo "  browser-mode [browsermcp|playwright|both]  Show or set the browser automation backend"
  echo "  queue-status  Show task counts by status/platform/type"
  echo "  schedule  Run one scheduler cycle without starting the agent"
  echo "  rebuild-focused-queue [--yes]  Rebuild pending queue for Quora/Substack/Facebook"
  echo "  reset-one-per-platform [--yes]  Delete task history and create one paused task per focused platform"
  echo "  setup-db  Initialise or migrate the SQLite database"
  echo "  test      Run the FlowAgent test suite"
  echo "  debug-bundle [task-id] [--watch <seconds>]  Print task/debug/diagnostic/state/tab snapshot"
  echo "  pause     Enable the DB kill switch without stopping processes"
  echo "  resume    Disable the DB kill switch"
  echo "  open      Open the dashboard URL in your browser"
  echo "  logs      Tail agent log"
  echo "  ui-logs   Tail UI server log"
  echo "  llm-logs  Tail LLM server log"
  echo "  diagnose [latest|<platform>|<task-id>|<plat>:<id>|--list]"
  echo "            Print a structured failure dossier for fast LLM/human triage"
  echo ""
  echo "  UI_PORT=3001       $0 start   (UI default: 3000)"
  echo "  LLM_PORT=8080      $0 start   (LLM default: 8080)"
  echo "  QWEN_MODEL=<name>  $0 start   (model default: mlx-community/Qwen3-14B-6bit)"
  echo "  FA_AGENT_AUTORESTART=0 $0 start   (disable crash auto-restart supervisor)"
  exit 1
}

case "${1:-}" in
  start)     cmd_start    ;;
  stop)      cmd_stop     ;;
  status)    cmd_status   ;;
  restart)   cmd_restart  ;;
  restart-ui) cmd_restart_ui ;;
  doctor)    cmd_doctor   ;;
  browser-help) cmd_browser_help ;;
  browser-login) cmd_browser_login ;;
  finish-login)  cmd_finish_login ;;
  login-check)   cmd_login_check ;;
  browser-mode) cmd_browser_mode "${2:-}" ;;
  queue-status) cmd_queue_status ;;
  schedule)   cmd_schedule ;;
  rebuild-focused-queue) cmd_rebuild_focused_queue "${2:-}" ;;
  reset-one-per-platform) cmd_reset_one_per_platform "${2:-}" ;;
  setup-db)  cmd_setup_db ;;
  test)      cmd_test     ;;
  pause)     cmd_pause    ;;
  resume)    cmd_resume   ;;
  open)      cmd_open     ;;
  logs)      cmd_logs     ;;
  ui-logs)   cmd_ui_logs  ;;
  llm-logs)  cmd_llm_logs ;;
  debug-bundle) cmd_debug_bundle "${@:2}" ;;
  diagnose)  cmd_diagnose "$@" ;;
  *)         usage        ;;
esac
