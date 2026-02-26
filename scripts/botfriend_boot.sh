#!/usr/bin/env bash
set -euo pipefail

# Boot/runtime launcher for Botronka.
# - Runs from project root
# - Uses project venv when available
# - Optionally starts local llama-server based on config/config.yaml
# - Starts main.py

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${BOTFRIEND_CONFIG:-$PROJECT_DIR/config/config.yaml}"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
SYS_PY="$(command -v python3 || true)"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

if [[ -x "$VENV_PY" ]]; then
  PY_BIN="$VENV_PY"
elif [[ -n "$SYS_PY" ]]; then
  PY_BIN="$SYS_PY"
else
  echo "ERROR: python3 not found. Install Python 3 and/or create .venv." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Load autostart config into shell variables.
eval "$($PY_BIN - "$CONFIG_PATH" <<'PY'
import shlex
import sys
import yaml

cfg_path = sys.argv[1]
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
a = cfg.get("autostart", {}) if isinstance(cfg, dict) else {}

vals = {
    "AUTOSTART_ENABLED": "1" if bool(a.get("enabled", True)) else "0",
    "START_LLM_SERVER": "1" if bool(a.get("start_llm_server", False)) else "0",
    "LLM_SERVER_BIN": str(a.get("llm_server_bin", "/home/botronka/llama.cpp/build/bin/llama-server")),
    "LLM_MODEL_PATH": str(a.get("llm_model_path", "/home/botronka/botfriend/models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf")),
    "LLM_HOST": str(a.get("llm_host", "127.0.0.1")),
    "LLM_PORT": str(a.get("llm_port", 8080)),
    "LLM_CTX_SIZE": str(a.get("llm_ctx_size", 1024)),
    "LLM_STARTUP_TIMEOUT_S": str(a.get("llm_startup_timeout_s", 20)),
}

for k, v in vals.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"

if [[ "$AUTOSTART_ENABLED" != "1" ]]; then
  echo "Autostart disabled in config (autostart.enabled=false). Exiting boot launcher."
  exit 0
fi

if [[ "$START_LLM_SERVER" == "1" ]]; then
  if [[ ! -x "$LLM_SERVER_BIN" ]]; then
    echo "WARNING: llama-server binary not executable: $LLM_SERVER_BIN" >&2
  elif [[ ! -f "$LLM_MODEL_PATH" ]]; then
    echo "WARNING: LLM model not found: $LLM_MODEL_PATH" >&2
  elif pgrep -f "llama-server.*--host[[:space:]]+$LLM_HOST.*--port[[:space:]]+$LLM_PORT" >/dev/null 2>&1; then
    echo "llama-server already running on $LLM_HOST:$LLM_PORT"
  else
    echo "Starting llama-server on $LLM_HOST:$LLM_PORT ..."
    "$LLM_SERVER_BIN" \
      -m "$LLM_MODEL_PATH" \
      --host "$LLM_HOST" \
      --port "$LLM_PORT" \
      -c "$LLM_CTX_SIZE" \
      >"$LOG_DIR/llama-server.log" 2>&1 &

    LLM_PID=$!
    echo "$LLM_PID" >"$LOG_DIR/llama-server.pid"

    # Wait briefly for port to become reachable.
    timeout_s="${LLM_STARTUP_TIMEOUT_S:-20}"
    start_ts=$(date +%s)
    while true; do
      if "$PY_BIN" - "$LLM_HOST" "$LLM_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(0.6)
try:
    s.connect((host, port))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
      then
        echo "llama-server is reachable at $LLM_HOST:$LLM_PORT"
        break
      fi

      now_ts=$(date +%s)
      if (( now_ts - start_ts >= timeout_s )); then
        echo "WARNING: llama-server did not become reachable within ${timeout_s}s" >&2
        break
      fi
      sleep 1
    done
  fi
fi

exec "$PY_BIN" "$PROJECT_DIR/main.py"
