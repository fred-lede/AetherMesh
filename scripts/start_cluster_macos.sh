#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

mkdir -p logs

WORKER_PORT="${AIIH_WORKER_RPC_PORT:-9300}"
NODE_PORT="${AIIH_NODE_PORT:-9400}"
HOST="${AIIH_HOST:-0.0.0.0}"

start_service() {
  local name="$1"
  local cmd="$2"
  local pid_file="logs/${name}.pid"
  local log_file="logs/${name}.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "${name} already running (PID ${old_pid})"
      return
    fi
  fi

  nohup bash -lc "$cmd" >"$log_file" 2>&1 &
  local new_pid=$!
  echo "$new_pid" >"$pid_file"
  echo "Started ${name} (PID ${new_pid})"
}

start_service "aiih-worker-agent" "python -m uvicorn node.worker_agent:app --host ${HOST} --port ${WORKER_PORT}"
start_service "aiih-node-agent" "python -m uvicorn node.node_agent:app --host ${HOST} --port ${NODE_PORT}"

echo "Logs:"
echo "  ${ROOT_DIR}/logs/aiih-worker-agent.log"
echo "  ${ROOT_DIR}/logs/aiih-node-agent.log"
