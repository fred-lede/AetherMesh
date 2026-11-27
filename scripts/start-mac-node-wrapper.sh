#!/usr/bin/env bash
set -euo pipefail

# launchd entry point: wait for the external drive to mount, then start the node.
# The control plane target is resolved by start-mac-node.sh from .env.

# Prefer the external drive when it is in use, otherwise fall back to wherever
# this script actually lives (e.g. /Users/<user>/AetherMesh).
AETHERMESH_DIR="${AETHERMESH_DIR:-/Volumes/Ai-2TB/ai/AetherMesh}"
if [ ! -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
  AETHERMESH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# Wait for the drive to mount (up to 60 seconds)
for i in $(seq 1 60); do
  if [ -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
    break
  fi
  sleep 1
done

# If still not present after timeout, exit (launchd will retry via KeepAlive)
if [ ! -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
  echo "ERROR: AetherMesh repo not found at $AETHERMESH_DIR" >&2
  exit 1
fi

exec /bin/bash "$AETHERMESH_DIR/scripts/start-mac-node.sh"
