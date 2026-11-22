#!/usr/bin/env bash
set -euo pipefail

# launchd entry point: wait for the external drive to mount, then start the node.
# The control plane target is resolved by start-mac-node.sh from .env.

AETHERMESH_DIR="/Volumes/Ai-2TB/ai/AetherMesh"

# Wait for external drive to mount (up to 60 seconds)
for i in $(seq 1 60); do
  if [ -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
    break
  fi
  sleep 1
done

# If still not mounted after timeout, exit (launchd will retry via KeepAlive)
if [ ! -f "$AETHERMESH_DIR/config/cluster.yaml" ]; then
  exit 1
fi

exec /bin/bash "$AETHERMESH_DIR/scripts/start-mac-node.sh"
