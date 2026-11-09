#!/usr/bin/env bash
# Pull results back from the GPU instance. Usage: scripts/fetch_results.sh [ssh_host]
set -euo pipefail
HOST=${1:-vast5090}
REMOTE_DIR=${REMOTE_DIR:-/workspace/hpml}
rsync -az "$HOST:$REMOTE_DIR/results/" ./results/
rsync -az --include '*/' --include '*.nsys-rep' --include '*.ncu-rep' --exclude '*' \
  "$HOST:$REMOTE_DIR/data/traces/" ./data/traces/ || true
echo "fetched results from $HOST"
