#!/usr/bin/env bash
# Push local code to the GPU instance (excludes large/private files). Usage: scripts/sync_to_remote.sh [ssh_host]
set -euo pipefail
HOST=${1:-vast5090}
REMOTE_DIR=${REMOTE_DIR:-/workspace/hpml}
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
  --exclude 'data/raw/' --exclude 'data/processed/' --exclude 'data/traces/' --exclude 'results/raw/' \
  --exclude 'EXPERIMENT_LOG.md' --exclude 'WRITING_STYLE.md' --exclude 'CLAUDE.md' --exclude '.claude/' \
  --exclude '*.pdf' --exclude 'wandb/' \
  ./ "$HOST:$REMOTE_DIR/"
echo "synced to $HOST:$REMOTE_DIR"
