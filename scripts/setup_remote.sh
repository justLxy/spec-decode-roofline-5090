#!/usr/bin/env bash
# One-time environment setup on the rented RTX 5090 instance.
# Usage (on the remote machine): bash scripts/setup_remote.sh
set -euo pipefail

WORK=${WORK:-/workspace}
VLLM_VERSION=${VLLM_VERSION:-0.29.0}
TORCH_BACKEND=${TORCH_BACKEND:-cu129}
export HF_HOME=${HF_HOME:-$WORK/.hf_home}

mkdir -p "$WORK" "$HF_HOME"
cd "$WORK/hpml" 2>/dev/null || { echo "Expected repo at $WORK/hpml"; exit 1; }

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
uv pip install "vllm==${VLLM_VERSION}" --torch-backend="${TORCH_BACKEND}"
uv pip install -e ".[gpu,dev]" --torch-backend="${TORCH_BACKEND}"
uv pip install -U "huggingface_hub[cli]"

# Nsight Systems CLI for timeline profiling (Nsight Compute ships with the CUDA devel image if present).
if ! command -v nsys >/dev/null; then
  apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli || \
    echo "WARN: nsight-systems-cli not available from apt; install manually."
fi

cat >> "$HOME/.bashrc" <<RC
export HF_HOME=$HF_HOME
export PATH=\$HOME/.local/bin:\$PATH
source $WORK/venv/bin/activate
cd $WORK/hpml
RC

echo "Setup done. Next: 'huggingface-cli login' and 'wandb login' (run them yourself), then scripts/collect_env.sh"
