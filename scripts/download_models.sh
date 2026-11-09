#!/usr/bin/env bash
# Download pinned model weights to $WORK/models (public repos, no token needed).
# Target: official meta-llama/Llama-3.1-8B-Instruct (gated; run `hf auth login` first). Its four safetensors
# shards have the same sha256 as the NousResearch/Meta-Llama-3.1-8B-Instruct mirror (verified 2026-11-09),
# so the mirror is a drop-in fallback for the weights; only tokenizer_config.json (chat template) differs.
set -euo pipefail
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
WORK=${WORK:-/workspace}
MODELS=$WORK/models
mkdir -p "$MODELS"
TARGET_REPO=meta-llama/Llama-3.1-8B-Instruct
TARGET_REV=0e9e39f249a16976918f6564b8830bc894c89659
DRAFT_REPO=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
DRAFT_REV=ada412b672e293d682423de84a095447bf38a637

hf download "$TARGET_REPO" --revision "$TARGET_REV" --exclude "original/*" --local-dir "$MODELS/Llama-3.1-8B-Instruct"
hf download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$MODELS/EAGLE3-LLaMA3.1-Instruct-8B"
( cd "$MODELS/Llama-3.1-8B-Instruct" && sha256sum model-*.safetensors ) | tee "$MODELS/target_sha256.txt"
( cd "$MODELS/EAGLE3-LLaMA3.1-Instruct-8B" && sha256sum pytorch_model.bin ) | tee "$MODELS/draft_sha256.txt"
du -sh "$MODELS"/*
