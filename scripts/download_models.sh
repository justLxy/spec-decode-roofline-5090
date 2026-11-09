#!/usr/bin/env bash
# Download pinned model weights to $WORK/models (public repos, no token needed).
# Target: NousResearch mirror of Llama-3.1-8B-Instruct. Its config.json, generation_config.json,
# tokenizer.json, special_tokens_map.json and safetensors index are byte-identical (same git blob id) to
# meta-llama/Llama-3.1-8B-Instruct; the four safetensors shards match the official sizes. Its
# tokenizer_config.json carries an older chat template, so we serve with configs/chat_templates/llama31_instruct.jinja.
set -euo pipefail
WORK=${WORK:-/workspace}
MODELS=$WORK/models
mkdir -p "$MODELS"
TARGET_REPO=NousResearch/Meta-Llama-3.1-8B-Instruct
TARGET_REV=d10aef7999a2b5ba950ab3974312feeedbfe0b77
DRAFT_REPO=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
DRAFT_REV=ada412b672e293d682423de84a095447bf38a637

hf download "$TARGET_REPO" --revision "$TARGET_REV" --exclude "original/*" --local-dir "$MODELS/Llama-3.1-8B-Instruct"
hf download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$MODELS/EAGLE3-LLaMA3.1-Instruct-8B"
( cd "$MODELS/Llama-3.1-8B-Instruct" && sha256sum model-*.safetensors ) | tee "$MODELS/target_sha256.txt"
( cd "$MODELS/EAGLE3-LLaMA3.1-Instruct-8B" && sha256sum pytorch_model.bin ) | tee "$MODELS/draft_sha256.txt"
du -sh "$MODELS"/*
