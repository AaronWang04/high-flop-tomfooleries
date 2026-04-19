#!/usr/bin/bash
# One-time setup for pretraining
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Dependencies ---
echo "==> Installing dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# --- Tokenizers ---
echo "==> Downloading tokenizers..."
huggingface-cli download Qwen/Qwen3-8B tokenizer.json tokenizer_config.json merges.txt vocab.json \
    --local-dir "${SCRIPT_DIR}/assets/hf/Qwen3-8B"

huggingface-cli download meta-llama/Llama-3.1-8B tokenizer.json tokenizer_config.json special_tokens_map.json original/tokenizer.model \
    --local-dir "${SCRIPT_DIR}/assets/hf/Llama-3.1-8B"

# --- Output dirs ---
mkdir -p "${SCRIPT_DIR}/outputs"

echo ""
echo "==> Setup complete. Run training with:"
echo "    cd ${SCRIPT_DIR} && bash train.sh"
