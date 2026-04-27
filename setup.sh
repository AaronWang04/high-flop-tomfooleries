#!/usr/bin/bash
# One-time setup for pretraining
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- TorchTitan nightly (local clone) ---
echo "==> Installing TorchTitan nightly from local clone..."
pip install -e "${REPOS_DIR}/torchtitan"

# --- Megatron-LM (local clone) ---
echo "==> Installing Megatron-LM from local clone..."
pip install -e "${REPOS_DIR}/Megatron-LM"

# --- Other dependencies ---
echo "==> Installing other dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# --- Tokenizers ---
echo "==> Downloading tokenizers..."
huggingface-cli download Qwen/Qwen3-8B tokenizer.json tokenizer_config.json merges.txt vocab.json \
    --local-dir "${SCRIPT_DIR}/assets/hf/Qwen3-8B"

huggingface-cli download meta-llama/Llama-3.1-8B tokenizer.json tokenizer_config.json special_tokens_map.json original/tokenizer.model \
    --local-dir "${SCRIPT_DIR}/assets/hf/Llama-3.1-8B"

huggingface-cli download Qwen/Qwen3.5-9B tokenizer.json tokenizer_config.json merges.txt vocab.json \
    --local-dir "${SCRIPT_DIR}/assets/hf/Qwen3.5-9B"

# --- Output dirs ---
mkdir -p "${SCRIPT_DIR}/outputs/torchtitan"
mkdir -p "${SCRIPT_DIR}/outputs/megatron"

echo ""
echo "==> Setup complete. Run training with:"
echo "    TorchTitan: cd ${SCRIPT_DIR} && bash torchtitan/train.sh"
echo "    Megatron:   cd ${SCRIPT_DIR} && RECIPE=llama3_8b bash megatron/train.sh"
