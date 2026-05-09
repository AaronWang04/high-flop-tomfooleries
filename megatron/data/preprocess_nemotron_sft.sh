#!/usr/bin/bash
# Pipeline: extend cl100k tokenizer with chat/think/tool tokens, download
# English-only Nemotron-Post-Training-Dataset-v2 parquets, format as ChatML
# with thinking traces, and tokenize to Megatron .bin/.idx.
#
# Usage:
#   sbatch megatron/data/preprocess_nemotron_sft.sh                       # smoke (1 file/split)
#   MAX_FILES_PER_SPLIT=99 sbatch megatron/data/preprocess_nemotron_sft.sh # full
#
#SBATCH --job-name=nemotron-sft-prep
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

# Smoke = 1 file per split (~30K examples). Set to 99 for full English (~1.4M).
MAX_FILES_PER_SPLIT=${MAX_FILES_PER_SPLIT:-1}

DATA_ROOT=${DATA_ROOT:-${SCRATCH_ROOT}/data/nemotron_sft_v2}
TOK_BASE="${REPO_ROOT}/assets/hf/cl100k_base"
TOK_OUT="${REPO_ROOT}/assets/hf/cl100k_chat"
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${DATA_ROOT}/nemotron_sft_v2_en}
WORKERS=${WORKERS:-64}

mkdir -p "${REPO_ROOT}/outputs/slurm"
mkdir -p "${DATA_ROOT}/parquet"
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/slurm/preprocess_nemotron_sft_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=\$(cat ${SCRATCH_ROOT}/.cache/huggingface/token)

echo "=== Step 1: Extend cl100k tokenizer with chat/think/tool tokens ==="
python3 << 'PY'
import os, shutil
from transformers import AutoTokenizer, AddedToken

base = "${TOK_BASE}"
out  = "${TOK_OUT}"

# cl100k_base already has <|im_start|>=100264, <|im_end|>=100265, <|endofprompt|>=100276.
# 14 new tokens — let transformers auto-assign IDs past existing added_tokens.
# Verified empirically: they land at 100277..100290 (next free past <|endofprompt|>).
NEW_TOKENS = [
    "<|think|>", "<|/think|>",
    "<|tool_call|>", "<|/tool_call|>",
    "<|tool_response|>", "<|/tool_response|>",
] + [f"<|reserved_{i}|>" for i in range(8)]

if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(base, out)

t = AutoTokenizer.from_pretrained(out)
n = t.add_tokens([AddedToken(c, special=True, normalized=False) for c in NEW_TOKENS], special_tokens=True)
print(f"Added {n} new special tokens")
t.save_pretrained(out)

# Reload to get fresh state and verify single-token encoding.
t = AutoTokenizer.from_pretrained(out)
print(f"Reloaded vocab len: {len(t)}; max added id check follows")
all_chat_tokens = ["<|im_start|>", "<|im_end|>", "<|endofprompt|>"] + NEW_TOKENS
for c in all_chat_tokens:
    tid = t.convert_tokens_to_ids(c)
    print(f"  {c!r:24s} -> {tid}")
# Smoke-encode a chat snippet to ensure the chat tokens land as single IDs.
sample = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<|think|>plan<|/think|>\nhi back<|im_end|>\n"
ids = t.encode(sample)
print(f"\nencoded ids ({len(ids)} tokens):")
print(ids)
# Sanity check: chat markers must appear as single tokens, not as BPE pieces.
expected_singles = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]
for c in expected_singles:
    cid = t.convert_tokens_to_ids(c)
    assert cid in ids, f"{c!r} (id={cid}) missing from encoded ids — would tokenize as BPE pieces!"
print("OK: all chat markers tokenize as single tokens.")
PY

echo "=== Step 2: Download English parquets (max ${MAX_FILES_PER_SPLIT}/split) ==="
python3 << 'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
all_files = api.list_repo_files("nvidia/Nemotron-Post-Training-Dataset-v2", repo_type="dataset")
EN_PREFIXES = ("data/chat-", "data/code-", "data/math-", "data/stem-")
matched = sorted([f for f in all_files if f.endswith(".parquet") and f.startswith(EN_PREFIXES)])
print(f"Matching English parquets: {len(matched)}")
by_split = {p: [] for p in EN_PREFIXES}
for f in matched:
    for p in EN_PREFIXES:
        if f.startswith(p):
            by_split[p].append(f); break
to_get = []
for p, lst in by_split.items():
    keep = lst[:${MAX_FILES_PER_SPLIT}]
    print(f"  {p} -> {len(keep)}/{len(lst)} files")
    to_get.extend(keep)
print(f"Downloading {len(to_get)} files...")
for i, fn in enumerate(to_get):
    hf_hub_download(
        repo_id="nvidia/Nemotron-Post-Training-Dataset-v2",
        repo_type="dataset",
        filename=fn,
        local_dir="${DATA_ROOT}/parquet",
    )
    print(f"  [{i+1}/{len(to_get)}] {fn}")
PY

echo "=== Step 3: Format with ChatML + thinking, write JSONL ==="
python3 << 'PY'
import pyarrow.parquet as pq
import json, glob, os
parquet_files = sorted(glob.glob("${DATA_ROOT}/parquet/data/*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

# ChatML chat-format: <|im_start|>{role}\n{content}<|im_end|>\n
# Reasoning trace (if present) wraps the assistant's content as
#   <|think|>{reasoning}<|/think|>\n{content}
def render(messages, reasoning):
    parts = []
    for i, m in enumerate(messages):
        role = m["role"]
        content = m["content"] or ""
        is_last_assistant = (role == "assistant" and i == len(messages) - 1)
        if is_last_assistant and reasoning:
            content = f"<|think|>{reasoning}<|/think|>\n{content}"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"

out_path = "${DATA_ROOT}/nemotron_sft_v2_en.jsonl"
total_rows = 0
skipped = 0
with open(out_path, "w") as out:
    for i, pf in enumerate(parquet_files):
        pq_file = pq.ParquetFile(pf)
        cols = ["messages", "reasoning"]
        for batch in pq_file.iter_batches(batch_size=2000, columns=cols):
            msgs_col = batch.column("messages").to_pylist()
            rsn_col = batch.column("reasoning").to_pylist()
            for messages, reasoning in zip(msgs_col, rsn_col):
                if not messages:
                    skipped += 1; continue
                text = render(messages, reasoning)
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total_rows += 1
        print(f"  [{i+1}/{len(parquet_files)}] {os.path.basename(pf)}: rows={total_rows} skipped={skipped}")
print(f"JSONL written: {out_path}, {total_rows} examples (skipped {skipped})")
print(f"Sample (first 800 chars):")
with open(out_path) as f:
    line = f.readline()
    rec = json.loads(line)
    print(rec["text"][:800])
PY

echo "=== Step 4: Tokenize with extended cl100k -> .bin/.idx ==="
python3 "${MEGATRON_ROOT}/tools/preprocess_data.py" \\
    --input "${DATA_ROOT}/nemotron_sft_v2_en.jsonl" \\
    --output-prefix "${OUTPUT_PREFIX}" \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-model "${TOK_OUT}" \\
    --json-keys text \\
    --workers ${WORKERS} \\
    --append-eod

echo "=== Done. Output:"
ls -lh "\$(dirname "${OUTPUT_PREFIX}")"/\$(basename "${OUTPUT_PREFIX}")*
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "MAX_FILES_PER_SPLIT: ${MAX_FILES_PER_SPLIT}"
echo "Output prefix:       ${OUTPUT_PREFIX}"
echo "Tokenizer out:       ${TOK_OUT}"
echo "Launch script:       ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
