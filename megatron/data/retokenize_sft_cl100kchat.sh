#!/usr/bin/bash
# Re-tokenize the existing nemotron SFT JSONL with the new Cl100kChat tokenizer
# (chat/think/tool tokens at explicit IDs). Overwrites the bin/idx in place.
#
#SBATCH --job-name=retokenize-sft
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

DATA_ROOT=${DATA_ROOT:-${SCRATCH_ROOT}/data/nemotron_sft_v2}
JSONL=${JSONL:-${DATA_ROOT}/nemotron_sft_v2_en.jsonl}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${DATA_ROOT}/nemotron_sft_v2_en_cl100kchat}
WORKERS=${WORKERS:-32}
export TIKTOKEN_CACHE_DIR=${SCRATCH_ROOT}/.cache/tiktoken
mkdir -p "${TIKTOKEN_CACHE_DIR}"

ls "${SCRATCH_ROOT}" > /dev/null
LAUNCH="${REPO_ROOT}/outputs/slurm/retokenize_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR}"
pip install --quiet --target="${SCRATCH_ROOT}/packages" tiktoken

# Pre-warm the tiktoken BPE cache before spawning workers to avoid contention.
python3 -c "import tiktoken; e = tiktoken.encoding_for_model('gpt-4'); print('cl100k merges loaded:', len(e._mergeable_ranks))"

python3 "${MEGATRON_ROOT}/tools/preprocess_data.py" \\
    --input "${JSONL}" \\
    --output-prefix "${OUTPUT_PREFIX}" \\
    --tokenizer-type Cl100kChat \\
    --json-keys text \\
    --workers ${WORKERS} \\
    --append-eod

echo "=== Done. Output:"
ls -lh "\$(dirname "${OUTPUT_PREFIX}")"/\$(basename "${OUTPUT_PREFIX}")*

# Verify chat tokens land as single ids in the .bin.
python3 - << 'PY'
import numpy as np
b = "${OUTPUT_PREFIX}_text_document.bin"
data = np.fromfile(b, dtype=np.uint32, count=2_000_000)
print("\\nFirst 60 ids:", np.fromfile(b, dtype=np.uint32, count=60).tolist())
for tid, name in [(100277,'<|im_start|>'),(100278,'<|im_end|>'),(100279,'<|think|>'),(100280,'<|/think|>'),(100281,'<|tool_call|>'),(100257,'<|endoftext|>')]:
    print(f"  {tid} ({name}): {(data==tid).sum():,}")
PY
LAUNCH
chmod +x "${LAUNCH}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH}"
