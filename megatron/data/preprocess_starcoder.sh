#!/usr/bin/bash
# Download bigcode/the-stack-smol (StarCoder-style code dataset) and preprocess
# to Megatron .bin/.idx using cl100k tokenizer. Targets ~2B tokens (~20% of fineweb sample-10BT).
#
# Usage: sbatch megatron/data/preprocess_starcoder.sh
#
#SBATCH --job-name=preprocess-starcoder
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/data/preprocess-starcoder-%j.out
#SBATCH --error=outputs/data/preprocess-starcoder-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

# bigcode/the-stack-smol is a 1B-token sample of The Stack.
# bigcode/the-stack-smol-xl is ~10B tokens (we'll grab a fraction).
# Default to "the-stack-smol" with all parquet files (~1B tokens after cl100k tokenization).
# codeparrot/codeparrot-clean is Python code (~50GB JSON.gz), public, no auth.
# 56 files of JSONL.gz, each record has 'content' field with code.
HF_REPO=${HF_REPO:-codeparrot/codeparrot-clean}
HF_PATTERN=${HF_PATTERN:-*.json.gz}
DATA_ROOT=${DATA_ROOT:-${SCRATCH_ROOT}/data/starcoder/codeparrot-clean}
TOKENIZER_PATH=${TOKENIZER_PATH:-${REPO_ROOT}/assets/hf/cl100k_base}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${DATA_ROOT}/starcoder_cl100k}
MAX_DOCS=${MAX_DOCS:-2000000}

TEXT_FIELD=${TEXT_FIELD:-content}
WORKERS=${WORKERS:-64}

mkdir -p "${REPO_ROOT}/outputs/data"
mkdir -p "${DATA_ROOT}/parquet"
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/data/preprocess_starcoder_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Step 1: Download ${HF_REPO} (${HF_PATTERN}) ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    repo_type='dataset',
    allow_patterns='${HF_PATTERN}',
    local_dir='${DATA_ROOT}/parquet',
    max_workers=8,
)
print('Download complete')
import os
total = 0
for root, _, files in os.walk('${DATA_ROOT}/parquet'):
    total += sum(1 for f in files if f.endswith('.parquet'))
print(f'parquet files on disk: {total}')
"

echo "=== Step 2: Convert JSONL.gz -> JSONL (cap at ${MAX_DOCS} docs) ==="
python3 -c "
import json, glob, os, gzip
json_files = sorted(glob.glob('${DATA_ROOT}/parquet/**/*.json.gz', recursive=True))
print(f'Found {len(json_files)} JSONL.gz files')
max_docs = ${MAX_DOCS}
total_rows = 0
out_path = '${DATA_ROOT}/starcoder.jsonl'
done = False
with open(out_path, 'w') as out:
    for i, jf in enumerate(json_files):
        if done:
            break
        try:
            with gzip.open(jf, 'rt') as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec.get('${TEXT_FIELD}')
                    if text is None:
                        continue
                    out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    total_rows += 1
                    if total_rows >= max_docs:
                        done = True
                        break
        except Exception as e:
            print(f'  skip {jf}: {e}')
            continue
        print(f'  [{i+1}/{len(json_files)}] {os.path.basename(jf)}: total rows={total_rows}')
print(f'JSONL written: {out_path}, {total_rows} documents')
"

echo "=== Step 3: Tokenize with Megatron preprocess_data.py ==="
python3 "${MEGATRON_ROOT}/tools/preprocess_data.py" \\
    --input "${DATA_ROOT}/starcoder.jsonl" \\
    --output-prefix "${OUTPUT_PREFIX}" \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-model "${TOKENIZER_PATH}" \\
    --json-keys text \\
    --workers ${WORKERS} \\
    --append-eod

echo "=== Done. Output:"
ls -lh "\$(dirname "${OUTPUT_PREFIX}")"/\$(basename "${OUTPUT_PREFIX}")*
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Repo:          ${HF_REPO}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Workers:       ${WORKERS}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
