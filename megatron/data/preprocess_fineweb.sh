#!/usr/bin/bash
# Download FineWeb sample-10BT and preprocess to Megatron .bin/.idx format.
# CPU-bound; runs on a single node, takes ~1-2h end to end.
#
# Usage: sbatch megatron/data/preprocess_fineweb.sh
#
#SBATCH --job-name=preprocess-fineweb
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=outputs/data/preprocess-fineweb-%j.out
#SBATCH --error=outputs/data/preprocess-fineweb-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

DATASET_SUBSET=${DATASET_SUBSET:-sample/10BT}
DATA_ROOT=${DATA_ROOT:-${SCRATCH_ROOT}/data/fineweb/${DATASET_SUBSET}}
TOKENIZER_PATH=${TOKENIZER_PATH:-${REPO_ROOT}/assets/hf/Qwen3.5-4B}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${DATA_ROOT}/fineweb_qwen35_${DATASET_SUBSET//\//_}}

WORKERS=${WORKERS:-64}

mkdir -p "${REPO_ROOT}/outputs/data"
mkdir -p "${DATA_ROOT}/parquet"
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

ls "${SCRATCH_ROOT}" > /dev/null  # autofs trigger

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/data/preprocess_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Step 1: Download FineWeb ${DATASET_SUBSET} parquet ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HuggingFaceFW/fineweb',
    repo_type='dataset',
    allow_patterns=['${DATASET_SUBSET}/*.parquet'],
    local_dir='${DATA_ROOT}/parquet',
    max_workers=8,
)
print('Download complete')
"

echo "=== Step 2: Convert parquet -> JSONL ==="
python3 -c "
import pyarrow.parquet as pq
import json, glob, os
parquet_files = sorted(glob.glob('${DATA_ROOT}/parquet/${DATASET_SUBSET}/*.parquet'))
print(f'Found {len(parquet_files)} parquet files')
total_rows = 0
out_path = '${DATA_ROOT}/fineweb.jsonl'
with open(out_path, 'w') as out:
    for i, pf in enumerate(parquet_files):
        pq_file = pq.ParquetFile(pf)
        for batch in pq_file.iter_batches(batch_size=10000, columns=['text']):
            for text in batch.column('text').to_pylist():
                out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                total_rows += 1
        print(f'  [{i+1}/{len(parquet_files)}] {os.path.basename(pf)} done; total rows={total_rows}')
print(f'JSONL written: {out_path}, {total_rows} documents')
"

echo "=== Step 3: Tokenize with Megatron preprocess_data.py ==="
python3 "${MEGATRON_ROOT}/tools/preprocess_data.py" \\
    --input "${DATA_ROOT}/fineweb.jsonl" \\
    --output-prefix "${OUTPUT_PREFIX}" \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-model "${TOKENIZER_PATH}" \\
    --json-keys text \\
    --workers ${WORKERS} \\
    --append-eod

echo "=== Done. Output:"
ls -lh "$(dirname "${OUTPUT_PREFIX}")"/$(basename "${OUTPUT_PREFIX}")*
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Subset:        ${DATASET_SUBSET}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Workers:       ${WORKERS}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
