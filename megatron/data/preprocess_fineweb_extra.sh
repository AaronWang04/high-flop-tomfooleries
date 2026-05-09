#!/usr/bin/bash
# Download a slice of FineWeb sample-100BT and preprocess to Megatron .bin/.idx (cl100k).
# Targets +10B tokens on top of the existing sample-10BT preprocessing.
# sample-100BT and sample-10BT are independent random samples from FineWeb's pool, so overlap
# is statistical — fine for our purposes.
#
# Usage: sbatch megatron/data/preprocess_fineweb_extra.sh
#
#SBATCH --job-name=preprocess-fineweb-extra
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=outputs/data/preprocess-fineweb-extra-%j.out
#SBATCH --error=outputs/data/preprocess-fineweb-extra-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

# Use sample/100BT but only grab the first N parquet shards (~700M tokens each in sample-10BT,
# so 15 shards ≈ 10B tokens. sample-100BT files are similar size.)
DATASET_SUBSET=${DATASET_SUBSET:-sample/100BT}
MAX_FILES=${MAX_FILES:-15}
DATA_ROOT=${DATA_ROOT:-${SCRATCH_ROOT}/data/fineweb/${DATASET_SUBSET}_part1}
TOKENIZER_PATH=${TOKENIZER_PATH:-${REPO_ROOT}/assets/hf/cl100k_base}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${DATA_ROOT}/fineweb_cl100k_${DATASET_SUBSET//\//_}_part1}
WORKERS=${WORKERS:-64}

mkdir -p "${REPO_ROOT}/outputs/data"
mkdir -p "${DATA_ROOT}/parquet"
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/data/preprocess_fineweb_extra_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Step 1: List + download first ${MAX_FILES} parquet files of ${DATASET_SUBSET} ==="
python3 -c "
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
all_files = api.list_repo_files('HuggingFaceFW/fineweb', repo_type='dataset')
matched = sorted([f for f in all_files if f.startswith('${DATASET_SUBSET}/') and f.endswith('.parquet')])
print(f'Total parquet in {repr(\"${DATASET_SUBSET}\")}: {len(matched)}')
to_get = matched[:${MAX_FILES}]
print(f'Downloading {len(to_get)} files...')
for i, fn in enumerate(to_get):
    hf_hub_download(
        repo_id='HuggingFaceFW/fineweb',
        repo_type='dataset',
        filename=fn,
        local_dir='${DATA_ROOT}/parquet',
    )
    print(f'  [{i+1}/{len(to_get)}] {fn}')
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
ls -lh "\$(dirname "${OUTPUT_PREFIX}")"/\$(basename "${OUTPUT_PREFIX}")*
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Subset:        ${DATASET_SUBSET} (first ${MAX_FILES} files)"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Workers:       ${WORKERS}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
