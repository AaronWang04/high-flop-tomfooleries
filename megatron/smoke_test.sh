#!/usr/bin/bash
# Smoke test: 1 node, 4 GPUs, 20 steps, short seq_len.
# Usage: sbatch megatron/smoke_test.sh
#        RECIPE=qwen3_8b sbatch megatron/smoke_test.sh
#
#SBATCH --job-name=smoke-deltanet-4b
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=outputs/megatron/experimental_deltanet_4b/logs/smoke-%j.out
#SBATCH --error=outputs/megatron/experimental_deltanet_4b/logs/smoke-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_deltanet_4b}
NGPU_PER_NODE=4

# Override training params for a quick smoke test
export TP_SIZE=4
export PP_SIZE=1
export SEQ_LEN=512
export GBS=4
export TRAIN_STEPS=20
export WARMUP_STEPS=2

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

# Override tokenizer for smoke test — NullTokenizer needs no files
DATA_ARGS="
    --mock-data
    --tokenizer-type NullTokenizer
"

mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/logs"

# Write a launch script so MODEL_ARGS/TRAIN_ARGS/DATA_ARGS single-quote values
# expand correctly as shell source when bash executes the file (same as slurm_multinode.sh).
LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/smoke_launch_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail

unset CC CXX AR LD  # prevent login-node cross-compiler vars leaking into aarch64 container
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

torchrun \\
    --nproc_per_node ${NGPU_PER_NODE} \\
    --rdzv_backend c10d \\
    --rdzv_endpoint "localhost:0" \\
    --local-ranks-filter 0 --role rank --tee 3 \\
    "${MEGATRON_ROOT}/pretrain_gpt.py" \\
    --use-mcore-models \\
    --tensor-model-parallel-size ${TP_SIZE} \\
    --pipeline-model-parallel-size ${PP_SIZE} \\
    --sequence-parallel \\
    $(printf '%s' "${MODEL_ARGS}" | tr '\n' ' ') \\
    $(printf '%s' "${TRAIN_ARGS}" | tr '\n' ' ') \\
    $(printf '%s' "${DATA_ARGS}" | tr '\n' ' ') \\
    --no-save-optim \\
    --no-save-rng
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

# Trigger autofs mount on this node before enroot tries to bind-mount it
ls "${SCRATCH_ROOT}" > /dev/null

echo "=== Smoke test: ${RECIPE} ==="
echo "Container: ${CONTAINER_IMAGE}"
echo "Node:     $(hostname)"
echo "GPUs:     ${NGPU_PER_NODE}"
echo "TP/PP:    ${TP_SIZE} / ${PP_SIZE}"
echo "Steps:    ${TRAIN_STEPS}  seq_len: ${SEQ_LEN}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"

echo "=== Smoke test PASSED ==="
