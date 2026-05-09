#!/usr/bin/bash
# Multinode smoke test on DLCluster gb200nvl72_ci (SC nodes only).
# 4 nodes × 4 GPUs = 16 GPUs, 20 steps, short seq_len, no checkpointing.
# Usage: sbatch megatron/smoke_test_multinode.sh
#
#SBATCH --job-name=smoke-deltanet-4b-multi
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=outputs/megatron/experimental_deltanet_4b/logs/smoke-multi-%j.out
#SBATCH --error=outputs/megatron/experimental_deltanet_4b/logs/smoke-multi-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_deltanet_4b}
NGPU_PER_NODE=4
NNODES=${SLURM_NNODES}
LOG_RANK=${LOG_RANK:-0}

# Smoke test overrides
export TP_SIZE=4
export PP_SIZE=1
export SEQ_LEN=512
export GBS=16    # NNODES * NGPU_PER_NODE / TP_SIZE = 4*4/4 = 4 DP, GBS must be multiple of micro_batch_size * DP
export TRAIN_STEPS=20
export WARMUP_STEPS=2

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

# Override tokenizer for smoke test — NullTokenizer needs no files
DATA_ARGS="
    --mock-data
    --tokenizer-type NullTokenizer
"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/logs"

# Trigger autofs on every allocated node before enroot tries to bind-mount.
srun --ntasks="${NNODES}" --ntasks-per-node=1 ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/smoke_multi_launch_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail

unset CC CXX AR LD
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

torchrun \\
    --nproc_per_node ${NGPU_PER_NODE} \\
    --nnodes ${NNODES} \\
    --node_rank \${SLURM_PROCID} \\
    --rdzv_backend c10d \\
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \\
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \\
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

echo "=== Multinode smoke test: ${RECIPE} ==="
echo "Container:     ${CONTAINER_IMAGE}"
echo "Nodes:         ${NNODES}"
echo "Nodelist:      ${SLURM_JOB_NODELIST}"
echo "Total GPUs:    $((NNODES * NGPU_PER_NODE))"
echo "TP/PP/DP:      ${TP_SIZE} / ${PP_SIZE} / $((NNODES * NGPU_PER_NODE / TP_SIZE / PP_SIZE))"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Steps:         ${TRAIN_STEPS}  seq_len: ${SEQ_LEN}  GBS: ${GBS}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"

echo "=== Multinode smoke test PASSED ==="
