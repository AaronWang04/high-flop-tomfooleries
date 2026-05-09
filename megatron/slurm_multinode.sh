#!/usr/bin/bash
# Multinode Slurm launcher for Megatron-LM recipes.
# Usage:
#   sbatch megatron/slurm_multinode.sh
#   RECIPE=qwen35_4b sbatch --nodes=4 megatron/slurm_multinode.sh
#
# Required env: SCRATCH_ROOT must point to a directory visible on every allocated
# compute node (mounted in the container via --container-mounts). The user's
# checkout, output, and packages dir all live under this path.
#
#SBATCH --job-name=deltanet-4b
#SBATCH --partition=
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=outputs/megatron/experimental_deltanet_4b/logs/slurm-%j.out
#SBATCH --error=outputs/megatron/experimental_deltanet_4b/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=${SCRATCH_ROOT:?SCRATCH_ROOT must be set to a path visible on all nodes}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_deltanet_4b}
NGPU_PER_NODE=4
NNODES=${SLURM_NNODES}
LOG_RANK=${LOG_RANK:-0}

# TP=4 keeps all tensor-parallel comms intra-node over NVLink.
# DP = NNODES * 4 / (TP * PP) handles gradient sync across nodes over IB.
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/logs"
mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/checkpoints"
mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/tensorboard"

# Write a per-node launch script so MODEL_ARGS/TRAIN_ARGS/DATA_ARGS expand
# cleanly without shell quoting issues across the srun boundary.
LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/launch_${SLURM_JOB_ID}.sh"
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
    --save "${REPO_ROOT}/outputs/megatron/${RECIPE}/checkpoints" \\
    --tensorboard-dir "${REPO_ROOT}/outputs/megatron/${RECIPE}/tensorboard"
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Nodes:         ${NNODES}"
echo "GPUs/node:     ${NGPU_PER_NODE}"
echo "Total GPUs:    $((NNODES * NGPU_PER_NODE))"
echo "TP/PP/DP:      ${TP_SIZE} / ${PP_SIZE} / $((NNODES * NGPU_PER_NODE / TP_SIZE / PP_SIZE))"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Recipe:        ${RECIPE}"
echo "Container:     ${CONTAINER_IMAGE}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
