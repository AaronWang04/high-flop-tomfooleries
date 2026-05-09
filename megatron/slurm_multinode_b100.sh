#!/usr/bin/bash
# Multinode Slurm launcher for Megatron-LM recipes on DLCluster b100_preprod (B100 x86_64).
# All B100 nodes are SC/ipp4 — same NFS region as our scratch.
# Uses /home/scratch.aarowang_ent/packages_x86 (built for x86_64) and the x86_64 helpers_cpp.so
# alongside the aarch64 one (Python's import picks the matching arch).
#
# Usage:
#   sbatch megatron/slurm_multinode_b100.sh
#   RECIPE=experimental_gqa_1_5b TRAIN_STEPS=80000 GBS=128 sbatch --nodes=4 megatron/slurm_multinode_b100.sh
#
#SBATCH --job-name=gqa-1_5b-b100
#SBATCH --partition=b100_preprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=outputs/megatron/experimental_gqa_1_5b/logs/slurm-%j.out
#SBATCH --error=outputs/megatron/experimental_gqa_1_5b/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_gqa_1_5b}
NGPU_PER_NODE=4
NNODES=${SLURM_NNODES}
LOG_RANK=${LOG_RANK:-0,3}

WANDB_PROJECT=${WANDB_PROJECT:-hft-pretrain}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-${RECIPE}-${SLURM_JOB_ID}}

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/logs"
mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/checkpoints"
mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/tensorboard"

# Trigger autofs on every allocated node before enroot tries to bind-mount.
srun --ntasks="${NNODES}" --ntasks-per-node=1 ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/launch_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail

unset CC CXX AR LD
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# B100 nodes have intra-node NVLink (4 GPUs) but no MNNVL across nodes — cross-node always IB.
# So no MNNVL/IMEX issues like on GB200.
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages_x86:\${PYTHONPATH:-}"

if [ "\${WANDB_MODE:-}" = "offline" ]; then
    :
elif [ -f "${SCRATCH_ROOT}/.wandb_api_key" ]; then
    export WANDB_API_KEY=\$(cat "${SCRATCH_ROOT}/.wandb_api_key")
else
    export WANDB_MODE=offline
fi
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
mkdir -p "\${WANDB_DIR}"

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
    --load "${REPO_ROOT}/outputs/megatron/${RECIPE}/checkpoints" \\
    --tensorboard-dir "${REPO_ROOT}/outputs/megatron/${RECIPE}/tensorboard" \\
    --wandb-project "${WANDB_PROJECT}" \\
    --wandb-exp-name "${WANDB_EXP_NAME}" \\
    --wandb-save-dir "${SCRATCH_ROOT}/wandb"
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Nodes:         ${NNODES}"
echo "Nodelist:      ${SLURM_JOB_NODELIST}"
echo "GPUs/node:     ${NGPU_PER_NODE}"
echo "Total GPUs:    $((NNODES * NGPU_PER_NODE))"
echo "TP/PP/DP:      ${TP_SIZE} / ${PP_SIZE} / $((NNODES * NGPU_PER_NODE / TP_SIZE / PP_SIZE))"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Recipe:        ${RECIPE}"
echo "Container:     ${CONTAINER_IMAGE}"
echo "Wandb:         ${WANDB_PROJECT}/${WANDB_EXP_NAME}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
