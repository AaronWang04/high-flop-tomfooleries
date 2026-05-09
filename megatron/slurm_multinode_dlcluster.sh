#!/usr/bin/bash
# Multinode Slurm launcher for Megatron-LM recipes on DLCluster gb200nvl72_ci.
# Constrains jobs to Santa Clara nodes so /home/scratch.aarowang_ent NFS is reachable.
# Hillsboro nodes (gb-nvl-147-*, gb200-nvl4-ts2-*) are excluded — scratch is not
# exported across regions; see dlcluster.md.
#
# Usage:
#   sbatch megatron/slurm_multinode_dlcluster.sh                           # 8 nodes, default recipe
#   RECIPE=qwen35_4b sbatch --nodes=4 megatron/slurm_multinode_dlcluster.sh
#
# Pre-trigger autofs on each node before Pyxis runs (otherwise enroot can't
# bind-mount /home/scratch.aarowang_ent into the container).
#
#SBATCH --job-name=deltanet-4b
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_deltanet_4b}
NGPU_PER_NODE=4
NNODES=${SLURM_NNODES}
# 0 captures rank 0 (memory/save logs); 3 captures last local rank, which on the last node
# is print_rank_last - that's where Megatron prints the per-iteration "lm loss" line.
LOG_RANK=${LOG_RANK:-0,3}

WANDB_PROJECT=${WANDB_PROJECT:-hft-pretrain}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-${RECIPE}-${SLURM_JOB_ID}}

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

# Trigger autofs on every allocated node before enroot tries to bind-mount.
# Each node accesses ${SCRATCH_ROOT} once, which causes autofs to mount it.
srun --ntasks="${NNODES}" --ntasks-per-node=1 ls "${SCRATCH_ROOT}" > /dev/null

# Write a per-node launch script so MODEL_ARGS/TRAIN_ARGS/DATA_ARGS expand
# cleanly without shell quoting issues across the srun boundary.
LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/launch_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail

unset CC CXX AR LD  # prevent login-node cross-compiler vars leaking into aarch64 container
export PYTHONUNBUFFERED=1  # flush stdout per-line so .out log file stays current with wandb
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Force IB-only traffic to avoid cross-chassis NVLink P2P which fails when 8+ nodes
# get allocated across chassis (no IMEX channels in container).
# (NCCL_P2P_DISABLE was tried for cross-chassis GBS=128, but rendezvous failed; reverted)
# (NFS-backed Triton cache caused concurrent-write corruption; using default per-node /tmp cache.)
# NCCL_MNNVL_ENABLE: leave at default (auto-detect). At ≤4-node clique sizes MNNVL works
# fine; at the 8-node (32-GPU clique) size we previously hit "Cuda failure 800" — but
# disabling it appears to cause a different multinode hang. Revisit if scaling back to 8.
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

# wandb: read API key from NFS file. If missing, run offline so training still proceeds.
# Caller can force offline mode by setting WANDB_MODE=offline before sbatch.
if [ "\${WANDB_MODE:-}" = "offline" ]; then
    : # respect caller's offline setting; don't load API key
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
