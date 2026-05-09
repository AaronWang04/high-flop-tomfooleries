#!/usr/bin/bash
# Pre-warm the Pyxis/enroot container cache on N SC nodes.
# Each node runs a no-op inside the container; this triggers the per-node download
# and leaves the squashfs in /tmp/enroot-data, so subsequent jobs landing on that
# node skip the 1-2 min import step.
#
# Usage:
#   sbatch megatron/data/prewarm_containers.sh
#   NNODES=8 sbatch megatron/data/prewarm_containers.sh
#
#SBATCH --job-name=prewarm-containers
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=outputs/data/prewarm-%j.out
#SBATCH --error=outputs/data/prewarm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

mkdir -p "${REPO_ROOT}/outputs/data"

echo "Pre-warming container ${CONTAINER_IMAGE} on ${SLURM_NNODES} nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo "---"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
     bash -c 'echo "  cached on $(hostname)"'

echo "---"
echo "Done. Subsequent jobs on these nodes skip container import."
