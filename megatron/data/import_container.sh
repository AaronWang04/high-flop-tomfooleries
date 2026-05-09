#!/usr/bin/bash
# One-time: import the PyTorch container into a squashfs file on scratch NFS.
# Subsequent jobs use --container-image=<sqsh path> instead of nvcr.io#... and
# skip the per-node download/import step entirely.
#
# Run on a compute node so enroot picks the aarch64 variant automatically.
#
# Usage: sbatch megatron/data/import_container.sh
#
#SBATCH --job-name=import-container
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --exclude=gb-nvl-115-compute[01-18]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=outputs/data/import-container-%j.out
#SBATCH --error=outputs/data/import-container-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_REF=${CONTAINER_REF:-nvcr.io#nvidia/pytorch:26.02-py3}
SQSH_DIR=${SQSH_DIR:-${SCRATCH_ROOT}/containers}
# Sanitize ref for filename: nvcr.io#nvidia/pytorch:26.02-py3 -> nvcr.io+nvidia+pytorch+26.02-py3.sqsh
SQSH_NAME=$(echo "${CONTAINER_REF}" | tr '/:#' '+').sqsh
SQSH_PATH="${SQSH_DIR}/${SQSH_NAME}"

mkdir -p "${REPO_ROOT}/outputs/data" "${SQSH_DIR}"

if [ -f "${SQSH_PATH}" ]; then
    echo "Already exists: ${SQSH_PATH} ($(du -h "${SQSH_PATH}" | cut -f1))"
    echo "Delete it first if you want to re-import."
    exit 0
fi

echo "Importing ${CONTAINER_REF} -> ${SQSH_PATH}"
echo "Host: $(hostname)  Arch: $(uname -m)"
echo "---"

cd "${SQSH_DIR}"
enroot import --output "${SQSH_PATH}" "docker://${CONTAINER_REF}"

echo "---"
echo "Done: $(ls -lh "${SQSH_PATH}")"
echo ""
echo "Use it in launch scripts via:"
echo "  srun --container-image=${SQSH_PATH} ..."
