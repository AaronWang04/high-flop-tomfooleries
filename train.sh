#!/usr/bin/bash
# Pretraining on 4x GB200
# Uses our custom hft module with torchtitan
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NGPU=${NGPU:-4}
CONFIG=${CONFIG:-"hft_qwen3_8b"}
export LOG_RANK=${LOG_RANK:-0}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --module hft --config ${CONFIG} \
    --dump_folder "${SCRIPT_DIR}/outputs" \
    "$@"
