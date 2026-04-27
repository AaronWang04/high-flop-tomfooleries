#!/usr/bin/bash
# Pretraining with TorchTitan
# Default: 4x GB200, Qwen3-8B
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NGPU=${NGPU:-4}
CONFIG=${CONFIG:-"qwen3_8b"}
export LOG_RANK=${LOG_RANK:-0}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --module models --config ${CONFIG} \
    --dump_folder "${SCRIPT_DIR}/../outputs/torchtitan" \
    "$@"
