#!/usr/bin/bash
# Pretraining with Megatron-LM
# Sources a recipe from megatron/recipes/ to set all model/training args.
# Usage:  RECIPE=llama3_8b bash train.sh
#         RECIPE=qwen3_8b NGPU=8 bash train.sh
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_ROOT="$(cd "${SCRIPT_DIR}/../../Megatron-LM" && pwd)"

NGPU=${NGPU:-4}
RECIPE=${RECIPE:-"llama3_8b"}
export LOG_RANK=${LOG_RANK:-0}

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    "${MEGATRON_ROOT}/pretrain_gpt.py" \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    ${MODEL_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    --save "${SCRIPT_DIR}/../outputs/megatron/${RECIPE}/checkpoints" \
    --tensorboard-dir "${SCRIPT_DIR}/../outputs/megatron/${RECIPE}/tensorboard" \
    "$@"
