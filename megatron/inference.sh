#!/usr/bin/bash
# Run static inference on the experimental_deltanet_4b checkpoint.
# 1 node, 4 GPUs (TP=4 PP=1) — same model parallel topology as training.
# Usage: sbatch megatron/inference.sh
#        PROMPT="Once upon a time" sbatch megatron/inference.sh
#
#SBATCH --job-name=infer-deltanet-4b
#SBATCH --partition=gb200nvl72_ci
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=outputs/megatron/experimental_deltanet_4b/logs/infer-%j.out
#SBATCH --error=outputs/megatron/experimental_deltanet_4b/logs/infer-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

RECIPE=${RECIPE:-experimental_deltanet_4b}
NGPU_PER_NODE=4

# Inference configuration
LOAD_DIR=${LOAD_DIR:-${REPO_ROOT}/outputs/megatron/experimental_deltanet_4b/checkpoints_archive/inference_iter_10000}
PROMPT=${PROMPT:-"The quick brown fox jumps over"}
NUM_TOKENS=${NUM_TOKENS:-100}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-50}

# Override training params for inference
export TP_SIZE=4
export PP_SIZE=1
export SEQ_LEN=4096
export GBS=1
export TRAIN_STEPS=10  # not used for inference but required by recipe
export WARMUP_STEPS=1

source "${SCRIPT_DIR}/recipes/${RECIPE}.sh"

mkdir -p "${REPO_ROOT}/outputs/megatron/${RECIPE}/logs"

ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/megatron/${RECIPE}/infer_launch_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail

unset CC CXX AR LD
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

torchrun \\
    --nproc_per_node ${NGPU_PER_NODE} \\
    --rdzv_backend c10d \\
    --rdzv_endpoint "localhost:0" \\
    --local-ranks-filter 0 --role rank --tee 3 \\
    "${MEGATRON_ROOT}/examples/inference/gpt/gpt_static_inference.py" \\
    --use-mcore-models \\
    --tensor-model-parallel-size ${TP_SIZE} \\
    --pipeline-model-parallel-size ${PP_SIZE} \\
    $(printf '%s' "${MODEL_ARGS}" | tr '\n' ' ') \\
    --micro-batch-size 1 \\
    --load "${LOAD_DIR}" \\
    --no-load-optim \\
    --no-load-rng \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-model "${REPO_ROOT}/assets/hf/Qwen3.5-4B" \\
    --prompts "${PROMPT}" \\
    --num-tokens-to-generate ${NUM_TOKENS} \\
    --temperature ${TEMPERATURE} \\
    --top_k ${TOP_K} \\
    --inference-max-requests 1 \\
    --inference-max-seq-length 1024
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "=== Inference: ${RECIPE} ==="
echo "Container:     ${CONTAINER_IMAGE}"
echo "Checkpoint:    ${LOAD_DIR}"
echo "Prompt:        ${PROMPT}"
echo "Num tokens:    ${NUM_TOKENS}"
echo "Temperature:   ${TEMPERATURE}"
echo "Top-k:         ${TOP_K}"
echo "TP/PP:         ${TP_SIZE}/${PP_SIZE}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
