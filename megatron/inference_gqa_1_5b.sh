#!/usr/bin/bash
# Single-GPU coherence inference for the experimental_gqa_1_5b checkpoint.
# torch_dist checkpoints reshard automatically, so TP4 → TP1 works for a sanity check.
#
# Usage: sbatch megatron/inference_gqa_1_5b.sh
#
#SBATCH --job-name=gqa-infer
#SBATCH --partition=gb200nvl72
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

CHECKPOINT_DIR="${REPO_ROOT}/outputs/megatron/experimental_gqa_1_5b/checkpoints"
TOKENIZER_PATH="${REPO_ROOT}/assets/hf/cl100k_base"

srun --ntasks=1 --ntasks-per-node=1 ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/slurm/inference_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

torchrun --nproc_per_node 1 \\
    "${MEGATRON_ROOT}/examples/inference/gpt/gpt_static_inference.py" \\
    --use-mcore-models \\
    --load "${CHECKPOINT_DIR}" \\
    --exit-on-missing-checkpoint \\
    --no-load-rng \\
    --no-load-optim \\
    --tensor-model-parallel-size 1 \\
    --pipeline-model-parallel-size 1 \\
    --num-layers 32 \\
    --hidden-size 2048 \\
    --num-attention-heads 16 \\
    --group-query-attention \\
    --num-query-groups 4 \\
    --kv-channels 128 \\
    --ffn-hidden-size 4096 \\
    --seq-length 4096 \\
    --max-position-embeddings 8192 \\
    --vocab-size 100352 \\
    --position-embedding-type rope \\
    --rotary-base 10000000 \\
    --rotary-percent 0.25 \\
    --no-rope-fusion \\
    --swiglu \\
    --normalization RMSNorm \\
    --apply-layernorm-1p \\
    --norm-epsilon 1e-6 \\
    --attention-output-gate \\
    --untie-embeddings-and-output-weights \\
    --no-position-embedding \\
    --disable-bias-linear \\
    --bf16 \\
    --micro-batch-size 1 \\
    --tokenizer-type HuggingFaceTokenizer \\
    --tokenizer-model "${TOKENIZER_PATH}" \\
    --inference-max-requests 4 \\
    --inference-max-seq-length 512 \\
    --num-tokens-to-generate 120 \\
    --temperature 0.8 \\
    --top_k 40 \\
    --prompts \\
        "The capital of France is" \\
        "Once upon a time, there was a small village" \\
        "def fibonacci(n):" \\
        "The three laws of motion are: 1."
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "Checkpoint:    ${CHECKPOINT_DIR}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "Container:     ${CONTAINER_IMAGE}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
