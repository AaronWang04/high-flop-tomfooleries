#!/usr/bin/bash
# Experimental ~1.5B GQA recipe for Megatron-LM (companion to experimental_deltanet_4b)
# Sourced by megatron/slurm_multinode_dlcluster.sh — sets TP_SIZE, PP_SIZE, MODEL_ARGS, TRAIN_ARGS, DATA_ARGS.
#
# Differences from experimental_deltanet_4b:
#   - Standard softmax GQA (not all-DeltaNet)
#   - hidden=2048, ffn=4096, vocab=100277 (cl100k_base via HF tokenizer)
#   - Same stability flags: layernorm-1p, RMSNorm, qk-layernorm WD, output gate, RoPE w/ partial rotation

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

SEQ_LEN=${SEQ_LEN:-4096}
GBS=${GBS:-128}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-200000}

# Architecture: 32 layers × hidden 2048 × FFN 4096, 16 Q heads / 4 KV heads (GQA 4:1), head_dim 128
# Body params: 32 × ~36M = ~1.14B
# Embed + LM head: 2 × 100277 × 2048 = ~411M
# Total: ~1.55B

MODEL_ARGS="
    --num-layers 32
    --hidden-size 2048
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --ffn-hidden-size 4096
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 8192
    --vocab-size 100352
    --position-embedding-type rope
    --rotary-base 10000000
    --rotary-percent 0.25
    --no-rope-fusion
    --swiglu
    --normalization RMSNorm
    --apply-layernorm-1p
    --norm-epsilon 1e-6
    --attention-output-gate
    --apply-wd-to-qk-layernorm
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --disable-bias-linear
    --bf16
"

TRAIN_ARGS="
    --micro-batch-size 1
    --global-batch-size ${GBS}
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-decay-style cosine
    --lr-warmup-iters ${WARMUP_STEPS}
    --train-iters ${TRAIN_STEPS}
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --log-interval 10
    --log-throughput
    --eval-iters 0
    --eval-interval 1000
    --save-interval ${SAVE_INTERVAL}
    --save-retain-interval ${SAVE_RETAIN_INTERVAL}
    --override-opt-param-scheduler
    --async-strategy mcore
    --recompute-activations
    --recompute-granularity selective
    --cross-entropy-loss-fusion
    --overlap-grad-reduce
"

# DATA_PREFIX must point at a Megatron .bin/.idx pair tokenized with the cl100k_base tokenizer.
# The current FineWeb data is tokenized with Qwen3.5 — needs re-preprocessing for this recipe.
# Use HuggingFaceTokenizer with Xenova/gpt-4 (HF wrapper around cl100k_base).
DATA_ARGS="
    --data-path
      0.425 ${SCRATCH_ROOT}/data/fineweb/sample/10BT/fineweb_cl100k_sample_10BT_text_document
      0.425 ${SCRATCH_ROOT}/data/fineweb/sample/100BT_part1/fineweb_cl100k_sample_100BT_part1_text_document
      0.150 ${SCRATCH_ROOT}/data/starcoder/codeparrot-clean/starcoder_cl100k_text_document
    --split 100,0,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_TOKENIZER_PATH:-${REPO_ROOT}/assets/hf/cl100k_base}
"
