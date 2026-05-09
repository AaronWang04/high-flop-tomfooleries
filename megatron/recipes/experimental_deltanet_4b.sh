#!/usr/bin/bash
# Experimental all-DeltaNet 4B recipe for Megatron-LM
# Sourced by megatron/train.sh — sets TP_SIZE, PP_SIZE, MODEL_ARGS, TRAIN_ARGS, DATA_ARGS.
#
# Mirrors the TorchTitan experimental_deltanet_4b config: Qwen3.5-4B dimensions,
# all 32 layers are GatedDeltaNet (no softmax attention).
#
# Default layout: TP=4 PP=1 DP=1  (1 node, 4 GPUs)
# For 8 GPUs:     NGPU=8 TP=4 PP=1 DP=2  bash train.sh

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

SEQ_LEN=${SEQ_LEN:-4096}
GBS=${GBS:-128}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
# save_interval=2000 steps ≈ 1.05B tokens at GBS=128, SEQ_LEN=4096
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
# Larger than TRAIN_STEPS so no intermediate save is retained → keeps only the latest checkpoint
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-200000}

MODEL_ARGS="
    --num-layers 32
    --hidden-size 2560
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --ffn-hidden-size 9216
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 8192
    --vocab-size 248320
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
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq '([1]*32)'
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32
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
"

DATA_ARGS="
    --data-path ${DATA_PREFIX:-${SCRATCH_ROOT}/data/fineweb/sample/10BT/fineweb_qwen35_sample_10BT_text_document}
    --split 99,1,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_TOKENIZER_PATH:-${REPO_ROOT}/assets/hf/Qwen3.5-4B}
"
