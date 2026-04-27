#!/usr/bin/bash
# Qwen3-8B recipe for Megatron-LM
# Sourced by megatron/train.sh — sets TP_SIZE, PP_SIZE, MODEL_ARGS, TRAIN_ARGS, DATA_ARGS.
#
# Default layout: TP=4 PP=1 DP=1  (1 node, 4 GPUs)
# For 8 GPUs:     NGPU=8 TP=4 PP=1 DP=2  bash train.sh
# For 2 nodes:    NGPU=16 TP=4 PP=2 DP=2  bash train.sh
#
# Qwen3-specific: QK-norm, GQA (32 heads / 8 KV), cos_sin RoPE, rope_theta=1M

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

SEQ_LEN=${SEQ_LEN:-8192}
GBS=${GBS:-128}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
TRAIN_STEPS=${TRAIN_STEPS:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-100}

MODEL_ARGS="
    --num-layers 36
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --ffn-hidden-size 22016
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 131072
    --vocab-size 151936
    --position-embedding-type rope
    --rotary-base 1000000
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
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
    --eval-iters 0
    --recompute-activations
    --recompute-granularity selective
"

DATA_ARGS="
    --mock-data
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_TOKENIZER_PATH:-assets/hf/Qwen3-8B}
"
