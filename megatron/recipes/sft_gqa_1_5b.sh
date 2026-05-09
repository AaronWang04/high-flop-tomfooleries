#!/usr/bin/bash
# SFT recipe for the experimental_gqa_1_5b base model on Nemotron-Post-Training-Dataset-v2
# (English only). Uses extended cl100k tokenizer with chat/think/tool tokens.
# Sourced by megatron/slurm_multinode_dlcluster.sh.
#
# Architecture matches pretrain (must match iter_0016000 checkpoint exactly).
# Hyperparameters are SFT-style: low LR, small batch, completion-style loss
# (full-sequence loss including system/user — proper masked-loss is a follow-up).

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

SEQ_LEN=${SEQ_LEN:-4096}
GBS=${GBS:-64}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-2e-6}
TRAIN_STEPS=${TRAIN_STEPS:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-200000}

# Architecture: identical to pretrain experimental_gqa_1_5b — must match the
# iter_0016000 checkpoint shape (vocab 100352 covers the 16 added chat tokens).
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

# --finetune: load model weights only, reset iter / optim / RNG → fresh SFT from step 1.
TRAIN_ARGS="
    --finetune
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
    --log-interval 5
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

DATA_ARGS="
    --data-path ${SCRATCH_ROOT}/data/nemotron_sft_v2/nemotron_sft_v2_en_cl100kchat_text_document
    --split 100,0,0
    --tokenizer-type Cl100kChat
"
