# High-Flop-Tomfooleries (HFT)

Pretraining experiments using [torchtitan](https://github.com/pytorch/torchtitan). Not to be confused with [High-Frequency-Trading](https://en.wikipedia.org/wiki/High-frequency_trading)

## Setup

```bash
pip install -r requirements.txt
```

Download tokenizers (requires `HF_TOKEN` or login with `huggingface-cli login`):

```bash
# Qwen3 (for Qwen3.5-9B)
huggingface-cli download Qwen/Qwen3-8B tokenizer.json tokenizer_config.json merges.txt vocab.json --local-dir assets/hf/Qwen3-8B

# Llama 3.1 (for Llama 8B)
huggingface-cli download meta-llama/Llama-3.1-8B tokenizer.json tokenizer_config.json special_tokens_map.json original/tokenizer.model --local-dir assets/hf/Llama-3.1-8B
```

## Training

```bash
# Qwen3.5-9B (default)
bash train.sh

# Llama 8B
CONFIG=hft_8b bash train.sh
```

Config params
```
--training.steps 5000              # train longer
--training.local-batch-size 2      # bigger batch per GPU
--training.seq-len 4096            # shorter sequences
--optimizer.lr 1e-4                # lower learning rate
--activation-checkpoint.mode full  # checkpoint all layers (save memory)
--compile.enable                   # enable torch.compile
--metrics.log-freq 5               # log every 5 steps
--checkpoint.interval 200          # save every 200 steps
```

