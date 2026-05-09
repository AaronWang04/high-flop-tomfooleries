#!/usr/bin/bash
# Upload Megatron checkpoints in checkpoints_keep/ to HuggingFace as
# Norapom/experimental_gqa_1_5b (private). Each iter goes to its own branch;
# the latest (iter_0016000) is also on main.
#
# Layout per branch:
#   iter_NNNNNNN/    (.distcp shards)
#   latest_checkpointed_iteration.txt
#   assets/hf/cl100k_base/   (tokenizer)
#   README.md
#
# Usage: sbatch megatron/data/upload_checkpoints_hf.sh
#
#SBATCH --job-name=hf-upload-ckpts
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

HF_REPO_ID=${HF_REPO_ID:-Norapom/experimental_gqa_1_5b}
KEEP_DIR=${KEEP_DIR:-${REPO_ROOT}/outputs/megatron/experimental_gqa_1_5b/checkpoints_keep}
TOKENIZER_DIR=${TOKENIZER_DIR:-${REPO_ROOT}/assets/hf/cl100k_base}

ls "${SCRATCH_ROOT}" > /dev/null

LAUNCH_SCRIPT="${REPO_ROOT}/outputs/slurm/upload_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH_SCRIPT}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=\$(cat ${SCRATCH_ROOT}/.cache/huggingface/token)

python3 << 'PY'
import os, sys, json
from huggingface_hub import HfApi, create_repo, create_branch, upload_folder, upload_file

REPO_ID  = "${HF_REPO_ID}"
KEEP_DIR = "${KEEP_DIR}"
TOK_DIR  = "${TOKENIZER_DIR}"

# Iter -> branch mapping. main mirrors the latest.
ITERS  = [10000, 12000, 13000, 16000]
LATEST = 16000

api = HfApi()

print(f"Creating private repo {REPO_ID} (exist_ok=True)...")
create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)

# README content (per-branch, parameterized by iter).
def make_readme(iter_num):
    tokens_pre = min(iter_num, 12000) * 128 * 4096
    tokens_post = max(iter_num - 12000, 0) * 512 * 4096
    total_tokens = (tokens_pre + tokens_post) / 1e9
    return f"""---
license: other
language: en
library_name: megatron-lm
tags:
- pretrained
- gqa
- megatron
- experimental
---

# experimental_gqa_1_5b — iter {iter_num:,}

Megatron-LM `torch_dist` checkpoint, trained from scratch on FineWeb sample-10BT/100BT_part1
(text) + codeparrot-clean (code), tokenized with cl100k_base.

**This branch:** iter {iter_num:,} (~{total_tokens:.2f}B tokens trained).
Other revisions: branches `iter_0010000`, `iter_0012000`, `iter_0013000`, `iter_0016000`.
The `main` branch tracks iter {LATEST:,}.

## Architecture
- Layers: 32, hidden 2048, FFN 4096, GQA 16Q/4KV, head_dim 128
- Vocab 100,352 (cl100k_base, padded for TP=4)
- RoPE base 1e7, partial 0.25; SwiGLU; RMSNorm + 1p
- Attention output gate; QK-LayerNorm with WD; untied embeddings
- Pretrain: TP=4 PP=1; bf16

## Loading
This is a Megatron-LM `torch_dist` checkpoint (sharded). To use:
```bash
git lfs install
git clone --branch iter_{iter_num:07d} https://huggingface.co/{REPO_ID}
```
Then point Megatron's `--load` at the cloned dir; tokenizer is at `assets/hf/cl100k_base`.
Architecture flags must match — see the recipe in the upstream training repo.

## Training schedule
- Iters 0–12,000: GBS=128, LR 3e-4 cosine, warmup 1000 — over 80,000-step schedule
- Iters 12,000–{LATEST:,}: GBS=512, LR cosine warm-restart over 30,000-step schedule
"""

for it in ITERS:
    branch  = f"iter_{it:07d}"
    src_dir = os.path.join(KEEP_DIR, f"iter_{it:07d}")
    is_main = (it == LATEST)

    print(f"\n=== Uploading iter {it} -> branch {branch}{' (also main)' if is_main else ''} ===")
    if not os.path.isdir(src_dir):
        print(f"  ERROR: {src_dir} does not exist, skipping"); continue

    # Branches we'll write to (in addition to main when this is the latest).
    targets = [branch] + (["main"] if is_main else [])
    for rev in targets:
        try:
            create_branch(REPO_ID, branch=rev, exist_ok=True)
        except Exception as e:
            # 'main' always exists; fine.
            print(f"  create_branch({rev}): {e!r}")

        print(f"  -> revision={rev}")

        # 1) Upload the checkpoint shards under iter_NNNNNNN/.
        print(f"     uploading {src_dir} -> {rev}:iter_{it:07d}/")
        upload_folder(
            folder_path=src_dir,
            path_in_repo=f"iter_{it:07d}",
            repo_id=REPO_ID,
            revision=rev,
            commit_message=f"Upload iter_{it} checkpoint shards",
        )

        # 2) Upload latest_checkpointed_iteration.txt at the root.
        latest_path = "/tmp/latest_checkpointed_iteration.txt"
        with open(latest_path, "w") as f:
            f.write(str(it))
        upload_file(
            path_or_fileobj=latest_path,
            path_in_repo="latest_checkpointed_iteration.txt",
            repo_id=REPO_ID,
            revision=rev,
            commit_message=f"Set latest iter to {it}",
        )

        # 3) Upload tokenizer at assets/hf/cl100k_base/.
        print(f"     uploading tokenizer -> {rev}:assets/hf/cl100k_base/")
        upload_folder(
            folder_path=TOK_DIR,
            path_in_repo="assets/hf/cl100k_base",
            repo_id=REPO_ID,
            revision=rev,
            commit_message="Upload cl100k_base tokenizer",
        )

        # 4) README for this branch.
        readme_path = "/tmp/README.md"
        with open(readme_path, "w") as f:
            f.write(make_readme(it))
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            revision=rev,
            commit_message=f"README for iter_{it}",
        )

print("\nDone. Repo URL: https://huggingface.co/${HF_REPO_ID}")
PY
LAUNCH
chmod +x "${LAUNCH_SCRIPT}"

echo "HF repo:       ${HF_REPO_ID}"
echo "Keep dir:      ${KEEP_DIR}"
echo "Tokenizer dir: ${TOKENIZER_DIR}"
echo "Launch script: ${LAUNCH_SCRIPT}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH_SCRIPT}"
