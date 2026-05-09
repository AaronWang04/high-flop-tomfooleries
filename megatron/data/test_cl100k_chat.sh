#!/usr/bin/bash
# Smoke test the new Cl100kChatTokenizer: verify chat tokens encode as
# single ids at the expected positions, decode round-trips.
#
#SBATCH --job-name=tok-test
#SBATCH --partition=gb200nvl72_qa24h
#SBATCH --exclude=gb-nvl-147-compute[01-09],gb-nvl-115-compute[01-18],gb200-nvl4-ts2-[64-65,68-69,72-75,77-82,86-93,97,99-100,102-105]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/slurm-%j.out
#SBATCH --error=outputs/slurm/slurm-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRATCH_ROOT=/home/scratch.aarowang_ent
MEGATRON_ROOT="${SCRATCH_ROOT}/repos/Megatron-LM"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.02-py3}

ls "${SCRATCH_ROOT}" > /dev/null
LAUNCH="${REPO_ROOT}/outputs/slurm/tok_test_${SLURM_JOB_ID}.sh"
cat > "${LAUNCH}" << LAUNCH
#!/usr/bin/bash
set -euo pipefail
unset CC CXX AR LD
export PYTHONPATH="${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages:\${PYTHONPATH:-}"

pip install --quiet --target="${SCRATCH_ROOT}/packages" tiktoken
python3 -c "import tiktoken; print('tiktoken', tiktoken.__version__)"

python3 - << 'PY'
from megatron.core.tokenizers.text.libraries.cl100k_chat_tokenizer import (
    Cl100kChatTokenizer, CL100K_CHAT_SPECIAL_TOKENS,
)

t = Cl100kChatTokenizer()
print(f"vocab_size={t.vocab_size}")
print(f"eod={t.eod}")

# 1) Single-token IDs of all specials.
print("\nSpecial token IDs (must all be unique and match the table):")
for name, expected_id in CL100K_CHAT_SPECIAL_TOKENS.items():
    got = t.text_to_ids(name)
    assert got == [expected_id], f"{name!r}: expected single id {expected_id}, got {got}"
    print(f"  {name!r:24s} -> {expected_id}  OK")

# 2) Round-trip a chat snippet.
chat = ("<|im_start|>system\nyou are helpful<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|think|>say hi back<|/think|>\nhello!<|im_end|>\n")
ids = t.text_to_ids(chat)
back = t.ids_to_text(ids)
print(f"\nChat encodes to {len(ids)} ids; specials present:")
for name in ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]:
    cid = CL100K_CHAT_SPECIAL_TOKENS[name]
    print(f"  {name!r:14s} (id {cid}): count={ids.count(cid)}")
assert back == chat, f"Round-trip mismatch:\nORIG: {chat!r}\nGOT : {back!r}"
print("Round-trip OK")

# 3) Tool call test.
tc = "<|tool_call|>{\"name\":\"calc\",\"args\":[1,2]}<|/tool_call|>"
ids = t.text_to_ids(tc)
assert CL100K_CHAT_SPECIAL_TOKENS["<|tool_call|>"] in ids
assert CL100K_CHAT_SPECIAL_TOKENS["<|/tool_call|>"] in ids
assert t.ids_to_text(ids) == tc
print("Tool-call round-trip OK")

# 4) Verify reserved tokens are addressable.
for i in range(8):
    name = f"<|reserved_{i}|>"
    cid = CL100K_CHAT_SPECIAL_TOKENS[name]
    assert t.text_to_ids(name) == [cid]
print("All 8 reserved tokens addressable.")

print("\nAll tokenizer smoke tests passed.")
PY
LAUNCH
chmod +x "${LAUNCH}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    bash "${LAUNCH}"
