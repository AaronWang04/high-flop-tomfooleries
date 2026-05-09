"""Upload the standalone Cl100kChatTokenizer to Norapom/experimental_gqa_1_5b.

Lives under tokenizer/ on the `main` branch. Future SFT'd checkpoints (which
need the chat tokens) will rebase off main; the historical iter_NNNNNNNN
branches keep the original cl100k_base tokenizer so they remain reproducible.
"""

import os
import shutil
import textwrap

from huggingface_hub import HfApi, upload_file, upload_folder

REPO_ID = "Norapom/experimental_gqa_1_5b"
SRC_TOKENIZER_PY = (
    "/home/scratch.aarowang_ent/repos/Megatron-LM/megatron/core/tokenizers/"
    "text/libraries/cl100k_chat_tokenizer.py"
)
STAGING = "/tmp/tokenizer_upload"
shutil.rmtree(STAGING, ignore_errors=True)
os.makedirs(STAGING, exist_ok=True)

# 1) The tokenizer class file.
shutil.copy(SRC_TOKENIZER_PY, os.path.join(STAGING, "cl100k_chat_tokenizer.py"))

# 2) A standalone helper so you can use it without Megatron installed.
helper = textwrap.dedent("""\
    \"\"\"Standalone shim: import Cl100kChatTokenizer without needing Megatron.

    The tokenizer class only depends on `tiktoken`; the abstract base classes
    from Megatron are stripped out at import-time here for portability.
    \"\"\"
    import sys, types

    # Stub the Megatron abstract base classes that the file inherits from.
    # We don't need their behaviour at runtime — only the class hierarchy.
    _abstract = types.ModuleType("megatron.core.tokenizers.text.libraries.abstract_tokenizer")
    class MegatronTokenizerTextAbstract: ...
    _abstract.MegatronTokenizerTextAbstract = MegatronTokenizerTextAbstract

    _chat = types.ModuleType("megatron.core.tokenizers.text.libraries.chat_template")
    class MegatronTokenizerChatTemplate: ...
    _chat.MegatronTokenizerChatTemplate = MegatronTokenizerChatTemplate

    sys.modules.setdefault(
        "megatron.core.tokenizers.text.libraries.abstract_tokenizer", _abstract
    )
    sys.modules.setdefault(
        "megatron.core.tokenizers.text.libraries.chat_template", _chat
    )

    from cl100k_chat_tokenizer import Cl100kChatTokenizer, CL100K_CHAT_SPECIAL_TOKENS  # noqa
""")
with open(os.path.join(STAGING, "load_tokenizer.py"), "w") as f:
    f.write(helper)

# 3) README with usage + the explicit special-token table.
specials_md_lines = [
    "| Token | ID |",
    "|---|---|",
    "| `<\\|endoftext\\|>` | 100257 |",
    "| `<\\|fim_prefix\\|>` | 100258 |",
    "| `<\\|fim_middle\\|>` | 100259 |",
    "| `<\\|fim_suffix\\|>` | 100260 |",
    "| `<\\|endofprompt\\|>` | 100276 |",
    "| `<\\|im_start\\|>` | 100277 |",
    "| `<\\|im_end\\|>` | 100278 |",
    "| `<\\|think\\|>` | 100279 |",
    "| `<\\|/think\\|>` | 100280 |",
    "| `<\\|tool_call\\|>` | 100281 |",
    "| `<\\|/tool_call\\|>` | 100282 |",
    "| `<\\|tool_response\\|>` | 100283 |",
    "| `<\\|/tool_response\\|>` | 100284 |",
    "| `<\\|reserved_0\\|>` … `<\\|reserved_7\\|>` | 100285 … 100292 |",
]
readme = textwrap.dedent("""\
    # Cl100kChatTokenizer

    Tiktoken-based extension of cl100k_base with **16 chat / thinking / tool**
    special tokens at fixed IDs. Designed to fit inside the 100,352-padded
    vocab of the experimental_gqa_1_5b model — IDs 100277..100292 are 16
    reserved positions in the model's existing embedding rows.

    ## Why not HuggingFace AutoTokenizer?

    `transformers` fast tokenizer for cl100k auto-renumbers added_tokens at
    reload time, ignoring the explicit IDs in `tokenizer.json` and creating
    collisions (e.g. `<|/think|>` lands on top of `<|im_start|>`). Going
    straight to `tiktoken` with explicit special-token IDs sidesteps the bug.

    ## Token IDs

    {table}

    Tokenizer `vocab_size` is 100293; the model is padded to 100352.

    ## Usage

    ```bash
    pip install tiktoken
    ```

    ```python
    # If you have Megatron-LM available:
    from megatron.core.tokenizers.text.libraries.cl100k_chat_tokenizer import (
        Cl100kChatTokenizer,
    )

    # Standalone (no Megatron — uses the helper shim):
    import sys; sys.path.insert(0, "tokenizer")
    from load_tokenizer import Cl100kChatTokenizer, CL100K_CHAT_SPECIAL_TOKENS

    tok = Cl100kChatTokenizer()
    print(tok.text_to_ids("<|im_start|>user\\nhi<|im_end|>"))
    # -> [100277, 882, 198, 6151, 100278]
    ```

    ## Files
    - `cl100k_chat_tokenizer.py` — the class (full version, depends on Megatron base classes)
    - `load_tokenizer.py` — standalone shim that stubs the Megatron base classes
""").format(table="\n".join(specials_md_lines))
with open(os.path.join(STAGING, "README.md"), "w") as f:
    f.write(readme)

# Upload everything in STAGING -> tokenizer/ on main.
print(f"Uploading {STAGING}/* to {REPO_ID} (main:tokenizer/)...")
upload_folder(
    folder_path=STAGING,
    path_in_repo="tokenizer",
    repo_id=REPO_ID,
    revision="main",
    commit_message="Add Cl100kChatTokenizer (chat/think/tool reserved tokens)",
)

# Also append a one-line pointer to the top-level README so the new tokenizer
# is discoverable from the root.
api = HfApi()
try:
    root_readme = api.hf_hub_download(repo_id=REPO_ID, filename="README.md", revision="main")
    with open(root_readme) as f:
        body = f.read()
except Exception as e:
    print(f"Couldn't fetch root README: {e}")
    body = ""

POINTER = "\n\n## Tokenizer (chat-extended)\nSee [`tokenizer/`](./tokenizer) for the Cl100kChatTokenizer with 16 reserved chat/think/tool tokens (IDs 100277-100292).\n"
if "tokenizer/" not in body:
    body = body.rstrip() + POINTER
    out = "/tmp/README_root.md"
    with open(out, "w") as f:
        f.write(body)
    upload_file(
        path_or_fileobj=out,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        revision="main",
        commit_message="README: link to chat-extended tokenizer",
    )
    print("Updated root README.md with tokenizer pointer.")
else:
    print("Root README already references tokenizer/, skipping.")

print(f"\nDone. https://huggingface.co/{REPO_ID}/tree/main/tokenizer")
