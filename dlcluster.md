# DL Cluster Environment Reference

## Node topology

| Layer | Host | Architecture | Home directory |
|---|---|---|---|
| Login node | `<login>` | x86_64 | `/home/scratch.aarowang_ent/` (NFS, autofs) |
| Compute — `gb200nvl72_qa24h` | `gb-nvl-0XX-compute09` | aarch64 (Grace-Blackwell) | `/home/aarowang/` (separate NFS) |
| Compute — `gb200nvl4` | `gb200-nvl4-ts2-XXX` | aarch64 | `/home/aarowang/` |

The login node's `~` is `/home/scratch.aarowang_ent/` — a 1TB scratch NFS. Compute nodes have a **separate** small NFS at `/home/aarowang/`. These are not the same path and not the same filesystem.

## Scratch NFS visibility

`/home/scratch.aarowang_ent/` is a Santa Clara Non-Farm Engineering Storage NFS export. **It only works within the SC region.** A compute node in another region (e.g. Hillsboro/`pdx02-ipp6`) will not see the path no matter what flags you pass.

- **`gb200nvl72_qa24h`** — all nodes in `santa_clara/mission_college`, scratch is visible (autofs mounts on first access).
- **`gb200nvl4`** — all 29 nodes (as of 2026-05-03) are in `hillsboro/pdx02-ipp6`. Scratch NFS is unreachable from this partition. Use CIFS or pick a different partition.

Check a node's region before debugging mount issues:
```bash
curl 'https://dlfw-api.dc6.k8s.nvidia.com/api/v1/node/location?node=gb200-nvl4-ts2-99'
# → ["hillsboro","pdx02-ipp6"]
```

**Cross-region access**: Use CIFS at `/mnt/cifs/home/scratch.aarowang_ent` instead of the NFS path. CIFS access requires a storage ticket granting DLCluster service accounts access to your scratch.

**Pitfall**: `srun --pty bash` on `gb200nvl72_qa24h` can access scratch because bash access triggers autofs. Pyxis containers do NOT inherit host mounts — scratch is invisible inside a container unless `--container-mounts` is specified AND autofs has already been triggered on the host before Pyxis runs.

**Fix pattern** in batch scripts for single-node jobs:
```bash
ls "${SCRATCH_ROOT}" > /dev/null   # trigger autofs before container launch
srun --container-image=... --container-mounts="${SCRATCH_ROOT}:${SCRATCH_ROOT}" ...
```

For multi-node jobs, each node needs the trigger. Use a bare srun step before the container step:
```bash
srun --ntasks="${NNODES}" --ntasks-per-node=1 ls "${SCRATCH_ROOT}" > /dev/null
```

## Container system (Pyxis / enroot)

The container is convenient because it ships a known-good PyTorch + CUDA + NCCL stack for aarch64/Blackwell, but its PyTorch is not a special NVIDIA-only build — equivalent versions are available upstream. Running directly on the host with a self-managed aarch64 conda env is viable if container friction outweighs the convenience.

Image format: `nvcr.io#nvidia/pytorch:26.02-py3` — use `#` as registry separator, not `/`.

```bash
srun --container-image=nvcr.io#nvidia/pytorch:26.02-py3 \
     --container-mounts=/home/scratch.aarowang_ent:/home/scratch.aarowang_ent \
     --container-workdir=/home/scratch.aarowang_ent/repos/high-flop-tomfooleries \
     bash script.sh
```

Key flags:
- `--container-mounts=HOST:CONTAINER` — bind-mount; host path must exist before Pyxis runs
- `--container-workdir` — sets working directory inside container
- `--container-env=VAR1,VAR2` — pass specific env vars into container (all env is not forwarded by default when using direct `torchrun`; safer to set vars explicitly inside the launch script)

**The container does not have Megatron-LM installed.** Must use PYTHONPATH.

## Conda/micromamba environment

The login node has a micromamba env at `/home/scratch.aarowang_ent/micromamba/envs/pretrain/` (x86_64). Activating it sets `CC`, `CXX`, `AR`, `LD` to conda cross-compilation toolchain paths (`x86_64-conda-linux-gnu-g++` etc.).

**Pitfall**: Slurm jobs inherit the submitting shell's environment, including these `CXX` vars. Inside an aarch64 container, those x86_64 compiler paths don't exist → any `make` call (including Megatron's on-import C++ extension build) fails with error 127.

**Fix**: Add to the top of every launch script that runs inside the container:
```bash
unset CC CXX AR LD
```

This is already added to the heredoc in `megatron/smoke_test.sh` and `megatron/slurm_multinode.sh`.

## Megatron-LM setup

Checkout lives at `/home/scratch.aarowang_ent/repos/Megatron-LM/`.

**`helpers_cpp` C++ extension**: Megatron auto-compiles this on first import. Must be pre-compiled once for aarch64:

```bash
srun --partition=gb200nvl72_qa24h --ntasks=1 \
     --container-image=nvcr.io#nvidia/pytorch:26.02-py3 \
     --container-mounts=/home/scratch.aarowang_ent:/home/scratch.aarowang_ent \
     --container-workdir=/home/scratch.aarowang_ent/repos/Megatron-LM/megatron/core/datasets \
     make CXX=g++
```

The compiled `.so` is stored on NFS and persists across jobs. If the file is ever deleted or Python version changes, re-run the above.

## FLA (flash-linear-attention) setup

Not included in the PyTorch container. Installed once to NFS:

```bash
srun --partition=gb200nvl72_qa24h --ntasks=1 \
     --container-image=nvcr.io#nvidia/pytorch:26.02-py3 \
     --container-mounts=/home/scratch.aarowang_ent:/home/scratch.aarowang_ent \
     pip install --only-binary=:all --target=/home/scratch.aarowang_ent/packages \
         flash-linear-attention
```

Use `--only-binary=:all` to avoid compilation. After install, remove conflicting packages that shadow the container's torch/CUDA:

```bash
rm -rf /home/scratch.aarowang_ent/packages/torch \
       /home/scratch.aarowang_ent/packages/torch-*.dist-info \
       /home/scratch.aarowang_ent/packages/torchgen \
       /home/scratch.aarowang_ent/packages/cuda \
       /home/scratch.aarowang_ent/packages/cuda_*.dist-info \
       /home/scratch.aarowang_ent/packages/nvidia \
       /home/scratch.aarowang_ent/packages/nvidia_*.dist-info
```

PYTHONPATH in launch scripts: `${MEGATRON_ROOT}:${SCRATCH_ROOT}/packages`

## Slurm script pitfalls

### `$0` path in sbatch scripts
Slurm copies the batch script to a temp location (`/var/lib/slurm/slurmd/jobXXX/slurm_script`). `dirname "$0"` returns that temp path, not the original location.

**Fix**: Use `SLURM_SUBMIT_DIR` (set by Slurm to the directory sbatch was called from):
```bash
REPO_ROOT="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_ROOT}/megatron"
MEGATRON_ROOT="$(cd "${REPO_ROOT}/../Megatron-LM" && pwd)"
```
Scripts must be submitted from the repo root: `sbatch megatron/smoke_test.sh`.

### Multiline MODEL_ARGS in heredocs
Recipe files define `MODEL_ARGS`, `TRAIN_ARGS`, `DATA_ARGS` as multiline bash strings (leading `\n`). Expanding them directly in a `torchrun` call via `${MODEL_ARGS}` breaks line continuations (the leading blank line terminates the command before the args are read).

**Fix**: Flatten to a single line before embedding in the heredoc:
```bash
$(printf '%s' "${MODEL_ARGS}" | tr '\n' ' ') \\
```
This preserves single-quoted values (like `'([1]*32)'`) which bash later parses correctly when executing the generated script file.

### `--linear-attention-freq` value format
The `la_freq_type` parser validates the string against `[,\d\[\]\(\)\+\*]` only. Single quotes passed as literal characters (from incorrect shell expansion) cause a validation error. Pattern length must match the total number of transformer layers (e.g., `'([1]*32)'` for 32 all-LA layers).

### `--eval-interval` required even with `--eval-iters 0`
Megatron computes eval data sample counts regardless of whether evals are disabled. Always include `--eval-interval` in TRAIN_ARGS.

### HuggingFace tokenizer paths
`huggingface_hub >= 1.0` validates tokenizer paths as repo IDs and rejects relative paths like `assets/hf/Qwen3.5-4B`. Use an absolute path via `HF_TOKENIZER_PATH`, or use `--tokenizer-type NullTokenizer` for smoke tests.

### Megatron checkpoint save: nvrx API mismatch
The 26.02 PyTorch container ships `nvidia_resiliency_ext` with a private API (`_get_write_results_queue`), but Megatron-LM expects the public name (`get_write_results_queue`). Megatron's sync save path hardcodes `"nvrx" if HAVE_NVRX else "mcore"` — so even passing `--async-strategy mcore` doesn't help; sync save still tries the broken nvrx path and crashes at the first checkpoint.

**Symptoms before the fix**: training works fine, then either fails with `ImportError: cannot import name 'get_write_results_queue'` at the first save (small `--save-interval`), or hangs silently after several iterations as ranks desync waiting for the broken save (large `--save-interval`).

**Fix**: shadow the broken package with a stub on NFS that raises `ImportError`, so `HAVE_NVRX` becomes False and Megatron uses the working mcore strategy:
```bash
mkdir -p ${SCRATCH_ROOT}/packages/nvidia_resiliency_ext
echo 'raise ImportError("shadowed: container nvrx has broken API")' \
    > ${SCRATCH_ROOT}/packages/nvidia_resiliency_ext/__init__.py
```
This works because `${SCRATCH_ROOT}/packages` is earlier in PYTHONPATH than the container's site-packages, so the stub is imported first.

### NCCL MNNVL on GB200 in containers
NCCL on GB200 probes for Multi-Node NVLink (MNNVL) and tries to use `/dev/nvidia-caps-imex-channels`. Pyxis containers don't expose those device files, so NCCL fails with `Cuda failure 800 'operation not permitted'` and the job hangs in init. Disable MNNVL in launch scripts:
```bash
export NCCL_MNNVL_ENABLE=0
```
Symptoms before the fix: 1-4 node jobs may work (depending on which NVLink domain they land in), but ≥8 nodes hang silently in distributed init for the entire job lifetime.
