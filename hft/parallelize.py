import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .model import HFTModel
from .qwen35 import Qwen35FullAttentionBlock, Qwen35LinearAttentionBlock, Qwen35Model


def parallelize_hft(
    model: HFTModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), (
        f"Sequence length {training.seq_len} must be divisible by the product "
        f"of TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = (
            float8_config is not None
            and float8_config.recipe_name in ("rowwise", "rowwise_with_gw_hp")
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        enable_sp = parallelism.enable_sequence_parallel

        tp_mesh = parallel_dims.get_mesh("tp")
        _apply_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    names = (
        ["dp_replicate", "fsdp"]
        if parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(names)
    _apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    logger.info("Applied fully_shard to the model")
    return model


def _apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_sp: bool = True,
):
    sp_layout = Shard(1) if enable_sp else Replicate()

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=enable_sp,
            ),
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    first_block = next(iter(model.layers.values()))
    use_fused_qkv = isinstance(first_block.attention.qkv_linear, FusedQKVLinear)

    for transformer_block in model.layers.values():
        if use_fused_qkv:
            qkv_plan = {"attention.qkv_linear.wqkv": colwise_parallel()}
        else:
            qkv_plan = {
                "attention.qkv_linear.wq": colwise_parallel(),
                "attention.qkv_linear.wk": colwise_parallel(),
                "attention.qkv_linear.wv": colwise_parallel(),
            }
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": prepare_module_input(
                input_layouts=(sp_layout, None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            **qkv_plan,
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
            "feed_forward": prepare_module_input(
                input_layouts=(sp_layout,),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_output_plan,
            "feed_forward.w3": colwise_parallel(),
        }
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def _apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)


# ---------------------------------------------------------------------------
# Qwen3.5 parallelization (hybrid: full attention + linear attention)
# ---------------------------------------------------------------------------


def parallelize_qwen35(
    model: Qwen35Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), (
        f"Sequence length {training.seq_len} must be divisible by the product "
        f"of TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = (
            float8_config is not None
            and float8_config.recipe_name in ("rowwise", "rowwise_with_gw_hp")
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        enable_sp = parallelism.enable_sequence_parallel

        tp_mesh = parallel_dims.get_mesh("tp")
        _apply_tp_qwen35(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.cp_enabled:
        full_attn_blocks = [
            block for block in model.layers.values()
            if isinstance(block, Qwen35FullAttentionBlock)
        ]
        if full_attn_blocks:
            apply_cp_to_attention_module(
                [block.attention.inner_attention for block in full_attn_blocks],
                parallel_dims.get_mesh("cp"),
            )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    names = (
        ["dp_replicate", "fsdp"]
        if parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(names)
    _apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    logger.info("Applied fully_shard to the Qwen3.5 model")
    return model


def _apply_tp_qwen35(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_sp: bool = True,
):
    sp_layout = Shard(1) if enable_sp else Replicate()

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=enable_sp,
            ),
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    ffn_plan = {
        "ffn_norm": norm_plan,
        "feed_forward": prepare_module_input(
            input_layouts=(sp_layout,),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": colwise_parallel(),
        "feed_forward.w2": rowwise_output_plan,
        "feed_forward.w3": colwise_parallel(),
    }

    for transformer_block in model.layers.values():
        if isinstance(transformer_block, Qwen35FullAttentionBlock):
            layer_plan = {
                "attention_norm": norm_plan,
                "attention": prepare_module_input(
                    input_layouts=(sp_layout, None, None, None),
                    desired_input_layouts=(Replicate(), None, None, None),
                ),
                "attention.q_proj": colwise_parallel(),
                "attention.k_proj": colwise_parallel(),
                "attention.v_proj": colwise_parallel(),
                "attention.o_proj": rowwise_output_plan,
                **ffn_plan,
            }
        elif isinstance(transformer_block, Qwen35LinearAttentionBlock):
            layer_plan = {
                "attention_norm": norm_plan,
                "attention": prepare_module_input(
                    input_layouts=(sp_layout, None, None, None),
                    desired_input_layouts=(Replicate(), None, None, None),
                ),
                "attention.in_proj_qkv": colwise_parallel(),
                "attention.in_proj_z": colwise_parallel(),
                "attention.out_proj": rowwise_output_plan,
                **ffn_plan,
            }
        else:
            raise TypeError(f"Unknown block type: {type(transformer_block)}")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the Qwen3.5 model"
    )
