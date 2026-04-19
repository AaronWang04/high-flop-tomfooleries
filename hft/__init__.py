from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import HFTModel, HFTTransformerBlock
from .parallelize import parallelize_hft

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


# --- Llama-style layer builder ---

def _build_llama_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    attn_backend: str = "sdpa",
) -> list[HFTTransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        HFTTransformerBlock.Config(
            attention_norm=RMSNorm.Config(
                normalized_shape=dim, param_init=_NORM_INIT
            ),
            ffn_norm=RMSNorm.Config(
                normalized_shape=dim, param_init=_NORM_INIT
            ),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(i),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="complex",
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(i),
            ),
        )
        for i in range(n_layers)
    ]


# --- Qwen3-style layer builder (QK-Norm, cos_sin RoPE, eps=1e-6) ---

_QWEN3_EPS = 1e-6


def _qwen3_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_QWEN3_EPS, param_init=_NORM_INIT)


def _build_qwen3_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str = "sdpa",
) -> list[HFTTransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        HFTTransformerBlock.Config(
            attention_norm=_qwen3_norm(dim),
            ffn_norm=_qwen3_norm(dim),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(i),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="cos_sin",
                qk_norm=_qwen3_norm(head_dim),
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(i),
            ),
        )
        for i in range(n_layers)
    ]


# --- Model configs ---

def _8b(attn_backend: str = "sdpa") -> HFTModel.Config:
    """Llama 3.1 8B architecture."""
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 128256
    return HFTModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_llama_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=compute_ffn_hidden_dim(
                dim, multiple_of=1024, ffn_dim_multiplier=1.3
            ),
            attn_backend=attn_backend,
        ),
    )


def _qwen35_9b(attn_backend: str = "sdpa") -> HFTModel.Config:
    """Qwen3.5-9B: Qwen3 architecture, ~9B params.

    Interpolates between Qwen3 8B (36 layers) and 14B (40 layers),
    keeping the 8B width with the 14B depth for ~9.0B total parameters.
    """
    dim = 4096
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8
    n_layers = 40
    hidden_dim = 12288
    vocab_size = 151936
    return HFTModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=_qwen3_norm(dim),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=4096,
            theta=1_000_000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
        ),
    )


hft_configs = {
    "8B": _8b,
    "qwen35-9B": _qwen35_9b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    config = hft_configs[flavor](attn_backend=attn_backend)
    return ModelSpec(
        name="hft",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_hft,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
