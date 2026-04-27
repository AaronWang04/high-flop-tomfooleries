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

from .model import Model, TransformerBlock
from .modules import GatedDeltaNetAttention, GemmaRMSNorm, Qwen35FullAttention
from .parallelize import parallelize, parallelize_qwen35
from .qwen35 import Qwen35FullAttentionBlock, Qwen35LinearAttentionBlock, Qwen35Model

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
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        TransformerBlock.Config(
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
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        TransformerBlock.Config(
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

def _8b(attn_backend: str = "sdpa") -> Model.Config:
    """Llama 3.1 8B architecture."""
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    n_layers = 32
    vocab_size = 128256
    return Model.Config(
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


def _qwen3_8b(attn_backend: str = "sdpa") -> Model.Config:
    """Qwen3-8B architecture."""
    dim = 4096
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8
    n_layers = 36
    hidden_dim = 12288
    vocab_size = 151936
    return Model.Config(
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


def _qwen35_9b(attn_backend: str = "sdpa") -> Model.Config:
    """Qwen3.5-9B: Qwen3 8B width with 14B depth, ~9B params."""
    dim = 4096
    head_dim = 128
    n_heads = 32
    n_kv_heads = 8
    n_layers = 40
    hidden_dim = 12288
    vocab_size = 151936
    return Model.Config(
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


# --- Qwen3.5 real architecture (hybrid: GatedDeltaNet + gated full attention) ---

_QWEN35_EPS = 1e-6
_QWEN35_LAYER_PATTERN = ["linear_attention"] * 3 + ["full_attention"]


def _gemma_norm(dim: int) -> GemmaRMSNorm.Config:
    return GemmaRMSNorm.Config(normalized_shape=dim, eps=_QWEN35_EPS)


def _build_qwen35_real_layers(
    *,
    dim: int = 4096,
    n_layers: int = 32,
    full_attn_n_heads: int = 16,
    full_attn_n_kv_heads: int = 4,
    full_attn_head_dim: int = 256,
    linear_n_k_heads: int = 16,
    linear_n_v_heads: int = 32,
    linear_key_head_dim: int = 128,
    linear_value_head_dim: int = 128,
    linear_conv_kernel: int = 4,
    hidden_dim: int = 12288,
) -> list:
    layers = []
    for i in range(n_layers):
        layer_type = _QWEN35_LAYER_PATTERN[i % len(_QWEN35_LAYER_PATTERN)]

        ffn = make_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim,
            w1_param_init=_LINEAR_INIT,
            w2w3_param_init=_depth_init(i),
        )

        if layer_type == "full_attention":
            layers.append(Qwen35FullAttentionBlock.Config(
                attention_norm=_gemma_norm(dim),
                ffn_norm=_gemma_norm(dim),
                attention=Qwen35FullAttention.Config(
                    dim=dim,
                    n_heads=full_attn_n_heads,
                    n_kv_heads=full_attn_n_kv_heads,
                    head_dim=full_attn_head_dim,
                    eps=_QWEN35_EPS,
                ),
                feed_forward=ffn,
            ))
        else:
            layers.append(Qwen35LinearAttentionBlock.Config(
                attention_norm=_gemma_norm(dim),
                ffn_norm=_gemma_norm(dim),
                attention=GatedDeltaNetAttention.Config(
                    dim=dim,
                    n_k_heads=linear_n_k_heads,
                    n_v_heads=linear_n_v_heads,
                    key_head_dim=linear_key_head_dim,
                    value_head_dim=linear_value_head_dim,
                    conv_kernel=linear_conv_kernel,
                    eps=_QWEN35_EPS,
                ),
                feed_forward=ffn,
            ))

    return layers


def _qwen35_9b_real(attn_backend: str = "sdpa") -> Qwen35Model.Config:
    """Real Qwen3.5-9B: hybrid GatedDeltaNet + gated full attention."""
    dim = 4096
    head_dim = 256
    partial_rotary_factor = 0.25
    rotary_dim = int(head_dim * partial_rotary_factor)
    vocab_size = 248320
    return Qwen35Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=1e7,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=_gemma_norm(dim),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=8192,
            theta=1e7,
            backend="cos_sin",
        ),
        layers=_build_qwen35_real_layers(),
    )


_configs = {
    "8B": _8b,
    "qwen3-8B": _qwen3_8b,
    "qwen35-9B": _qwen35_9b,
    "qwen35-9B-real": _qwen35_9b_real,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> ModelSpec:
    config = _configs[flavor](attn_backend=attn_backend)
    par_fn = parallelize_qwen35 if flavor == "qwen35-9B-real" else parallelize
    return ModelSpec(
        name="models",
        flavor=flavor,
        model=config,
        parallelize_fn=par_fn,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
