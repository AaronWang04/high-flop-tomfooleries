"""Wrappers around flash-linear-attention (FLA) layers.

Adapts the FLA forward signature:
    (hidden_states, attention_mask, past_key_values, ...) -> (output, attn, kv_cache)
to the TorchTitan convention:
    (x, freqs_cis, attention_masks, positions) -> Tensor

so every class here is a drop-in for the ``attention`` slot of a TransformerBlock.

Required:   pip install flash-linear-attention
Optional:   pip install mamba-ssm   (Mamba2 CUDA fast path; Triton fallback otherwise)
            pip install flash-attn  (required for MultiheadLatentAttention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchtitan.models.common.attention import AttentionMasksType

try:
    import fla  # noqa: F401
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


def _require_fla(cls_name: str) -> None:
    if not _FLA_AVAILABLE:
        raise ImportError(
            f"{cls_name} requires flash-linear-attention: "
            "pip install flash-linear-attention"
        )


# ---------------------------------------------------------------------------
# Kimi Linear (KimiDeltaAttention)
# ---------------------------------------------------------------------------


class KimiLinear(nn.Module):
    """Kimi Linear attention from "Kimi Linear: An Expressive, Efficient
    Attention Architecture" (arXiv:2510.26692).

    Extends GatedDeltaNet with per-dim-per-head vector gating instead of a
    scalar-per-head gate. No RoPE — freqs_cis and positions are ignored.
    """

    @dataclass
    class Config:
        dim: int = 4096
        head_dim: int = 128
        num_heads: int = 16
        expand_v: int = 1
        num_v_heads: Optional[int] = None
        mode: str = "chunk"
        use_short_conv: bool = True
        conv_size: int = 4
        norm_eps: float = 1e-5
        layer_idx: Optional[int] = None

        def build(self) -> "KimiLinear":
            return KimiLinear(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("KimiLinear")
        from fla.layers import KimiDeltaAttention
        self._attn = KimiDeltaAttention(
            hidden_size=config.dim,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            expand_v=config.expand_v,
            num_v_heads=config.num_v_heads,
            mode=config.mode,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        out, _, _ = self._attn(x)
        return out


# ---------------------------------------------------------------------------
# DeepSeek Native Sparse Attention (NSA)
# ---------------------------------------------------------------------------


class NativeSparseAttention(nn.Module):
    """DeepSeek Native Sparse Attention (NSA).

    Three-way gated combination: compressed blocks + selected top-k blocks +
    local sliding window. RoPE is managed internally by FLA; pass positions
    as position_ids when tokens are not sequentially ordered.

    Reference: "Native Sparse Attention: Hardware-Aligned and Natively
    Trainable Sparse Attention" (DeepSeek, 2025).
    """

    @dataclass
    class Config:
        dim: int = 4096
        num_heads: int = 64
        num_kv_heads: int = 4
        head_dim: int = 64
        block_size: int = 64      # tokens per compressed block
        block_counts: int = 16    # top-k blocks to attend per query
        window_size: int = 512    # local sliding window tokens
        rope_theta: float = 10000.0
        max_position_embeddings: Optional[int] = None
        qkv_bias: bool = False
        layer_idx: Optional[int] = None

        def build(self) -> "NativeSparseAttention":
            return NativeSparseAttention(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("NativeSparseAttention")
        from fla.layers import NativeSparseAttention as _NSA
        self._attn = _NSA(
            hidden_size=config.dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            qkv_bias=config.qkv_bias,
            block_size=config.block_size,
            block_counts=config.block_counts,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        # NSA computes RoPE internally; pass position_ids for non-causal ordering.
        out, _, _ = self._attn(x, position_ids=positions)
        return out


# ---------------------------------------------------------------------------
# DeepSeek Multi-head Latent Attention (MLA)
# ---------------------------------------------------------------------------


class MultiheadLatentAttention(nn.Module):
    """DeepSeek Multi-head Latent Attention (MLA).

    Low-rank KV compression: projects keys and values through a shared latent
    vector (kv_lora_rank << n_heads * head_dim), dramatically reducing KV cache
    size. Optionally compresses queries too (q_lora_rank).

    Requires: pip install flash-attn

    Reference: DeepSeek-V2/V3 technical reports.
    """

    @dataclass
    class Config:
        dim: int = 4096
        num_heads: int = 16
        q_lora_rank: Optional[int] = None   # None = no Q compression
        qk_rope_head_dim: int = 64
        kv_lora_rank: int = 512
        v_head_dim: int = 128
        qk_nope_head_dim: int = 128
        rope_theta: float = 10000.0
        layer_idx: Optional[int] = None

        def build(self) -> "MultiheadLatentAttention":
            return MultiheadLatentAttention(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("MultiheadLatentAttention")
        try:
            from fla.layers import MultiheadLatentAttention as _MLA
        except ImportError as e:
            raise ImportError(
                "MultiheadLatentAttention also requires flash-attn: "
                "pip install flash-attn"
            ) from e
        self._attn = _MLA(
            hidden_size=config.dim,
            num_heads=config.num_heads,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        out, _, _ = self._attn(x, position_ids=positions)
        return out


# ---------------------------------------------------------------------------
# Gated Linear Attention (GLA)
# ---------------------------------------------------------------------------


class GatedLinearAttention(nn.Module):
    """Gated Linear Attention (GLA).

    Data-dependent gating applied to a linear attention formulation.
    No RoPE — freqs_cis and positions are ignored.

    Reference: "Gated Linear Attention Transformers with Hardware-Efficient
    Training" (Yang et al., 2023).
    """

    @dataclass
    class Config:
        dim: int = 4096
        expand_k: float = 0.5
        expand_v: float = 1.0
        num_heads: int = 4
        num_kv_heads: Optional[int] = None
        mode: str = "chunk"
        use_short_conv: bool = False
        conv_size: int = 4
        use_output_gate: bool = True
        layer_idx: Optional[int] = None

        def build(self) -> "GatedLinearAttention":
            return GatedLinearAttention(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("GatedLinearAttention")
        from fla.layers import GatedLinearAttention as _GLA
        self._attn = _GLA(
            hidden_size=config.dim,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            mode=config.mode,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            use_output_gate=config.use_output_gate,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        out, _, _ = self._attn(x)
        return out


# ---------------------------------------------------------------------------
# Mamba2
# ---------------------------------------------------------------------------


class Mamba2(nn.Module):
    """Mamba2 state space model.

    Uses the FLA implementation which has a pure Triton fallback if mamba-ssm
    is not installed (install mamba-ssm for the faster CUDA kernel path).

    Mamba2 replaces the attention sublayer; a standard FFN can still follow it
    in a TransformerBlock for extra capacity. freqs_cis and positions are
    ignored (SSMs are position-implicit).

    Reference: "Transformers are SSMs: Generalized Models and Efficient
    Algorithms through Structured State Space Duality" (Dao & Gu, 2024).
    """

    @dataclass
    class Config:
        dim: int = 4096
        head_dim: int = 64
        num_heads: Optional[int] = None   # inferred as dim*expand // head_dim if None
        state_size: int = 128
        expand: int = 2
        n_groups: int = 1
        conv_kernel: int = 4
        chunk_size: int = 256
        layer_idx: Optional[int] = None

        def build(self) -> "Mamba2":
            return Mamba2(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("Mamba2")
        from fla.layers import Mamba2 as _Mamba2
        self._ssm = _Mamba2(
            hidden_size=config.dim,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            state_size=config.state_size,
            expand=config.expand,
            n_groups=config.n_groups,
            conv_kernel=config.conv_kernel,
            chunk_size=config.chunk_size,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        out, _, _ = self._ssm(x)
        return out


# ---------------------------------------------------------------------------
# RWKV-7
# ---------------------------------------------------------------------------


class RWKV7(nn.Module):
    """RWKV-7 attention (Eagle/Finch architecture).

    Token-mixing via data-dependent state-space recurrence. No RoPE.

    Reference: "RWKV-7 "Goose" with Expressive Dynamic State Evolution"
    (Peng et al., 2025).
    """

    @dataclass
    class Config:
        dim: int = 4096
        head_dim: int = 64
        num_heads: Optional[int] = None
        mode: str = "chunk"
        layer_idx: Optional[int] = None

        def build(self) -> "RWKV7":
            return RWKV7(self)

    def __init__(self, config: Config):
        super().__init__()
        _require_fla("RWKV7")
        from fla.layers import RWKV7Attention
        self._attn = RWKV7Attention(
            hidden_size=config.dim,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            mode=config.mode,
            layer_idx=config.layer_idx,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        out, _, _ = self._attn(x)
        return out
