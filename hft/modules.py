"""Qwen3.5-specific modules: GemmaRMSNorm, RMSNormGated, GatedDeltaNet, gated full attention."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductAttention,
)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class GemmaRMSNorm(nn.Module):
    """RMSNorm with zero-initialized weight: ``(1 + w) * x / rms``."""

    @dataclass
    class Config:
        normalized_shape: int
        eps: float = 1e-6

        def build(self) -> "GemmaRMSNorm":
            return GemmaRMSNorm(self.normalized_shape, self.eps)

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(nn.Module):
    """``rms_norm(x) * silu(gate)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (self.weight * x).to(input_dtype)
        return x * F.silu(gate.float()).to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE helpers (partial rotary, cos_sin backend)
# ---------------------------------------------------------------------------


def precompute_partial_rope_cache(
    head_dim: int,
    max_seq_len: int,
    partial_rotary_factor: float = 0.25,
    theta: float = 1e7,
) -> Tensor:
    rotary_dim = int(head_dim * partial_rotary_factor)
    freqs = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=freqs.dtype)
    idx_theta = torch.outer(t, freqs)
    idx_theta = torch.cat([idx_theta, idx_theta], dim=-1)
    return torch.cat([idx_theta.cos(), idx_theta.sin()], dim=-1)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_emb(
    xq: Tensor,
    xk: Tensor,
    rope_cache: Tensor,
    positions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to the first ``rotary_dim`` dims of q and k only."""
    if positions is None:
        seqlen = xq.shape[1]
        rope_cache = rope_cache[:seqlen]
    elif positions.ndim == 1 or (positions.ndim == 2 and positions.shape[0] == 1):
        rope_cache = rope_cache[positions.view(-1)]
    else:
        bz, seqlen = positions.shape
        rope_cache = rope_cache[None].expand(bz, -1, -1)
        rope_cache = torch.gather(
            rope_cache, 1,
            positions.unsqueeze(-1).expand(-1, -1, rope_cache.shape[-1]),
        )

    rotary_dim = rope_cache.shape[-1] // 2
    cos = rope_cache[..., :rotary_dim]
    sin = rope_cache[..., rotary_dim:]

    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    cos = cos.to(xq.dtype)
    sin = sin.to(xq.dtype)

    q_rot, q_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    k_rot, k_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    return (
        torch.cat([q_rot, q_pass], dim=-1),
        torch.cat([k_rot, k_pass], dim=-1),
    )


# ---------------------------------------------------------------------------
# GatedDeltaNet (linear attention)
# ---------------------------------------------------------------------------


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _naive_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
) -> Tensor:
    """Recurrent gated delta rule — naive O(T) loop, used as fallback."""
    B, H, T, d_k = k.shape
    d_v = v.shape[-1]
    S = torch.zeros(B, H, d_k, d_v, device=v.device, dtype=v.dtype)
    out = torch.zeros_like(v)

    for t in range(T):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)

        S = S * g_t
        delta = (v_t - (S * k_t.unsqueeze(-1)).sum(-2)) * beta_t
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (S * q_t.unsqueeze(-1)).sum(-2)

    return out


def _try_import_fla_chunk():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception:
        return None


class GatedDeltaNetAttention(nn.Module):
    """Gated Delta Rule linear attention (Qwen3.5 linear_attention layers)."""

    @dataclass
    class Config:
        dim: int = 4096
        n_k_heads: int = 16
        n_v_heads: int = 32
        key_head_dim: int = 128
        value_head_dim: int = 128
        conv_kernel: int = 4
        eps: float = 1e-6

        def build(self) -> "GatedDeltaNetAttention":
            return GatedDeltaNetAttention(self)

    def __init__(self, config: Config):
        super().__init__()
        self.n_k_heads = config.n_k_heads
        self.n_v_heads = config.n_v_heads
        self.d_k = config.key_head_dim
        self.d_v = config.value_head_dim
        self.key_dim = self.n_k_heads * self.d_k
        self.value_dim = self.n_v_heads * self.d_v

        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(config.dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.dim, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.dim, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.dim, self.n_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.dim, bias=False)

        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, config.conv_kernel,
            groups=conv_dim, padding=config.conv_kernel - 1, bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(
            torch.empty(self.n_v_heads).uniform_(0, 16).log()
        )
        self.norm = RMSNormGated(self.d_v, eps=config.eps)

        self._chunk_fn = _try_import_fla_chunk()

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = x.shape

        qkv = self.in_proj_qkv(x)
        qkv = F.silu(
            self.conv1d(qkv.transpose(1, 2))[:, :, :T]
        ).transpose(1, 2)

        z = self.in_proj_z(x).view(B, T, self.n_v_heads, self.d_v)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )

        q, k, v = torch.split(
            qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        q = q.view(B, T, self.n_k_heads, self.d_k)
        k = k.view(B, T, self.n_k_heads, self.d_k)
        v = v.view(B, T, self.n_v_heads, self.d_v)

        if self.n_v_heads > self.n_k_heads:
            r = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        if self._chunk_fn is not None and x.is_cuda:
            y, _ = self._chunk_fn(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                g=g, beta=beta,
                scale=self.d_k ** -0.5,
                output_final_state=False,
            )
            y = y.transpose(1, 2)
        else:
            q = q.transpose(1, 2).contiguous().float()
            k = k.transpose(1, 2).contiguous().float()
            v = v.transpose(1, 2).contiguous().float()
            beta = beta.unsqueeze(-1).float() if beta.ndim == 3 else beta.float()
            g_float = g.float()

            q = _l2norm(q) / (q.shape[-1] ** 0.5)
            k = _l2norm(k)

            if beta.ndim == 3:
                beta = beta.transpose(1, 2).contiguous()
            if g_float.ndim == 3:
                g_float = g_float.transpose(1, 2).contiguous()

            y = _naive_gated_delta_rule(q, k, v, g_float, beta)
            y = y.transpose(1, 2).contiguous().to(x.dtype)

        y = self.norm(
            y.reshape(-1, self.d_v),
            z.reshape(-1, self.d_v),
        )
        return self.out_proj(y.view(B, T, -1))


# ---------------------------------------------------------------------------
# Gated Full Attention (Qwen3.5 full_attention layers)
# ---------------------------------------------------------------------------


class Qwen35FullAttention(nn.Module):
    """GQA with output gating, QK-Norm, and partial RoPE."""

    @dataclass
    class Config:
        dim: int = 4096
        n_heads: int = 16
        n_kv_heads: int = 4
        head_dim: int = 256
        eps: float = 1e-6

        def build(self) -> "Qwen35FullAttention":
            return Qwen35FullAttention(self)

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.q_proj = nn.Linear(
            config.dim, config.n_heads * config.head_dim * 2, bias=False,
        )
        self.k_proj = nn.Linear(
            config.dim, config.n_kv_heads * config.head_dim, bias=False,
        )
        self.v_proj = nn.Linear(
            config.dim, config.n_kv_heads * config.head_dim, bias=False,
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.dim, bias=False,
        )

        self.q_norm = GemmaRMSNorm(config.head_dim, eps=config.eps)
        self.k_norm = GemmaRMSNorm(config.head_dim, eps=config.eps)

        self.inner_attention = ScaledDotProductAttention(None)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        bs, seqlen, _ = x.shape

        qg = self.q_proj(x).view(bs, seqlen, -1, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(bs, seqlen, -1)

        k = self.k_proj(x).view(bs, seqlen, -1, self.head_dim)
        v = self.v_proj(x).view(bs, seqlen, -1, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_partial_rotary_emb(q, k, freqs_cis, positions)

        y = self.inner_attention(
            q, k, v,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        y = y.contiguous().reshape(bs, seqlen, -1)

        y = y * torch.sigmoid(gate)
        return self.o_proj(y)
