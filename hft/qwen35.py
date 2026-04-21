"""Qwen3.5 hybrid transformer: full-attention blocks + linear-attention blocks."""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.tools.logging import logger

from .modules import (
    GatedDeltaNetAttention,
    GemmaRMSNorm,
    Qwen35FullAttention,
    precompute_partial_rope_cache,
)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class Qwen35FullAttentionBlock(TransformerBlock):

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions,
        )
        return h + self.feed_forward(self.ffn_norm(h))


class Qwen35LinearAttentionBlock(TransformerBlock):

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_masks: AttentionMasksType | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions,
        )
        return h + self.feed_forward(self.ffn_norm(h))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Qwen35Model(Decoder):

    def verify_module_protocol(self) -> None:
        pass

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 248320
        head_dim: int = 256
        partial_rotary_factor: float = 0.25
        rope_theta: float = 1e7

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len

            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for layer_cfg in self.layers:
                    attn = layer_cfg.attention
                    if isinstance(attn, Qwen35FullAttention.Config):
                        if attn.n_heads % tp != 0:
                            raise ValueError(
                                f"TP degree ({tp}) must divide full attention "
                                f"n_heads ({attn.n_heads})."
                            )
                        if attn.n_kv_heads % tp != 0:
                            raise ValueError(
                                f"TP degree ({tp}) must divide full attention "
                                f"n_kv_heads ({attn.n_kv_heads})."
                            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int,
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            flops = 0
            for block in model.layers.values():
                for p in block.parameters():
                    flops += 2 * p.numel() * seq_len
            return nparams, flops
