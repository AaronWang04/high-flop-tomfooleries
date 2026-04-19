import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


class HFTTransformerBlock(TransformerBlock):
    """Pre-norm transformer block with GQA + SwiGLU FFN."""

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
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class HFTModel(Decoder):
    """Decoder-only transformer for pretraining experiments."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 128256
        enable_weight_tying: bool = False

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
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_kv_heads ({n_kv_heads})."
                    )

            if (
                self.enable_weight_tying
                and parallelism.pipeline_parallel_degree > 1
            ):
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight
        super().init_states(buffer_device=buffer_device)
