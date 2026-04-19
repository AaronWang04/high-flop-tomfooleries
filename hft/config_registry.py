import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hft_8b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path=os.path.join(_SCRIPT_DIR, "assets", "hf", "Llama-3.1-8B"),
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def hft_qwen35_9b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path=os.path.join(_SCRIPT_DIR, "assets", "hf", "Qwen3-8B"),
        model_spec=model_registry("qwen35-9B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )
