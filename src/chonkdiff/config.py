"""Configuration helpers for the diffusion-first elliptic benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ForcingConfig:
    period_length: float = 0.5
    lengthscale: float = 10.0
    jitter: float = 1.0e-10


@dataclass
class DatasetConfig:
    train_size: int = 896
    val_size: int = 128
    seed: int = 42
    out_path: str = "data/chonkdiff_elliptic_dataset.npz"


@dataclass
class BenchmarkConfig:
    nx: int = 63
    domain_length: float = 1.0
    kappa: float = 50.0
    warmup_iterations: int = 5
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


@dataclass
class OracleConfig:
    max_iterations: int = 64
    tolerance: float = 1.0e-12
    lambda_init: float = 1.0e-4
    lambda_min: float = 1.0e-10
    lambda_max: float = 1.0e8
    lambda_decay: float = 0.5
    lambda_inflate: float = 10.0
    alpha_init: float = 1.0
    alpha_decay: float = 0.5
    min_alpha: float = 1.0e-6
    line_search_steps: int = 8


@dataclass
class ModelConfig:
    hidden_channels: int = 64
    num_blocks: int = 8
    time_embedding_dim: int = 128


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 80
    seed: int = 1234
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    num_workers: int = 0
    checkpoint_dir: str = "outputs/chonkdiff"
    pde_weight: float = 3.0e-5
    bc_weight: float = 0.0
    stage_a_fraction: float = 0.2
    stage_b_fraction: float = 0.75


@dataclass
class SamplingConfig:
    guidance_mode: str = "gn"
    guidance_strength: float = 1.0e-3
    guidance_start_fraction: float = 0.35
    guidance_lambda: float = 1.0e-3
    projector_iterations: int = 8
    num_eval_samples: int = 32
    seed: int = 123


@dataclass
class ExperimentConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(default: Any, values: Dict[str, Any] | None) -> Any:
    values = values or {}
    kwargs = {}
    for field_info in default.__dataclass_fields__.values():
        current = getattr(default, field_info.name)
        if field_info.name not in values:
            kwargs[field_info.name] = current
            continue
        incoming = values[field_info.name]
        if hasattr(current, "__dataclass_fields__"):
            kwargs[field_info.name] = _merge_dataclass(current, incoming)
        else:
            try:
                kwargs[field_info.name] = type(current)(incoming)
            except (TypeError, ValueError):
                kwargs[field_info.name] = incoming
    return type(default)(**kwargs)


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    default = ExperimentConfig()
    return _merge_dataclass(default, raw)
