from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProblemConfig:
    family: str = "nonlinear_elliptic"
    equation: str = "-Delta v + kappa v^3 = u"
    kappa: float = 50.0
    nx: int = 63
    domain_length: float = 1.0
    reference_oracle_backend: str = "chonkdiff"
    reference_dataset_path: str = "data/chonkdiff_elliptic_dataset.npz"
    bootstrap_checkpoint: str | None = None
    forcing_period_length: float = 0.5
    forcing_lengthscale: float = 10.0
    forcing_jitter: float = 1.0e-10


@dataclass(slots=True)
class PosteriorConfig:
    mode: str = "partial_observation"
    observed_fraction: float = 0.1
    observation_noise_std: float = 0.0
    observation_pattern: str = "random_mask"
    seed: int = 7


@dataclass(slots=True)
class ModelConfig:
    hidden_channels: int = 64
    num_fno_layers: int = 4
    modes: int = 16
    time_embedding_dim: int = 64
    mlp_channels: int = 128


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 80
    seed: int = 1234
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    num_workers: int = 0
    checkpoint_dir: str = "outputs/posterior_projection"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    log_every: int = 1


@dataclass(slots=True)
class SamplingConfig:
    num_steps: int = 100
    observation_guidance_strength: float = 2.5e-1


@dataclass(slots=True)
class ProjectionConfig:
    schedules: list[str] = field(
        default_factory=lambda: [
            "none",
            "final_only",
            "every_step",
            "every_2",
            "every_5",
            "late_only",
            "adaptive_residual",
        ]
    )
    orders: list[str] = field(default_factory=lambda: ["first_order", "gauss_newton"])
    late_start_fraction: float = 0.7
    adaptive_warmup_fraction: float = 0.5
    adaptive_residual_threshold: float = 1.0e-2
    first_order_lambda: float = 1.0e-6
    second_order_lambda_init: float = 1.0e-4
    second_order_lambda_min: float = 1.0e-10
    second_order_lambda_max: float = 1.0e8
    second_order_lambda_decay: float = 0.5
    second_order_lambda_inflate: float = 10.0
    second_order_alpha_init: float = 1.0
    second_order_alpha_decay: float = 0.5
    second_order_min_alpha: float = 1.0e-6
    second_order_line_search_steps: int = 8
    max_projection_steps: int = 6
    final_cleanup: bool = True
    final_cleanup_iterations: int = 8
    final_cleanup_tolerance: float = 1.0e-10


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(
        default_factory=lambda: [
            "posterior_quality",
            "physical_consistency",
            "runtime",
            "trajectory_stability",
        ]
    )
    oracle_relative_error: bool = True
    num_eval_samples: int = 128
    num_observation_seeds: int = 3
    observation_seed_base: int = 17
    sample_seed_base: int = 123


@dataclass(slots=True)
class ExperimentConfig:
    study_name: str = "final_project_projection_posterior"
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_value(current: Any, incoming: Any) -> Any:
    if incoming is None:
        return None
    if current is None:
        return incoming
    if hasattr(current, "__dataclass_fields__"):
        return _merge_dataclass(current, incoming)
    if isinstance(current, bool):
        if isinstance(incoming, str):
            lowered = incoming.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(incoming)
    if isinstance(current, float):
        return float(incoming)
    if isinstance(current, int):
        return int(incoming)
    if isinstance(current, str):
        return str(incoming)
    if isinstance(current, list):
        return list(incoming)
    return incoming


def _merge_dataclass(default: Any, values: dict[str, Any] | None) -> Any:
    values = values or {}
    kwargs = {}
    for field_info in default.__dataclass_fields__.values():
        current = getattr(default, field_info.name)
        if field_info.name not in values:
            kwargs[field_info.name] = current
            continue
        incoming = values[field_info.name]
        kwargs[field_info.name] = _coerce_value(current, incoming)
    return type(default)(**kwargs)


def load_config_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return _merge_dataclass(ExperimentConfig(), raw)


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return load_config_dict(raw)
