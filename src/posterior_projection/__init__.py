"""Posterior projection study package."""

from .config import (
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PosteriorConfig,
    ProblemConfig,
    ProjectionConfig,
    SamplingConfig,
    TrainingConfig,
    load_config,
)
from .dataset import JointNormalizationStats, JointStateDataset, sample_observation_mask
from .flow import euler_sample, flow_matching_loss
from .model import FlowMatchingFNO1D
from .objectives import (
    physics_consistency,
    posterior_quality,
    runtime_cost,
    scalarized_objective,
    trajectory_stability,
)
from .problem import JointPosteriorProblem
from .projection import first_order_project, second_order_project
from .study import (
    ProjectionCase,
    build_default_cases,
    materialize_schedule,
    should_project,
)

__all__ = [
    "EvaluationConfig",
    "ExperimentConfig",
    "FlowMatchingFNO1D",
    "JointNormalizationStats",
    "JointPosteriorProblem",
    "JointStateDataset",
    "ModelConfig",
    "PosteriorConfig",
    "ProblemConfig",
    "ProjectionCase",
    "ProjectionConfig",
    "SamplingConfig",
    "TrainingConfig",
    "build_default_cases",
    "euler_sample",
    "first_order_project",
    "flow_matching_loss",
    "load_config",
    "materialize_schedule",
    "physics_consistency",
    "posterior_quality",
    "runtime_cost",
    "sample_observation_mask",
    "scalarized_objective",
    "second_order_project",
    "should_project",
    "trajectory_stability",
]
