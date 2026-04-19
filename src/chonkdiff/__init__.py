"""Diffusion-first nonlinear elliptic solver utilities."""

from .benchmark import NonlinearElliptic1D
from .config import ExperimentConfig, load_config
from .diffusion import GaussianDiffusion1D
from .guidance import PhysicsGuidance
from .model import ConditionalDiffusionCNN
from .oracle import LMResult, lm_project
from .pipeline import DiffusionProjectorSolver

__all__ = [
    "ConditionalDiffusionCNN",
    "DiffusionProjectorSolver",
    "ExperimentConfig",
    "GaussianDiffusion1D",
    "LMResult",
    "NonlinearElliptic1D",
    "PhysicsGuidance",
    "lm_project",
    "load_config",
]
