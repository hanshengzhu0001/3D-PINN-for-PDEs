"""High-level sampling pipeline for diffusion + projector evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from .benchmark import NonlinearElliptic1D
from .config import ExperimentConfig, load_config
from .dataset import NormalizationStats
from .diffusion import GaussianDiffusion1D
from .guidance import PhysicsGuidance
from .model import ConditionalDiffusionCNN
from .oracle import LMResult, lm_project


class DiffusionProjectorSolver:
    """Wraps the conditional diffusion model and final LM/NK projector."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: ConditionalDiffusionCNN,
        diffusion: GaussianDiffusion1D,
        stats: NormalizationStats,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.stats = stats.to(self.device)
        self.benchmark = NonlinearElliptic1D(config.benchmark, device=self.device)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: Optional[torch.device] = None
    ) -> "DiffusionProjectorSolver":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        if "config_dict" in checkpoint:
            config = _config_from_dict(checkpoint["config_dict"])
        else:
            config = load_config(checkpoint["config_path"])
        model = ConditionalDiffusionCNN(config.model)
        model.load_state_dict(checkpoint["model_state"])
        diffusion = GaussianDiffusion1D(config.diffusion)
        stats = NormalizationStats(
            u_mean=torch.tensor(checkpoint["stats"]["u_mean"], dtype=torch.float32),
            u_std=torch.tensor(checkpoint["stats"]["u_std"], dtype=torch.float32),
            v_mean=torch.tensor(checkpoint["stats"]["v_mean"], dtype=torch.float32),
            v_std=torch.tensor(checkpoint["stats"]["v_std"], dtype=torch.float32),
        )
        return cls(config, model, diffusion, stats, device=device)

    def sample(
        self,
        forcing_u: torch.Tensor,
        guidance_mode: Optional[str] = None,
        guidance_strength: Optional[float] = None,
        guidance_start_fraction: Optional[float] = None,
        guidance_lambda: Optional[float] = None,
        projector: bool = False,
        projector_iterations: Optional[int] = None,
        projector_tolerance: Optional[float] = None,
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[LMResult]]:
        if forcing_u.ndim == 1:
            forcing_u = forcing_u.unsqueeze(0)
        forcing_u = forcing_u.to(self.device, dtype=torch.float32)
        forcing_cond = self.stats.normalize_u(forcing_u).unsqueeze(1)

        guidance = None
        if guidance_mode and guidance_mode != "none":
            guidance = PhysicsGuidance(
                self.benchmark,
                self.stats,
                total_timesteps=self.config.diffusion.timesteps,
                mode=guidance_mode,
                strength=(
                    self.config.sampling.guidance_strength
                    if guidance_strength is None
                    else guidance_strength
                ),
                start_fraction=(
                    self.config.sampling.guidance_start_fraction
                    if guidance_start_fraction is None
                    else guidance_start_fraction
                ),
                lm_lambda=(
                    self.config.sampling.guidance_lambda
                    if guidance_lambda is None
                    else guidance_lambda
                ),
            )

        sample_norm = self.diffusion.sample(
            self.model,
            forcing_cond,
            signal_shape=(forcing_u.shape[0], 1, self.config.benchmark.nx),
            guidance_fn=guidance,
            x_init=x_init,
            generator=generator,
        )
        sample_phys = self.stats.denormalize_v(sample_norm.squeeze(1)).to(torch.float64)

        if not projector:
            return sample_phys, None

        result = lm_project(
            self.benchmark,
            forcing_u.squeeze(0).to(torch.float64),
            sample_phys.squeeze(0),
            self.config.oracle,
            max_iterations=(
                self.config.sampling.projector_iterations
                if projector_iterations is None
                else projector_iterations
            ),
            tolerance=projector_tolerance,
        )
        return result.solution.unsqueeze(0), result


def _config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig()

    def apply(target: Any, values: dict[str, Any]) -> Any:
        kwargs = {}
        for field_info in target.__dataclass_fields__.values():
            current = getattr(target, field_info.name)
            if field_info.name not in values:
                kwargs[field_info.name] = current
                continue
            incoming = values[field_info.name]
            if hasattr(current, "__dataclass_fields__"):
                kwargs[field_info.name] = apply(current, incoming)
            else:
                kwargs[field_info.name] = incoming
        return type(target)(**kwargs)

    return apply(config, raw)
