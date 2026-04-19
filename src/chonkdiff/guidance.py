"""Physics-guided reverse diffusion corrections."""

from __future__ import annotations

import torch

from .benchmark import NonlinearElliptic1D
from .dataset import NormalizationStats


class PhysicsGuidance:
    """Applies light J^T F or damped Gauss-Newton corrections during sampling."""

    def __init__(
        self,
        benchmark: NonlinearElliptic1D,
        stats: NormalizationStats,
        total_timesteps: int,
        mode: str = "gn",
        strength: float = 1.0e-3,
        start_fraction: float = 0.35,
        lm_lambda: float = 1.0e-3,
    ) -> None:
        self.benchmark = benchmark
        self.stats = stats
        self.total_timesteps = max(int(total_timesteps), 1)
        self.mode = "jtf" if mode == "residual" else mode
        self.strength = float(strength)
        self.start_fraction = float(start_fraction)
        self.lm_lambda = float(lm_lambda)

    def _guidance_scale(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0.0:
            return torch.zeros_like(timesteps, dtype=torch.float64)

        if self.total_timesteps <= 1:
            progress = torch.ones_like(timesteps, dtype=torch.float64)
        else:
            progress = 1.0 - timesteps.to(torch.float64) / float(self.total_timesteps - 1)

        start = min(max(self.start_fraction, 0.0), 1.0)
        if start >= 1.0:
            ramp = (timesteps == 0).to(torch.float64)
        else:
            ramp = ((progress - start) / max(1.0 - start, 1.0e-12)).clamp(0.0, 1.0)
        return self.strength * ramp

    def _normalize_direction(self, direction: torch.Tensor) -> torch.Tensor:
        direction = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-12)
        return direction / norm

    def _gauss_newton_direction(self, u_phys: torch.Tensor, v_phys: torch.Tensor) -> torch.Tensor:
        residual = self.benchmark.residual(u_phys, v_phys)
        jacobian = self.benchmark.jacobian_matrix(v_phys)
        jt = jacobian.transpose(-1, -2)
        jt_j = torch.matmul(jt, jacobian)
        jt_r = torch.matmul(jt, residual.unsqueeze(-1))

        eye = torch.eye(
            self.benchmark.nx,
            dtype=v_phys.dtype,
            device=v_phys.device,
        )
        if v_phys.ndim == 1:
            system = jt_j + self.lm_lambda * eye
        else:
            system = jt_j + self.lm_lambda * eye.unsqueeze(0)

        try:
            step = torch.linalg.solve(system, jt_r).squeeze(-1)
        except RuntimeError:
            step = self.benchmark.jtf(u_phys, v_phys)
        return step

    def __call__(
        self,
        x_t: torch.Tensor,
        forcing_u: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "none" or self.strength <= 0.0:
            return x_t

        guidance_scale = self._guidance_scale(timesteps)
        if torch.all(guidance_scale <= 0.0):
            return x_t

        x_dtype = x_t.dtype
        u_phys = self.stats.denormalize_u(forcing_u.squeeze(1)).to(torch.float64)
        v_phys = self.stats.denormalize_v(x_t.squeeze(1)).to(torch.float64)

        if self.mode == "jtf":
            direction = self.benchmark.jtf(u_phys, v_phys)
        elif self.mode == "gn":
            direction = self._gauss_newton_direction(u_phys, v_phys)
        else:
            raise ValueError(f"Unsupported guidance mode: {self.mode}")

        direction = self._normalize_direction(direction)
        correction = guidance_scale.to(direction.device).unsqueeze(-1) * direction
        corrected = torch.nan_to_num(
            v_phys - correction, nan=0.0, posinf=0.0, neginf=0.0
        )
        return torch.nan_to_num(
            self.stats.normalize_v(corrected.to(dtype=x_dtype)).unsqueeze(1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
