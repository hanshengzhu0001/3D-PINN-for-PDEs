"""Benchmark operators for the CHONKNORIS nonlinear elliptic problem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import BenchmarkConfig


@dataclass
class SampledForcing:
    x: torch.Tensor
    u: torch.Tensor


class NonlinearElliptic1D:
    """Periodic 1D nonlinear elliptic benchmark with dense Jacobians."""

    def __init__(
        self,
        config: BenchmarkConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.nx = config.nx
        self.domain_length = config.domain_length
        self.kappa = float(config.kappa)
        self.dx = self.domain_length / self.nx
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.x = torch.arange(self.nx, dtype=dtype, device=self.device) * self.dx
        self.base_operator = self._build_minus_laplacian_matrix().to(self.device)
        self._covariance_cache: Optional[np.ndarray] = None

    def _build_minus_laplacian_matrix(self) -> torch.Tensor:
        matrix = torch.zeros((self.nx, self.nx), dtype=self.dtype)
        scale = 1.0 / (self.dx ** 2)
        for idx in range(self.nx):
            matrix[idx, idx] = 2.0 * scale
            matrix[idx, (idx - 1) % self.nx] = -1.0 * scale
            matrix[idx, (idx + 1) % self.nx] = -1.0 * scale
        return matrix

    def periodic_kernel_covariance(self) -> np.ndarray:
        if self._covariance_cache is not None:
            return self._covariance_cache

        x = self.x.detach().cpu().numpy()
        delta = x[:, None] - x[None, :]
        period = self.config.forcing.period_length
        lengthscale = self.config.forcing.lengthscale
        covariance = np.exp(
            (-2.0 / lengthscale) * np.sin((math.pi / period) * delta) ** 2
        )
        covariance += self.config.forcing.jitter * np.eye(self.nx)
        self._covariance_cache = covariance
        return covariance

    def sample_forcing(self, n_samples: int, seed: int) -> SampledForcing:
        rng = np.random.default_rng(seed)
        covariance = self.periodic_kernel_covariance()
        samples = rng.multivariate_normal(
            mean=np.zeros(self.nx), cov=covariance, size=n_samples
        )
        u = torch.from_numpy(samples).to(dtype=self.dtype, device=self.device)
        return SampledForcing(x=self.x.clone(), u=u)

    def apply_minus_laplacian(self, v: torch.Tensor) -> torch.Tensor:
        return (2.0 * v - torch.roll(v, 1, dims=-1) - torch.roll(v, -1, dims=-1)) / (
            self.dx ** 2
        )

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.apply_minus_laplacian(v) + self.kappa * v.pow(3) - u

    def jacobian_matrix(self, v: torch.Tensor) -> torch.Tensor:
        base = self.base_operator.to(device=v.device, dtype=v.dtype)
        diagonal = 3.0 * self.kappa * v.pow(2)
        if v.ndim == 1:
            return base + torch.diag(diagonal)
        return base.unsqueeze(0) + torch.diag_embed(diagonal)

    def jtf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        residual = self.residual(u, v)
        jacobian = self.jacobian_matrix(v)
        if v.ndim == 1:
            return jacobian.transpose(0, 1) @ residual
        return torch.matmul(jacobian.transpose(-1, -2), residual.unsqueeze(-1)).squeeze(-1)

    def residual_norm(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.residual(u, v), dim=-1)

    def relative_l2_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
        denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1.0e-12)
        return numerator / denominator

    def periodic_boundary_loss(self, v: torch.Tensor) -> torch.Tensor:
        # The state is already represented on a periodic grid, so the discrete
        # finite-difference operator lives directly on the torus. We therefore
        # expose an explicit BC loss hook for training schedules, but it is
        # identically zero for this representation.
        shape = v.shape[:-1]
        return torch.zeros(shape, dtype=v.dtype, device=v.device)

    def periodic_bc_violation(self, _: torch.Tensor) -> torch.Tensor:
        # Periodicity is built into the finite-difference stencil, so the discrete
        # representation carries zero explicit boundary violation.
        return torch.zeros((), dtype=self.dtype, device=self.device)
