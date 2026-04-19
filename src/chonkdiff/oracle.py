"""Float64 Levenberg-Marquardt / Newton-Kantorovich projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .benchmark import NonlinearElliptic1D
from .config import OracleConfig


@dataclass
class LMResult:
    solution: torch.Tensor
    residual_history: List[float]
    lambda_history: List[float]
    alpha_history: List[float]
    converged: bool


def lm_project(
    benchmark: NonlinearElliptic1D,
    u: torch.Tensor,
    v0: torch.Tensor,
    config: OracleConfig,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> LMResult:
    """Run a damped LM/NK cleanup step in float64."""

    iterations = max_iterations if max_iterations is not None else config.max_iterations
    tol = tolerance if tolerance is not None else config.tolerance

    u_vec = u.detach().clone().to(dtype=torch.float64)
    v = v0.detach().clone().to(dtype=torch.float64)
    identity = torch.eye(benchmark.nx, dtype=torch.float64, device=v.device)

    lam = float(config.lambda_init)
    alpha = float(config.alpha_init)
    residual_history: List[float] = []
    lambda_history: List[float] = []
    alpha_history: List[float] = []

    for _ in range(iterations):
        residual = benchmark.residual(u_vec, v)
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        residual_history.append(residual_norm)
        lambda_history.append(lam)
        alpha_history.append(alpha)

        if residual_norm <= tol:
            return LMResult(v, residual_history, lambda_history, alpha_history, True)

        jacobian = benchmark.jacobian_matrix(v)
        jt_j = jacobian.transpose(0, 1) @ jacobian
        jt_r = jacobian.transpose(0, 1) @ residual

        accepted = False
        local_lambda = lam
        local_alpha = alpha
        for _ in range(config.line_search_steps):
            system = jt_j + local_lambda * identity
            step = torch.linalg.solve(system, jt_r)
            candidate = v - local_alpha * step
            candidate_residual = benchmark.residual(u_vec, candidate)
            candidate_norm = float(torch.linalg.vector_norm(candidate_residual).item())
            if candidate_norm < residual_norm:
                v = candidate
                lam = max(config.lambda_min, local_lambda * config.lambda_decay)
                alpha = min(1.0, local_alpha / max(config.alpha_decay, 1.0e-12))
                accepted = True
                break
            local_alpha *= config.alpha_decay
            local_lambda = min(config.lambda_max, local_lambda * config.lambda_inflate)

        if not accepted:
            lam = min(config.lambda_max, lam * config.lambda_inflate)
            alpha = max(config.min_alpha, alpha * config.alpha_decay)
            if alpha <= config.min_alpha and lam >= config.lambda_max:
                break

    return LMResult(v, residual_history, lambda_history, alpha_history, False)
