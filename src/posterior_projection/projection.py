from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import ProjectionConfig
from .problem import JointPosteriorProblem


@dataclass(slots=True)
class ProjectionResult:
    state: torch.Tensor
    residual_history: list[float]
    lambda_history: list[float]
    alpha_history: list[float]
    converged: bool
    elapsed_seconds: float

    @property
    def iterations(self) -> int:
        return len(self.residual_history)


def _flatten_state(state: torch.Tensor) -> torch.Tensor:
    return state.reshape(-1)


def _unflatten_state(vector: torch.Tensor, nx: int) -> torch.Tensor:
    return vector.reshape(2, nx)


def first_order_project(
    problem: JointPosteriorProblem,
    state: torch.Tensor,
    damping: float = 1.0e-6,
) -> ProjectionResult:
    start_time = time.perf_counter()
    x = state.detach().clone().to(dtype=torch.float64)
    residual = problem.residual_from_state(x)
    jacobian = problem.joint_jacobian_from_state(x)
    identity = torch.eye(problem.nx, dtype=torch.float64, device=x.device)
    schur = jacobian @ jacobian.transpose(0, 1) + damping * identity
    step_rhs = torch.linalg.solve(schur, residual)
    delta = jacobian.transpose(0, 1) @ step_rhs
    projected = _flatten_state(x) - delta
    state_out = _unflatten_state(projected, problem.nx)
    elapsed = time.perf_counter() - start_time
    return ProjectionResult(
        state=state_out,
        residual_history=[float(torch.linalg.vector_norm(residual).item())],
        lambda_history=[float(damping)],
        alpha_history=[1.0],
        converged=False,
        elapsed_seconds=elapsed,
    )


def second_order_project(
    problem: JointPosteriorProblem,
    state_hat: torch.Tensor,
    config: ProjectionConfig,
    max_iterations: int | None = None,
    tolerance: float | None = None,
) -> ProjectionResult:
    start_time = time.perf_counter()
    x_hat = state_hat.detach().clone().to(dtype=torch.float64)
    x = x_hat.clone()
    nx = problem.nx
    identity = torch.eye(2 * nx, dtype=torch.float64, device=x.device)
    lambda_dual = torch.zeros(nx, dtype=torch.float64, device=x.device)

    iterations = max_iterations if max_iterations is not None else config.max_projection_steps
    tol = tolerance if tolerance is not None else config.final_cleanup_tolerance
    lam = float(config.second_order_lambda_init)
    alpha = float(config.second_order_alpha_init)

    residual_history: list[float] = []
    lambda_history: list[float] = []
    alpha_history: list[float] = []

    for _ in range(iterations):
        residual = problem.residual_from_state(x)
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        residual_history.append(residual_norm)
        lambda_history.append(lam)
        alpha_history.append(alpha)
        if residual_norm <= tol:
            return ProjectionResult(
                state=x,
                residual_history=residual_history,
                lambda_history=lambda_history,
                alpha_history=alpha_history,
                converged=True,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        jacobian = problem.joint_jacobian_from_state(x)
        x_vector = _flatten_state(x)
        x_hat_vector = _flatten_state(x_hat)
        grad_lagrangian = (1.0 + lam) * (x_vector - x_hat_vector) + jacobian.transpose(0, 1) @ lambda_dual

        kkt = torch.zeros((3 * nx, 3 * nx), dtype=torch.float64, device=x.device)
        kkt[: 2 * nx, : 2 * nx] = (1.0 + lam) * identity
        kkt[: 2 * nx, 2 * nx :] = jacobian.transpose(0, 1)
        kkt[2 * nx :, : 2 * nx] = jacobian

        rhs = -torch.cat([grad_lagrangian, residual], dim=0)

        accepted = False
        local_lambda = lam
        local_alpha = alpha
        for _ in range(config.second_order_line_search_steps):
            if local_lambda != lam:
                kkt[: 2 * nx, : 2 * nx] = (1.0 + local_lambda) * identity
                rhs[: 2 * nx] = -(
                    (1.0 + local_lambda) * (x_vector - x_hat_vector)
                    + jacobian.transpose(0, 1) @ lambda_dual
                )
            delta = torch.linalg.solve(kkt, rhs)
            delta_x = delta[: 2 * nx]
            delta_lambda = delta[2 * nx :]

            candidate = _unflatten_state(x_vector + local_alpha * delta_x, nx)
            candidate_lambda = lambda_dual + local_alpha * delta_lambda
            candidate_residual = problem.residual_from_state(candidate)
            candidate_norm = float(torch.linalg.vector_norm(candidate_residual).item())

            if candidate_norm < residual_norm:
                x = candidate
                lambda_dual = candidate_lambda
                lam = max(config.second_order_lambda_min, local_lambda * config.second_order_lambda_decay)
                alpha = min(1.0, local_alpha / max(config.second_order_alpha_decay, 1.0e-12))
                accepted = True
                break

            local_alpha *= config.second_order_alpha_decay
            local_lambda = min(config.second_order_lambda_max, local_lambda * config.second_order_lambda_inflate)

        if not accepted:
            lam = min(config.second_order_lambda_max, lam * config.second_order_lambda_inflate)
            alpha = max(config.second_order_min_alpha, alpha * config.second_order_alpha_decay)
            if alpha <= config.second_order_min_alpha and lam >= config.second_order_lambda_max:
                break

    return ProjectionResult(
        state=x,
        residual_history=residual_history,
        lambda_history=lambda_history,
        alpha_history=alpha_history,
        converged=False,
        elapsed_seconds=time.perf_counter() - start_time,
    )
