from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def posterior_quality(neg_log_prior: float, neg_log_likelihood: float) -> float:
    """Posterior-quality surrogate J_post = -log p_theta(u) - log p(y | u)."""
    return float(neg_log_prior + neg_log_likelihood)


def physics_consistency(residual: np.ndarray) -> float:
    """Physical-consistency objective J_phys = ||h(u)||_2^2."""
    residual_array = np.asarray(residual, dtype=np.float64)
    return float(np.sum(residual_array * residual_array))


def runtime_cost(
    base_sampling_seconds: float,
    projection_step_seconds: Iterable[float],
    final_projection_seconds: float = 0.0,
) -> float:
    """Runtime objective J_time from baseline, intermediate, and final projection cost."""
    return float(
        base_sampling_seconds
        + sum(float(value) for value in projection_step_seconds)
        + final_projection_seconds
    )


def trajectory_stability(constrained_path: np.ndarray, baseline_path: np.ndarray) -> float:
    """Trajectory-stability objective J_traj = sum_k ||u_k^c - u_k^base||_2^2."""
    constrained = np.asarray(constrained_path, dtype=np.float64)
    baseline = np.asarray(baseline_path, dtype=np.float64)
    if constrained.shape != baseline.shape:
        raise ValueError(
            f"trajectory shapes must match, got {constrained.shape} and {baseline.shape}"
        )
    delta = constrained - baseline
    return float(np.sum(delta * delta))


def scalarized_objective(
    j_post: float,
    j_phys: float,
    j_time: float,
    j_traj: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Scalarized study objective J_post + alpha J_phys + beta J_time + gamma J_traj."""
    return float(j_post + alpha * j_phys + beta * j_time + gamma * j_traj)
