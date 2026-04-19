from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def flow_matching_loss(
    model: nn.Module,
    data_state: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = data_state.shape[0]
    x0 = torch.randn_like(data_state)
    t = torch.rand(batch, device=data_state.device, dtype=data_state.dtype)
    xt = (1.0 - t).view(batch, 1, 1) * x0 + t.view(batch, 1, 1) * data_state
    target_velocity = data_state - x0
    predicted_velocity = model(xt, t)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    return loss, {"loss_total": float(loss.detach().item())}


def euler_sample(
    model: nn.Module,
    initial_state: torch.Tensor,
    num_steps: int,
    guidance_fn: Callable[[torch.Tensor, int, float], torch.Tensor] | None = None,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dt = 1.0 / num_steps
    state = initial_state.clone()
    trajectory = [state.clone()] if return_trajectory else None

    for step in range(num_steps):
        t = torch.full(
            (state.shape[0],),
            float(step) / float(num_steps),
            device=state.device,
            dtype=state.dtype,
        )
        with torch.no_grad():
            velocity = model(state, t)
        state = state + dt * velocity
        if guidance_fn is not None:
            state = guidance_fn(state, step, dt)
        if trajectory is not None:
            trajectory.append(state.clone())

    if trajectory is None:
        return state, None
    return state, torch.stack(trajectory, dim=1)
