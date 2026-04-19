"""DDPM utilities for the conditional elliptic solver."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import NonlinearElliptic1D
from .config import DiffusionConfig
from .dataset import NormalizationStats


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.view(timesteps.shape[0], *([1] * (len(target_shape) - 1)))


class GaussianDiffusion1D(nn.Module):
    """Minimal conditional DDPM implementation."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.timesteps = config.timesteps

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        ) * noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            x_t
            - _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape) * noise
        ) / _extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)

    def loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        forcing_u: torch.Tensor,
        benchmark: Optional[NonlinearElliptic1D] = None,
        stats: Optional[NormalizationStats] = None,
        pde_weight: float = 0.0,
        bc_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x_start.shape[0]
        timesteps = self.sample_timesteps(batch_size, x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        predicted_noise = model(x_t, forcing_u, timesteps)

        diffusion_loss = F.mse_loss(predicted_noise, noise)
        total_loss = diffusion_loss
        metrics = {
            "loss_diff": float(diffusion_loss.detach().item()),
            "loss_pde": 0.0,
            "loss_bc": 0.0,
        }

        if (pde_weight > 0.0 or bc_weight > 0.0) and benchmark is not None and stats is not None:
            x0_pred = self.predict_start_from_noise(x_t, timesteps, predicted_noise)
            u_phys = stats.denormalize_u(forcing_u.squeeze(1))
            v_phys = stats.denormalize_v(x0_pred.squeeze(1))
            pde_loss = benchmark.residual(u_phys, v_phys).pow(2).mean()
            bc_loss = benchmark.periodic_boundary_loss(v_phys).pow(2).mean()
            total_loss = total_loss + pde_weight * pde_loss + bc_weight * bc_loss
            metrics["loss_pde"] = float(pde_loss.detach().item())
            metrics["loss_bc"] = float(bc_loss.detach().item())

        metrics["loss_total"] = float(total_loss.detach().item())
        return total_loss, metrics

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        forcing_u: torch.Tensor,
        timesteps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        predicted_noise = model(x_t, forcing_u, timesteps)
        beta_t = _extract(self.betas, timesteps, x_t.shape)
        sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, timesteps, x_t.shape)
        sqrt_one_minus_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )

        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * predicted_noise / sqrt_one_minus_cumprod_t
        )
        posterior_variance_t = _extract(self.posterior_variance, timesteps, x_t.shape)

        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        nonzero_mask = (timesteps > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t.clamp_min(1.0e-12)) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        forcing_u: torch.Tensor,
        signal_shape: torch.Size,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if x_init is None:
            x_t = torch.randn(signal_shape, device=forcing_u.device, generator=generator)
        else:
            x_t = x_init.clone()

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (signal_shape[0],), step, device=forcing_u.device, dtype=torch.long
            )
            x_t = self.p_sample(model, x_t, forcing_u, timesteps, generator=generator)
            if guidance_fn is not None:
                x_t = guidance_fn(x_t, forcing_u, timesteps)
        return x_t
