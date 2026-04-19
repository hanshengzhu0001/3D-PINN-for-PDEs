from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, size = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes],
            self.weight[:, :, :modes],
        )
        return torch.fft.irfft(out_ft, n=size, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int, time_dim: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.spectral(x) + self.pointwise(x)
        h = h + self.time_proj(time_embedding).unsqueeze(-1)
        h = torch.nn.functional.silu(self.norm(h))
        return x + h


class FlowMatchingFNO1D(nn.Module):
    """Compact 1D FNO-style vector field for joint (u, v) flow matching."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.mlp_channels),
            nn.SiLU(),
            nn.Linear(config.mlp_channels, config.time_embedding_dim),
        )
        self.input_proj = nn.Conv1d(5, config.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock1d(
                    channels=config.hidden_channels,
                    modes=config.modes,
                    time_dim=config.time_embedding_dim,
                )
                for _ in range(config.num_fno_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, config.hidden_channels), config.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(config.hidden_channels, 2, kernel_size=1),
        )

    def forward(self, state: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch, _, nx = state.shape
        grid = torch.linspace(0.0, 1.0, nx, device=state.device, dtype=state.dtype)
        grid = grid.unsqueeze(0).expand(batch, -1)
        grid_features = torch.stack(
            [
                grid,
                torch.sin(2.0 * math.pi * grid),
                torch.cos(2.0 * math.pi * grid),
            ],
            dim=1,
        )
        x = torch.cat([state, grid_features], dim=1)
        x = self.input_proj(x)
        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        for block in self.blocks:
            x = block(x, time_embedding)
        return self.output_proj(x)
