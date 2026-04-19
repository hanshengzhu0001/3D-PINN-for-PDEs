"""Small conditional 1D diffusion backbone."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """Residual 1D convolution block with timestep conditioning."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalDiffusionCNN(nn.Module):
    """A compact 1D residual CNN for epsilon prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )
        self.input_proj = nn.Conv1d(2, config.hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(config.hidden_channels, config.time_embedding_dim)
                for _ in range(config.num_blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(min(8, config.hidden_channels), config.hidden_channels)
        self.output_proj = nn.Conv1d(config.hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, noisy_v: torch.Tensor, forcing_u: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_v, forcing_u], dim=1)
        x = self.input_proj(x)
        t_embed = self.time_mlp(self.time_embedding(timesteps))
        for block in self.blocks:
            x = block(x, t_embed)
        x = self.output_proj(F.silu(self.output_norm(x)))
        return x
