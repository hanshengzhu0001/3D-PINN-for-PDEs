from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class JointNormalizationStats:
    u_mean: torch.Tensor
    u_std: torch.Tensor
    v_mean: torch.Tensor
    v_std: torch.Tensor

    @classmethod
    def from_npz(cls, data: dict[str, np.ndarray] | np.lib.npyio.NpzFile) -> "JointNormalizationStats":
        return cls(
            u_mean=torch.tensor(float(data["u_mean"]), dtype=torch.float32),
            u_std=torch.tensor(float(data["u_std"]), dtype=torch.float32),
            v_mean=torch.tensor(float(data["v_mean"]), dtype=torch.float32),
            v_std=torch.tensor(float(data["v_std"]), dtype=torch.float32),
        )

    def to(self, device: torch.device) -> "JointNormalizationStats":
        return JointNormalizationStats(
            u_mean=self.u_mean.to(device),
            u_std=self.u_std.to(device),
            v_mean=self.v_mean.to(device),
            v_std=self.v_std.to(device),
        )

    def normalize_u(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.u_mean.to(value.device, value.dtype)) / self.u_std.to(
            value.device, value.dtype
        )

    def denormalize_u(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.u_std.to(value.device, value.dtype) + self.u_mean.to(
            value.device, value.dtype
        )

    def normalize_v(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.v_mean.to(value.device, value.dtype)) / self.v_std.to(
            value.device, value.dtype
        )

    def denormalize_v(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.v_std.to(value.device, value.dtype) + self.v_mean.to(
            value.device, value.dtype
        )

    def normalize_state(self, u_value: torch.Tensor, v_value: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.normalize_u(u_value), self.normalize_v(v_value)], dim=0)

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.denormalize_u(state[..., 0, :]),
                self.denormalize_v(state[..., 1, :]),
            ],
            dim=-2,
        )


def sample_observation_mask(
    nx: int,
    observed_fraction: float,
    seed: int,
    pattern: str = "random_mask",
) -> torch.Tensor:
    if pattern != "random_mask":
        raise ValueError(f"unsupported observation pattern: {pattern}")
    count = max(1, int(round(nx * observed_fraction)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(nx, generator=generator)[:count]
    mask = torch.zeros(nx, dtype=torch.float32)
    mask[indices] = 1.0
    return mask


class JointStateDataset(Dataset):
    """Joint (u, v) dataset with on-the-fly partial observations on v."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        observed_fraction: float = 0.1,
        observation_noise_std: float = 0.0,
        observation_pattern: str = "random_mask",
        seed: int = 7,
        max_samples: int | None = None,
    ) -> None:
        data = np.load(dataset_path)
        u_all = torch.from_numpy(data[f"{split}_u"]).float()
        v_all = torch.from_numpy(data[f"{split}_v"]).float()
        if max_samples is not None:
            u_all = u_all[:max_samples]
            v_all = v_all[:max_samples]
        self.u_phys = u_all
        self.v_phys = v_all
        self.x_grid = torch.from_numpy(data["x"]).float()
        self.stats = JointNormalizationStats.from_npz(data)
        self.observed_fraction = observed_fraction
        self.observation_noise_std = observation_noise_std
        self.observation_pattern = observation_pattern
        self.seed = seed

    def __len__(self) -> int:
        return self.u_phys.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        u_phys = self.u_phys[index]
        v_phys = self.v_phys[index]
        state = self.stats.normalize_state(u_phys, v_phys)

        mask_seed = self.seed + index
        obs_mask = sample_observation_mask(
            nx=v_phys.numel(),
            observed_fraction=self.observed_fraction,
            seed=mask_seed,
            pattern=self.observation_pattern,
        )
        obs_noise = torch.zeros_like(v_phys)
        if self.observation_noise_std > 0.0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(mask_seed + 10_000)
            obs_noise = torch.randn(v_phys.shape, generator=generator) * self.observation_noise_std
        obs_v_phys = v_phys + obs_noise * obs_mask
        obs_v = self.stats.normalize_v(obs_v_phys)

        return {
            "state": state,
            "u_phys": u_phys,
            "v_phys": v_phys,
            "obs_mask": obs_mask,
            "obs_v": obs_v,
            "obs_v_phys": obs_v_phys,
            "x_grid": self.x_grid,
        }
