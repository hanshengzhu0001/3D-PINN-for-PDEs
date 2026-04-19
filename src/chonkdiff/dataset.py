"""Oracle dataset generation and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .benchmark import NonlinearElliptic1D
from .config import ExperimentConfig
from .oracle import lm_project


@dataclass
class NormalizationStats:
    u_mean: torch.Tensor
    u_std: torch.Tensor
    v_mean: torch.Tensor
    v_std: torch.Tensor

    @classmethod
    def from_npz(cls, data: Dict[str, np.ndarray]) -> "NormalizationStats":
        return cls(
            u_mean=torch.tensor(float(data["u_mean"]), dtype=torch.float32),
            u_std=torch.tensor(float(data["u_std"]), dtype=torch.float32),
            v_mean=torch.tensor(float(data["v_mean"]), dtype=torch.float32),
            v_std=torch.tensor(float(data["v_std"]), dtype=torch.float32),
        )

    def to(self, device: torch.device) -> "NormalizationStats":
        return NormalizationStats(
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


def generate_oracle_dataset(config: ExperimentConfig, force: bool = False) -> Path:
    """Generate the periodic-forcing benchmark dataset if it does not exist."""

    out_path = Path(config.benchmark.dataset.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return out_path

    benchmark = NonlinearElliptic1D(config.benchmark)
    total = config.benchmark.dataset.train_size + config.benchmark.dataset.val_size
    forcings = benchmark.sample_forcing(total, config.benchmark.dataset.seed).u

    warm_solutions = []
    oracle_solutions = []
    warm_residuals = []
    oracle_residuals = []

    for index, forcing in enumerate(forcings):
        warm = lm_project(
            benchmark,
            forcing,
            torch.zeros_like(forcing),
            config.oracle,
            max_iterations=config.benchmark.warmup_iterations,
            tolerance=0.0,
        )
        oracle = lm_project(
            benchmark,
            forcing,
            warm.solution,
            config.oracle,
        )
        warm_solutions.append(warm.solution.detach().cpu().numpy())
        oracle_solutions.append(oracle.solution.detach().cpu().numpy())
        warm_residuals.append(warm.residual_history[-1])
        oracle_residuals.append(oracle.residual_history[-1])

        if (index + 1) % 64 == 0 or index + 1 == total:
            print(f"Generated {index + 1}/{total} oracle pairs")

    warm_array = np.stack(warm_solutions, axis=0)
    oracle_array = np.stack(oracle_solutions, axis=0)
    forcings_array = forcings.detach().cpu().numpy()

    train_size = config.benchmark.dataset.train_size
    train_u = forcings_array[:train_size]
    train_v = oracle_array[:train_size]
    train_warm = warm_array[:train_size]
    val_u = forcings_array[train_size:]
    val_v = oracle_array[train_size:]
    val_warm = warm_array[train_size:]

    u_mean = float(train_u.mean())
    u_std = float(train_u.std() + 1.0e-6)
    v_mean = float(train_v.mean())
    v_std = float(train_v.std() + 1.0e-6)

    np.savez_compressed(
        out_path,
        x=benchmark.x.detach().cpu().numpy(),
        train_u=train_u,
        train_v=train_v,
        train_warm=train_warm,
        val_u=val_u,
        val_v=val_v,
        val_warm=val_warm,
        train_warm_residual=np.asarray(warm_residuals[:train_size], dtype=np.float64),
        train_oracle_residual=np.asarray(oracle_residuals[:train_size], dtype=np.float64),
        val_warm_residual=np.asarray(warm_residuals[train_size:], dtype=np.float64),
        val_oracle_residual=np.asarray(oracle_residuals[train_size:], dtype=np.float64),
        u_mean=np.asarray(u_mean, dtype=np.float32),
        u_std=np.asarray(u_std, dtype=np.float32),
        v_mean=np.asarray(v_mean, dtype=np.float32),
        v_std=np.asarray(v_std, dtype=np.float32),
        kappa=np.asarray(config.benchmark.kappa, dtype=np.float32),
        nx=np.asarray(config.benchmark.nx, dtype=np.int32),
    )
    return out_path


class OracleSolutionDataset(Dataset):
    """Torch dataset wrapping normalized forcing/solution pairs."""

    def __init__(self, dataset_path: str | Path, split: str = "train") -> None:
        data = np.load(dataset_path)
        self.split = split
        self.u_phys = torch.from_numpy(data[f"{split}_u"]).float()
        self.v_phys = torch.from_numpy(data[f"{split}_v"]).float()
        self.warm_phys = torch.from_numpy(data[f"{split}_warm"]).float()
        self.x = torch.from_numpy(data["x"]).float()
        self.stats = NormalizationStats.from_npz(data)

    def __len__(self) -> int:
        return self.u_phys.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        u_phys = self.u_phys[index]
        v_phys = self.v_phys[index]
        warm_phys = self.warm_phys[index]
        return {
            "u": self.stats.normalize_u(u_phys).unsqueeze(0),
            "v": self.stats.normalize_v(v_phys).unsqueeze(0),
            "u_phys": u_phys,
            "v_phys": v_phys,
            "warm_phys": warm_phys,
        }
