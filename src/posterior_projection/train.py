from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .dataset import JointStateDataset
from .flow import flow_matching_loss
from .model import FlowMatchingFNO1D


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: ExperimentConfig) -> Path:
    _set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = JointStateDataset(
        config.problem.reference_dataset_path,
        split="train",
        observed_fraction=config.posterior.observed_fraction,
        observation_noise_std=config.posterior.observation_noise_std,
        observation_pattern=config.posterior.observation_pattern,
        seed=config.posterior.seed,
        max_samples=config.training.max_train_samples,
    )
    val_dataset = JointStateDataset(
        config.problem.reference_dataset_path,
        split="val",
        observed_fraction=config.posterior.observed_fraction,
        observation_noise_std=config.posterior.observation_noise_std,
        observation_pattern=config.posterior.observation_pattern,
        seed=config.posterior.seed + 100_000,
        max_samples=config.training.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = FlowMatchingFNO1D(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    best_val = float("inf")

    print(
        f"device={device.type} train_size={len(train_dataset)} val_size={len(val_dataset)} "
        f"dataset={config.problem.reference_dataset_path}"
    )

    for epoch in range(config.training.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            state = batch["state"].to(device)
            loss, _ = flow_matching_loss(model, state)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.training.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"].to(device)
                loss, _ = flow_matching_loss(model, state)
                val_losses.append(float(loss.detach().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if epoch % config.training.log_every == 0:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

        checkpoint = {
            "config_dict": config.to_dict(),
            "model_state": model.state_dict(),
            "stats": {
                "u_mean": float(train_dataset.stats.u_mean.item()),
                "u_std": float(train_dataset.stats.u_std.item()),
                "v_mean": float(train_dataset.stats.v_mean.item()),
                "v_std": float(train_dataset.stats.v_std.item()),
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, latest_path)
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

    print(f"Best checkpoint: {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/posterior_projection.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_train_samples is not None:
        config.training.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        config.training.max_val_samples = args.max_val_samples

    best_path = train(config)
    print(json.dumps({"best_checkpoint": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
