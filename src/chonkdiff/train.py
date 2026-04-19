"""Training entrypoint for the conditional DDPM baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .benchmark import NonlinearElliptic1D
from .config import ExperimentConfig, load_config
from .dataset import OracleSolutionDataset, generate_oracle_dataset
from .diffusion import GaussianDiffusion1D
from .model import ConditionalDiffusionCNN


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def physics_weights_for_epoch(
    epoch: int, config: ExperimentConfig, diffusion_only: bool = False
) -> tuple[float, float]:
    if diffusion_only:
        return 0.0, 0.0
    progress = epoch / max(config.training.epochs - 1, 1)
    if progress < config.training.stage_a_fraction:
        return 0.0, 0.0
    if progress < config.training.stage_b_fraction:
        ratio = (progress - config.training.stage_a_fraction) / max(
            config.training.stage_b_fraction - config.training.stage_a_fraction, 1.0e-6
        )
        return (
            0.5 * config.training.pde_weight * ratio,
            0.5 * config.training.bc_weight * ratio,
        )
    ratio = (progress - config.training.stage_b_fraction) / max(
        1.0 - config.training.stage_b_fraction, 1.0e-6
    )
    return (
        config.training.pde_weight * (0.5 + 0.5 * ratio),
        config.training.bc_weight * (0.5 + 0.5 * ratio),
    )


def run_epoch(
    model: ConditionalDiffusionCNN,
    diffusion: GaussianDiffusion1D,
    loader: DataLoader,
    optimizer: AdamW | None,
    benchmark: NonlinearElliptic1D,
    pde_weight: float,
    bc_weight: float,
    gradient_clip: float,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss_total": 0.0,
        "loss_diff": 0.0,
        "loss_pde": 0.0,
        "loss_bc": 0.0,
    }
    total_items = 0
    stats = loader.dataset.stats.to(device)

    for batch in loader:
        u = batch["u"].to(device)
        v = batch["v"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        loss, metrics = diffusion.loss(
            model,
            x_start=v,
            forcing_u=u,
            benchmark=benchmark,
            stats=stats,
            pde_weight=pde_weight,
            bc_weight=bc_weight,
        )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

        batch_size = u.shape[0]
        for key, value in metrics.items():
            totals[key] += float(value) * batch_size
        total_items += batch_size

    return {key: value / max(total_items, 1) for key, value in totals.items()}


def train(
    config: ExperimentConfig,
    config_path: str,
    diffusion_only: bool = False,
    force_regenerate: bool = False,
) -> Path:
    set_training_seed(config.training.seed)
    dataset_path = generate_oracle_dataset(config, force=force_regenerate)
    train_dataset = OracleSolutionDataset(dataset_path, split="train")
    val_dataset = OracleSolutionDataset(dataset_path, split="val")
    train_generator = torch.Generator()
    train_generator.manual_seed(config.training.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionCNN(config.model).to(device)
    diffusion = GaussianDiffusion1D(config.diffusion).to(device)
    benchmark = NonlinearElliptic1D(config.benchmark, device=device, dtype=torch.float32)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_total_path = checkpoint_dir / "best_total.pt"
    best_diff_path = checkpoint_dir / "best_diff.pt"
    latest_path = checkpoint_dir / "latest.pt"

    best_val_total = float("inf")
    best_val_diff = float("inf")
    stats = train_dataset.stats

    print(
        f"device={device} train_size={len(train_dataset)} val_size={len(val_dataset)} "
        f"dataset={dataset_path} diffusion_only={diffusion_only} "
        f"periodic_bc_term=inactive seed={config.training.seed}"
    )

    for epoch in range(config.training.epochs):
        pde_weight, bc_weight = physics_weights_for_epoch(
            epoch, config, diffusion_only=diffusion_only
        )
        train_metrics = run_epoch(
            model,
            diffusion,
            train_loader,
            optimizer,
            benchmark,
            pde_weight,
            bc_weight,
            config.training.gradient_clip,
            device,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                diffusion,
                val_loader,
                None,
                benchmark,
                pde_weight,
                bc_weight,
                config.training.gradient_clip,
                device,
            )
        scheduler.step()
        train_loss = train_metrics["loss_total"]
        val_loss = val_metrics["loss_total"]

        checkpoint = {
            "model_state": model.state_dict(),
            "config_path": str(Path(config_path).resolve()),
            "config_dict": config.to_dict(),
            "diffusion_only": diffusion_only,
            "pde_weight": pde_weight,
            "bc_weight": bc_weight,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "stats": {
                "u_mean": float(stats.u_mean.item()),
                "u_std": float(stats.u_std.item()),
                "v_mean": float(stats.v_mean.item()),
                "v_std": float(stats.v_std.item()),
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, latest_path)
        if val_metrics["loss_total"] < best_val_total:
            best_val_total = val_metrics["loss_total"]
            torch.save(checkpoint, best_total_path)
        if val_metrics["loss_diff"] < best_val_diff:
            best_val_diff = val_metrics["loss_diff"]
            torch.save(checkpoint, best_diff_path)
            torch.save(checkpoint, best_path)

        print(
            f"epoch={epoch:03d} pde_weight={pde_weight:.2e} bc_weight_inactive={bc_weight:.2e} "
            f"train_total={train_metrics['loss_total']:.6e} train_diff={train_metrics['loss_diff']:.6e} "
            f"train_pde={train_metrics['loss_pde']:.6e} train_bc_inactive={train_metrics['loss_bc']:.6e} "
            f"val_total={val_metrics['loss_total']:.6e} val_diff={val_metrics['loss_diff']:.6e} "
            f"val_pde={val_metrics['loss_pde']:.6e} val_bc_inactive={val_metrics['loss_bc']:.6e}"
        )

    return best_diff_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/chonkdiff_elliptic.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--pde-weight", type=float, default=None, help="Override final PDE loss weight"
    )
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=None,
        help="Override final periodic BC loss weight (inactive for this benchmark)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override checkpoint directory"
    )
    parser.add_argument(
        "--diffusion-only",
        action="store_true",
        help="Disable PDE loss scheduling and train pure DDPM",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate the oracle dataset even if it already exists",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.pde_weight is not None:
        config.training.pde_weight = args.pde_weight
    if args.bc_weight is not None:
        config.training.bc_weight = args.bc_weight
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    best_path = train(
        config,
        args.config,
        diffusion_only=args.diffusion_only,
        force_regenerate=args.force_regenerate,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
