"""Evaluation entrypoint for diffusion-only, guided, and projected solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .benchmark import NonlinearElliptic1D
from .dataset import OracleSolutionDataset, generate_oracle_dataset
from .pipeline import DiffusionProjectorSolver


def evaluate(
    checkpoint_path: str,
    num_samples: int | None = None,
    guidance_mode: Optional[str] = None,
    guidance_strength: Optional[float] = None,
    guidance_start_fraction: Optional[float] = None,
    guidance_lambda: Optional[float] = None,
    projector_iterations: Optional[int] = None,
    projector_tolerance: Optional[float] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = DiffusionProjectorSolver.from_checkpoint(checkpoint_path, device=device)
    dataset_path = generate_oracle_dataset(solver.config)
    val_dataset = OracleSolutionDataset(dataset_path, split="val")
    benchmark = NonlinearElliptic1D(solver.config.benchmark, device=device)

    n_eval = num_samples or min(solver.config.sampling.num_eval_samples, len(val_dataset))
    active_guidance = (
        solver.config.sampling.guidance_mode if guidance_mode is None else guidance_mode
    )
    metrics = {
        "diffusion_error": [],
        "guided_error": [],
        "projected_error": [],
        "diffusion_residual": [],
        "guided_residual": [],
        "projected_residual": [],
        "projection_steps": [],
        "projection_converged_rate": [],
        "projection_final_lambda": [],
        "projection_final_alpha": [],
    }

    for index in range(n_eval):
        sample = val_dataset[index]
        forcing_u = sample["u_phys"].to(device)
        oracle_v = sample["v_phys"].to(device, dtype=torch.float64).unsqueeze(0)
        seed = solver.config.sampling.seed + index
        init_generator = torch.Generator(device=device)
        init_generator.manual_seed(seed)
        diffusion_generator = torch.Generator(device=device)
        diffusion_generator.manual_seed(seed)
        guided_generator = torch.Generator(device=device)
        guided_generator.manual_seed(seed)
        projected_generator = torch.Generator(device=device)
        projected_generator.manual_seed(seed)
        x_init = torch.randn(
            (1, 1, solver.config.benchmark.nx), generator=init_generator, device=device
        )

        diffusion_v, _ = solver.sample(
            forcing_u,
            guidance_mode="none",
            projector=False,
            x_init=x_init,
            generator=diffusion_generator,
        )
        guided_v, _ = solver.sample(
            forcing_u,
            guidance_mode=active_guidance,
            guidance_strength=guidance_strength,
            guidance_start_fraction=guidance_start_fraction,
            guidance_lambda=guidance_lambda,
            projector=False,
            x_init=x_init,
            generator=guided_generator,
        )
        projected_v, projector_result = solver.sample(
            forcing_u,
            guidance_mode=active_guidance,
            guidance_strength=guidance_strength,
            guidance_start_fraction=guidance_start_fraction,
            guidance_lambda=guidance_lambda,
            projector=True,
            projector_iterations=projector_iterations,
            projector_tolerance=projector_tolerance,
            x_init=x_init,
            generator=projected_generator,
        )

        forcing_batch = forcing_u.unsqueeze(0).to(torch.float64)
        metrics["diffusion_error"].append(
            float(benchmark.relative_l2_error(diffusion_v, oracle_v).item())
        )
        metrics["guided_error"].append(float(benchmark.relative_l2_error(guided_v, oracle_v).item()))
        metrics["projected_error"].append(
            float(benchmark.relative_l2_error(projected_v, oracle_v).item())
        )
        metrics["diffusion_residual"].append(
            float(benchmark.residual_norm(forcing_batch, diffusion_v).item())
        )
        metrics["guided_residual"].append(
            float(benchmark.residual_norm(forcing_batch, guided_v).item())
        )
        metrics["projected_residual"].append(
            float(benchmark.residual_norm(forcing_batch, projected_v).item())
        )
        metrics["projection_steps"].append(
            0 if projector_result is None else len(projector_result.residual_history)
        )
        metrics["projection_converged_rate"].append(
            0.0 if projector_result is None else float(projector_result.converged)
        )
        metrics["projection_final_lambda"].append(
            0.0 if projector_result is None else float(projector_result.lambda_history[-1])
        )
        metrics["projection_final_alpha"].append(
            0.0 if projector_result is None else float(projector_result.alpha_history[-1])
        )

    summary = {key: float(sum(values) / max(len(values), 1)) for key, values in metrics.items()}

    for key in [
        "diffusion_error",
        "guided_error",
        "projected_error",
        "diffusion_residual",
        "guided_residual",
        "projected_residual",
        "projection_steps",
    ]:
        values = torch.tensor(metrics[key], dtype=torch.float64)
        summary[f"{key}_median"] = float(values.median().item())
        summary[f"{key}_p95"] = float(torch.quantile(values, 0.95).item())
        summary[f"{key}_max"] = float(values.max().item())

    summary["projected_error_count_gt_1e-6"] = float(
        sum(value > 1.0e-6 for value in metrics["projected_error"])
    )
    summary["projected_residual_count_gt_1e-10"] = float(
        sum(value > 1.0e-10 for value in metrics["projected_residual"])
    )
    summary["projected_residual_count_gt_1e-8"] = float(
        sum(value > 1.0e-8 for value in metrics["projected_residual"])
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of validation samples to evaluate",
    )
    parser.add_argument(
        "--guidance-mode",
        default=None,
        help="Override guidance mode: none, jtf, residual, or gn",
    )
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=None,
        help="Override per-step guidance strength used during reverse sampling",
    )
    parser.add_argument(
        "--guidance-start-fraction",
        type=float,
        default=None,
        help="Override reverse-time fraction at which PDE guidance starts ramping in",
    )
    parser.add_argument(
        "--guidance-lambda",
        type=float,
        default=None,
        help="Override the damping used by Gauss-Newton reverse guidance",
    )
    parser.add_argument(
        "--projector-iterations",
        type=int,
        default=None,
        help="Override the number of final LM/NK cleanup iterations",
    )
    parser.add_argument(
        "--projector-tolerance",
        type=float,
        default=None,
        help="Override the residual tolerance used by the final projector",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for writing metrics as JSON",
    )
    args = parser.parse_args()

    metrics = evaluate(
        args.checkpoint,
        args.num_samples,
        args.guidance_mode,
        args.guidance_strength,
        args.guidance_start_fraction,
        args.guidance_lambda,
        args.projector_iterations,
        args.projector_tolerance,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
