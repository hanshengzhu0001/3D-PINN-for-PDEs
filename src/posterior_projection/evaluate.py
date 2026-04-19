from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import JointStateDataset
from .pipeline import PosteriorProjectionPipeline
from .problem import JointPosteriorProblem
from .study import build_default_cases


def _observation_error(v_pred: torch.Tensor, obs_v: torch.Tensor, obs_mask: torch.Tensor) -> float:
    mask = obs_mask.to(v_pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return float(torch.sqrt(((mask * (v_pred - obs_v)) ** 2).sum() / denom).item())


def evaluate(
    checkpoint_path: str,
    json_out: str | None = None,
    csv_out: str | None = None,
    num_samples: int | None = None,
    num_observation_seeds: int | None = None,
) -> list[dict[str, float | str]]:
    pipeline = PosteriorProjectionPipeline.from_checkpoint(checkpoint_path)
    config = pipeline.config
    if num_samples is not None:
        config.evaluation.num_eval_samples = num_samples
    if num_observation_seeds is not None:
        config.evaluation.num_observation_seeds = num_observation_seeds
    dataset = JointStateDataset(
        config.problem.reference_dataset_path,
        split="val",
        observed_fraction=config.posterior.observed_fraction,
        observation_noise_std=config.posterior.observation_noise_std,
        observation_pattern=config.posterior.observation_pattern,
        seed=config.posterior.seed,
        max_samples=config.evaluation.num_eval_samples,
    )
    problem = JointPosteriorProblem(config.problem)
    cases = build_default_cases()

    summaries: list[dict[str, float | str]] = []
    per_case: dict[str, list[dict[str, float]]] = {case.name: [] for case in cases}

    for sample_index in range(min(config.evaluation.num_eval_samples, len(dataset))):
        sample = dataset[sample_index]
        u_true = sample["u_phys"].to(torch.float64)
        v_true = sample["v_phys"].to(torch.float64)
        obs_mask = sample["obs_mask"]
        obs_v = sample["obs_v"]

        for obs_seed in range(config.evaluation.num_observation_seeds):
            sample_seed = config.evaluation.sample_seed_base + 10_000 * obs_seed + sample_index
            for case in cases:
                output = pipeline.sample_posterior(
                    obs_mask=obs_mask,
                    obs_v=obs_v,
                    schedule=case.schedule,
                    order=case.order,
                    sample_seed=sample_seed,
                )
                baseline_trajectory = output.baseline_trajectory.to(torch.float64)
                trajectory = output.trajectory.to(torch.float64)
                pre_state = output.pre_cleanup_state_phys
                post_state = output.post_cleanup_state_phys

                traj_deviation = float(
                    ((trajectory - baseline_trajectory) ** 2).sum().item()
                )
                pre_u, pre_v = pre_state[0], pre_state[1]
                post_u, post_v = post_state[0], post_state[1]

                posterior_quality = float(
                    (
                        problem.relative_l2(pre_u.unsqueeze(0), u_true.unsqueeze(0))
                        + problem.relative_l2(pre_v.unsqueeze(0), v_true.unsqueeze(0))
                    ).mean().item()
                ) + _observation_error(
                    pipeline.stats.normalize_v(pre_v.float()),
                    obs_v,
                    obs_mask,
                )

                row = {
                    "posterior_quality": posterior_quality,
                    "physical_consistency": float(problem.residual_norm_from_state(pre_state).item()),
                    "runtime": output.total_runtime_seconds,
                    "trajectory_stability": traj_deviation,
                    "u_error": float(problem.relative_l2(pre_u.unsqueeze(0), u_true.unsqueeze(0)).item()),
                    "v_error": float(problem.relative_l2(pre_v.unsqueeze(0), v_true.unsqueeze(0)).item()),
                    "obs_error": _observation_error(
                        pipeline.stats.normalize_v(pre_v.float()),
                        obs_v,
                        obs_mask,
                    ),
                    "ce_squared": float((problem.residual_from_state(pre_state) ** 2).sum().item()),
                    "projection_time": output.projection_time_seconds,
                    "projection_calls": float(output.projection_calls),
                    "projection_iterations": float(output.projection_iterations),
                    "post_u_error": float(problem.relative_l2(post_u.unsqueeze(0), u_true.unsqueeze(0)).item()),
                    "post_v_error": float(problem.relative_l2(post_v.unsqueeze(0), v_true.unsqueeze(0)).item()),
                    "post_ce": float(problem.residual_norm_from_state(post_state).item()),
                }
                per_case[case.name].append(row)

    for case in cases:
        values = per_case[case.name]
        summary: dict[str, float | str] = {"schedule": case.schedule, "order": case.order}
        for key in values[0].keys():
            summary[key] = float(np.mean([row[key] for row in values]))
        summaries.append(summary)

    for metric_name in ["posterior_quality", "physical_consistency", "runtime", "trajectory_stability"]:
        ranking = sorted(summaries, key=lambda row: float(row[metric_name]))
        for rank, row in enumerate(ranking, start=1):
            row[f"rank_{metric_name}"] = float(rank)

    if json_out is not None:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if csv_out is not None:
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json-out", default="outputs/posterior_projection/eval_summary.json")
    parser.add_argument("--csv-out", default="outputs/posterior_projection/eval_summary.csv")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-observation-seeds", type=int, default=None)
    args = parser.parse_args()

    rows = evaluate(
        args.checkpoint,
        json_out=args.json_out,
        csv_out=args.csv_out,
        num_samples=args.num_samples,
        num_observation_seeds=args.num_observation_seeds,
    )
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
