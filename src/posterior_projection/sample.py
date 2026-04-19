from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import JointStateDataset
from .pipeline import PosteriorProjectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--schedule", default="none")
    parser.add_argument("--order", default="none")
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--npz-out", default=None)
    args = parser.parse_args()

    pipeline = PosteriorProjectionPipeline.from_checkpoint(args.checkpoint)
    dataset = JointStateDataset(
        pipeline.config.problem.reference_dataset_path,
        split=args.split,
        observed_fraction=pipeline.config.posterior.observed_fraction,
        observation_noise_std=pipeline.config.posterior.observation_noise_std,
        observation_pattern=pipeline.config.posterior.observation_pattern,
        seed=pipeline.config.posterior.seed,
    )
    sample = dataset[args.index]
    output = pipeline.sample_posterior(
        obs_mask=sample["obs_mask"],
        obs_v=sample["obs_v"],
        schedule=args.schedule,
        order=args.order,
        sample_seed=args.sample_seed,
    )

    payload = {
        "schedule": args.schedule,
        "order": args.order,
        "sample_seed": args.sample_seed,
        "projection_calls": output.projection_calls,
        "projection_time_seconds": output.projection_time_seconds,
        "projection_iterations": output.projection_iterations,
        "total_runtime_seconds": output.total_runtime_seconds,
    }
    print(json.dumps(payload, indent=2))

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.npz_out is not None:
        import numpy as np

        out_path = Path(args.npz_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            trajectory=output.trajectory.numpy(),
            baseline_trajectory=output.baseline_trajectory.numpy(),
            pre_cleanup_state=output.pre_cleanup_state_phys.numpy(),
            post_cleanup_state=output.post_cleanup_state_phys.numpy(),
        )


if __name__ == "__main__":
    main()
