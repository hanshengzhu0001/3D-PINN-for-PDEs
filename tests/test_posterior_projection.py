from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from posterior_projection.config import ExperimentConfig
from posterior_projection.dataset import JointStateDataset, sample_observation_mask
from posterior_projection.flow import euler_sample, flow_matching_loss
from posterior_projection.model import FlowMatchingFNO1D
from posterior_projection.pipeline import PosteriorProjectionPipeline
from posterior_projection.problem import JointPosteriorProblem
from posterior_projection.projection import first_order_project, second_order_project
from posterior_projection.study import build_default_cases, materialize_schedule, should_project
from posterior_projection.train import train


DATASET_PATH = "data/chonkdiff_elliptic_dataset.npz"


class PosteriorProjectionDatasetTests(unittest.TestCase):
    def test_joint_state_dataset_shapes(self) -> None:
        dataset = JointStateDataset(DATASET_PATH, split="train", max_samples=2)
        sample = dataset[0]
        self.assertEqual(tuple(sample["state"].shape), (2, 63))
        self.assertEqual(tuple(sample["obs_mask"].shape), (63,))
        self.assertEqual(tuple(sample["obs_v"].shape), (63,))

    def test_observation_mask_is_reproducible(self) -> None:
        mask_a = sample_observation_mask(63, 0.1, seed=11)
        mask_b = sample_observation_mask(63, 0.1, seed=11)
        mask_c = sample_observation_mask(63, 0.1, seed=12)
        self.assertTrue(torch.equal(mask_a, mask_b))
        self.assertFalse(torch.equal(mask_a, mask_c))


class PosteriorProjectionProblemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = JointPosteriorProblem(ExperimentConfig().problem)
        self.dataset = JointStateDataset(DATASET_PATH, split="train", max_samples=1)
        sample = self.dataset[0]
        self.state = self.dataset.stats.denormalize_state(sample["state"]).to(torch.float64)

    def test_joint_jacobian_matches_finite_difference(self) -> None:
        eps = 1.0e-6
        residual = self.problem.residual_from_state(self.state)
        jacobian = self.problem.joint_jacobian_from_state(self.state)
        finite_difference = []
        flat_state = self.state.reshape(-1)
        for index in range(flat_state.numel()):
            perturb = torch.zeros_like(flat_state)
            perturb[index] = eps
            plus = self.problem.residual_from_state((flat_state + perturb).reshape_as(self.state))
            minus = self.problem.residual_from_state((flat_state - perturb).reshape_as(self.state))
            finite_difference.append(((plus - minus) / (2.0 * eps)).unsqueeze(-1))
        fd_matrix = torch.cat(finite_difference, dim=-1)
        self.assertLess(float(torch.max(torch.abs(jacobian - fd_matrix)).item()), 2.5e-3)

    def test_first_order_projection_reduces_residual(self) -> None:
        noisy_state = self.state.clone()
        noisy_state[0] += 0.2
        before = float(self.problem.residual_norm_from_state(noisy_state).item())
        projected = first_order_project(self.problem, noisy_state)
        after = float(self.problem.residual_norm_from_state(projected.state).item())
        self.assertLess(after, before)

    def test_second_order_projection_is_stronger(self) -> None:
        noisy_state = self.state.clone()
        noisy_state[1] += 0.3
        config = ExperimentConfig().projection
        first = first_order_project(self.problem, noisy_state)
        second = second_order_project(self.problem, noisy_state, config=config)
        first_residual = float(self.problem.residual_norm_from_state(first.state).item())
        second_residual = float(self.problem.residual_norm_from_state(second.state).item())
        self.assertLessEqual(second_residual, first_residual + 1.0e-8)


class PosteriorProjectionScheduleTests(unittest.TestCase):
    def test_materialize_schedule_every_five(self) -> None:
        self.assertEqual(materialize_schedule("every_5", 12), (0, 5, 10))

    def test_adaptive_schedule_trigger(self) -> None:
        self.assertFalse(
            should_project(
                "adaptive_residual",
                step_index=2,
                num_steps=10,
                residual_norm=1.0,
                adaptive_threshold=1.0e-2,
                adaptive_warmup_fraction=0.5,
            )
        )
        self.assertTrue(
            should_project(
                "adaptive_residual",
                step_index=7,
                num_steps=10,
                residual_norm=1.0,
                adaptive_threshold=1.0e-2,
                adaptive_warmup_fraction=0.5,
            )
        )

    def test_build_default_cases_contains_baseline_and_orders(self) -> None:
        names = {case.name for case in build_default_cases()}
        self.assertIn("none__none", names)
        self.assertIn("adaptive_residual__gauss_newton", names)


class PosteriorProjectionFlowTests(unittest.TestCase):
    def test_flow_matching_loss_is_finite(self) -> None:
        config = ExperimentConfig()
        model = FlowMatchingFNO1D(config.model)
        batch = torch.randn(4, 2, config.problem.nx)
        loss, metrics = flow_matching_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_total", metrics)

    def test_euler_sampler_is_deterministic(self) -> None:
        config = ExperimentConfig()
        model = FlowMatchingFNO1D(config.model)
        torch.manual_seed(123)
        init = torch.randn(1, 2, config.problem.nx)
        sample_a, traj_a = euler_sample(model, init, num_steps=6, return_trajectory=True)
        sample_b, traj_b = euler_sample(model, init, num_steps=6, return_trajectory=True)
        self.assertTrue(torch.allclose(sample_a, sample_b))
        self.assertTrue(torch.allclose(traj_a, traj_b))


class PosteriorProjectionSmokeTests(unittest.TestCase):
    def test_end_to_end_smoke(self) -> None:
        config = ExperimentConfig()
        config.training.epochs = 1
        config.training.batch_size = 4
        config.training.max_train_samples = 8
        config.training.max_val_samples = 4
        config.model.hidden_channels = 16
        config.model.num_fno_layers = 2
        config.model.modes = 8
        config.sampling.num_steps = 8
        config.projection.max_projection_steps = 3
        config.projection.final_cleanup_iterations = 3
        config.evaluation.num_eval_samples = 2
        config.evaluation.num_observation_seeds = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            config.training.checkpoint_dir = tmpdir
            checkpoint = train(config)
            self.assertTrue(Path(checkpoint).exists())

            pipeline = PosteriorProjectionPipeline.from_checkpoint(checkpoint)
            dataset = JointStateDataset(
                DATASET_PATH,
                split="val",
                observed_fraction=config.posterior.observed_fraction,
                observation_noise_std=config.posterior.observation_noise_std,
                observation_pattern=config.posterior.observation_pattern,
                seed=config.posterior.seed,
                max_samples=1,
            )
            sample = dataset[0]
            output = pipeline.sample_posterior(
                obs_mask=sample["obs_mask"],
                obs_v=sample["obs_v"],
                schedule="every_2",
                order="gauss_newton",
                sample_seed=321,
            )
            self.assertGreaterEqual(output.projection_calls, 1)
            self.assertEqual(output.trajectory.shape[0], config.sampling.num_steps + 1)
            self.assertEqual(tuple(output.pre_cleanup_state_phys.shape), (2, 63))


if __name__ == "__main__":
    unittest.main()
