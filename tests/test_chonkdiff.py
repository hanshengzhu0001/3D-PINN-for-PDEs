"""Unit tests for the diffusion-first nonlinear elliptic utilities."""

import tempfile
import unittest
from pathlib import Path

import torch

from chonkdiff.benchmark import NonlinearElliptic1D
from chonkdiff.config import ExperimentConfig
from chonkdiff.dataset import NormalizationStats, generate_oracle_dataset
from chonkdiff.diffusion import GaussianDiffusion1D
from chonkdiff.guidance import PhysicsGuidance
from chonkdiff.model import ConditionalDiffusionCNN
from chonkdiff.oracle import lm_project


class ChonkDiffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()
        self.config.benchmark.dataset.train_size = 4
        self.config.benchmark.dataset.val_size = 2
        self.config.benchmark.dataset.out_path = str(
            Path(tempfile.gettempdir()) / "chonkdiff_test_dataset.npz"
        )
        self.config.training.epochs = 1
        self.benchmark = NonlinearElliptic1D(self.config.benchmark)

    def test_minus_laplacian_annihilates_constants(self) -> None:
        values = torch.ones(self.config.benchmark.nx, dtype=torch.float64)
        laplacian = self.benchmark.apply_minus_laplacian(values)
        self.assertTrue(torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1.0e-10))

    def test_jacobian_matches_finite_difference(self) -> None:
        forcing = self.benchmark.sample_forcing(1, seed=7).u.squeeze(0)
        state = torch.randn(self.config.benchmark.nx, dtype=torch.float64) * 0.05
        direction = torch.randn(self.config.benchmark.nx, dtype=torch.float64)
        direction = direction / torch.linalg.vector_norm(direction)
        epsilon = 1.0e-6

        finite_difference = (
            self.benchmark.residual(forcing, state + epsilon * direction)
            - self.benchmark.residual(forcing, state - epsilon * direction)
        ) / (2.0 * epsilon)
        jacobian_vector = self.benchmark.jacobian_matrix(state) @ direction
        self.assertTrue(torch.allclose(finite_difference, jacobian_vector, atol=1.0e-5, rtol=1.0e-4))

    def test_projector_reduces_residual(self) -> None:
        forcing = self.benchmark.sample_forcing(1, seed=11).u.squeeze(0)
        result = lm_project(
            self.benchmark,
            forcing,
            torch.zeros_like(forcing),
            self.config.oracle,
            max_iterations=6,
        )
        self.assertGreater(result.residual_history[0], result.residual_history[-1])
        self.assertLessEqual(len(result.residual_history), 6)
        self.assertEqual(len(result.lambda_history), len(result.residual_history))
        self.assertEqual(len(result.alpha_history), len(result.residual_history))

    def test_dataset_generation_and_model_shapes(self) -> None:
        dataset_path = generate_oracle_dataset(self.config, force=True)
        self.assertTrue(Path(dataset_path).exists())

        model = ConditionalDiffusionCNN(self.config.model)
        diffusion = GaussianDiffusion1D(self.config.diffusion)
        noisy = torch.randn(2, 1, self.config.benchmark.nx)
        forcing = torch.randn(2, 1, self.config.benchmark.nx)
        timesteps = torch.tensor([1, 3], dtype=torch.long)
        predicted = model(noisy, forcing, timesteps)
        self.assertEqual(predicted.shape, noisy.shape)

        sampled = diffusion.q_sample(noisy, timesteps, torch.zeros_like(noisy))
        self.assertEqual(sampled.shape, noisy.shape)

    def test_guidance_start_fraction_defers_correction(self) -> None:
        stats = NormalizationStats(
            u_mean=torch.tensor(0.0),
            u_std=torch.tensor(1.0),
            v_mean=torch.tensor(0.0),
            v_std=torch.tensor(1.0),
        )
        guidance = PhysicsGuidance(
            self.benchmark,
            stats,
            total_timesteps=10,
            mode="jtf",
            strength=1.0e-2,
            start_fraction=0.5,
        )
        forcing = self.benchmark.sample_forcing(1, seed=3).u.to(torch.float32)
        state = torch.zeros((1, 1, self.config.benchmark.nx), dtype=torch.float32)

        inactive = guidance(state, forcing.unsqueeze(1), torch.tensor([9], dtype=torch.long))
        active = guidance(state, forcing.unsqueeze(1), torch.tensor([0], dtype=torch.long))

        self.assertTrue(torch.allclose(inactive, state))
        self.assertFalse(torch.allclose(active, state))

    def test_gauss_newton_guidance_reduces_residual(self) -> None:
        stats = NormalizationStats(
            u_mean=torch.tensor(0.0),
            u_std=torch.tensor(1.0),
            v_mean=torch.tensor(0.0),
            v_std=torch.tensor(1.0),
        )
        guidance = PhysicsGuidance(
            self.benchmark,
            stats,
            total_timesteps=10,
            mode="gn",
            strength=1.0e-2,
            start_fraction=0.0,
            lm_lambda=1.0e-3,
        )
        forcing = self.benchmark.sample_forcing(1, seed=17).u.to(torch.float32)
        state = torch.zeros((1, 1, self.config.benchmark.nx), dtype=torch.float32)

        before = self.benchmark.residual_norm(forcing.to(torch.float64), state.squeeze(1).to(torch.float64))
        corrected = guidance(state, forcing.unsqueeze(1), torch.tensor([0], dtype=torch.long))
        after = self.benchmark.residual_norm(
            forcing.to(torch.float64),
            corrected.squeeze(1).to(torch.float64),
        )

        self.assertLess(float(after.item()), float(before.item()))


if __name__ == "__main__":
    unittest.main()
