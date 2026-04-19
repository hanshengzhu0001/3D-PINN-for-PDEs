from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig, load_config, load_config_dict
from .dataset import JointNormalizationStats
from .flow import euler_sample
from .model import FlowMatchingFNO1D
from .problem import JointPosteriorProblem
from .projection import ProjectionResult, first_order_project, second_order_project
from .study import should_project


@dataclass(slots=True)
class SampleOutput:
    trajectory: torch.Tensor
    baseline_trajectory: torch.Tensor
    pre_cleanup_state_phys: torch.Tensor
    post_cleanup_state_phys: torch.Tensor
    projection_calls: int
    projection_time_seconds: float
    projection_iterations: int
    cleanup_result: ProjectionResult | None
    total_runtime_seconds: float


class PosteriorProjectionPipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        model: FlowMatchingFNO1D,
        stats: JointNormalizationStats,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.stats = stats.to(self.device)
        self.problem = JointPosteriorProblem(config.problem, device=self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
    ) -> "PosteriorProjectionPipeline":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        if "config_dict" in checkpoint:
            config = load_config_dict(checkpoint["config_dict"])
        else:
            config = load_config(checkpoint["config_path"])
        model = FlowMatchingFNO1D(config.model)
        model.load_state_dict(checkpoint["model_state"])
        stats = JointNormalizationStats(
            u_mean=torch.tensor(checkpoint["stats"]["u_mean"], dtype=torch.float32),
            u_std=torch.tensor(checkpoint["stats"]["u_std"], dtype=torch.float32),
            v_mean=torch.tensor(checkpoint["stats"]["v_mean"], dtype=torch.float32),
            v_std=torch.tensor(checkpoint["stats"]["v_std"], dtype=torch.float32),
        )
        return cls(config=config, model=model, stats=stats, device=device)

    def _observation_guidance(
        self,
        state: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        state_for_grad = state.detach().clone().requires_grad_(True)
        loss = self.problem.observation_loss(state_for_grad, obs_mask, obs_v).mean()
        grad = torch.autograd.grad(loss, state_for_grad)[0]
        return (state_for_grad - self.config.sampling.observation_guidance_strength * dt * grad).detach()

    def _apply_projection(
        self,
        state: torch.Tensor,
        order: str,
    ) -> ProjectionResult:
        state_phys = self.stats.denormalize_state(state).squeeze(0).to(torch.float64)
        if order == "first_order":
            result = first_order_project(
                self.problem,
                state_phys,
                damping=self.config.projection.first_order_lambda,
            )
        elif order == "gauss_newton":
            result = second_order_project(
                self.problem,
                state_phys,
                config=self.config.projection,
                max_iterations=self.config.projection.max_projection_steps,
            )
        else:
            raise ValueError(f"unsupported projection order: {order}")
        normalized = self.stats.normalize_state(result.state[0].float(), result.state[1].float()).unsqueeze(0)
        return ProjectionResult(
            state=normalized.to(self.device),
            residual_history=result.residual_history,
            lambda_history=result.lambda_history,
            alpha_history=result.alpha_history,
            converged=result.converged,
            elapsed_seconds=result.elapsed_seconds,
        )

    def sample_posterior(
        self,
        obs_mask: torch.Tensor,
        obs_v: torch.Tensor,
        schedule: str = "none",
        order: str = "none",
        sample_seed: int = 123,
        return_baseline: bool = True,
    ) -> SampleOutput:
        self.model.eval()
        start_time = time.perf_counter()
        num_steps = self.config.sampling.num_steps
        obs_mask = obs_mask.to(self.device, dtype=torch.float32).unsqueeze(0)
        obs_v = obs_v.to(self.device, dtype=torch.float32).unsqueeze(0)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(sample_seed)
        initial_state = torch.randn((1, 2, self.config.problem.nx), generator=generator, device=self.device)

        def baseline_guidance(state: torch.Tensor, _: int, dt: float) -> torch.Tensor:
            return self._observation_guidance(state, obs_mask, obs_v, dt)

        _, baseline_trajectory = euler_sample(
            self.model,
            initial_state=initial_state,
            num_steps=num_steps,
            guidance_fn=baseline_guidance,
            return_trajectory=True,
        )
        if baseline_trajectory is None:
            raise RuntimeError("baseline trajectory was not recorded")

        projection_calls = 0
        projection_time = 0.0
        projection_iterations = 0

        def guided_sampler(state: torch.Tensor, step_index: int, dt: float) -> torch.Tensor:
            nonlocal projection_calls, projection_time, projection_iterations
            updated = self._observation_guidance(state, obs_mask, obs_v, dt)
            if schedule == "none" or order == "none":
                return updated

            state_phys = self.stats.denormalize_state(updated).squeeze(0).to(torch.float64)
            residual_norm = float(self.problem.residual_norm_from_state(state_phys).item())
            if not should_project(
                schedule=schedule,
                step_index=step_index,
                num_steps=num_steps,
                residual_norm=residual_norm,
                late_start_fraction=self.config.projection.late_start_fraction,
                adaptive_threshold=self.config.projection.adaptive_residual_threshold,
                adaptive_warmup_fraction=self.config.projection.adaptive_warmup_fraction,
            ):
                return updated

            result = self._apply_projection(updated, order)
            projection_calls += 1
            projection_time += result.elapsed_seconds
            projection_iterations += result.iterations if order == "gauss_newton" else 1
            return result.state.to(self.device, dtype=updated.dtype)

        pre_cleanup_state, trajectory = euler_sample(
            self.model,
            initial_state=initial_state,
            num_steps=num_steps,
            guidance_fn=guided_sampler,
            return_trajectory=True,
        )
        if trajectory is None:
            raise RuntimeError("posterior trajectory was not recorded")

        pre_cleanup_phys = self.stats.denormalize_state(pre_cleanup_state).squeeze(0).to(torch.float64)
        cleanup_result = None
        post_cleanup_phys = pre_cleanup_phys
        if self.config.projection.final_cleanup:
            cleanup_result = second_order_project(
                self.problem,
                pre_cleanup_phys,
                config=self.config.projection,
                max_iterations=self.config.projection.final_cleanup_iterations,
                tolerance=self.config.projection.final_cleanup_tolerance,
            )
            projection_time += cleanup_result.elapsed_seconds
            projection_iterations += cleanup_result.iterations
            post_cleanup_phys = cleanup_result.state

        total_runtime = time.perf_counter() - start_time
        return SampleOutput(
            trajectory=trajectory.squeeze(0).cpu(),
            baseline_trajectory=baseline_trajectory.squeeze(0).cpu() if return_baseline else torch.empty(0),
            pre_cleanup_state_phys=pre_cleanup_phys.cpu(),
            post_cleanup_state_phys=post_cleanup_phys.cpu(),
            projection_calls=projection_calls,
            projection_time_seconds=projection_time,
            projection_iterations=projection_iterations,
            cleanup_result=cleanup_result,
            total_runtime_seconds=total_runtime,
        )
