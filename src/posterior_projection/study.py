from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectionCase:
    schedule: str
    order: str

    @property
    def name(self) -> str:
        return f"{self.schedule}__{self.order}"


def materialize_schedule(
    schedule: str,
    num_steps: int,
    late_start_fraction: float = 0.7,
) -> tuple[int, ...]:
    """Turn a static schedule name into explicit sample-step indices."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if schedule == "none":
        return ()
    if schedule == "final_only":
        return (num_steps - 1,)
    if schedule == "every_step":
        return tuple(range(num_steps))
    if schedule == "every_2":
        return tuple(range(0, num_steps, 2))
    if schedule == "every_5":
        return tuple(range(0, num_steps, 5))
    if schedule == "late_only":
        start = max(0, int(num_steps * late_start_fraction))
        return tuple(range(start, num_steps))
    if schedule == "adaptive_residual":
        raise ValueError("adaptive_residual is dynamic and should use should_project()")
    raise ValueError(f"unknown schedule: {schedule}")


def should_project(
    schedule: str,
    step_index: int,
    num_steps: int,
    residual_norm: float | None = None,
    late_start_fraction: float = 0.7,
    adaptive_threshold: float = 1.0e-2,
    adaptive_warmup_fraction: float = 0.5,
) -> bool:
    """Decide whether a schedule triggers projection at this sample step."""
    if schedule == "adaptive_residual":
        if residual_norm is None:
            raise ValueError("adaptive_residual schedule requires a residual_norm value")
        warmup_start = max(0, int(num_steps * adaptive_warmup_fraction))
        return step_index >= warmup_start and residual_norm > adaptive_threshold
    return step_index in materialize_schedule(
        schedule,
        num_steps=num_steps,
        late_start_fraction=late_start_fraction,
    )


def build_default_cases() -> list[ProjectionCase]:
    cases = [ProjectionCase(schedule="none", order="none")]
    schedules = (
        "final_only",
        "every_step",
        "every_2",
        "every_5",
        "late_only",
        "adaptive_residual",
    )
    orders = ("first_order", "gauss_newton")
    cases.extend(
        ProjectionCase(schedule=schedule, order=order)
        for schedule in schedules
        for order in orders
    )
    return cases
