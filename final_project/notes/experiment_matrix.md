# Experiment Matrix

This note turns the proposal into an implementation checklist.

## Variables We Control

- `projection schedule`
  - `none`
  - `final_only`
  - `every_step`
  - `every_2`
  - `every_5`
  - `late_only`
  - `adaptive_residual`
- `projection order`
  - `first_order`
  - `gauss_newton`

## Targets We Measure

- `posterior quality`
  - negative log posterior surrogate
  - reconstruction / relative error when oracle targets are available
- `physical consistency`
  - residual norm
  - constraint error
- `runtime`
  - total wall-clock
  - projection-only wall-clock
  - number of projection calls
- `trajectory stability`
  - squared deviation from the unconstrained sampling path

## First Benchmark

- problem: `-Delta v + kappa v^3 = u`
- `kappa = 50`
- periodic grid with `N_x = 63`
- oracle and dataset already exist under `src/chonkdiff`
- but the new study should not assume the `chonkdiff` learner itself is the final-project model

## Recommended Order Of Work

1. Start from the existing `chonkdiff` benchmark and oracle only as a reference backend.
2. Add partial-observation masking so the task becomes a posterior-sampling problem.
3. Build the new posterior-sampling pipeline in its own package.
4. Wrap the sampler with schedule-controlled projection.
5. Compare first-order vs Gauss-Newton projection under the same observation pattern.
6. Report the four core targets together, not just residual error.
