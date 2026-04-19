# Final Project: Posterior Projection for PDE Generative Sampling

This folder is the clean workspace for the final project only. The project studies a specific question:

Given a generative sampler for PDE states, which projection schedule and which projection order are best for posterior quality, physical consistency, runtime, and trajectory stability?

The implementation for this project lives in [src/posterior_projection](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection). The older [src/chonkdiff](/Users/hanszhu/Desktop/3D_PINN/src/chonkdiff) code is reused only as a numerical backend for the nonlinear elliptic benchmark, oracle solves, and the reference dataset. It is not the final-project learner.

## 1. Scope

This final project uses one benchmark first:

- PDE: `-Delta v + kappa v^3 = u`
- boundary condition: periodic
- `kappa = 50`
- grid size: `N_x = 63`

The inverse task is:

- unknown: full forcing `u`
- observed: partial entries of the solution `v`
- target: posterior samples of the joint state `(u, v)` that both fit observations and satisfy the PDE

In this v1 implementation, the project does not try to estimate exact posterior log density. Instead, it compares projection strategies using practical surrogate metrics.

## 2. Current Environment Setup

### Python and packaging

- Python requirement: `>= 3.12` in [pyproject.toml](/Users/hanszhu/Desktop/3D_PINN/pyproject.toml)
- package root: `src/`
- most commands are run with `PYTHONPATH=src`

### Main dependencies used by the final project

From [requirements.txt](/Users/hanszhu/Desktop/3D_PINN/requirements.txt):

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `pyyaml`

The final-project pipeline itself is PyTorch-based. The JAX/Equinox dependencies in this repo belong to older PINN code and are not required conceptually for the `posterior_projection` implementation path.

### Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or, if you prefer the editable package route:

```bash
pip install -e .
```

### Data and outputs

- reference dataset: `data/chonkdiff_elliptic_dataset.npz`
- checkpoints default directory: `outputs/posterior_projection`
- evaluation outputs default directory: `outputs/posterior_projection`

These paths are intentionally gitignored, so the repo tracks code/config/docs, not generated artifacts.

If the reference dataset does not exist on a fresh clone, bootstrap it with:

```bash
PYTHONPATH=src python -m chonkdiff.generate_dataset \
  --config configs/chonkdiff_elliptic.yaml
```

## 3. Project Question

This project is motivated by a larger research problem:

Generative models can produce diverse samples for inverse PDE problems, but enforcing physics during sampling is delicate. Too little correction gives physically invalid samples. Too much correction can distort the trajectory, increase runtime, or collapse diversity.

Our concrete question is:

- which projection schedule is best
  - `none`
  - `final_only`
  - `every_step`
  - `every_2`
  - `every_5`
  - `late_only`
  - `adaptive_residual`
- which projection order is best
  - `first_order`
  - `gauss_newton`
- for which target
  - posterior quality
  - physical consistency
  - runtime
  - trajectory stability

The study is designed to answer not just whether projection helps, but where, when, and at what cost.

## 4. Mathematical Problem Definition

We work with the joint state

```text
x = [u, v] in R^(2 x N_x)
```

and the nonlinear elliptic constraint

```text
h(u, v) = -Delta v + kappa v^3 - u.
```

The inverse problem conditions on partial observations of `v`. If `M` is the observation mask and `y` is the observed data, then the observation-fit term is

```text
L_obs(x) = || M odot (v - y) ||_2^2 / max(1, |M|).
```

The project evaluates a projection strategy by four objectives:

```text
J_post  = quality of recovered (u, v) under oracle and observation-fit surrogates
J_phys  = || h(u, v) ||_2^2
J_time  = total runtime including projection cost
J_traj  = sum_k || x_k^(proj) - x_k^(base) ||_2^2
```

The research decision variable is:

```text
(S, q)
```

where:

- `S` = projection schedule
- `q` = projection order

The study asks how `(S, q)` changes the four objectives above.

## 5. Method Overview

### 5.1 Generative prior

We train an unconditional joint flow-matching prior over oracle samples `(u, v)`.

- model family: compact 1D FNO-style vector field
- implementation: [model.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/model.py)
- training objective: OT-style flow matching
- implementation: [flow.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/flow.py)

Training uses:

- `x_0 ~ N(0, I)`
- `x_1 ~ p_data(u, v)`
- `x_t = (1 - t) x_0 + t x_1`
- target velocity `x_1 - x_0`

### 5.2 Posterior sampling

At inference time, sampling starts from Gaussian noise in joint space and integrates the learned flow with explicit Euler steps.

At each sample step, we may apply:

- observation guidance on `v`
- physics projection according to the chosen schedule/order

Implementation:

- sampler: [pipeline.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/pipeline.py)
- problem wrapper: [problem.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/problem.py)

### 5.3 Projection orders

First-order projection:

```text
x_proj = x - J^T (J J^T + lambda I)^(-1) h(x)
```

Second-order projection:

- damped Gauss-Newton / Newton-KKT style correction
- float64 linear algebra
- line search
- small iteration budget

Implementation:

- [projection.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/projection.py)

### 5.4 Final cleanup

After the reverse trajectory ends, the pipeline can still run a final cleanup projection. This matters because the project compares:

- no intermediate projection
- intermediate projection schedules
- and the effect of a final deterministic cleanup

## 6. Final-Project File Map

- [configs/posterior_projection.yaml](/Users/hanszhu/Desktop/3D_PINN/configs/posterior_projection.yaml)
  - source of truth for benchmark, training, sampling, projection, and evaluation defaults
- [src/posterior_projection/config.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/config.py)
  - typed config loading and checkpoint config recovery
- [src/posterior_projection/dataset.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/dataset.py)
  - joint dataset and on-the-fly observation mask generation
- [src/posterior_projection/problem.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/problem.py)
  - PDE residual, Jacobian, observation loss, oracle comparisons
- [src/posterior_projection/model.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/model.py)
  - 1D FNO-style flow model
- [src/posterior_projection/flow.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/flow.py)
  - flow-matching objective and Euler sampler
- [src/posterior_projection/projection.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/projection.py)
  - first-order and Gauss-Newton projection operators
- [src/posterior_projection/pipeline.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/pipeline.py)
  - full posterior-sampling pipeline
- [src/posterior_projection/evaluate.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/evaluate.py)
  - schedule/order study sweep and ranking table
- [src/posterior_projection/study.py](/Users/hanszhu/Desktop/3D_PINN/src/posterior_projection/study.py)
  - static schedules and dynamic schedule logic
- [tests/test_posterior_projection.py](/Users/hanszhu/Desktop/3D_PINN/tests/test_posterior_projection.py)
  - unit and smoke coverage for the final project
- [src/chonkdiff](/Users/hanszhu/Desktop/3D_PINN/src/chonkdiff)
  - benchmark/oracle backend used by the final project for PDE residuals and oracle data generation
- [configs/chonkdiff_elliptic.yaml](/Users/hanszhu/Desktop/3D_PINN/configs/chonkdiff_elliptic.yaml)
  - benchmark/oracle defaults for the nonlinear elliptic backend

## 7. Current Config Defaults

Current defaults from [posterior_projection.yaml](/Users/hanszhu/Desktop/3D_PINN/configs/posterior_projection.yaml):

- observation mode: `partial_observation`
- observed fraction: `0.1`
- observation noise: `0.0`
- flow steps: `100`
- observation guidance strength: `2.5e-1`
- training epochs: `80`
- batch size: `64`
- projection schedules:
  - `none`
  - `final_only`
  - `every_step`
  - `every_2`
  - `every_5`
  - `late_only`
  - `adaptive_residual`
- projection orders:
  - `first_order`
  - `gauss_newton`
- `late_only` threshold start: `t >= 0.7`
- `adaptive_residual` warmup: `0.5`
- final cleanup: enabled
- final cleanup iterations: `8`

## 8. Step-by-Step Plan

### Step 1: train a joint prior

Train the flow-matching model on oracle `(u, v)` pairs from the elliptic benchmark.

Goal:

- learn a good unconditional prior over physically meaningful joint states

### Step 2: define the inverse posterior task

Hide most of `v`, keep only sparse partial observations, and ask the model to reconstruct a plausible posterior sample for `(u, v)`.

Goal:

- make the final project genuinely generative and inverse-problem focused

### Step 3: run guided sampling

During sampling:

- integrate the learned flow
- apply observation guidance
- optionally apply projection according to schedule `S` and order `q`

Goal:

- study how physics affects the posterior trajectory, not just the final cleaned result

### Step 4: compare projection schedules

We explicitly compare:

- no projection
- final-only projection
- frequent projection
- sparse periodic projection
- late-stage projection
- adaptive residual-based projection

Goal:

- answer when projection should be used

### Step 5: compare projection orders

We compare:

- first-order Jacobian correction
- second-order Gauss-Newton correction

Goal:

- answer whether stronger physics is worth the runtime and trajectory distortion

### Step 6: rank by the four project objectives

For every `(schedule, order)` pair, evaluate:

- posterior quality
- physical consistency
- runtime
- trajectory stability

Goal:

- produce a clean final-project result table showing different winners for different objectives if that is what the data says

## 9. Current Status

What is already working:

- full train / sample / evaluate pipeline
- end-to-end nonlinear elliptic benchmark path
- observation guidance during sampling
- first-order and Gauss-Newton projection
- schedule sweep and ranking table
- unit and smoke tests

What is not finished scientifically:

- posterior recovery quality is still much weaker than physical consistency after final cleanup
- the next real work is tuning and stronger training, not just more code plumbing

In other words:

- the infrastructure is ready
- the research comparison is running
- the model quality still needs improvement before the final conclusions are strong

## 10. Commands

Train:

```bash
PYTHONPATH=src python -m posterior_projection.train \
  --config configs/posterior_projection.yaml
```

Sample one trajectory:

```bash
PYTHONPATH=src python -m posterior_projection.sample \
  --checkpoint outputs/posterior_projection/best.pt \
  --schedule every_2 \
  --order gauss_newton
```

Quick evaluation:

```bash
PYTHONPATH=src python -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection/best.pt \
  --num-samples 4 \
  --num-observation-seeds 1
```

Larger evaluation:

```bash
PYTHONPATH=src python -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection/best.pt \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --json-out outputs/posterior_projection/eval_32x2.json \
  --csv-out outputs/posterior_projection/eval_32x2.csv
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_posterior_projection.py'
```

## 11. References Stored in This Workspace

- [SOURCE_CATALOG.md](/Users/hanszhu/Desktop/3D_PINN/final_project/references/SOURCE_CATALOG.md)
- [pcfm.pdf](/Users/hanszhu/Desktop/3D_PINN/final_project/references/papers/pcfm.pdf)
- [diffusionpde.pdf](/Users/hanszhu/Desktop/3D_PINN/final_project/references/papers/diffusionpde.pdf)
- [pcfm.txt](/Users/hanszhu/Desktop/3D_PINN/final_project/references/text/pcfm.txt)
- [diffusionpde.txt](/Users/hanszhu/Desktop/3D_PINN/final_project/references/text/diffusionpde.txt)

These are the local paper references that ground the proposal and implementation.
