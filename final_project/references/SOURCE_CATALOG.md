# Source Catalog

This catalog records the key sources we are using for the final project and what each source contributes.

## Local Paper Copies

| Source | Local file | Why it matters |
| --- | --- | --- |
| Physics-Constrained Flow Matching (PCFM) | `papers/pcfm.pdf` | Main algorithmic template for projection during generative sampling; gives the strongest motivation for schedule/order comparisons. |
| DiffusionPDE | `papers/diffusionpde.pdf` | Main posterior-sampling / inverse-problem baseline under partial observation; shows how PDE guidance improves diffusion-based inference. |

## Searchable Text Copies

| Extract | Local file | Use |
| --- | --- | --- |
| PCFM text | `text/pcfm.txt` | Fast grep for metrics, algorithm steps, and runtime notes. |
| DiffusionPDE text | `text/diffusionpde.txt` | Fast grep for inverse-problem setup, partial-observation framing, and PDE-guidance results. |

## Key Results We Will Reuse

### PCFM

- Hard-constraint generative sampling via projection during flow matching.
- Reported metrics include `MMSE`, `SMSE`, `CE`, and `FPD`.
- Notable comparisons against DiffusionPDE:
  - Heat `MMSE`: `0.241` vs `4.49` (`x 1e-2`)
  - Burgers-IC `MMSE`: `0.052` vs `14.3` (`x 1e-2`)
- Runtime note:
  - the final projection step contributes only about `1-3%` of total runtime according to the paper text.

### DiffusionPDE

- Joint generative modeling of solution and coefficient under partial observation.
- Strong inverse-problem framing for posterior sampling.
- PDE-guidance ablation result used in the proposal:
  - Helmholtz solution relative error drops from `9.3%` to `0.6%`
  - coefficient relative error drops from `13.2%` to `9.4%`

## Additional External Sources To Keep In Scope

These are important references for the implementation, even though the repo currently stores only the two local PDFs above.

- Diffusion Posterior Sampling
  - https://arxiv.org/abs/2209.14687
  - Useful for posterior-sampling language and inverse-problem conditioning.
- Flow Matching for Generative Modeling
  - https://arxiv.org/abs/2210.02747
  - Useful for the base generative dynamics perspective behind PCFM.
- Operator Learning at Machine Precision
  - https://arxiv.org/abs/2511.19980
  - Useful because the existing elliptic benchmark/oracle in this repo comes from this benchmark family.
- PCFM code
  - https://github.com/cpfpengfei/PCFM
  - Useful for implementation details around schedule/projection logic.

## Implementation Mapping

- `PCFM -> projection algorithm`
  - schedule choices
  - first-order constraint projector
  - runtime / physical-consistency metrics
- `DiffusionPDE -> posterior-sampling framing`
  - partial observations
  - inverse-problem evaluation
  - reconstruction metrics
- `CHONKNORIS benchmark/oracle -> numerical backbone`
  - nonlinear elliptic testbed
  - float64 projector
  - solver-quality ground truth
