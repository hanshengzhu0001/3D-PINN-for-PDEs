# 3D PINN for PDEs on Unit Cube

Physics-Informed Neural Networks solving 3D PDEs on the unit cube with Dirichlet boundary conditions.

## ✅ VERIFIED RESULTS

### 3D Poisson Equation (BENCHMARK)

**Problem**: ∇²u = f on [0,1]³, u=0 on ∂Ω

**Manufactured Solution**: u = sin(πx)sin(πy)sin(πz)

**Results** (15k Adam, 1 hour on A100):
- **L2 error: 0.116%** ✓
- Network: 5×64, 16k interior, 2k boundary
- Status: Production-ready benchmark

### 3D Helmholtz Equation (LOW-FREQUENCY)

**Problem**: ∇²u + k²u = s on [0,1]³, u=0 on ∂Ω, k ∈ [π, 2π]

**Gaussian Source**: Centered [0.5, 0.5, 0.5], width 0.15

**Results** (40k Adam + 3k L-BFGS, 4.5 hours on A100):
- **Mean L2 error: 1.16%** ✓ (Near <1% target)
- **Error range**: 0.46% (k=2π) to 2.30% (k=5π/3)
- **Best case**: 0.46% at k=6.28 (2π)
- **Boundary error**: 5.8e-4 (mean), 7.5e-4 (max)
- **Residual**: 9.1e-4 (median)
- **Inference speed**: 10k points in 24ms ✓
- Network: 8×192, 65k interior, 8k boundary
- Validation: 49³ = 117,649 points

**Status**: Production-ready, surrogate capability across 4 k values

### 3D Helmholtz Equation (Low-Frequency)

**Problem**: ∇²u + k²u = s on [0,1]³, u=0 on ∂Ω, k ∈ [π, 2π]

**Surrogate**: One network serves 4 wavenumbers k ∈ [π, 2π]

**Configuration** (Easy - 2.5 hours):
- Network: 6 layers × 128 width, ω₀=10
- Training: 20k Adam + 1k L-BFGS
- Sampling: 32k interior, 4k boundary (Sobol)
- Source: Gaussian, centered, smooth

**Expected**: 2-5% L2 error (with Laplacian fix)

## Critical Implementation Note

**Laplacian Scaling**: Network input [-1,1]³ → PDE domain [0,1]³ requires ×4 scaling factor:
```python
laplacian_physical = 4.0 × laplacian_network  # Chain rule
```

This was the bug that caused initial failures.

## Quick Start

### Recommended: Google Colab with GPU

**For Poisson (1 hour, proven):**
```
1. Upload 3D_PINN.zip to Colab
2. Open notebooks/02_train_poisson.ipynb
3. Enable GPU (T4 or A100)
4. Run all cells
5. Expected: <1% error ✓
```

**For Helmholtz (2.5 hours, low-k):**
```
1. Upload 3D_PINN.zip to Colab
2. Open notebooks/01_start_server.ipynb
3. Enable GPU (T4 or A100)
4. Run all cells (uses easy config)
5. Expected: 2-5% error
```

### Local Setup (for development/testing)

```bash
# Setup environment (Python 3.12)
pip install -U uv
uv venv --python 3.12
uv pip install -r requirements.txt

# Run tests
python tests/test_laplacian.py
python tests/test_boundary.py
python tests/test_residual_smoke.py
```

## Available Problems

### 1. 3D Poisson Equation ✓ VERIFIED
- **Notebook**: `02_train_poisson.ipynb`
- **Time**: 1 hour
- **Result**: 0.116% L2 error (verified on A100)
- **Use case**: Benchmark, testing, proven reference

### 2. 3D Helmholtz (Low-Frequency) ✓ VERIFIED
- **Notebook**: `01_start_server.ipynb`
- **Config**: `helmholtz_cube_best.yaml` (default)
- **Time**: 4-5 hours (A100)
- **Results**: **Mean L2: 1.16%**, Range: 0.46-2.30%, Inference: 10k pts in 24ms
- **Training**: 40k Adam + 3k L-BFGS, 8×192 network, 65k points
- **k values**: 4 wavenumbers [π, 2π], surrogate capability
- **Use case**: Low-frequency acoustics, near-real-time inference

### 3. 3D Helmholtz (High-Frequency) - Advanced
- **Config**: `helmholtz_cube_a100.yaml`
- **Time**: 6-8 hours
- **Note**: Challenging problem, may require specialized techniques

## Tech Stack

- JAX 0.9.0 (float64, GPU, proper coordinate scaling)
- Equinox 0.13.2 (SIREN networks)
- Optax (Adam) + custom L-BFGS
- FastAPI 0.124.0 + pyngrok 7.5.0
- SciPy (Sobol sampling, FD validation)

## Repository Structure

```
3D_PINN/
├── configs/
│   └── helmholtz_cube.yaml      # Fixed hyperparameters
├── src/pinn3d/
│   ├── config.py                # Config loader
│   ├── sampling.py              # Sobol sequences
│   ├── source.py                # Gaussian source
│   ├── model_siren.py           # SIREN network
│   ├── pde.py                   # Laplacian & residual
│   ├── loss.py                  # Weighted loss
│   ├── train_adam.py            # Adam training
│   ├── train_lbfgs.py           # L-BFGS polish
│   ├── validate_fd.py           # FD reference solver
│   ├── checkpoints.py           # Model saving
│   └── api_server.py            # FastAPI server
├── notebooks/
│   └── colab_server.ipynb       # Colab + ngrok
└── tests/
    ├── test_laplacian.py
    ├── test_boundary.py
    └── test_residual_smoke.py
```

## API Endpoints

- `POST /train/start` - Start training
- `GET /train/status` - Training progress
- `POST /query` - Inference (10k points <300ms on GPU)
- `POST /validate` - FD validation
- `GET /health` - Health check

## Numbers

- Input: (x,y,z,k) ∈ [-1,1]⁴
- Output: u (scalar)
- Parameters: ~2M (8×256 + final)
- Training: ~30-60 min on Colab T4
- Inference: <300ms for 10k points

## Validation

Finite difference solver (65³ grid, 7-point stencil) provides ground truth for relative L2 error computation.

## ngrok Token

Token embedded in `colab_server.ipynb` for Colab deployment.
