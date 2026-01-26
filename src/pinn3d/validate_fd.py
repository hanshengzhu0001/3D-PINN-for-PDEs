"""Finite difference validation for PINN."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict
from .sampling import scale_to_input_range, scale_k_to_input_range
from .pde import batch_prediction, batch_residual


def build_3d_laplacian_matrix(n: int) -> sparse.csr_matrix:
    """Build 3D Laplacian matrix with 7-point stencil.
    
    Args:
        n: Grid size (n x n x n grid)
        
    Returns:
        Sparse Laplacian matrix of size (n^3, n^3)
    """
    N = n ** 3
    h = 1.0 / (n - 1)  # Grid spacing
    
    # 7-point stencil coefficients
    # L u_ijk = (u_{i-1,j,k} + u_{i+1,j,k} + u_{i,j-1,k} + u_{i,j+1,k} + u_{i,j,k-1} + u_{i,j,k+1} - 6*u_{i,j,k}) / h^2
    
    diagonals = []
    offsets = []
    
    # Center: -6/h^2
    diagonals.append(-6.0 / h**2 * np.ones(N))
    offsets.append(0)
    
    # x-direction: 1/h^2 at offset ±1
    diag_x = np.ones(N) / h**2
    # Zero out boundaries
    for k in range(n):
        for j in range(n):
            idx = k * n * n + j * n
            diag_x[idx] = 0  # x=0 boundary
            if idx > 0:
                diag_x[idx - 1] = 0  # Connection from x=n-1
    diagonals.append(diag_x[1:])
    offsets.append(1)
    diagonals.append(diag_x[:-1])
    offsets.append(-1)
    
    # y-direction: 1/h^2 at offset ±n
    diag_y = np.ones(N) / h**2
    for k in range(n):
        for i in range(n):
            idx = k * n * n + i
            diag_y[idx] = 0  # y=0 boundary
            idx2 = k * n * n + (n-1) * n + i
            if idx2 < N:
                diag_y[idx2] = 0  # y=n-1 boundary
    diagonals.append(diag_y[n:])
    offsets.append(n)
    diagonals.append(diag_y[:-n])
    offsets.append(-n)
    
    # z-direction: 1/h^2 at offset ±n^2
    diag_z = np.ones(N) / h**2
    # Zero out z boundaries
    for j in range(n):
        for i in range(n):
            idx = j * n + i
            diag_z[idx] = 0  # z=0 boundary
            idx2 = (n-1) * n * n + j * n + i
            if idx2 < N:
                diag_z[idx2] = 0  # z=n-1 boundary
    diagonals.append(diag_z[n*n:])
    offsets.append(n * n)
    diagonals.append(diag_z[:-n*n])
    offsets.append(-n * n)
    
    L = sparse.diags(diagonals, offsets, shape=(N, N), format='csr')
    
    return L


def solve_helmholtz_fd(n: int, k: float, source_fn) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Helmholtz equation using finite difference.
    
    Solves: (∇² + k²) u = s with Dirichlet BC u=0 on boundary
    
    Args:
        n: Grid size (n x n x n)
        k: Wavenumber
        source_fn: Source term function (takes physical coordinates)
        
    Returns:
        Tuple of (solution_grid, coordinates_grid)
    """
    # Build grid
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Build Laplacian matrix
    L = build_3d_laplacian_matrix(n)
    
    # Helmholtz operator: L + k²I
    I = sparse.identity(n**3, format='csr')
    A = L + k**2 * I
    
    # Source term
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    s = np.array([source_fn(jnp.array(c)) for c in coords])
    
    # Apply Dirichlet BC: set boundary points to 0
    is_boundary = np.zeros(n**3, dtype=bool)
    for i in range(n):
        for j in range(n):
            for k_idx in range(n):
                idx = i * n * n + j * n + k_idx
                if i == 0 or i == n-1 or j == 0 or j == n-1 or k_idx == 0 or k_idx == n-1:
                    is_boundary[idx] = True
    
    # Modify A and s for boundary conditions
    A = A.tolil()
    for idx in range(n**3):
        if is_boundary[idx]:
            A[idx, :] = 0
            A[idx, idx] = 1.0
            s[idx] = 0.0
    A = A.tocsr()
    
    # Solve linear system
    u = spsolve(A, s)
    
    # Reshape to grid
    u_grid = u.reshape((n, n, n))
    coords_grid = np.stack([X, Y, Z], axis=-1)
    
    return u_grid, coords_grid


def validate_with_fd(model, config: dict, source_fn, k_physical: float, 
                     verbose: bool = True) -> Dict:
    """Validate PINN against finite difference solution.
    
    Args:
        model: Trained PINN model
        config: Configuration dictionary
        source_fn: Source term function
        k_physical: Physical wavenumber to validate
        verbose: Whether to print results
        
    Returns:
        Dictionary with validation metrics
    """
    jax.config.update("jax_enable_x64", True)
    
    # Get validation config
    grid_size = config['validation']['grid_size']
    
    # Get k scaling parameters
    k_min = config['pde']['k_train_min']
    k_max = config['pde']['k_train_max']
    k_scaled = scale_k_to_input_range(k_physical, k_min, k_max)
    
    if verbose:
        print(f"\nValidating at k = {k_physical:.4f} (k_scaled = {k_scaled:.4f})")
        print(f"FD grid size: {grid_size}^3 = {grid_size**3:,} points")
    
    # Solve with finite difference
    if verbose:
        print("Solving with finite difference...")
    u_fd, coords_grid = solve_helmholtz_fd(grid_size, k_physical, source_fn)
    
    # Get PINN predictions
    if verbose:
        print("Computing PINN predictions...")
    
    # Flatten coordinates
    coords_flat = coords_grid.reshape(-1, 3)
    coords_scaled = scale_to_input_range(jnp.array(coords_flat))
    
    # Predict in batches to avoid memory issues
    batch_size = 10000
    u_pinn_list = []
    for i in range(0, len(coords_flat), batch_size):
        batch = coords_scaled[i:i+batch_size]
        u_batch = batch_prediction(model, batch, k_scaled)
        u_pinn_list.append(np.array(u_batch))
    
    u_pinn = np.concatenate(u_pinn_list)
    u_pinn_grid = u_pinn.reshape(u_fd.shape)
    
    # Compute errors (interior points only, exclude boundary)
    interior_mask = np.zeros_like(u_fd, dtype=bool)
    interior_mask[1:-1, 1:-1, 1:-1] = True
    
    u_fd_interior = u_fd[interior_mask]
    u_pinn_interior = u_pinn_grid[interior_mask]
    
    # L2 relative error
    l2_error_abs = np.linalg.norm(u_pinn_interior - u_fd_interior)
    l2_norm_fd = np.linalg.norm(u_fd_interior)
    l2_error_rel = l2_error_abs / (l2_norm_fd + 1e-10)
    
    # Boundary errors
    boundary_mask = ~interior_mask
    u_pinn_boundary = u_pinn_grid[boundary_mask]
    boundary_max_error = np.max(np.abs(u_pinn_boundary))
    
    # Residual statistics on random interior points
    n_residual_points = 100000
    rng = np.random.RandomState(42)
    random_interior = rng.uniform(0.01, 0.99, size=(n_residual_points, 3))
    random_interior_scaled = scale_to_input_range(jnp.array(random_interior))
    
    residuals = batch_residual(
        model, random_interior_scaled, k_scaled, k_physical, source_fn
    )
    residuals_np = np.array(residuals)
    median_residual = np.median(np.abs(residuals_np))
    max_residual = np.max(np.abs(residuals_np))
    mean_residual = np.mean(np.abs(residuals_np))
    
    metrics = {
        'k': k_physical,
        'l2_error_relative': float(l2_error_rel),
        'l2_error_absolute': float(l2_error_abs),
        'boundary_max_error': float(boundary_max_error),
        'median_residual': float(median_residual),
        'max_residual': float(max_residual),
        'mean_residual': float(mean_residual),
        'fd_grid_size': grid_size,
        'n_residual_points': n_residual_points
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"L2 relative error: {l2_error_rel:.6f} ({l2_error_rel*100:.3f}%)")
        print(f"L2 absolute error: {l2_error_abs:.6e}")
        print(f"Boundary max error: {boundary_max_error:.6e}")
        print(f"Residual median: {median_residual:.6e}")
        print(f"Residual mean: {mean_residual:.6e}")
        print(f"Residual max: {max_residual:.6e}")
        print("=" * 80)
        
        # Check acceptance criteria
        tol_l2 = config['validation']['tolerance_l2_error']
        tol_boundary = config['validation']['tolerance_boundary_error']
        tol_residual = config['validation']['tolerance_median_residual']
        
        passed = True
        print("\nACCEPTANCE CRITERIA:")
        
        if l2_error_rel < tol_l2:
            print(f"✓ L2 error {l2_error_rel:.6f} < {tol_l2}")
        else:
            print(f"✗ L2 error {l2_error_rel:.6f} >= {tol_l2}")
            passed = False
        
        if boundary_max_error < tol_boundary:
            print(f"✓ Boundary error {boundary_max_error:.6e} < {tol_boundary}")
        else:
            print(f"✗ Boundary error {boundary_max_error:.6e} >= {tol_boundary}")
            passed = False
        
        if median_residual < tol_residual:
            print(f"✓ Median residual {median_residual:.6e} < {tol_residual}")
        else:
            print(f"✗ Median residual {median_residual:.6e} >= {tol_residual}")
            passed = False
        
        if passed:
            print("\n🎉 ALL ACCEPTANCE CRITERIA PASSED!")
        else:
            print("\n⚠️  Some acceptance criteria not met")
        print("=" * 80)
    
    return metrics


def validate_all_k_values(model, config: dict, source_fn, verbose: bool = True) -> Dict:
    """Validate PINN for all k values in training grid.
    
    Args:
        model: Trained PINN model
        config: Configuration dictionary
        source_fn: Source term function
        verbose: Whether to print results
        
    Returns:
        Dictionary with all validation metrics
    """
    from .config import get_k_train_grid
    
    k_train_grid = get_k_train_grid(config)
    
    all_metrics = []
    for k in k_train_grid:
        metrics = validate_with_fd(model, config, source_fn, k, verbose=verbose)
        all_metrics.append(metrics)
    
    # Summary statistics
    l2_errors = [m['l2_error_relative'] for m in all_metrics]
    boundary_errors = [m['boundary_max_error'] for m in all_metrics]
    median_residuals = [m['median_residual'] for m in all_metrics]
    
    summary = {
        'individual_metrics': all_metrics,
        'l2_error_mean': float(np.mean(l2_errors)),
        'l2_error_max': float(np.max(l2_errors)),
        'boundary_error_mean': float(np.mean(boundary_errors)),
        'boundary_error_max': float(np.max(boundary_errors)),
        'median_residual_mean': float(np.mean(median_residuals)),
        'median_residual_max': float(np.max(median_residuals))
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL K VALUES")
        print("=" * 80)
        print(f"Mean L2 error: {summary['l2_error_mean']:.6f}")
        print(f"Max L2 error: {summary['l2_error_max']:.6f}")
        print(f"Mean boundary error: {summary['boundary_error_mean']:.6e}")
        print(f"Max boundary error: {summary['boundary_error_max']:.6e}")
        print(f"Mean median residual: {summary['median_residual_mean']:.6e}")
        print(f"Max median residual: {summary['median_residual_max']:.6e}")
        print("=" * 80)
    
    return summary
