"""Smoke test for PDE residual computation."""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pinn3d.config import load_config
from pinn3d.model_siren import create_model
from pinn3d.source import get_source_fn
from pinn3d.sampling import sample_interior_sobol, scale_to_input_range, scale_k_to_input_range
from pinn3d.pde import batch_residual, compute_residual


def test_residual_computation():
    """Test that residual can be computed without errors."""
    jax.config.update("jax_enable_x64", True)
    
    # Load config
    config = load_config()
    
    # Create model
    model = create_model(config)
    
    # Create source function
    source_fn = get_source_fn(
        config['source']['center'],
        config['source']['width'],
        config['source']['amplitude']
    )
    
    # Sample interior points
    n_points = 1000
    interior_points = sample_interior_sobol(n_points, seed=42)
    interior_points_scaled = scale_to_input_range(interior_points)
    
    # Get k
    k_physical = config['pde']['k_train_min']
    k_scaled = scale_k_to_input_range(
        k_physical,
        config['pde']['k_train_min'],
        config['pde']['k_train_max']
    )
    
    # Compute residuals
    residuals = batch_residual(
        model,
        interior_points_scaled,
        k_scaled,
        k_physical,
        source_fn
    )
    
    # Check shape
    assert residuals.shape == (n_points,), f"Wrong residual shape: {residuals.shape}"
    
    # Check that residuals are finite
    assert jnp.all(jnp.isfinite(residuals)), "Some residuals are not finite!"
    
    print(f"Residual statistics (untrained model):")
    print(f"  Mean: {jnp.mean(jnp.abs(residuals)):.6e}")
    print(f"  Std: {jnp.std(residuals):.6e}")
    print(f"  Min: {jnp.min(residuals):.6e}")
    print(f"  Max: {jnp.max(residuals):.6e}")
    print(f"  Median: {jnp.median(jnp.abs(residuals)):.6e}")
    
    print("✓ Test passed: Residual computation works")


def test_single_point_residual():
    """Test residual computation for a single point."""
    jax.config.update("jax_enable_x64", True)
    
    # Load config
    config = load_config()
    
    # Create model
    model = create_model(config)
    
    # Create source function
    source_fn = get_source_fn(
        config['source']['center'],
        config['source']['width'],
        config['source']['amplitude']
    )
    
    # Single point in center
    x = jnp.array([0.0, 0.0, 0.0])  # Scaled to [-1, 1]
    k_physical = config['pde']['k_train_min']
    k_scaled = scale_k_to_input_range(
        k_physical,
        config['pde']['k_train_min'],
        config['pde']['k_train_max']
    )
    
    # Compute residual
    residual = compute_residual(model, x, k_scaled, k_physical, source_fn)
    
    # Check that it's a scalar
    assert residual.shape == (), f"Residual should be scalar, got shape: {residual.shape}"
    
    # Check that it's finite
    assert jnp.isfinite(residual), "Residual is not finite!"
    
    print(f"Single point residual: {residual:.6e}")
    
    print("✓ Test passed: Single point residual computation works")


def test_source_evaluation():
    """Test source term evaluation."""
    # Load config
    config = load_config()
    
    # Create source function
    source_fn = get_source_fn(
        config['source']['center'],
        config['source']['width'],
        config['source']['amplitude']
    )
    
    # Evaluate at center
    center = jnp.array(config['source']['center'])
    s_center = source_fn(center)
    
    # Should be close to amplitude
    amplitude = config['source']['amplitude']
    print(f"Source at center: {s_center:.6f} (expected ~{amplitude})")
    assert jnp.abs(s_center - amplitude) < 0.01, "Source at center should be close to amplitude"
    
    # Evaluate far from center
    far_point = jnp.array([0.0, 0.0, 0.0])
    s_far = source_fn(far_point)
    
    # Should be much smaller
    print(f"Source far from center: {s_far:.6e}")
    assert s_far < amplitude * 0.1, "Source far from center should be small"
    
    print("✓ Test passed: Source evaluation correct")


if __name__ == "__main__":
    print("Running smoke tests for residual computation...")
    print("=" * 80)
    test_source_evaluation()
    print()
    test_single_point_residual()
    print()
    test_residual_computation()
    print("=" * 80)
    print("All smoke tests passed!")
