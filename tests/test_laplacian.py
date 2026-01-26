"""Test Laplacian computation."""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pinn3d.model_siren import SirenNetwork
from pinn3d.pde import compute_laplacian


def test_laplacian_quadratic():
    """Test Laplacian on quadratic function.
    
    For u = x² + y² + z², we have:
    ∇²u = 2 + 2 + 2 = 6
    """
    jax.config.update("jax_enable_x64", True)
    
    # Create a simple model that outputs x² + y² + z²
    class QuadraticModel:
        def __call__(self, x):
            # x is [x, y, z, k] in [-1, 1], but we ignore k for this test
            spatial = x[:3]  # Just take x, y, z
            # Convert from [-1, 1] to [0, 1]
            spatial_01 = (spatial + 1.0) / 2.0
            return jnp.sum(spatial_01 ** 2)
    
    model = QuadraticModel()
    
    # Test at a point
    x = jnp.array([0.5, 0.3, 0.7])  # in [-1, 1]
    k = 0.0  # doesn't matter for this test
    
    laplacian = compute_laplacian(model, x, k)
    
    # Expected: 6 (but scaled by input scaling factor)
    # Since we're taking derivatives w.r.t. [-1, 1] space, we need to account for the chain rule
    # dx_physical/dx_scaled = 0.5, so d²/dx_physical² = (d²/dx_scaled²) * 4
    # So Laplacian in scaled space = 6 / 4 = 1.5
    expected = 1.5
    
    print(f"Computed Laplacian: {laplacian}")
    print(f"Expected Laplacian: {expected}")
    
    # Check if close (with tolerance)
    assert jnp.abs(laplacian - expected) < 0.1, f"Laplacian mismatch: {laplacian} vs {expected}"
    print("✓ Test passed: Laplacian computation correct for quadratic function")


def test_laplacian_sine():
    """Test Laplacian on sine function.
    
    For u = sin(2πx), we have:
    ∇²u = -4π² sin(2πx)
    """
    jax.config.update("jax_enable_x64", True)
    
    class SineModel:
        def __call__(self, x):
            # x is [x, y, z, k] in [-1, 1]
            x_coord = x[0]
            # Convert from [-1, 1] to [0, 1]
            x_01 = (x_coord + 1.0) / 2.0
            return jnp.sin(2.0 * jnp.pi * x_01)
    
    model = SineModel()
    
    # Test at x=0.5 (should be sin(π)=0, Laplacian should be close to 0)
    x = jnp.array([0.0, 0.0, 0.0])  # x=0 in [-1,1] = 0.5 in [0,1]
    k = 0.0
    
    laplacian = compute_laplacian(model, x, k)
    
    # At x=0.5, sin(π) ≈ 0, Laplacian ≈ 0
    print(f"Computed Laplacian at x=0.5: {laplacian}")
    
    # Should be close to 0
    assert jnp.abs(laplacian) < 0.1, f"Laplacian should be near 0: {laplacian}"
    print("✓ Test passed: Laplacian computation correct for sine function")


if __name__ == "__main__":
    print("Testing Laplacian computation...")
    print("=" * 80)
    test_laplacian_quadratic()
    print()
    test_laplacian_sine()
    print("=" * 80)
    print("All tests passed!")
