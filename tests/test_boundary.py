"""Test boundary condition enforcement."""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pinn3d.config import load_config
from pinn3d.model_siren import create_model
from pinn3d.sampling import sample_boundary_sobol, scale_to_input_range, scale_k_to_input_range
from pinn3d.pde import batch_prediction


def test_boundary_sampling():
    """Test boundary sampling covers all 6 faces."""
    n_per_face = 100
    boundary_points = sample_boundary_sobol(n_per_face, seed=42)
    
    # Should have 6 * n_per_face points
    assert boundary_points.shape == (6 * n_per_face, 3), f"Wrong shape: {boundary_points.shape}"
    
    # Check that points are on boundaries (at 0 or 1)
    tolerance = 1e-6
    
    # Each point should have at least one coordinate at 0 or 1
    on_boundary = jnp.any(
        (jnp.abs(boundary_points) < tolerance) | 
        (jnp.abs(boundary_points - 1.0) < tolerance),
        axis=1
    )
    
    assert jnp.all(on_boundary), "Some points are not on boundary!"
    
    # Check that we have points on all 6 faces
    faces = {
        'x0': jnp.sum(jnp.abs(boundary_points[:, 0]) < tolerance),
        'x1': jnp.sum(jnp.abs(boundary_points[:, 0] - 1.0) < tolerance),
        'y0': jnp.sum(jnp.abs(boundary_points[:, 1]) < tolerance),
        'y1': jnp.sum(jnp.abs(boundary_points[:, 1] - 1.0) < tolerance),
        'z0': jnp.sum(jnp.abs(boundary_points[:, 2]) < tolerance),
        'z1': jnp.sum(jnp.abs(boundary_points[:, 2] - 1.0) < tolerance),
    }
    
    print("Points per face:")
    for face_name, count in faces.items():
        print(f"  {face_name}: {count}")
        assert count > 0, f"No points on face {face_name}"
    
    print("✓ Test passed: Boundary sampling correct")


def test_boundary_predictions_untrained():
    """Test that untrained model predictions on boundary can be computed."""
    jax.config.update("jax_enable_x64", True)
    
    # Load config
    config = load_config()
    
    # Create model
    model = create_model(config)
    
    # Sample boundary points
    n_per_face = 100
    boundary_points = sample_boundary_sobol(n_per_face, seed=42)
    boundary_points_scaled = scale_to_input_range(boundary_points)
    
    # Get k
    k_physical = config['pde']['k_train_min']
    k_scaled = scale_k_to_input_range(
        k_physical, 
        config['pde']['k_train_min'],
        config['pde']['k_train_max']
    )
    
    # Predict
    predictions = batch_prediction(model, boundary_points_scaled, k_scaled)
    
    # Check shape
    assert predictions.shape == (6 * n_per_face,), f"Wrong prediction shape: {predictions.shape}"
    
    # Check that predictions are finite
    assert jnp.all(jnp.isfinite(predictions)), "Some predictions are not finite!"
    
    print(f"Prediction statistics:")
    print(f"  Mean: {jnp.mean(predictions):.6e}")
    print(f"  Std: {jnp.std(predictions):.6e}")
    print(f"  Min: {jnp.min(predictions):.6e}")
    print(f"  Max: {jnp.max(predictions):.6e}")
    
    print("✓ Test passed: Boundary predictions computable")


if __name__ == "__main__":
    print("Testing boundary conditions...")
    print("=" * 80)
    test_boundary_sampling()
    print()
    test_boundary_predictions_untrained()
    print("=" * 80)
    print("All tests passed!")
