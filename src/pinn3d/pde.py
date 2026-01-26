"""PDE residual computation for Helmholtz equation."""

import jax
import jax.numpy as jnp
from typing import Callable


def compute_laplacian(model, x: jnp.ndarray, k: float) -> jnp.ndarray:
    """Compute Laplacian of model output using JAX autodiff with proper scaling.
    
    CRITICAL: Network input is in [-1, 1] but PDE is in [0, 1].
    Chain rule: d²u/dx_phys² = 4 * d²u/dx_net² (since dx_net = 2*dx_phys)
    
    Args:
        model: Neural network model
        x: Spatial coordinates (x, y, z) in network coords [-1, 1]
        k: Wavenumber (scalar) already scaled to [-1, 1]
        
    Returns:
        Laplacian value in physical coordinates [0, 1]
    """
    def u_fn(spatial_coords):
        """Model output as function of spatial coordinates only."""
        # spatial_coords is (x, y, z) already scaled to [-1, 1]
        # k is already scaled to [-1, 1]
        input_vec = jnp.concatenate([spatial_coords, jnp.array([k])])
        return model(input_vec)
    
    # Second derivatives in network coordinates (diagonal of Hessian)
    def u_xx_fn(spatial_coords):
        return jax.grad(u_fn)(spatial_coords)[0]
    
    def u_yy_fn(spatial_coords):
        return jax.grad(u_fn)(spatial_coords)[1]
    
    def u_zz_fn(spatial_coords):
        return jax.grad(u_fn)(spatial_coords)[2]
    
    u_xx_net = jax.grad(u_xx_fn)(x)[0]
    u_yy_net = jax.grad(u_yy_fn)(x)[1]
    u_zz_net = jax.grad(u_zz_fn)(x)[2]
    
    # CRITICAL FIX: Scale to physical coordinates
    # x_phys = (x_net + 1)/2, so dx_net/dx_phys = 2
    # d²u/dx_phys² = (d²u/dx_net²) * (dx_net/dx_phys)² = 4 * d²u/dx_net²
    scaling_factor = 4.0
    
    laplacian_phys = scaling_factor * (u_xx_net + u_yy_net + u_zz_net)
    
    return laplacian_phys


def compute_residual(model, x: jnp.ndarray, k_scaled: float, k_physical: float, 
                     source_fn: Callable) -> jnp.ndarray:
    """Compute PDE residual.
    
    Residual: r = ∇²u + k²u - s(x)
    
    Args:
        model: Neural network model
        x: Spatial coordinates in [-1, 1] scale (for network input)
        k_scaled: Wavenumber scaled to [-1, 1] (for network input)
        k_physical: Physical wavenumber (for PDE equation)
        source_fn: Source term function (takes physical coordinates [0,1])
        
    Returns:
        Residual value
    """
    # Compute Laplacian
    laplacian = compute_laplacian(model, x, k_scaled)
    
    # Get model output
    input_vec = jnp.concatenate([x, jnp.array([k_scaled])])
    u = model(input_vec)
    
    # Convert x back to physical coordinates [0, 1] for source term
    x_physical = (x + 1.0) / 2.0
    s = source_fn(x_physical)
    
    # Compute residual: ∇²u + k²u - s
    residual = laplacian + (k_physical ** 2) * u - s
    
    return residual


def batch_residual(model, x_batch: jnp.ndarray, k_scaled: float, k_physical: float,
                   source_fn: Callable) -> jnp.ndarray:
    """Compute residuals for a batch of points.
    
    Args:
        model: Neural network model
        x_batch: Batch of spatial coordinates (n, 3) in [-1, 1]
        k_scaled: Wavenumber scaled to [-1, 1]
        k_physical: Physical wavenumber
        source_fn: Source term function
        
    Returns:
        Residuals of shape (n,)
    """
    # Vectorize over batch
    residual_fn = lambda x: compute_residual(model, x, k_scaled, k_physical, source_fn)
    residuals = jax.vmap(residual_fn)(x_batch)
    return residuals


def batch_prediction(model, x_batch: jnp.ndarray, k_scaled: float) -> jnp.ndarray:
    """Predict u values for a batch of points.
    
    Args:
        model: Neural network model
        x_batch: Batch of spatial coordinates (n, 3) in [-1, 1]
        k_scaled: Wavenumber scaled to [-1, 1]
        
    Returns:
        Predictions of shape (n,)
    """
    # Create input batch: (n, 4)
    k_vec = jnp.full((x_batch.shape[0], 1), k_scaled)
    inputs = jnp.concatenate([x_batch, k_vec], axis=1)
    
    # Vectorize prediction
    predictions = jax.vmap(model)(inputs)
    return predictions
