"""Poisson equation implementation (simpler than Helmholtz)."""

import jax
import jax.numpy as jnp
from typing import Callable


def manufactured_solution_source(x: jnp.ndarray, pi: float = jnp.pi) -> jnp.ndarray:
    """Source term for manufactured solution u = sin(πx)sin(πy)sin(πz).
    
    For u = sin(πx)sin(πy)sin(πz):
    ∇²u = -3π²sin(πx)sin(πy)sin(πz) = -3π²u
    
    So source f = -3π²sin(πx)sin(πy)sin(πz)
    
    Args:
        x: Points of shape (..., 3) with coordinates in [0, 1]
        pi: Value of π
        
    Returns:
        Source values of shape (...)
    """
    # x is in [0, 1]
    f = -3.0 * pi**2 * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1]) * jnp.sin(pi * x[..., 2])
    return f


def exact_solution(x: jnp.ndarray, pi: float = jnp.pi) -> jnp.ndarray:
    """Exact solution u = sin(πx)sin(πy)sin(πz).
    
    Args:
        x: Points of shape (..., 3) with coordinates in [0, 1]
        pi: Value of π
        
    Returns:
        Exact solution values of shape (...)
    """
    u = jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1]) * jnp.sin(pi * x[..., 2])
    return u


def compute_laplacian_poisson(model, x: jnp.ndarray) -> jnp.ndarray:
    """Compute Laplacian for Poisson equation with proper coordinate scaling.
    
    CRITICAL: Network input is in [-1, 1] but PDE is in [0, 1].
    Chain rule: d²u/dx_phys² = 4 * d²u/dx_net² (since dx_net = 2*dx_phys)
    
    Args:
        model: Neural network model
        x: Spatial coordinates (x, y, z) scaled to [-1, 1] (network coords)
        
    Returns:
        Laplacian value in physical coordinates [0, 1]
    """
    def u_fn(spatial_coords):
        """Model output as function of spatial coordinates."""
        return model(spatial_coords)
    
    # Second derivatives in network coordinates
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


def compute_residual_poisson(model, x: jnp.ndarray, source_fn: Callable) -> jnp.ndarray:
    """Compute Poisson equation residual.
    
    Residual: r = ∇²u - f(x)
    
    Args:
        model: Neural network model
        x: Spatial coordinates in [-1, 1] (for network input)
        source_fn: Source term function (takes physical coordinates [0,1])
        
    Returns:
        Residual value
    """
    # Compute Laplacian
    laplacian = compute_laplacian_poisson(model, x)
    
    # Convert x back to physical coordinates [0, 1] for source term
    x_physical = (x + 1.0) / 2.0
    f = source_fn(x_physical)
    
    # Residual: ∇²u - f = 0
    residual = laplacian - f
    
    return residual


def batch_residual_poisson(model, x_batch: jnp.ndarray, source_fn: Callable) -> jnp.ndarray:
    """Compute Poisson residuals for a batch of points.
    
    Args:
        model: Neural network model
        x_batch: Batch of spatial coordinates (n, 3) in [-1, 1]
        source_fn: Source term function
        
    Returns:
        Residuals of shape (n,)
    """
    residual_fn = lambda x: compute_residual_poisson(model, x, source_fn)
    residuals = jax.vmap(residual_fn)(x_batch)
    return residuals


def batch_prediction_poisson(model, x_batch: jnp.ndarray) -> jnp.ndarray:
    """Predict u values for a batch of points.
    
    Args:
        model: Neural network model
        x_batch: Batch of spatial coordinates (n, 3) in [-1, 1]
        
    Returns:
        Predictions of shape (n,)
    """
    predictions = jax.vmap(model)(x_batch)
    return predictions
