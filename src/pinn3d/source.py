"""Source term for the Helmholtz equation."""

import jax.numpy as jnp
import jax


def gaussian_source(x: jnp.ndarray, 
                    center: jnp.ndarray, 
                    width: float, 
                    amplitude: float = 1.0) -> jnp.ndarray:
    """Compute Gaussian source term.
    
    s(x) = A * exp(-||x - c||^2 / (2 * sigma^2))
    
    Args:
        x: Points of shape (..., 3)
        center: Source center of shape (3,)
        width: Gaussian width (sigma)
        amplitude: Source amplitude
        
    Returns:
        Source values of shape (...)
    """
    # Compute squared distance
    diff = x - center
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    
    # Gaussian
    s = amplitude * jnp.exp(-dist_sq / (2.0 * width ** 2))
    
    return s


def get_source_fn(center: list, width: float, amplitude: float = 1.0):
    """Create source function with fixed parameters.
    
    Args:
        center: Source center [x, y, z]
        width: Gaussian width
        amplitude: Source amplitude
        
    Returns:
        Callable source function s(x) -> scalar or array
    """
    center_array = jnp.array(center, dtype=jnp.float64)
    
    def source_fn(x: jnp.ndarray) -> jnp.ndarray:
        return gaussian_source(x, center_array, width, amplitude)
    
    return source_fn
