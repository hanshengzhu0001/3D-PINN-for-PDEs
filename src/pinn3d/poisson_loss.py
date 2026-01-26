"""Loss function for Poisson PINN."""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple
from .poisson_pde import batch_prediction_poisson, batch_residual_poisson


def compute_poisson_loss(model, 
                        x_interior: jnp.ndarray,
                        x_boundary: jnp.ndarray,
                        source_fn: Callable,
                        boundary_weight: float = 10.0,
                        residual_weight: float = 1.0) -> Tuple[jnp.ndarray, dict]:
    """Compute weighted loss for Poisson equation.
    
    Loss = boundary_weight * L_b + residual_weight * L_f
    
    Args:
        model: Neural network model
        x_interior: Interior collocation points (n_f, 3) in [-1, 1]
        x_boundary: Boundary points (n_b, 3) in [-1, 1]
        source_fn: Source term function
        boundary_weight: Weight for boundary loss
        residual_weight: Weight for residual loss
        
    Returns:
        Tuple of (total_loss, info_dict)
    """
    # Boundary loss
    u_boundary = batch_prediction_poisson(model, x_boundary)
    loss_boundary = jnp.mean(u_boundary ** 2)
    
    # Residual loss
    residuals = batch_residual_poisson(model, x_interior, source_fn)
    loss_residual = jnp.mean(residuals ** 2)
    
    # Weighted total loss
    total_loss = boundary_weight * loss_boundary + residual_weight * loss_residual
    
    # Info for logging
    info = {
        'loss_total': total_loss,
        'loss_boundary': loss_boundary,
        'loss_residual': loss_residual,
        'max_boundary_error': jnp.max(jnp.abs(u_boundary)),
        'max_residual': jnp.max(jnp.abs(residuals)),
        'median_residual': jnp.median(jnp.abs(residuals))
    }
    
    return total_loss, info


def make_poisson_loss_fn(config: dict, source_fn: Callable):
    """Create Poisson loss function.
    
    Args:
        config: Configuration dictionary
        source_fn: Source term function
        
    Returns:
        Loss function
    """
    boundary_weight = config['loss']['boundary_weight']
    residual_weight = config['loss']['residual_weight']
    
    def loss_fn(model, x_interior, x_boundary):
        return compute_poisson_loss(
            model, x_interior, x_boundary,
            source_fn, boundary_weight, residual_weight
        )
    
    return loss_fn
