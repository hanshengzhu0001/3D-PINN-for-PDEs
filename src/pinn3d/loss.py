"""Loss function for PINN training."""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple
from .pde import batch_prediction, batch_residual


def compute_loss(model, 
                 x_interior: jnp.ndarray,
                 x_boundary: jnp.ndarray,
                 k_scaled: float,
                 k_physical: float,
                 source_fn: Callable,
                 boundary_weight: float = 100.0,
                 residual_weight: float = 1.0) -> Tuple[jnp.ndarray, dict]:
    """Compute weighted loss.
    
    Loss = boundary_weight * L_b + residual_weight * L_f
    
    where:
        L_b = MSE of boundary predictions (target = 0)
        L_f = MSE of PDE residuals (target = 0)
    
    Args:
        model: Neural network model
        x_interior: Interior collocation points (n_f, 3) in [-1, 1]
        x_boundary: Boundary points (n_b, 3) in [-1, 1]
        k_scaled: Wavenumber scaled to [-1, 1]
        k_physical: Physical wavenumber
        source_fn: Source term function
        boundary_weight: Weight for boundary loss
        residual_weight: Weight for residual loss
        
    Returns:
        Tuple of (total_loss, info_dict)
    """
    # Boundary loss: MSE of predictions on boundary (should be 0)
    u_boundary = batch_prediction(model, x_boundary, k_scaled)
    loss_boundary = jnp.mean(u_boundary ** 2)
    
    # Residual loss: MSE of PDE residuals at interior points
    residuals = batch_residual(model, x_interior, k_scaled, k_physical, source_fn)
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


def make_loss_fn(config: dict, source_fn: Callable):
    """Create loss function with fixed configuration.
    
    Args:
        config: Configuration dictionary
        source_fn: Source term function
        
    Returns:
        Loss function with signature (model, x_interior, x_boundary, k_scaled, k_physical)
    """
    boundary_weight = config['loss']['boundary_weight']
    residual_weight = config['loss']['residual_weight']
    
    def loss_fn(model, x_interior, x_boundary, k_scaled, k_physical):
        return compute_loss(
            model, x_interior, x_boundary, k_scaled, k_physical,
            source_fn, boundary_weight, residual_weight
        )
    
    return loss_fn
