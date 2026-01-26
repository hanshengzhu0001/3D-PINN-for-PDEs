"""L-BFGS training implementation for PINN."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Tuple, Optional, List
from .sampling import sample_interior_sobol, sample_boundary_sobol, scale_to_input_range, scale_k_to_input_range
from .config import get_k_train_grid
import time


def lbfgs_two_loop_recursion(s_hist: List, y_hist: List, grad: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Two-loop recursion for L-BFGS direction computation.
    
    Args:
        s_hist: List of s vectors (parameter differences)
        y_hist: List of y vectors (gradient differences)
        grad: Current gradient
        gamma: Initial Hessian approximation scale
        
    Returns:
        Search direction
    """
    q = grad
    m = len(s_hist)
    alphas = []
    
    # First loop: backward
    for i in range(m - 1, -1, -1):
        s_i = s_hist[i]
        y_i = y_hist[i]
        rho_i = 1.0 / (jnp.dot(y_i, s_i) + 1e-10)
        alpha_i = rho_i * jnp.dot(s_i, q)
        alphas.insert(0, alpha_i)
        q = q - alpha_i * y_i
    
    # Initial Hessian approximation
    r = gamma * q
    
    # Second loop: forward
    for i in range(m):
        s_i = s_hist[i]
        y_i = y_hist[i]
        rho_i = 1.0 / (jnp.dot(y_i, s_i) + 1e-10)
        beta = rho_i * jnp.dot(y_i, r)
        r = r + s_i * (alphas[i] - beta)
    
    return -r  # Return negative for descent direction


def flatten_pytree(pytree):
    """Flatten pytree to 1D array."""
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    flat = jnp.concatenate([jnp.reshape(x, [-1]) for x in leaves])
    return flat, treedef, [x.shape for x in leaves]


def unflatten_pytree(flat, treedef, shapes):
    """Unflatten 1D array to pytree."""
    leaves = []
    idx = 0
    for shape in shapes:
        size = jnp.prod(jnp.array(shape))
        leaves.append(jnp.reshape(flat[idx:idx+size], shape))
        idx += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


def backtracking_line_search(loss_grad_fn, x0, grad0, direction, loss0, 
                             c1=1e-4, c2=0.9, max_iter=20):
    """Backtracking line search with strong Wolfe conditions.
    
    Args:
        loss_grad_fn: Function returning (loss, grad)
        x0: Current parameters (flat)
        grad0: Current gradient (flat)
        direction: Search direction (flat)
        loss0: Current loss value
        c1: Armijo condition parameter
        c2: Curvature condition parameter
        max_iter: Maximum line search iterations
        
    Returns:
        Step size alpha
    """
    alpha = 1.0
    directional_deriv = jnp.dot(grad0, direction)
    
    for _ in range(max_iter):
        x_new = x0 + alpha * direction
        loss_new, grad_new = loss_grad_fn(x_new)
        
        # Armijo condition
        if loss_new <= loss0 + c1 * alpha * directional_deriv:
            # Curvature condition
            directional_deriv_new = jnp.dot(grad_new, direction)
            if jnp.abs(directional_deriv_new) <= c2 * jnp.abs(directional_deriv):
                return alpha
        
        alpha *= 0.5
    
    return alpha


def train_lbfgs(model,
                config: dict,
                loss_fn: Callable,
                max_iterations: Optional[int] = None,
                checkpoint_fn: Optional[Callable] = None,
                verbose: bool = True) -> Tuple:
    """Train model with L-BFGS optimizer.
    
    Args:
        model: Initial SIREN model (typically pre-trained with Adam)
        config: Configuration dictionary
        loss_fn: Loss function
        max_iterations: Maximum iterations (overrides config)
        checkpoint_fn: Optional checkpointing callback
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Enable float64
    jax.config.update("jax_enable_x64", True)
    
    # Training config
    lbfgs_config = config['lbfgs']
    max_iter = max_iterations if max_iterations is not None else lbfgs_config['max_iterations']
    tolerance = lbfgs_config['tolerance']
    history_size = lbfgs_config['history_size']
    
    # Sampling config
    sampling_config = config['sampling']
    n_interior = sampling_config['n_interior']
    n_boundary = sampling_config['n_boundary']
    n_boundary_per_face = n_boundary // 6
    seed = sampling_config['seed']
    
    # PDE config
    k_train_grid = get_k_train_grid(config)
    k_min = config['pde']['k_train_min']
    k_max = config['pde']['k_train_max']
    
    # Sample fixed collocation points for L-BFGS (cycle through k values to save memory)
    # Use only one k value at a time instead of all simultaneously
    k_idx = 0  # Start with first k value
    k_physical = k_train_grid[k_idx]
    
    if verbose:
        print("=" * 80)
        print("L-BFGS TRAINING")
        print("=" * 80)
        print(f"Max iterations: {max_iter}")
        print(f"Tolerance: {tolerance}")
        print(f"History size: {history_size}")
        print(f"Using 1 k value at a time (cycling through {len(k_train_grid)} values)")
        print("=" * 80)
    
    def get_collocation_data(k_idx):
        """Get collocation data for a specific k index."""
        k_phys = k_train_grid[k_idx]
        step_seed = seed + 100000 + k_idx
        x_int = sample_interior_sobol(n_interior, seed=step_seed)
        x_bnd = sample_boundary_sobol(n_boundary_per_face, seed=step_seed)
        x_int_scaled = scale_to_input_range(x_int)
        x_bnd_scaled = scale_to_input_range(x_bnd)
        k_scaled = scale_k_to_input_range(k_phys, k_min, k_max)
        return x_int_scaled, x_bnd_scaled, k_scaled, k_phys
    
    # Define loss and gradient function for current k value
    @eqx.filter_jit
    def compute_total_loss_and_grad(model, x_interior, x_boundary, k_scaled, k_physical):
        """Compute loss and gradients for current k value."""
        loss_fn_k = lambda m: loss_fn(m, x_interior, x_boundary, k_scaled, k_physical)[0]
        loss, grads = eqx.filter_value_and_grad(loss_fn_k)(model)
        return loss, grads
    
    # Flatten model parameters
    params = eqx.filter(model, eqx.is_array)
    params_flat, treedef, shapes = flatten_pytree(params)
    
    # Get initial collocation data
    x_interior, x_boundary, k_scaled, k_physical = get_collocation_data(k_idx)
    
    def loss_grad_flat(params_flat):
        """Loss and gradient as function of flat parameters."""
        params_tree = unflatten_pytree(params_flat, treedef, shapes)
        model_temp = eqx.combine(params_tree, model)
        loss, grad_tree = compute_total_loss_and_grad(model_temp, x_interior, x_boundary, k_scaled, k_physical)
        grad_flat, _, _ = flatten_pytree(grad_tree)
        return loss, grad_flat
    
    # L-BFGS history
    s_hist = []
    y_hist = []
    
    # Training history
    history = {
        'iteration': [],
        'loss': [],
        'grad_norm': [],
        'time': []
    }
    
    start_time = time.time()
    loss_prev = float('inf')
    
    for iteration in range(max_iter):
        # Cycle through k values every 100 iterations to get better coverage
        if iteration % 100 == 0 and iteration > 0:
            k_idx = (k_idx + 1) % len(k_train_grid)
            x_interior, x_boundary, k_scaled, k_physical = get_collocation_data(k_idx)
        
        # Compute loss and gradient
        loss, grad = loss_grad_flat(params_flat)
        grad_norm = jnp.linalg.norm(grad)
        
        # Logging
        elapsed = time.time() - start_time
        history['iteration'].append(iteration)
        history['loss'].append(float(loss))
        history['grad_norm'].append(float(grad_norm))
        history['time'].append(elapsed)
        
        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:5d} | Loss: {loss:.6e} | Grad norm: {grad_norm:.6e} | Time: {elapsed:.1f}s")
        
        # Check convergence
        if iteration > 0:
            rel_improvement = abs(loss_prev - loss) / (abs(loss_prev) + 1e-10)
            if rel_improvement < tolerance:
                if verbose:
                    print(f"Converged: relative improvement {rel_improvement:.6e} < {tolerance}")
                break
        
        loss_prev = loss
        
        # Compute search direction
        if len(s_hist) == 0:
            # First iteration: steepest descent
            direction = -grad
        else:
            # L-BFGS direction
            gamma = jnp.dot(s_hist[-1], y_hist[-1]) / (jnp.dot(y_hist[-1], y_hist[-1]) + 1e-10)
            direction = lbfgs_two_loop_recursion(s_hist, y_hist, grad, gamma)
        
        # Line search
        alpha = backtracking_line_search(loss_grad_flat, params_flat, grad, direction, loss)
        
        # Update parameters
        params_flat_new = params_flat + alpha * direction
        
        # Compute new gradient for history update
        _, grad_new = loss_grad_flat(params_flat_new)
        
        # Update L-BFGS history
        s = params_flat_new - params_flat
        y = grad_new - grad
        
        if len(s_hist) >= history_size:
            s_hist.pop(0)
            y_hist.pop(0)
        
        s_hist.append(s)
        y_hist.append(y)
        
        params_flat = params_flat_new
        
        # Checkpointing
        if checkpoint_fn is not None and iteration % 100 == 0:
            params_tree = unflatten_pytree(params_flat, treedef, shapes)
            model_temp = eqx.combine(params_tree, model)
            checkpoint_fn(model_temp, iteration, loss, {})
    
    # Final model
    params_tree = unflatten_pytree(params_flat, treedef, shapes)
    model_final = eqx.combine(params_tree, model)
    
    if verbose:
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"L-BFGS TRAINING COMPLETE")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final loss: {loss:.6e}")
        print(f"Iterations: {iteration + 1}")
        print("=" * 80)
    
    return model_final, history
