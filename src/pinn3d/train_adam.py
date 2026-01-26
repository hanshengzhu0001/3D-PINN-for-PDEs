"""Adam training loop for PINN."""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Callable, Tuple, Optional
from .sampling import sample_interior_sobol, sample_boundary_sobol, scale_to_input_range, scale_k_to_input_range
from .config import get_k_train_grid
import time


def cosine_decay_schedule(lr_init: float, lr_min: float, total_steps: int):
    """Create cosine decay learning rate schedule.
    
    Args:
        lr_init: Initial learning rate
        lr_min: Minimum learning rate
        total_steps: Total number of training steps
        
    Returns:
        Optax learning rate schedule
    """
    return optax.cosine_decay_schedule(
        init_value=lr_init,
        decay_steps=total_steps,
        alpha=lr_min / lr_init
    )


def train_adam(model,
               config: dict,
               loss_fn: Callable,
               steps: Optional[int] = None,
               checkpoint_fn: Optional[Callable] = None,
               verbose: bool = True) -> Tuple:
    """Train model with Adam optimizer.
    
    Args:
        model: Initial SIREN model
        config: Configuration dictionary
        loss_fn: Loss function
        steps: Number of training steps (overrides config)
        checkpoint_fn: Optional checkpointing callback
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Enable float64
    jax.config.update("jax_enable_x64", True)
    
    # Training config
    adam_config = config['adam']
    n_steps = steps if steps is not None else adam_config['steps']
    lr_init = adam_config['learning_rate']
    lr_min = adam_config['lr_min']
    grad_clip = adam_config['gradient_clip']
    
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
    
    # Setup optimizer
    schedule = cosine_decay_schedule(lr_init, lr_min, n_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # JIT compile loss and gradient computation
    @eqx.filter_jit
    def compute_loss_and_grad(model, x_interior, x_boundary, k_scaled, k_physical):
        loss_fn_with_model = lambda m: loss_fn(m, x_interior, x_boundary, k_scaled, k_physical)[0]
        loss, grads = eqx.filter_value_and_grad(loss_fn_with_model)(model)
        _, info = loss_fn(model, x_interior, x_boundary, k_scaled, k_physical)
        return loss, grads, info
    
    @eqx.filter_jit
    def update_step(model, opt_state, x_interior, x_boundary, k_scaled, k_physical):
        loss, grads, info = compute_loss_and_grad(model, x_interior, x_boundary, k_scaled, k_physical)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, info
    
    # Training history
    history = {
        'step': [],
        'loss': [],
        'loss_boundary': [],
        'loss_residual': [],
        'k_value': [],
        'time': []
    }
    
    if verbose:
        print("=" * 80)
        print("ADAM TRAINING")
        print("=" * 80)
        print(f"Steps: {n_steps}")
        print(f"Learning rate: {lr_init} -> {lr_min} (cosine decay)")
        print(f"Interior points per step: {n_interior}")
        print(f"Boundary points per step: {n_boundary}")
        print(f"K training grid: {len(k_train_grid)} values")
        print("=" * 80)
    
    start_time = time.time()
    best_loss = float('inf')
    
    for step in range(n_steps):
        # Cycle through k values deterministically
        k_idx = step % len(k_train_grid)
        k_physical = k_train_grid[k_idx]
        k_scaled = scale_k_to_input_range(k_physical, k_min, k_max)
        
        # Sample points (deterministic via seed + step)
        step_seed = seed + step
        x_interior = sample_interior_sobol(n_interior, seed=step_seed)
        x_boundary = sample_boundary_sobol(n_boundary_per_face, seed=step_seed)
        
        # Scale to network input range [-1, 1]
        x_interior_scaled = scale_to_input_range(x_interior)
        x_boundary_scaled = scale_to_input_range(x_boundary)
        
        # Update step
        model, opt_state, loss, info = update_step(
            model, opt_state, x_interior_scaled, x_boundary_scaled, k_scaled, k_physical
        )
        
        # Track best model
        if loss < best_loss:
            best_loss = loss
        
        # Logging
        if step % 100 == 0 or step == n_steps - 1:
            elapsed = time.time() - start_time
            history['step'].append(step)
            history['loss'].append(float(loss))
            history['loss_boundary'].append(float(info['loss_boundary']))
            history['loss_residual'].append(float(info['loss_residual']))
            history['k_value'].append(float(k_physical))
            history['time'].append(elapsed)
            
            if verbose and step % 1000 == 0:
                print(f"Step {step:6d} | Loss: {loss:.6e} | "
                      f"L_b: {info['loss_boundary']:.6e} | "
                      f"L_f: {info['loss_residual']:.6e} | "
                      f"k: {k_physical:.3f} | "
                      f"Time: {elapsed:.1f}s")
        
        # Checkpointing
        if checkpoint_fn is not None and step % config['checkpoints']['save_frequency'] == 0:
            checkpoint_fn(model, step, loss, info)
    
    if verbose:
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"ADAM TRAINING COMPLETE")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final loss: {loss:.6e}")
        print(f"Best loss: {best_loss:.6e}")
        print("=" * 80)
    
    return model, history
