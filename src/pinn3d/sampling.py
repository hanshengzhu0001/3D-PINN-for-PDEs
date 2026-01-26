"""Deterministic sampling using Sobol sequences for 3D PINN."""

import jax.numpy as jnp
from scipy.stats import qmc
import numpy as np
from typing import Tuple


def sample_interior_sobol(n_points: int, seed: int = 42) -> jnp.ndarray:
    """Sample interior points from unit cube using Sobol sequence.
    
    Args:
        n_points: Number of points to sample
        seed: Random seed for determinism
        
    Returns:
        Array of shape (n_points, 3) with coordinates in [0, 1]^3
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sampler.random(n_points)
    return jnp.array(points, dtype=jnp.float64)


def sample_boundary_sobol(n_points_per_face: int, seed: int = 42) -> jnp.ndarray:
    """Sample boundary points from cube faces using Sobol sequence.
    
    Distributes points evenly across 6 faces of the unit cube.
    
    Args:
        n_points_per_face: Number of points per face
        seed: Random seed for determinism
        
    Returns:
        Array of shape (6*n_points_per_face, 3) with boundary coordinates
    """
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    face_points_2d = sampler.random(n_points_per_face)
    
    all_boundary_points = []
    
    # Face 1: x = 0
    face1 = np.zeros((n_points_per_face, 3))
    face1[:, 0] = 0.0
    face1[:, 1:3] = face_points_2d
    all_boundary_points.append(face1)
    
    # Face 2: x = 1
    face2 = np.zeros((n_points_per_face, 3))
    face2[:, 0] = 1.0
    face2[:, 1:3] = face_points_2d
    all_boundary_points.append(face2)
    
    # Face 3: y = 0
    face3 = np.zeros((n_points_per_face, 3))
    face3[:, 0] = face_points_2d[:, 0]
    face3[:, 1] = 0.0
    face3[:, 2] = face_points_2d[:, 1]
    all_boundary_points.append(face3)
    
    # Face 4: y = 1
    face4 = np.zeros((n_points_per_face, 3))
    face4[:, 0] = face_points_2d[:, 0]
    face4[:, 1] = 1.0
    face4[:, 2] = face_points_2d[:, 1]
    all_boundary_points.append(face4)
    
    # Face 5: z = 0
    face5 = np.zeros((n_points_per_face, 3))
    face5[:, 0:2] = face_points_2d
    face5[:, 2] = 0.0
    all_boundary_points.append(face5)
    
    # Face 6: z = 1
    face6 = np.zeros((n_points_per_face, 3))
    face6[:, 0:2] = face_points_2d
    face6[:, 2] = 1.0
    all_boundary_points.append(face6)
    
    boundary_points = np.vstack(all_boundary_points)
    return jnp.array(boundary_points, dtype=jnp.float64)


def scale_to_input_range(points: jnp.ndarray) -> jnp.ndarray:
    """Scale points from [0, 1] to [-1, 1] for network input.
    
    Args:
        points: Points in [0, 1]^3
        
    Returns:
        Scaled points in [-1, 1]^3
    """
    return 2.0 * points - 1.0


def scale_k_to_input_range(k: float, k_min: float, k_max: float) -> float:
    """Scale wavenumber k to [-1, 1] for network input.
    
    Args:
        k: Wavenumber value
        k_min: Minimum k in training range
        k_max: Maximum k in training range
        
    Returns:
        Scaled k in [-1, 1]
    """
    return 2.0 * (k - k_min) / (k_max - k_min) - 1.0
