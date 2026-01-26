"""Configuration management for 3D PINN."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/helmholtz_cube.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_k_train_grid(config: Dict[str, Any]) -> list:
    """Generate deterministic k training grid.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of k values for training
    """
    import numpy as np
    
    k_min = config['pde']['k_train_min']
    k_max = config['pde']['k_train_max']
    n_k = config['pde']['n_k_train']
    
    return np.linspace(k_min, k_max, n_k).tolist()
