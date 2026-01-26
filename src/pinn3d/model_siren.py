"""SIREN network implementation using Equinox."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class SirenLayer(eqx.Module):
    """Single SIREN layer with sine activation."""
    
    weight: jnp.ndarray
    bias: jnp.ndarray
    omega: float
    
    def __init__(self, in_features: int, out_features: int, omega: float = 1.0, 
                 is_first: bool = False, key: Optional[jax.random.PRNGKey] = None):
        """Initialize SIREN layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega: Frequency multiplier for sine activation
            is_first: Whether this is the first layer (different init)
            key: PRNG key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        self.omega = omega
        
        # SIREN initialization
        if is_first:
            # First layer: uniform in [-1/n, 1/n]
            bound = 1.0 / in_features
        else:
            # Hidden layers: uniform in [-sqrt(6/n)/omega, sqrt(6/n)/omega]
            bound = jnp.sqrt(6.0 / in_features) / omega
        
        w_key, b_key = jax.random.split(key)
        self.weight = jax.random.uniform(
            w_key, 
            shape=(out_features, in_features),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float64
        )
        self.bias = jax.random.uniform(
            b_key,
            shape=(out_features,),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float64
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with sine activation.
        
        Args:
            x: Input of shape (..., in_features)
            
        Returns:
            Output of shape (..., out_features)
        """
        linear = jnp.dot(x, self.weight.T) + self.bias
        return jnp.sin(self.omega * linear)


class SirenNetwork(eqx.Module):
    """SIREN network for 3D+k input to scalar output."""
    
    layers: list
    final_layer: eqx.nn.Linear
    
    def __init__(self, hidden_layers: int = 8, width: int = 256, 
                 omega_0: float = 30.0, key: Optional[jax.random.PRNGKey] = None):
        """Initialize SIREN network.
        
        Architecture: 4 -> [256]*8 -> 1 with sine activations
        
        Args:
            hidden_layers: Number of hidden layers
            width: Width of hidden layers
            omega_0: First-layer frequency multiplier
            key: PRNG key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, hidden_layers + 1)
        
        # Input is (x, y, z, k) scaled to [-1, 1]
        in_features = 4
        
        # Build layers
        self.layers = []
        
        # First layer
        self.layers.append(
            SirenLayer(in_features, width, omega=omega_0, is_first=True, key=keys[0])
        )
        
        # Hidden layers
        for i in range(1, hidden_layers):
            self.layers.append(
                SirenLayer(width, width, omega=1.0, is_first=False, key=keys[i])
            )
        
        # Final linear layer (no activation)
        # Initialize with same scheme as hidden layers
        bound = jnp.sqrt(6.0 / width)
        w_key, b_key = jax.random.split(keys[-1])
        weight = jax.random.uniform(
            w_key,
            shape=(1, width),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float64
        )
        bias = jax.random.uniform(
            b_key,
            shape=(1,),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float64
        )
        
        # Create Linear layer with initialized weights
        self.final_layer = eqx.nn.Linear(width, 1, use_bias=True, key=keys[-1])
        # Override with our initialized weights
        self.final_layer = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.final_layer,
            (weight, bias)
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            x: Input of shape (..., 4) where last dim is (x, y, z, k) in [-1, 1]
            
        Returns:
            Output of shape (...,) scalar field value
        """
        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)
        
        # Final linear layer
        x = self.final_layer(x)
        
        # Squeeze last dimension
        return jnp.squeeze(x, axis=-1)


def create_model(config: dict, key: Optional[jax.random.PRNGKey] = None) -> SirenNetwork:
    """Create SIREN model from configuration.
    
    Args:
        config: Configuration dictionary
        key: PRNG key (uses seed from config if None)
        
    Returns:
        Initialized SIREN network
    """
    if key is None:
        seed = config['sampling']['seed']
        key = jax.random.PRNGKey(seed)
    
    network_config = config['network']
    
    return SirenNetwork(
        hidden_layers=network_config['hidden_layers'],
        width=network_config['width'],
        omega_0=network_config['omega_0'],
        key=key
    )
