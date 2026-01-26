"""Model checkpointing utilities."""

import equinox as eqx
import json
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_checkpoint_id = None
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': []}
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(self, model, step: int, loss: float, info: Dict, 
             prefix: str = "checkpoint") -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            step: Training step
            loss: Loss value
            info: Additional info dictionary
            prefix: Checkpoint file prefix
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{prefix}_step{step}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.eqx"
        
        # Save model
        eqx.tree_serialise_leaves(checkpoint_path, model)
        
        # Update metadata
        checkpoint_meta = {
            'id': checkpoint_id,
            'step': step,
            'loss': float(loss),
            'info': {k: float(v) if hasattr(v, 'item') else v for k, v in info.items()},
            'path': str(checkpoint_path)
        }
        self.metadata['checkpoints'].append(checkpoint_meta)
        self._save_metadata()
        
        # Track best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_checkpoint_id = checkpoint_id
            # Save best model separately
            best_path = self.checkpoint_dir / "best_model.eqx"
            eqx.tree_serialise_leaves(best_path, model)
            self.metadata['best'] = checkpoint_meta
            self._save_metadata()
        
        return checkpoint_id
    
    def load(self, model_template, checkpoint_id: Optional[str] = None) -> Any:
        """Load model checkpoint.
        
        Args:
            model_template: Template model with correct structure
            checkpoint_id: Checkpoint ID to load (loads best if None)
            
        Returns:
            Loaded model
        """
        if checkpoint_id is None:
            # Load best model
            checkpoint_path = self.checkpoint_dir / "best_model.eqx"
            if not checkpoint_path.exists():
                raise FileNotFoundError("No best model checkpoint found")
        else:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.eqx"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        model = eqx.tree_deserialise_leaves(checkpoint_path, model_template)
        return model
    
    def get_best_checkpoint_id(self) -> Optional[str]:
        """Get best checkpoint ID.
        
        Returns:
            Best checkpoint ID or None
        """
        return self.best_checkpoint_id
    
    def list_checkpoints(self) -> list:
        """List all checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        return self.metadata.get('checkpoints', [])
