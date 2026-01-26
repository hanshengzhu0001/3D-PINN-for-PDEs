"""FastAPI server for PINN inference and training."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import threading
import time

from .config import load_config
from .source import get_source_fn
from .model_siren import create_model
from .loss import make_loss_fn
from .train_adam import train_adam
from .train_lbfgs import train_lbfgs
from .validate_fd import validate_with_fd, validate_all_k_values
from .checkpoints import CheckpointManager
from .sampling import scale_to_input_range, scale_k_to_input_range
from .pde import batch_prediction


# Global state
app = FastAPI(title="3D PINN Helmholtz Solver", version="0.1.0")

class ServerState:
    """Global server state."""
    def __init__(self):
        self.model = None
        self.config = None
        self.source_fn = None
        self.checkpoint_manager = None
        self.training_status = {
            'is_training': False,
            'stage': None,
            'epoch': 0,
            'loss': None,
            'best_checkpoint_id': None,
            'start_time': None,
            'elapsed_time': 0
        }
        self.training_lock = threading.Lock()

state = ServerState()


# Request/Response models
class TrainRequest(BaseModel):
    """Training request."""
    config_name: str = Field(default="helmholtz_cube", description="Configuration file name")
    adam_steps: Optional[int] = Field(default=None, description="Override Adam steps")
    lbfgs_iterations: Optional[int] = Field(default=None, description="Override L-BFGS iterations")


class QueryRequest(BaseModel):
    """Query request for PINN predictions."""
    k: float = Field(description="Wavenumber")
    points: List[List[float]] = Field(description="List of [x, y, z] coordinates in [0, 1]")


class QueryResponse(BaseModel):
    """Query response with predictions."""
    k: float
    predictions: List[float]
    computation_time: float


class TrainingStatusResponse(BaseModel):
    """Training status response."""
    is_training: bool
    stage: Optional[str]
    epoch: int
    loss: Optional[float]
    best_checkpoint_id: Optional[str]
    elapsed_time: float


class ValidationRequest(BaseModel):
    """Validation request."""
    k: Optional[float] = Field(default=None, description="Wavenumber (validates all if None)")


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "config_loaded": state.config is not None
    }


@app.post("/train/start")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start training in background."""
    with state.training_lock:
        if state.training_status['is_training']:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Load config
        config_path = f"configs/{request.config_name}.yaml"
        try:
            state.config = load_config(config_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        
        # Initialize components
        state.source_fn = get_source_fn(
            state.config['source']['center'],
            state.config['source']['width'],
            state.config['source']['amplitude']
        )
        
        state.checkpoint_manager = CheckpointManager(
            state.config['checkpoints']['directory']
        )
        
        # Mark training as started
        state.training_status['is_training'] = True
        state.training_status['stage'] = 'initializing'
        state.training_status['epoch'] = 0
        state.training_status['loss'] = None
        state.training_status['start_time'] = time.time()
    
    # Run training in background
    background_tasks.add_task(
        run_training, 
        request.adam_steps, 
        request.lbfgs_iterations
    )
    
    return {
        "status": "Training started",
        "config": request.config_name
    }


def run_training(adam_steps: Optional[int], lbfgs_iterations: Optional[int]):
    """Background training task."""
    jax.config.update("jax_enable_x64", True)
    
    try:
        # Create model
        state.model = create_model(state.config)
        loss_fn = make_loss_fn(state.config, state.source_fn)
        
        # Checkpoint callback
        def checkpoint_callback(model, step, loss, info):
            state.checkpoint_manager.save(model, step, loss, info)
            state.training_status['epoch'] = step
            state.training_status['loss'] = float(loss)
            state.training_status['best_checkpoint_id'] = state.checkpoint_manager.get_best_checkpoint_id()
        
        # Stage A: Adam
        state.training_status['stage'] = 'adam'
        state.model, _ = train_adam(
            state.model,
            state.config,
            loss_fn,
            steps=adam_steps,
            checkpoint_fn=checkpoint_callback,
            verbose=True
        )
        
        # Stage B: L-BFGS
        state.training_status['stage'] = 'lbfgs'
        state.model, _ = train_lbfgs(
            state.model,
            state.config,
            loss_fn,
            max_iterations=lbfgs_iterations,
            checkpoint_fn=checkpoint_callback,
            verbose=True
        )
        
        # Save final model
        final_loss = state.training_status['loss']
        state.checkpoint_manager.save(state.model, -1, final_loss, {}, prefix="final")
        
        state.training_status['stage'] = 'completed'
        
    except Exception as e:
        state.training_status['stage'] = 'failed'
        print(f"Training failed: {e}")
        raise
    
    finally:
        state.training_status['is_training'] = False
        elapsed = time.time() - state.training_status['start_time']
        state.training_status['elapsed_time'] = elapsed


@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get training status."""
    elapsed = 0
    if state.training_status['is_training'] and state.training_status['start_time']:
        elapsed = time.time() - state.training_status['start_time']
    elif state.training_status['elapsed_time'] > 0:
        elapsed = state.training_status['elapsed_time']
    
    return TrainingStatusResponse(
        is_training=state.training_status['is_training'],
        stage=state.training_status['stage'],
        epoch=state.training_status['epoch'],
        loss=state.training_status['loss'],
        best_checkpoint_id=state.training_status['best_checkpoint_id'],
        elapsed_time=elapsed
    )


@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """Query model for predictions."""
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Train first.")
    
    if state.config is None:
        raise HTTPException(status_code=400, detail="Config not loaded")
    
    # Validate inputs
    if not request.points:
        raise HTTPException(status_code=400, detail="No points provided")
    
    for point in request.points:
        if len(point) != 3:
            raise HTTPException(status_code=400, detail="Each point must have 3 coordinates")
    
    start_time = time.time()
    
    # Convert to JAX arrays
    points_array = jnp.array(request.points, dtype=jnp.float64)
    
    # Scale to network input range
    points_scaled = scale_to_input_range(points_array)
    
    # Scale k
    k_min = state.config['pde']['k_train_min']
    k_max = state.config['pde']['k_train_max']
    k_scaled = scale_k_to_input_range(request.k, k_min, k_max)
    
    # Predict
    predictions = batch_prediction(state.model, points_scaled, k_scaled)
    predictions_list = [float(p) for p in predictions]
    
    computation_time = time.time() - start_time
    
    return QueryResponse(
        k=request.k,
        predictions=predictions_list,
        computation_time=computation_time
    )


@app.post("/validate")
async def validate_model(request: ValidationRequest):
    """Validate model against finite difference solution."""
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Train first.")
    
    if state.config is None or state.source_fn is None:
        raise HTTPException(status_code=400, detail="Config or source function not loaded")
    
    if request.k is None:
        # Validate all k values
        metrics = validate_all_k_values(state.model, state.config, state.source_fn, verbose=False)
    else:
        # Validate single k value
        metrics = validate_with_fd(state.model, state.config, state.source_fn, request.k, verbose=False)
    
    return metrics


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if state.config is None:
        raise HTTPException(status_code=400, detail="Config not loaded")
    
    return state.config


@app.get("/checkpoints")
async def list_checkpoints():
    """List available checkpoints."""
    if state.checkpoint_manager is None:
        raise HTTPException(status_code=400, detail="Checkpoint manager not initialized")
    
    return {
        "checkpoints": state.checkpoint_manager.list_checkpoints(),
        "best_checkpoint_id": state.checkpoint_manager.get_best_checkpoint_id()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
