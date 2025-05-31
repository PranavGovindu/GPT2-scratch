from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import os

DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DEFAULT_PROJECT_ROOT)  


@dataclass
class TrainingArgs:
    # Data paths
    data_path: str = f"{DEFAULT_PROJECT_ROOT}/train.txt"
    val_data_path: Optional[str] = f"{DEFAULT_PROJECT_ROOT}/validation.txt"
    output_dir_base: str = f"{DEFAULT_PROJECT_ROOT}/checkpoints"
    runs_dir_base: str = f"{PARENT_DIR}/runs"
    run_name: Optional[str] = field(default_factory=lambda: f"moe_run_{int(time.time())}")
    
    # Model configuration
    model_config_overrides: Dict[str, Any] = field(default_factory=dict)
    retrain_tokenizer: bool = False
    
    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1  # Better default for MoE models
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"
    warmup_steps_ratio: float = 0.03  # 3% for MoE stability
    
    # Evaluation and logging
    eval_every_n_steps: int = 250
    log_every_n_steps: int = 1
    generate_every_n_steps: int = 250
    num_samples_to_generate: int = 1
    max_gen_len: int = 64
    val_split_ratio: float = 0.1
    
    # Checkpointing
    save_every_n_steps: int = 500
    save_every_epoch: bool = True
    early_stopping_patience: int = 5
    
    # Hardware and optimization
    device: str = "cuda"
    use_bf16: bool = True
    seed: int = 42
    num_workers: int = 0
    
    # Output format
    save_hf_format: bool = True
    
    # debugging
    use_tqdm: bool = True
    estimate_training_reqs: bool = True
    
    # MoE specific
    aux_loss_weight: float = 0.1
    
    # Generation settings
    generation_temperature: float = 0.8
    generation_top_k: Optional[int] = 50
