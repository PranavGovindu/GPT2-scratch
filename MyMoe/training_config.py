# training_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import time
import os

# Default paths - consider making these relative or configurable via CLI
# Get parent directory of MyMoe to access shared resources
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DEFAULT_PROJECT_ROOT)  # This will be GPT2-scratch

@dataclass
class TrainingArgs:    # Data and paths
    data_path: str = f"{DEFAULT_PROJECT_ROOT}/train.txt"
    val_data_path: Optional[str] =f"{DEFAULT_PROJECT_ROOT}/validation.txt"  # Optional validation data path
    output_dir_base: str = f"{DEFAULT_PROJECT_ROOT}/checkpoints"
    runs_dir_base: str = f"{PARENT_DIR}/runs"  # TensorBoard logs in parent directory
    run_name: Optional[str] = field(default_factory=lambda: "moe_run")

    # Model loading
    model_config_overrides: Optional[Dict] = field(default_factory=dict)
    retrain_tokenizer: bool = False

    # Training hyperparams
    epochs: int = 5
    batch_size: int = 4 
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0
    
    # Scheduler
    lr_scheduler_type: str = "cosine" # Options: "cosine", "linear", "none" / "constant"
    warmup_steps_ratio: float = 0.05

    # Validation and logging
    eval_every_n_steps: int = 250 # Set to a lower value for small datasets, e.g., 10 or 1
    log_every_n_steps: int = 1   # e.g. 1
    generate_every_n_steps: int = 250 # e.g. 10 or 2
    num_samples_to_generate: int = 1
    max_gen_len: int = 64
    val_split_ratio: float = 0.1
    save_every_n_steps: int = 500 # Save model every 500 steps
    save_every_epoch: bool = True # Save model at the end of each epoch

    # Early stopping
    early_stopping_patience: int = 5 # Number of evaluations without improvement
    
    # Hardware
    # device will be determined in train.py based on availability if "cuda" is set
    device: str = "cuda" # "cuda" or "cpu"
    use_bf16: bool = True 
    use_fp16: bool = False

    # Reproducibility
    seed: int = 42

    # Hugging Face saving structure
    save_hf_format: bool = True

    # DataLoader
    num_workers: int = 0 # Set to > 0 if you want parallel data loading
    model_config_overrides: Dict[str, any] = field(default_factory=dict)
    use_tqdm: bool = True # Use tqdm for progress bars
    estimate_training_reqs: bool = True # Estimate total training steps for progress bar
    aux_loss_weight: float = 0.1 # Weight for auxiliary loss, if applicable

        # sampling
    generation_prompt: Optional[str] = "Once upon a time" # Default prompt, or None to start with BOS
    generation_temperature: float = 0.8 # Sampling temperature
    generation_top_k: Optional[int] = 50 # Top-k filtering
    max_gen_len: int = 200# Maximum length of generated sequences
    
    # sampling
    generation_prompt: Optional[str] = "Once upon a time" # Default prompt, or None to start with BOS
    generation_temperature: float = 0.8 # Sampling temperature
    generation_top_k: Optional[int] = 50 # Top-k filtering
    max_gen_len: int = 200# Maximum length of generated sequences
