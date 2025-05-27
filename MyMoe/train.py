#!/usr/bin/env python3
"""
Improved LLM Training Script
Simplified, cleaner, with comprehensive logging and sample generation
"""

import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import sys
import webbrowser
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import safetensors.torch
from tqdm import tqdm

# --- Module Imports ---
# Ensure these can be imported. If not, the script will fail early.
try:
    from config import ModelArgs 
    from Transformer_Block import LLM 
    from BpeTokenizer import get_bpe_tokenizer, BPEDataset, SPECIAL_TOKENS 
    from training_config import TrainingArgs 
except ImportError as e:
    print(f"FATAL: Failed to import a required module: {e}", file=sys.stderr)
    print("Please ensure config.py, Transformer_Block.py, BpeTokenizer.py, and training_config.py are in PYTHONPATH or the current directory.", file=sys.stderr)
    sys.exit(1)


def launch_tensorboard(log_dir: str, port: int = 6006):
    """Launch TensorBoard in a background process."""
    tensorboard_url = f"http://localhost:{port}"
    print(f"[Setup] Attempting to launch TensorBoard on {tensorboard_url} for logdir: {log_dir}")
    try:
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3) # Give TensorBoard a moment to start
        if tb_process.poll() is None: # Check if process started
            print(f"[Setup] TensorBoard process started (PID: {tb_process.pid}). Attempting to open browser...")
            try:
                webbrowser.open(tensorboard_url)
            except webbrowser.Error:
                print(f"[Setup] Could not open browser automatically. Please navigate to {tensorboard_url}")
            return tb_process
        else:
            print(f"[Setup] TensorBoard process failed to start. Return code: {tb_process.returncode}")
            stderr_output = tb_process.stderr.read().decode(errors='ignore')
            if stderr_output:
                print(f"[Setup] TensorBoard stderr:\n{stderr_output}")
            return None
    except FileNotFoundError:
        print("[Setup] Error: 'tensorboard' command not found. Is TensorBoard installed and in your PATH?", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Setup] Error launching TensorBoard: {e}", file=sys.stderr)
        return None

@dataclass
class TrainingState:
    """Track training state"""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    epochs_no_improve: int = 0 
    train_loss_sum: float = 0.0 
    train_tokens: int = 0 

class ModelTrainer:
    """Main training class"""

    def __init__(self, args: TrainingArgs):
        self.args = args
        self.state = TrainingState()
        self.avg_tokens_per_batch_estimate: float = 0.0 

        self._initialize_paths_and_create_dirs() # Creates dirs needed for logging
        self.setup_logging() # Call before other logs
        
        self.logger.info("ModelTrainer initialized. Args validated (implicitly by dataclass).")
        self._log_directory_info_and_save_configs()
        self.setup_device()

        self.tokenizer = None
        self.model: Optional[LLM] = None
        self.pad_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.writer: Optional[SummaryWriter] = None
        self.training_start_time: Optional[float] = None 
        self.tb_process: Optional[subprocess.Popen] = None


    def _initialize_paths_and_create_dirs(self):
        """Define paths and create physical directories."""
        print(f"[Setup] Initializing paths. Run name: {self.args.run_name}")
        self.output_dir = Path(self.args.output_dir_base) / self.args.run_name
        self.tb_dir = Path(self.args.runs_dir_base) / self.args.run_name
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Setup] Output directory created/exists: {self.output_dir}")
            print(f"[Setup] TensorBoard directory created/exists: {self.tb_dir}")
        except OSError as e:
            print(f"FATAL: Could not create directories {self.output_dir} or {self.tb_dir}: {e}", file=sys.stderr)
            sys.exit(1)
    
    def setup_logging(self):
        log_file_path = self.output_dir / 'training.log'
        print(f"[Setup] Setting up logging. Log file will be: {log_file_path}")
        
        # Create logger instance for this trainer
        self.logger = logging.getLogger(__name__) 
        self.logger.setLevel(logging.INFO) 
        self.logger.propagate = False # Important: prevent messages going to root logger's handlers

        # Clear any existing handlers from THIS logger instance
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logger at {log_file_path}: {e}", file=sys.stderr)
        
        # Stream Handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter('%(message)s') 
        )
        self.logger.addHandler(stream_handler)

        try:
            if hasattr(stream_handler.stream, 'reconfigure'):
                 stream_handler.stream.reconfigure(encoding='utf-8')
        except Exception: 
            pass 

        self.logger.info(f"Logging initialized. Log file: {log_file_path}")


    def _log_directory_info_and_save_configs(self):
        """Logs information about directories and saves training arguments."""
        self.logger.info(f"Output directory: {self.output_dir.resolve()}")
        self.logger.info(f"Tensorboard directory: {self.tb_dir.resolve()}")
        config_save_path = self.output_dir / "training_args.json"
        try:
            with open(config_save_path, "w", encoding='utf-8') as f:
                json.dump(asdict(self.args), f, indent=2)
            self.logger.info(f"Training arguments saved to {config_save_path}")
        except Exception as e:
            self.logger.error(f"Could not save training_args.json: {e}")


    def setup_device(self):
        self.logger.info(f"Requested device: {self.args.device}")
        if self.args.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info(f"CUDA is available. Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.args.device = "cpu" # Update args to reflect actual device
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Effective device: {self.device}")

        self.mixed_precision_dtype = None
        self.scaler = None
        if self.device.type == 'cuda':
            if self.args.use_bf16:
                if torch.cuda.is_bf16_supported():
                    self.mixed_precision_dtype = torch.bfloat16
                    self.logger.info("Using bfloat16 mixed precision. Tensor Cores will be leveraged.")
                else:
                    self.logger.warning("BF16 requested but not supported by this CUDA device. Will use FP32.")
            elif self.args.use_fp16: # Only if BF16 is not used/supported and FP16 is requested
                self.mixed_precision_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
                self.logger.info("Using float16 mixed precision with GradScaler. Tensor Cores will be leveraged.")
            else:
                self.logger.info("Not using BF16 or FP16. Training in FP32.")
        else: # CPU
             self.logger.info("Device is CPU. Mixed precision (BF16/FP16) is typically for CUDA.")


    def setup_model_and_tokenizer(self):
        self.logger.info("Setting up model and tokenizer...")
        model_args_dict = self.args.model_config_overrides if isinstance(self.args.model_config_overrides, dict) else {}
        
        try:
            model_args_obj = ModelArgs(**model_args_dict)
        except TypeError as e:
            self.logger.error(f"Error creating ModelArgs with overrides: {e}. Check ModelArgs definition and overrides.")
            self.logger.warning("Falling back to default ModelArgs.")
            model_args_obj = ModelArgs()

        try:
            self.tokenizer, updated_model_args_obj = get_bpe_tokenizer(
                model_args_obj, self.args.data_path, self.args.retrain_tokenizer
            )
        except Exception as e:
            self.logger.error(f"Failed to get BPE tokenizer: {e}", exc_info=True)
            raise
        
        model_config_path = self.output_dir / "model_config.json"
        try:
            with open(model_config_path, "w", encoding='utf-8') as f:
                json.dump(asdict(updated_model_args_obj), f, indent=2)
            self.logger.info(f"Model config saved to {model_config_path}")
        except Exception as e:
            self.logger.error(f"Could not save model_config.json: {e}")
        
        try:
            self.model = LLM(updated_model_args_obj).to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM model: {e}", exc_info=True)
            raise

        # SPECIAL_TOKENS: 0: <unk>, 1: <eos>, 2: <bos>, 3: <pad>
        try:
            self.pad_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[3]) 
            self.bos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[2]) 
            self.eos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[1]) 
        except Exception as e:
            self.logger.error(f"Failed to get IDs for special tokens from tokenizer: {e}", exc_info=True)
            raise

        if self.pad_token_id is None or self.bos_token_id is None or self.eos_token_id is None:
            msg = "CRITICAL: One or more special tokens (PAD, BOS, EOS) are None after tokenizer lookup."
            self.logger.error(msg)
            raise ValueError(msg)
            
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {param_count:,} trainable parameters on device {self.device}.")
        return updated_model_args_obj

    def setup_data(self):
        self.logger.info("Setting up data loaders...")
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not initialized before setup_data.")
            raise RuntimeError("Model/tokenizer setup must precede data setup.")
        if not hasattr(self.model, 'args') or not hasattr(self.model.args, 'max_seq_len'):
            self.logger.error("Model.args.max_seq_len not available for BPEDataset.")
            raise AttributeError("Model configuration (model.args.max_seq_len) is missing.")

        try:
            train_dataset = BPEDataset(
                self.args.data_path, self.tokenizer, self.model.args.max_seq_len
            )
        except FileNotFoundError:
            self.logger.error(f"Training data file not found: {self.args.data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error creating training BPEDataset from {self.args.data_path}: {e}", exc_info=True)
            raise

        if len(train_dataset) == 0:
            self.logger.error(f"Training dataset is empty: {self.args.data_path}. Cannot proceed.")
            raise ValueError("Training dataset is empty.")
            
        val_dataset = None
        if self.args.val_data_path:
            try:
                val_dataset = BPEDataset(
                    self.args.val_data_path, self.tokenizer, self.model.args.max_seq_len
                )
                if len(val_dataset) == 0:
                    self.logger.warning(f"Validation dataset specified ({self.args.val_data_path}) but it's empty. No validation will be performed with it.")
                    val_dataset = None 
                else:
                    self.logger.info(f"Using separate validation data: {len(val_dataset)} samples from {self.args.val_data_path}")
            except FileNotFoundError:
                self.logger.warning(f"Validation data file specified ({self.args.val_data_path}) but not found. Proceeding without it.")
            except Exception as e:
                self.logger.warning(f"Error creating validation BPEDataset from {self.args.val_data_path}: {e}. Proceeding without it.")


        elif self.args.val_split_ratio > 0 and len(train_dataset) > 1: # Min dataset size for split
            # Ensure we have enough samples for a meaningful split
            min_samples_for_split = int(1 / self.args.val_split_ratio) if self.args.val_split_ratio > 0 else float('inf')
            min_samples_for_split = max(min_samples_for_split, 10) # Heuristic: need at least 10 samples to consider splitting

            if len(train_dataset) >= min_samples_for_split:
                val_size = max(1, int(len(train_dataset) * self.args.val_split_ratio))
                train_size = len(train_dataset) - val_size
                if train_size > 0 and val_size > 0 :
                    train_dataset_split, val_dataset_split = random_split(
                        train_dataset, [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.args.seed)
                    )
                    train_dataset = train_dataset_split # reassign
                    val_dataset = val_dataset_split   # assign
                    self.logger.info(f"Split data - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
                else:
                    self.logger.warning(f"Cannot split dataset (size {len(train_dataset)}) with ratio {self.args.val_split_ratio} to get non-empty splits. Using full for training.")
            else:
                self.logger.warning(f"Dataset too small (size {len(train_dataset)}) to split with ratio {self.args.val_split_ratio}. Need at least {min_samples_for_split}. Using full for training.")
        else:
            self.logger.info("No validation data path and val_split_ratio is 0 or dataset too small. No validation set created from training data.")


        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=(self.device.type == 'cuda'),
            drop_last=True # Good practice if batch-dependent operations exist (e.g. BatchNorm, though less common in LLMs)
        )
        self.val_loader = None
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.args.batch_size, shuffle=False, # No shuffle for val
                num_workers=self.args.num_workers, pin_memory=(self.device.type == 'cuda')
            )
        
        if not self.train_loader or len(self.train_loader) == 0 : # After drop_last, could be 0 if dataset smaller than batch_size
            self.logger.error(f"Training loader is empty after DataLoader initialization (possibly due to drop_last and small dataset). Original dataset size: {len(train_dataset)}")
            raise ValueError("Training loader is empty.")

        self.logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        if self.val_loader:
            self.logger.info(f"Validation batches: {len(self.val_loader)}")
        else:
            self.logger.info("No validation loader configured.")


    def setup_optimizer_and_scheduler(self):
        if not self.train_loader or len(self.train_loader) == 0:
             self.logger.error("Train loader not available or empty for optimizer setup. Ensure data is loaded and processed correctly.")
             raise RuntimeError("Cannot setup optimizer without a valid train_loader.")
        if self.model is None:
            self.logger.error("Model not initialized before optimizer setup.")
            raise RuntimeError("Model must be initialized before optimizer setup.")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay
        )
        # Ensure epochs is positive
        if self.args.epochs <= 0 :
            self.logger.warning(f"Number of epochs is {self.args.epochs}. Setting to 1 for scheduler calculation, but training might not run as expected.")
            effective_epochs_for_scheduler = 1
        else:
            effective_epochs_for_scheduler = self.args.epochs

        total_steps = len(self.train_loader) * effective_epochs_for_scheduler
        if total_steps == 0:
            self.logger.warning("Total_steps for scheduler is 0. Scheduler might not work as expected. Check train_loader and epochs.")
            # Fallback or raise error depending on strictness
            # For now, allow scheduler to be None or misconfigured if total_steps is 0

        warmup_steps = int(total_steps * self.args.warmup_steps_ratio) if total_steps > 0 else 0

        if self.args.lr_scheduler_type == "cosine" and total_steps > 0 and (total_steps - warmup_steps > 0):
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(1, total_steps - warmup_steps),
                eta_min=self.args.learning_rate * 0.1 
            )
        else:
            self.scheduler = None
            if self.args.lr_scheduler_type == "cosine":
                 self.logger.warning(f"Cosine scheduler requested but total_steps ({total_steps}) or T_0 ({total_steps - warmup_steps}) is too small. No scheduler will be used.")
            else:
                self.logger.info("No LR scheduler configured or type not recognized.")

        self.logger.info(f"Optimizer: AdamW. LR Scheduler: {self.args.lr_scheduler_type if self.scheduler else 'None'}.")
        self.logger.info(f"Calculated total training steps: {total_steps}. Warmup steps: {warmup_steps}.")


    @torch.no_grad()
    def _generate_sample_text(self):
        if not self.model or not self.tokenizer:
            self.logger.warning("Generation skipped: Model or tokenizer not ready.")
            return

        self.logger.info(f"--- Generating sample text (Global Step {self.state.global_step}) ---")
        self.model.eval()

        prompt_text = getattr(self.args, 'generation_prompt', "The story begins") # Default prompt
        max_gen_len = getattr(self.args, 'max_gen_len', 100)
        temperature = getattr(self.args, 'generation_temperature', 0.7) # Slightly lower default
        top_k = getattr(self.args, 'generation_top_k', 40) # Slightly lower default
        num_samples = getattr(self.args, 'num_samples_to_generate', 1)

        for i in range(num_samples):
            if prompt_text:
                try:
                    input_ids = self.tokenizer.encode(prompt_text).ids
                except Exception as e:
                    self.logger.error(f"Error encoding prompt '{prompt_text}': {e}. Skipping sample generation.")
                    continue
            else: 
                if self.bos_token_id is None:
                    self.logger.warning("BOS token ID is None and no prompt provided. Skipping sample generation.")
                    continue
                input_ids = [self.bos_token_id]
            
            if not input_ids: # Should be caught by above checks, but as a safeguard
                self.logger.warning("Input IDs for generation are empty. Skipping sample.")
                continue

            generated_ids = list(input_ids) 
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

            for _ in range(max_gen_len):
                if input_tensor.size(1) == 0: break # Should not happen
                
                # Context truncation for the model
                context_tensor = input_tensor[:, -self.model.args.max_seq_len:]
                attention_mask = torch.ones_like(context_tensor, device=self.device)

                with torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=(self.mixed_precision_dtype is not None)):
                    logits, _, _ = self.model(context_tensor, kv_context_ids=context_tensor, kv_context_mask=attention_mask)
                
                next_token_logits = logits[:, -1, :] 

                if temperature > 0:
                    scaled_logits = next_token_logits / temperature
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
                else: 
                    next_token_id = torch.argmax(next_token_logits, dim=-1)

                if self.eos_token_id is not None and next_token_id.item() == self.eos_token_id:
                    generated_ids.append(next_token_id.item()) # Append EOS before breaking
                    break
                
                generated_ids.append(next_token_id.item())
                # Update input_tensor for the next iteration (no need to unsqueeze if next_token_id is already 1D)
                input_tensor = torch.cat((input_tensor, next_token_id.view(1, 1)), dim=1)


            try:
                generated_text = self.tokenizer.decode(generated_ids)
            except Exception as e:
                self.logger.error(f"Error decoding generated IDs: {generated_ids}. Error: {e}")
                generated_text = "[Decoding Error]"

            self.logger.info(f"Sample {i+1}/{num_samples}: {generated_text}")
            if self.writer:
                # Replace newlines for better display in TensorBoard text plugin
                tb_text = generated_text.replace('\n', '  \n')
                self.writer.add_text(f'GeneratedText/Epoch{self.state.epoch+1}_Sample_{i+1}', tb_text, self.state.global_step)
        
        self.model.train() # Ensure model is back in training mode
        self.logger.info("--- Finished generating sample text ---")


    @torch.no_grad()
    def evaluate(self):
        if not self.val_loader:
            return None
        self.logger.info("Running validation...")
        self.model.eval()
        total_loss_sum = 0.0 # Sum of (batch_loss * active_tokens_in_batch)
        total_active_tokens = 0
        
        # Using CrossEntropyLoss with reduction='mean' to get per-token loss for each batch
        # Then we'll average these weighted by active tokens.
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
            reduction='mean' # Get average loss for active tokens in the batch
        )
        
        val_pbar = tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.args.use_tqdm)

        for batch in val_pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device) 
            
            with torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=(self.mixed_precision_dtype is not None)):
                logits, _, _ = self.model(input_ids, kv_context_ids=input_ids, kv_context_mask=attention_mask)
                # loss here is the average loss for active tokens in THIS batch
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            active_tokens_in_batch = (target_ids != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_ids.numel()
            
            if active_tokens_in_batch > 0 and not torch.isnan(loss) and not torch.isinf(loss):
                total_loss_sum += loss.item() * active_tokens_in_batch 
                total_active_tokens += active_tokens_in_batch
        
        self.model.train() # Switch back to train mode
        if total_active_tokens == 0:
            self.logger.warning("Validation: No active tokens found across all batches. Returning inf loss.")
            return float('inf')
        
        avg_val_loss = total_loss_sum / total_active_tokens
        self.logger.info(f"Validation loss: {avg_val_loss:.4f} (TotalLossSum: {total_loss_sum:.2f}, TotalActiveTokens: {total_active_tokens})")
        return avg_val_loss

    def save_checkpoint(self, is_best: bool = False):
        if not self.model:
            self.logger.warning("Attempted to save checkpoint, but model is not initialized.")
            return

        state_dict = self.model.state_dict()
        if ('tok_embeddings.weight' in state_dict and 'lm_head.weight' in state_dict and
            torch.is_tensor(state_dict['tok_embeddings.weight']) and 
            torch.is_tensor(state_dict['lm_head.weight']) and
            state_dict['tok_embeddings.weight'].data_ptr() == state_dict['lm_head.weight'].data_ptr()):
            
            self.logger.info("Cloning lm_head.weight for saving (tied weights).")
            state_dict = state_dict.copy() 
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()
        
        filename_stem = "best_model" if is_best else f"checkpoint_epoch_{self.state.epoch+1}_step_{self.state.global_step}"
        save_path = self.output_dir / f"{filename_stem}.safetensors"
        
        log_message = f"Saving {'best model' if is_best else 'checkpoint'} to {save_path}"
        self.logger.info(log_message)
        
        try:
            safetensors.torch.save_file(state_dict, str(save_path))
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {save_path}: {e}")


    def _calculate_grad_norm(self) -> float:
        if not self.model: return 0.0
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return (total_norm_sq ** 0.5) if total_norm_sq > 0 else 0.0

    def training_step(self, batch: Dict[str, torch.Tensor]):
        step_start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True) 
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=(self.mixed_precision_dtype is not None)):
            logits, _, aux_loss_tensor = self.model(input_ids, kv_context_ids=input_ids, kv_context_mask=attention_mask)
            
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.pad_token_id if self.pad_token_id is not None else -100, 
                reduction='mean' # Average loss over active tokens in the batch
            )
            main_loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss_for_backward = main_loss
            aux_loss_item = None # For logging
            if aux_loss_tensor is not None and self.args.aux_loss_weight > 0: 
                total_loss_for_backward = main_loss + self.args.aux_loss_weight * aux_loss_tensor
                aux_loss_item = aux_loss_tensor.item()


        actual_grad_norm = 0.0
        if self.scaler: # FP16
            self.scaler.scale(total_loss_for_backward).backward()
            self.scaler.unscale_(self.optimizer) 
            if self.args.grad_clip_norm and self.args.grad_clip_norm > 0:
                actual_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm).item()
            else:
                actual_grad_norm = self._calculate_grad_norm() 
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: # FP32 or BF16
            total_loss_for_backward.backward()
            if self.args.grad_clip_norm and self.args.grad_clip_norm > 0:
                actual_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm).item()
            else:
                actual_grad_norm = self._calculate_grad_norm() 
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step() 

        active_tokens_in_batch = (target_ids != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_ids.numel()
        
        # main_loss.item() is already average loss per token for this batch
        # For epoch average: sum(batch_avg_loss * batch_tokens) / sum(batch_tokens)
        if not torch.isnan(main_loss) and not torch.isinf(main_loss):
             self.state.train_loss_sum += main_loss.item() * active_tokens_in_batch
             self.state.train_tokens += active_tokens_in_batch
        else:
            self.logger.warning(f"NaN or Inf main_loss detected at step {self.state.global_step}. Loss: {main_loss.item()}. Skipping accumulation for this step.")


        step_duration = time.time() - step_start_time
        tokens_per_second = active_tokens_in_batch / step_duration if step_duration > 0 else 0.0

        loss_dict_for_log = {'main': main_loss.item()}
        if aux_loss_item is not None: # Use the scalar item for logging
            loss_dict_for_log['aux'] = aux_loss_item
            loss_dict_for_log['total'] = total_loss_for_backward.item() 
        
        return loss_dict_for_log, actual_grad_norm, tokens_per_second


    def estimate_training_requirements(self, model, train_loader, epochs, device_name):
        self.logger.info("[ESTIMATE] Analyzing training requirements...") 
        if not model: 
            self.logger.warning("Model not available for estimation.")
            return {}
        model_params = sum(p.numel() for p in model.parameters())
        
        sample_batches = 0
        if train_loader and len(train_loader) > 0 :
            sample_batches = min(10, len(train_loader))
        
        if sample_batches == 0:
            self.logger.warning("Train loader empty or not available, cannot estimate requirements accurately.")
            self.avg_tokens_per_batch_estimate = 0.0
        else:
            num_tokens_in_sampled_batches = 0
            actual_sampled_batches = 0
            try:
                for i, batch_data in enumerate(train_loader): 
                    if i >= sample_batches: break
                    target_ids = batch_data.get('target_ids', batch_data.get('input_ids')) 
                    if target_ids is not None:
                        active_tokens_in_batch = (target_ids != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_ids.numel()
                        num_tokens_in_sampled_batches += active_tokens_in_batch
                        actual_sampled_batches +=1
                    elif 'input_ids' in batch_data: # Fallback if only input_ids
                        num_tokens_in_sampled_batches += batch_data['input_ids'].numel() 
                        actual_sampled_batches +=1
                if actual_sampled_batches > 0:
                    self.avg_tokens_per_batch_estimate = num_tokens_in_sampled_batches / actual_sampled_batches
                else: self.avg_tokens_per_batch_estimate = 0.0
            except Exception as e:
                self.logger.error(f"Error sampling batches for estimation: {e}. Using 0 tokens/batch.")
                self.avg_tokens_per_batch_estimate = 0.0
        
        total_batches = (len(train_loader) * epochs) if train_loader and epochs > 0 else 0
        total_tokens = int(self.avg_tokens_per_batch_estimate * total_batches)
        flops_per_token = 6 * model_params 
        total_flops = total_tokens * flops_per_token
        
        rtx_3060_eff_tflops = 3 
        rtx_3060_fps = rtx_3060_eff_tflops * 1e12
        train_hours = (total_flops / rtx_3060_fps / 3600) if rtx_3060_fps > 0 and total_flops > 0 else 0.0
        
        # Determine bytes per param based on effective mixed precision type
        if self.mixed_precision_dtype == torch.bfloat16: bytes_per_param = 12 # model(2)+grad(2)+optimizer(8)
        elif self.mixed_precision_dtype == torch.float16: bytes_per_param = 10 # model(2)+grad(2)+optimizer(FP16 can be 2*N for master weights) + scaler overhead. Let's use 10-12.
        else: bytes_per_param = 16 # FP32: model(4)+grad(4)+optimizer(8)

        mem_gb = (model_params * bytes_per_param) / (1024**3)

        self.logger.info(f"  Model: {model_params:,} params ({model_params/1e6:.1f}M)")
        self.logger.info(f"  Dataset: Avg active tokens/batch: {self.avg_tokens_per_batch_estimate:.0f}")
        self.logger.info(f"  Total active tokens ({epochs} epochs): {total_tokens:,} ({total_tokens/1e6:.2f}M)")
        self.logger.info(f"  Compute: Est. total {total_flops/1e15:.2f} PFLOPs")
        self.logger.info(f"  Time: Est. RTX 3060 ({rtx_3060_eff_tflops} eff. TFLOPs) train time: {train_hours:.1f}h")
        self.logger.info(f"  VRAM: Est. (params, grads, opt): {mem_gb:.1f}GB (Activations memory not included!)")
        
        gpu_vram_gb = 0
        if self.device.type == 'cuda':
            try: 
                gpu_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                if mem_gb > gpu_vram_gb:
                    self.logger.warning(f"  Est. VRAM ({mem_gb:.1f}GB) may exceed device VRAM ({gpu_vram_gb:.1f}GB for {self.device}).")
            except Exception as e:
                 self.logger.warning(f"  Could not get device VRAM properties: {e}")
        
        if train_hours > 48: 
            self.logger.warning("  Est. training time is long (>48 hours).")
        return {
            'model_params': model_params, 'total_tokens': total_tokens, 'total_flops': total_flops, 
            'train_hours_rtx_3060_eff': train_hours, 'mem_gb_est_params_opt': mem_gb, 
            'avg_tokens_per_batch': self.avg_tokens_per_batch_estimate
        }


    def train(self):
        self.logger.info("Attempting to launch TensorBoard...")
        tensorboard_port = getattr(self.args, 'tensorboard_port', 6006) # Get port from args or default
        self.tb_process = launch_tensorboard(str(self.tb_dir), port=tensorboard_port)
        if self.tb_process:
            self.logger.info(f"TensorBoard launched or launching. Check http://localhost:{tensorboard_port}")
        else:
            self.logger.warning(f"TensorBoard did not launch. You may need to start it manually: tensorboard --logdir {self.tb_dir} --port {tensorboard_port}")
        
        self.logger.info("Starting training process...")
        try:
            _ = self.setup_model_and_tokenizer() 
            self.setup_data() 
            self.setup_optimizer_and_scheduler() 
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR during setup: {e}", exc_info=True)
            self.finalize_training(None) # Attempt cleanup
            return # Cannot proceed

        if not self.train_loader or len(self.train_loader) == 0:
            self.logger.error("Train loader is empty after setup. Aborting training.")
            self.finalize_training(None) 
            return

        if self.args.estimate_training_reqs:
            self.estimate_training_requirements(
                model=self.model, train_loader=self.train_loader,
                epochs=self.args.epochs, device_name=str(self.device)
            )
        
        try:
            self.writer = SummaryWriter(log_dir=str(self.tb_dir))
            self.logger.info(f"TensorBoard SummaryWriter initialized. Logging to: {self.tb_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard SummaryWriter: {e}. TensorBoard logging will be disabled.")
            self.writer = None


        self.training_start_time = time.time() # Set after all setups

        self.logger.info(f"--- Starting Training for {self.args.epochs} Epochs ---")
        for epoch in range(self.state.epoch, self.args.epochs):
            self.state.epoch = epoch
            self.state.train_loss_sum = 0.0
            self.state.train_tokens = 0
            
            epoch_msg = f"--- Starting Epoch {epoch + 1}/{self.args.epochs} ---"
            if self.avg_tokens_per_batch_estimate > 0 and self.train_loader and len(self.train_loader) > 0:
                epoch_msg += f" | Est. active tokens this epoch: {self.avg_tokens_per_batch_estimate * len(self.train_loader) / 1e6:.2f}M"
            self.logger.info(epoch_msg)

            if self.model: self.model.train()
            else:
                self.logger.error("Model is None at start of epoch. Aborting.")
                break # Exit epoch loop
            
            train_pbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f"Epoch {epoch+1}/{self.args.epochs}", 
                disable=not self.args.use_tqdm,
                leave=True # Keep bar after epoch finishes
            )

            for batch_idx, batch in train_pbar:
                try:
                    loss_dict, grad_norm, tps = self.training_step(batch)
                except Exception as e:
                    self.logger.error(f"Error in training_step at G:{self.state.global_step}, E:{epoch+1}, B:{batch_idx}: {e}", exc_info=True)
                    # Decide whether to continue to next step or stop training
                    # For now, let's try to continue if it's a recoverable error, but log it seriously
                    continue # Skip to next batch

                current_lr = self.optimizer.param_groups[0]['lr']

                if self.state.global_step % self.args.log_every_n_steps == 0:
                    self.log_console_step(loss_dict, current_lr, grad_norm, tps, batch_idx)

                if self.writer:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Loss/train_{key}', value, self.state.global_step)
                    self.writer.add_scalar('Params/learning_rate', current_lr, self.state.global_step)
                    self.writer.add_scalar('Params/epoch_current', self.state.epoch + 1, self.state.global_step)
                    self.writer.add_scalar('Params/gradient_norm', grad_norm, self.state.global_step)
                    self.writer.add_scalar('Performance/tokens_per_second', tps, self.state.global_step)
                
                # Update tqdm description minimally, console log has details
                if self.args.use_tqdm:
                     train_pbar.set_description(f"E{epoch+1} B:{batch_idx+1}/{len(self.train_loader)} L:{loss_dict['main']:.3f}")


                # --- Evaluation Logic ---
                if (self.val_loader and self.args.eval_every_n_steps > 0 and
                        self.state.global_step > 0 and
                        self.state.global_step % self.args.eval_every_n_steps == 0):
                    val_loss = self.evaluate() 
                    if val_loss is not None and self.writer:
                        self.writer.add_scalar('Loss/validation_on_step', val_loss, self.state.global_step)
                    if val_loss is not None: # Check if evaluation happened and returned a valid loss
                        if val_loss < self.state.best_val_loss:
                            self.logger.info(f"New best val_loss: {val_loss:.4f} (was {self.state.best_val_loss:.4f}) at G:{self.state.global_step}")
                            self.state.best_val_loss = val_loss
                            self.state.epochs_no_improve = 0
                            self.save_checkpoint(is_best=True) 
                        else:
                            self.state.epochs_no_improve += 1
                            self.logger.info(f"Val_loss {val_loss:.4f} (no improvement). Best: {self.state.best_val_loss:.4f}. No-improve count: {self.state.epochs_no_improve} at G:{self.state.global_step}")
                        
                        if self.args.early_stopping_patience > 0 and self.state.epochs_no_improve >= self.args.early_stopping_patience:
                            self.logger.info(f"Early stopping triggered: {self.state.epochs_no_improve} evaluations without validation loss improvement.")
                            self.finalize_training(self.training_start_time)
                            return # Exit train() method
                
                # --- Sample Generation Logic ---
                if (self.args.generate_every_n_steps > 0 and
                        self.state.global_step > 0 and
                        self.state.global_step % self.args.generate_every_n_steps == 0):
                    self._generate_sample_text()

                # --- Checkpointing Logic ---
                if self.args.save_every_n_steps > 0 and self.state.global_step > 0 and self.state.global_step % self.args.save_every_n_steps == 0:
                    self.save_checkpoint(is_best=False) # Regular step-based checkpoint

                self.state.global_step += 1
            
            # --- End of Epoch Summary ---
            train_pbar.close() # Explicitly close epoch progress bar
            avg_epoch_train_loss = (self.state.train_loss_sum / self.state.train_tokens) if self.state.train_tokens > 0 else float('nan')
            self.logger.info(f"--- Epoch {epoch+1} Completed. Avg Train Loss: {avg_epoch_train_loss:.4f}. Current Best Val Loss: {self.state.best_val_loss if self.state.best_val_loss != float('inf') else 'N/A'} ---")
            if self.writer:
                self.writer.add_scalar('Loss/train_epoch_avg', avg_epoch_train_loss, self.state.epoch + 1) # Log against epoch number

            # End of epoch evaluation (if not done by steps or if val_loader exists)
            if self.val_loader and self.args.eval_every_n_steps <= 0: 
                val_loss_epoch_end = self.evaluate() 
                if val_loss_epoch_end is not None and self.writer:
                    self.writer.add_scalar('Loss/validation_epoch_end', val_loss_epoch_end, self.state.epoch + 1) 
                if val_loss_epoch_end is not None:
                    if val_loss_epoch_end < self.state.best_val_loss:
                        self.logger.info(f"New best val_loss (epoch end): {val_loss_epoch_end:.4f}")
                        self.state.best_val_loss = val_loss_epoch_end
                        self.state.epochs_no_improve = 0 
                        self.save_checkpoint(is_best=True)
            
            # Save checkpoint at end of epoch if configured
            if self.args.save_every_epoch:
                 self.save_checkpoint(is_best=False) # Epoch-end checkpoint

        self.finalize_training(self.training_start_time)


    def finalize_training(self, training_start_time: Optional[float]):
        self.logger.info("Finalizing training run...")
        
        # Final evaluation if possible
        if self.val_loader and self.model: 
            final_val_loss = self.evaluate() 
            if final_val_loss is not None and final_val_loss <= self.state.best_val_loss: # Check if this is a new best
                self.state.best_val_loss = final_val_loss
                self.save_checkpoint(is_best=True) # Save if it's the absolute best

        # Save final model state if model exists
        if self.model : 
            self.logger.info("Saving final model state...")
            final_model_path = self.output_dir / "final_model.safetensors"
            try:
                final_state_dict = self.model.state_dict()
                # Handle tied weights for lm_head and token embeddings
                if ('tok_embeddings.weight' in final_state_dict and 'lm_head.weight' in final_state_dict and
                    torch.is_tensor(final_state_dict['tok_embeddings.weight']) and 
                    torch.is_tensor(final_state_dict['lm_head.weight']) and
                    final_state_dict['tok_embeddings.weight'].data_ptr() == final_state_dict['lm_head.weight'].data_ptr()):
                    self.logger.info("Cloning lm_head.weight for final save (tied weights).")
                    final_state_dict = final_state_dict.copy() # Make a copy before modifying
                    final_state_dict['lm_head.weight'] = final_state_dict['lm_head.weight'].clone()        
                safetensors.torch.save_file(final_state_dict, str(final_model_path))
                self.logger.info(f"Final model saved to {final_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save final model: {e}")
        
        if self.writer:
            try:
                self.writer.close()
                self.logger.info("TensorBoard writer closed.")
            except Exception as e:
                self.logger.error(f"Error closing TensorBoard writer: {e}")

        if hasattr(self, 'tb_process') and self.tb_process and self.tb_process.poll() is None: # Check if process exists and is running
            self.logger.info("Attempting to shut down TensorBoard process...")
            self.tb_process.terminate()
            try:
                self.tb_process.wait(timeout=5) 
                self.logger.info("TensorBoard process terminated.")
            except subprocess.TimeoutExpired:
                self.logger.warning("TensorBoard process did not terminate gracefully after 5s, killing.")
                self.tb_process.kill()
            except Exception as e:
                self.logger.error(f"Error terminating TensorBoard process: {e}")
            
        if training_start_time is not None:
            total_time_seconds = time.time() - training_start_time
            self.logger.info(f"Total training time: {total_time_seconds:.2f}s ({total_time_seconds/3600:.2f}h).")
        else:
            self.logger.info("Training complete (actual start time not available for total duration).")
        
        best_loss_str = f"{self.state.best_val_loss:.4f}" if self.state.best_val_loss != float('inf') else "N/A (no validation or no improvement)"
        self.logger.info(f"Best validation loss achieved during run: {best_loss_str}")
        self.logger.info(f"--- Training Run '{self.args.run_name}' Finished ---")


    def log_console_step(self, loss_dict: Dict[str, float], lr: float, grad_norm: float, tps: float, batch_idx: int):
        """Log compact training step info to console."""
        # Calculate average epoch loss so far for this epoch
        avg_epoch_loss_so_far = (self.state.train_loss_sum / self.state.train_tokens) if self.state.train_tokens > 0 else float('nan')
        
        current_step_in_epoch = batch_idx + 1 
        total_steps_in_epoch = len(self.train_loader) if self.train_loader else 0

        # Compact format for console
        # E1 10/100 (G:1234) L:1.23 AuxL:0.10 AvgEpL:1.50 GN:0.95 LR:1.0e-4 TPS:12345
        log_msg_parts = [
            f"E{self.state.epoch+1}",
            f"{current_step_in_epoch}/{total_steps_in_epoch}" if total_steps_in_epoch > 0 else f"B:{current_step_in_epoch}",
            f"(G:{self.state.global_step})",
            f"L:{loss_dict['main']:.3f}"
        ]
        if 'aux' in loss_dict and loss_dict['aux'] is not None: # Check for None as aux_loss_item can be None
            log_msg_parts.append(f"AuxL:{loss_dict['aux']:.3f}")
        
        log_msg_parts.extend([
            f"AvgEpL:{avg_epoch_loss_so_far:.3f}", 
            f"GN:{grad_norm:.2f}",
            f"LR:{lr:.1e}", # Compact LR
            f"TPS:{tps:.0f}"
        ])
        self.logger.info(" ".join(log_msg_parts))


def main():
    print("[Main] Script started.")
    try:
        training_args = TrainingArgs() 
        print(f"[Main] TrainingArgs loaded. Run name from config: {training_args.run_name}")
    except ImportError as e:
        print(f"FATAL: training_config.py not found or TrainingArgs class missing/error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        print(f"FATAL: Error initializing TrainingArgs from training_config.py: {e}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(training_args.seed)
    print(f"[Main] PyTorch manual seed set to: {training_args.seed}")
    
    # Optional: more determinism for CUDA, can impact performance
    # if training_args.device == "cuda" and torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     print("[Main] CUDA deterministic backend set (can affect performance).")
    
    trainer = None # Initialize trainer to None for robust error handling
    try:
        print("[Main] Initializing ModelTrainer...")
        trainer = ModelTrainer(training_args)
        print("[Main] ModelTrainer initialization complete. Starting training...")
        trainer.train()
        print("[Main] trainer.train() completed.")
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user (KeyboardInterrupt). Finalizing...", file=sys.stderr)
        if trainer: # If trainer was initialized
            start_time = trainer.training_start_time if hasattr(trainer, 'training_start_time') and trainer.training_start_time is not None else time.time()
            trainer.finalize_training(start_time) 
        else: # Trainer initialization failed
            print("[Main] Finalizing called, but trainer was not fully initialized.", file=sys.stderr)
    except Exception as e:
        print(f"\n[Main] FATAL Unhandled exception during training process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if trainer: # If trainer was initialized, try to finalize
            print("[Main] Attempting to finalize training after unhandled exception...", file=sys.stderr)
            start_time = trainer.training_start_time if hasattr(trainer, 'training_start_time') and trainer.training_start_time is not None else time.time()
            trainer.finalize_training(start_time)
        else:
            print("[Main] Trainer was not initialized when unhandled exception occurred. Minimal cleanup.", file=sys.stderr)
        sys.exit(1) # Indicate failure
    finally:
        print("[Main] Script execution finished.")


if __name__ == "__main__":
    # Ensure PYTHONUTF8=1 for best console emoji/UTF-8 support on Windows
    # Example: $env:PYTHONUTF8=1; python train.py (PowerShell)
    # Or set it system-wide.
    if os.name == 'nt' and os.getenv('PYTHONUTF8') != '1':
        print("Warning: On Windows, for best console UTF-8 support, consider setting the PYTHONUTF8=1 environment variable.", file=sys.stderr)
    main()
