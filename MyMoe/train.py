#!/usr/bin/env python3
"""
Improved LLM Training Script
Simplified, cleaner, with comprehensive logging
"""

import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import sys
import webbrowser
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import safetensors.torch
from tqdm import tqdm

def launch_tensorboard(log_dir: str, port: int = 6006):
    """Launch TensorBoard in a background process."""
    tensorboard_url = f"http://localhost:{port}"
    
    # Start TensorBoard process
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", str(log_dir), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    
    # Wait a bit for TensorBoard to start
    time.sleep(3)
    
    # Open the default web browser
    webbrowser.open(tensorboard_url)
    return tb_process

# Import your modules (assumed to exist)
from config import ModelArgs
from Transformer_Block import LLM
from BpeTokenizer import get_bpe_tokenizer, BPEDataset, SPECIAL_TOKENS
from training_config import TrainingArgs # Make sure this has all necessary fields


@dataclass
class TrainingState:
    """Track training state"""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    epochs_no_improve: int = 0 # For early stopping based on validation
    train_loss_sum: float = 0.0 # Cumulative loss for current epoch
    train_tokens: int = 0 # Cumulative tokens for current epoch

class ModelTrainer:
    """Main training class"""

    def __init__(self, args: TrainingArgs):
        self.args = args
        self.state = TrainingState()
        self.avg_tokens_per_batch_estimate: float = 0.0 # For epoch token estimation

        self._initialize_paths_and_create_dirs()
        self.setup_logging()
        self._log_directory_info_and_save_configs()
        self.setup_device()

    def _initialize_paths_and_create_dirs(self):
        """Define paths and create physical directories."""
        self.output_dir = Path(self.args.output_dir_base) / self.args.run_name
        self.tb_dir = Path(self.args.runs_dir_base) / self.args.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_file_path = self.output_dir / 'training.log'

        # Ensure UTF-8 encoding for the file handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout) # Explicitly use sys.stdout

        # For StreamHandler, try to set UTF-8 if possible, or handle errors
        # This is tricky because console encoding is platform/config dependent
        # Python 3.7+ with PYTHONUTF8=1 env var is the best way to ensure UTF-8 console
        try:
            stream_handler.stream.reconfigure(encoding='utf-8')
        except AttributeError: # In case reconfigure is not available (e.g. older Python or different stream type)
            pass # Keep default encoding for console, emojis might not display correctly

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file_path}")

    def _log_directory_info_and_save_configs(self):
        """Logs information about directories and saves training arguments."""
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Tensorboard directory: {self.tb_dir}")
        config_save_path = self.output_dir / "training_args.json"
        with open(config_save_path, "w", encoding='utf-8') as f:
            json.dump(asdict(self.args), f, indent=2)
        self.logger.info(f"Training arguments saved to {config_save_path}")

    def setup_device(self):
        if self.args.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Using CPU.")
            self.args.device = "cpu"
        self.device = torch.device(self.args.device)
        self.logger.info(f"Using device: {self.device}")
        self.mixed_precision_dtype = None
        self.scaler = None
        if self.device.type == 'cuda':
            if self.args.use_bf16 and torch.cuda.is_bf16_supported():
                self.mixed_precision_dtype = torch.bfloat16
                self.logger.info("Using bfloat16 mixed precision")
            elif self.args.use_fp16:
                self.mixed_precision_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
                self.logger.info("Using float16 mixed precision with GradScaler")

    def setup_model_and_tokenizer(self):
        self.logger.info("Setting up model and tokenizer...")
        model_args_dict = self.args.model_config_overrides if isinstance(self.args.model_config_overrides, dict) else {}
        model_args = ModelArgs(**model_args_dict)
        self.tokenizer, model_args = get_bpe_tokenizer(
            model_args, self.args.data_path, self.args.retrain_tokenizer
        )
        model_config_path = self.output_dir / "model_config.json"
        with open(model_config_path, "w", encoding='utf-8') as f:
            json.dump(asdict(model_args), f, indent=2)
        self.logger.info(f"Model config saved to {model_config_path}")
        self.model = LLM(model_args).to(self.device)
        self.pad_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[3]) # <pad>
        self.bos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[2]) # <bos>
        self.eos_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[1]) # <eos>
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {param_count:,} trainable parameters")
        return model_args

    def setup_data(self):
        self.logger.info("Setting up data loaders...")
        train_dataset = BPEDataset(
            self.args.data_path, self.tokenizer, self.model.args.max_seq_len
        )
        val_dataset = None
        if self.args.val_data_path:
            val_dataset = BPEDataset(
                self.args.val_data_path, self.tokenizer, self.model.args.max_seq_len
            )
            self.logger.info(f"Using separate validation data: {len(val_dataset)} samples")
        elif self.args.val_split_ratio > 0 and len(train_dataset) > 100:
            val_size = max(1, int(len(train_dataset) * self.args.val_split_ratio))
            train_size = len(train_dataset) - val_size
            if train_size <= 0 or val_size <= 0:
                self.logger.warning(f"Cannot split dataset of size {len(train_dataset)} with ratio {self.args.val_split_ratio}. Using full dataset for training.")
            else:
                train_dataset, val_dataset = random_split(
                    train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.args.seed)
                )
                self.logger.info(f"Split data - Train: {train_size}, Val: {val_size}")
        else:
            self.logger.warning("No validation data available.")

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=(self.device.type == 'cuda')
        )
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=(self.device.type == 'cuda')
            )
        self.logger.info(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            self.logger.info(f"Validation batches: {len(self.val_loader)}")

    def setup_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay
        )
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_steps_ratio)
        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(1, total_steps - warmup_steps),
                eta_min=self.args.learning_rate * 0.1
            )
        else:
            self.scheduler = None
            self.logger.info("No learning rate scheduler configured.")
        self.logger.info(f"Optimizer: AdamW")
        self.logger.info(f"LR Scheduler: {self.args.lr_scheduler_type if self.scheduler else 'None'}")
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {warmup_steps}")

    def log_step(self, loss_dict: Dict[str, float], lr: float, grad_norm: float, tps: float):
        """Log training step information"""
        if self.state.global_step % self.args.log_every_n_steps == 0:
            avg_epoch_loss = self.state.train_loss_sum / max(self.state.train_tokens, 1)
            log_msg_parts = [
                f"Step {self.state.global_step:6d}",
                f"Epoch {self.state.epoch+1:3d}",
                f"Loss: {loss_dict['main']:.4f}",
                f"AvgEpochLoss: {avg_epoch_loss:.4f}",
                f"GradNorm: {grad_norm:.2f}",
                f"LR: {lr:.2e}",
                f"Tokens/sec: {tps:.0f}"
            ]
            if 'aux' in loss_dict:
                 log_msg_parts.insert(4, f"AuxLoss: {loss_dict['aux']:.4f}") # Insert aux loss if present
            self.logger.info(" | ".join(log_msg_parts))

        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Train/loss_{key}', value, self.state.global_step)
        self.writer.add_scalar('Train/learning_rate', lr, self.state.global_step)
        self.writer.add_scalar('Train/epoch', self.state.epoch + 1, self.state.global_step)
        self.writer.add_scalar('Train/gradient_norm', grad_norm, self.state.global_step)
        self.writer.add_scalar('Performance/tokens_per_second', tps, self.state.global_step)

    @torch.no_grad()
    def evaluate(self):
        if not self.val_loader:
            self.logger.info("No validation loader available, skipping evaluation.")
            return None
        self.logger.info("Running validation...")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
            reduction='sum'
        )
        for batch in tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.args.use_tqdm):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=(self.mixed_precision_dtype is not None)):
                logits, _, _ = self.model(input_ids, kv_context_ids=input_ids, kv_context_mask=attention_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            active_tokens = (target_ids != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_ids.numel()
            if active_tokens > 0:
                total_loss += loss.item()
                total_tokens += active_tokens
        self.model.train()
        if total_tokens == 0:
            self.logger.warning("No active tokens found during validation. Returning inf loss.")
            return float('inf')
        avg_loss = total_loss / total_tokens
        self.logger.info(f"Validation loss: {avg_loss:.4f} (TotalLoss: {total_loss:.2f}, TotalTokens: {total_tokens})")
        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        state_dict = self.model.state_dict()
        if ('tok_embeddings.weight' in state_dict and 'lm_head.weight' in state_dict and
            torch.is_tensor(state_dict['tok_embeddings.weight']) and torch.is_tensor(state_dict['lm_head.weight']) and
            state_dict['tok_embeddings.weight'].data_ptr() == state_dict['lm_head.weight'].data_ptr()):
            state_dict = state_dict.copy()
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()
        
        if is_best:
            save_path = self.output_dir / "best_model.safetensors"
            self.logger.info(f"Saving best model to {save_path}")
        else:
            save_path = self.output_dir / f"checkpoint_epoch_{self.state.epoch+1}_step_{self.state.global_step}.safetensors"
            self.logger.info(f"Saving checkpoint to {save_path}")
        safetensors.torch.save_file(state_dict, str(save_path))

    def _calculate_grad_norm(self) -> float:
        """Calculates the L2 norm of gradients for all trainable parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return (total_norm ** 0.5) if total_norm > 0 else 0.0

    def training_step(self, batch):
        step_start_time = time.time()
        self.optimizer.zero_grad()
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=(self.mixed_precision_dtype is not None)):
            logits, _, aux_loss = self.model(input_ids, kv_context_ids=input_ids, kv_context_mask=attention_mask)
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id if self.pad_token_id is not None else -100)
            main_loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss = main_loss
            if aux_loss is not None:
                total_loss = main_loss + self.args.aux_loss_weight * aux_loss

        actual_grad_norm = 0.0
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer) # Unscale before clipping or norm calculation
            if self.args.grad_clip_norm and self.args.grad_clip_norm > 0:
                actual_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm).item()
            else:
                actual_grad_norm = self._calculate_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if self.args.grad_clip_norm and self.args.grad_clip_norm > 0:
                actual_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm).item()
            else:
                actual_grad_norm = self._calculate_grad_norm()
            self.optimizer.step()

        if self.scheduler:
            # For CosineAnnealingWarmRestarts, step per batch/iteration
            self.scheduler.step(self.state.epoch + self.state.global_step / len(self.train_loader))


        active_tokens = (target_ids != self.pad_token_id).sum().item() if self.pad_token_id is not None else target_ids.numel()
        self.state.train_loss_sum += main_loss.item() * active_tokens
        self.state.train_tokens += active_tokens

        step_duration = time.time() - step_start_time
        tokens_per_second = active_tokens / step_duration if step_duration > 0 else 0.0

        loss_dict = {'main': main_loss.item()}
        if aux_loss is not None:
            loss_dict['aux'] = aux_loss.item()
            loss_dict['total'] = total_loss.item()
        return loss_dict, actual_grad_norm, tokens_per_second

    def estimate_training_requirements(self, model, train_loader, epochs, device_name):
        self.logger.info("[ESTIMATE] Analyzing training requirements...") # Emojis removed for console compatibility
        model_params = sum(p.numel() for p in model.parameters())
        sample_batches = min(10, len(train_loader))
        total_sample_tokens = 0
        if sample_batches == 0:
            self.logger.warning("Train loader is empty, cannot estimate training requirements accurately.")
            self.avg_tokens_per_batch_estimate = 0.0
        else:
            try:
                for i, batch_data in enumerate(train_loader): # Renamed batch to batch_data to avoid conflict
                    if i >= sample_batches: break
                    input_ids = batch_data['input_ids']
                    total_sample_tokens += input_ids.numel()
                self.avg_tokens_per_batch_estimate = total_sample_tokens / sample_batches
            except Exception as e:
                self.logger.error(f"Error sampling batches for estimation: {e}. Using 0 tokens/batch.")
                self.avg_tokens_per_batch_estimate = 0.0
        
        total_batches = len(train_loader) * epochs
        total_tokens_overall_training = int(self.avg_tokens_per_batch_estimate * total_batches)
        flops_per_token = 6 * model_params
        total_flops = total_tokens_overall_training * flops_per_token
        rtx_3060_flops = 13e12
        training_time_hours = (total_flops / rtx_3060_flops / 3600) if rtx_3060_flops > 0 else float('inf')
        memory_gb = (model_params * 4 * 3) / (1024**3)

        self.logger.info(f"[MODEL INFO] Model: {model_params:,} parameters ({model_params/1e6:.1f}M)")
        self.logger.info(f"[DATASET INFO] Avg tokens/batch: {self.avg_tokens_per_batch_estimate:.0f}, Total tokens for training ({epochs} epochs): {total_tokens_overall_training:,} ({total_tokens_overall_training/1e6:.2f}M)")
        self.logger.info(f"[COMPUTE] Estimated total compute (Chinchilla): {total_flops/1e12:.1f} TFLOP (for {epochs} epochs)")
        self.logger.info(f"[TIME] Estimated RTX 3060 training time: {training_time_hours:.1f}h ({training_time_hours/24:.1f} days)")
        self.logger.info(f"[VRAM] Estimated VRAM (simplified): {memory_gb:.1f}GB (model, grads, optimizer)")
        if memory_gb > 12:
            self.logger.warning("Estimated VRAM may exceed RTX 3060 VRAM (12GB)!")
        if training_time_hours > 48:
            self.logger.warning("Estimated training time is long.")
        return {
            'model_params': model_params, 'total_tokens': total_tokens_overall_training,
            'total_flops': total_flops, 'training_hours_rtx3060': training_time_hours,
            'memory_gb_estimate': memory_gb, 'avg_tokens_per_batch': self.avg_tokens_per_batch_estimate
        }

    def train(self):
        # Launch TensorBoard before starting training
        self.logger.info("Launching TensorBoard...")
        self.tb_process = launch_tensorboard(str(self.tb_dir))
        self.logger.info("TensorBoard launched. Open http://localhost:6006 in your browser if it didn't open automatically")
        
        # Initial logs (outside the main loop) still use self.logger.info
        self.logger.info("Starting training...")
        _ = self.setup_model_and_tokenizer()
        self.setup_data()
        if self.args.estimate_training_reqs:
            self.estimate_training_requirements(
                model=self.model, train_loader=self.train_loader,
                epochs=self.args.epochs, device_name=str(self.device)
            )
        self.setup_optimizer_and_scheduler()
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.logger.info(f"TensorBoard writer initialized. Logging to: {self.tb_dir}")

        training_start_time = time.time()

        # Loop through epochs
        for epoch in range(self.state.epoch, self.args.epochs):
            self.state.epoch = epoch
            self.state.train_loss_sum = 0.0
            self.state.train_tokens = 0
            
            # Log start of epoch information
            epoch_start_message = f"--- Starting Epoch {epoch + 1}/{self.args.epochs} ---"
            if self.avg_tokens_per_batch_estimate > 0 and len(self.train_loader) > 0:
                tokens_this_epoch_est = self.avg_tokens_per_batch_estimate * len(self.train_loader)
                epoch_start_message += f" | Est. tokens this epoch: {tokens_this_epoch_est / 1e6:.2f}M"
            self.logger.info(epoch_start_message)

            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                loss_dict, grad_norm, tps = self.training_step(batch)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Console logging for steps, based on frequency
                if self.state.global_step % self.args.log_every_n_steps == 0:
                    self.log_step(loss_dict, current_lr, grad_norm, tps)

                # TensorBoard logging (every step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/loss_{key}', value, self.state.global_step)
                self.writer.add_scalar('Train/learning_rate', current_lr, self.state.global_step)
                self.writer.add_scalar('Train/epoch_current', self.state.epoch + 1, self.state.global_step) # Log current epoch for each step
                self.writer.add_scalar('Train/gradient_norm', grad_norm, self.state.global_step)
                self.writer.add_scalar('Performance/tokens_per_second', tps, self.state.global_step)

                # --- Evaluation and Checkpointing Logic ---
                if (self.val_loader and self.args.eval_every_n_steps > 0 and
                        self.state.global_step > 0 and
                        self.state.global_step % self.args.eval_every_n_steps == 0):
                    
                    val_loss = self.evaluate() 

                    if val_loss is not None:
                        self.writer.add_scalar('Val/loss_step', val_loss, self.state.global_step)
                        if val_loss < self.state.best_val_loss:
                            self.state.best_val_loss = val_loss
                            self.state.epochs_no_improve = 0
                            self.save_checkpoint(is_best=True) 
                        else:
                            self.state.epochs_no_improve += 1
                        if self.args.early_stopping_patience > 0 and self.state.epochs_no_improve >= self.args.early_stopping_patience:
                            self.logger.info(f"Early stopping triggered after {self.state.epochs_no_improve} evaluations without improvement.")
                            self.writer.close()
                            total_time = time.time() - training_start_time
                            self.logger.info(f"Total training time: {total_time:.2f}s ({total_time/3600:.2f}h).")
                            return
                
                if self.args.save_every_n_steps > 0 and self.state.global_step > 0 and self.state.global_step % self.args.save_every_n_steps == 0:
                    self.save_checkpoint(is_best=False)

                self.state.global_step += 1
            
            # --- End of Epoch Summary ---
            avg_epoch_train_loss = self.state.train_loss_sum / max(self.state.train_tokens, 1)
            self.logger.info(f"--- Epoch {epoch+1} completed. Avg Train Loss: {avg_epoch_train_loss:.4f}. Best Val Loss: {self.state.best_val_loss if self.state.best_val_loss != float('inf') else 'N/A'} ---")
            self.writer.add_scalar('Train/loss_epoch_avg', avg_epoch_train_loss, epoch + 1) # Log against epoch number

            # End of epoch evaluation (if not done by steps)
            if self.val_loader and self.args.eval_every_n_steps <= 0: 
                val_loss_epoch = self.evaluate() 
                if val_loss_epoch is not None:
                    self.writer.add_scalar('Val/loss_epoch_end', val_loss_epoch, epoch + 1) # Log against epoch num
                    if val_loss_epoch < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss_epoch
                        self.state.epochs_no_improve = 0 
                        self.save_checkpoint(is_best=True)
            
            # Save checkpoint at end of epoch if configured
            if self.args.save_every_epoch:
                 self.save_checkpoint(is_best=False)

        # --- Final Evaluation and Saving (after all epochs) ---
        if self.val_loader:
            final_val_loss = self.evaluate()
            if final_val_loss is not None:
                # self.logger.info(f"Final validation loss: {final_val_loss:.4f}") # evaluate() already logs this
                if final_val_loss <= self.state.best_val_loss: # Check if this is the new best
                    self.state.best_val_loss = final_val_loss
                    self.save_checkpoint(is_best=True)

        self.logger.info("Saving final model state...")
        final_model_path = self.output_dir / "final_model.safetensors"
        final_state_dict = self.model.state_dict()
        if ('tok_embeddings.weight' in final_state_dict and 'lm_head.weight' in final_state_dict and
            torch.is_tensor(final_state_dict['tok_embeddings.weight']) and torch.is_tensor(final_state_dict['lm_head.weight']) and
            final_state_dict['tok_embeddings.weight'].data_ptr() == final_state_dict['lm_head.weight'].data_ptr()):
            self.logger.info("Cloning lm_head.weight for final save due to tied weights.")
            final_state_dict = final_state_dict.copy()
            final_state_dict['lm_head.weight'] = final_state_dict['lm_head.weight'].clone()        
            safetensors.torch.save_file(final_state_dict, str(final_model_path))
        self.logger.info(f"Final model saved to {final_model_path}")
        
        self.writer.close()
        
        # Clean up TensorBoard process
        if hasattr(self, 'tb_process'):
            self.logger.info("Shutting down TensorBoard...")
            self.tb_process.terminate()
            self.tb_process.wait()
            
        total_time = time.time() - training_start_time
        self.logger.info(f"Training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h).")

    # Ensure log_step method is defined as before to handle console output based on frequency
    def log_step(self, loss_dict: Dict[str, float], lr: float, grad_norm: float, tps: float):
        """Log training step information TO CONSOLE based on frequency"""
        avg_epoch_loss_so_far = self.state.train_loss_sum / max(self.state.train_tokens, 1)
        log_msg_parts = [
            f"Step {self.state.global_step % len(self.train_loader):6d}/{len(self.train_loader):6d}",
            f"Epoch {self.state.epoch+1:3d}", # Current epoch
            f"Loss: {loss_dict['main']:.4f}",
            f"AvgEpochLoss: {avg_epoch_loss_so_far:.4f}"
        ]
        if 'aux' in loss_dict:
            log_msg_parts.insert(4, f"AuxLoss: {loss_dict['aux']:.4f}")
        
        log_msg_parts.extend([
            f"GradNorm: {grad_norm:.2f}",
            f"LR: {lr:.2e}",
            f"Tokens/sec: {tps:.0f}"
        ])
        self.logger.info(" | ".join(log_msg_parts))


def main():
    training_args = TrainingArgs()
    torch.manual_seed(training_args.seed)
    # if training_args.device == "cuda": # Optional: for more determinism with CUDA
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # Example: Override run_name if needed for specific experiments
    # training_args.run_name = f"experiment_X_{time.strftime('%Y%m%d_%H%M%S')}"
    
    trainer = ModelTrainer(training_args)
    trainer.train()

if __name__ == "__main__":
    # Ensure PYTHONUTF8=1 is set in your environment for best console emoji support on Windows
    # e.g., in PowerShell: $env:PYTHONUTF8=1; python train.py
    # or permanently set it in system environment variables.
    main()
