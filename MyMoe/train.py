import os
import json
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import safetensors.torch

from config import ModelArgs
from Transformer_Block import LLM
from BpeTokenizer import get_bpe_tokenizer, BPEDataset, SPECIAL_TOKENS
from training_config import TrainingArgs


class Trainer:
    def __init__(self, args: TrainingArgs):
        self.args = args
        self.step = 0
        self.best_loss = float('inf')
        
        # Setup paths
        self.output_dir = Path(args.output_dir_base) / args.run_name
        self.tb_dir = Path(args.runs_dir_base) / args.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Cnfigure mixed precision based on args
        self.use_amp = True
        self.dtype = torch.bfloat16
        print(f"Device: {self.device}, AMP: {self.use_amp}, dtype: {self.dtype}")

    def get_lr_scheduler(self, optimizer, num_training_steps):
        """
        Create learning rate scheduler based on configuration
        """
        warmup_steps = int(self.args.warmup_steps_ratio * num_training_steps)
        
        if self.args.lr_scheduler_type == "cosine":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Warmup with gradual increase
                    warmup_factor = 0.03
                    return warmup_factor + (1.0 - warmup_factor) * (current_step / warmup_steps)
                else:
                    # Cosine decay
                    progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
                    min_lr_ratio = 0.05
                    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return LambdaLR(optimizer, lr_lambda), warmup_steps
        else:
            # Default to cosine if unknown scheduler type
            return self.get_lr_scheduler(optimizer, num_training_steps)

    def _move_batch_to_device(self, batch):
        """Helper method to move batch data to device."""
        return {
            'input_ids': batch['input_ids'].to(self.device, non_blocking=True),
            'target_ids': batch['target_ids'].to(self.device, non_blocking=True),
            'attention_mask': batch['attention_mask'].to(self.device, non_blocking=True)
        }

    def setup(self):
        # Set random seed
        torch.manual_seed(self.args.seed)
        
        # Model and tokenizer
        model_args = ModelArgs(**self.args.model_config_overrides)
        self.tokenizer, model_args = get_bpe_tokenizer(model_args, self.args.data_path, self.args.retrain_tokenizer)
        self.model = LLM(model_args).to(self.device)
        
        # Special tokens
        self.pad_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[3])
        self.eos_id = self.tokenizer.token_to_id(SPECIAL_TOKENS[1])
        
        # Data
        train_dataset = BPEDataset(self.args.data_path, self.tokenizer, model_args.max_seq_len)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_dataset = None
        if self.args.val_data_path and os.path.exists(self.args.val_data_path):
            val_dataset = BPEDataset(self.args.val_data_path, self.tokenizer, model_args.max_seq_len)
            self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        else:
            self.val_loader = None
        
        # Pre-compute constants
        self.batches_per_epoch = len(self.train_loader)
        self.total_steps = self.batches_per_epoch * self.args.epochs
        
        # Optimizer with configuration parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler, self.warmup_steps = self.get_lr_scheduler(self.optimizer, self.total_steps)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        
        # Logging
        self.writer = SummaryWriter(self.tb_dir)
        
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"Data: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")
        print(f"Training: {self.args.epochs} epochs, {self.total_steps} steps")
        print(f"LR Schedule: {self.warmup_steps} warmup steps, {self.args.lr_scheduler_type} decay")
        print(f"Batch size: {self.args.batch_size}, Learning rate: {self.args.learning_rate}")

    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        batch = self._move_batch_to_device(batch)
        
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.use_amp):
            logits, _, aux_loss = self.model(
                batch['input_ids'], 
                kv_context_ids=batch['input_ids'], 
                kv_context_mask=batch['attention_mask']
            )
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), batch['target_ids'].view(-1))
            if aux_loss is not None:
                loss = loss + self.args.aux_loss_weight * aux_loss
        
        loss.backward()
        
        if self.args.grad_clip_norm and self.args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), aux_loss.item() if aux_loss is not None else 0.0

    @torch.no_grad()
    def validate(self):
        if not self.val_loader:
            return None
            
        self.model.eval()
        total_loss = 0
        total_aux_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits, _, aux_loss = self.model(
                    batch['input_ids'], 
                    kv_context_ids=batch['input_ids'], 
                    kv_context_mask=batch['attention_mask']
                )
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), batch['target_ids'].view(-1))
            
            total_loss += loss.item()
            if aux_loss is not None:
                total_aux_loss += aux_loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches, total_aux_loss / num_batches

    @torch.no_grad()
    def generate_sample(self, prompt="The story begins", max_len=None):
        if max_len is None:
            max_len = self.args.max_gen_len
            
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        for _ in range(max_len):
            context = input_tensor[:, -self.model.args.max_seq_len:]
            mask = torch.ones_like(context)
            
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits, _, _ = self.model(context, kv_context_ids=context, kv_context_mask=mask)
            
            # Apply temperature and top-k sampling if configured
            if self.args.generation_temperature != 1.0:
                logits = logits / self.args.generation_temperature
            
            if self.args.generation_top_k:
                top_k = min(self.args.generation_top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits[:, -1, :], top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx).squeeze()
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
            
            if next_token.item() == self.eos_id:
                break
                
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
        
        text = self.tokenizer.decode(input_tensor[0].tolist())
        self.model.train()
        return text

    def _handle_generation(self):
        """Generate samples based on configuration."""
        for i in range(self.args.num_samples_to_generate):
            sample = self.generate_sample()
            print(f"Sample {i+1}: {sample}")
            self.writer.add_text(f'generation/sample_{i+1}', sample, self.step)

    def save_model(self, path):
        state_dict = self.model.state_dict()
        safetensors.torch.save_file(state_dict, str(path))
        
        # Save in HuggingFace format if requested
        if self.args.save_hf_format:
            # Save model configuration
            config = {
            "model_type": "custom_transformer",
            "vocab_size": self.model.args.vocab_size,
            "hidden_size": self.model.args.n_embd,
            "num_hidden_layers": self.model.args.n_layers,
            "num_attention_heads": self.model.args.n_heads,
            "max_position_embeddings": self.model.args.max_seq_len,
            "layer_norm_eps": self.model.args.layer_norm_epsilon,
            "architectures": ["CustomTransformerLLM"],
            }
            
            with open(str(path).replace('.safetensors', '-config.json'), 'w') as f:
                json.dump(config, f, indent=2)

            # Save tokenizer in HF format 
            self.tokenizer.save(str(path).replace('.safetensors', '-tokenizer.json'))

    def train(self):
        print("Starting training...")
        start_time = time.time()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_aux_loss = 0
            
            # Progress tracking
            if self.args.use_tqdm:
                try:
                    from tqdm import tqdm
                    loader = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
                except ImportError:
                    loader = self.train_loader
                    print(f"TQDM not available, using standard progress")
            else:
                loader = self.train_loader
            
            for batch_idx, batch in enumerate(loader):
                loss, aux_loss = self.train_step(batch)
                epoch_loss += loss
                epoch_aux_loss += aux_loss
                
                # Enhanced logging
                if self.step % self.args.log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    log_msg = f"Step {self.step}: loss={loss:.4f}, aux_loss={aux_loss:.4f}, lr={current_lr:.2e}"
                    
                    if not self.args.use_tqdm:
                        print(log_msg)
                    
                    self.writer.add_scalar('train/loss', loss, self.step)
                    self.writer.add_scalar('train/aux_loss', aux_loss, self.step)
                    self.writer.add_scalar('train/lr', current_lr, self.step)
                    
                    # Log warmup progress
                    if self.step < self.warmup_steps:
                        self.writer.add_scalar('train/warmup_progress', self.step / self.warmup_steps, self.step)
                
                # Validation
                if self.val_loader and self.step % self.args.eval_every_n_steps == 0 and self.step > 0:
                    val_loss, val_aux_loss = self.validate()
                    print(f"Validation - loss: {val_loss:.4f}, aux_loss: {val_aux_loss:.4f}")
                    
                    self.writer.add_scalar('val/loss', val_loss, self.step)
                    self.writer.add_scalar('val/aux_loss', val_aux_loss, self.step)
                    
                    # Best model saving
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_model(self.output_dir / "best_model.safetensors")
                        print(f"New best model saved: {val_loss:.4f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.args.early_stopping_patience:
                            print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                            return
                
                # Generation
                if self.step % self.args.generate_every_n_steps == 0 and self.step > 0:
                    self._handle_generation()
                
                # Checkpointing
                if self.args.save_every_n_steps > 0 and self.step % self.args.save_every_n_steps == 0:
                    self.save_model(self.output_dir / f"checkpoint_{self.step}.safetensors")
                
                self.step += 1
            
            # End of epoch logging
            avg_loss = epoch_loss / self.batches_per_epoch
            avg_aux_loss = epoch_aux_loss / self.batches_per_epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.args.epochs}: avg_loss={avg_loss:.4f}, avg_aux_loss={avg_aux_loss:.4f}, lr={current_lr:.2e}")
            
            self.writer.add_scalar('epoch/avg_loss', avg_loss, epoch)
            self.writer.add_scalar('epoch/avg_aux_loss', avg_aux_loss, epoch)
            self.writer.add_scalar('epoch/lr', current_lr, epoch)
            
            # Save every epoch if requested
            if self.args.save_every_epoch:
                self.save_model(self.output_dir / f"epoch_{epoch+1}_model.safetensors")
        
        # Final save
        self.save_model(self.output_dir / "final_model.safetensors")
        self.tokenizer.save(str(self.output_dir / "tokenizer.json"))
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Final LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Best validation loss: {self.best_loss:.4f}")
        
        self.writer.close()


def main():
    # Create training arguments
    args = TrainingArgs()
    
    # Example of how to override default configurations
    # args.epochs = 10
    # args.batch_size = 8
    # args.learning_rate = 1e-4
    # args.generate_every_n_steps = 500
    
    # Initialize trainer
    trainer = Trainer(args)
    trainer.setup()
    
    # Estimate training requirements 
    if args.estimate_training_reqs:
        print(f"Estimated training requirements:")
        print(f"  - Total steps: {trainer.total_steps}")
        print(f"  - Warmup steps: {trainer.warmup_steps}")
        print(f"  - Memory usage: ~{sum(p.numel() * 4 for p in trainer.model.parameters()) / 1e9:.2f} GB (FP32)")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
