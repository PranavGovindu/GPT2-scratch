from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import os
from torch.utils.tensorboard import SummaryWriter
import inspect

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

@dataclass
class GPTconfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

class CausalSelfAttention(nn.Module):
    def _init_(self, config):
        super()._init_()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def _init_(self, config):
        super()._init_()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def _init_(self, config):
        super()._init_()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def _init_(self, config: GPTconfig):
        super()._init_()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized: {n_params/1e6:.2f}M parameters")
        print(f"Vocab size: {config.vocab_size}, Block size: {config.block_size}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if module == self.transformer.h[0].mlp.c_proj or module == self.transformer.h[0].attn.c_proj:
                 std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.config.block_size:
             idx = idx[:, -self.config.block_size:]
             T = self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            targets_cropped = targets[:, -T:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_cropped.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    print("Tokenizer does not have pad_token, setting to eos_token")
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
print(f"Tokenizer loaded. Vocab size: {vocab_size}")
print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

current_dir = os.path.dirname(_file) if "file_" in locals() else "."
train_tokens_path = os.path.join(current_dir, 'tinystories_train_tokens.pt')
val_tokens_path = os.path.join(current_dir, 'tinystories_val_tokens.pt')

if os.path.exists(train_tokens_path) and os.path.exists(val_tokens_path):
    print(f"Loading pre-tokenized data from {train_tokens_path} and {val_tokens_path}...")
    train_data = torch.load(train_tokens_path)
    val_data = torch.load(val_tokens_path)
    print(f"Loaded train tokens: {len(train_data):,}")
    print(f"Loaded validation tokens: {len(val_data):,}")
else:
    print("Tokenized data not found. Processing dataset...")
    print("Loading TinyStories dataset from Hugging Face Hub...")
    try:
        raw_train_ds = load_dataset("roneneldan/TinyStories", split="train")
        raw_val_ds = load_dataset("roneneldan/TinyStories", split="validation")
        print(f"Raw dataset loaded. Train examples: {len(raw_train_ds):,}, Validation examples: {len(raw_val_ds):,}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and 'datasets' library installed.")
        exit()

    def tokenize_and_concat(dataset, tokenizer):
        all_tokens = []
        print(f"Tokenizing and concatenating {len(dataset):,} stories...")
        for example in tqdm(dataset):
            text = example['text']
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens.append(tokenizer.eos_token_id)
                all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)

    train_data = tokenize_and_concat(raw_train_ds, tokenizer)
    val_data = tokenize_and_concat(raw_val_ds, tokenizer)

    print("Tokenization complete.")
    print(f"Total train tokens: {len(train_data):,}")
    print(f"Total validation tokens: {len(val_data):,}")

    print(f"Saving tokenized data to {train_tokens_path} and {val_tokens_path}...")
    torch.save(train_data, train_tokens_path)
    torch.save(val_data, val_tokens_path)
    print("Data saved.")

batch_size = 16
block_size = 256
max_steps = 10000
gradient_accumulation_steps = 8
effective_batch_size = batch_size * gradient_accumulation_steps
print(f"Effective batch size: {batch_size} * {gradient_accumulation_steps} = {effective_batch_size}")

eval_interval = 500
learning_rate = 3e-4
weight_decay = 0.1
warmup_steps = 200
min_lr = 3e-5
grad_clip = 1.0

log_dir = "runs/gpt2_tinystories"
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using device: {device}")

if device_type == 'cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
     print("Enabling TF32")
     torch.backends.cuda.matmul.allow_tf32 = True
     torch.backends.cudnn.allow_tf32 = True

use_amp = (device_type == 'cuda')
pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
ctx_amp = torch.autocast(device_type=device_type, dtype=pt_dtype, enabled=use_amp)
print(f"Using Automatic Mixed Precision (AMP): {use_amp} with dtype {pt_dtype if use_amp else 'torch.float32'}")

config = GPTconfig(block_size=block_size, vocab_size=vocab_size)
model = GPT(config)
model.to(device)

compile_model = False
if compile_model and hasattr(torch, 'compile'):
    print("Compiling the model)")
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")
else:
    if compile_model:
        print(" Proceeding without compilation.")
    else:
        print("torch.compile disabled.")

optimizer = model.configure_optimizers(weight_decay, learning_rate, device_type)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_index = len(data) - block_size
    ix = torch.randint(max_index, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

def get_lr(it):
    if it < warmup_steps:
        return learning_rate * it / warmup_steps
    decay_end_step = max_steps
    if it > decay_end_step:
        return min_lr
    decay_ratio = (it - warmup_steps) / (decay_end_step - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Decay ratio out of bounds: {decay_ratio} at step {it}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_iters = 200
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            eval_model = model._orig_mod if compile_model and hasattr(model, '_orig_mod') else model
            logits, loss = eval_model(X, Y)
            if loss is not None:
                 losses[k] = loss.item()
            else:
                 losses[k] = float('nan')
        out[split] = losses.mean()
    model.train()
    return out

print("\nStarting training loop")
t_total_start = time.time()
t0 = time.time()
best_val_loss = float('inf')
step = 0

optimizer.zero_grad(set_to_none=True)

while step < max_steps:
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('LearningRate', lr, step)

    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"\nStep {step:5d}: Est. train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        writer.add_scalar('Loss/val', losses['val'], step)
        writer.add_scalar('Loss/train_eval', losses['train'], step)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'step': step,
                'best_val_loss': best_val_loss,
                'tokenizer_name': 'gpt2',
            }
            ckpt_path = os.path.join(current_dir, 'gpt2_tinystories_best.pth')
            print(f"  -> Saving best checkpoint to {ckpt_path} (val loss: {best_val_loss:.4f})")
            torch.save(checkpoint, ckpt_path)

        latest_checkpoint = {
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'config': config,
             'step': step,
             'best_val_loss': best_val_loss,
             'tokenizer_name': 'gpt2',
         }
        periodic_ckpt_path = os.path.join(current_dir, 'gpt2_tinystories_latest.pth')
        torch.save(latest_checkpoint, periodic_ckpt_path)

    accumulated_raw_loss = 0.0
    micro_steps_ran = 0
    model.train()

    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        with ctx_amp:
            logits, raw_loss = model(xb, yb)

        if raw_loss is not None:
            micro_steps_ran += 1
            accumulated_raw_loss += raw_loss.item()
            scaled_loss = raw_loss / gradient_accumulation_steps
            scaled_loss.backward()
        else:
            print(f"Warning: Loss is None in micro_step {micro_step} of step {step}. Skipping backward.")

    if micro_steps_ran > 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        writer.add_scalar('GradientNorm', norm.item(), step)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        avg_loss_for_step = accumulated_raw_loss / micro_steps_ran
        writer.add_scalar('Loss/train_step', avg_loss_for_step, step)

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = batch_size * block_size * gradient_accumulation_steps
        tokens_per_sec = tokens_processed / (dt / 1000) if dt > 0 else 0

        writer.add_scalar('Timing/step_time_ms', dt, step)
        writer.add_scalar('Timing/tokens_per_sec', tokens_per_sec, step)

        print(f"Step {step:5d}/{max_steps} | Avg Loss: {avg_loss_for_step:.4f} | LR: {lr:.2e} | Norm: {norm:.4f} | dt: {dt:.2f}ms | Tokens/sec: {tokens_per_sec:.0f}")
        t0 = t1

    else:
         print(f"Step {step}: Skipping optimizer .")
         optimizer.zero_grad(set_to_none=True)
         t0 = time.time()

    step += 1

t_total_end = time.time()
print("\nTraining finished.")
print(f"Total training time: {(t_total_end - t_total_start)/60:.2f} minutes")
print(f"Best validation loss achieved: {best_val_loss:.4f}")
writer.close()

print("\n Generating Sample")

model_to_load = None
checkpoint_path = os.path.join(current_dir, 'gpt2_tinystories_best.pth')

if os.path.exists(checkpoint_path):
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint['config']
        tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
        print(f"Loading tokenizer '{tokenizer_name}' used during training")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer reloaded.")

        model_gen = GPT(config)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model_gen.load_state_dict(state_dict)
        print("Model state dict loaded successfully.")

        model_gen.eval()
        model_gen.to(device)
        model_to_load = model_gen
        print(f"Loaded model from step {checkpoint.get('step', 'N/A')} with val loss {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}.")
        model_to_load = None

else:
    print(f"Checkpoint file '{checkpoint_path}' not found.")

if model_to_load is not None:
    start_context_ids = [tokenizer.eos_token_id]
    context = torch.tensor([start_context_ids], dtype=torch.long, device=device)

    print(f"\nGenerating text  ({tokenizer.eos_token})...")
    print("-" * 30)

    with torch.no_grad():
        with ctx_amp:
             generated_tokens = model_to_load.generate(idx=context,
                                                max_new_tokens=200,
                                                temperature=0.75,
                                                top_k=40)
             generated_text = tokenizer.decode(generated_tokens[0, 1:].tolist())
             print(generated_text)

    print("-" * 30)
else:
    print("Model was not successfully loaded.")
