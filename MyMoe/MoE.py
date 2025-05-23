import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelArgs
import math
from typing import Tuple

class FeedForwardExpert(nn.Module):
    """Single feed-forward network expert using SwiGLU activation."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout_rate: float):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) :
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoELayer(nn.Module):
    """Mixture of Experts layer with capacity factor and router noise."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.moe_num_experts
        self.top_k = args.moe_top_k
        self.capacity_factor = args.moe_capacity_factor
        self.router_noise_std = args.moe_router_noise
        self.aux_loss_coef = 0.01
        
        ffn_dim = int(args.embed_dim * getattr(args, 'moe_ffn_dim_multiplier', 4.0))
        
        self.experts = nn.ModuleList([
            FeedForwardExpert(args.embed_dim, ffn_dim, args.moe_dropout_rate)
            for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(args.embed_dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) :
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.shape[0]

        if num_tokens == 0:
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Router logits with optional noise
        router_logits = self.gate(x_flat)
        if self.training and self.router_noise_std > 0:
            router_logits += torch.randn_like(router_logits) * self.router_noise_std
        
        # Top-k routing
        raw_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(raw_weights, dim=-1, dtype=torch.float32)

        # Auxiliary loss
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        tokens_per_expert = expert_mask.float().sum(dim=0)
        token_fractions = tokens_per_expert / num_tokens if num_tokens > 0 else torch.zeros_like(tokens_per_expert)
        router_probs = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32).mean(dim=0)
        aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(token_fractions * router_probs)

        # Capacity-based token dropping
        capacity = min(max(1, math.ceil(num_tokens * self.capacity_factor / self.num_experts)), num_tokens)
        
        # Use cumsum to enforce capacity limits
        flat_experts = expert_indices.flatten()
        expert_one_hot = F.one_hot(flat_experts, self.num_experts)
        ranks = (torch.cumsum(expert_one_hot, dim=0) - 1) * expert_one_hot
        capacity_mask = torch.gather(ranks < capacity, dim=1, index=flat_experts.unsqueeze(1)).squeeze(1)
        capacity_mask = capacity_mask.view(num_tokens, self.top_k)
        
        # Apply capacity mask to routing weights
        routing_weights *= capacity_mask.to(routing_weights.dtype)

        # Expert computation
        flat_inputs = x_flat.repeat_interleave(self.top_k, dim=0)
        expert_outputs = torch.zeros_like(flat_inputs)
        
        for i in range(self.num_experts):
            mask = (flat_experts == i) & capacity_mask.flatten()
            if mask.any():
                expert_outputs[mask] = self.experts[i](flat_inputs[mask])

        # Combine expert outputs
        expert_outputs = expert_outputs.view(num_tokens, self.top_k, dim)
        final_output = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)
        
        return final_output.view(bsz, seq_len, dim), aux_loss.to(x.dtype)
