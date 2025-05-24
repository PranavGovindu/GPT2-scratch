#------moe.py--------#

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

    def forward(self, x):
        # SwiGLU 
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoELayer(nn.Module):
    """Mixture of Experts layer with capacity factor and router noise."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.moe_num_experts
        self.top_k = args.moe_top_k
        self.capacity_factor = args.moe_capacity_factor
        self.router_noise_std = args.moe_router_noise
        self.aux_loss_coef = args.moe_aux_loss_coef
        
        ffn_dim_multiplier = args.moe_ffn_dim_multiplier
        ffn_dim = int(args.embed_dim * ffn_dim_multiplier)
        
        # experts 
        self.experts = nn.ModuleList([
            FeedForwardExpert(args.embed_dim, ffn_dim, args.moe_dropout_rate)
            for _ in range(self.num_experts)
        ])
        
        # router
        self.gate = nn.Linear(args.embed_dim, self.num_experts, bias=False)

    def forward(self, x) :
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.shape[0]

        
        if num_tokens == 0:
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)

        router_logits = self.gate(x_flat)
        if self.training and self.router_noise_std > 0:
            noise = torch.randn_like(router_logits) * self.router_noise_std
            router_logits = router_logits + noise
        
        # top 2 experts
        raw_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        routing_weights_after_softmax = F.softmax(raw_weights, dim=-1, dtype=torch.float32)

        # aux loss
        router_probs_all_experts = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # avg router probs
        mean_router_probs_per_expert = torch.mean(router_probs_all_experts, dim=0)
        
        tokens_per_expert_one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).sum(dim=1)
        # fractions 
        fraction_tokens_per_expert = tokens_per_expert_one_hot.float().mean(dim=0)
        
        aux_loss = self.aux_loss_coef * torch.sum(fraction_tokens_per_expert * mean_router_probs_per_expert) * self.num_experts

        ideal_capacity_per_expert = (num_tokens / self.num_experts) * self.capacity_factor
        capacity = math.floor(ideal_capacity_per_expert)
        capacity = max(1, capacity)
        
        flat_expert_indices = expert_indices.flatten()
        
        position_in_expert_assignment = torch.zeros_like(flat_expert_indices)
        for i in range(self.num_experts):
            expert_mask_i = (flat_expert_indices == i)
            position_in_expert_assignment[expert_mask_i] = torch.arange(expert_mask_i.sum(), device=x.device)
        
        within_capacity = (position_in_expert_assignment < capacity)
        capacity_mask = within_capacity.view(num_tokens, self.top_k)

        routing_weights = routing_weights_after_softmax * capacity_mask.to(routing_weights_after_softmax.dtype)
        sum_routing_weights = routing_weights.sum(dim=1, keepdim=True)
        routing_weights = torch.nan_to_num(routing_weights / (sum_routing_weights + 1e-8))

        final_output_flat = torch.zeros_like(x_flat)
        
        # sending tokens to experts loop
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                # check if capacity is not exceeded
                if capacity_mask[token_idx, k_idx]:
                    expert_idx = expert_indices[token_idx, k_idx].item()
                    weight = routing_weights[token_idx, k_idx]
                    if weight > 0:
                         expert_output = self.experts[expert_idx](x_flat[token_idx].unsqueeze(0))
                         final_output_flat[token_idx] += weight * expert_output.squeeze(0)
        
        return final_output_flat.view(bsz, seq_len, dim), aux_loss.to(x.dtype)
