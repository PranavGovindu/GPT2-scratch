#------moe.py--------#

import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import ModelArgs # Not strictly needed in this file if args is passed
import math
# from typing import Tuple # Removed as per no type hints

class FeedForwardExpert(nn.Module):
    """Single feed-forward network expert using SwiGLU activation."""
    def __init__(self, embed_dim, ffn_dim, dropout_rate):
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
    def __init__(self, args): # args is an instance of ModelArgs
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
        """
        Forward pass of Mixture of Experts (MoE) layer implementing a modified version of the routing algorithm 
        from "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (Lepikhin et al., 2020)
        and "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2021).

    
         
        Implementation Notes:
            - Derived from GShard paper's original algorithm but modified for better parallelization
            - Uses vectorized operations instead of explicit loops for expert assignment
            - Incorporates load balancing techniques from Switch Transformers
            - Capacity limiting prevents expert overloading while maintaining efficiency
            - Router noise addition during training improves expert utilization
        Reference Papers:
            - GShard: https://arxiv.org/abs/2006.16668
            - Switch Transformers: https://arxiv.org/abs/2101.03961
        """
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.shape[0]

        
        if num_tokens == 0:
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)

        router_logits = self.gate(x_flat)
        if self.training and self.router_noise_std > 0:
            noise = torch.randn_like(router_logits) * self.router_noise_std
            router_logits = router_logits + noise
        
        # top 2 experts (actually top_k)
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
        capacity = max(1, capacity) # Ensure capacity is at least 1
        
        flat_expert_indices = expert_indices.flatten()
        
        position_in_expert_assignment = torch.zeros_like(flat_expert_indices)
        for i in range(self.num_experts):
            expert_mask_i = (flat_expert_indices == i)
            # Calculate rank for assignments to expert i
            position_in_expert_assignment[expert_mask_i] = torch.arange(expert_mask_i.sum(), device=x.device) 
        
        within_capacity = (position_in_expert_assignment < capacity)
        capacity_mask = within_capacity.view(num_tokens, self.top_k)

        routing_weights = routing_weights_after_softmax * capacity_mask.to(routing_weights_after_softmax.dtype)
        sum_routing_weights = routing_weights.sum(dim=1, keepdim=True)
        routing_weights = torch.nan_to_num(routing_weights / (sum_routing_weights + 1e-8), nan=0.0)

        final_output_flat = torch.zeros_like(x_flat)
        

    
        for expert_idx_loop in range(self.num_experts):
            is_routed_to_current_expert = (expert_indices == expert_idx_loop)
            is_valid_assignment_for_expert = is_routed_to_current_expert & capacity_mask
            
            source_token_indices_for_expert, source_k_choices_for_expert = is_valid_assignment_for_expert.nonzero(as_tuple=True)

            if source_token_indices_for_expert.numel() == 0:
                continue

            inputs_for_this_expert = x_flat[source_token_indices_for_expert] 
            
            weights_for_this_expert = routing_weights[source_token_indices_for_expert, source_k_choices_for_expert]
            
            current_expert = self.experts[expert_idx_loop]
            expert_output = current_expert(inputs_for_this_expert) 
            
            weighted_expert_output = expert_output * weights_for_this_expert.unsqueeze(-1)
            
            final_output_flat.index_add_(0, source_token_indices_for_expert, weighted_expert_output.to(final_output_flat.dtype))
        
        return final_output_flat.view(bsz, seq_len, dim), aux_loss.to(x.dtype)
