# attention_test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import ModelArgs 
from ROPE import RotaryEmbedding

class BaseGroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs, apply_rope_to_q: bool, apply_rope_to_k: bool):
        super().__init__()



        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.head_dim

        self.apply_rope_to_q = apply_rope_to_q
        self.apply_rope_to_k = apply_rope_to_k

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        self.attn_dropout_p = args.dropout_rate

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        rotary_emb_fn: RotaryEmbedding,
        freqs_cis_q: Optional[torch.Tensor],
        freqs_cis_k: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Grouped Query Attention forward pass.

        Args:
            query_states (torch.Tensor): Query input of shape (batch_size, query_length, embed_dim)
            key_value_states (torch.Tensor): Key/Value input of shape (batch_size, kv_length, embed_dim)
            rotary_emb_fn (RotaryEmbedding): Function to apply rotary embeddings
            freqs_cis_q (Optional[torch.Tensor]): RoPE frequencies for queries
            freqs_cis_k (Optional[torch.Tensor]): RoPE frequencies for keys 
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_length, embed_dim)
        """
        bsz, q_len, _ = query_states.shape
        _bsz, kv_seq_len, _ = key_value_states.shape
        q_proj = self.q_proj(query_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k_proj = self.k_proj(key_value_states).view(bsz, kv_seq_len, self.num_kv_heads, self.head_dim)
        v_proj = self.v_proj(key_value_states).view(bsz, kv_seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE if specified
        if self.apply_rope_to_q and freqs_cis_q is not None:
            q = rotary_emb_fn(q_proj, freqs_cis_q)
        else:
            q = q_proj

        if self.apply_rope_to_k and freqs_cis_k is not None:
            k = rotary_emb_fn(k_proj, freqs_cis_k)
        else:
            k = k_proj
        v = v_proj

        num_key_value_groups = self.num_heads // self.num_kv_heads
        k = k.unsqueeze(2).expand(bsz, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
        k = k.reshape(bsz, kv_seq_len, self.num_heads, self.head_dim)
        v = v.unsqueeze(2).expand(bsz, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, kv_seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        attn_output_intermediate = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        
        attn_output = attn_output_intermediate.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        return self.o_proj(attn_output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()


        self.num_latents = args.num_latents
        self.embed_dim = args.embed_dim
        self.latent_queries = nn.Parameter(torch.randn(1, args.num_latents, self.embed_dim))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        self.latents_to_input_attn = BaseGroupedQueryAttention(args, apply_rope_to_q=False, apply_rope_to_k=True) # stage 1
        
        self.input_to_latents_attn = BaseGroupedQueryAttention(args, apply_rope_to_q=True, apply_rope_to_k=False) # stage 2

    def forward(
        self,
        x_query_sequence: torch.Tensor,
        x_kv_context: torch.Tensor,
        rotary_emb_fn: RotaryEmbedding,
        freqs_cis_query_seq: torch.Tensor,
        freqs_cis_kv_context: torch.Tensor,
        kv_context_mask: Optional[torch.Tensor] = None,
        past_latent_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, #depends on the stages
        use_cache: bool = False, 
    ) :
        
        """
        Multi-Head Latent Attention implementation
        """
        bsz, q_len, _ = x_query_sequence.shape
        
        current_latent_kv_to_cache = None


        if past_latent_kv is not None:
            latent_k_prime, latent_v_prime = past_latent_kv
        else:
            expanded_latent_queries = self.latent_queries.expand(bsz, -1, -1)

            
            aggregated_latents = self.latents_to_input_attn(
                query_states=expanded_latent_queries,
                key_value_states=x_kv_context,
                rotary_emb_fn=rotary_emb_fn,
                freqs_cis_q=None,
                freqs_cis_k=freqs_cis_kv_context,
                attention_mask=kv_context_mask,
            )
            latent_k_prime, latent_v_prime = aggregated_latents, aggregated_latents
            if use_cache:
                current_latent_kv_to_cache = (latent_k_prime, latent_v_prime)

        output = self.input_to_latents_attn(
            query_states=x_query_sequence,
            key_value_states=latent_k_prime,
            rotary_emb_fn=rotary_emb_fn,
            freqs_cis_q=freqs_cis_query_seq,
            freqs_cis_k=None,
            attention_mask=None,
        )
        return output, current_latent_kv_to_cache
