import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelArgs
from ROPE import RotaryEmbedding 

class BaseGroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs, apply_rope_to_q: bool, apply_rope_to_k: bool):
        super().__init__()

        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads 

        self.head_dim = args.embed_dim // args.num_heads 

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
    ) :
        bsz, q_len, _ = query_states.shape
        _bsz_kv, kv_seq_len, _ = key_value_states.shape

        q_proj_out = self.q_proj(query_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k_proj_out = self.k_proj(key_value_states).view(_bsz_kv, kv_seq_len, self.num_kv_heads, self.head_dim)
        v_proj_out = self.v_proj(key_value_states).view(_bsz_kv, kv_seq_len, self.num_kv_heads, self.head_dim)

        if self.apply_rope_to_q and freqs_cis_q is not None:
            q = rotary_emb_fn(q_proj_out, freqs_cis_q)
        else:
            q = q_proj_out

        if self.apply_rope_to_k and freqs_cis_k is not None:
            k = rotary_emb_fn(k_proj_out, freqs_cis_k)
        else:
            k = k_proj_out
        v = v_proj_out

        num_key_value_groups = self.num_heads // self.num_kv_heads
        if num_key_value_groups > 1:
            k = k.unsqueeze(2).expand(_bsz_kv, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
            k = k.reshape(_bsz_kv, kv_seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(2).expand(_bsz_kv, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
            v = v.reshape(_bsz_kv, kv_seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        prepared_attention_mask = attention_mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                prepared_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        attn_output_intermediate = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=prepared_attention_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        attn_output = attn_output_intermediate.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.num_latents = args.num_latents
        self.embed_dim = args.embed_dim
        self.latent_queries = nn.Parameter(torch.randn(1, args.num_latents, self.embed_dim))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        self.latents_to_input_attn = BaseGroupedQueryAttention(args, apply_rope_to_q=False, apply_rope_to_k=True)
        self.input_to_latents_attn = BaseGroupedQueryAttention(args, apply_rope_to_q=True, apply_rope_to_k=False)

    def forward(
        self,
        x_query_sequence: torch.Tensor,
        x_kv_context: torch.Tensor,
        rotary_emb_fn: RotaryEmbedding,
        freqs_cis_q: Optional[torch.Tensor],
        freqs_cis_k: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_latent_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len_input_seq, _ = x_query_sequence.shape
        current_latent_kv_to_cache = None

        if past_latent_kv is not None:
            latent_k_prime, latent_v_prime = past_latent_kv
            if use_cache:
                current_latent_kv_to_cache = past_latent_kv
        else:
            expanded_latent_queries = self.latent_queries.expand(bsz, -1, -1)

            aggregated_latents = self.latents_to_input_attn(
                query_states=expanded_latent_queries,
                key_value_states=x_kv_context,
                rotary_emb_fn=rotary_emb_fn,
                freqs_cis_q=None,
                freqs_cis_k=freqs_cis_k,
                attention_mask=attention_mask,
            )
            latent_k_prime, latent_v_prime = aggregated_latents, aggregated_latents
            if use_cache:
                current_latent_kv_to_cache = (latent_k_prime, latent_v_prime)

        output = self.input_to_latents_attn(
            query_states=x_query_sequence,
            key_value_states=latent_k_prime,
            rotary_emb_fn=rotary_emb_fn,
            freqs_cis_q=freqs_cis_q,
            freqs_cis_k=None,
            attention_mask=None,
        )
        return output, current_latent_kv_to_cache
