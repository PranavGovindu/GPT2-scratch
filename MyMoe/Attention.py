# attention_test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import ModelArgs # Use the test config
from ROPE import RotaryEmbedding

class BaseGroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs, apply_rope_to_q: bool, apply_rope_to_k: bool):
        super().__init__()

        if not (args.num_kv_heads is not None): # Check if num_kv_heads is provided
            raise ValueError("num_kv_heads must be specified in ModelArgs for BaseGroupedQueryAttention.")
        if args.num_kv_heads >= args.num_heads:
            raise ValueError(
                f"For BaseGroupedQueryAttention, num_kv_heads ({args.num_kv_heads}) "
                f"must be less than num_heads ({args.num_heads})."
            )
        if args.num_heads % args.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({args.num_heads}) must be divisible by "
                f"num_kv_heads ({args.num_kv_heads}) for GQA."
            )

        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.head_dim # Access derived property

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
        bsz, q_len, _ = query_states.shape
        _bsz_kv, kv_seq_len, _ = key_value_states.shape
        assert bsz == _bsz_kv, "Batch sizes for query and key/value states must match"        # Project and reshape: (batch, seq_len, num_*_heads, head_dim)
        q_proj = self.q_proj(query_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k_proj = self.k_proj(key_value_states).view(bsz, kv_seq_len, self.num_kv_heads, self.head_dim)
        v_proj = self.v_proj(key_value_states).view(bsz, kv_seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE if specified
        if self.apply_rope_to_q and freqs_cis_q is not None:
            assert freqs_cis_q.shape[0] == q_len, f"freqs_cis_q length {freqs_cis_q.shape[0]} != q_len {q_len}"
            q = rotary_emb_fn(q_proj, freqs_cis_q)
        else:
            q = q_proj

        if self.apply_rope_to_k and freqs_cis_k is not None:
            assert freqs_cis_k.shape[0] == kv_seq_len, f"freqs_cis_k length {freqs_cis_k.shape[0]} != kv_seq_len {kv_seq_len}"
            k = rotary_emb_fn(k_proj, freqs_cis_k)
        else:
            k = k_proj
        v = v_proj

        # Adjust k and v for grouped query attention by repeating
        num_key_value_groups = self.num_heads // self.num_kv_heads
        k = k.unsqueeze(2).expand(bsz, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
        k = k.reshape(bsz, kv_seq_len, self.num_heads, self.head_dim)
        v = v.unsqueeze(2).expand(bsz, kv_seq_len, num_key_value_groups, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, kv_seq_len, self.num_heads, self.head_dim)

        # Prepare for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # print(f"BaseGQA before F.sdpa: q={q.shape}, k={k.shape}, v={v.shape}")
        # if attention_mask is not None: print(f"  mask={attention_mask.shape}")

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
        if not (args.num_kv_heads is not None and args.num_kv_heads < args.num_heads):
            raise ValueError("MLA requires ModelArgs to be GQA-compliant (num_kv_heads < num_heads).")
        if args.num_heads % args.num_kv_heads != 0:
             raise ValueError(f"num_heads must be divisible by num_kv_heads.")

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
        freqs_cis_query_seq: torch.Tensor,
        freqs_cis_kv_context: torch.Tensor,
        kv_context_mask: Optional[torch.Tensor] = None,
        past_latent_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_main_len, _ = x_query_sequence.shape
        # ctx_len = x_kv_context.shape[1] # Not directly used here but good for clarity
        current_latent_kv_to_cache = None

        print(f"\n--- MLA Forward Pass ---")
        print(f"  Input x_query_sequence: {x_query_sequence.shape}")
        print(f"  Input x_kv_context: {x_kv_context.shape}")

        if past_latent_kv is not None:
            latent_k_prime, latent_v_prime = past_latent_kv
            print(f"  Using past_latent_kv. K' shape: {latent_k_prime.shape}")
        else:
            print(f"  Stage 1: Latents attend to x_kv_context")
            expanded_latent_queries = self.latent_queries.expand(bsz, -1, -1)
            print(f"    Q (latents): {expanded_latent_queries.shape}, K/V (context): {x_kv_context.shape}")
            if kv_context_mask is not None: print(f"    kv_context_mask: {kv_context_mask.shape}")
            
            aggregated_latents = self.latents_to_input_attn(
                query_states=expanded_latent_queries,
                key_value_states=x_kv_context,
                rotary_emb_fn=rotary_emb_fn,
                freqs_cis_q=None,
                freqs_cis_k=freqs_cis_kv_context,
                attention_mask=kv_context_mask,
            )
            latent_k_prime, latent_v_prime = aggregated_latents, aggregated_latents
            print(f"    Aggregated latents shape: {aggregated_latents.shape}")
            if use_cache:
                current_latent_kv_to_cache = (latent_k_prime, latent_v_prime)

        print(f"  Stage 2: x_query_sequence attends to aggregated latents")
        print(f"    Q (query_seq): {x_query_sequence.shape}, K/V (latents): {latent_k_prime.shape}")
        output = self.input_to_latents_attn(
            query_states=x_query_sequence,
            key_value_states=latent_k_prime,
            rotary_emb_fn=rotary_emb_fn,
            freqs_cis_q=freqs_cis_query_seq,
            freqs_cis_k=None,
            attention_mask=None,
        )
        print(f"  Output of MLA shape: {output.shape}")
        return output, current_latent_kv_to_cache
