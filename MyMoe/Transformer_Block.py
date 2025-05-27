import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import ModelArgs
from ROPE import RotaryEmbedding
from RMSNorm import RMSNorm
from Attention import MultiHeadLatentAttention
from MoE import MoELayer
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from BpeTokenizer import BpeTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = MultiHeadLatentAttention(args)
        self.norm1 = RMSNorm(args.embed_dim, eps=args.norm_eps)
        self.moe_layer = MoELayer(args)
        self.norm2 = RMSNorm(args.embed_dim, eps=args.norm_eps)

    def _forward_impl(self, x_query_sequence, x_kv_context, rotary_emb_fn,
                      freqs_cis_query_seq, freqs_cis_kv_context,
                      kv_context_mask=None, past_layer_latent_kv=None, use_cache=False):
        normed_x_query = self.norm1(x_query_sequence)
        h_attn, cache = self.attention(
            normed_x_query, x_kv_context, rotary_emb_fn, freqs_cis_query_seq,
            freqs_cis_kv_context, kv_context_mask, past_layer_latent_kv, use_cache
        )
        x_query_sequence = x_query_sequence + h_attn
        normed_x_after_attn = self.norm2(x_query_sequence)
        moe_out, aux_loss = self.moe_layer(normed_x_after_attn)
        output_x = x_query_sequence + moe_out
        return output_x, cache, aux_loss

    def forward(self, x_query_sequence, x_kv_context, rotary_emb_fn,
                freqs_cis_query_seq, freqs_cis_kv_context,
                kv_context_mask=None, past_layer_latent_kv=None, use_cache=False):
        
        # grad checkpointing works only in training mode so need to use flag 
        if self.training and self.args.use_gradient_checkpointing:
            return grad_checkpoint(
                self._forward_impl, x_query_sequence, x_kv_context, rotary_emb_fn,
                freqs_cis_query_seq, freqs_cis_kv_context, kv_context_mask,
                past_layer_latent_kv, use_cache, use_reentrant=False
            )
        else:
            return self._forward_impl(
                x_query_sequence, x_kv_context, rotary_emb_fn,
                freqs_cis_query_seq, freqs_cis_kv_context,
                kv_context_mask, past_layer_latent_kv, use_cache
            )

class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.embed_dim)
        self.rotary_emb = RotaryEmbedding(
            dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta
        )
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_layers)])
        self.norm_out = RMSNorm(args.embed_dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.embed_dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight 

    def forward(self, input_ids, kv_context_ids, kv_context_mask=None,
                start_pos=0, past_layer_latents_list=None, use_cache=False):
        _, q_len = input_ids.shape
        _, kv_ctx_len = kv_context_ids.shape

        query_embeddings = self.tok_embeddings(input_ids)
        kv_context_embeddings = self.tok_embeddings(kv_context_ids)

        freqs_cis_query_seq = self.rotary_emb.get_freqs_cis(q_len, query_embeddings.device, offset=start_pos)
        freqs_cis_kv_context = self.rotary_emb.get_freqs_cis(kv_ctx_len, kv_context_embeddings.device, offset=0)

        new_layer_latents_list = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=query_embeddings.device, dtype=query_embeddings.dtype)
        current_query_sequence = query_embeddings

        for i, layer in enumerate(self.layers):
            past_layer_latent_kv = past_layer_latents_list[i] if past_layer_latents_list and i < len(past_layer_latents_list) else None

            current_query_sequence, layer_cache, block_aux_loss = layer(
                current_query_sequence,
                kv_context_embeddings,
                self.rotary_emb,
                freqs_cis_query_seq,
                freqs_cis_kv_context,
                kv_context_mask,
                past_layer_latent_kv,
                use_cache
            )

            total_aux_loss += block_aux_loss
            if use_cache:
                new_layer_latents_list.append(layer_cache)

        final_output = self.norm_out(current_query_sequence)
        logits = self.lm_head(final_output)
        return logits, new_layer_latents_list, total_aux_loss
    

