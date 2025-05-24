#--------model.py--------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import ModelArgs
from ROPE import RotaryEmbedding
from RMSNorm import RMSNorm
from Transformer_Block import TransformerBlock 

class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.embed_dim)
        # RotaryEmbedding instance shared by all layers
        self.rotary_emb = RotaryEmbedding(
            dim=args.head_dim, # RoPE is applied up to head_dim
            max_seq_len=args.max_seq_len,
            theta=args.rope_theta
        )
        
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_layers)])
        
        self.norm_out = RMSNorm(args.embed_dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.embed_dim, args.vocab_size, bias=False)

        # Weight tying
        self.tok_embeddings.weight = self.lm_head.weight

    def forward(
        self,
        input_ids,
        kv_context_ids,
        kv_context_mask=None,
        start_pos=0,
        past_layer_latents_list=None,
        use_cache=False,
    ):
        _, q_len = input_ids.shape
        _, kv_ctx_len = kv_context_ids.shape

        query_embeddings = self.tok_embeddings(input_ids)
        kv_context_embeddings = self.tok_embeddings(kv_context_ids)

        # rope for query
        freqs_cis_query_seq = self.rotary_emb.get_freqs_cis(
            seq_len=q_len, 
            device=query_embeddings.device, 
            offset=start_pos
        )
        
        # rope for kv context
        freqs_cis_kv_context = self.rotary_emb.get_freqs_cis(
            seq_len=kv_ctx_len, 
            device=kv_context_embeddings.device, 
            offset=0
        )
        
        
        # cache initialization
        new_layer_latents_list = [] if use_cache else None
        
         # aux loss init 
        total_aux_loss = torch.tensor(0.0, device=query_embeddings.device, dtype=query_embeddings.dtype)
        
        # query sequence
        current_query_sequence = query_embeddings

        for i, layer in enumerate(self.layers):
            # past_layer_latents_list is a list of cached latents for each layer
            past_latent_kv_for_this_layer = None
            if past_layer_latents_list is not None and i < len(past_layer_latents_list) and past_layer_latents_list[i] is not None:
                past_latent_kv_for_this_layer = past_layer_latents_list[i]

            # pass the query sequence and context embeddings through the layer
            
            current_query_sequence, current_block_latent_kv_to_cache, block_aux_loss = layer(
                x_query_sequence=current_query_sequence,
                x_kv_context=kv_context_embeddings,
                rotary_emb_fn=self.rotary_emb,
                freqs_cis_query_seq=freqs_cis_query_seq,
                freqs_cis_kv_context=freqs_cis_kv_context,
                kv_context_mask=kv_context_mask,
                past_latent_kv=past_latent_kv_for_this_layer,
                use_cache=use_cache,
            )
            
            total_aux_loss += block_aux_loss

            # need this because of the way we cache 
            if use_cache:
                new_layer_latents_list.append(current_block_latent_kv_to_cache)

        # final lm head 
        final_output_normed = self.norm_out(current_query_sequence)
        logits = self.lm_head(final_output_normed)

        return logits, new_layer_latents_list, total_aux_loss
