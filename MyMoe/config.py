# config.py
# from typing import NamedTuple, Optional # Remove NamedTuple if not used elsewhere for it
from typing import Optional # Keep Optional
from dataclasses import dataclass # Import dataclass

@dataclass # Decorate with @dataclass
class ModelArgs:
    vocab_size: int = 8000     
    embed_dim: int = 768     
    num_layers: int = 12      
    num_heads: int = 4        
    num_kv_heads: int = 2 

    # Keep your property if it's still relevant
    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads
    
    num_latents: int = 16    
    moe_num_experts: int = 2  
    moe_top_k: int = 1
    moe_ffn_dim_multiplier: Optional[float] = 2.0 
    moe_dropout_rate: float = 0.05 
    max_seq_len: int = 128    
    dropout_rate: float = 0.05 
    norm_eps: float = 1e-5    
    rope_theta: float = 10000.0 
    rope_traditional: bool = False 
    # use_gradient_checkpointing: bool = True # You had this twice, ensure only one
    moe_capacity_factor: float = 1.25
    moe_router_noise: float = 0.1
    use_gradient_checkpointing: bool = True
    moe_aux_loss_coef: float = 0.01
