from typing import NamedTuple, Optional


class ModelArgs(NamedTuple):
    """
    Configuration class.
    """
    
    vocab_size: int = 8000     
    embed_dim: int = 256      
    num_layers: int = 4       
    num_heads: int = 4        
    num_kv_heads: Optional[int] = 2 

    @property
    def head_dim(self) -> int:
        """
        Calculate the dimension of each attention head.
        Returns the embedding dimension divided by the number of heads.
        """
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
    use_gradient_checkpointing: bool = True 
