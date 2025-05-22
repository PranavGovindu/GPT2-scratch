from typing import NamedTuple, Optional


class ModelArgs(NamedTuple):
    """
    Configuration class for the LLM model and training setup.
    Using NamedTuple for immutable, structured configuration.
    """
    vocab_size: int = 8000    # Target vocabulary size for BPE tokenizer. Actual size might differ slightly after training.
    embed_dim: int = 256      # Dimensionality of token embeddings and hidden states.
    num_layers: int = 4       # Number of transformer blocks in the model.
    num_heads: int = 4        # Number of attention heads in Multi-Head Attention.
    num_kv_heads: Optional[int] = 2 # Number of key/value heads for Grouped Query Attention (GQA). If None, defaults to num_heads (MHA).
    num_latents: int = 16     # Number of latent vectors used in Multi-Head Latent Attention (MLA).
    moe_num_experts: int = 2  # Number of experts in the Mixture of Experts (MoE) layer.
    moe_top_k: int = 1        # Number of top experts to route tokens to in MoE.
    moe_ffn_dim_multiplier: Optional[float] = 2.0 # Multiplier for FFN hidden dim in MoE experts (e.g., 2.0 * embed_dim). If None, defaults to 4.0.
    moe_dropout_rate: float = 0.05 # Dropout rate within MoE expert FFNs.
    max_seq_len: int = 128    # Maximum sequence length for model inputs. Enforced by tokenizer padding/truncation.
    dropout_rate: float = 0.05 # General dropout rate for embeddings and attention.
    norm_eps: float = 1e-5    # Epsilon value for RMSNorm to prevent division by zero.
    rope_theta: float = 10000.0 # Theta parameter for Rotary Positional Embeddings (RoPE).
    rope_traditional: bool = False # Legacy RoPE implementation detail (not used here).
    use_gradient_checkpointing: bool = True # Whether to use gradient checkpointing to save memory during training.
