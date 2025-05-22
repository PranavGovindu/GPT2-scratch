import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch.nn import functional as F

class RotaryEmbedding(nn.Module):
    """
    implements rope 
    """
    
    def __init__(self, dim: int,max_seq_len:int,theta:float=10000.0):
        """

        Args:
            dim (int): dimensions to be rotated
            max_seq_len (int):max sequence length
            theta (float, optional): hyperparameter for the rotation. Defaults to 10000.0.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

 
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False) # Not part of state_dict

     
        # Use 'meta' device initially if freqs is on meta, otherwise cpu for buffer init
        init_device = freqs.device if freqs.device.type != 'meta' else torch.device('cpu')
        self._update_freqs_cis_cache(max_seq_len, init_device)
        
    def _update_freqs_cis_cache(self, seq_len: int, device: torch.device):
        """
        Precomputes and caches the complex numbers `cos(m*theta_i) + i*sin(m*theta_i)`
        for positions m from 0 to seq_len-1.
        Args:
            seq_len: The sequence length for which to compute `freqs_cis`.
            device: The device to store the cached `freqs_cis` on.
        """
        #no need to reshape the dimensions to (seq_len, 1) because we are using outer product
        # and the freqs are already in the right shape
        m= torch.arange(seq_len, device=device)
 
        freqs_for_m = torch.outer(m, self.freqs.to(device))
  
        freqs_cis = torch.polar(torch.ones_like(freqs_for_m), freqs_for_m)
        self.register_buffer("freqs_cis_cached", freqs_cis, persistent=False)
        
    def get_freqs_cis(self, seq_len: int, device: torch.device, offset: int = 0):
        """
        Returns the cached complex rotation values  
        starting from position `offset`
        Recomputes the cache if needed.
        """
        required_len = offset + seq_len

        # Recompute if cache is missing, too short, or on wrong device
        if not hasattr(self, "freqs_cis_cached") or self.freqs_cis_cached.device != device or self.freqs_cis_cached.shape[0] < required_len:
            self._update_freqs_cis_cache(max(self.max_seq_len, required_len), device)

        return self.freqs_cis_cached[offset : offset + seq_len]

        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary embeddings to the input tensor x.

        Args:
            x: Tensor of shape (B, S, H, D) or (B, H, S, D), where D = self.dim.
            
            the conditional checking is required for the type or architecture of the model
            
            freqs_cis: Precomputed complex rotations of shape (S, D/2).

        Returns:
            Tensor of same shape as x, with rotary embeddings applied.
        """
        x_shape = x.shape

        # Detect sequence dimension based on freqs_cis compatibility
        if x.shape[1] == freqs_cis.shape[0]:
            seq_dim = 1  # (B, S, H, D)
        elif x.shape[2] == freqs_cis.shape[0]:
            seq_dim = 2  # (B, H, S, D)
        else:
            raise ValueError(f"Incompatible shapes: x={x.shape}, freqs_cis={freqs_cis.shape}")

        # first : convert to float 
        # second : reshape from (B,S,H,D) to (B,S,H,D/2,2) (the (-1,2) does the split automatically)
        # third : convert to complex pairs 
        x_complex = torch.view_as_complex(x.float().reshape(*x_shape[:-1], -1, 2))

        # Reshape freqs_cis for broadcasting
        shape = [1] * x_complex.ndim
        shape[seq_dim] = freqs_cis.shape[0]
        shape[-1] = freqs_cis.shape[1]
        freqs_cis = freqs_cis.view(*shape)

        # Apply rotation
        x_rotated = x_complex * freqs_cis
        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)

        
        
# example usage

# if __name__ == "__main__":
#     B, S, H, D = 1, 4, 1, 8  # smaller for easier visualization
#     x = torch.arange(B*S*H*D).float().reshape(B, S, H, D)  # simple increasing numbers for clarity

#     print("Original x:")
#     print(x)

#     rope = RotatoryEmbedding(dim=D, max_seq_len=S)

#     # Get the freqs_cis for the input sequence length
#     freqs_cis = rope.get_freqs_cis(seq_len=S, device=x.device)

#     # Apply the rotary embeddings
#     x_rotated = rope(x, freqs_cis)

#     print("\nRotated x:")
#     print(x_rotated)

#     print("\nShape before:", x.shape)
#     print("Shape after:", x_rotated.shape)
