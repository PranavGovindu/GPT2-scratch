import torch
from config import ModelArgs
from Transformer_Block import TransformerBlock

# Dummy args for testing
args = ModelArgs()

# Create a block
block = TransformerBlock(args)

# Dummy input: (batch, seq, dim)
batch = 2
seq = 8
dim = args.embed_dim
x = torch.randn(batch, seq, dim)

# Forward pass
y, aux_loss, cache = block(x,use_cache=True)
print('Output shape:', y.shape)
print('Aux loss:', aux_loss)
print('Cache:', cache)
