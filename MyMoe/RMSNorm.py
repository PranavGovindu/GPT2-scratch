import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

    
# testing

# if __name__ == "__main__":
#     rmsnorm = RMSNorm(10)
#     x = torch.randn(2, 10)
#     print(x)
#     print(rmsnorm.weight)
#     output = rmsnorm(x)
#     print(output)


