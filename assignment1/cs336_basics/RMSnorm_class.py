import torch
from torch import nn

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super.__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
    def forward(self,x:torch.Tensor)->torch.Tensor:
        