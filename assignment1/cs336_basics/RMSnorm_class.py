import torch
from torch import nn
from math import sqrt
from einops import einsum

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        #process an input tensor of shape (batch_size,sequence_length,d_model) and
        #return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = [
            [
                [t/(sqrt(sum(l**2 for l in row)/self.d_model)+self.eps) for t in row]
                for row in layer
            ]
            for layer in x
        ]
        result_tensor = torch.tensor(result,dtype = torch.float32)
        true_resule = einsum(self.weight,result_tensor,'d_model,... d_model -> ... d_model')
        return true_resule.to(in_dtype)

