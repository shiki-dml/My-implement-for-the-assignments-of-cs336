import torch
from torch import nn
from einops import rearrange,einsum
import math


class Linear(nn.Module):
    def __init__(self,in_features,out_features,device = None,dtype = None):
        """
        in_features: int final dimnension of input
        out_features: int final dimnension of output
        device:torch.device | None = None Device to store the parameters on
        dtype:torch.dtype | None = None Data type of the parameters
        """
        #we perform a linear transformation with colomn-major weight matrix and column vector input
        #so the figure of input x is (in_features,1)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std =math.sqrt(2/(self.in_features+self.out_features))
        self.weight = nn.Parameter(torch.empty((self.out_features,self.in_features),device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight,std =std,a = -3*std,b = 3*std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight,x,'output input,... input -> ... output')