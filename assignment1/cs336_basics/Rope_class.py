import torch
from torch import nn
import math
import einops
from einops import rearrange

def rotate_half(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RoPe(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device = None):
        """
        theta: float \theta value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        indice  = torch.arange(0,d_k,2,dtype = torch.float32,device = device)
        freq = 1.0/(theta**(indice/d_k))
        t = torch.arange(max_seq_len,dtype = torch.float32,device = device)
        freqs = torch.outer(t,freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos = self.cos_cached[token_positions] # (batch, seq, d)
        sin = self.sin_cached[token_positions]
        
        return (x * cos) + (rotate_half(x) * sin)