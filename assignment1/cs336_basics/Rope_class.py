import torch
from torch import nn

class RoPe(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device = None):
        """
        theta: float \theta value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        