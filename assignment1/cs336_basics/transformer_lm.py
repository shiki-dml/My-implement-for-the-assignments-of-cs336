import torch
from torch import nn
import einops
from cs336_basics.RMSnorm_class import RMSnorm
from cs336_basics.Causal_Multi_Head_Self_Attention import CausalMultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta=None,max_seq_len=None,device = None):
        """
        d_model: int Dimensionality of the Transformer block inputs
        num_heads: int Number of heads to use in multi-head self-attention
        d_ff: int Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        self.CMHA = CausalMultiHeadSelfAttention(d_model,num_heads,theta,max_seq_len)
        self.RMS1 = RMSnorm(d_model)
        self.RMS2 = RMSnorm(d_model)
        self.SwiGLU= SwiGLU(d_model,d_ff)
    def forward(self,x:torch.Tensor):
        prenorm = x+self.CMHA(self.RMS1(x))
        output = prenorm + self.SwiGLU(self.RMS2(prenorm))
        return output        
        