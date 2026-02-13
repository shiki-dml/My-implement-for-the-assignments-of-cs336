import torch
from torch import nn
from einops import einsum
from cs336_basics.Linear_class import Linear
def SiLU(x):
    return x*torch.sigmoid(x)
def GLU(x,W1,W2):
    W1_0 = einsum(W1,x,'... d_model,d_model -> ... d_model')
    W2_0 = einsum(W2,x,'... d_model,d_model -> ... d_model')
    return torch.sigmoid(W1_0)*W2_0

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff = None):
        super().__init__()
        if(d_ff == None):
            self.d_ff = int(d_model*8/3+0.5)*64
        else:
            self.d_ff = d_ff
        self.d_model = d_model
        self.W1 = Linear(d_model,self.d_ff)
        self.W2 = Linear(self.d_ff,d_model)
        self.W3 = Linear(d_model,self.d_ff)

    def forward(self,x):
        W1_0 = einsum(self.W1.weight,x,'d_ff d_model,... d_model -> ... d_ff')
        W3_0 = einsum(self.W3.weight,x,'d_ff d_model,... d_model -> ... d_ff')
        return einsum(self.W2.weight,SiLU(W1_0)*W3_0,'d_model d_ff,... d_ff ->... d_model')
