import torch
from torch import nn
import einops
from cs336_basics.Dot_Product_Attention import softmax

def cross_entropy(input:torch.Tensor, target):
    c = input.max(dim = -1,keepdim = True).values
    test = input -c
    idx = torch.arange(input.shape[0])
    out = torch.log(torch.exp(test).sum(dim = 1))
    return (out[idx] - test[idx,target]).mean()


