import torch
from torch import nn
import einops
import math
from einops import einsum


def softmax(x:torch.Tensor,i:int):
    """
    x: the input tensor
    i: the target dimension i
    """
    x = x.transpose(i,-1)
    max_val = x.max(dim = -1,keepdim = True).values
    exp_x = torch.exp(x-max_val)
    sum = exp_x.sum(dim = -1,keepdim = True)
    x = exp_x/sum
    x = x.transpose(-1,i)
    return x

def Scaled_dot_product_attention(Q,K,V,mask = None):
    d_k = Q.shape[-1]
    temp = einsum(Q,K,'... q d_k, ... p d_k -> ... q p')/torch.sqrt(torch.tensor(d_k))
    temp = temp.masked_fill(mask == False, float('-inf'))
    res = softmax(temp,-1)#find the most likely key, so the last dimension
    return einsum(res,V,'... q p, ... p v -> ... q v')


