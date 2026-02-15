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

from collections.abc import Callable,Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3):
        if lr<0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad == None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss

weights = torch.nn.Parameter(5*torch.randn(10,10))
opt = SGD([weights],lr = 1e3)

for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,weight_decay,betas = [0.9,0.999],eps = 10e-8,):
        if lr<0:
            raise ValueError("Invalid learning rate {lr}")
        m = 0
        v = 0 
        defaults = {"lr":lr,"m":m,"v":v,"beta_1":betas[0],"beta_2":betas[1],"weight_decay":weight_decay,"eps":eps}
        super().__init__(params,defaults)
    def step(self,closure:Optional[Callable] = None):
        lr = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            m = group["m"]
            v = group["v"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            lamda = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad == None:
                    continue
                state = self.state[p]
                t = state.get("t",1)
                grad = p.grad.data
                m = beta_1*m+(1-beta_1)*grad
                v = beta_2*v+(1-beta_2)*grad*grad
                group["m"] = m
                group["v"] = v
                lr_t = lr*math.sqrt(1-beta_2**t)/(1-beta_1**t)
                p.data -= lr_t*m/(eps+torch.sqrt(v))
                p.data -= lr*lamda*p.data
                state["t"] = t+1
        return loss

def cosine_annealing_schedule(t,alpha_max,alpha_min,t_w,t_c):
    if t<t_w:
        return t*alpha_max/t_w
    elif t>t_c:
        return alpha_min
    else:
        return alpha_min+(1+math.cos(math.pi*(t-t_w)/(t_c-t_w)))*(alpha_max-alpha_min)/2