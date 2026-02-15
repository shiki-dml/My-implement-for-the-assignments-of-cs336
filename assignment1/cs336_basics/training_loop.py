import torch
import numpy.typing as npt
from torch import nn
import os
from typing import BinaryIO,IO

def data_loading(x: npt.NDArray,batch_size:int,context_length:int,device)->tuple[torch.tensor,torch.tensor]:
    res = torch.from_numpy(x).to(device=device)
    x_long = res.long()
    length = len(res)
    idx_1 = torch.randint(0,x_long[0:length-context_length].numel(),(batch_size,)).long().to(x_long.device)
    offset = torch.arange(context_length,device = x_long.device)
    gather_idx_1 = idx_1[:,None]+offset[None,:]
    out_1 = x_long[gather_idx_1]
    idx_2 = idx_1+1
    gather_idx_2 = idx_2[:,None]+offset[None,:]
    out_2 = x_long[gather_idx_2]
    return (out_1,out_2)
def save_checkpoint(model:torch.nn.Module,optimizer:torch.optim.Optimizer,iteration:int,out:str|os.PathLike|BinaryIO|IO[bytes]):
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()
    obj = {"model_state":model_state,"optim_state":optim_state,"itr":iteration}
    torch.save(obj,out)
def load_checkpoint(src:str|os.PathLike|BinaryIO|IO[bytes],model:torch.nn.Module,optimizer:torch.optim.Optimizer):
    state = torch.load(src)
    #Note that the state is like {"model_state":..,"itr":...,"optim_state":...},
    #where "itr" comes from the function save_checkpoint above
    model.load_state_dict(state["model_state"]) 
    optimizer.load_state_dict(state['optim_state'])
    return state["itr"]
