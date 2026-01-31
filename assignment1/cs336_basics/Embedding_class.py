import torch
from torch import nn
from einops import einsum
class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device = None,dtype =None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimensionality of each embedding vector i.e. d_model
        device: torch.device|None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        std = 1
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight,std =std,a = -3,b =3)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        ids =token_ids.clone().detach().long()
        return self.weight[ids]