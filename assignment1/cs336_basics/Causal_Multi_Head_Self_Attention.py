import torch
from torch import nn
from einops import einsum
from cs336_basics.Dot_Product_Attention import Scaled_dot_product_attention
from cs336_basics.Rope_class import RoPe
from cs336_basics.Linear_class import Linear
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,theta = None,max_seq_len = None,device = None):
        """
        d_model: Dimensionality of the Transformer block inputs
        num_heads: Number of heads to uses in multi-head self-attention
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if theta is not None:
            self.rope = RoPe(theta,self.head_dim,max_seq_len,device=device)
        self.q = Linear(d_model,d_model)
        self.k = Linear(d_model,d_model)
        self.v = Linear(d_model,d_model)
        self.o = Linear(d_model,d_model)
    def forward(self, x, mask=None,token_positions = None):
        *batch_shape, seq_len, d_in = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(*batch_shape, seq_len, self.num_heads, self.head_dim)
        k = k.view(*batch_shape, seq_len, self.num_heads, self.head_dim)
        v = v.view(*batch_shape, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        if hasattr(self, "rope") and self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        ones = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.uint8)
        upper_tri = torch.triu(ones, diagonal=1)
        causal_mask = (upper_tri == 0)
        if mask is not None:
            causal_mask = causal_mask & mask.bool()
        print("q", q.shape, "k", k.shape, "v", v.shape)
        att_out = Scaled_dot_product_attention(q, k, v, mask=causal_mask)#shape: ... d_k d_v

        att_out = att_out.transpose(-3, -2).contiguous()
        att_out = att_out.view(*batch_shape, seq_len, self.d_model)

        output = self.o(att_out)

        return output
        
