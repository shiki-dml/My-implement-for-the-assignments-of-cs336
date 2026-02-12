import torch
from torch import nn
from einops import einsum
from cs336_basics.Dot_Product_Attention import Scaled_dot_product_attention
from cs336_basics.Rope_class import RoPe

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
    def forward(self, w_q, w_k, w_v, w_o, x, mask=None,token_positions = None):
        *batch_shape, seq_len, d_in = x.shape

        q = einsum(x,w_q,'... sequence_length d_in, d_k d_in -> ... sequence_length d_k')
        k = einsum(x,w_k,'... sequence_length d_in, d_k d_in -> ... sequence_length d_k')
        v = einsum(x,w_v,'... sequence_length d_in, d_v d_in -> ... sequence_length d_v')

        q = q.view(*batch_shape, seq_len, self.num_heads, self.head_dim)
        k = k.view(*batch_shape, seq_len, self.num_heads, self.head_dim)
        v = v.view(*batch_shape, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        ones = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.uint8)
        upper_tri = torch.triu(ones, diagonal=1)
        causal_mask = (upper_tri == 0)
        if mask is not None:
            causal_mask = causal_mask & mask.bool()

        att_out = Scaled_dot_product_attention(q, k, v, mask=causal_mask)#shape: ... d_k d_v

        att_out = att_out.transpose(-3, -2).contiguous()
        att_out = att_out.view(*batch_shape, seq_len, self.d_model)

        output = einsum(att_out,w_o,'... d_k d_v, d_model d_v -> ... d_k d_model')

        return output
        
