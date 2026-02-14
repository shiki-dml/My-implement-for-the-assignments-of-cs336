import torch
from torch import nn
from cs336_basics.Embedding_class import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.RMSnorm_class import RMSnorm
from cs336_basics.Linear_class import Linear
from cs336_basics.Dot_Product_Attention import softmax

class TransfromerLM(nn.Module):
    def __init__(self,vocab_size:int,context_length:int,num_layers:int,d_model,num_heads,d_ff,theta=None,device = None,dtype = None):
        """
        vocab_size:int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix
        context_length:int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        num_layers:int The number of Transformer blocks to use
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size,d_model)
        self.TransformerLayers = nn.Sequential(*[TransformerBlock(d_model,num_heads,d_ff,theta,context_length,device) for i in range(num_layers)])
        self.norm = RMSnorm(d_model)
        self.Linear = Linear(d_model,self.vocab_size)
    def forward(self,x):
        out = self.embedding(x)
        for layer in self.TransformerLayers:
            out = layer(out)
        out = self.norm(out)
        out = self.Linear(out)
        #out = softmax(out,-1)
        return out