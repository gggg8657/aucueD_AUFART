import torch
import torch.nn as nn
import torch.nn.functional as f

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class TDC(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads, batchsize):
        super(TDC, self).__init__()
        self.dropout = dropout
        self.batchsize = batchsize
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, batchsize=self.batchsize) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 28, nhid))
        self.cls_token = nn.Parameter(torch.randn(1, 1, nhid))
        self.Tdropout = nn.Dropout(0.0)

        self.transformer = Transformer(nhid, 6, nheads, 64, 256, 0.0)
        self.pool = 'cls'
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(nhid, noutput)
        nn.init.xavier_uniform_(self.mlp_head.weight, gain=0.01)
        
    def forward(self, x, y):

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.Tdropout(x)

        x = self.transformer(x, y)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)