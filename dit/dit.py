# dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLNZero(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super(AdaLNZero, self).__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, 3 * embed_dim)
        )

    def forward(self, x, cond):
        x = self.norm(x)

        cond = self.mlp(cond)
        scale, shift, gate = cond.chunk(3, dim=-1)

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)

        x = x * (1 + scale) + shift
        
        return x, gate
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.net = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x, _ = self.net(x, x, x)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_ratio=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout

        self.fc1 = nn.Linear(embed_dim, ffn_ratio * embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_ratio * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x
    

class DiTBlock(nn.Module):
    def __init__(self, embed_dim, cond_dim, num_heads=8, ffn_ratio=4, dropout=0.1):
        super(DiTBlock, self).__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout

        self.adaln1 = AdaLNZero(embed_dim, cond_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.adaln2 = AdaLNZero(embed_dim, cond_dim)
        self.ffn = FeedForward(embed_dim, ffn_ratio, dropout)
    
    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        h = self.mhsa(h)
        x = x + gate1 * h

        h, gate2 = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + gate2 * h

        return x
    

class DiT(nn.Module):
    def __init__(self, embed_dim, cond_dim, depth=6, num_heads=8, ffn_ratio=4, dropout=0.1):
        super(DiT, self).__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout

        self.layers = nn.ModuleList([
            DiTBlock(embed_dim, cond_dim, num_heads, ffn_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
            
        return x