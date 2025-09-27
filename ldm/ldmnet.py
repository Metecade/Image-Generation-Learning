# ldmnet.py - Latent Diffusion Model Network

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TimeEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块，包含时间嵌入"""
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_conv(x)

class SpatialTransformer(nn.Module):
    """空间Transformer块（用于cross-attention）"""
    def __init__(self, channels, n_heads=8, d_head=64, context_dim=None):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.d_head = d_head
        
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, n_heads * d_head, 1)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(n_heads * d_head, n_heads, d_head, context_dim=context_dim)
        ])
        
        self.proj_out = nn.Conv2d(n_heads * d_head, channels, 1)
    
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Reshape for transformer
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, -1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        # Reshape back and project out
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        
        return x + x_in

class BasicTransformerBlock(nn.Module):
    """基础Transformer块"""
    def __init__(self, dim, n_heads, d_head, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class CrossAttention(nn.Module):
    """交叉注意力层"""
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape to (batch, seq_len, heads, dim_per_head) then transpose to (batch, heads, seq_len, dim_per_head)
        q, k, v = map(lambda t: t.view(*t.shape[:-1], h, -1).transpose(-2, -3), (q, k, v))
        
        # Use 4D einsum for multi-head attention
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # Transpose back and reshape: (batch, heads, seq_len, dim_per_head) -> (batch, seq_len, heads * dim_per_head)
        out = out.transpose(-2, -3).contiguous().view(*out.shape[:-3], out.shape[-2], -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class LatentUNet(nn.Module):
    """用于Latent Diffusion的U-Net - 适配潜在空间"""
    def __init__(self, in_channels=4, model_channels=320, num_res_blocks=2, 
                 attention_resolutions=[4, 2, 1], channel_mult=[1, 2, 4, 4],
                 use_spatial_transformer=True, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.use_spatial_transformer = use_spatial_transformer
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 计算各层的通道数
        ch_mult = channel_mult
        self.num_resolutions = len(ch_mult)
        
        # 编码器路径
        self.down = nn.ModuleList()
        ch = model_channels
        input_ch = ch
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(SpatialTransformer(ch, context_dim=context_dim))
                    else:
                        layers.append(AttentionBlock(ch))
                self.down.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(ch_mult) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2
        
        # 中间层
        self.middle = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim),
            SpatialTransformer(ch, context_dim=context_dim) if use_spatial_transformer else AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim)
        )
        
        # 解码器路径
        self.up = nn.ModuleList()
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch + input_block_chans.pop(), mult * model_channels, time_embed_dim)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(SpatialTransformer(ch, context_dim=context_dim))
                    else:
                        layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.up.append(nn.Sequential(*layers))
        
        # Output layers
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
    
    def forward(self, x, timesteps, context=None):
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Store skip connections
        hs = [h]
        
        # Down sampling
        for module in self.down:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, context)
            else:
                h = layer(h)
        
        # Up sampling
        for module in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h

class AttentionBlock(nn.Module):
    """简单的自注意力块（不使用spatial transformer时的备用）"""
    def __init__(self, channels, heads=8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        h_input = self.norm(x)
        qkv = self.qkv(h_input)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(b, self.heads, c // self.heads, h * w).transpose(-2, -1)
        k = k.view(b, self.heads, c // self.heads, h * w)
        v = v.view(b, self.heads, c // self.heads, h * w).transpose(-2, -1)
        
        # Attention
        scale = (c // self.heads) ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out