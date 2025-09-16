# ddimnet.py

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
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
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

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Reshape for attention
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        v = v.reshape(batch, channels, height * width).permute(0, 2, 1)
        
        # Compute attention
        attn = torch.bmm(q, k) * (channels ** (-0.5))
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(batch, channels, height, width)
        h = self.proj_out(h)
        
        return x + h

class UNet(nn.Module):
    """DDIM的U-Net架构 - 适配28x28图像"""
    def __init__(self, in_channels=1, model_channels=64, num_res_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
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
        
        # 编码器路径 (28x28 -> 14x14 -> 7x7)
        # Level 0: 28x28, channels: 64
        self.down_res_0 = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_embed_dim) 
            for _ in range(num_res_blocks)
        ])
        self.down_sample_0 = nn.Conv2d(model_channels, model_channels*2, 3, stride=2, padding=1)
        
        # Level 1: 14x14, channels: 128  
        self.down_res_1 = nn.ModuleList([
            ResidualBlock(model_channels*2, model_channels*2, time_embed_dim) 
            for _ in range(num_res_blocks)
        ])
        self.down_sample_1 = nn.Conv2d(model_channels*2, model_channels*4, 3, stride=2, padding=1)
        
        # Level 2: 7x7, channels: 256
        self.down_res_2 = nn.ModuleList([
            ResidualBlock(model_channels*4, model_channels*4, time_embed_dim) 
            for _ in range(num_res_blocks)
        ])
        
        # 中间层
        self.mid_block1 = ResidualBlock(model_channels*4, model_channels*4, time_embed_dim)
        self.mid_attn = AttentionBlock(model_channels*4)
        self.mid_block2 = ResidualBlock(model_channels*4, model_channels*4, time_embed_dim)
        
        # 解码器路径 (7x7 -> 14x14 -> 28x28)
        # Level 2: 7x7, channels: 256 -> 256
        self.up_res_2 = nn.ModuleList([
            ResidualBlock(model_channels*4 + model_channels*4, model_channels*4, time_embed_dim),
            ResidualBlock(model_channels*4, model_channels*4, time_embed_dim)
        ])
        self.up_sample_2 = nn.ConvTranspose2d(model_channels*4, model_channels*2, 4, stride=2, padding=1)
        
        # Level 1: 14x14, channels: 128 -> 128
        self.up_res_1 = nn.ModuleList([
            ResidualBlock(model_channels*2 + model_channels*2, model_channels*2, time_embed_dim),
            ResidualBlock(model_channels*2, model_channels*2, time_embed_dim)
        ])
        self.up_sample_1 = nn.ConvTranspose2d(model_channels*2, model_channels, 4, stride=2, padding=1)
        
        # Level 0: 28x28, channels: 64 -> 64
        self.up_res_0 = nn.ModuleList([
            ResidualBlock(model_channels + model_channels, model_channels, time_embed_dim),
            ResidualBlock(model_channels, model_channels, time_embed_dim)
        ])
        
        # Output layers
        self.out_norm = nn.GroupNorm(8, model_channels)
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
    
    def forward(self, x, timesteps):
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Initial conv
        h = self.input_conv(x)  # [B, 64, 28, 28]
        
        # 编码器
        # Level 0
        skip_0 = h
        for res_block in self.down_res_0:
            h = res_block(h, t_emb)
        h = self.down_sample_0(h)  # [B, 128, 14, 14]
        
        # Level 1
        skip_1 = h
        for res_block in self.down_res_1:
            h = res_block(h, t_emb)
        h = self.down_sample_1(h)  # [B, 256, 7, 7]
        
        # Level 2
        skip_2 = h
        for res_block in self.down_res_2:
            h = res_block(h, t_emb)
        
        # 中间层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 解码器
        # Level 2
        h = torch.cat([h, skip_2], dim=1)  # [B, 512, 7, 7]
        h = self.up_res_2[0](h, t_emb)
        h = self.up_res_2[1](h, t_emb)
        h = self.up_sample_2(h)  # [B, 128, 14, 14]
        
        # Level 1  
        h = torch.cat([h, skip_1], dim=1)  # [B, 256, 14, 14]
        h = self.up_res_1[0](h, t_emb)
        h = self.up_res_1[1](h, t_emb)
        h = self.up_sample_1(h)  # [B, 64, 28, 28]
        
        # Level 0
        h = torch.cat([h, skip_0], dim=1)  # [B, 128, 28, 28]
        h = self.up_res_0[0](h, t_emb)
        h = self.up_res_0[1](h, t_emb)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)  # [B, 1, 28, 28]
        
        return h

class DDIM:
    """DDIM扩散模型 - 支持确定性采样和可变推理步数"""
    def __init__(self, model, beta_start=1e-4, beta_end=2e-2, timesteps=1000, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 计算beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算用于去噪的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """前向扩散过程：在x_0上添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def ddim_sample_step(self, x_t, t, t_prev, eta=0.0):
        """DDIM采样步骤 - 支持确定性和随机采样"""
        with torch.no_grad():
            # 预测噪声
            predicted_noise = self.model(x_t, t)
            
            # 获取alpha值 - 更安全的索引处理
            alpha_t = self.alphas_cumprod[t][:, None, None, None]
            
            # 处理t_prev的边界情况
            if len(t_prev) == 0 or t_prev[0] < 0:
                alpha_t_prev = torch.ones_like(alpha_t)
            else:
                alpha_t_prev = self.alphas_cumprod[t_prev][:, None, None, None]
            
            # 计算预测的x_0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            pred_x0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            
            # 计算方向
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)
            
            # DDIM更新公式
            pred_dir = sqrt_one_minus_alpha_t_prev * predicted_noise
            
            # 添加随机性（eta=0表示确定性采样）
            if eta > 0 and len(t_prev) > 0 and t_prev[0] >= 0:
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(x_t)
                pred_dir = pred_dir + sigma * noise
            
            x_t_prev = sqrt_alpha_t_prev * pred_x0 + pred_dir
            
            return x_t_prev
    
    def ddim_sample(self, batch_size, img_size=28, ddim_steps=50, eta=0.0):
        """DDIM采样 - 支持更少的推理步数"""
        self.model.eval()
        
        # 创建采样时间表
        skip = self.timesteps // ddim_steps
        seq = range(0, self.timesteps, skip)
        seq = list(seq)
        seq_next = [-1] + list(seq[:-1])
        
        # 从纯噪声开始
        img = torch.randn(batch_size, 1, img_size, img_size, device=self.device)
        
        # 逐步去噪
        for i, (t, t_prev) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_prev_tensor = torch.full((batch_size,), t_prev, device=self.device, dtype=torch.long)
            
            img = self.ddim_sample_step(img, t_tensor, t_prev_tensor, eta=eta)
        
        return img
    
    def sample(self, batch_size, img_size=28, ddim_steps=50, eta=0.0):
        """生成图像的主要接口"""
        return self.ddim_sample(batch_size, img_size, ddim_steps, eta)
    
    def compute_loss(self, x_0):
        """计算训练损失（与DDPM相同）"""
        batch_size = x_0.shape[0]
        
        # 随机选择时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # 添加噪声
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def interpolate(self, x1, x2, t, num_steps=10):
        """在潜在空间中插值（简化版）"""
        self.model.eval()
        
        with torch.no_grad():
            # 简单的线性插值方法
            interpolated = []
            
            for alpha in torch.linspace(0, 1, num_steps):
                # 在原始图像空间中插值
                x_interp = alpha * x1 + (1 - alpha) * x2
                interpolated.append(x_interp)
            
            return torch.cat(interpolated, dim=0)