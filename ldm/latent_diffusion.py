# latent_diffusion.py - Main Latent Diffusion Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vae import VQModel
from ldmnet import LatentUNet

class LatentDiffusion(nn.Module):
    """Latent Diffusion Model - 在潜在空间进行扩散"""
    
    def __init__(self, 
                 unet_config,
                 vae=None,  # 接受预训练的VAE
                 vae_config=None,  # 或者VAE配置来创建新的VAE
                 beta_start=1e-4, 
                 beta_end=2e-2, 
                 timesteps=1000,
                 scale_factor=1.0,
                 device='cpu'):
        super().__init__()
        
        # 初始化VAE
        if vae is not None:
            self.vae = vae
        elif vae_config is not None:
            self.vae = VQModel(vae_config, embed_dim=4)
        else:
            raise ValueError("Either vae or vae_config must be provided")
        
        self.vae.eval()  # VAE通常预训练并冻结
        
        # 初始化扩散U-Net
        self.unet = LatentUNet(**unet_config)
        
        # 扩散参数
        self.timesteps = timesteps
        self.device = device
        self.scale_factor = scale_factor
        
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
    
    def encode_to_latent(self, x):
        """将图像编码到潜在空间"""
        with torch.no_grad():
            posterior = self.vae.encode(x)
            z = posterior.mode() * self.scale_factor  # 使用mode而不是sample
            return z
    
    def decode_from_latent(self, z):
        """从潜在空间解码到图像"""
        with torch.no_grad():
            z = z / self.scale_factor
            x_recon = self.vae.decode(z)
            return x_recon
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：在潜在表示上添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, z_t, t, context=None):
        """逆向去噪过程：从z_t预测z_{t-1}"""
        with torch.no_grad():
            # 预测噪声
            predicted_noise = self.unet(z_t, t, context=context)
            
            # 计算系数
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
            betas_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            
            # 计算均值
            mean = sqrt_recip_alphas_t * (z_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if t[0] == 0:
                return mean
            else:
                # 添加噪声
                posterior_variance_t = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(z_t)
                return mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample_latent(self, batch_size, latent_shape=(4, 8, 8), context=None):
        """在潜在空间中采样"""
        self.unet.eval()
        
        # 从纯噪声开始
        z = torch.randn(batch_size, *latent_shape, device=self.device)
        
        # 逐步去噪
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            z = self.p_sample(z, t, context=context)
        
        return z
    
    def sample(self, batch_size, latent_shape=(4, 8, 8), context=None):
        """生成图像的主要接口"""
        # 在潜在空间采样
        z = self.sample_latent(batch_size, latent_shape, context)
        
        # 解码到图像空间
        images = self.decode_from_latent(z)
        
        return images
    
    def compute_loss(self, x_0, context=None):
        """计算训练损失"""
        batch_size = x_0.shape[0]
        
        # 编码到潜在空间
        z_0 = self.encode_to_latent(x_0)
        
        # 随机选择时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # 添加噪声
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        
        # 预测噪声
        predicted_noise = self.unet(z_t, t, context=context)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def forward(self, x, context=None):
        """前向传播"""
        return self.compute_loss(x, context)
    
    def get_latent_shape(self, image_size):
        """获取对应图像尺寸的潜在空间形状"""
        # 通常VAE的下采样率是8，所以潜在空间尺寸是图像尺寸的1/8
        latent_size = image_size // 8
        return (4, latent_size, latent_size)  # 4是VAE的潜在维度
    
    def interpolate_latent(self, z1, z2, num_steps=10):
        """在潜在空间中插值"""
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        interpolated_latents = []
        
        for alpha in alphas:
            z_interp = alpha * z1 + (1 - alpha) * z2
            interpolated_latents.append(z_interp)
        
        return torch.cat(interpolated_latents, dim=0)
    
    def interpolate_images(self, img1, img2, num_steps=10):
        """图像插值"""
        # 编码到潜在空间
        z1 = self.encode_to_latent(img1)
        z2 = self.encode_to_latent(img2)
        
        # 潜在空间插值
        z_interp = self.interpolate_latent(z1, z2, num_steps)
        
        # 解码回图像
        interpolated_images = self.decode_from_latent(z_interp)
        
        return interpolated_images


class DDIMSampler:
    """DDIM采样器用于LDM"""
    
    def __init__(self, ldm):
        self.ldm = ldm
        self.device = ldm.device
    
    def ddim_sample_step(self, z_t, t, t_prev, eta=0.0, context=None):
        """DDIM采样步骤"""
        with torch.no_grad():
            # 预测噪声
            predicted_noise = self.ldm.unet(z_t, t, context=context)
            
            # 获取alpha值
            alpha_t = self.ldm.alphas_cumprod[t][:, None, None, None]
            if len(t_prev) > 0 and t_prev[0] >= 0:
                alpha_t_prev = self.ldm.alphas_cumprod[t_prev][:, None, None, None]
            else:
                alpha_t_prev = torch.ones_like(alpha_t)
            
            # 计算预测的z_0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            pred_z0 = (z_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            
            # 计算方向
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)
            
            # DDIM更新公式
            pred_dir = sqrt_one_minus_alpha_t_prev * predicted_noise
            
            # 添加随机性（eta=0表示确定性采样）
            if eta > 0 and len(t_prev) > 0 and t_prev[0] >= 0:
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(z_t)
                pred_dir = pred_dir + sigma * noise
            
            z_t_prev = sqrt_alpha_t_prev * pred_z0 + pred_dir
            
            return z_t_prev
    
    def sample(self, batch_size, latent_shape, ddim_steps=50, eta=0.0, context=None):
        """DDIM快速采样"""
        self.ldm.unet.eval()
        
        # 创建采样时间表
        skip = self.ldm.timesteps // ddim_steps
        seq = range(0, self.ldm.timesteps, skip)
        seq = list(seq)
        seq_next = [-1] + list(seq[:-1])
        
        # 从纯噪声开始
        z = torch.randn(batch_size, *latent_shape, device=self.device)
        
        # 逐步去噪
        for i, (t, t_prev) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            t_prev_tensor = torch.full((batch_size,), t_prev, device=self.device, dtype=torch.long)
            
            z = self.ddim_sample_step(z, t_tensor, t_prev_tensor, eta=eta, context=context)
        
        return z