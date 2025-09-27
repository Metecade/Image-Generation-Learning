# visualize.py - Visualization utilities for LDM

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_ldm_samples(ldm, device, num_samples=16, latent_shape=(4, 8, 8), save_path=None, epoch=None):
    """可视化LDM生成的样本"""
    ldm.unet.eval()
    
    with torch.no_grad():
        # 生成样本
        samples = ldm.sample(num_samples, latent_shape)
        
        # 转换到[0, 1]范围
        samples = (samples + 1) / 2.0
        samples = torch.clamp(samples, 0, 1).cpu()
    
    # 创建图像网格
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    title = f'LDM Generated Samples'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                img = samples[idx]
                if img.shape[0] == 1:  # 灰度图
                    axes[i, j].imshow(img.squeeze(), cmap='gray')
                else:  # 彩色图
                    axes[i, j].imshow(img.permute(1, 2, 0))
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LDM samples saved to: {save_path}")
    
    plt.close()
    ldm.unet.train()

def visualize_ddim_comparison(ldm, device, latent_shape=(4, 8, 8), save_path=None):
    """比较不同DDIM步数的效果"""
    from latent_diffusion import DDIMSampler
    
    ldm.unet.eval()
    ddim_sampler = DDIMSampler(ldm)
    
    steps_list = [10, 20, 50, 200]
    num_samples = 4
    
    with torch.no_grad():
        all_samples = []
        for steps in steps_list:
            z_samples = ddim_sampler.sample(num_samples, latent_shape, ddim_steps=steps)
            samples = ldm.decode_from_latent(z_samples)
            samples = (samples + 1) / 2.0
            samples = torch.clamp(samples, 0, 1).cpu()
            all_samples.append(samples)
    
    # 可视化比较
    fig, axes = plt.subplots(len(steps_list), num_samples, figsize=(num_samples * 2, len(steps_list) * 2))
    fig.suptitle('DDIM Steps Comparison for LDM', fontsize=16)
    
    for row, (steps, samples) in enumerate(zip(steps_list, all_samples)):
        for col in range(num_samples):
            img = samples[col]
            if img.shape[0] == 1:
                axes[row, col].imshow(img.squeeze(), cmap='gray')
            else:
                axes[row, col].imshow(img.permute(1, 2, 0))
            
            if col == 0:
                axes[row, col].set_ylabel(f'{steps} Steps', fontsize=12, rotation=90, va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDIM comparison saved to: {save_path}")
    
    plt.close()
    ldm.unet.train()

def visualize_reconstruction_quality(ldm, dataloader, device, save_path=None):
    """可视化VAE重建质量"""
    ldm.vae.eval()
    
    # 获取一批测试数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:8].to(device)  # 取8个样本
    
    with torch.no_grad():
        # VAE重建
        recon_data, _ = ldm.vae(real_data, sample_posterior=False)
        
        # 转换到显示范围
        real_display = (real_data + 1) / 2.0
        recon_display = (recon_data + 1) / 2.0
        
        real_display = torch.clamp(real_display, 0, 1).cpu()
        recon_display = torch.clamp(recon_display, 0, 1).cpu()
    
    # 可视化对比
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('VAE Reconstruction Quality', fontsize=16)
    
    for i in range(8):
        # 原图
        img = real_display[i]
        if img.shape[0] == 1:
            axes[0, i].imshow(img.squeeze(), cmap='gray')
        else:
            axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # 重建图
        img = recon_display[i]
        if img.shape[0] == 1:
            axes[1, i].imshow(img.squeeze(), cmap='gray')
        else:
            axes[1, i].imshow(img.permute(1, 2, 0))
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction comparison saved to: {save_path}")
    
    plt.close()
    ldm.vae.train()

def visualize_latent_interpolation(ldm, dataloader, device, save_path=None):
    """可视化潜在空间插值"""
    ldm.eval()
    
    # 获取两个样本
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    
    if len(real_data) < 2:
        print("Not enough samples for interpolation")
        return
    
    img1 = real_data[0:1].to(device)
    img2 = real_data[1:2].to(device)
    
    with torch.no_grad():
        # 生成插值
        interpolated = ldm.interpolate_images(img1, img2, num_steps=10)
        
        # 转换显示范围
        interpolated = (interpolated + 1) / 2.0
        interpolated = torch.clamp(interpolated, 0, 1).cpu()
    
    # 可视化插值结果
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    fig.suptitle('Latent Space Interpolation', fontsize=16)
    
    for i in range(10):
        img = interpolated[i]
        if img.shape[0] == 1:
            axes[i].imshow(img.squeeze(), cmap='gray')
        else:
            axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Interpolation saved to: {save_path}")
    
    plt.close()
    ldm.train()

def visualize_denoising_process(ldm, device, latent_shape=(4, 8, 8), save_path=None):
    """可视化去噪过程"""
    ldm.unet.eval()
    
    with torch.no_grad():
        # 从纯噪声开始
        z = torch.randn(1, *latent_shape, device=device)
        
        # 选择要显示的时间步
        timesteps_to_show = [999, 800, 600, 400, 200, 100, 50, 0]
        images = []
        
        # 完整的去噪过程，但只保存指定时间步的结果
        for i in reversed(range(ldm.timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            z = ldm.p_sample(z, t)
            
            if i in timesteps_to_show:
                # 解码到图像空间
                img = ldm.decode_from_latent(z)
                img = (img + 1) / 2.0
                img = torch.clamp(img, 0, 1).cpu()
                images.append(img.squeeze())
    
    # 可视化去噪过程
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(len(timesteps_to_show) * 2, 2))
    fig.suptitle('LDM Denoising Process', fontsize=16)
    
    for i, (ax, img, timestep) in enumerate(zip(axes, images, timesteps_to_show)):
        if img.dim() == 2:  # 灰度图
            ax.imshow(img, cmap='gray')
        else:  # 彩色图
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f't={timestep}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Denoising process saved to: {save_path}")
    
    plt.close()
    ldm.unet.train()

def plot_ldm_training_curves(train_losses, val_losses, save_path="results/visualizations/ldm_training_curves.png"):
    """绘制LDM训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LDM Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习曲线（平滑版）
    plt.subplot(1, 2, 2)
    if len(train_losses) > 10:
        # 使用移动平均平滑曲线
        window = max(len(train_losses) // 10, 1)
        train_smooth = []
        val_smooth = []
        for i in range(len(train_losses)):
            start = max(0, i - window + 1)
            train_smooth.append(np.mean(train_losses[start:i+1]))
            val_smooth.append(np.mean(val_losses[start:i+1]))
        
        plt.plot(epochs, train_smooth, 'b-', label='Training Loss (Smoothed)', linewidth=2)
        plt.plot(epochs, val_smooth, 'r-', label='Validation Loss (Smoothed)', linewidth=2)
    else:
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LDM Smoothed Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"LDM training curves saved to: {save_path}")
    plt.close()

def create_ldm_comparison_grid(ldm, device, save_path=None):
    """创建LDM功能对比网格"""
    from latent_diffusion import DDIMSampler
    
    ldm.unet.eval()
    ddim_sampler = DDIMSampler(ldm)
    latent_shape = (4, 8, 8)
    
    with torch.no_grad():
        # 生成不同方法的样本
        samples_dict = {
            'DDPM (1000 steps)': ldm.sample(4, latent_shape),
            'DDIM (50 steps)': ldm.decode_from_latent(ddim_sampler.sample(4, latent_shape, ddim_steps=50)),
            'DDIM (20 steps)': ldm.decode_from_latent(ddim_sampler.sample(4, latent_shape, ddim_steps=20)),
            'DDIM (10 steps)': ldm.decode_from_latent(ddim_sampler.sample(4, latent_shape, ddim_steps=10)),
        }
        
        # 转换显示范围
        for key in samples_dict:
            samples_dict[key] = (samples_dict[key] + 1) / 2.0
            samples_dict[key] = torch.clamp(samples_dict[key], 0, 1).cpu()
    
    # 创建对比网格
    fig, axes = plt.subplots(len(samples_dict), 4, figsize=(8, len(samples_dict) * 2))
    fig.suptitle('LDM Sampling Methods Comparison', fontsize=16)
    
    for row, (method, samples) in enumerate(samples_dict.items()):
        for col in range(4):
            img = samples[col]
            if img.shape[0] == 1:
                axes[row, col].imshow(img.squeeze(), cmap='gray')
            else:
                axes[row, col].imshow(img.permute(1, 2, 0))
            
            if col == 0:
                axes[row, col].set_ylabel(method, fontsize=10, rotation=90, va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"LDM comparison grid saved to: {save_path}")
    
    plt.close()
    ldm.unet.train()