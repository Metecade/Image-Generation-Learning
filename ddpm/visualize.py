# visualize.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体或默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def visualize_ddpm_samples(ddpm, device, num_samples=16, save_path=None, epoch=None):
    """
    可视化DDPM生成的样本
    """
    ddpm.model.eval()
    
    with torch.no_grad():
        # 生成样本
        samples = ddpm.sample(num_samples, img_size=28)
        
        # 将数据从[-1, 1]转换到[0, 1]
        samples = (samples + 1) / 2.0
        samples = samples.cpu()
    
    # 创建图像网格
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle(f'DDPM Generated Samples' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(samples[idx].squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDPM samples saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()


def visualize_denoising_process(ddpm, device, save_path=None, epoch=None, steps_to_show=8):
    """
    可视化DDPM的去噪过程
    """
    ddpm.model.eval()
    
    with torch.no_grad():
        # 从纯噪声开始
        img = torch.randn(1, 1, 28, 28, device=device)
        images = []
        
        # 选择要显示的时间步
        timesteps_to_show = np.linspace(ddpm.timesteps-1, 0, steps_to_show, dtype=int)
        current_step = 0
        
        # 逐步去噪并保存关键步骤
        for i in reversed(range(ddpm.timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = ddpm.p_sample(img, t)
            
            if i in timesteps_to_show:
                # 转换到[0, 1]范围并保存
                img_to_save = (img + 1) / 2.0
                images.append(img_to_save.cpu().squeeze().numpy())
                current_step += 1
    
    # 创建去噪过程可视化
    fig, axes = plt.subplots(1, steps_to_show, figsize=(steps_to_show * 2, 2))
    fig.suptitle(f'DDPM Denoising Process' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Step {timesteps_to_show[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDPM denoising process saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()


def compare_original_noisy_denoised(ddpm, dataloader, device, save_path=None, epoch=None):
    """
    对比原图、加噪图像和去噪结果（优化版本，不做完整去噪）
    """
    ddpm.model.eval()
    
    # 获取一批真实数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:6].to(device)  # 取6个样本
    
    # 标准化到[-1, 1]
    real_data_norm = real_data * 2.0 - 1.0
    
    with torch.no_grad():
        # 选择几个不同的噪声时间步
        t_values = [200, 500, 800]  # 轻度、中度、重度噪声
        
        fig, axes = plt.subplots(len(t_values), 6 * 3, figsize=(18, len(t_values) * 3))
        fig.suptitle(f'Original vs Noisy vs Predicted' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
        
        for row, t_val in enumerate(t_values):
            t = torch.full((6,), t_val, device=device, dtype=torch.long)
            
            # 加噪
            noise = torch.randn_like(real_data_norm)
            noisy_data = ddpm.q_sample(real_data_norm, t, noise)
            
            # 预测噪声并估算去噪结果（一步去噪近似）
            predicted_noise = ddpm.model(noisy_data, t)
            alpha_t = ddpm.alphas_cumprod[t_val]
            beta_t = 1 - alpha_t
            predicted_clean = (noisy_data - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # 显示对比结果
            for i in range(6):
                col_base = i * 3
                
                # 原图
                original_img = real_data[i].squeeze().cpu()
                axes[row, col_base].imshow(original_img, cmap='gray')
                axes[row, col_base].set_title(f'Original {i+1}' if row == 0 else '')
                axes[row, col_base].axis('off')
                if i == 0:
                    axes[row, col_base].set_ylabel(f'Noise step: {t_val}', fontsize=12, rotation=90, va='center')
                
                # 加噪图像
                noisy_img = (noisy_data[i] + 1) / 2.0
                axes[row, col_base + 1].imshow(noisy_img.squeeze().cpu(), cmap='gray')
                axes[row, col_base + 1].set_title(f'Noisy {i+1}' if row == 0 else '')
                axes[row, col_base + 1].axis('off')
                
                # 预测的干净图像
                predicted_img = torch.clamp((predicted_clean[i] + 1) / 2.0, 0, 1)
                axes[row, col_base + 2].imshow(predicted_img.squeeze().cpu(), cmap='gray')
                axes[row, col_base + 2].set_title(f'Predicted {i+1}' if row == 0 else '')
                axes[row, col_base + 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()
    
    plt.close()
    ddpm.model.train()


def visualize_reconstruction_quality(ddpm, dataloader, device, save_path=None, epoch=None):
    """
    可视化重建质量 - 使用一步预测近似（优化版本）
    """
    ddpm.model.eval()
    
    # 获取一批真实数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:4].to(device)  # 取4个样本
    
    # 标准化到[-1, 1]
    real_data_norm = real_data * 2.0 - 1.0
    
    with torch.no_grad():
        # 测试不同程度的噪声重建能力
        noise_levels = [0, 100, 300, 500, 700]  # 减少噪声级别数量
        
        fig, axes = plt.subplots(4, len(noise_levels), figsize=(len(noise_levels) * 2, 8))
        fig.suptitle(f'Reconstruction Quality Test' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
        
        for col, noise_level in enumerate(noise_levels):
            axes[0, col].set_title(f'Noise: {noise_level}' if noise_level > 0 else 'Original', fontsize=10)
            
            for row in range(4):
                if noise_level == 0:
                    # 显示原图
                    img_to_show = real_data[row].squeeze().cpu()
                else:
                    # 加噪并使用一步预测
                    t = torch.full((1,), noise_level, device=device, dtype=torch.long)
                    noise = torch.randn_like(real_data_norm[row:row+1])
                    noisy_img = ddpm.q_sample(real_data_norm[row:row+1], t, noise)
                    
                    # 一步预测去噪
                    predicted_noise = ddpm.model(noisy_img, t)
                    alpha_t = ddpm.alphas_cumprod[noise_level]
                    beta_t = 1 - alpha_t
                    predicted_clean = (noisy_img - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(alpha_t)
                    
                    img_to_show = torch.clamp((predicted_clean + 1) / 2.0, 0, 1)
                    img_to_show = img_to_show.squeeze().cpu()
                
                axes[row, col].imshow(img_to_show, cmap='gray')
                axes[row, col].axis('off')
                if col == 0:
                    axes[row, col].set_ylabel(f'Sample {row+1}', fontsize=10, rotation=90, va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction quality test saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()
    
    plt.close()
    ddpm.model.train()


def visualize_full_denoising_comparison(ddpm, dataloader, device, save_path=None, epoch=None, max_steps=100):
    """
    完整去噪对比（慢速版本，仅在需要时使用）
    """
    print(f"Warning: This function performs full denoising up to {max_steps} steps. It will be slow!")
    ddpm.model.eval()
    
    # 获取一批真实数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:3].to(device)  # 只取3个样本
    
    # 标准化到[-1, 1]
    real_data_norm = real_data * 2.0 - 1.0
    
    with torch.no_grad():
        # 选择较小的噪声时间步
        t_values = [50, max_steps]  # 只测试两个级别
        
        fig, axes = plt.subplots(len(t_values), 3 * 3, figsize=(9, len(t_values) * 3))
        fig.suptitle(f'Full Denoising Comparison' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
        
        for row, t_val in enumerate(t_values):
            t = torch.full((3,), t_val, device=device, dtype=torch.long)
            
            # 加噪
            noise = torch.randn_like(real_data_norm)
            noisy_data = ddpm.q_sample(real_data_norm, t, noise)
            
            # 完整去噪过程
            denoised_data = noisy_data.clone()
            for timestep in reversed(range(t_val + 1)):
                t_tensor = torch.full((3,), timestep, device=device, dtype=torch.long)
                denoised_data = ddpm.p_sample(denoised_data, t_tensor)
            
            # 显示对比结果
            for i in range(3):
                col_base = i * 3
                
                # 原图
                original_img = real_data[i].squeeze().cpu()
                axes[row, col_base].imshow(original_img, cmap='gray')
                axes[row, col_base].set_title(f'Original {i+1}' if row == 0 else '')
                axes[row, col_base].axis('off')
                
                # 加噪图像
                noisy_img = (noisy_data[i] + 1) / 2.0
                axes[row, col_base + 1].imshow(noisy_img.squeeze().cpu(), cmap='gray')
                axes[row, col_base + 1].set_title(f'Noisy {i+1}' if row == 0 else '')
                axes[row, col_base + 1].axis('off')
                
                # 完整去噪
                denoised_img = (denoised_data[i] + 1) / 2.0
                axes[row, col_base + 2].imshow(denoised_img.squeeze().cpu(), cmap='gray')
                axes[row, col_base + 2].set_title(f'Full Denoised {i+1}' if row == 0 else '')
                axes[row, col_base + 2].axis('off')
                
                if i == 0:
                    axes[row, col_base].set_ylabel(f'Steps: {t_val}', fontsize=12, rotation=90, va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Full denoising comparison saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()


def plot_ddpm_training_curves(train_losses, val_losses, save_path=None):
    """
    绘制DDPM训练曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 训练和验证损失
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax.set_title('DDPM Training Progress', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加最小值标记
    min_train_idx = np.argmin(train_losses)
    min_val_idx = np.argmin(val_losses)
    ax.plot(min_train_idx + 1, train_losses[min_train_idx], 'bo', markersize=8)
    ax.plot(min_val_idx + 1, val_losses[min_val_idx], 'ro', markersize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDPM training curves saved to: {save_path}")
    
    plt.close()


def generate_interpolation(ddpm, device, num_steps=10, save_path=None):
    """
    生成插值动画（在潜在空间中）
    """
    ddpm.model.eval()
    
    with torch.no_grad():
        # 生成两个随机噪声作为起点和终点
        noise1 = torch.randn(1, 1, 28, 28, device=device)
        noise2 = torch.randn(1, 1, 28, 28, device=device)
        
        interpolated_images = []
        
        for i in range(num_steps):
            # 线性插值
            alpha = i / (num_steps - 1)
            interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
            
            # 完整的去噪过程
            img = interpolated_noise.clone()
            for t_idx in reversed(range(ddpm.timesteps)):
                t = torch.full((1,), t_idx, device=device, dtype=torch.long)
                img = ddpm.p_sample(img, t)
            
            # 转换到[0, 1]范围
            img = (img + 1) / 2.0
            interpolated_images.append(img.cpu().squeeze().numpy())
    
    # 可视化插值结果
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    fig.suptitle('DDPM Interpolation', fontsize=16)
    
    for i, (ax, img) in enumerate(zip(axes, interpolated_images)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Step {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDPM interpolation saved to: {save_path}")
    
    plt.close()
    ddpm.model.train()
    
    return interpolated_images