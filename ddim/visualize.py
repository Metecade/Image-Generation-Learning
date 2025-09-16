# visualize.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体或默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def visualize_ddim_samples(ddim, device, num_samples=16, ddim_steps=50, eta=0.0, save_path=None, epoch=None):
    """
    可视化DDIM生成的样本
    """
    ddim.model.eval()
    
    with torch.no_grad():
        # 生成样本
        samples = ddim.sample(num_samples, img_size=28, ddim_steps=ddim_steps, eta=eta)
        
        # 将数据从[-1, 1]转换到[0, 1]
        samples = (samples + 1) / 2.0
        samples = samples.cpu()
    
    # 创建图像网格
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    title = f'DDIM Generated Samples (Steps: {ddim_steps}, η: {eta})'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)
    
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
        print(f"DDIM samples saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def visualize_denoising_process(ddim, device, ddim_steps=20, save_path=None, epoch=None, steps_to_show=8):
    """
    可视化DDIM的去噪过程
    """
    ddim.model.eval()
    
    with torch.no_grad():
        # 从纯噪声开始
        img = torch.randn(1, 1, 28, 28, device=device)
        images = []
        
        # 创建DDIM采样时间表
        skip = ddim.timesteps // ddim_steps
        seq = range(0, ddim.timesteps, skip)
        seq = list(seq)
        seq_next = [-1] + list(seq[:-1])
        
        # 选择要显示的步骤
        show_indices = np.linspace(0, len(seq)-1, steps_to_show, dtype=int)
        
        # 逐步去噪并保存关键步骤
        for i, (t, t_prev) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            t_prev_tensor = torch.full((1,), t_prev, device=device, dtype=torch.long)
            
            img = ddim.ddim_sample_step(img, t_tensor, t_prev_tensor, eta=0.0)
            
            if (len(seq) - 1 - i) in show_indices:
                # 转换到[0, 1]范围并保存
                img_to_save = (img + 1) / 2.0
                images.append(img_to_save.cpu().squeeze().numpy())
    
    # 创建去噪过程可视化
    fig, axes = plt.subplots(1, steps_to_show, figsize=(steps_to_show * 2, 2))
    title = f'DDIM Denoising Process (Steps: {ddim_steps})'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray')
        step_num = show_indices[len(show_indices) - 1 - i]
        ax.set_title(f'Step {step_num}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDIM denoising process saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def compare_original_noisy_denoised(ddim, dataloader, device, save_path=None, epoch=None):
    """
    对比原图、加噪图像和去噪结果
    """
    ddim.model.eval()
    
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
        title = f'Original vs Noisy vs DDIM Predicted'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        fig.suptitle(title, fontsize=16)
        
        for row, t_val in enumerate(t_values):
            t = torch.full((6,), t_val, device=device, dtype=torch.long)
            
            # 加噪
            noise = torch.randn_like(real_data_norm)
            noisy_data = ddim.q_sample(real_data_norm, t, noise)
            
            # 预测噪声并估算去噪结果（一步去噪近似）
            predicted_noise = ddim.model(noisy_data, t)
            alpha_t = ddim.alphas_cumprod[t_val]
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
        print(f"DDIM comparison saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def visualize_reconstruction_quality(ddim, dataloader, device, save_path=None, epoch=None):
    """
    可视化重建质量
    """
    ddim.model.eval()
    
    # 获取一批真实数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:4].to(device)  # 取4个样本
    
    # 标准化到[-1, 1]
    real_data_norm = real_data * 2.0 - 1.0
    
    with torch.no_grad():
        # 测试不同程度的噪声重建能力
        noise_levels = [0, 100, 300, 500, 700]  
        
        fig, axes = plt.subplots(4, len(noise_levels), figsize=(len(noise_levels) * 2, 8))
        title = f'DDIM Reconstruction Quality Test'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        fig.suptitle(title, fontsize=16)
        
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
                    noisy_img = ddim.q_sample(real_data_norm[row:row+1], t, noise)
                    
                    # 预测噪声并重建
                    predicted_noise = ddim.model(noisy_img, t)
                    alpha_t = ddim.alphas_cumprod[noise_level]
                    predicted_clean = (noisy_img - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                    
                    # 转换到[0, 1]范围
                    img_to_show = torch.clamp((predicted_clean + 1) / 2.0, 0, 1).squeeze().cpu()
                
                axes[row, col].imshow(img_to_show, cmap='gray')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDIM reconstruction saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def visualize_ddim_interpolation(ddim, dataloader, device, save_path=None, epoch=None):
    """
    可视化DDIM插值功能（简化版）
    """
    ddim.model.eval()
    
    # 获取两个真实样本
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    
    # 确保我们有足够的样本
    if len(real_data) < 2:
        print("Warning: Not enough samples for interpolation, skipping...")
        return
    
    x1 = real_data[0:1].to(device) * 2.0 - 1.0
    x2 = real_data[1:2].to(device) * 2.0 - 1.0
    
    with torch.no_grad():
        try:
            # 选择较低的噪声水平进行插值
            t = torch.full((1,), 200, device=device, dtype=torch.long)
            
            # 生成插值序列
            interpolated = ddim.interpolate(x1, x2, t, num_steps=10)
            
            # 转换到[0, 1]范围
            interpolated = (interpolated + 1) / 2.0
            interpolated = interpolated.cpu()
            
            # 可视化插值结果
            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
            title = f'DDIM Interpolation'
            if epoch is not None:
                title += f' - Epoch {epoch}'
            fig.suptitle(title, fontsize=16)
            
            for i in range(10):
                axes[i].imshow(interpolated[i].squeeze(), cmap='gray')
                axes[i].set_title(f'α={i/9:.1f}')
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"DDIM interpolation saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to generate interpolation: {e}")
            print("Skipping interpolation visualization...")
    
    ddim.model.train()


def compare_ddim_steps(ddim, device, save_path=None, epoch=None):
    """
    比较不同DDIM步数的采样效果
    """
    ddim.model.eval()
    
    steps_list = [10, 20, 50, 100]
    num_samples = 4
    
    with torch.no_grad():
        all_samples = []
        for steps in steps_list:
            samples = ddim.sample(num_samples, img_size=28, ddim_steps=steps, eta=0.0)
            samples = (samples + 1) / 2.0
            all_samples.append(samples.cpu())
    
    # 可视化比较
    fig, axes = plt.subplots(len(steps_list), num_samples, figsize=(num_samples * 2, len(steps_list) * 2))
    title = f'DDIM Steps Comparison'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)
    
    for row, (steps, samples) in enumerate(zip(steps_list, all_samples)):
        for col in range(num_samples):
            axes[row, col].imshow(samples[col].squeeze(), cmap='gray')
            if col == 0:
                axes[row, col].set_ylabel(f'{steps} Steps', fontsize=12, rotation=90, va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDIM steps comparison saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def compare_eta_values(ddim, device, save_path=None, epoch=None):
    """
    比较不同eta值(随机性程度)的采样效果
    """
    ddim.model.eval()
    
    eta_values = [0.0, 0.5, 1.0]  # 确定性 -> 随机性
    num_samples = 4
    ddim_steps = 50
    
    with torch.no_grad():
        all_samples = []
        for eta in eta_values:
            samples = ddim.sample(num_samples, img_size=28, ddim_steps=ddim_steps, eta=eta)
            samples = (samples + 1) / 2.0
            all_samples.append(samples.cpu())
    
    # 可视化比较
    fig, axes = plt.subplots(len(eta_values), num_samples, figsize=(num_samples * 2, len(eta_values) * 2))
    title = f'DDIM η (Stochasticity) Comparison'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)
    
    for row, (eta, samples) in enumerate(zip(eta_values, all_samples)):
        for col in range(num_samples):
            axes[row, col].imshow(samples[col].squeeze(), cmap='gray')
            if col == 0:
                label = 'Deterministic' if eta == 0.0 else f'η = {eta}'
                axes[row, col].set_ylabel(label, fontsize=12, rotation=90, va='center')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DDIM eta comparison saved to: {save_path}")
    
    plt.close()
    ddim.model.train()


def plot_ddim_training_curves(train_losses, val_losses, save_path="results/visualizations/ddim_training_curves.png"):
    """
    绘制DDIM训练曲线
    """
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDIM Training and Validation Loss')
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
    plt.title('DDIM Smoothed Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"DDIM training curves saved to: {save_path}")
    plt.close()