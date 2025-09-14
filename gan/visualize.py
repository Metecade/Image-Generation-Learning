# visualize.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid

def visualize_reconstruction(model, dataloader, device, num_samples=8, save_path=None, epoch=None):
    """
    可视化VAE的重建结果
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 要显示的样本数量
        save_path: 保存路径
        epoch: 当前epoch（用于文件命名）
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(dataloader)
    data, _ = next(data_iter)
    data = data[:num_samples].to(device)
    
    with torch.no_grad():
        recon_data, mu, logvar = model(data)
    
    # 移到CPU并转换格式用于显示
    original = data.cpu()
    reconstructed = recon_data.cpu()
    
    # 创建图像网格
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle(f'VAE Reconstruction Comparison' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i in range(num_samples):
        # 原始图像
        axes[0, i].imshow(original[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # 重建图像
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # plt.show()
    model.train()


def visualize_latent_sampling(model, device, num_samples=16, save_path=None, epoch=None):
    """
    从潜在空间采样生成新图像
    
    Args:
        model: VAE模型
        device: 设备
        num_samples: 要生成的样本数量
        save_path: 保存路径
        epoch: 当前epoch（用于文件命名）
    """
    model.eval()
    
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated = model.decode(z).cpu()
    
    # 创建图像网格
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle(f'Generated Samples from Latent Space' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(generated[idx].squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generated samples saved to: {save_path}")
    
    # plt.show()
    model.train()


def visualize_latent_space_2d(model, dataloader, device, save_path=None, num_batches=10):
    """
    可视化2D潜在空间（如果潜在维度为2）
    
    Args:
        model: VAE模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
        num_batches: 要处理的批次数量
    """
    if model.latent_dim != 2:
        print(f"Warning: Latent dimension is {model.latent_dim}, not 2. Skipping 2D visualization.")
        return
    
    model.eval()
    
    mu_list = []
    labels_list = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            data = data.to(device)
            mu, _ = model.encode(data)
            mu_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # 合并所有数据
    all_mu = np.concatenate(mu_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    # 绘制2D潜在空间
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Latent space visualization saved to: {save_path}")
    
    # plt.show()
    model.train()


def save_comparison_grid(original, reconstructed, save_path, title="VAE Reconstruction"):
    """
    保存原图和重建图的对比网格
    
    Args:
        original: 原始图像张量
        reconstructed: 重建图像张量
        save_path: 保存路径
        title: 图像标题
    """
    # 将原图和重建图拼接
    comparison = torch.cat([original, reconstructed], dim=0)
    
    # 创建网格
    grid = make_grid(comparison, nrow=original.size(0), normalize=True, padding=2)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {save_path}")


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表 [(total, bce, kld), ...]
        val_losses: 验证损失列表 [(total, bce, kld), ...]
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 分离不同类型的损失
    train_total = [loss[0] for loss in train_losses]
    train_bce = [loss[1] for loss in train_losses]
    train_kld = [loss[2] for loss in train_losses]
    
    val_total = [loss[0] for loss in val_losses]
    val_bce = [loss[1] for loss in val_losses]
    val_kld = [loss[2] for loss in val_losses]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 总损失
    axes[0].plot(epochs, train_total, 'b-', label='Train')
    axes[0].plot(epochs, val_total, 'r-', label='Validation')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # BCE损失
    axes[1].plot(epochs, train_bce, 'b-', label='Train')
    axes[1].plot(epochs, val_bce, 'r-', label='Validation')
    axes[1].set_title('Reconstruction Loss (BCE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # KLD损失
    axes[2].plot(epochs, train_kld, 'b-', label='Train')
    axes[2].plot(epochs, val_kld, 'r-', label='Validation')
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    # plt.show()


# ==================== GAN 专用可视化函数 ====================

def visualize_gan_samples(generator, device, num_samples=16, save_path=None, epoch=None):
    """
    可视化GAN生成的样本
    """
    generator.eval()
    
    with torch.no_grad():
        # 生成随机噪声
        noise = torch.randn(num_samples, generator.latent_dim, device=device)
        fake_data = generator(noise)
        
        # 将数据从[-1, 1]转换到[0, 1]
        fake_data = (fake_data + 1) / 2.0
        fake_data = fake_data.cpu()
    
    # 创建图像网格
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle(f'GAN Generated Samples' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(fake_data[idx].squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GAN samples saved to: {save_path}")
    
    plt.close()
    generator.train()


def compare_real_fake(generator, dataloader, device, num_samples=8, save_path=None, epoch=None):
    """
    对比真实图像和生成图像
    """
    generator.eval()
    
    # 获取真实数据
    data_iter = iter(dataloader)
    real_data, _ = next(data_iter)
    real_data = real_data[:num_samples]
    
    # 生成假数据
    with torch.no_grad():
        noise = torch.randn(num_samples, generator.latent_dim, device=device)
        fake_data = generator(noise)
        # 将生成数据从[-1, 1]转换到[0, 1]
        fake_data = (fake_data + 1) / 2.0
        fake_data = fake_data.cpu()
    
    # 创建对比图
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle(f'Real vs Generated Comparison' + (f' - Epoch {epoch}' if epoch is not None else ''), fontsize=16)
    
    for i in range(num_samples):
        # 真实图像
        axes[0, i].imshow(real_data[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
        
        # 生成图像
        axes[1, i].imshow(fake_data[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Generated')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Real vs fake comparison saved to: {save_path}")
    
    plt.close()
    generator.train()


def plot_gan_training_curves(g_train_losses, g_val_losses, d_train_losses, d_val_losses, save_path=None):
    """
    绘制GAN训练曲线
    """
    epochs = range(1, len(g_train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generator损失
    axes[0].plot(epochs, g_train_losses, 'b-', label='Train')
    axes[0].plot(epochs, g_val_losses, 'r-', label='Validation')
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Discriminator损失
    axes[1].plot(epochs, d_train_losses, 'b-', label='Train')
    axes[1].plot(epochs, d_val_losses, 'r-', label='Validation')
    axes[1].set_title('Discriminator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GAN training curves saved to: {save_path}")
    
    plt.close()