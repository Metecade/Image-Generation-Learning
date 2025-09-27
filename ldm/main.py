# main.py - Main script for Latent Diffusion Model

import torch
import torch.optim as optim
import os

from dataset import get_loaders
from vae import VQModel
from ldmnet import LatentUNet
from latent_diffusion import LatentDiffusion, DDIMSampler
from trainer import train_ldm, load_ldm_model, train_vae_first

def get_model_configs():
    """获取模型配置"""
    
    # VAE配置
    vae_config = {
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "in_channels": 1,  # MNIST是灰度图像
        "resolution": 64,
        "z_channels": 4,
        "double_z": True,
        "out_ch": 1,  # 输出也是灰度图像
        "resamp_with_conv": True
    }
    
    # U-Net配置 - 优化后的轻量版本
    unet_config = {
        "in_channels": 4,  # VAE的潜在维度
        "model_channels": 128,  # 减少到128以提高速度
        "num_res_blocks": 2,
        "attention_resolutions": [2],  # 只在一个分辨率使用注意力
        "channel_mult": [1, 2, 4],  # 减少层数
        "use_spatial_transformer": True,
        "context_dim": None  # 暂不使用条件生成
    }
    
    return vae_config, unet_config

def main():
    # 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # LDM参数 - 优化后的设置
    img_size = 64  # 使用64x64图像
    batch_size = 16  # 增加batch size以提高效率
    lr = 2e-4  # 稍微提高学习率
    num_epochs = 1
    timesteps = 1000
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_loaders(img_size=img_size, batch_size=batch_size)
    print(f"Data loaders ready. Train batches: {len(train_loader)}")
    
    # 获取模型配置
    vae_config, unet_config = get_model_configs()
    
    # 创建保存目录
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # 初始化VAE
    print("Initializing VAE...")
    vae = VQModel(vae_config, embed_dim=4).to(device)
    
    # 选择：使用预训练VAE或重新训练
    vae_path = "results/models/best_vae.pth"
    if os.path.exists(vae_path):
        print("Loading pre-trained VAE...")
        vae.load_state_dict(torch.load(vae_path, weights_only=False))
    else:
        print("Pre-training VAE...")
        vae = train_vae_first(vae, train_loader, val_loader, num_epochs=2, 
                             device=device, save_dir="results/models")
    
    # 冻结VAE参数
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    
    # 初始化LDM
    print("Initializing Latent Diffusion Model...")
    ldm = LatentDiffusion(
        unet_config=unet_config,
        vae=vae,  # 传入预训练的VAE
        timesteps=timesteps,
        scale_factor=0.18215,  # 标准LDM缩放因子
        device=device
    ).to(device)
    
    # 不需要再赋值VAE，因为我们已经传入了预训练的VAE
    
    print(f"LDM U-Net parameters: {sum(p.numel() for p in ldm.unet.parameters()):,}")
    
    # 优化器 - 优化版本
    optimizer = optim.AdamW(ldm.unet.parameters(), lr=lr, weight_decay=0.01, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 启用混合精度训练（如果支持）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print("Starting LDM training...")
    print(f"Using mixed precision: {scaler is not None}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    # 训练模型
    train_losses, val_losses = train_ldm(
        ldm=ldm,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir="results/models",
        scaler=scaler  # 传入scaler用于混合精度
    )
    
    # 最终评估和样本生成
    print("\n" + "="*50)
    print("Final Evaluation and Sample Generation")
    print("="*50)
    
    # 加载最佳模型进行测试
    load_ldm_model("results/models/best_ldm_model.pth", ldm)
    
    # 生成样本
    print("Generating samples...")
    
    # 标准采样
    with torch.no_grad():
        ldm.eval()
        latent_shape = ldm.get_latent_shape(img_size)
        print(f"Latent shape: {latent_shape}")
        
        # 生成16个样本
        samples = ldm.sample(batch_size=16, latent_shape=latent_shape)
        
        # 保存样本
        save_samples(samples, "results/visualizations/ldm_samples.png")
    
    # DDIM快速采样
    print("Testing DDIM fast sampling...")
    ddim_sampler = DDIMSampler(ldm)
    
    with torch.no_grad():
        for steps in [50, 20, 10]:
            z_samples = ddim_sampler.sample(
                batch_size=16, 
                latent_shape=latent_shape, 
                ddim_steps=steps
            )
            samples = ldm.decode_from_latent(z_samples)
            save_samples(samples, f"results/visualizations/ldm_ddim_{steps}steps.png")
            print(f"✓ Generated samples with {steps} DDIM steps")
    
    # 插值测试
    print("Testing interpolation...")
    test_interpolation(ldm, test_loader, device)
    
    print("\n" + "="*50)
    print("LDM Training and Evaluation Completed!")
    print("="*50)
    print("Key LDM Features:")
    print("✓ Efficient training in latent space")
    print("✓ High-quality image generation")
    print("✓ Fast DDIM sampling")
    print("✓ Smooth interpolation in latent space")
    print("✓ Memory efficient compared to pixel-space diffusion")
    print(f"\nResults saved in: results/")

def save_samples(samples, path):
    """保存生成的样本"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 转换到[0, 1]范围
    samples = (samples + 1) / 2.0
    samples = torch.clamp(samples, 0, 1).cpu()
    
    # 创建网格
    grid_size = 4  # 4x4网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(samples):
                img = samples[idx]
                if img.shape[0] == 1:  # 灰度图
                    axes[i, j].imshow(img.squeeze(), cmap='gray')
                else:  # 彩色图
                    axes[i, j].imshow(img.permute(1, 2, 0))
            axes[i, j].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved to: {path}")

def test_interpolation(ldm, test_loader, device):
    """测试插值功能"""
    # 获取测试图像
    data_iter = iter(test_loader)
    test_data, _ = next(data_iter)
    
    img1 = test_data[0:1].to(device)
    img2 = test_data[1:2].to(device)
    
    # 插值
    with torch.no_grad():
        interpolated = ldm.interpolate_images(img1, img2, num_steps=8)
        save_samples(interpolated, "results/visualizations/ldm_interpolation.png")
    
    print("✓ Interpolation completed")


if __name__ == "__main__":
    main()
