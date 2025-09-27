# evaluate.py - Evaluation utilities for LDM

import torch
from tqdm import tqdm
import numpy as np

def evaluate_ldm(ldm, dataloader, device, num_samples=100):
    """评估LDM模型"""
    ldm.unet.eval()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc='Evaluating LDM', leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            
            # 计算损失
            loss = ldm.compute_loss(data)
            running_loss += loss.item()
            
            loop.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{running_loss/(batch_idx+1):.6f}'
            })
    
    avg_loss = running_loss / num_batches
    return avg_loss

def evaluate_reconstruction_quality(ldm, dataloader, device, num_samples=50):
    """评估VAE重建质量"""
    ldm.vae.eval()
    
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # VAE重建
            recon, _ = ldm.vae(data, sample_posterior=False)
            
            # 计算MSE
            mse = torch.nn.functional.mse_loss(recon, data, reduction='sum')
            total_mse += mse.item()
            total_samples += data.size(0)
            
            if total_samples >= num_samples:
                break
    
    avg_mse = total_mse / total_samples
    print(f"VAE Reconstruction MSE: {avg_mse:.6f}")
    
    return avg_mse

def evaluate_sampling_speed(ldm, device, latent_shape=(4, 8, 8), batch_size=16):
    """评估采样速度"""
    import time
    from latent_diffusion import DDIMSampler
    
    ldm.unet.eval()
    
    print("\nEvaluating LDM sampling speed:")
    print("-" * 40)
    
    # DDPM采样
    start_time = time.time()
    with torch.no_grad():
        _ = ldm.sample_latent(batch_size, latent_shape)
    ddpm_time = time.time() - start_time
    
    # DDIM采样 - 不同步数
    ddim_sampler = DDIMSampler(ldm)
    steps_list = [10, 20, 50, 100]
    
    for steps in steps_list:
        start_time = time.time()
        with torch.no_grad():
            _ = ddim_sampler.sample(batch_size, latent_shape, ddim_steps=steps)
        ddim_time = time.time() - start_time
        
        speedup = ddpm_time / ddim_time if ddim_time > 0 else float('inf')
        print(f"DDIM {steps:3d} steps: {ddim_time:.3f}s | Speedup: {speedup:.1f}x")
    
    print(f"DDPM 1000 steps: {ddpm_time:.3f}s | Baseline")
    print("-" * 40)

def calculate_fid_score(real_images, generated_images, device):
    """计算FID分数（简化版）"""
    # 注意：这是一个简化的FID实现，实际应用中需要使用预训练的Inception网络
    try:
        from scipy.linalg import sqrtm
        
        # 计算特征统计
        def get_statistics(images):
            # 简化：使用图像的均值和方差作为特征
            images_flat = images.view(images.size(0), -1)
            mu = torch.mean(images_flat, dim=0)
            sigma = torch.cov(images_flat.T)
            return mu.cpu().numpy(), sigma.cpu().numpy()
        
        mu1, sigma1 = get_statistics(real_images)
        mu2, sigma2 = get_statistics(generated_images)
        
        # 计算FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        return fid
        
    except ImportError:
        print("scipy not available, skipping FID calculation")
        return None

def evaluate_interpolation_quality(ldm, test_data, device, num_pairs=5):
    """评估插值质量"""
    ldm.eval()
    
    interpolation_scores = []
    
    with torch.no_grad():
        for i in range(num_pairs):
            # 随机选择两个图像
            idx1, idx2 = torch.randint(0, len(test_data), (2,))
            img1 = test_data[idx1:idx1+1].to(device)
            img2 = test_data[idx2:idx2+1].to(device)
            
            # 生成插值
            interpolated = ldm.interpolate_images(img1, img2, num_steps=5)
            
            # 计算插值的平滑度（相邻帧之间的差异）
            smoothness = 0.0
            for j in range(len(interpolated) - 1):
                diff = torch.nn.functional.mse_loss(interpolated[j], interpolated[j+1])
                smoothness += diff.item()
            
            smoothness /= (len(interpolated) - 1)
            interpolation_scores.append(smoothness)
    
    avg_smoothness = np.mean(interpolation_scores)
    print(f"Interpolation smoothness score: {avg_smoothness:.6f}")
    
    return avg_smoothness