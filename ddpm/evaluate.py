# evaluate.py

import torch
from tqdm import tqdm
from visualize import (
    visualize_ddpm_samples, 
    visualize_denoising_process, 
    compare_original_noisy_denoised,
    visualize_reconstruction_quality
)

def evaluate(ddpm, dataloader, device, visualize=False, epoch=None):
    ddpm.model.eval()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            
            # 将数据标准化到[-1, 1]范围
            data = data * 2.0 - 1.0
            
            # 计算损失
            loss = ddpm.compute_loss(data)
            running_loss += loss.item()
            
            loop.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{running_loss/(batch_idx+1):.6f}'
            })
    
    avg_loss = running_loss / num_batches
    
    # 添加可视化功能
    if visualize:
        # 创建保存目录
        save_dir = "results/visualizations"
        
        # 可视化生成样本
        sampling_save_path = f"{save_dir}/ddpm_samples_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddpm_samples.png"
        visualize_ddpm_samples(ddpm, device, num_samples=16, save_path=sampling_save_path, epoch=epoch)
        
        # 可视化去噪过程
        denoising_save_path = f"{save_dir}/ddpm_denoising_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddpm_denoising.png"
        visualize_denoising_process(ddpm, device, save_path=denoising_save_path, epoch=epoch)
        
        # 可视化原图vs去噪对比
        comparison_save_path = f"{save_dir}/ddpm_comparison_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddpm_comparison.png"
        compare_original_noisy_denoised(ddpm, dataloader, device, save_path=comparison_save_path, epoch=epoch)
        
        # 可视化重建质量
        reconstruction_save_path = f"{save_dir}/ddpm_reconstruction_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddpm_reconstruction.png"
        visualize_reconstruction_quality(ddpm, dataloader, device, save_path=reconstruction_save_path, epoch=epoch)
    
    return avg_loss