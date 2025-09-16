# evaluate.py

import torch
from tqdm import tqdm
from visualize import (
    visualize_ddim_samples, 
    visualize_denoising_process, 
    compare_original_noisy_denoised,
    visualize_reconstruction_quality,
    visualize_ddim_interpolation
)

def evaluate(ddim, dataloader, device, visualize=False, epoch=None):
    """评估DDIM模型"""
    ddim.model.eval()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc='Evaluating DDIM', leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            
            # 将数据标准化到[-1, 1]范围
            data = data * 2.0 - 1.0
            
            # 计算损失
            loss = ddim.compute_loss(data)
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
        
        # 可视化生成样本 (不同的DDIM步数)
        for ddim_steps in [50, 20, 10]:
            sampling_save_path = f"{save_dir}/ddim_samples_steps{ddim_steps}_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddim_samples_steps{ddim_steps}.png"
            visualize_ddim_samples(ddim, device, num_samples=16, ddim_steps=ddim_steps, save_path=sampling_save_path, epoch=epoch)
        
        # 可视化去噪过程
        denoising_save_path = f"{save_dir}/ddim_denoising_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddim_denoising.png"
        visualize_denoising_process(ddim, device, save_path=denoising_save_path, epoch=epoch, ddim_steps=20)
        
        # 可视化原图vs去噪对比
        comparison_save_path = f"{save_dir}/ddim_comparison_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddim_comparison.png"
        compare_original_noisy_denoised(ddim, dataloader, device, save_path=comparison_save_path, epoch=epoch)
        
        # 可视化重建质量
        reconstruction_save_path = f"{save_dir}/ddim_reconstruction_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddim_reconstruction.png"
        visualize_reconstruction_quality(ddim, dataloader, device, save_path=reconstruction_save_path, epoch=epoch)
        
        # 可视化DDIM插值
        interpolation_save_path = f"{save_dir}/ddim_interpolation_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/ddim_interpolation.png"
        visualize_ddim_interpolation(ddim, dataloader, device, save_path=interpolation_save_path, epoch=epoch)
    
    return avg_loss

def evaluate_sampling_speed(ddim, device, img_size=28, batch_size=16):
    """评估不同DDIM步数的采样速度"""
    import time
    
    ddim.model.eval()
    
    print("\nEvaluating DDIM sampling speed:")
    print("-" * 40)
    
    steps_list = [10, 20, 50, 100, 200, 1000]
    
    for steps in steps_list:
        # 预热
        with torch.no_grad():
            _ = ddim.sample(1, img_size, ddim_steps=steps, eta=0.0)
        
        # 计时
        start_time = time.time()
        with torch.no_grad():
            _ = ddim.sample(batch_size, img_size, ddim_steps=steps, eta=0.0)
        end_time = time.time()
        
        time_per_sample = (end_time - start_time) / batch_size
        print(f"Steps: {steps:4d} | Time per sample: {time_per_sample:.3f}s")
    
    print("-" * 40)

def compare_deterministic_vs_stochastic(ddim, device, img_size=28, num_samples=8, ddim_steps=50):
    """比较确定性采样(eta=0)和随机采样(eta=1)"""
    ddim.model.eval()
    
    print(f"\nComparing deterministic vs stochastic DDIM sampling:")
    print(f"DDIM steps: {ddim_steps}")
    
    # 固定随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 确定性采样
    with torch.no_grad():
        deterministic_samples = ddim.sample(num_samples, img_size, ddim_steps=ddim_steps, eta=0.0)
    
    # 随机采样  
    with torch.no_grad():
        stochastic_samples = ddim.sample(num_samples, img_size, ddim_steps=ddim_steps, eta=1.0)
    
    return deterministic_samples, stochastic_samples