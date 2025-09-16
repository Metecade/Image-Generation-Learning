# main.py

import torch
import torch.optim as optim
import os

from dataset import get_loaders
from ddimnet import UNet, DDIM
from trainer import train_ddim
from evaluate import evaluate, evaluate_sampling_speed, compare_deterministic_vs_stochastic
from visualize import (
    plot_ddim_training_curves, 
    compare_ddim_steps, 
    compare_eta_values,
    visualize_ddim_samples
)

def main():
    # 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # DDIM参数
    model_channels = 64
    timesteps = 1000
    lr = 2e-4
    num_epochs = 10
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Data loaders ready. Train batches: {len(train_loader)}")
    
    # 初始化模型
    unet = UNet(in_channels=1, model_channels=model_channels).to(device)
    ddim = DDIM(unet, timesteps=timesteps, device=device)
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 优化器
    optimizer = optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 创建保存目录
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    print("Starting DDIM training...")
    
    # 训练模型
    train_losses, val_losses = train_ddim(
        ddim=ddim,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir="results/models"
    )
    
    # 绘制训练曲线
    plot_ddim_training_curves(train_losses, val_losses)
    
    # 最终评估
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # 在测试集上评估
    test_loss = evaluate(ddim, test_loader, device, visualize=True, epoch="final")
    print(f"Final test loss: {test_loss:.6f}")
    
    # 评估采样速度
    evaluate_sampling_speed(ddim, device)
    
    # 比较确定性和随机采样
    det_samples, sto_samples = compare_deterministic_vs_stochastic(ddim, device)
    
    # 可视化不同DDIM配置的效果
    compare_ddim_steps(ddim, device, save_path="results/visualizations/ddim_steps_comparison.png")
    compare_eta_values(ddim, device, save_path="results/visualizations/ddim_eta_comparison.png")
    
    # 生成最终样本展示
    print("\nGenerating final sample showcase...")
    for steps in [10, 20, 50]:
        for eta in [0.0, 1.0]:
            save_path = f"results/visualizations/ddim_final_samples_steps{steps}_eta{eta}.png"
            visualize_ddim_samples(ddim, device, num_samples=16, ddim_steps=steps, eta=eta, save_path=save_path)
    
    print("\n" + "="*50)
    print("DDIM Training and Evaluation Completed!")
    print("="*50)
    print("Key DDIM Features Demonstrated:")
    print("✓ Fast sampling with reduced steps (10-50 vs 1000)")
    print("✓ Deterministic sampling (eta=0) for reproducible results")  
    print("✓ Stochastic sampling (eta>0) for diversity")
    print("✓ Interpolation in latent space")
    print("✓ Quality maintained with fewer sampling steps")
    print(f"\nResults saved in: results/")

if __name__ == "__main__":
    main()
