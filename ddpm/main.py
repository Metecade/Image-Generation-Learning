# main.py

import torch
import torch.optim as optim
import os

from dataset import get_loaders
from ddpmnet import UNet, DDPM
from trainer import train_epoch
from evaluate import evaluate
from visualize import plot_ddpm_training_curves, visualize_ddpm_samples, compare_original_noisy_denoised

def main():
    # 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # DDPM参数
    model_channels = 64
    timesteps = 1000
    lr = 2e-4
    num_epochs = 10
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Data loaders ready. Train batches: {len(train_loader)}")
    
    # 初始化模型
    unet = UNet(in_channels=1, model_channels=model_channels).to(device)
    ddpm = DDPM(unet, timesteps=timesteps, device=device)
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 优化器
    optimizer = optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 创建保存目录
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting DDPM training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练一个epoch
        train_loss = train_epoch(ddpm, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.6f}")
        
        # 验证
        visualize_flag = True
        val_loss = evaluate(ddpm, val_loader, device, visualize=visualize_flag, epoch=epoch+1)
        print(f"Val Loss: {val_loss:.6f}")
        
        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet.state_dict(), "results/models/best_ddpm_model.pth")
            print(f"Best model saved with Val Loss: {best_val_loss:.6f}")
    
    # 绘制训练曲线
    plot_ddpm_training_curves(train_losses, val_losses, save_path="results/visualizations/ddpm_training_curves.png")
    
    # 加载最佳模型进行最终评估
    unet.load_state_dict(torch.load("results/models/best_ddpm_model.pth"))
    ddpm_final = DDPM(unet, timesteps=timesteps, device=device)
    
    test_loss = evaluate(ddpm_final, test_loader, device, visualize=True, epoch="final")
    print(f"\nTest Loss: {test_loss:.6f}")
    
    # 生成最终样本展示
    print("Generating final samples...")
    from visualize import visualize_ddpm_samples, generate_interpolation
    
    visualize_ddpm_samples(ddpm_final, device, num_samples=64, 
                          save_path="results/visualizations/final_ddpm_samples.png", epoch="Final")
    
    generate_interpolation(ddpm_final, device, num_steps=10,
                          save_path="results/visualizations/ddpm_interpolation.png")
    
    compare_original_noisy_denoised(ddpm_final, test_loader, device,
                                   save_path="results/visualizations/ddpm_noise_comparison.png", epoch="Final")
    
    print("DDPM training completed!")
    print("Check 'results/visualizations/' for generated images")
    print("Check 'results/models/' for saved models")

if __name__ == "__main__":
    main()