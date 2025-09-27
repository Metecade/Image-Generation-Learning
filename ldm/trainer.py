# trainer.py - Training utilities for Latent Diffusion Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

def train_epoch(ldm, train_loader, optimizer, device, epoch, use_conditioning=False, scaler=None):
    """训练一个epoch"""
    ldm.unet.train()  # 只训练U-Net，VAE保持冻结
    
    running_loss = 0.0
    num_batches = len(train_loader)
    
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch_idx, batch_data in enumerate(loop):
        if use_conditioning:
            data, labels = batch_data
            # 这里可以添加条件编码逻辑
            context = None  # 简化版本暂不使用条件
        else:
            data, _ = batch_data
            context = None
        
        data = data.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = ldm.compute_loss(data, context=context)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ldm.unet.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            loss = ldm.compute_loss(data, context=context)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ldm.unet.parameters(), 1.0)
            
            optimizer.step()
        
        running_loss += loss.item()
        
        # 更新进度条
        loop.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg_Loss': f'{running_loss/(batch_idx+1):.6f}'
        })
    
    avg_loss = running_loss / num_batches
    return avg_loss

def validate_epoch(ldm, dataloader, device, use_conditioning=False):
    """验证一个epoch"""
    ldm.unet.eval()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc='Validation', leave=False)
        
        for batch_data in loop:
            if use_conditioning:
                data, labels = batch_data
                context = None
            else:
                data, _ = batch_data
                context = None
            
            data = data.to(device)
            
            # 计算损失
            loss = ldm.compute_loss(data, context=context)
            
            running_loss += loss.item()
            
            # 更新进度条
            loop.set_postfix({
                'Val_Loss': f'{loss.item():.6f}',
                'Avg_Val_Loss': f'{running_loss/(loop.n+1):.6f}'
            })
    
    avg_loss = running_loss / num_batches
    return avg_loss

def train_ldm(ldm, train_loader, val_loader, optimizer, scheduler, 
              num_epochs, device, save_dir="models", use_conditioning=False, scaler=None):
    """完整的LDM训练流程"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting Latent Diffusion Model training...")
    print(f"Training on device: {device}")
    print(f"U-Net parameters: {sum(p.numel() for p in ldm.unet.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(ldm, train_loader, optimizer, device, epoch, use_conditioning, scaler)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate_epoch(ldm, val_loader, device, use_conditioning)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'unet_state_dict': ldm.unet.state_dict(),
                'vae_state_dict': ldm.vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'unet_config': ldm.unet.__dict__ if hasattr(ldm.unet, '__dict__') else {},
                'scale_factor': ldm.scale_factor,
            }, f"{save_dir}/best_ldm_model.pth")
            print(f"✓ Saved best model (Val Loss: {val_loss:.6f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'unet_state_dict': ldm.unet.state_dict(),
                'vae_state_dict': ldm.vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{save_dir}/ldm_checkpoint_epoch_{epoch+1}.pth")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs-1,
        'unet_state_dict': ldm.unet.state_dict(),
        'vae_state_dict': ldm.vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, f"{save_dir}/ldm_final_model.pth")
    
    print("\n" + "="*50)
    print("LDM Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses

def load_ldm_model(model_path, ldm, optimizer=None, scheduler=None):
    """加载LDM模型"""
    checkpoint = torch.load(model_path, map_location=ldm.device, weights_only=False)
    
    # 加载U-Net权重
    ldm.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    # 如果有VAE权重也加载（通常VAE是预训练的）
    if 'vae_state_dict' in checkpoint:
        ldm.vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    
    print(f"✓ Loaded LDM model from epoch {epoch+1}")
    
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    
    return epoch + 1

def train_vae_first(vae, train_loader, val_loader, num_epochs, device, save_dir):
    """可选：先训练VAE（如果没有预训练的VAE）"""
    print("Pre-training VAE...")
    
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练VAE
        vae.train()
        train_loss = 0.0
        
        for data, _ in tqdm(train_loader, desc=f'VAE Epoch {epoch+1}'):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # VAE前向传播
            recon, posterior = vae(data)
            
            # VAE损失：重建损失 + KL散度
            recon_loss = F.mse_loss(recon, data)
            kl_loss = posterior.kl().mean()
            loss = recon_loss + 0.1 * kl_loss  # KL权重
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证VAE
        vae.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon, posterior = vae(data)
                recon_loss = F.mse_loss(recon, data)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + 0.1 * kl_loss
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"VAE Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳VAE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), f"{save_dir}/best_vae.pth")
        
        scheduler.step()
    
    print(f"VAE pre-training completed. Best val loss: {best_val_loss:.6f}")
    
    # 加载最佳VAE权重
    vae.load_state_dict(torch.load(f"{save_dir}/best_vae.pth", weights_only=False))
    
    return vae