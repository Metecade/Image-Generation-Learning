# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(ddim, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    ddim.model.train()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)
        
        # 将数据标准化到[-1, 1]范围
        data = data * 2.0 - 1.0
        
        optimizer.zero_grad()
        
        # 计算损失
        loss = ddim.compute_loss(data)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(ddim.model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # 更新进度条
        loop.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg_Loss': f'{running_loss/(batch_idx+1):.6f}'
        })
    
    avg_loss = running_loss / num_batches
    return avg_loss

def validate_epoch(ddim, dataloader, device):
    """验证一个epoch"""
    ddim.model.eval()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc='Validation', leave=False)
        
        for data, _ in loop:
            data = data.to(device)
            
            # 将数据标准化到[-1, 1]范围
            data = data * 2.0 - 1.0
            
            # 计算损失
            loss = ddim.compute_loss(data)
            
            running_loss += loss.item()
            
            # 更新进度条
            loop.set_postfix({
                'Val_Loss': f'{loss.item():.6f}',
                'Avg_Val_Loss': f'{running_loss/(loop.n+1):.6f}'
            })
    
    avg_loss = running_loss / num_batches
    return avg_loss

def train_ddim(ddim, train_loader, val_loader, optimizer, scheduler, num_epochs, device, save_dir):
    """完整的DDIM训练流程"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting DDIM training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(ddim, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate_epoch(ddim, val_loader, device)
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
                'model_state_dict': ddim.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{save_dir}/best_ddim_model.pth")
            print(f"✓ Saved best model (Val Loss: {val_loss:.6f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddim.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{save_dir}/ddim_checkpoint_epoch_{epoch+1}.pth")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': ddim.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, f"{save_dir}/ddim_final_model.pth")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses

def load_ddim_model(model_path, ddim, optimizer=None, scheduler=None):
    """加载DDIM模型"""
    checkpoint = torch.load(model_path, map_location=ddim.device)
    
    ddim.model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    
    print(f"✓ Loaded model from epoch {epoch+1}")
    
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    
    return epoch + 1