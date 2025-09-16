# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(ddpm, dataloader, optimizer, device, epoch):
    ddpm.model.train()
    
    running_loss = 0.0
    num_batches = len(dataloader)
    
    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)
        
        # 将数据标准化到[-1, 1]范围
        data = data * 2.0 - 1.0
        
        optimizer.zero_grad()
        
        # 计算损失
        loss = ddpm.compute_loss(data)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # 更新进度条
        loop.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg_Loss': f'{running_loss/(batch_idx+1):.6f}'
        })
    
    avg_loss = running_loss / num_batches
    return avg_loss