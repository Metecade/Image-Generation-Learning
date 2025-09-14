# evaluate.py

import torch
from tqdm import tqdm
from visualize import visualize_reconstruction, visualize_latent_sampling

def evaluate(model, dataloader, device, loss_fn, visualize=False, epoch=None):
    model.eval()
    running_loss = 0.0
    bce_loss = 0.0
    kld_loss = 0.0

    loop = tqdm(dataloader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)

            recon_data, mu, logvar = model(data)
            loss, bce, kld = loss_fn(recon_data, data, mu, logvar)

            running_loss += loss.item()
            bce_loss += bce.item() 
            kld_loss += kld.item()

            loop.set_postfix(loss=running_loss/(batch_idx+1), 
                             bce=bce_loss/(batch_idx+1),    
                             kld=kld_loss/(batch_idx+1))
            
    avg_loss = running_loss / len(dataloader)
    avg_bce = bce_loss / len(dataloader)
    avg_kld = kld_loss / len(dataloader)

    # 添加可视化功能
    if visualize:
        # 创建保存目录
        save_dir = "results/visualizations"
        
        # 可视化重建结果
        recon_save_path = f"{save_dir}/reconstruction_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/reconstruction.png"
        visualize_reconstruction(model, dataloader, device, num_samples=8, save_path=recon_save_path, epoch=epoch)
        
        # 可视化从潜在空间生成的样本
        sampling_save_path = f"{save_dir}/sampling_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/sampling.png"
        visualize_latent_sampling(model, device, num_samples=16, save_path=sampling_save_path, epoch=epoch)

    return avg_loss, avg_bce, avg_kld
