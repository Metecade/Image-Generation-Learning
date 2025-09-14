# evaluate.py

import torch
from tqdm import tqdm
from visualize import visualize_gan_samples, compare_real_fake

def evaluate(generator, discriminator, dataloader, device, loss_fn, visualize=False, epoch=None):
    generator.eval()
    discriminator.eval()
    generator_loss = 0.0
    discriminator_loss = 0.0

    loop = tqdm(dataloader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(device)
            batch_size = data.size(0)

            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_data = generator(noise)

            fake_output = discriminator(fake_data)
            real_output = discriminator(data)

            g_loss = loss_fn(fake_output, True)

            d_loss_real = loss_fn(real_output, True)
            d_loss_fake = loss_fn(fake_output, False)
            d_loss = (d_loss_real + d_loss_fake) / 2

            generator_loss += g_loss.item()
            discriminator_loss += d_loss.item()

    avg_g_loss = generator_loss / len(dataloader)
    avg_d_loss = discriminator_loss / len(dataloader)

    # 添加可视化功能
    if visualize:
        # 创建保存目录
        save_dir = "results/visualizations"
        
        # 可视化生成样本
        sampling_save_path = f"{save_dir}/gan_samples_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/gan_samples.png"
        visualize_gan_samples(generator, device, num_samples=16, save_path=sampling_save_path, epoch=epoch)
        
        # 可视化真实vs生成对比
        compare_save_path = f"{save_dir}/gan_comparison_epoch_{epoch}.png" if epoch is not None else f"{save_dir}/gan_comparison.png"
        compare_real_fake(generator, dataloader, device, num_samples=8, save_path=compare_save_path, epoch=epoch)

    return avg_g_loss, avg_d_loss
