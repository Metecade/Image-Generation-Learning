# visualize.py
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_vae_recon(vae, dataloader, device, save_path):
    vae.eval()
    x, _ = next(iter(dataloader))
    x = x[:8].to(device)
    with torch.no_grad():
        x_recon, _, _ = vae(x)
    grid = torch.cat([x, x_recon], dim=0)
    grid = (grid * 0.5 + 0.5).clamp(0,1)
    grid_np = make_grid(grid, nrow=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12,4))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('Top: Original, Bottom: Recon')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_grid(tensor, nrow=8):
    # tensor: (N,C,H,W)
    tensor = tensor.detach().cpu()
    N,C,H,W = tensor.shape
    ncol = nrow
    nrow_grid = int(np.ceil(N / ncol))
    canvas = torch.zeros(C, nrow_grid*H, ncol*W)
    for idx in range(N):
        r = idx // ncol
        c = idx % ncol
        canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = tensor[idx]
    canvas = canvas.permute(1,2,0)
    if C == 1:
        return canvas.squeeze().numpy()
    return canvas.numpy()


def visualize_class_conditional_samples(vae, ldt, diffusion, device, num_classes=10, samples_per_class=8, img_size=32, save_path=None):
    vae.eval(); ldt.eval()
    with torch.no_grad():
        latent_channels = 4  # encoder output channels
        batch_size = num_classes * samples_per_class
        # latent spatial size inferred from ldt.img_size (should match encoder output 32/(2^3)=4)
        latent_spatial = ldt.img_size if hasattr(ldt, 'img_size') else img_size // 8
        z = torch.randn(batch_size, latent_channels, latent_spatial, latent_spatial, device=device)
        # Build time schedule
        skip = diffusion.num_timesteps // diffusion.ddim_timesteps
        sequence = list(range(0, diffusion.num_timesteps, skip))
        sequence_next = [-1] + sequence[:-1]
        labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
        for i, j in zip(reversed(sequence), reversed(sequence_next)):
            t_tensor = torch.full((batch_size,), i, dtype=torch.long, device=device)
            alpha = diffusion.alphas_cumprod[t_tensor].view(-1,1,1,1)
            alpha_prev = diffusion.alphas_cumprod[j].view(-1,1,1,1) if j>=0 else torch.tensor(1.0, device=device).view(1,1,1,1)
            beta = diffusion.betas[t_tensor].view(-1,1,1,1)
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1-alpha)
            pred_noise = ldt(z, t_tensor, labels)
            x0_pred = (z - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            dir_xt = torch.sqrt(1.0 - alpha_prev - diffusion.eta**2 * beta) * pred_noise
            noise = diffusion.eta * torch.sqrt(beta) * torch.randn_like(z) if j>0 else 0.0
            z = torch.sqrt(alpha_prev) * x0_pred + dir_xt + noise
        # decode
        x_gen = vae.decoder(z)
        imgs = (x_gen * 0.5 + 0.5).clamp(0,1)
    # arrange grid by class rows
    rows = []
    C = imgs.size(1)
    for cls in range(num_classes):
        row_imgs = imgs[cls*samples_per_class:(cls+1)*samples_per_class]
        rows.append(row_imgs)
    grid = torch.cat(rows, dim=0)
    grid_np = make_grid(grid, nrow=samples_per_class)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(samples_per_class*1.5, num_classes*1.5))
        plt.imshow(grid_np if C!=1 else grid_np, cmap=None if C!=1 else 'gray')
        plt.axis('off')
        plt.title('Class-Conditional Samples')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return imgs
