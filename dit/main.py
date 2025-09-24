# main.py

import torch, os
from torch import optim
from dataset import get_loaders
from vae import VAE
from ldt import LDT
from ddim import DDIM
from trainer import train_vae_epoch, eval_vae, train_ldt_epoch, eval_ldt
from visualize import visualize_vae_recon, visualize_class_conditional_samples
import matplotlib.pyplot as plt


def plot_curves(train_hist, val_hist, title, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_hist)+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_hist, label='train')
    plt.plot(epochs, val_hist, label='val')
    plt.title(title)
    plt.xlabel('epoch'); plt.ylabel('loss')
    plt.legend(); plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader, val_loader, test_loader = get_loaders()

    # 1. Train VAE
    vae = VAE(in_channels=3, latent_dim=64, out_channels=4).to(device)
    vae_opt = optim.AdamW(vae.parameters(), lr=2e-4, weight_decay=1e-4)

    vae_epochs = 1
    best_val = float('inf')
    train_loss_hist = []
    val_loss_hist = []

    if os.path.exists('dit/results/models/best_vae.pth'):
        vae.load_state_dict(torch.load('dit/results/models/best_vae.pth', weights_only=False))
        print("Loaded pre-trained VAE from dit/results/models/best_vae.pth")
    else:
        print("No pre-trained VAE found, starting training...")
        for epoch in range(vae_epochs):
            tl = train_vae_epoch(vae, train_loader, vae_opt, device)
            vl = eval_vae(vae, val_loader, device)
            train_loss_hist.append(tl) 
            val_loss_hist.append(vl)
            print(f"[VAE] Epoch {epoch+1}/{vae_epochs} train {tl:.4f} val {vl:.4f}")

            if vl < best_val:
                best_val = vl
                os.makedirs('dit/results/models', exist_ok=True)
                torch.save(vae.state_dict(), 'dit/results/models/best_vae.pth')

        plot_curves(train_loss_hist, val_loss_hist, 'VAE Loss', 'dit/results/visualizations/vae_loss.png')
        visualize_vae_recon(vae, val_loader, device, 'dit/results/visualizations/vae_recon.png')

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # 2. Train LDT
    ldt = LDT(channels=4, patch_size=2, img_size=32, hidden_size=768, time_embed_dim=512, cond_embed_dim=512, num_layers=6, num_heads=8).to(device)
    ldt_opt = optim.AdamW(ldt.parameters(), lr=1e-4, weight_decay=1e-4)

    diffusion = DDIM(model=ldt, beta_start=0.0001, beta_end=0.02, num_timesteps=1000, ddim_timesteps=50, eta=0.0)

    ldt_epochs = 1
    best_val_ldt = float('inf')
    ldt_train_hist = [] 
    ldt_val_hist = []

    for epoch in range(ldt_epochs):
        tl = train_ldt_epoch(ldt, vae, diffusion, train_loader, ldt_opt, device, diffusion.num_timesteps)
        vl = eval_ldt(ldt, vae, diffusion, val_loader, device, diffusion.num_timesteps)
        ldt_train_hist.append(tl) 
        ldt_val_hist.append(vl)
        print(f"[LDT] Epoch {epoch+1}/{ldt_epochs} train {tl:.4f} val {vl:.4f}")

        if vl < best_val_ldt:
            best_val_ldt = vl
            torch.save(ldt.state_dict(), 'dit/results/models/best_ldt.pth')

    plot_curves(ldt_train_hist, ldt_val_hist, 'LDT Diffusion Loss', 'dit/results/visualizations/ldt_loss.png')

    os.makedirs('dit/results/visualizations', exist_ok=True)
    visualize_class_conditional_samples(vae, ldt, diffusion, device, save_path='dit/results/visualizations/ldt_ddim_samples.png')

    print('Pipeline completed. Results saved under dit/results')

if __name__ == '__main__':
    main()
