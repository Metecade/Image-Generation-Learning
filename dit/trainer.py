# trainer.py
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import Tuple
from vae import vae_loss

def train_vae_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0
    loop = tqdm(dataloader, desc='Train VAE', leave=False)

    for x, _ in loop:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)

        loss = vae_loss(x, x_recon, mu, logvar)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/num_batches:.4f}")

    return running_loss / len(dataloader)


def eval_vae(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    loop = tqdm(dataloader, desc='Eval VAE', leave=False)

    with torch.no_grad():
        for x, _ in loop:
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            loss = vae_loss(x, x_recon, mu, logvar)

            running_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/num_batches:.4f}")

    return running_loss / len(dataloader)


def train_ldt_epoch(ldt, vae, diffusion, dataloader, optimizer, device, num_timesteps: int):
    ldt.train()
    running_loss = 0.0
    num_batches = 0
    loop = tqdm(dataloader, desc='Train LDT', leave=False)

    for x, labels in loop:
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            mu, logvar = vae.encoder(x)
            z = mu  

        b = z.size(0)
        t = torch.randint(0, num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(z)

        x_t = diffusion.q_sample(z, t, noise)
        pred_noise = ldt(x_t, t, labels)

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(ldt.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/num_batches:.4f}")

    return running_loss / len(dataloader)


def eval_ldt(ldt, vae, diffusion, dataloader, device, num_timesteps: int):
    ldt.eval()
    running_loss = 0.0
    num_batches = 0
    loop = tqdm(dataloader, desc='Eval LDT', leave=False)

    with torch.no_grad():
        for x, labels in loop:
            x = x.to(device)
            labels = labels.to(device)

            mu, logvar = vae.encoder(x)
            z = mu

            b = z.size(0)
            t = torch.randint(0, num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(z)

            x_t = diffusion.q_sample(z, t, noise)
            pred_noise = ldt(x_t, t, labels)

            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            running_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/num_batches:.4f}")

    return running_loss / len(dataloader)
