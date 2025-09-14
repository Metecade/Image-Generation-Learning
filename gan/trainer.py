# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, loss_fn, epoch):
    generator.train()
    discriminator.train()

    generator_loss = 0.0
    discriminator_loss = 0.0

    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)

    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)
        batch_size = data.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        real_output = discriminator(data)
        d_loss_real = loss_fn(real_output, True)

        # Fake data
        noise = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        d_loss_fake = loss_fn(fake_output, False)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        fake_output = discriminator(fake_data)
        g_loss = loss_fn(fake_output, True)
        g_loss.backward()
        optimizer_G.step()

        generator_loss += g_loss.item()
        discriminator_loss += d_loss.item()

        loop.set_postfix(generator_loss=generator_loss/(batch_idx+1), 
                         discriminator_loss=discriminator_loss/(batch_idx+1))
        
    avg_g_loss = generator_loss / len(dataloader)
    avg_d_loss = discriminator_loss / len(dataloader)

    return avg_g_loss, avg_d_loss