# trainer.py

import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    bce_loss = 0.0
    kld_loss = 0.0

    loop = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)

    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)
        optimizer.zero_grad()

        recon_data, mu, logvar = model(data)
        loss, bce, kld =  loss_fn(recon_data, data, mu, logvar)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()

        loop.set_postfix(loss=running_loss/(batch_idx+1), 
                         bce=bce_loss/(batch_idx+1), 
                         kld=kld_loss/(batch_idx+1))
        
    avg_loss = running_loss / len(dataloader)
    avg_bce = bce_loss / len(dataloader)
    avg_kld = kld_loss / len(dataloader)

    return avg_loss, avg_bce, avg_kld
