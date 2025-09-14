# vaenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_channels = 1
        self.latent_dim = 20

        # Encoder
        self.encoder = nn.Sequential(
            # 1 * 28 * 28 -> 32 * 14 * 14
            nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 * 14 * 14 -> 64 * 7 * 7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 * 7 * 7 -> 128 * 4 * 4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

        self.fc_decode = nn.Linear(self.latent_dim, 128 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            # 128 * 4 * 4 -> 64 * 8 * 8 
            # output = (4-1)*2 - 2*1 + 3 + 1 = 6 - 2 + 3 + 1 = 8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 * 8 * 8 -> 32 * 16 * 16
            # output = (8-1)*2 - 2*1 + 4 + 0 = 14 - 2 + 4 = 16  
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 * 16 * 16 -> 1 * 28 * 28
            # output = (16-1)*2 - 2*2 + 4 + 0 = 30 - 4 + 4 = 30, but we want 28
            # So use padding=3: output = (16-1)*2 - 2*3 + 4 + 0 = 30 - 6 + 4 = 28
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=3, output_padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    

def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + 0.5 * kld, bce, kld
