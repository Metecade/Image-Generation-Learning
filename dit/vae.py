# vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.act(h + self.shortcut(x))
        return h

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, out_channels=4):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.initial_conv = nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)

        self.resblock1 = ResidualBlock(latent_dim, latent_dim)
        self.downsample1 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1)

        self.resblock2 = ResidualBlock(latent_dim, latent_dim * 2)
        self.downsample2 = nn.Conv2d(latent_dim * 2, latent_dim * 2, kernel_size=3, stride=2, padding=1)

        self.resblock3 = ResidualBlock(latent_dim * 2, latent_dim * 4)
        self.downsample3 = nn.Conv2d(latent_dim * 4, latent_dim * 4, kernel_size=3, stride=2, padding=1)

        self.resblock4 = ResidualBlock(latent_dim * 4, latent_dim * 4)

        self.conv_mu = nn.Conv2d(latent_dim * 4, out_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(latent_dim * 4, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.initial_conv(x)

        h = self.resblock1(h)
        h = self.downsample1(h)

        h = self.resblock2(h)
        h = self.downsample2(h)

        h = self.resblock3(h)
        h = self.downsample3(h)

        h = self.resblock4(h)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)

        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=64, in_channels=4):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.initial_conv = nn.Conv2d(in_channels, latent_dim * 4, kernel_size=3, padding=1)

        self.resblock1 = ResidualBlock(latent_dim * 4, latent_dim * 4)
        self.upsample1 = nn.ConvTranspose2d(latent_dim * 4, latent_dim * 4, kernel_size=4, stride=2, padding=1)

        self.resblock2 = ResidualBlock(latent_dim * 4, latent_dim * 2)
        self.upsample2 = nn.ConvTranspose2d(latent_dim * 2, latent_dim * 2, kernel_size=4, stride=2, padding=1)

        self.resblock3 = ResidualBlock(latent_dim * 2, latent_dim)
        self.upsample3 = nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1)

        self.resblock4 = ResidualBlock(latent_dim, latent_dim)

        self.final_conv = nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1)
        self.act = nn.Tanh()

    def forward(self, z):
        h = self.initial_conv(z)

        h = self.resblock1(h)
        h = self.upsample1(h)

        h = self.resblock2(h)
        h = self.upsample2(h)

        h = self.resblock3(h)
        h = self.upsample3(h)

        h = self.resblock4(h)

        x_recon = self.act(self.final_conv(h))

        return x_recon
    

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, out_channels=4):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, out_channels)
        self.decoder = Decoder(in_channels, latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss
