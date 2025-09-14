# gannet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 20
        self.output_channels = 1

        self.fc = nn.Linear(self.latent_dim, 128 * 4 * 4)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, self.output_channels, kernel_size=4, stride=2, padding=3, output_padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.model(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_channels = 1

        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        x = self.model(x) 
        return x
    


def gan_loss(output, target_is_real):
    """
    GAN损失函数
    Args:
        output: 判别器输出
        target_is_real: 目标是否为真实样本 (布尔值)
    """
    if target_is_real:
        target = torch.ones_like(output)
    else:
        target = torch.zeros_like(output)
    
    bce = F.binary_cross_entropy(output, target)
    return bce