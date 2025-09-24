# dataset.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

def get_loaders():
    print(f"Start Loading Dataset...")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])
    test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    print(f"Dataset Loaded Successfully!")

    return train_loader, val_loader, test_loader