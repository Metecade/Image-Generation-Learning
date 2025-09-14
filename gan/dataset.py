# dataset.py

import os
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

def get_loaders():
    print("Preparing data loaders...")

    # Define transformations for the training and validation sets
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset
    full_train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Split the training set into training and validation sets
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader