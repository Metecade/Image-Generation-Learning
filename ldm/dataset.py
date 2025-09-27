# dataset.py - Dataset for Latent Diffusion Model

import os
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

def get_loaders(img_size=64, batch_size=16):
    """获取适用于LDM的数据加载器"""
    print("Preparing data loaders for Latent Diffusion Model...")

    # LDM的数据预处理
    # 由于LDM在潜在空间工作，通常使用更高分辨率的图像
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])

    # Load the dataset
    full_train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    print(f"Using MNIST dataset, resized to {img_size}x{img_size}")

    # Split the training set into training and validation sets
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # Create data loaders - 优化版本
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}, Image size: {img_size}x{img_size}")

    return train_loader, val_loader, test_loader

def get_conditional_loaders(img_size=64, batch_size=16, num_classes=10):
    """获取条件生成的数据加载器（保留类别标签）"""
    print("Preparing conditional data loaders for LDM...")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    full_train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    class_names = [str(i) for i in range(10)]

    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Conditional generation with {num_classes} classes: {class_names}")
    
    return train_loader, val_loader, test_loader, class_names

class ConditionalWrapper:
    """条件生成的包装器"""
    def __init__(self, num_classes=10, embed_dim=512):
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_embed = torch.nn.Embedding(num_classes, embed_dim)
    
    def encode_class(self, class_labels):
        """将类别标签编码为嵌入向量"""
        return self.class_embed(class_labels)

def create_latent_dataset(vae_model, dataloader, save_path, device):
    """预计算潜在表示以加速训练"""
    print("Creating latent dataset...")
    
    vae_model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            
            # 编码到潜在空间
            posterior = vae_model.encode(data)
            z = posterior.mode()
            
            latents.append(z.cpu())
            labels.append(label)
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx}/{len(dataloader)} batches")
    
    # 保存预计算的潜在表示
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    
    torch.save({
        'latents': latents,
        'labels': labels
    }, save_path)
    
    print(f"Latent dataset saved to {save_path}")
    print(f"Latent shape: {latents.shape}")
    
    return latents, labels

class LatentDataset(torch.utils.data.Dataset):
    """预计算潜在表示的数据集"""
    def __init__(self, latent_path):
        data = torch.load(latent_path, weights_only=False)
        self.latents = data['latents']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

def get_latent_loaders(latent_path, batch_size=32):
    """从预计算的潜在表示创建数据加载器"""
    dataset = LatentDataset(latent_path)
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader