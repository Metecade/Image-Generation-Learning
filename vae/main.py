# main.py

import torch
from torch import optim, nn
import os

from dataset import get_loaders
from vaenet import VAE, vae_loss
from trainer import train_epoch
from evaluate import evaluate
from visualize import plot_training_curves, visualize_latent_space_2d

def main():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Data loaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Initialize model, optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Model and optimizer initialized.")

    # 创建结果保存目录
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    # 记录训练历史
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")

        train_loss, train_bce, train_kld = train_epoch(model, train_loader, optimizer, device, vae_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, BCE: {train_bce:.4f}, KLD: {train_kld:.4f}")

        # 每3个epoch进行一次可视化
        visualize_flag = (epoch + 1) % 3 == 0 or epoch == 0
        val_loss, val_bce, val_kld = evaluate(model, val_loader, device, vae_loss, visualize=visualize_flag, epoch=epoch+1)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, BCE: {val_bce:.4f}, KLD: {val_kld:.4f}")

        # 记录训练历史
        train_history.append((train_loss, train_bce, train_kld))
        val_history.append((val_loss, val_bce, val_kld))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/models/best_vae_model.pth")
            print(f"Best model saved with Val Loss: {best_val_loss:.4f}")

    # 绘制训练曲线
    plot_training_curves(train_history, val_history, save_path="results/visualizations/training_curves.png")

    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load("results/models/best_vae_model.pth"))

    test_loss, test_bce, test_kld = evaluate(model, test_loader, device, vae_loss, visualize=True, epoch="final")
    print(f"Test Loss: {test_loss:.4f}, BCE: {test_bce:.4f}, KLD: {test_kld:.4f}")
    
    # 如果潜在维度是2，可视化潜在空间
    if model.latent_dim == 2:
        visualize_latent_space_2d(model, test_loader, device, save_path="results/visualizations/latent_space_2d.png")

if __name__ == "__main__":
    main()
