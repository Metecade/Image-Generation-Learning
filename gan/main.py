# main.py

import torch
from torch import optim, nn
import os

from dataset import get_loaders
from gannet import Generator, Discriminator, gan_loss
from trainer import train_epoch
from evaluate import evaluate
from visualize import plot_gan_training_curves, visualize_gan_samples, compare_real_fake

def main():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Data loaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Initialize model, optimizer
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    print("Models and optimizers initialized.")

    # 创建结果保存目录
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')

    # 记录训练历史
    g_train_history = []
    g_val_history = []
    d_train_history = []
    d_val_history = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")

        g_train_loss, d_train_loss = train_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, device, gan_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss - G: {g_train_loss:.4f}, D: {d_train_loss:.4f}")

        # 每3个epoch进行一次可视化
        visualize_flag = (epoch + 1) % 3 == 0 or epoch == 0
        g_val_loss, d_val_loss = evaluate(generator, discriminator, val_loader, device, gan_loss, visualize_flag, epoch)
        print(f"Epoch {epoch+1} Val Loss - G: {g_val_loss:.4f}, D: {d_val_loss:.4f}")

        # 记录训练历史
        g_train_history.append(g_train_loss)
        g_val_history.append(g_val_loss)
        d_train_history.append(d_train_loss)
        d_val_history.append(d_val_loss)

        # Save the best model based on generator validation loss
        if g_val_loss < best_val_loss:
            best_val_loss = g_val_loss
            torch.save(generator.state_dict(), "results/models/best_gan_generator.pth")
            torch.save(discriminator.state_dict(), "results/models/best_gan_discriminator.pth")
            print(f"Best models saved with G Val Loss: {best_val_loss:.4f}")

    # 绘制训练曲线
    plot_gan_training_curves(g_train_history, g_val_history, d_train_history, d_val_history, save_path="results/visualizations/gan_training_curves.png")

    # load best model for final evaluation on test set
    generator.load_state_dict(torch.load("results/models/best_gan_generator.pth"))
    discriminator.load_state_dict(torch.load("results/models/best_gan_discriminator.pth"))

    g_test_loss, d_test_loss = evaluate(generator, discriminator, test_loader, device, gan_loss, visualize=True, epoch="final")
    print(f"Test Loss - G: {g_test_loss:.4f}, D: {d_test_loss:.4f}")
        
if __name__ == "__main__":
    main()
