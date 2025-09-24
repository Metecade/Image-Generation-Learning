import torch
import torch.nn as nn
import torch.nn.functional as F

class DDIM:
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_timesteps=1000, ddim_timesteps=50, eta=0.0):
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

        # Get device from model
        device = next(model.parameters()).device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, batch_size, channels=4, img_size=32):
        device = next(self.model.parameters()).device
        
        skip = self.num_timesteps // self.ddim_timesteps
        sequence = list(range(0, self.num_timesteps, skip))
        sequence_next = [-1] + sequence[:-1]

        x_t = torch.randn(batch_size, channels, img_size, img_size, device=device)

        for i, j in zip(reversed(sequence), reversed(sequence_next)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)

            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_prev = self.alphas_cumprod[j].view(-1, 1, 1, 1) if j >= 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)

            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            pred_noise = self.model(x_t, t)

            x_0_pred = (x_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            dir_xt = torch.sqrt(1.0 - alpha_prev - self.eta ** 2 * beta) * pred_noise
            noise = self.eta * torch.sqrt(beta) * torch.randn_like(x_t) if j > 0 else 0.0

            x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + noise

        return x_t
    
    def compute_loss(self, x_0):
        batch_size = x_0.size(0)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)

        pred_noise = self.model(x_t, t)

        loss = F.mse_loss(noise, pred_noise)

        return loss