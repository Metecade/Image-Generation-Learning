import torch
import torch.nn as nn
import torch.nn.functional as F

from dit import DiT


class PatchfyEmbed(nn.Module):
    def __init__(self, channels=4, patch_size=2, img_size=32, hidden_size=768):
        super(PatchfyEmbed, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.proj = nn.Linear(channels * patch_size * patch_size, hidden_size)

        self.pos = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, hidden_size))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        h = height // self.patch_size
        w = width // self.patch_size

        x = x.view(batch_size, channels, h, self.patch_size, w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(batch_size, h * w, self.patch_size * self.patch_size * channels)

        x = self.proj(x)

        x = x + self.pos

        return x
    

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=512, time_embed_dim=512):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim

        self.fc1 = nn.Linear(embed_dim, time_embed_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        emb = self.fc1(emb)
        emb = self.act(emb)
        emb = self.fc2(emb)

        return emb
    

class UnPatchfy(nn.Module):
    def __init__(self, channels=4, patch_size=2, img_size=32, hidden_size=768):
        super(UnPatchfy, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.channels = channels
        self.img_size = img_size

        self.ln = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, channels * patch_size * patch_size)

    def forward(self, x):
        batch_size, num_patches, hidden_size = x.shape
        h = self.img_size // self.patch_size
        w = self.img_size // self.patch_size

        x = self.ln(x)
        x = self.proj(x)

        x = x.view(batch_size, h, w, self.patch_size, self.patch_size, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.channels, h * self.patch_size, w * self.patch_size)

        return x
    

class LDT(nn.Module):
    def __init__(self, channels=4, patch_size=2, img_size=32, hidden_size=768, time_embed_dim=512, cond_embed_dim=512, num_layers=12, num_heads=8, ffn_ratio=4, dropout=0.1):
        super(LDT, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout

        self.patchfy = PatchfyEmbed(channels=channels, patch_size=patch_size, img_size=img_size, hidden_size=hidden_size)
        self.time_embed = TimeEmbedding(embed_dim=time_embed_dim, time_embed_dim=cond_embed_dim)
        self.cond_embed = nn.Embedding(10, cond_embed_dim)

        self.dit = DiT(embed_dim=hidden_size, cond_dim=cond_embed_dim, depth=num_layers, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        self.unpatchfy = UnPatchfy(channels=channels, patch_size=patch_size, img_size=img_size, hidden_size=hidden_size)

    def forward(self, x, t, cond):
        x = self.patchfy(x)

        t = self.time_embed(t)

        cond = self.cond_embed(cond)
        cond = cond + t

        x = self.dit(x, cond)

        x = self.unpatchfy(x)

        return x