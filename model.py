"""
model.py – WaveGAN with IMPROVED Sentinel Conditioner
=====================================================
IMPROVEMENTS:
- LayerNorm for stability
- Deeper continuous MLP (2→32→16)
- Dropout(0.1) for regularization
- Proper weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def phase_shuffle(x, rad=2):
    if rad == 0:
        return x
    B, C, T = x.shape
    shifts = torch.randint(-rad, rad + 1, (B,), device=x.device)
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s == 0:
            out[i] = x[i]
        elif s > 0:
            out[i, :, s:]  = x[i, :, :T - s]
            out[i, :, :s]  = x[i, :, 1:s + 1].flip(dims=[-1])
        else:
            s = abs(s)
            out[i, :, :T - s] = x[i, :, s:]
            out[i, :, T - s:] = x[i, :, T - s - 1:T - 1].flip(dims=[-1])
    return out


def lrelu(x, alpha=0.2):
    return F.leaky_relu(x, negative_slope=alpha)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, emb_dim: int = 32):
        super().__init__()
        inp_dim = latent_dim + emb_dim
        self.fc = nn.Linear(inp_dim, 4 * 256)
        self.upconv0 = nn.ConvTranspose1d(256, 128, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        self.upconv1 = nn.ConvTranspose1d(128,  64, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        self.upconv2 = nn.ConvTranspose1d( 64,   1, kernel_size=25,
                                          stride=2, padding=12, output_padding=1)

    def forward(self, z, cond_emb):
        x = torch.cat([z, cond_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4)
        x = F.relu(self.upconv0(x))
        x = F.relu(self.upconv1(x))
        x = torch.tanh(self.upconv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, emb_dim: int = 32, phaseshuffle_rad: int = 2):
        super().__init__()
        self.rad = phaseshuffle_rad
        self.cond_proj = nn.Linear(emb_dim, 128)
        self.conv0 = nn.Conv1d(  2,  64, kernel_size=25, stride=4, padding=12)
        self.conv1 = nn.Conv1d( 64, 128, kernel_size=25, stride=4, padding=12)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=12)
        self.fc = nn.Linear(2 * 256, 1)

    def forward(self, x, cond_emb):
        c = self.cond_proj(cond_emb).unsqueeze(1)
        x = torch.cat([x, c], dim=1)
        x = lrelu(self.conv0(x))
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv1(x))
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


class SentinelConditioner(nn.Module):
    """
    IMPROVED Conditioning Architecture
    
    Changes from old version:
    - Deeper continuous MLP: 2 → 32 → 16 (was 2 → 16)  
    - Added Dropout(0.1) for regularization
    - Added LayerNorm after fusion for stability
    - Proper weight initialization
    
    This addresses your concern about "just adding" the embeddings.
    Now we have:
    1. Better feature extraction from continuous vars
    2. Normalization for stable training
    3. Regularization to prevent overfitting
    """
    
    def __init__(self, num_surf_types: int = 3, emb_dim: int = 32):
        super().__init__()
        half = emb_dim // 2
        
        # Discrete branch
        self.surf_emb = nn.Embedding(num_surf_types, half)
        
        # Continuous branch - IMPROVED: deeper + dropout
        self.cont_mlp = nn.Sequential(
            nn.Linear(2, 32),           # Expand to 32 first
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),            # Regularization
            nn.Linear(32, half),        # Project to half
        )
        
        # Fusion - IMPROVED: added LayerNorm
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),      # Stability
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization"""
        nn.init.normal_(self.surf_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, surf_type, range_norm, rr_norm):
        disc = self.surf_emb(surf_type)                          # (B, 16)
        cont_in = torch.stack([range_norm, rr_norm], dim=1)     # (B, 2)
        cont = self.cont_mlp(cont_in)                            # (B, 16)
        fused = self.fusion(torch.cat([disc, cont], dim=1))      # (B, 32)
        return fused