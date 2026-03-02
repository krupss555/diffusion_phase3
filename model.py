"""
model.py – WaveGAN adapted for Sentinel-3 SRAL 128-sample power waveforms.

Architecture (single-channel, 128 samples):

Generator
  z (100) + cond_emb (32)  →  FC  →  reshape (256, 4)
  → ConvTranspose1d s=4  →  (128, 16)
  → ConvTranspose1d s=4  →  (64,  64)
  → ConvTranspose1d s=2  →  (1,  128)   ← power waveform output, tanh

Discriminator
  Input (1, 128) waveform
  + cond projected to (1, 128) → cat → (2, 128)
  → Conv1d s=4  →  (64,  32)   + phase-shuffle
  → Conv1d s=4  →  (128,  8)   + phase-shuffle
  → Conv1d s=4  →  (256,  2)
  → flatten (512)  →  FC  →  scalar

Conditioning (mixed: discrete + continuous)
  SentinelConditioner:
    nn.Embedding(3, emb_dim//2)    for surf_type (0=Ocean, 1=Ice, 2=Land)
    MLP([2, 16, emb_dim//2])       for (range_norm, range_rate_norm)
    concat → Linear(emb_dim, emb_dim)

Weight transfer from DeepRadar checkpoints
  See transfer_weights.py for loading compatible layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def phase_shuffle(x, rad=2):
    """
    Randomly shift each sample by ±rad along the time axis (reflect-padded).
    x: (B, C, T)
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Generator  (single-channel output)
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Input:  z (B, latent_dim=100)  +  cond_emb (B, emb_dim=32)
    Output: (B, 1, 128)  log1p-normalised power waveform in [-1, 1]

    ConvTranspose1d sizes:
      kernel=25, padding=12, output_padding=3  for stride 4
      kernel=25, padding=12, output_padding=1  for stride 2
    """

    def __init__(self, latent_dim: int = 100, emb_dim: int = 32):
        super().__init__()
        inp_dim = latent_dim + emb_dim   # 132

        self.fc = nn.Linear(inp_dim, 4 * 256)

        self.upconv0 = nn.ConvTranspose1d(256, 128, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        self.upconv1 = nn.ConvTranspose1d(128,  64, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        # Output: 1 channel (power waveform, single real value per time step)
        self.upconv2 = nn.ConvTranspose1d( 64,   1, kernel_size=25,
                                          stride=2, padding=12, output_padding=1)

    def forward(self, z, cond_emb):
        """
        z        : (B, 100)
        cond_emb : (B, 32)
        returns  : (B, 1, 128)
        """
        x = torch.cat([z, cond_emb], dim=1)   # (B, 132)
        x = self.fc(x)                          # (B, 1024)
        x = x.view(x.size(0), 256, 4)          # (B, 256, 4)
        x = F.relu(self.upconv0(x))             # (B, 128, 16)
        x = F.relu(self.upconv1(x))             # (B,  64, 64)
        x = torch.tanh(self.upconv2(x))         # (B,   1, 128)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator  (single-channel input)
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Input: (B, 1, 128) waveform
    cond_emb (B, 32) projected to (B, 1, 128) → cat → (B, 2, 128)

    Spatial downsampling: 128 → 32 → 8 → 2
    Final flatten: 2 * 256 = 512 → FC → 1
    """

    def __init__(self, emb_dim: int = 32, phaseshuffle_rad: int = 2):
        super().__init__()
        self.rad = phaseshuffle_rad

        self.cond_proj = nn.Linear(emb_dim, 128)

        # 2 input channels: 1 waveform + 1 cond projection
        self.conv0 = nn.Conv1d(  2,  64, kernel_size=25, stride=4, padding=12)
        self.conv1 = nn.Conv1d( 64, 128, kernel_size=25, stride=4, padding=12)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=12)

        self.fc = nn.Linear(2 * 256, 1)   # T=2 after 3×stride-4, 256 ch

    def forward(self, x, cond_emb):
        """
        x        : (B, 1, 128)
        cond_emb : (B, 32)
        returns  : (B,)  raw WGAN logit
        """
        c = self.cond_proj(cond_emb).unsqueeze(1)   # (B, 1, 128)
        x = torch.cat([x, c], dim=1)                 # (B, 2, 128)

        x = lrelu(self.conv0(x))                      # (B,  64, 32)
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv1(x))                      # (B, 128,  8)
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv2(x))                      # (B, 256,  2)

        x = x.view(x.size(0), -1)                     # (B, 512)
        return self.fc(x).squeeze(-1)                  # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# Conditioner  –  Mixed discrete + continuous
# ──────────────────────────────────────────────────────────────────────────────

class SentinelConditioner(nn.Module):
    """
    Maps Sentinel-3 conditioning variables → embedding vector (B, emb_dim).

    Inputs
    ------
    surf_type  : (B,)  long   0=Ocean, 1=Ice, 2=Land
    range_norm : (B,)  float  z-scored range [m]
    rr_norm    : (B,)  float  z-scored radial velocity [m/s]

    Architecture
    ------------
    Discrete branch : Embedding(3, emb_dim//2)
    Continuous branch: Linear(2, 16) → ReLU → Linear(16, emb_dim//2)
    Fusion          : cat → Linear(emb_dim, emb_dim) → ReLU
    """

    def __init__(self, num_surf_types: int = 3, emb_dim: int = 32):
        super().__init__()
        half = emb_dim // 2   # 16

        # Discrete branch
        self.surf_emb = nn.Embedding(num_surf_types, half)

        # Continuous branch
        self.cont_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, half),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, surf_type, range_norm, rr_norm):
        """
        surf_type  : (B,) long
        range_norm : (B,) float32
        rr_norm    : (B,) float32
        returns    : (B, emb_dim)
        """
        disc = self.surf_emb(surf_type)                              # (B, 16)
        cont_in = torch.stack([range_norm, rr_norm], dim=1)         # (B, 2)
        cont = self.cont_mlp(cont_in)                                # (B, 16)
        fused = self.fusion(torch.cat([disc, cont], dim=1))          # (B, 32)
        return fused
