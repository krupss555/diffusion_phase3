"""
model_diffusion.py  –  Physics-Informed 1D UNet for DDPM
=========================================================
Denoiser backbone for Sentinel-3 SRAL 128-sample power waveforms.

Architecture
─────────────────────────────────────────────────────────────────
Input  : (B, 1, 128)  noisy waveform  +  timestep t  +  cond_emb
Output : (B, 1, 128)  predicted noise ε

UNet encoder/decoder (base_ch=64):
  Resolution  128  →  64  →  32  →  16  (bottleneck, with attention)
  Channels    64  →  128  → 256  → 256

Time embedding  : sinusoidal(128) → MLP → t_hidden=512
Cond injection  : SentinelConditioner output projected to t_hidden,
                  added to time embedding (classifier-free guidance ready)
Block type      : ResBlock1D with AdaGN (scale+shift from t_emb)
Attention       : Multi-head self-attention at 16 and 32 samples

Reference: Li et al. (2025) Physics-Informed Diffusion Model for
           Complex-Valued Radar Sea Clutter Generation, IEEE SPL
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Positional / Time Embedding
# ──────────────────────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    timesteps : (B,) long    →    (B, dim) float
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                         device=timesteps.device) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# ──────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Residual block with AdaGN (Adaptive Group Normalisation).
    Time + conditioning is injected as per-channel scale & shift on norm2.
    """

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

        # AdaGN: t_emb → scale+shift for norm2
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch * 2),
        )

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # AdaGN scale + shift
        ts = self.t_proj(t_emb)              # (B, 2*out_ch)
        scale, shift = ts.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)           # (B, out_ch, 1)
        shift = shift.unsqueeze(-1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the time axis (batch_first)."""

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = self.norm(x).permute(0, 2, 1)      # (B, T, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)           # (B, C, T)


class Downsample1D(nn.Module):
    """Stride-2 convolution: T → T//2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Transposed stride-2 convolution: T → T*2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# UNet
# ──────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    1D UNet denoiser for DDPM.

    Signal length    : 128 samples
    Resolutions      : 128 → 64 → 32 → 16 (bottleneck)
    Channels         : base_ch=64 → 128 → 256 → 256
    Attention at     : 32 and 16 samples
    Time/cond emb    : t_emb_dim=128 sinusoidal → MLP → t_hidden=512
    Classifier-free  : pass cond_emb=None to use null embedding (zeros)

    Parameters
    ----------
    in_ch     : 1  (single-channel power waveform)
    base_ch   : 64
    t_emb_dim : 128   (sinusoidal embedding dimension)
    cond_dim  : 32    (SentinelConditioner output dim)
    """

    def __init__(self,
                 in_ch:     int = 1,
                 base_ch:   int = 64,
                 t_emb_dim: int = 128,
                 cond_dim:  int = 32):
        super().__init__()

        self.t_emb_dim = t_emb_dim
        t_hidden = t_emb_dim * 4   # 512

        # ── Time + Cond projection ────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_hidden),
            nn.SiLU(),
            nn.Linear(t_hidden, t_hidden),
        )
        # Null embedding for classifier-free guidance
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.cond_proj = nn.Linear(cond_dim, t_hidden)

        c = base_ch   # 64

        # ── Input projection ─────────────────────────────────────────────
        self.enc_in = nn.Conv1d(in_ch, c, 3, padding=1)   # (B, 64, 128)

        # ── Encoder ──────────────────────────────────────────────────────
        # Level 0: 128 samples, 64 ch
        self.enc0_r0 = ResBlock1D(c,   c,   t_hidden)
        self.enc0_r1 = ResBlock1D(c,   c,   t_hidden)
        self.down0   = Downsample1D(c)

        # Level 1: 64 samples, 128 ch
        self.enc1_r0 = ResBlock1D(c,   c*2, t_hidden)
        self.enc1_r1 = ResBlock1D(c*2, c*2, t_hidden)
        self.down1   = Downsample1D(c*2)

        # Level 2: 32 samples, 256 ch  + attention
        self.enc2_r0 = ResBlock1D(c*2, c*4, t_hidden)
        self.enc2_r1 = ResBlock1D(c*4, c*4, t_hidden)
        self.enc2_at = SelfAttention1D(c*4)
        self.down2   = Downsample1D(c*4)

        # ── Bottleneck: 16 samples, 256 ch  + attention ──────────────────
        self.mid_r0 = ResBlock1D(c*4, c*4, t_hidden)
        self.mid_at = SelfAttention1D(c*4)
        self.mid_r1 = ResBlock1D(c*4, c*4, t_hidden)

        # ── Decoder ──────────────────────────────────────────────────────
        # Level 2: 16→32, skip from enc2 (256 ch)
        self.up2     = Upsample1D(c*4)
        self.dec2_r0 = ResBlock1D(c*4 + c*4, c*4, t_hidden)
        self.dec2_r1 = ResBlock1D(c*4,        c*2, t_hidden)
        self.dec2_at = SelfAttention1D(c*2)

        # Level 1: 32→64, skip from enc1 (128 ch)
        self.up1     = Upsample1D(c*2)
        self.dec1_r0 = ResBlock1D(c*2 + c*2, c*2, t_hidden)
        self.dec1_r1 = ResBlock1D(c*2,        c,   t_hidden)

        # Level 0: 64→128, skip from enc0 (64 ch)
        self.up0     = Upsample1D(c)
        self.dec0_r0 = ResBlock1D(c + c, c, t_hidden)
        self.dec0_r1 = ResBlock1D(c,     c, t_hidden)

        # ── Output ────────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, c)
        self.out_conv = nn.Conv1d(c, in_ch, 3, padding=1)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self,
                x:        torch.Tensor,
                t:        torch.Tensor,
                cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        x        : (B, 1, 128)  noisy waveform
        t        : (B,)         long timestep indices
        cond_emb : (B, 32)      from SentinelConditioner; None → null cond
        returns  : (B, 1, 128)  predicted noise ε
        """
        B = x.shape[0]

        # Time embedding
        t_emb = get_timestep_embedding(t, self.t_emb_dim)  # (B, 128)
        t_emb = self.time_mlp(t_emb)                        # (B, 512)

        # Conditioning (classifier-free guidance: null if None)
        if cond_emb is None:
            cond_emb = self.null_cond.unsqueeze(0).expand(B, -1)
        t_emb = t_emb + self.cond_proj(cond_emb)           # (B, 512)

        # Encoder
        h   = self.enc_in(x)                               # (B, 64, 128)
        h0  = self.enc0_r1(self.enc0_r0(h,  t_emb), t_emb)  # (B, 64, 128)

        h1  = self.down0(h0)                                # (B, 64,  64)
        h1  = self.enc1_r0(h1, t_emb)                      # (B,128,  64)
        h1  = self.enc1_r1(h1, t_emb)

        h2  = self.down1(h1)                                # (B,128,  32)
        h2  = self.enc2_r0(h2, t_emb)                      # (B,256,  32)
        h2  = self.enc2_r1(h2, t_emb)
        h2  = self.enc2_at(h2)

        # Bottleneck
        hb  = self.down2(h2)                                # (B,256,  16)
        hb  = self.mid_r0(hb, t_emb)
        hb  = self.mid_at(hb)
        hb  = self.mid_r1(hb, t_emb)

        # Decoder
        d2 = torch.cat([self.up2(hb), h2], dim=1)          # (B,512,  32)
        d2 = self.dec2_r0(d2, t_emb)                        # (B,256,  32)
        d2 = self.dec2_r1(d2, t_emb)                        # (B,128,  32)
        d2 = self.dec2_at(d2)

        d1 = torch.cat([self.up1(d2), h1], dim=1)          # (B,256,  64)
        d1 = self.dec1_r0(d1, t_emb)                        # (B,128,  64)
        d1 = self.dec1_r1(d1, t_emb)                        # (B, 64,  64)

        d0 = torch.cat([self.up0(d1), h0], dim=1)          # (B,128, 128)
        d0 = self.dec0_r0(d0, t_emb)                        # (B, 64, 128)
        d0 = self.dec0_r1(d0, t_emb)

        out = F.silu(self.out_norm(d0))
        return self.out_conv(out)                            # (B,  1, 128)


# ──────────────────────────────────────────────────────────────────────────────
# EMA wrapper
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a shadow copy updated with:  shadow = decay * shadow + (1-decay) * param
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Copy shadow weights → model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()
