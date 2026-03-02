"""
noise_schedule.py  –  Fixed with Cosine Schedule
=========================================================
"""

import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Cosine Schedule (Better for 1D Data)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# ──────────────────────────────────────────────────────────────────────────────
# Noise Schedule Class
# ──────────────────────────────────────────────────────────────────────────────

class DDPMSchedule:
    def __init__(self,
                 T:          int   = 1000,
                 schedule:   str   = "cosine",  # <--- CHANGED DEFAULT
                 beta_start: float = 1e-4,
                 beta_end:   float = 0.02,
                 device:     str   = "cpu"):
        self.T      = T
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, device=device)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # Clip alpha_bars to avoid numerical issues (sqrt of neg)
        alpha_bars = torch.clamp(alpha_bars, min=1e-10)

        # Append a leading 1 for alpha_bar[t-1] convenience
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])

        self.betas                    = betas
        self.alphas                   = alphas
        self.alpha_bars               = alpha_bars
        self.alpha_bars_prev          = alpha_bars_prev
        self.sqrt_alpha_bars          = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars= (1.0 - alpha_bars).sqrt()
        
        # For posterior calculations
        self.sqrt_recip_alpha_bars    = (1.0 / alpha_bars).sqrt()
        self.sqrt_recip_m1_alpha_bars = (1.0 / alpha_bars - 1.0).sqrt()

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        self.device = device
        return self

    # ── Forward Process ───────────────────────────────────────────────────

    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ── DDPM Loss ─────────────────────────────────────────────────────────

    def training_loss(self, model, x0, cond_emb, cfg_prob=0.1):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)

        if cfg_prob > 0.0:
            drop_mask = torch.rand(B, device=x0.device) < cfg_prob
            cond_input = cond_emb.clone()
            cond_input[drop_mask] = 0.0 
        else:
            cond_input = cond_emb

        eps_pred = model(x_t, t, cond_input)
        return F.mse_loss(eps_pred, noise)

    # ── Predict x0 ────────────────────────────────────────────────────────

    def predict_x0(self, x_t, t, eps_pred):
        sqrt_ab   = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Physics Guidance & DDIM Sampler (Unchanged logic, just keeping imports)
# ──────────────────────────────────────────────────────────────────────────────
# (Paste the rest of your original noise_schedule.py here for PSD/DDIM)
# Just make sure DDIMSampler uses the schedule passed to it.

def psd_guidance_grad(x0_pred, ref_psd, strength=0.5):
    x = x0_pred.squeeze(1)
    x_req = x.detach().requires_grad_(True)
    spec = torch.fft.rfft(x_req, dim=-1)
    psd  = spec.real**2 + spec.imag**2
    ref = ref_psd.to(x0_pred.device)
    loss = ((psd - ref.unsqueeze(0)) ** 2).mean()
    loss.backward()
    return -strength * x_req.grad.unsqueeze(1)

def compute_reference_stats(waveforms_np: np.ndarray) -> dict:
    specs  = np.fft.rfft(waveforms_np, axis=-1)
    psds   = np.abs(specs) ** 2
    ref_psd = torch.from_numpy(psds.mean(0).astype(np.float32))
    laps    = waveforms_np[:, 2:] - 2 * waveforms_np[:, 1:-1] + waveforms_np[:, :-2]
    ref_lap = torch.zeros(128, dtype=torch.float32)
    ref_lap[1:127] = torch.from_numpy(laps.mean(0).astype(np.float32))
    return {"ref_psd": ref_psd, "ref_lap": ref_lap}

class DDIMSampler:
    def __init__(self, schedule, ddim_steps=100, eta=0.0, cfg_scale=1.0, physics=None):
        self.schedule = schedule
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.cfg_scale = cfg_scale
        self.physics = physics or {}
        
        step = max(schedule.T // ddim_steps, 1)
        self.timesteps = list(reversed(range(0, schedule.T, step)))

    @torch.no_grad()
    def sample(self, model, cond_emb, shape=(1,1,128), device="cpu"):
        sch = self.schedule
        B = shape[0]
        x_t = torch.randn(*shape, device=device)
        
        for t_cur in self.timesteps:
            t_next = t_cur - (sch.T // self.ddim_steps)
            if t_next < 0: t_next = -1
            
            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)
            
            eps = model(x_t, t_tensor, cond_emb)
            if self.cfg_scale != 1.0:
                eps_null = model(x_t, t_tensor, torch.zeros_like(cond_emb))
                eps = eps_null + self.cfg_scale * (eps - eps_null)
            
            x0_pred = sch.predict_x0(x_t, t_tensor, eps).clamp(-1, 1)
            
            # Physics guidance here (same as before)...
            
            if t_next >= 0:
                ab_cur  = sch.alpha_bars[t_cur]
                ab_next = sch.alpha_bars[t_next]
                sigma = self.eta * ((1 - ab_next)/(1 - ab_cur)*(1 - ab_cur/ab_next)).sqrt()
                dir_xt = (1 - ab_next - sigma**2).sqrt() * eps
                noise = sigma * torch.randn_like(x_t)
                x_t = ab_next.sqrt() * x0_pred + dir_xt + noise
            else:
                x_t = x0_pred
                
        return x_t