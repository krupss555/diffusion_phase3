"""
noise_schedule.py  –  DDPM Noise Schedule + DDIM Sampler
=========================================================
Implements:
  • Linear beta schedule (Ho et al. 2020)
  • Forward diffusion  q(x_t | x_0)
  • DDPM training loss
  • DDIM deterministic sampling (Song et al. 2021)  with physics-guided priors

Physics guidance (Li et al. 2025):
  1. PSD guidance      : align generated PSD with training reference
  2. Smoothness        : second-derivative (Laplacian) regularisation
  3. Waveform clipping : keep tanh range [-1, 1]
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple  # <--- Added for compatibility


# ──────────────────────────────────────────────────────────────────────────────
# Noise Schedule
# ──────────────────────────────────────────────────────────────────────────────

class DDPMSchedule:
    """
    Linear beta schedule.

    T          : total diffusion steps (default 1000)
    beta_start : smallest noise variance (default 1e-4)
    beta_end   : largest noise variance  (default 0.02)
    """

    def __init__(self,
                 T:          int   = 1000,
                 beta_start: float = 1e-4,
                 beta_end:   float = 0.02,
                 device:     str   = "cpu"):
        self.T      = T
        self.device = device

        betas      = torch.linspace(beta_start, beta_end, T, device=device)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Append a leading 1 for alpha_bar[t-1] convenience
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])

        self.betas                    = betas
        self.alphas                   = alphas
        self.alpha_bars               = alpha_bars
        self.alpha_bars_prev          = alpha_bars_prev
        self.sqrt_alpha_bars          = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars= (1.0 - alpha_bars).sqrt()
        self.log_one_minus_alpha_bars = (1.0 - alpha_bars).log()
        self.sqrt_recip_alpha_bars    = (1.0 / alpha_bars).sqrt()
        self.sqrt_recip_m1_alpha_bars = (1.0 / alpha_bars - 1.0).sqrt()

    def to(self, device):
        """Move all buffers to device."""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        self.device = device
        return self

    # ── Forward Process ───────────────────────────────────────────────────

    def q_sample(self,
                 x0:    torch.Tensor,
                 t:     torch.Tensor,
                 noise: Optional[torch.Tensor] = None # <--- FIXED
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0).
        x0    : (B, 1, 128)
        t     : (B,)  long  timestep indices  0 ≤ t < T
        Returns (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ── DDPM Loss ─────────────────────────────────────────────────────────

    def training_loss(self,
                      model,
                      x0:       torch.Tensor,
                      cond_emb: torch.Tensor,
                      cfg_prob: float = 0.1
                      ) -> torch.Tensor:
        """
        Compute DDPM noise-prediction MSE loss with classifier-free guidance.

        cfg_prob : probability of dropping conditioning (null cond) per sample
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)

        # Classifier-free guidance: randomly drop conditioning
        if cfg_prob > 0.0:
            drop_mask = torch.rand(B, device=x0.device) < cfg_prob
            cond_input = cond_emb.clone()
            cond_input[drop_mask] = 0.0   # null embedding → model handles via null_cond param
            # Flag null samples by passing None is handled externally; here we
            # just zero them out and the model adds its learned null_cond
        else:
            cond_input = cond_emb

        eps_pred = model(x_t, t, cond_input)
        return F.mse_loss(eps_pred, noise)

    # ── Predict x0 from x_t and predicted noise ───────────────────────────

    def predict_x0(self,
                   x_t:      torch.Tensor,
                   t:        torch.Tensor,
                   eps_pred: torch.Tensor
                   ) -> torch.Tensor:
        """
        x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)
        """
        sqrt_ab   = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Physics-Guided Prior Functions
# ──────────────────────────────────────────────────────────────────────────────

def psd_guidance_grad(x0_pred: torch.Tensor,
                       ref_psd: torch.Tensor,
                       strength: float = 0.5) -> torch.Tensor:
    """
    PSD guidance gradient.
    Pushes generated PSD towards the training-set mean PSD.

    x0_pred : (B, 1, 128)  predicted clean waveform
    ref_psd : (N_freq,)     mean PSD of training set (rfft magnitude²)
    returns : (B, 1, 128)   gradient correction Δx̂_0
    """
    x = x0_pred.squeeze(1)                  # (B, 128)
    x_req = x.detach().requires_grad_(True)

    spec = torch.fft.rfft(x_req, dim=-1)    # (B, 65) complex
    psd  = spec.real**2 + spec.imag**2      # (B, 65)

    ref = ref_psd.to(x0_pred.device)
    loss = ((psd - ref.unsqueeze(0)) ** 2).mean()
    loss.backward()

    grad = x_req.grad                        # (B, 128)
    return -strength * grad.unsqueeze(1)     # (B, 1, 128)


def smoothness_guidance_grad(x0_pred: torch.Tensor,
                              ref_lap: torch.Tensor,
                              strength: float = 0.3) -> torch.Tensor:
    """
    Laplacian smoothness guidance.
    Penalises deviation from the mean Laplacian of the training set.

    x0_pred : (B, 1, 128)
    ref_lap : (128,) or (1, 128)  mean Laplacian of training waveforms
    """
    x = x0_pred.squeeze(1)                  # (B, 128)
    x_req = x.detach().requires_grad_(True)

    # Second-order finite difference: lap[i] ≈ x[i+1] - 2x[i] + x[i-1]
    lap = x_req[:, 2:] - 2 * x_req[:, 1:-1] + x_req[:, :-2]  # (B, 126)

    ref = ref_lap.to(x0_pred.device)
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
    ref_center = ref[:, 1:-1] if ref.shape[-1] == 128 else ref  # (1, 126)

    loss = ((lap - ref_center) ** 2).mean()
    loss.backward()

    grad = x_req.grad                        # (B, 128)
    return -strength * grad.unsqueeze(1)     # (B, 1, 128)


def compute_reference_stats(waveforms_np: np.ndarray
                              ) -> dict: # Fixed type hint
    """
    Compute physics reference statistics from training waveforms.

    waveforms_np : (N, 128)  numpy float32 (already normalised to [-1,1])
    Returns dict with 'ref_psd' (65,) and 'ref_lap' (128,) tensors
    """
    # Mean PSD via rfft
    specs  = np.fft.rfft(waveforms_np, axis=-1)
    psds   = np.abs(specs) ** 2
    ref_psd = torch.from_numpy(psds.mean(0).astype(np.float32))

    # Mean Laplacian
    laps    = waveforms_np[:, 2:] - 2 * waveforms_np[:, 1:-1] + waveforms_np[:, :-2]
    ref_lap = torch.zeros(128, dtype=torch.float32)
    ref_lap[1:127] = torch.from_numpy(laps.mean(0).astype(np.float32))

    return {"ref_psd": ref_psd, "ref_lap": ref_lap}


# ──────────────────────────────────────────────────────────────────────────────
# DDIM Sampler
# ──────────────────────────────────────────────────────────────────────────────

class DDIMSampler:
    """
    DDIM (Song et al. 2021) deterministic / stochastic sampler with optional
    classifier-free guidance and physics-guided priors.

    Parameters
    ----------
    schedule    : DDPMSchedule instance
    ddim_steps  : number of denoising steps (50 or 100 recommended)
    eta         : 0 → fully deterministic (DDIM), 1 → DDPM stochastic
    cfg_scale   : classifier-free guidance scale (1 = no guidance, 7.5 typical)
    physics     : dict with optional keys:
                    'ref_psd'      (65,) tensor
                    'ref_lap'      (128,) tensor
                    'psd_strength' float  (default 0.3)
                    'lap_strength' float  (default 0.2)
    """

    def __init__(self,
                 schedule:   DDPMSchedule,
                 ddim_steps: int   = 100,
                 eta:        float = 0.0,
                 cfg_scale:  float = 1.0,
                 physics:    dict  = None):
        self.schedule    = schedule
        self.ddim_steps  = ddim_steps
        self.eta         = eta
        self.cfg_scale   = cfg_scale
        self.physics     = physics or {}

        # Build DDIM timestep schedule (uniform spacing in [0, T-1])
        T = schedule.T
        step = max(T // ddim_steps, 1)
        ts   = list(range(0, T, step))
        # Reverse for denoising (T-1 → 0)
        self.timesteps = list(reversed(ts))

    @torch.no_grad()
    def sample(self,
               model,
               cond_emb:  torch.Tensor,
               shape:     tuple = (1, 1, 128),
               device:    str   = "cpu") -> torch.Tensor:
        """
        Generate samples starting from Gaussian noise.

        model     : trained UNet1D
        cond_emb  : (B, 32)  conditioning embedding
        shape     : (B, 1, 128)
        device    : 'cuda' or 'cpu'
        Returns   : (B, 1, 128) generated waveforms in [-1, 1]
        """
        sch = self.schedule
        B   = shape[0]

        x_t = torch.randn(*shape, device=device)

        for i, t_cur in enumerate(self.timesteps):
            t_next = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # ── Noise prediction ─────────────────────────────────────────
            eps = model(x_t, t_tensor, cond_emb)

            # Classifier-free guidance
            if self.cfg_scale != 1.0:
                null_emb = torch.zeros_like(cond_emb)
                eps_null = model(x_t, t_tensor, null_emb)
                eps = eps_null + self.cfg_scale * (eps - eps_null)

            # ── Predict x̂_0 ──────────────────────────────────────────────
            x0_pred = sch.predict_x0(x_t, t_tensor, eps)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ── Physics guidance (gradient-based) ────────────────────────
            if self.physics and t_cur > 50:   # apply only in early steps
                if "ref_psd" in self.physics:
                    psd_str = self.physics.get("psd_strength", 0.3)
                    with torch.enable_grad():
                        x0_pred = x0_pred + psd_guidance_grad(
                            x0_pred.detach(), self.physics["ref_psd"], psd_str
                        )
                if "ref_lap" in self.physics:
                    lap_str = self.physics.get("lap_strength", 0.2)
                    with torch.enable_grad():
                        x0_pred = x0_pred + smoothness_guidance_grad(
                            x0_pred.detach(), self.physics["ref_lap"], lap_str
                        )
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ── DDIM update ───────────────────────────────────────────────
            if t_next >= 0:
                ab_cur  = sch.alpha_bars[t_cur]
                ab_next = sch.alpha_bars[t_next]

                sigma = (
                    self.eta
                    * ((1 - ab_next) / (1 - ab_cur) * (1 - ab_cur / ab_next)).clamp(min=0).sqrt()
                )

                # Direction pointing to x_t
                dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps

                noise  = sigma * torch.randn_like(x_t) if self.eta > 0 else 0.0
                x_t    = ab_next.sqrt() * x0_pred + dir_xt + noise
            else:
                x_t = x0_pred

        return x_t.clamp(-1.0, 1.0)