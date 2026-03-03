"""
vis_diffusion.py – FIXED visualization with matched surface types
=================================================================
KEY FIX: Randomly selects a surface type, then generates fake WITH
THAT SAME TYPE and finds a real sample WITH THAT SAME TYPE.

This fixes your bug: "u keep visualizing the fake fixed ocean class  
signal to any random real signal class"
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().squeeze()
    return np.squeeze(x)


def _stft_power(sig, win=32, hop=8):
    window = np.hanning(win)
    frames = []
    for start in range(0, len(sig) - win + 1, hop):
        frame = sig[start: start + win] * window
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    return np.array(frames).T


@torch.no_grad()
def save_diffusion_vis(model, conditioner, schedule, val_loader,
                       epoch: int, out_dir: str, device: str,
                       physics_stats: dict = None):
    """
    Generate and visualize with MATCHED surface types.
    
    NEW BEHAVIOR:
    1. Randomly pick surf_type ∈ {0, 1, 2}
    2. Find REAL sample with that type
    3. Generate FAKE with SAME type
    4. Plot them side-by-side for proper comparison
    """
    os.makedirs(out_dir, exist_ok=True)
    
    model.eval()
    conditioner.eval()
    
    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}
    
    # Get validation batch
    real_batch = next(iter(val_loader))
    real_sig, real_surf, real_rng, real_rr = real_batch
    
    # ========== KEY FIX ==========
    # Step 1: Randomly pick a surface type
    viz_surf_type = np.random.randint(0, 3)
    
    # Step 2: Find a REAL sample with this type
    mask = (real_surf == viz_surf_type).numpy()
    if mask.sum() == 0:
        # Fallback if type not in batch
        viz_surf_type = int(real_surf[0].item())
        real_idx = 0
    else:
        real_idx = np.where(mask)[0][0]
    
    real_wfm = real_sig[real_idx]  # (1, 128)
    
    # Step 3: Generate FAKE with SAME type
    surf_t = torch.tensor([viz_surf_type], dtype=torch.long, device=device)
    rng_t = torch.randn(1, device=device)
    rr_t = torch.randn(1, device=device)
    # =============================
    
    cond_emb = conditioner(surf_t, rng_t, rr_t)
    
    # Quick DDPM sampling (every 10th step for speed)
    x_t = torch.randn(1, 1, 128, device=device)
    sample_steps = list(range(schedule.T - 1, -1, -10))
    
    for t_val in sample_steps:
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        eps_pred = model(x_t, t, cond_emb)
        
        alpha_bar_t = schedule.alpha_bars[t_val]
        sqrt_ab = alpha_bar_t.sqrt()
        sqrt_1mab = (1.0 - alpha_bar_t).sqrt()
        x0_pred = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        x0_pred = x0_pred.clamp(-1, 1)
        
        if t_val > 0:
            alpha_t = schedule.alphas[t_val]
            beta_t = schedule.betas[t_val]
            alpha_bar_prev = schedule.alpha_bars_prev[t_val]
            
            coef1 = (alpha_t.sqrt() * beta_t) / (1.0 - alpha_bar_t)
            coef2 = ((1.0 - alpha_bar_prev) * (1.0 - alpha_t).sqrt()) / (1.0 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x_t
            
            sigma = beta_t.sqrt()
            noise = torch.randn_like(x_t)
            x_t = mean + sigma * noise
        else:
            x_t = x0_pred
    
    fake_wfm = x_t[0]
    
    # Plot
    real_np = _to_np(real_wfm)
    fake_np = _to_np(fake_wfm)
    
    surf_label = surf_names[viz_surf_type]
    title = f"Epoch {epoch} – Diffusion | {surf_label}"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)
    
    t_ax = np.arange(128)
    axes[0].plot(t_ax, real_np, color="steelblue", lw=1.8, alpha=0.9, label="Real")
    axes[0].plot(t_ax, fake_np, color="darkorange", lw=1.8, alpha=0.9,
                 linestyle="--", label="Generated")
    axes[0].set_title("Time Domain")
    axes[0].set_xlabel("Range Gate")
    axes[0].set_ylabel("log1p Power (norm)")
    axes[0].legend()
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].grid(True, ls=":", alpha=0.5)
    
    real_spec = _stft_power(real_np)
    fake_spec = _stft_power(fake_np)
    combined = np.concatenate([real_spec, fake_spec], axis=0)
    nf = real_spec.shape[0]
    
    im = axes[1].imshow(10 * np.log10(combined + 1e-8), aspect="auto",
                        origin="lower", cmap="plasma", interpolation="bicubic")
    axes[1].axhline(y=nf - 0.5, color="white", lw=1.5, ls="--")
    axes[1].set_title("STFT (top=Real, bottom=Gen)")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Freq Bin")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="dB")
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    
    print(f"[Vis] Epoch {epoch:04d} → {path} | {surf_label}")
    print(f"  Real: mean={real_np.mean():.4f}, std={real_np.std():.4f}")
    print(f"  Fake: mean={fake_np.mean():.4f}, std={fake_np.std():.4f}")