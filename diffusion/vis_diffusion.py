"""
vis_diffusion.py - FIXED with proper DDIM sampling
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
                       epoch, out_dir, device, physics_stats=None):
    """FIXED: Uses proper DDIM sampling instead of broken quick sampling."""
    os.makedirs(out_dir, exist_ok=True)
    
    model.eval()
    conditioner.eval()
    
    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}
    
    # Get validation batch
    real_batch = next(iter(val_loader))
    real_sig, real_surf, real_rng, real_rr = real_batch
    
    # Randomly pick surface type and find matching real sample
    viz_surf_type = np.random.randint(0, 3)
    mask = (real_surf == viz_surf_type).numpy()
    if mask.sum() == 0:
        viz_surf_type = int(real_surf[0].item())
        real_idx = 0
    else:
        real_idx = np.where(mask)[0][0]
    
    real_wfm = real_sig[real_idx]
    
    # Generate fake with SAME type
    surf_t = torch.tensor([viz_surf_type], dtype=torch.long, device=device)
    rng_t = torch.randn(1, device=device)
    rr_t = torch.randn(1, device=device)
    cond_emb = conditioner(surf_t, rng_t, rr_t)
    
    # ========== FIXED: PROPER DDIM SAMPLING ==========
    # Import DDIM sampler
    from diffusion.noise_schedule import DDIMSampler
    
    sampler = DDIMSampler(
        schedule, 
        ddim_steps=50,   # 50 DDIM steps is enough for visualization
        eta=0.0,         # Deterministic
        cfg_scale=2.0    # Classifier-free guidance
    )
    
    fake_wfm = sampler.sample(
        model, cond_emb,
        shape=(1, 1, 128),
        device=device
    )[0]  # (1, 128)
    # =================================================
    
    # Convert to numpy
    real_np = _to_np(real_wfm)
    fake_np = _to_np(fake_wfm)
    
    # ========== VARIANCE CHECK ==========
    if fake_np.std() < 0.05:
        print(f"  ⚠️  WARNING: Generated std={fake_np.std():.4f} is TOO LOW!")
    # ====================================
    
    # Create visualization
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