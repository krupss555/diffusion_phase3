"""
vis_diffusion.py  –  Visualisation helper for the diffusion model.
Mirrors utils.py but adapted for diffusion training.
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
    return np.array(frames).T   # (freq_bins, time_frames)


def save_diffusion_vis(real_wfm, fake_wfm, epoch: int, out_dir: str,
                       surf_label: str = "", val_loss: float = None):
    """
    Save a 2-panel comparison PNG:  time-domain + STFT spectrogram.
    """
    os.makedirs(out_dir, exist_ok=True)

    real_np = _to_np(real_wfm)
    fake_np = _to_np(fake_wfm)

    title = f"Epoch {epoch} – Diffusion Waveform"
    if surf_label:
        title += f" | {surf_label}"
    if val_loss is not None:
        title += f" | val_loss={val_loss:.5f}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    t_ax = np.arange(128)
    axes[0].plot(t_ax, real_np, color="steelblue",  lw=1.8, alpha=0.9, label="Real")
    axes[0].plot(t_ax, fake_np, color="darkorange",  lw=1.8, alpha=0.9,
                 linestyle="--", label="Diffusion")
    axes[0].set_title("Time Domain (normalised)")
    axes[0].set_xlabel("Range Gate")
    axes[0].set_ylabel("log1p Power (norm)")
    axes[0].legend()
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].grid(True, ls=":", alpha=0.5)

    real_spec = _stft_power(real_np)
    fake_spec = _stft_power(fake_np)
    combined  = np.concatenate([real_spec, fake_spec], axis=0)
    nf        = real_spec.shape[0]

    im = axes[1].imshow(
        10 * np.log10(combined + 1e-8),
        aspect="auto", origin="lower", cmap="plasma",
        interpolation="bicubic",
    )
    axes[1].axhline(y=nf - 0.5, color="white", lw=1.5, ls="--")
    axes[1].set_title("STFT Power dB (top=Real, bottom=Diffusion)")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Freq Bin")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="dB")

    axes[1].text(0.02, 0.97, "Real",     transform=axes[1].transAxes,
                 color="white", fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="steelblue", alpha=0.7))
    axes[1].text(0.02, 0.52, "Diffusion", transform=axes[1].transAxes,
                 color="white", fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="darkorange", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Vis] Diffusion epoch {epoch:04d} → {path}")
