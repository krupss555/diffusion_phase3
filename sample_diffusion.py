"""
sample_diffusion.py  –  Generate Sentinel-3 SRAL waveforms from trained model
===============================================================================
Loads a diffusion checkpoint, runs DDIM sampling and saves:
  • generated_waveforms.npy   (N, 128)  normalised [-1,1]
  • generated_metadata.npz    (surf_type, range_norm, rr_norm arrays)
  • sample_grid.png           visualisation grid

Usage
──────────────────────────────────────────────────────────────────────────────
# Generate 5000 samples (balanced across surf types by default)
python diffusion/sample_diffusion.py \\
    --ckpt       ./runs/diffusion_phase3/checkpoints/ckpt_epoch_0299.pt \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir    ./generated/diffusion \\
    --n_samples  5000 \\
    --ddim_steps 100 \\
    --cfg_scale  2.0

# Generate for a specific surface type only (0=Ocean, 1=Ice, 2=Land)
python diffusion/sample_diffusion.py \\
    --ckpt      ./runs/diffusion_phase3/checkpoints/ckpt_epoch_0299.pt \\
    --nc_path   data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir   ./generated/diffusion_ocean \\
    --n_samples 2000 \\
    --surf_type 0

# Use EMA weights (recommended, default)
# Add --no_ema to use raw model weights instead
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset         import SentinelDataset
from model           import SentinelConditioner
from diffusion.model_diffusion import UNet1D, EMA
from diffusion.noise_schedule  import DDPMSchedule, DDIMSampler, compute_reference_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_sample_grid(waveforms: np.ndarray,
                     surf_types: np.ndarray,
                     out_path:   str,
                     n_cols:     int = 5,
                     n_rows:     int = 4):
    """Save a grid of random generated waveforms."""
    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}
    N = min(n_cols * n_rows, len(waveforms))
    idx = np.random.choice(len(waveforms), N, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = axes.flatten()

    t_ax = np.arange(128)
    for i, ax in enumerate(axes[:N]):
        ax.plot(t_ax, waveforms[idx[i]], lw=1.0, color="darkorange", alpha=0.85)
        ax.set_title(surf_names.get(int(surf_types[idx[i]]), "?"), fontsize=8)
        ax.set_ylim(-1.15, 1.15)
        ax.tick_params(labelsize=6)
        ax.grid(True, ls=":", alpha=0.4)

    for ax in axes[N:]:
        ax.set_visible(False)

    fig.suptitle("Diffusion Model – Generated Waveforms", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[Sample] Grid saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Conditioning helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_balanced_conditions(n: int, device: str) -> tuple:
    """
    Create n balanced surf_type labels (0/1/2 roughly equal, or proportional).
    Range and range_rate sampled from N(0,1) (normalised domain).
    """
    surf = torch.tensor(
        [i % 3 for i in range(n)], dtype=torch.long, device=device
    )
    rng  = torch.randn(n, device=device)
    rr   = torch.randn(n, device=device)
    return surf, rng, rr


def make_fixed_conditions(n: int, surf_type_id: int, device: str) -> tuple:
    """All samples with the same surface type, varied range/rr."""
    surf = torch.full((n,), surf_type_id, dtype=torch.long, device=device)
    rng  = torch.randn(n, device=device)
    rr   = torch.randn(n, device=device)
    return surf, rng, rr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"[Sample] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    norm_stats    = ckpt.get("norm_stats", None)
    physics_stats = {k: v.to(device) for k, v in ckpt.get("physics_stats", {}).items()}

    # ── Models ────────────────────────────────────────────────────────────
    model       = UNet1D(in_ch=1, base_ch=64, t_emb_dim=128, cond_dim=args.emb_dim).to(device)
    conditioner = SentinelConditioner(num_surf_types=3, emb_dim=args.emb_dim).to(device)

    if args.no_ema:
        model.load_state_dict(ckpt["model_state"])
        print("[Sample] Using raw model weights")
    else:
        # Apply EMA weights
        ema = EMA(model, decay=0.9999)
        for k, v in ckpt["ema_shadow"].items():
            ema.shadow[k] = v.to(device)
        ema.apply_shadow()
        print("[Sample] Using EMA weights")

    conditioner.load_state_dict(ckpt["cond_state"])
    model.eval()
    conditioner.eval()

    # ── Noise schedule + sampler ──────────────────────────────────────────
    schedule = DDPMSchedule(T=args.T, device=str(device))
    sampler  = DDIMSampler(
        schedule,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        cfg_scale=args.cfg_scale,
        physics={
            **physics_stats,
            "psd_strength": args.psd_strength,
            "lap_strength": args.lap_strength,
        } if physics_stats else {},
    )

    # ── Generate in batches ───────────────────────────────────────────────
    all_wfm  = []
    all_surf = []
    all_rng  = []
    all_rr   = []

    n_done = 0
    bs     = args.batch_size

    print(f"[Sample] Generating {args.n_samples} waveforms …")

    with torch.no_grad():
        while n_done < args.n_samples:
            cur_bs = min(bs, args.n_samples - n_done)

            if args.surf_type is None:
                surf, rng, rr = make_balanced_conditions(cur_bs, str(device))
            else:
                surf, rng, rr = make_fixed_conditions(cur_bs, args.surf_type, str(device))

            cond_emb = conditioner(surf, rng, rr)
            fakes    = sampler.sample(
                model, cond_emb,
                shape=(cur_bs, 1, 128),
                device=str(device),
            )

            all_wfm.append(fakes.squeeze(1).cpu().numpy())
            all_surf.append(surf.cpu().numpy())
            all_rng.append(rng.cpu().numpy())
            all_rr.append(rr.cpu().numpy())

            n_done += cur_bs
            print(f"  {n_done}/{args.n_samples}")

    wfm_arr  = np.concatenate(all_wfm,  axis=0)  # (N, 128)
    surf_arr = np.concatenate(all_surf, axis=0)
    rng_arr  = np.concatenate(all_rng,  axis=0)
    rr_arr   = np.concatenate(all_rr,   axis=0)

    # ── Save ─────────────────────────────────────────────────────────────
    wfm_path  = os.path.join(args.out_dir, "generated_waveforms.npy")
    meta_path = os.path.join(args.out_dir, "generated_metadata.npz")
    grid_path = os.path.join(args.out_dir, "sample_grid.png")

    np.save(wfm_path, wfm_arr)
    np.savez(meta_path, surf_type=surf_arr, range_norm=rng_arr, rr_norm=rr_arr)

    print(f"[Sample] Waveforms saved → {wfm_path}  shape={wfm_arr.shape}")
    print(f"[Sample] Metadata  saved → {meta_path}")

    # Statistics
    print(f"[Sample] Generated stats: "
          f"mean={wfm_arr.mean():.4f}  "
          f"std={wfm_arr.std():.4f}  "
          f"min={wfm_arr.min():.4f}  "
          f"max={wfm_arr.max():.4f}")
    for s, name in zip([0, 1, 2], ["Ocean", "Ice", "Land"]):
        cnt = (surf_arr == s).sum()
        print(f"  surf_type {s} ({name}): {cnt} samples")

    save_sample_grid(wfm_arr, surf_arr, grid_path)

    # Save norm stats for eval scripts
    if norm_stats:
        ns_path = os.path.join(args.out_dir, "norm_stats.json")
        with open(ns_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"[Sample] Norm stats saved → {ns_path}")

    print("[Sample] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diffusion Model Sampling")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--nc_path",     required=True)
    p.add_argument("--out_dir",     default="./generated/diffusion")
    p.add_argument("--n_samples",   type=int,   default=5000)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--emb_dim",     type=int,   default=32)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--ddim_steps",  type=int,   default=100)
    p.add_argument("--eta",         type=float, default=0.0,
                   help="0=DDIM deterministic, 1=DDPM stochastic")
    p.add_argument("--cfg_scale",   type=float, default=2.0)
    p.add_argument("--psd_strength",type=float, default=0.3)
    p.add_argument("--lap_strength",type=float, default=0.2)
    p.add_argument("--surf_type",   type=int,   default=None,
                   help="0=Ocean,1=Ice,2=Land or None for balanced")
    p.add_argument("--no_ema",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Diffusion Model Sampling – Sentinel-3 SRAL")
    print("=" * 60)
    generate(args)
