"""
sample_gan.py  –  Generate GAN samples in the same format as diffusion samples
===============================================================================
Produces generated_waveforms.npy and generated_metadata.npz for eval scripts.

Usage
──────────────────────────────────────────────────────────────────────────────
python diffusion/sample_gan.py \\
    --ckpt      ./runs/sentinel_gan/checkpoints/ckpt_epoch_0199.pt \\
    --nc_path   data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir   ./generated/gan \\
    --n_samples 5000
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Generator, SentinelConditioner

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_balanced_conds(n, device):
    surf = torch.tensor([i % 3 for i in range(n)], dtype=torch.long, device=device)
    rng  = torch.randn(n, device=device)
    rr   = torch.randn(n, device=device)
    return surf, rng, rr


def save_sample_grid(waveforms, surf_types, out_path, n_cols=5, n_rows=4):
    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}
    N   = min(n_cols * n_rows, len(waveforms))
    idx = np.random.choice(len(waveforms), N, replace=False)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    axes = axes.flatten()
    t_ax = np.arange(128)
    for i, ax in enumerate(axes[:N]):
        ax.plot(t_ax, waveforms[idx[i]], lw=1.0, color="tomato")
        ax.set_title(surf_names.get(int(surf_types[idx[i]]), "?"), fontsize=8)
        ax.set_ylim(-1.15, 1.15)
        ax.tick_params(labelsize=6)
        ax.grid(True, ls=":", alpha=0.4)
    for ax in axes[N:]:
        ax.set_visible(False)
    fig.suptitle("GAN – Generated Waveforms", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[GAN Sample] Grid → {out_path}")


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)

    G           = Generator(latent_dim=args.latent, emb_dim=args.emb_dim).to(device)
    conditioner = SentinelConditioner(num_surf_types=3, emb_dim=args.emb_dim).to(device)

    G.load_state_dict(ckpt["G_state"])
    conditioner.load_state_dict(ckpt["cond_state"])
    G.eval(); conditioner.eval()

    norm_stats = ckpt.get("norm_stats", None)

    all_wfm = []; all_surf = []; all_rng = []; all_rr = []
    n_done  = 0

    with torch.no_grad():
        while n_done < args.n_samples:
            cur_bs = min(args.batch_size, args.n_samples - n_done)
            surf, rng, rr = make_balanced_conds(cur_bs, str(device))
            cond_emb = conditioner(surf, rng, rr)
            z        = torch.randn(cur_bs, args.latent, device=device)
            fakes    = G(z, cond_emb).squeeze(1).cpu().numpy()

            all_wfm.append(fakes)
            all_surf.append(surf.cpu().numpy())
            all_rng.append(rng.cpu().numpy())
            all_rr.append(rr.cpu().numpy())
            n_done += cur_bs
            print(f"  {n_done}/{args.n_samples}")

    wfm_arr  = np.concatenate(all_wfm,  0)
    surf_arr = np.concatenate(all_surf, 0)
    rng_arr  = np.concatenate(all_rng,  0)
    rr_arr   = np.concatenate(all_rr,   0)

    np.save(os.path.join(args.out_dir, "generated_waveforms.npy"), wfm_arr)
    np.savez(os.path.join(args.out_dir, "generated_metadata.npz"),
             surf_type=surf_arr, range_norm=rng_arr, rr_norm=rr_arr)
    save_sample_grid(wfm_arr, surf_arr,
                     os.path.join(args.out_dir, "sample_grid.png"))

    if norm_stats:
        with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
            json.dump(norm_stats, f, indent=2)

    print(f"[GAN Sample] {wfm_arr.shape}  mean={wfm_arr.mean():.4f}  std={wfm_arr.std():.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--nc_path",     required=True)
    p.add_argument("--out_dir",     default="./generated/gan")
    p.add_argument("--n_samples",   type=int, default=5000)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--latent",      type=int, default=100)
    p.add_argument("--emb_dim",     type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
