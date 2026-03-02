"""
train_diffusion.py  –  Phase-3 Diffusion Model Training
========================================================
Physics-informed DDPM on Sentinel-3 SRAL Level-1B altimeter data.

Features
─────────────────────────────────────────────────────────────────────
• UNet1D denoiser with AdaGN time/cond injection
• Classifier-free guidance (10% null-cond dropout)
• EMA weight averaging  (decay=0.9999)
• Physics reference stats computed from training set
• Validation: mean DDPM loss + generated waveform visualisation
• Checkpoints include EMA weights + norm_stats + physics_stats

Usage
─────────────────────────────────────────────────────────────────────
python diffusion/train_diffusion.py \\
    --nc_path  data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir  ./runs/diffusion_phase3 \\
    --epochs   300 \\
    --batch    64  \\
    --lr       2e-4

python diffusion/train_diffusion.py \\
    --nc_path  data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir  ./runs/diffusion_phase3 \\
    --epochs   300 \\
    --batch    64  \\
    --lr       2e-4 \\
    --resume   ./runs/diffusion_phase3/checkpoints/ckpt_epoch_0099.pt
─────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset          import SentinelDataset
from model            import SentinelConditioner
from diffusion.model_diffusion  import UNet1D, EMA
from diffusion.noise_schedule   import DDPMSchedule, DDIMSampler, compute_reference_stats
from diffusion.vis_diffusion    import save_diffusion_vis


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def save_ckpt(ckpt_dir, epoch, step, model, conditioner, opt,
              ema, norm_stats, physics_stats):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch":          epoch,
        "step":           step,
        "model_state":    model.state_dict(),
        "cond_state":     conditioner.state_dict(),
        "opt_state":      opt.state_dict(),
        "ema_shadow":     ema.shadow,
        "norm_stats":     norm_stats,
        "physics_stats":  {k: v.cpu() for k, v in physics_stats.items()},
    }, path)
    print(f"[Ckpt] Saved → {path}")
    return path


def load_ckpt(path, model, conditioner, opt, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    conditioner.load_state_dict(ckpt["cond_state"])
    opt.load_state_dict(ckpt["opt_state"])
    for k, v in ckpt["ema_shadow"].items():
        ema.shadow[k] = v.to(device)
    norm_stats   = ckpt.get("norm_stats", None)
    physics_stats = {k: v.to(device) for k, v in ckpt.get("physics_stats", {}).items()}
    print(f"[Ckpt] Loaded epoch {ckpt['epoch']}, step {ckpt['step']} from {path}")
    return ckpt["epoch"], ckpt["step"], norm_stats, physics_stats


# ──────────────────────────────────────────────────────────────────────────────
# Cosine annealing with warm restarts
# ──────────────────────────────────────────────────────────────────────────────

def warmup_cosine_schedule(opt, step, warmup_steps, total_steps, lr_max, lr_min=1e-6):
    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diffusion Train] device={device}")

    ckpt_dir  = os.path.join(args.out_dir, "checkpoints")
    vis_dir   = os.path.join(args.out_dir, "vis")
    stats_path = os.path.join(args.out_dir, "norm_stats.json")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    train_ds = SentinelDataset(
        args.nc_path, split="train",
        val_frac=0.10, test_frac=0.10,
        stats_path=stats_path,
    )
    norm_stats = train_ds.norm_stats

    val_ds = SentinelDataset(
        args.nc_path, split="val",
        val_frac=0.10, test_frac=0.10,
        norm_stats=norm_stats,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── Physics reference statistics ──────────────────────────────────────
    print("[Diffusion Train] Computing physics reference stats …")
    all_wfm = train_ds.waveform.squeeze() if train_ds.waveform.ndim == 3 else train_ds.waveform
    # waveform is (N, 128), already normalised
    physics_stats = compute_reference_stats(all_wfm)
    physics_stats = {k: v.to(device) for k, v in physics_stats.items()}
    print(f"  ref_psd shape={physics_stats['ref_psd'].shape}, "
          f"ref_lap shape={physics_stats['ref_lap'].shape}")

    # ── Models ────────────────────────────────────────────────────────────
    model       = UNet1D(in_ch=1, base_ch=64, t_emb_dim=128, cond_dim=args.emb_dim).to(device)
    conditioner = SentinelConditioner(num_surf_types=3, emb_dim=args.emb_dim).to(device)
    ema         = EMA(model, decay=0.9999)
    schedule    = DDPMSchedule(T=args.T, device=str(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Diffusion Train] UNet1D params: {n_params/1e6:.2f}M")

    # ── Optimiser ─────────────────────────────────────────────────────────
    all_params = list(model.parameters()) + list(conditioner.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.999),
                             weight_decay=1e-4)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    if args.resume:
        start_epoch, global_step, ns, ps = load_ckpt(
            args.resume, model, conditioner, opt, ema, device
        )
        if ns:
            norm_stats = ns
        if ps:
            physics_stats.update(ps)
        start_epoch += 1

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = min(2000, total_steps // 20)

    # ── DDIM sampler for vis (fast, 50 steps) ────────────────────────────
    sampler = DDIMSampler(
        schedule,
        ddim_steps=50,
        eta=0.0,
        cfg_scale=args.cfg_scale,
        physics={"ref_psd": physics_stats["ref_psd"],
                 "ref_lap": physics_stats["ref_lap"],
                 "psd_strength": 0.3,
                 "lap_strength": 0.2},
    )

    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}

    print(f"[Diffusion Train] Training for {args.epochs} epochs, "
          f"{total_steps} steps …")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        conditioner.train()

        epoch_loss = 0.0
        n_batches  = 0

        for batch_idx, (real_sig, surf, rng, rr) in enumerate(train_loader):
            real_sig = real_sig.to(device)
            surf     = surf.to(device)
            rng      = rng.to(device)
            rr       = rr.to(device)

            # Learning-rate schedule
            warmup_cosine_schedule(opt, global_step, warmup_steps,
                                   total_steps, args.lr)

            cond_emb = conditioner(surf, rng, rr)

            # DDPM loss with classifier-free guidance dropout
            loss = schedule.training_loss(
                model, real_sig, cond_emb, cfg_prob=args.cfg_prob
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            ema.update()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % 200 == 0:
                lr_cur = opt.param_groups[0]["lr"]
                print(f"Epoch {epoch:04d} | "
                      f"Batch {batch_idx+1:05d}/{len(train_loader)} | "
                      f"Loss {loss.item():.5f} | "
                      f"LR {lr_cur:.2e} | "
                      f"Step {global_step}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation loss ───────────────────────────────────────────────
        model.eval()
        conditioner.eval()
        ema.apply_shadow()

        val_losses = []
        with torch.no_grad():
            for real_v, surf_v, rng_v, rr_v in val_loader:
                real_v = real_v.to(device)
                surf_v = surf_v.to(device)
                rng_v  = rng_v.to(device)
                rr_v   = rr_v.to(device)
                cond_v = conditioner(surf_v, rng_v, rr_v)
                vl = schedule.training_loss(model, real_v, cond_v, cfg_prob=0.0)
                val_losses.append(vl.item())

        avg_val_loss = float(np.mean(val_losses))
        print(f"[Epoch {epoch:04d}] train_loss={avg_train_loss:.5f}  "
              f"val_loss={avg_val_loss:.5f}")

        # ── Visualisation (every 5 epochs) ───────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                vis_surf = torch.zeros(1, dtype=torch.long, device=device)
                vis_rng  = torch.zeros(1, dtype=torch.float32, device=device)
                vis_rr   = torch.zeros(1, dtype=torch.float32, device=device)
                vis_cond = conditioner(vis_surf, vis_rng, vis_rr)

                fake_sig = sampler.sample(
                    model, vis_cond,
                    shape=(1, 1, 128), device=str(device)
                )

            # Grab a real sample for comparison
            real_ref = train_ds[0][0].unsqueeze(0).to(device)

            save_diffusion_vis(
                real_wfm=real_ref[0],
                fake_wfm=fake_sig[0],
                epoch=epoch,
                out_dir=vis_dir,
                surf_label=surf_names[0],
                val_loss=avg_val_loss,
            )

        ema.restore()

        # ── Checkpoint ───────────────────────────────────────────────────
        save_ckpt(
            ckpt_dir, epoch, global_step,
            model, conditioner, opt, ema,
            norm_stats, physics_stats,
        )

    print("[Diffusion Train] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Physics-Informed Diffusion Model – Sentinel-3 SRAL"
    )
    p.add_argument("--nc_path",     required=True,  help="Path to measurement.nc")
    p.add_argument("--out_dir",     default="./runs/diffusion_phase3")
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--emb_dim",     type=int,   default=32,
                   help="Conditioning embedding dimension")
    p.add_argument("--T",           type=int,   default=1000,
                   help="DDPM total timesteps")
    p.add_argument("--cfg_scale",   type=float, default=2.0,
                   help="Classifier-free guidance scale (1=disabled)")
    p.add_argument("--cfg_prob",    type=float, default=0.10,
                   help="Null-cond dropout prob during training")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      default="", help="Resume from checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("Phase-3: Physics-Informed Diffusion Model")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)
    train(args)
