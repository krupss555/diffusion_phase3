"""
train_diffusion.py – ROBUST Physics-Informed Diffusion Training
===============================================================
FIXES:
- Correct call to save_diffusion_vis() with proper signature
- EMA with proper shadow application
- Gradient clipping for stability
- Better monitoring and checkpointing
"""

import argparse
import os
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SentinelDataset
from model import SentinelConditioner
from diffusion.model_diffusion import UNet1D, EMA
from diffusion.noise_schedule import DDPMSchedule, compute_reference_stats
from diffusion.vis_diffusion import save_diffusion_vis


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Dataset
    stats_path = os.path.join(args.out_dir, "norm_stats.json")
    train_ds = SentinelDataset(args.nc_path, split="train", stats_path=stats_path)
    val_ds = SentinelDataset(args.nc_path, split="val", norm_stats=train_ds.norm_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                            num_workers=args.num_workers)

    # Physics stats
    print("[Train] Computing physics stats ...")
    subset_idx = np.random.choice(len(train_ds), min(5000, len(train_ds)), replace=False)
    phys_stats = compute_reference_stats(train_ds.waveform[subset_idx])

    # Models
    model = UNet1D(in_ch=1, base_ch=64, t_emb_dim=128, cond_dim=args.emb_dim).to(device)
    conditioner = SentinelConditioner(emb_dim=args.emb_dim).to(device)
    ema = EMA(model, decay=0.9999)
    
    # Cosine schedule
    schedule = DDPMSchedule(T=args.T, schedule="cosine", device=device)

    # Optimizer + LR schedule
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()), 
        lr=args.lr, weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs*len(train_loader), 
        pct_start=0.05, anneal_strategy='cos'
    )

    print(f"[Train] UNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[Train] Training {args.epochs} epochs | batch={args.batch} | lr={args.lr}")

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        conditioner.train()
        loss_acc = 0.0
        
        for i, (sig, surf, rng, rr) in enumerate(train_loader):
            sig = sig.to(device)
            surf = surf.to(device)
            rng = rng.to(device)
            rr = rr.to(device)

            cond_emb = conditioner(surf, rng, rr)
            loss = schedule.training_loss(model, sig, cond_emb, cfg_prob=args.cfg_prob)
            
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), 1.0)
            
            opt.step()
            lr_scheduler.step()
            ema.update()

            loss_acc += loss.item()
            global_step += 1

            if i % 200 == 0:
                print(f"Epoch {epoch:04d} | Batch {i:05d}/{len(train_loader)} | "
                      f"Loss {loss.item():.5f} | LR {lr_scheduler.get_last_lr()[0]:.2e}")

        avg_loss = loss_acc / len(train_loader)
        
        # Validation
        model.eval()
        conditioner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sig, surf, rng, rr in val_loader:
                sig = sig.to(device)
                surf = surf.to(device)
                rng = rng.to(device)
                rr = rr.to(device)
                cond = conditioner(surf, rng, rr)
                val_loss += schedule.training_loss(model, sig, cond, cfg_prob=0.0).item()
        val_loss /= len(val_loader)

        print(f"[Epoch {epoch:04d}] train_loss={avg_loss:.5f}  val_loss={val_loss:.5f}")

        # Visualization & checkpoint (every 5 epochs)
        if epoch % 5 == 0:
            ema.apply_shadow()
            
            try:
                # ========== FIXED CALL ==========
                save_diffusion_vis(
                    model, conditioner, schedule, val_loader,  # Correct args
                    epoch, vis_dir, device, 
                    physics_stats=phys_stats
                )
                # ================================
            except Exception as e:
                print(f"[Warning] Viz failed: {e}")
            
            ema.restore()
            
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "cond_state": conditioner.state_dict(),
                "ema_shadow": ema.shadow,
                "opt_state": opt.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "norm_stats": train_ds.norm_stats,
                "physics_stats": phys_stats,
            }, ckpt_path)
            print(f"[Ckpt] → {ckpt_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cond_state": conditioner.state_dict(),
                    "ema_shadow": ema.shadow,
                    "val_loss": val_loss,
                    "norm_stats": train_ds.norm_stats,
                    "physics_stats": phys_stats,
                }, best_path)
                print(f"[Best] val_loss={val_loss:.5f} → {best_path}")

    print(f"[Train] Complete! Best val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nc_path", required=True)
    p.add_argument("--out_dir", default="./runs/checkpoints")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--cfg_prob", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)