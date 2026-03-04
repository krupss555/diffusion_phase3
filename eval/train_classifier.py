"""
eval/train_classifier.py
=========================
Train a 1D ResNet classifier on Sentinel-3 SRAL waveforms (3 surface types).
The penultimate feature layer is used for FID computation.

Architecture  : 1D ResNet with 4 residual blocks
Classes       : 3  (0=Ocean, 1=Ice, 2=Land)
Training      : cross-entropy + cosine LR schedule
Saves         : classifier checkpoint (used by fid_1d.py and inception_score.py)

Usage
──────────────────────────────────────────────────────────────────────────────
python eval/train_classifier.py \\
    --nc_path  data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir  ./eval/classifier_ckpt \\
    --epochs   100 \\
    --batch    256
──────────────────────────────────────────────────────────────────────────────
"""

"""
eval/train_classifier.py
=========================
Train a 1D ResNet classifier on Sentinel-3 SRAL waveforms.
"""

import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import SentinelDataset


# ──────────────────────────────────────────────────────────────────────────────
# 1D ResNet Classifier
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock1DCls(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.GroupNorm(groups, out_ch)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))


class ResNet1DClassifier(nn.Module):
    """
    Input  : (B, 1, 128) waveform
    Output : (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 3, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),  # (B,32,64)
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # 4 blocks, doubling channels, halving temporal dim
        self.layer1 = ResBlock1DCls(32,  64,  stride=2)   # (B,64,32)
        self.layer2 = ResBlock1DCls(64,  128, stride=2)   # (B,128,16)
        self.layer3 = ResBlock1DCls(128, 256, stride=2)   # (B,256, 8)
        self.layer4 = ResBlock1DCls(256, feat_dim, stride=2) # (B,256, 4)

        self.pool  = nn.AdaptiveAvgPool1d(1)               # (B, 256, 1)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(feat_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feat_dim) feature vector before classification head."""
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).squeeze(-1)   # (B, feat_dim)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.drop(h)
        return self.fc(h)


# ─── Added Helper Function ────────────────────────────────────────────────────

def load_classifier(ckpt_path: str, device: str) -> ResNet1DClassifier:
    """Load the trained ResNet1D classifier (eval mode)."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ResNet1DClassifier(
        num_classes=ckpt.get("num_classes", 3),
        feat_dim=ckpt.get("feat_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    stats_path = os.path.join(args.out_dir, "norm_stats.json")
    train_ds = SentinelDataset(args.nc_path, split="train",
                                stats_path=stats_path)
    val_ds   = SentinelDataset(args.nc_path, split="val",
                                norm_stats=train_ds.norm_stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=4, pin_memory=True)

    model = ResNet1DClassifier(num_classes=3, feat_dim=256).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Classifier] params={n_par/1e3:.1f}K  device={device}")

    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_acc = 0.0
    best_path    = os.path.join(args.out_dir, "best_classifier.pt")

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_correct = n_total = 0

        for sig, surf, rng, rr in train_loader:
            sig  = sig.to(device)
            surf = surf.to(device)

            logits = model(sig)
            loss   = F.cross_entropy(logits, surf)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss    += loss.item() * len(sig)
            total_correct += (logits.argmax(1) == surf).sum().item()
            n_total       += len(sig)

        sched.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for sig, surf, _, _ in val_loader:
                sig  = sig.to(device)
                surf = surf.to(device)
                preds = model(sig).argmax(1)
                val_correct += (preds == surf).sum().item()
                val_total   += len(sig)

        train_acc = 100.0 * total_correct / n_total
        val_acc   = 100.0 * val_correct   / val_total
        avg_loss  = total_loss / n_total
        print(f"Epoch {epoch:04d} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "feat_dim":     model.feat_dim,
                "num_classes":  3,
            }, best_path)
            print(f"  [Best] val_acc={val_acc:.1f}% → {best_path}")

    # Also save final checkpoint
    final_path = os.path.join(args.out_dir, "final_classifier.pt")
    torch.save({
        "model_state": model.state_dict(),
        "feat_dim":    model.feat_dim,
        "num_classes": 3,
    }, final_path)
    print(f"[Classifier] Best val acc = {best_val_acc:.1f}%")
    print(f"[Classifier] Final saved → {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nc_path",  required=True)
    p.add_argument("--out_dir",  default="./eval/classifier_ckpt")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--lr",       type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print("Training 1D ResNet Classifier for IS / FID")
    print("=" * 55)
    train(args)
