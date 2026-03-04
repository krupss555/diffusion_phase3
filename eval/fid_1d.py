"""
eval/fid_1d.py
==============
Fréchet Inception Distance (FID) adapted for 1D radar waveforms.

Instead of the 2D ImageNet Inception-v3 used in standard image FID,
we use the penultimate layer of the trained ResNet1DClassifier as
feature extractor (256-dimensional features).

Theory  (Heusel et al. 2017)
─────────────────────────────────────────────────────────────────────
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})

where (μ_r, Σ_r) are the mean/cov of real features,
      (μ_g, Σ_g) are the mean/cov of generated features.

Lower FID  ←→  distributions are more similar.

Usage (standalone)
──────────────────────────────────────────────────────────────────────
python eval/fid_1d.py \\
    --real_npy   data/real_waveforms.npy \\
    --fake_npy   generated/diffusion/generated_waveforms.npy \\
    --cls_ckpt   eval/classifier_ckpt/best_classifier.pt \\
    --device     cuda

python eval/fid_1d.py \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --fake_npy   generated/diffusion/generated_waveforms.npy \\
    --cls_ckpt   eval/classifier_ckpt/best_classifier.pt
──────────────────────────────────────────────────────────────────────
"""


"""
eval/fid_1d.py
==============
Fréchet Inception Distance (FID) adapted for 1D radar waveforms.
"""

"""
eval/fid_1d.py
==============
Fréchet Inception Distance (FID) adapted for 1D radar waveforms.
"""

"""
eval/fid_1d.py
==============
Fréchet Inception Distance (FID) adapted for 1D radar waveforms.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.train_classifier import ResNet1DClassifier, load_classifier # <--- IMPORT FIXED


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(waveforms: np.ndarray,
                     model:     ResNet1DClassifier,
                     device:    str,
                     batch_size: int = 512) -> np.ndarray:
    """
    Extract feature vectors from waveforms using the classifier backbone.
    """
    all_feats = []
    for i in range(0, len(waveforms), batch_size):
        batch = torch.from_numpy(waveforms[i: i + batch_size]).unsqueeze(1).to(device)
        feats = model.features(batch)          # (bs, feat_dim)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# FID Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_activation_statistics(features: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of feature matrix."""
    mu  = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    return mu, cov


def fid_from_stats(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """
    Compute FID from pre-computed mean and covariance.
    Handles numerical instabilities in matrix square root.
    """
    diff  = mu1 - mu2
    diff2 = diff @ diff

    # Matrix sqrt of Σ_r @ Σ_g
    cov_product = sigma1 @ sigma2
    sqrt_cov, _ = sqrtm(cov_product, disp=False)

    # Numerical clean-up
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_cov)
    fid_score  = float(diff2 + trace_term)
    return fid_score


def compute_fid(real_wfm:   np.ndarray,
                fake_wfm:   np.ndarray,
                cls_ckpt:   str,
                device:     str = "cpu",
                batch_size: int = 512) -> dict:
    """
    End-to-end FID computation.
    """
    model = load_classifier(cls_ckpt, device)

    print(f"[FID] Extracting real features …  n={len(real_wfm)}")
    real_feats = extract_features(real_wfm, model, device, batch_size)
    print(f"[FID] Extracting fake features …  n={len(fake_wfm)}")
    fake_feats = extract_features(fake_wfm, model, device, batch_size)

    mu_r, cov_r = compute_activation_statistics(real_feats)
    mu_g, cov_g = compute_activation_statistics(fake_feats)

    fid = fid_from_stats(mu_r, cov_r, mu_g, cov_g)

    return {
        "fid":      fid,
        "feat_dim": real_feats.shape[1],
        "n_real":   len(real_wfm),
        "n_fake":   len(fake_wfm),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-class FID
# ──────────────────────────────────────────────────────────────────────────────

def compute_per_class_fid(real_wfm:   np.ndarray,
                           real_surf:  np.ndarray,
                           fake_wfm:   np.ndarray,
                           fake_surf:  np.ndarray,
                           cls_ckpt:   str,
                           device:     str = "cpu") -> dict:
    """
    Compute FID for each surface type independently.
    """
    model = load_classifier(cls_ckpt, device)

    def feat(arr):
        return extract_features(arr, model, device)

    surf_names = {0: "Ocean", 1: "Ice", 2: "Land"}
    results    = {}

    # Overall
    mu_r, cov_r = compute_activation_statistics(feat(real_wfm))
    mu_g, cov_g = compute_activation_statistics(feat(fake_wfm))
    results["overall"] = fid_from_stats(mu_r, cov_r, mu_g, cov_g)

    # Per-class
    for cls_id, name in surf_names.items():
        rm = real_surf == cls_id
        fm = fake_surf == cls_id
        if rm.sum() < 100 or fm.sum() < 100:
            results[name] = float("nan")
            continue
        mu_r, cov_r = compute_activation_statistics(feat(real_wfm[rm]))
        mu_g, cov_g = compute_activation_statistics(feat(fake_wfm[fm]))
        results[name] = fid_from_stats(mu_r, cov_r, mu_g, cov_g)

    return results


def load_real_waveforms_from_nc(nc_path: str,
                                 split:   str = "test",
                                 n_max:   int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real waveforms from the NetCDF dataset.
    """
    from dataset import SentinelDataset
    
    # Load training stats first
    if split != "train":
        train_ds = SentinelDataset(nc_path, split="train")
        stats = train_ds.norm_stats
        ds = SentinelDataset(nc_path, split=split, norm_stats=stats)
    else:
        ds = SentinelDataset(nc_path, split=split)

    wfm  = ds.waveform[:n_max]
    surf = ds.surf_type[:n_max]
    return wfm.astype(np.float32), surf.astype(np.int64)


def parse_args():
    p = argparse.ArgumentParser(description="1D FID for radar waveforms")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--real_npy", help="(N, 128) .npy of real waveforms")
    g.add_argument("--nc_path",  help="Path to measurement.nc (uses test split)")
    p.add_argument("--fake_npy",  required=True, help="(M, 128) .npy of generated waveforms")
    p.add_argument("--cls_ckpt",  required=True, help="Trained classifier checkpoint")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_max",     type=int, default=10000, help="Max real samples to use")
    p.add_argument("--per_class", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load real waveforms
    if args.real_npy:
        real_wfm  = np.load(args.real_npy).astype(np.float32)
        real_surf = np.zeros(len(real_wfm), dtype=np.int64)  # unknown
    else:
        real_wfm, real_surf = load_real_waveforms_from_nc(
            args.nc_path, split="test", n_max=args.n_max
        )

    # Load generated waveforms
    fake_wfm  = np.load(args.fake_npy).astype(np.float32)

    # Check for metadata
    meta_path = args.fake_npy.replace("generated_waveforms.npy", "generated_metadata.npz")
    if os.path.exists(meta_path):
        meta = np.load(meta_path)
        fake_surf = meta["surf_type"].astype(np.int64)
    else:
        fake_surf = np.zeros(len(fake_wfm), dtype=np.int64)

    print(f"[FID] Real: {real_wfm.shape}, Fake: {fake_wfm.shape}")

    if args.per_class:
        results = compute_per_class_fid(
            real_wfm, real_surf, fake_wfm, fake_surf,
            args.cls_ckpt, args.device
        )
        print("\n══ FID Results (per class) ══")
        for k, v in results.items():
            print(f"  {k:10s}: {v:.4f}")
    else:
        result = compute_fid(real_wfm, fake_wfm, args.cls_ckpt, args.device)
        print(f"\n══ FID = {result['fid']:.4f} "
              f"(feat_dim={result['feat_dim']}, "
              f"n_real={result['n_real']}, n_fake={result['n_fake']}) ══")