"""
eval/eval_all.py
================
Unified evaluation scorecard: GAN vs Physics-Informed Diffusion Model.

Computes the full set of metrics and prints a side-by-side comparison table:

  ┌────────────────────────────┬────────────┬────────────┬────────────┐
  │ Metric                     │ Real       │ GAN        │ Diffusion  │
  ├────────────────────────────┼────────────┼────────────┼────────────┤
  │ IS (↑ better, max=3.0)     │ x.xx±x.xx  │ x.xx±x.xx  │ x.xx±x.xx  │
  │ FID (↓ better)             │ 0.00       │ xx.xx      │ xx.xx      │
  │ DTW mean (↓ better)        │ xx.xx      │ xx.xx      │ xx.xx      │
  │ PSD MSE (↓ better)         │ 0.000000   │ 0.000000   │ 0.000000   │
  │ PSD KL (↓ better)          │ 0.0000     │ x.xxxx     │ x.xxxx     │
  │ Wasserstein-1 amp (↓)      │ 0.000000   │ 0.000000   │ 0.000000   │
  │ MMD (↓ better)             │ 0.000000   │ 0.000000   │ 0.000000   │
  │ |D|_self  (norm, ↑ = div.) │ 1.00       │ x.xx       │ x.xx       │
  │ |D|_train (norm, ↑ = diff) │ 1.00       │ x.xx       │ x.xx       │
  └────────────────────────────┴────────────┴────────────┴────────────┘

Saves results to a JSON file for further analysis.

Usage
──────────────────────────────────────────────────────────────────────
python eval/eval_all.py \\
    --nc_path      data/sral_s3_level1b_2023/measurement.nc \\
    --gan_dir      generated/gan \\
    --diff_dir     generated/diffusion \\
    --cls_ckpt     eval/classifier_ckpt/best_classifier.pt \\
    --out_json     eval/results/comparison.json \\
    --n_dtw        500 \\
    --n_max        5000

# Skip one model if not yet generated:
python eval/eval_all.py \\
    --nc_path      data/sral_s3_level1b_2023/measurement.nc \\
    --diff_dir     generated/diffusion \\
    --cls_ckpt     eval/classifier_ckpt/best_classifier.pt
──────────────────────────────────────────────────────────────────────
"""

"""
eval/eval_all.py
================
Unified evaluation scorecard: GAN vs Physics-Informed Diffusion Model.
"""

"""
eval/eval_all.py
================
Unified evaluation scorecard: GAN vs Physics-Informed Diffusion Model.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fid_1d           import compute_fid, compute_per_class_fid
from eval.inception_score  import compute_inception_score
from eval.dtw_similarity   import compute_all_similarity_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_real_data(nc_path: str, split: str = "test", n_max: int = 5000
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real waveforms + surface type labels from the dataset.
    Returns (waveforms (N, 128), surf_types (N,))
    """
    from dataset import SentinelDataset
    print(f"[Eval] Loading real data (split={split}, n_max={n_max}) …")
    
    # ── FIX: Load training stats first for consistency ──
    if split != "train":
        print("[Eval] Pre-loading training stats to normalize test data...")
        # We assume the training split is available to compute valid normalization
        train_ds = SentinelDataset(nc_path, split="train")
        stats = train_ds.norm_stats
        # Now load the requested split using those stats
        ds = SentinelDataset(nc_path, split=split, norm_stats=stats)
    else:
        ds = SentinelDataset(nc_path, split=split)
    # ────────────────────────────────────────────────────

    wfm  = ds.waveform[:n_max].astype(np.float32)
    surf = ds.surf_type[:n_max].astype(np.int64)
    return wfm, surf


def load_generated_data(gen_dir: str, n_max: int = 5000
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load generated waveforms + surface type labels from a generated/ directory.
    """
    wfm_path  = os.path.join(gen_dir, "generated_waveforms.npy")
    meta_path = os.path.join(gen_dir, "generated_metadata.npz")

    assert os.path.exists(wfm_path), f"Not found: {wfm_path}"
    wfm = np.load(wfm_path).astype(np.float32)[:n_max]

    if os.path.exists(meta_path):
        meta = np.load(meta_path)
        surf = meta["surf_type"].astype(np.int64)[:n_max]
    else:
        surf = np.zeros(len(wfm), dtype=np.int64)

    return wfm, surf


def eval_one_model(name:      str,
                   real_wfm:  np.ndarray,
                   real_surf: np.ndarray,
                   fake_wfm:  np.ndarray,
                   fake_surf: np.ndarray,
                   cls_ckpt:  str,
                   device:    str,
                   n_dtw:     int) -> dict:
    """Run all metrics for one model. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Real: {real_wfm.shape}  Fake: {fake_wfm.shape}")
    print(f"{'='*60}")

    results = {"model": name, "n_real": len(real_wfm), "n_fake": len(fake_wfm)}

    # Inception Score
    print(f"\n[{name}] Computing Inception Score …")
    is_result = compute_inception_score(fake_wfm, cls_ckpt, device)
    results["is_mean"] = is_result["is_mean"]
    results["is_std"]  = is_result["is_std"]

    # FID (overall)
    print(f"\n[{name}] Computing FID …")
    fid_result = compute_fid(real_wfm, fake_wfm, cls_ckpt, device)
    results["fid"] = fid_result["fid"]

    # FID per class
    print(f"\n[{name}] Computing per-class FID …")
    per_class = compute_per_class_fid(
        real_wfm, real_surf, fake_wfm, fake_surf, cls_ckpt, device
    )
    results["fid_per_class"] = per_class

    # Waveform similarity
    print(f"\n[{name}] Computing waveform similarity metrics …")
    sim = compute_all_similarity_metrics(real_wfm, fake_wfm, n_dtw=n_dtw,
                                          verbose=False)
    results.update(sim)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: real vs real (self-comparison)
# ──────────────────────────────────────────────────────────────────────────────

def eval_real_baseline(real_wfm:  np.ndarray,
                        real_surf: np.ndarray,
                        cls_ckpt:  str,
                        device:    str,
                        n_dtw:     int) -> dict:
    """
    Evaluate real test data against real train data (establishes upper bound).
    Splits real data 50/50 into two halves.
    """
    print("\n[Eval] Computing real baseline (real vs real split) …")
    mid  = len(real_wfm) // 2
    r1   = real_wfm[:mid];  s1 = real_surf[:mid]
    r2   = real_wfm[mid:];  s2 = real_surf[mid:]

    results = {"model": "Real (baseline)", "n_real": mid, "n_fake": mid}

    is_res = compute_inception_score(r2, cls_ckpt, device)
    results["is_mean"] = is_res["is_mean"]
    results["is_std"]  = is_res["is_std"]

    fid_r  = compute_fid(r1, r2, cls_ckpt, device)
    results["fid"] = fid_r["fid"]
    results["fid_per_class"] = {}

    sim = compute_all_similarity_metrics(r1, r2, n_dtw=min(n_dtw, 200), verbose=False)
    results.update(sim)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print table
# ──────────────────────────────────────────────────────────────────────────────

def print_scorecard(all_results: List[dict]):
    """Print a formatted comparison table."""
    names   = [r["model"] for r in all_results]
    col_w   = max(12, max(len(n) for n in names) + 2)
    hdr_w   = 30

    def row(label, vals, fmt=".4f"):
        cells = [f"{v:{fmt}}" if not isinstance(v, str) else v for v in vals]
        return f"  {label:<{hdr_w}}" + "".join(f"{c:>{col_w}}" for c in cells)

    header = f"  {'Metric':<{hdr_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    sep    = "  " + "─" * (hdr_w + col_w * len(names))

    print(f"\n{'═'*70}")
    print("  EVALUATION SCORECARD")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*70}")
    print(header)
    print(sep)

    def vals(key, default=float("nan")):
        return [r.get(key, default) for r in all_results]

    print(row("IS (↑ better,  max=3.0)",    vals("is_mean")))
    print(row("IS std",                      vals("is_std")))
    print(row("FID (↓ better)",              vals("fid")))
    print(row("FID Ocean (↓)",               [r.get("fid_per_class", {}).get("Ocean", float("nan"))
                                               for r in all_results]))
    print(row("FID Ice   (↓)",               [r.get("fid_per_class", {}).get("Ice", float("nan"))
                                               for r in all_results]))
    print(row("FID Land  (↓)",               [r.get("fid_per_class", {}).get("Land", float("nan"))
                                               for r in all_results]))
    print(sep)
    print(row("DTW mean  (↓ better)",        vals("dtw_mean")))
    print(row("DTW std",                     vals("dtw_std")))
    print(row("PSD MSE   (↓ better)",        vals("psd_mse"), fmt=".6f"))
    print(row("PSD KL    (↓ better)",        vals("psd_kl")))
    print(row("Spectral centroid MAE (↓)",   vals("centroid_mae")))
    print(row("Wasserstein-1 amp (↓)",       vals("wasserstein_amplitude"), fmt=".6f"))
    print(row("MMD       (↓ better)",        vals("mmd"), fmt=".6f"))
    print(sep)
    print(row("|D|_self  (↑ = diverse)",     vals("D_self_norm")))
    print(row("|D|_train (↑ = not overfit)", vals("D_train_norm")))
    print(f"{'═'*70}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # ── Load real data ────────────────────────────────────────────────────
    real_wfm, real_surf = load_real_data(args.nc_path, split="test",
                                          n_max=args.n_max)

    all_results = []

    # ── Real baseline ─────────────────────────────────────────────────────
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        base = eval_real_baseline(real_wfm, real_surf,
                                   args.cls_ckpt, device, args.n_dtw)
        all_results.append(base)

    # ── GAN ───────────────────────────────────────────────────────────────
    if args.gan_dir and os.path.exists(os.path.join(args.gan_dir, "generated_waveforms.npy")):
        gan_wfm, gan_surf = load_generated_data(args.gan_dir, n_max=args.n_max)
        gan_res = eval_one_model(
            name="WaveGAN",
            real_wfm=real_wfm, real_surf=real_surf,
            fake_wfm=gan_wfm,  fake_surf=gan_surf,
            cls_ckpt=args.cls_ckpt, device=device, n_dtw=args.n_dtw,
        )
        all_results.append(gan_res)
    elif args.gan_dir:
        print(f"[Eval] GAN dir not found or empty: {args.gan_dir}")

    # ── Diffusion ─────────────────────────────────────────────────────────
    if args.diff_dir and os.path.exists(os.path.join(args.diff_dir, "generated_waveforms.npy")):
        diff_wfm, diff_surf = load_generated_data(args.diff_dir, n_max=args.n_max)
        diff_res = eval_one_model(
            name="Diffusion",
            real_wfm=real_wfm,  real_surf=real_surf,
            fake_wfm=diff_wfm,  fake_surf=diff_surf,
            cls_ckpt=args.cls_ckpt, device=device, n_dtw=args.n_dtw,
        )
        all_results.append(diff_res)
    elif args.diff_dir:
        print(f"[Eval] Diffusion dir not found or empty: {args.diff_dir}")

    if not all_results:
        print("[Eval] No results to report. Check --gan_dir / --diff_dir / --cls_ckpt.")
        return

    # ── Print table ───────────────────────────────────────────────────────
    print_scorecard(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────
    def to_serialisable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serialisable(v) for k, v in obj.items()}
        return obj

    out = {
        "timestamp":  datetime.now().isoformat(),
        "n_max":      args.n_max,
        "n_dtw":      args.n_dtw,
        "results":    [to_serialisable(r) for r in all_results],
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Eval] Results saved → {args.out_json}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full evaluation scorecard")
    p.add_argument("--nc_path",   required=True,  help="measurement.nc")
    p.add_argument("--cls_ckpt",  required=True,  help="Classifier checkpoint")
    p.add_argument("--gan_dir",   default="",     help="GAN generated/ dir")
    p.add_argument("--diff_dir",  default="",     help="Diffusion generated/ dir")
    p.add_argument("--out_json",  default="eval/results/comparison.json")
    p.add_argument("--n_max",     type=int, default=5000,
                   help="Max samples per split")
    p.add_argument("--n_dtw",     type=int, default=500,
                   help="DTW pairs to evaluate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)