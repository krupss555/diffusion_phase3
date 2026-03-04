"""
eval/inception_score.py
=======================
Inception Score (IS) for Sentinel-3 SRAL generated waveforms.

Adapts the standard IS (Salimans et al. 2016) used in WaveGAN (Donahue 2019)
for 1D radar waveforms:
  IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )

• p(y|x)  = classifier softmax probabilities for generated sample x
• p(y)    = marginal class distribution = mean of p(y|x) over all x

Higher IS  →  generated samples are both diverse AND semantically clear.
Maximum IS for 3 classes = 3.0.

Reference
─────────────────────────────────────────────────────────────────────
Donahue et al. (2019), WaveGAN – Table 1 evaluation methodology.
Salimans et al. (2016), Improved Techniques for Training GANs.

Usage
──────────────────────────────────────────────────────────────────────
python eval/inception_score.py \\
    --fake_npy  generated/diffusion/generated_waveforms.npy \\
    --cls_ckpt  eval/classifier_ckpt/best_classifier.pt \\
    --splits    10
──────────────────────────────────────────────────────────────────────
"""

"""
eval/inception_score.py
=======================
Inception Score (IS) for Sentinel-3 SRAL generated waveforms.
"""
"""
eval/inception_score.py
=======================
Inception Score (IS) for Sentinel-3 SRAL generated waveforms.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.train_classifier import ResNet1DClassifier, load_classifier # <--- IMPORT FIXED


# ──────────────────────────────────────────────────────────────────────────────
# Probability Extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_softmax_probs(waveforms: np.ndarray,
                      model:     ResNet1DClassifier,
                      device:    str,
                      batch_size: int = 512) -> np.ndarray:
    """
    Compute softmax class probabilities for each generated waveform.
    """
    model.eval()
    all_probs = []

    for i in range(0, len(waveforms), batch_size):
        batch = torch.from_numpy(waveforms[i: i + batch_size]).unsqueeze(1).to(device)
        logits = model(batch)                                # (bs, C)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()    # (bs, C)
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)                # (N, C)


# ──────────────────────────────────────────────────────────────────────────────
# IS Computation
# ──────────────────────────────────────────────────────────────────────────────

def inception_score_from_probs(probs:  np.ndarray,
                                splits: int = 10
                                ) -> Tuple[float, float]:
    """
    Compute IS and its standard deviation across splits.
    """
    N = len(probs)
    split_is = []

    for k in range(splits):
        idx   = np.arange(k * N // splits, (k + 1) * N // splits)
        p_yx  = probs[idx]              # (split_size, C)  p(y|x)
        p_y   = p_yx.mean(axis=0)      # (C,)              p(y)

        # KL(p(y|x) || p(y))  for each x
        kl    = p_yx * (np.log(p_yx + 1e-10) - np.log(p_y[None, :] + 1e-10))
        kl    = kl.sum(axis=1).mean()   # scalar

        split_is.append(np.exp(kl))

    return float(np.mean(split_is)), float(np.std(split_is))


def compute_inception_score(waveforms: np.ndarray,
                             cls_ckpt:  str,
                             device:    str = "cpu",
                             splits:    int = 10
                             ) -> dict:
    """
    End-to-end IS computation.
    """
    # Use the shared loading function
    model = load_classifier(cls_ckpt, device)

    print(f"[IS] Computing softmax probs for {len(waveforms)} samples …")
    probs = get_softmax_probs(waveforms, model, device)

    is_mean, is_std = inception_score_from_probs(probs, splits=splits)

    # Also compute accuracy proxy: if IS ≈ num_classes, each class is equally predicted
    marginal_entropy = -(probs.mean(0) * np.log(probs.mean(0) + 1e-10)).sum()
    
    # Get num_classes from the model's fc layer
    num_classes = model.fc.out_features

    return {
        "is_mean":          is_mean,
        "is_std":           is_std,
        "num_classes":      num_classes,
        "marginal_entropy": float(marginal_entropy),
        "n_samples":        len(waveforms),
        "n_splits":         splits,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inception Score for 1D waveforms")
    p.add_argument("--fake_npy",  required=True)
    p.add_argument("--cls_ckpt",  required=True)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits",    type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fake_wfm = np.load(args.fake_npy).astype(np.float32)
    result   = compute_inception_score(
        fake_wfm, args.cls_ckpt, args.device, args.splits
    )

    print(f"\n══ Inception Score (IS) ══")
    print(f"  IS       = {result['is_mean']:.4f} ± {result['is_std']:.4f}")
    print(f"  Max IS   = {result['num_classes']}.0  (for {result['num_classes']} classes)")
    print(f"  Marginal entropy = {result['marginal_entropy']:.4f}  "
          f"(max={np.log(result['num_classes']):.4f})")
    print(f"  N samples = {result['n_samples']}  |  splits = {result['n_splits']}")