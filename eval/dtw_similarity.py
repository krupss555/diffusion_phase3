"""
eval/dtw_similarity.py
======================
Waveform similarity metrics for GAN vs Diffusion evaluation.

Metrics implemented (from referenced papers):
─────────────────────────────────────────────────────────────────────
1. DTW Distance         (Truong & Yanushkevich 2019 radar GAN paper)
   Dynamic Time Warping – shape similarity ignoring minor temporal shifts.
   Lower  =  more similar.

2. Nearest-Neighbour Distance  |D|_train  and  |D|_self
   From WaveGAN (Donahue et al. 2019, Table 1).
   |D|_train : distance of generated samples to training set  (overfitting check)
   |D|_self  : diversity within generated set
   Both measured in frequency domain (magnitude of rfft).

3. PSD MSE / PSD Kullback-Leibler Divergence
   Mean Squared Error between mean Power Spectral Densities.
   Used in Truong 2019 (ensemble variance MSE) and Li et al. 2025.

4. Amplitude Distribution Wasserstein Distance
   1-D Wasserstein-1 distance between amplitude histograms.
   Referenced in Li et al. 2025 (Table I Texture Similarity via
   Wasserstein distance).

5. Maximum Mean Discrepancy (MMD)
   Kernel-based two-sample test statistic.
   Primary metric in Li et al. 2025 (Table I).

6. Spectral Centroid Error
   Mean absolute error of spectral centroid frequency.

Usage
──────────────────────────────────────────────────────────────────────
python eval/dtw_similarity.py \\
    --real_npy  data/real_test_waveforms.npy \\
    --fake_npy  generated/diffusion/generated_waveforms.npy \\
    --n_dtw     500

python eval/dtw_similarity.py \\
    --nc_path   data/sral_s3_level1b_2023/measurement.nc \\
    --fake_npy  generated/gan/generated_waveforms.npy \\
    --n_dtw     500
──────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Time Warping
# ──────────────────────────────────────────────────────────────────────────────

def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    DTW distance between two 1-D sequences using standard DP.
    Uses Euclidean point-wise cost.
    """
    n, m = len(s1), len(s2)
    # Sakoe-Chiba band: window = max(n, m) // 10
    w = max(1, max(n, m) // 10)

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                    dtw[i, j - 1],
                                    dtw[i - 1, j - 1])
    return float(np.sqrt(dtw[n, m]))


def dtw_distance_fast(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Fast DTW approximation using resampling (for large-scale eval).
    Downsamples to 32 points before DTW.
    """
    step = max(1, len(s1) // 32)
    s1d  = s1[::step]
    s2d  = s2[::step]
    n, m = len(s1d), len(s2d)
    dtw  = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1d[i-1] - s2d[j-1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(np.sqrt(dtw[n, m]))


def compute_mean_dtw(real:  np.ndarray,
                      fake:  np.ndarray,
                      n_pairs: int = 500,
                      fast:    bool = True) -> float:
    """
    Sample random real-fake pairs and compute mean DTW distance.
    """
    n = min(n_pairs, len(real), len(fake))
    ri = np.random.choice(len(real), n, replace=False)
    fi = np.random.choice(len(fake), n, replace=False)
    fn = dtw_distance_fast if fast else dtw_distance
    dists = [fn(real[ri[i]], fake[fi[i]]) for i in range(n)]
    return float(np.mean(dists)), float(np.std(dists))


# ──────────────────────────────────────────────────────────────────────────────
# Nearest-Neighbour Distances  (WaveGAN metric)
# ──────────────────────────────────────────────────────────────────────────────

def _rfft_magnitude(waveforms: np.ndarray) -> np.ndarray:
    """Compute rfft magnitude for each waveform: (N, 65) float32."""
    return np.abs(np.fft.rfft(waveforms, axis=-1)).astype(np.float32)


def nn_distance(query: np.ndarray,
                reference: np.ndarray,
                n_query: int = 1000,
                batch_size: int = 512) -> float:
    """
    For each query sample, find the L2 distance to its nearest neighbour
    in the reference set (in frequency domain).  Returns mean NN distance.
    """
    qf  = _rfft_magnitude(query[:n_query])       # (n_q, 65)
    rf  = _rfft_magnitude(reference)              # (N_r, 65)

    nn_dists = []
    for i in range(0, len(qf), batch_size):
        batch = qf[i: i + batch_size]             # (bs, 65)
        # L2 distances to all reference samples
        diffs = batch[:, None, :] - rf[None, :, :]  # (bs, N_r, 65)
        dists = np.linalg.norm(diffs, axis=-1)       # (bs, N_r)
        nn_dists.append(dists.min(axis=1))           # (bs,)

    return float(np.concatenate(nn_dists).mean())


def compute_nn_metrics(real_train: np.ndarray,
                        fake:       np.ndarray,
                        n_samples:  int = 1000
                        ) -> dict:
    """
    WaveGAN nearest-neighbour metrics (Table 1):
      |D|_self  : intra-fake diversity
      |D|_train : distance fake → train  (overfitting signal)
    Normalised by real-test self-distance.
    """
    # Self-distance of fake
    fake_nn  = nn_distance(fake, fake,       n_query=n_samples)
    # Distance of fake to training set
    train_nn = nn_distance(fake, real_train, n_query=n_samples)

    # Real self-distance for normalisation
    real_nn  = nn_distance(real_train[:n_samples], real_train, n_query=n_samples)

    d_self  = fake_nn  / (real_nn + 1e-8)
    d_train = train_nn / (real_nn + 1e-8)

    return {
        "D_self_raw":    fake_nn,
        "D_train_raw":   train_nn,
        "D_real_raw":    real_nn,
        "D_self_norm":   d_self,
        "D_train_norm":  d_train,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PSD metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psd_mse(real: np.ndarray, fake: np.ndarray) -> dict:
    """
    Compare mean Power Spectral Densities.
    PSD = |rfft(x)|²
    Returns: psd_mse, psd_kl, centroid_mae
    """
    def mean_psd(wfm):
        psd = np.abs(np.fft.rfft(wfm, axis=-1)) ** 2    # (N, 65)
        return psd.mean(0)                                # (65,)

    psd_r = mean_psd(real)
    psd_g = mean_psd(fake)

    psd_mse = float(((psd_r - psd_g) ** 2).mean())

    # Normalised KL: treat normalised PSDs as probability distributions
    eps   = 1e-10
    p_r   = psd_r / (psd_r.sum() + eps)
    p_g   = psd_g / (psd_g.sum() + eps)
    psd_kl = float((p_r * np.log((p_r + eps) / (p_g + eps))).sum())

    # Spectral centroid
    freqs     = np.arange(len(psd_r), dtype=float)
    centroid_r = (freqs * p_r).sum()
    centroid_g = (freqs * p_g).sum()
    centroid_mae = float(abs(centroid_r - centroid_g))

    return {
        "psd_mse":       psd_mse,
        "psd_kl":        psd_kl,
        "centroid_mae":  centroid_mae,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Amplitude distribution (Wasserstein-1)
# ──────────────────────────────────────────────────────────────────────────────

def compute_amplitude_wasserstein(real: np.ndarray, fake: np.ndarray) -> float:
    """
    Wasserstein-1 distance between amplitude distributions.
    Flattens all sample values.
    """
    r_flat = real.flatten()
    f_flat = fake.flatten()
    return float(wasserstein_distance(r_flat, f_flat))


# ──────────────────────────────────────────────────────────────────────────────
# MMD (Li et al. 2025, Table I)
# ──────────────────────────────────────────────────────────────────────────────

def compute_mmd(real:  np.ndarray,
                fake:  np.ndarray,
                sigma: float = 1.0,
                n_max: int   = 2000) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    k(x,y) = exp(-||x-y||² / (2σ²))

    Computed in frequency domain (rfft magnitude) for stability.
    """
    rf  = _rfft_magnitude(real[:n_max]).astype(np.float64)
    gf  = _rfft_magnitude(fake[:n_max]).astype(np.float64)

    # Subsample for tractability
    nr, ng = min(len(rf), 1000), min(len(gf), 1000)
    rf = rf[:nr]; gf = gf[:ng]

    def rbf_kernel(X, Y, sig):
        XX = (X ** 2).sum(1, keepdims=True)
        YY = (Y ** 2).sum(1, keepdims=True)
        XY = X @ Y.T
        dist2 = XX + YY.T - 2 * XY
        return np.exp(-dist2 / (2 * sig ** 2))

    Krr = rbf_kernel(rf, rf, sigma)
    Kgg = rbf_kernel(gf, gf, sigma)
    Krg = rbf_kernel(rf, gf, sigma)

    n, m = len(rf), len(gf)
    np.fill_diagonal(Krr, 0.0)
    np.fill_diagonal(Kgg, 0.0)

    mmd2 = (Krr.sum() / (n * (n - 1))
            + Kgg.sum() / (m * (m - 1))
            - 2.0 * Krg.mean())

    return float(np.sqrt(max(mmd2, 0.0)))


# ──────────────────────────────────────────────────────────────────────────────
# Combined runner
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_similarity_metrics(real:    np.ndarray,
                                    fake:    np.ndarray,
                                    n_dtw:   int = 500,
                                    verbose: bool = True) -> dict:
    """
    Compute all waveform similarity metrics.

    real  : (N, 128) float32  normalised to [-1, 1]
    fake  : (M, 128) float32  normalised to [-1, 1]
    """
    results = {}

    print("[Similarity] Computing DTW …")
    dtw_mean, dtw_std = compute_mean_dtw(real, fake, n_pairs=n_dtw)
    results["dtw_mean"] = dtw_mean
    results["dtw_std"]  = dtw_std

    print("[Similarity] Computing PSD metrics …")
    psd = compute_psd_mse(real, fake)
    results.update(psd)

    print("[Similarity] Computing Wasserstein-1 amplitude distance …")
    results["wasserstein_amplitude"] = compute_amplitude_wasserstein(real, fake)

    print("[Similarity] Computing MMD …")
    results["mmd"] = compute_mmd(real, fake)

    print("[Similarity] Computing NN distances …")
    nn = compute_nn_metrics(real, fake, n_samples=min(1000, len(real), len(fake)))
    results.update(nn)

    if verbose:
        print("\n══ Waveform Similarity Metrics ══")
        print(f"  DTW (mean±std)         : {results['dtw_mean']:.4f} ± {results['dtw_std']:.4f}")
        print(f"  PSD MSE                : {results['psd_mse']:.6f}")
        print(f"  PSD KL divergence      : {results['psd_kl']:.4f}")
        print(f"  Spectral centroid MAE  : {results['centroid_mae']:.4f} bins")
        print(f"  Amplitude Wasserstein-1: {results['wasserstein_amplitude']:.6f}")
        print(f"  MMD                    : {results['mmd']:.6f}")
        print(f"  |D|_self  (norm)       : {results['D_self_norm']:.4f}")
        print(f"  |D|_train (norm)       : {results['D_train_norm']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Waveform similarity metrics")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--real_npy", help="(N, 128) .npy real waveforms")
    g.add_argument("--nc_path",  help="Path to measurement.nc")
    p.add_argument("--fake_npy",  required=True)
    p.add_argument("--n_dtw",    type=int, default=500,
                   help="Number of DTW pairs to sample")
    p.add_argument("--n_max",    type=int, default=10000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.real_npy:
        real_wfm = np.load(args.real_npy).astype(np.float32)
    else:
        from dataset import SentinelDataset
        ds       = SentinelDataset(args.nc_path, split="test")
        real_wfm = ds.waveform[:args.n_max].astype(np.float32)

    fake_wfm = np.load(args.fake_npy).astype(np.float32)

    print(f"Real: {real_wfm.shape}  Fake: {fake_wfm.shape}")
    compute_all_similarity_metrics(real_wfm, fake_wfm, n_dtw=args.n_dtw)
