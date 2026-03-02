"""
dataset.py – PyTorch Dataset for Sentinel-3 SRAL Level-1B altimeter data.

Expected file layout
─────────────────────────
data/sral_s3_level1b_2023/measurement.nc

Variables loaded
────────────────
  i2q2_meas_ku_l1b_echo_sar_ku : (N, 128)  float32  power waveform
  surf_type_l1b_echo_sar_ku    : (N,)       float32  surface type 0/1/2/3
  range_ku_l1b_echo_sar_ku     : (N,)       float32  range [m]
  range_rate_l1b_echo_sar_ku   : (N,)       float32  radial velocity [m/s]

Preprocessing
─────────────
1. Remove class 1 (Enclosed Seas / Lakes) — too few samples to model
2. Remap surf_type  {0→0, 2→1, 3→2}  (3 classes remain)
3. Waveform: log1p  →  scale to [-1, 1] using training-split percentiles
4. Range & range_rate: z-score normalise using training-split stats
5. Random train / val / test split (80 / 10 / 10) with fixed seed

Norm stats (waveform_log_p99, range_mean/std, rr_mean/std) are stored on
self.norm_stats and can be saved/loaded for inference consistency.

Output shapes
─────────────
  signal   : (1, 128)  float32  — waveform channel for Conv1d
  surf_type: scalar    long
  range    : scalar    float32  (normalised)
  range_rate: scalar   float32  (normalised)
"""

import os
import json
import numpy as np
import netCDF4 as nc4
import torch
from torch.utils.data import Dataset


# Surface type remapping after removing class 1
_SURF_REMAP = {0: 0, 2: 1, 3: 2}
NUM_SURF_TYPES = 3


def _load_nc(nc_path: str):
    """Load all required variables from the NetCDF file.
    Returns raw numpy arrays (before any filtering/normalisation).
    """
    with nc4.Dataset(nc_path, "r") as ds:
        waveform   = np.array(ds["i2q2_meas_ku_l1b_echo_sar_ku"][:], dtype=np.float32)
        surf_type  = np.array(ds["surf_type_l1b_echo_sar_ku"][:],     dtype=np.float32)
        range_ku   = np.array(ds["range_ku_l1b_echo_sar_ku"][:],      dtype=np.float32)
        range_rate = np.array(ds["range_rate_l1b_echo_sar_ku"][:],    dtype=np.float32)
    return waveform, surf_type, range_ku, range_rate


def compute_norm_stats(waveform_log: np.ndarray,
                       range_ku: np.ndarray,
                       range_rate: np.ndarray) -> dict:
    """Compute normalisation statistics from training-split arrays."""
    # Waveform is already log1p-transformed when passed here
    stats = {
        "waveform_log_p01":  float(np.percentile(waveform_log,  1)),
        "waveform_log_p99":  float(np.percentile(waveform_log, 99)),
        "range_mean":        float(range_ku.mean()),
        "range_std":         float(range_ku.std()),
        "rr_mean":           float(range_rate.mean()),
        "rr_std":            float(range_rate.std()),
    }
    return stats


def apply_waveform_norm(waveform_log: np.ndarray, stats: dict) -> np.ndarray:
    """Scale log1p waveform to [-1, 1] using p01/p99 from training stats."""
    lo = stats["waveform_log_p01"]
    hi = stats["waveform_log_p99"]
    clipped = np.clip(waveform_log, lo, hi)
    # Map [lo, hi] → [-1, 1]
    return 2.0 * (clipped - lo) / (hi - lo + 1e-8) - 1.0


class SentinelDataset(Dataset):
    """
    Parameters
    ----------
    nc_path    : path to measurement.nc
    split      : 'train' | 'val' | 'test'
    val_frac   : fraction of data for validation (default 0.10)
    test_frac  : fraction of data for test       (default 0.10)
    seed       : random seed for reproducible split
    norm_stats : pre-computed stats dict (pass val/test the training stats)
                 If None and split=='train', stats are computed automatically.
    stats_path : if set, save/load norm stats JSON here
    """

    def __init__(self,
                 nc_path:    str,
                 split:      str   = "train",
                 val_frac:   float = 0.10,
                 test_frac:  float = 0.10,
                 seed:       int   = 42,
                 norm_stats: dict  = None,
                 stats_path: str   = None):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"

        print(f"[Sentinel] Loading {nc_path} …")
        waveform, surf_type_raw, range_ku, range_rate = _load_nc(nc_path)

        # ── Remove Enclosed Seas (class 1) ───────────────────────────────
        mask = surf_type_raw != 1
        waveform   = waveform[mask]
        surf_type_raw = surf_type_raw[mask]
        range_ku   = range_ku[mask]
        range_rate = range_rate[mask]
        print(f"[Sentinel] After removing class 1: {len(waveform)} samples")

        # ── Remap surf_type  {0→0, 2→1, 3→2} ────────────────────────────
        surf_type_int = np.vectorize(_SURF_REMAP.get)(surf_type_raw.astype(int))

        # ── Train / val / test split ─────────────────────────────────────
        N = len(waveform)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_test = int(N * test_frac)
        n_val  = int(N * val_frac)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]

        split_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]

        waveform   = waveform[split_idx]
        surf_type_int = surf_type_int[split_idx]
        range_ku   = range_ku[split_idx]
        range_rate = range_rate[split_idx]

        # ── Log1p waveform (before normalisation) ────────────────────────
        waveform_log = np.log1p(waveform)   # (N, 128)

        # ── Normalisation stats ───────────────────────────────────────────
        if norm_stats is None:
            if split == "train":
                norm_stats = compute_norm_stats(
                    waveform_log.ravel(), range_ku, range_rate
                )
                if stats_path:
                    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
                    with open(stats_path, "w") as f:
                        json.dump(norm_stats, f, indent=2)
                    print(f"[Sentinel] Norm stats saved → {stats_path}")
            else:
                raise ValueError(
                    "norm_stats must be provided for val/test splits. "
                    "Pass the dict returned by the train dataset."
                )
        self.norm_stats = norm_stats

        # ── Apply normalisation ───────────────────────────────────────────
        # Waveform → [-1, 1]
        waveform_norm = apply_waveform_norm(waveform_log, norm_stats)

        # Range → z-score
        range_norm = (range_ku - norm_stats["range_mean"]) / (norm_stats["range_std"] + 1e-8)

        # Range rate → z-score
        rr_norm = (range_rate - norm_stats["rr_mean"]) / (norm_stats["rr_std"] + 1e-8)

        # ── Store as tensors ──────────────────────────────────────────────
        self.waveform   = waveform_norm.astype(np.float32)      # (N, 128)
        self.surf_type  = surf_type_int.astype(np.int64)        # (N,)
        self.range_ku   = range_norm.astype(np.float32)         # (N,)
        self.range_rate = rr_norm.astype(np.float32)            # (N,)

        print(f"[Sentinel] {split}: {len(self.waveform)} samples | "
              f"surf classes: {np.unique(surf_type_int, return_counts=True)}")

    def __len__(self):
        return len(self.waveform)

    def __getitem__(self, idx):
        # Waveform: add channel dim → (1, 128)
        signal = torch.from_numpy(self.waveform[idx]).unsqueeze(0)   # (1, 128)

        # Conditioning
        surf   = torch.tensor(self.surf_type[idx],  dtype=torch.long)
        rng    = torch.tensor(self.range_ku[idx],   dtype=torch.float32)
        rr     = torch.tensor(self.range_rate[idx], dtype=torch.float32)

        return signal, surf, rng, rr
