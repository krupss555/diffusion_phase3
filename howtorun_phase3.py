#!/bin/bash
# COMPLETE RUN COMMANDS - ALL FIXES APPLIED

# ══════════════════════════════════════════════════════════════
# STEP 1: Train Classifier (for IS/FID evaluation)
# ══════════════════════════════════════════════════════════════
nohup python3 eval/train_classifier.py \
    --nc_path  data/sral_s3_level1b_2023/measurement.nc \
    --out_dir  ./eval/classifier_ckpt \
    --epochs   100 \
    --batch    256 \
> logs/classifier.log 2>&1 &

# ══════════════════════════════════════════════════════════════
# STEP 2: Train Diffusion Model (FIXED VERSION)
# ══════════════════════════════════════════════════════════════
nohup python3 diffusion/train_diffusion.py \
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \
    --out_dir    ./runs/diffusion_fixed \
    --epochs     500 \
    --batch      64  \
    --lr         2e-4 \
    --T          1000 \
    --cfg_prob   0.1  \
    --num_workers 4   \
> logs/diffusion_fixed.log 2>&1 &

# Monitor training:
# tail -f logs/diffusion_fixed.log

# ══════════════════════════════════════════════════════════════
# STEP 3: Generate Samples (after training completes)
# ══════════════════════════════════════════════════════════════
python3 diffusion/sample_diffusion.py \
    --ckpt        ./runs/diffusion_fixed/checkpoints/best_model.pt \
    --nc_path     data/sral_s3_level1b_2023/measurement.nc \
    --out_dir     ./generated/diffusion \
    --n_samples   5000 \
    --ddim_steps  100 \
    --cfg_scale   2.0

# ══════════════════════════════════════════════════════════════
# STEP 4: Full Evaluation
# ══════════════════════════════════════════════════════════════
python3 eval/eval_all.py \
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \
    --cls_ckpt   ./eval/classifier_ckpt/best_classifier.pt \
    --diff_dir   ./generated/diffusion \
    --out_json   ./eval/results/comparison.json \
    --n_max      5000 \
    --n_dtw      500
```

---

## 🎯 KEY IMPROVEMENTS EXPLAINED

### 1. **Conditioning Architecture**
**Your concern**: *"it seems stupid to just add it"*

**Old approach**: Simple concatenation
```
surf_emb (16) + cont_mlp(2→16) → concat → 32
```

**NEW approach**: Better feature extraction
```
surf_emb (16) + cont_mlp(2→32→16 with Dropout) → concat → LayerNorm → 32