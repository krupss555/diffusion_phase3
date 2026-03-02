"""
howtorun_phase3.py
==================
All commands needed to run Phase-3 (Diffusion) and evaluation.

STEP 1: Train the 1D ResNet classifier (needed for IS and FID)
STEP 2: Train the diffusion model
STEP 3: Generate samples (diffusion + GAN for comparison)
STEP 4: Run the full evaluation scorecard
"""

# ── STEP 1: Train classifier (IS / FID feature extractor) ─────────────────
# Trains a 1D ResNet on the 3-class surf_type task.
# Best checkpoint → eval/classifier_ckpt/best_classifier.pt
# Expected val accuracy: > 85%

nohup_step1 = """
nohup python3 eval/train_classifier.py \\
    --nc_path  data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir  ./eval/classifier_ckpt \\
    --epochs   100 \\
    --batch    256 \\
    --lr       1e-3 \\
> logs/classifier_v1.log 2>&1 &
"""

# ── STEP 2: Train diffusion model ─────────────────────────────────────────
# Physics-informed DDPM with UNet1D denoiser.
# Key hyperparameters:
#   --T          1000   (total DDPM timesteps)
#   --cfg_scale  2.0    (classifier-free guidance at inference time)
#   --cfg_prob   0.10   (10% null-cond dropout during training)
#   --epochs     300    (increase to 500 for best results)
#   --batch      64     (DDPM prefers smaller batches than GAN)
#   --lr         2e-4   (with cosine schedule + warmup)

nohup_step2_fresh = """
nohup python3 diffusion/train_diffusion.py \
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \
    --out_dir    ./runs/checkpoints \
    --epochs     500 \
    --batch      64  \
    --lr         2e-4 \
    --emb_dim    32  \
    --T          1000 \
    --cfg_scale  2.0  \
    --cfg_prob   0.10 \
    --num_workers 4   \
> logs/train_ph3_v2.log 2>&1 &
"""

nohup_step2_resume = """
nohup python3 diffusion/train_diffusion.py \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir    ./runs/diffusion_phase3 \\
    --epochs     300 \\
    --batch      64  \\
    --lr         2e-4 \\
    --resume     ./runs/diffusion_phase3/checkpoints/ckpt_epoch_0099.pt \\
> logs/diffusion_phase3_resume.log 2>&1 &
"""

# ── STEP 3: Generate samples ───────────────────────────────────────────────
# Generate 5000 samples from each model (balanced across surf types)

generate_diffusion = """
python3 diffusion/sample_diffusion.py \\
    --ckpt        ./runs/diffusion_phase3/checkpoints/ckpt_epoch_0299.pt \\
    --nc_path     data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir     ./generated/diffusion \\
    --n_samples   5000 \\
    --ddim_steps  100 \\
    --cfg_scale   2.0 \\
    --batch_size  64
"""

generate_gan = """
python3 diffusion/sample_gan.py \\
    --ckpt        ./runs/sentinel_gan/checkpoints/ckpt_epoch_0199.pt \\
    --nc_path     data/sral_s3_level1b_2023/measurement.nc \\
    --out_dir     ./generated/gan \\
    --n_samples   5000 \\
    --batch_size  128
"""

# ── STEP 4: Full evaluation scorecard ────────────────────────────────────
# Computes IS, FID, FID per class, DTW, PSD MSE/KL, Wasserstein,
# MMD, |D|_self, |D|_train and prints a comparison table.

evaluate_all = """
python3 eval/eval_all.py \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --cls_ckpt   ./eval/classifier_ckpt/best_classifier.pt \\
    --gan_dir    ./generated/gan \\
    --diff_dir   ./generated/diffusion \\
    --out_json   ./eval/results/comparison.json \\
    --n_max      5000 \\
    --n_dtw      500
"""

# ── Individual metric scripts (if you want to run them standalone) ─────────

run_fid_only = """
python3 eval/fid_1d.py \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --fake_npy   ./generated/diffusion/generated_waveforms.npy \\
    --cls_ckpt   ./eval/classifier_ckpt/best_classifier.pt \\
    --per_class
"""

run_is_only = """
python3 eval/inception_score.py \\
    --fake_npy   ./generated/diffusion/generated_waveforms.npy \\
    --cls_ckpt   ./eval/classifier_ckpt/best_classifier.pt \\
    --splits     10
"""

run_dtw_only = """
python3 eval/dtw_similarity.py \\
    --nc_path    data/sral_s3_level1b_2023/measurement.nc \\
    --fake_npy   ./generated/diffusion/generated_waveforms.npy \\
    --n_dtw      500
"""

if __name__ == "__main__":
    print("Phase 3 commands printed above. Copy-paste to run.")
    print("\nSteps:")
    print("  1. Train classifier:   eval/train_classifier.py")
    print("  2. Train diffusion:    diffusion/train_diffusion.py")
    print("  3. Generate samples:   diffusion/sample_diffusion.py + sample_gan.py")
    print("  4. Full evaluation:    eval/eval_all.py")
