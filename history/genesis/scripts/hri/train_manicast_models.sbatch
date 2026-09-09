#!/bin/bash
# Train AvatarPoseNet then the STS-GCN forecaster (one GPU, sequential).
# Usage: sbatch scripts/hri/train_manicast_models.sbatch

#SBATCH --job-name=hri_mctrain
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=job_logs/hri_mctrain_%j.log
#SBATCH --error=job_logs/hri_mctrain_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/env.sh
DATA_ROOT="${SKEL_OUT_ROOT:-data/hri_manicast/skeleton_data}"
OUT_DIR="runs/hri_manicast"
mkdir -p "$OUT_DIR/pose" "$OUT_DIR/forecaster"

echo "=== node $(hostname) job $SLURM_JOB_ID ==="
nvidia-smi -L | head -2

echo "--- training AvatarPoseNet ---"
"$MAWM_PY" scripts/hri/train_avatar_pose.py \
  --data-root "$DATA_ROOT" \
  --output "$OUT_DIR/pose/avatar_pose_v1.pt" \
  --epochs 15 --frame-stride 2 --batch-size 128 --workers 8

echo "--- training STS-GCN forecaster (ManiCast recipe) ---"
"$MAWM_PY" scripts/hri/train_manicast.py \
  --data-root "$DATA_ROOT" \
  --output "$OUT_DIR/forecaster/stsgcn_v1.pt" \
  --epochs 40 --steps-per-epoch 300 --batch-size 256

echo "=== training done ==="
