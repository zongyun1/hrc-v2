#!/bin/bash
# Train a Diffusion Policy on the already-built LeRobot dataset (same demos as
# the ACT run). Diffusion samples the action distribution -> tends to grasp
# more reliably than ACT's mean-regression. lerobot v0.1.0 via pi0 env.
#
# Usage: sbatch --export=ALL,LEROBOT_HOME=<dir>,REPO_ID=<ghb/...>,OUT=<dir> \
#   scripts/hri/train_diffusion_1p2.sbatch
#SBATCH --job-name=hri_diff1p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=job_logs/hri_diff1p2_%j.log
#SBATCH --error=job_logs/hri_diff1p2_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PI0=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python
export WANDB_MODE=offline
# Use DS_HOME (not LEROBOT_HOME — that name is deprecated and lerobot errors
# if it is present in the environment at all).
DS_HOME="${DS_HOME:?}"; REPO_ID="${REPO_ID:?}"; OUT="${OUT:?}"
unset LEROBOT_HOME 2>/dev/null || true
STEPS="${STEPS:-20000}"; BATCH="${BATCH:-64}"; WORKERS="${WORKERS:-8}"

echo "=== node $(hostname) diffusion train repo=$REPO_ID steps=$STEPS ==="
RESUME=()
[[ -d "$OUT/checkpoints/last" ]] && RESUME=(--resume=true)

HF_LEROBOT_HOME="$DS_HOME" "$PI0" -m lerobot.scripts.train \
  --policy.type=diffusion \
  --dataset.repo_id="$REPO_ID" \
  --output_dir="$OUT" \
  --job_name="hri_diffusion_burger" \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --num_workers="$WORKERS" \
  --policy.device=cuda \
  --save_freq=5000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --wandb.enable=false \
  "${RESUME[@]}"
echo "=== diffusion training done ==="
