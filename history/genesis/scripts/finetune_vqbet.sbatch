#!/bin/bash
# SLURM wrapper for LeRobot VQ-BeT finetuning on genesis-hr-bench.
#
# Submit with:
#   sbatch --time=1:00:00  scripts/finetune_vqbet.sbatch --sanity
#   sbatch --time=8:00:00  scripts/finetune_vqbet.sbatch --full
#
# Notes:
#   * Reuses the pi0 venv (lerobot already installed). No bootstrap needed.
#   * VQ-BeT is larger than ACT (residual VQ-VAE + transformer head); A100
#     batch=32 fits comfortably. Single-GPU full run ~6-8h for 100k steps.
#   * Preempt-resume: $SLURM_JOB_ID is stable across requeues, so the
#     output_dir is shared and lerobot's --resume picks up the last ckpt.

#SBATCH --job-name=vqbet_finetune
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=runs/vqbet/finetune/slurm_%j.out
#SBATCH --error=runs/vqbet/finetune/slurm_%j.err

set -e
mkdir -p runs/vqbet/finetune

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
exec scripts/finetune_vqbet.sh "$@"
