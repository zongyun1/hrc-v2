#!/bin/bash
# SLURM wrapper for LeRobot Diffusion Policy finetuning on genesis-hr-bench.
# Distinct from scripts/finetune_diffusion_policy.sbatch, which trains
# Chi et al.'s standalone Diffusion Policy.
#
# Submit with:
#   sbatch --time=1:00:00  scripts/finetune_lerobot_diffusion.sbatch --sanity
#   sbatch --time=8:00:00  scripts/finetune_lerobot_diffusion.sbatch --full
#
# Notes:
#   * Reuses the pi0 venv (lerobot already installed). No bootstrap needed.
#   * LeRobot's Diffusion (~250M params + DDPM head) needs more VRAM than
#     ACT but A40/A100 batch=32 still fits. Single-GPU full run ~10-12h.

#SBATCH --job-name=lerobot_diffusion_finetune
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=runs/lerobot_diffusion/finetune/slurm_%j.out
#SBATCH --error=runs/lerobot_diffusion/finetune/slurm_%j.err

set -e
mkdir -p runs/lerobot_diffusion/finetune

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
exec scripts/finetune_lerobot_diffusion.sh "$@"
