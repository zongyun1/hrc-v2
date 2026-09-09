#!/bin/bash
# SLURM wrapper for LeRobot ACT finetuning on genesis-hr-bench.
#
# Submit with:
#   sbatch --time=1:00:00  scripts/finetune_act.sbatch --sanity
#   sbatch --time=8:00:00  scripts/finetune_act.sbatch --full
#   sbatch --time=12:00:00 scripts/finetune_act.sbatch --full -- --batch_size=64
#
# Notes:
#   * Reuses the pi0 venv (lerobot already installed). No bootstrap needed.
#   * ACT is small (~50M params); A40 / L40S / A100 / H100 all comfortable
#     at batch=32. Single-GPU full run ~5h for 100k steps on an A100.
#   * Preempt-resume: $SLURM_JOB_ID is stable across requeues, so the
#     output_dir is shared and lerobot's --resume picks up the last ckpt.

#SBATCH --job-name=act_finetune
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=runs/act/finetune/slurm_%j.out
#SBATCH --error=runs/act/finetune/slurm_%j.err

set -e
mkdir -p runs/act/finetune

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
exec scripts/finetune_act.sh "$@"
