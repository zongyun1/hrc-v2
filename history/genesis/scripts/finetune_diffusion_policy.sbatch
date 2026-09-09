#!/bin/bash
# SLURM wrapper for Diffusion Policy fine-tuning on genesis_hr_bench.
#
# Submit with one of:
#   sbatch --time=1:00:00  scripts/finetune_diffusion_policy.sbatch --task pour_water --sanity
#   sbatch --time=12:00:00 scripts/finetune_diffusion_policy.sbatch --task pour_water --full
#   sbatch --time=12:00:00 scripts/finetune_diffusion_policy.sbatch --task pour_water --full -- training.num_epochs=2000
#
# Notes:
#   * Dev node TITAN X is cc 5.2 — torch+cu121 wheels won't init it. Run under
#     sbatch on gpu-preempt with a100/h100/l40s/a40.
#   * Cold start trains the policy from scratch — no checkpoint download.
#   * The zarr replay buffer is built on first launch by the inner script
#     (reads vla_data/<task>/<renderer>/seed_*/).

#SBATCH --job-name=dp_finetune
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=runs/diffusion_policy/slurm_%j.out
#SBATCH --error=runs/diffusion_policy/slurm_%j.err

set -e
mkdir -p runs/diffusion_policy

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
exec scripts/finetune_diffusion_policy.sh "$@"
