#!/bin/bash
# SLURM wrapper for DP3 (3D Diffusion Policy) fine-tuning on genesis_hr_bench.
#
# Submit:
#   sbatch --time=1:00:00  scripts/finetune_dp3.sbatch --task pour_water --sanity
#   sbatch --time=12:00:00 scripts/finetune_dp3.sbatch --task pour_water --full
#   sbatch --time=12:00:00 scripts/finetune_dp3.sbatch --task pour_water --full -- training.num_epochs=2000
#
# Notes:
#   * Dev node TITAN X is cc 5.2 — torch+cu121 wheels won't init it. Run under
#     sbatch on gpu-preempt with a100/h100/l40s/a40.
#   * No checkpoint download — DP3 trains from scratch.
#   * The zarr replay buffer is built on first launch by the inner script.

#SBATCH --job-name=dp3_finetune
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=runs/dp3/slurm_%j.out
#SBATCH --error=runs/dp3/slurm_%j.err

set -e
mkdir -p runs/dp3

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo

cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench
exec scripts/finetune_dp3.sh "$@"
