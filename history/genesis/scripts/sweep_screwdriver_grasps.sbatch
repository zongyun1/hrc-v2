#!/bin/bash
#SBATCH --job-name=screw_hold_sweep
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH -o job_logs/screw_hold_%A_%a.log
#SBATCH -e job_logs/screw_hold_%A_%a.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench}"

CLOSES=(0.10 0.14 0.18 0.22)
CLOSE=${CLOSES[$SLURM_ARRAY_TASK_ID]}
OUT=data/object/grasp_sweeps/screwdriver_factory_flat
mkdir -p "${OUT}"

srun /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python \
  scripts/object_pool_grasp_demo.py \
  --profile screwdriver_m0 \
  --close-value "${CLOSE}" \
  --output-stem "screwdriver_close_${CLOSE}" \
  --output-dir "${OUT}" \
  --hold-steps 100
