#!/bin/bash
# In-process VLA/heuristic eval wrapper. Use for policies registered in
# scripts.vla_client.VLA_REGISTRY that do not need a model server.
#
# Usage:
#   sbatch scripts/hri/eval_hri_inprocess.sbatch [eval_vla.py args...]

#SBATCH --job-name=hri_inproc
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=job_logs/hri_inprocess_%j.log
#SBATCH --error=job_logs/hri_inprocess_%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/scripts/eval_vla.py" ]]; then
  echo "Could not resolve repo root from SLURM_SUBMIT_DIR=$REPO_ROOT" >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p job_logs

# shellcheck source=../env.sh
source scripts/env.sh
cd "$REPO_ROOT"

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo "repo: $REPO_ROOT"
echo "args: $*"
echo

export GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"
echo "[eval_hri_inprocess] GENESIS_BACKEND=$GENESIS_BACKEND"
SETUPTOOLS_USE_DISTUTILS=stdlib "$MAWM_PY" scripts/eval_vla.py "$@"
