#!/bin/bash
#SBATCH --job-name=evalset_init
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o job_logs/evalset_init_%j.log
#SBATCH -e job_logs/evalset_init_%j.log

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench}"
mkdir -p job_logs

# shellcheck source=env.sh
source scripts/env.sh

export SETUPTOOLS_USE_DISTUTILS=stdlib
export GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"

echo "=== eval-set init validation ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "args: $*"
echo "python=${MAWM_PY}"
echo "GENESIS_BACKEND=${GENESIS_BACKEND}"

srun "${MAWM_PY}" scripts/validate_eval_set_init.py --subprocess-per-episode "$@"
