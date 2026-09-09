#!/bin/bash
# Avatar skeleton + head-camera data collection (CPU genesis).
# Usage: sbatch scripts/hri/collect_skeleton.sbatch <task1> [<task2> ...]
# Env: SKEL_SEEDS (default 40), SKEL_START_SEED (default 0), SKEL_OUT_ROOT.

#SBATCH --job-name=hri_skel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=job_logs/hri_skel_%j.log
#SBATCH --error=job_logs/hri_skel_%j.err

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
source scripts/env.sh

export GENESIS_BACKEND=cpu
SEEDS="${SKEL_SEEDS:-40}"
START="${SKEL_START_SEED:-0}"
OUT_ROOT="${SKEL_OUT_ROOT:-data/hri_manicast/skeleton_data}"

echo "=== node $(hostname) job $SLURM_JOB_ID tasks: $* seeds=$SEEDS start=$START ==="
for task in "$@"; do
  echo "--- collecting $task ---"
  "$MAWM_PY" scripts/hri/collect_avatar_skeleton.py \
    --task "$task" \
    --start-seed "$START" \
    --seeds "$SEEDS" \
    --randomize-avatar \
    --out-root "$OUT_ROOT/$task" || echo "[warn] $task collection exited nonzero"
done
echo "=== done ==="
