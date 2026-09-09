#!/bin/bash
# Record-mode VLA demo collection over a seed range (one task per job).
# Usage: sbatch scripts/hri/collect_vla_range.sbatch <task> <start_seed> <n_seeds>
# Env: VLA_OUT_ROOT (default data/hri_manicast/vla_data_retrain)

#SBATCH --job-name=hri_vla_col
#SBATCH --partition=cpu-preempt
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=job_logs/hri_vla_col_%j.log
#SBATCH --error=job_logs/hri_vla_col_%j.err

set -o pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/env.sh
export GENESIS_BACKEND=cpu

TASK="${1:?task}"; START="${2:?start_seed}"; N="${3:?n_seeds}"
OUT_ROOT="${VLA_OUT_ROOT:-data/hri_manicast/vla_data_retrain}"
echo "=== node $(hostname) job $SLURM_JOB_ID task=$TASK seeds=$START..$((START+N-1)) ==="

for ((s=START; s<START+N; s++)); do
  echo "--- seed $s ($(date +%H:%M)) ---"
  timeout 2400 "$MAWM_PY" scripts/vla_data/collect_vla.py \
    --task "$TASK" --seed "$s" --mode record \
    --out-root "$OUT_ROOT/$TASK/rasterizer" \
    --randomize-avatar \
    || echo "[warn] seed $s exited nonzero"
done
echo "=== done ==="
