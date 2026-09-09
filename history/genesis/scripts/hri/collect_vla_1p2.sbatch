#!/bin/bash
# Record-mode VLA demo collection on Genesis 1.2.0 (MAWM_latest_genesis).
# Usage: sbatch scripts/hri/collect_vla_1p2.sbatch <task> <start_seed> <n_seeds> <out_root>
#SBATCH --job-name=hri_col12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=job_logs/hri_col12_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python
export GENESIS_BACKEND=cpu
TASK="${1:?}"; START="${2:?}"; N="${3:?}"; OUT="${4:?}"
echo "=== node $(hostname) task=$TASK seeds=$START..$((START+N-1)) ==="
for ((s=START; s<START+N; s++)); do
  "$PY" scripts/vla_data/collect_vla.py --task "$TASK" --seed "$s" --mode record \
    --out-root "$OUT/$TASK/rasterizer" --randomize-avatar || echo "[warn] seed $s nonzero"
done
echo "=== done ==="
