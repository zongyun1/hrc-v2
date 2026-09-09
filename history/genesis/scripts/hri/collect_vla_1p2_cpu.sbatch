#!/bin/bash
# !!! DO NOT USE FOR RECORD-MODE COLLECTION !!!
# Record mode renders camera images via the rasterizer, which needs an EGL/GL
# context = a GPU. On the cpu partition scene.build() dies with
# "EGLError: No EGL context could be initialized" and EVERY episode fails at
# task_init. GENESIS_BACKEND=cpu only moves PHYSICS to CPU; image capture still
# needs a GPU. Use scripts/hri/collect_vla_1p2.sbatch (gpu partition) instead.
# Kept only as a cautionary record. (verdict-only mode w/o capture might work on
# cpu, untested.)
#
# Record-mode VLA demo collection on Genesis 1.2.0 (MAWM_latest_genesis), CPU
# partition.
# Usage: sbatch scripts/hri/collect_vla_1p2_cpu.sbatch <task> <start_seed> <n_seeds> <out_root>
#SBATCH --job-name=hri_col12
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
