#!/bin/bash
# Baseline task-native grasp/lift/hold pass over the categorize object pool.
# Usage: sbatch scripts/categorize_grasp_sweep.sbatch [output_dir] [seed]
#SBATCH --job-name=cate_grasp
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080_ti|l40s|a16|a40|l4|rtx_8000|a4000
#SBATCH --exclude=uri-gpu[001-017],umd-cscdr-gpu[001-002],gpu013,gpu014,gpu015,gpu017,gpu018,gpu019,gpu020,gpu021,gpu051,gypsum-gpu151,gypsum-gpu055
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-7%8
#SBATCH --output=job_logs/cate_grasp_%A_%a.log
#SBATCH --error=job_logs/cate_grasp_%A_%a.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench}"

PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python
OUT=${1:-data/categorize/grasp_sweep/baseline}
SEED=${2:-0}
OBJECTS=(
  048_stapler@0
  031_jam-jar@0
  cube
  038_milk-box@0
  035_apple@1
  086_woodenblock@1
  100_seal@1
  073_rubikscube@1
)
OBJECT=${OBJECTS[$SLURM_ARRAY_TASK_ID]}

mkdir -p job_logs "$OUT"
export GENESIS_BACKEND=cpu
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

echo "object=$OBJECT seed=$SEED host=$(hostname) output=$OUT"
srun "$PY" scripts/categorize_grasp_sweep.py \
  --object "$OBJECT" \
  --seed "$SEED" \
  --output-dir "$OUT"
