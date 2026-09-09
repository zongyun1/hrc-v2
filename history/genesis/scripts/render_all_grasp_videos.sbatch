#!/bin/bash
#SBATCH --job-name=all_grasp_videos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-28%8
#SBATCH -o job_logs/all_grasp_videos_%A_%a.log
#SBATCH -e job_logs/all_grasp_videos_%A_%a.log

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench}"
OUTPUT_DIR=${1:-data/object/grasp_videos_all}
mkdir -p job_logs "${OUTPUT_DIR}"

PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python
PROFILES=(
  stapler_m0
  jam_jar_m0
  cube_40mm
  milk_box_m0
  apple_m1
  woodenblock_m1
  seal_m1
  rubikscube_m1
  coffee_box_m0
  tea_box_m0
  tissue_box_m0
  rubikscube_m0
  woodenblock_m0
  shoe_m0
  bowl_m1
  bread_m0
  bread_m1
  olive_oil_m0
  mug_m0
  knife_m0
  hamburger_m5
  french_fries_m0
  skillet_m0
  apple_m0
  screwdriver_m0
  hammer_m0
  block_36mm
  block_48mm
  block_60mm
)

PROFILE=${PROFILES[$SLURM_ARRAY_TASK_ID]}
echo "profile=${PROFILE} job=${SLURM_JOB_ID} host=$(hostname)"
srun "${PY}" scripts/object_pool_grasp_demo.py \
  --profile "${PROFILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --hold-steps 100
