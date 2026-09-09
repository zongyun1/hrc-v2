#!/bin/bash
# Train one per-task ACT on the freshly collected Genesis-1.2.0 demos, using the
# proven qpos_target recipe (chunk 100, 25k steps, novae) but WITHOUT sourcing
# the migration-broken scripts/env.sh (which `exit 1`s trying to activate the
# nonexistent .venv-genesis-latest and would kill this job under `set -e`).
# This is the training analogue of scripts/hri/eval_1p2.sh.
#
# The converter + trainer are lerobot v0.1.0 code (pi0 venv) -- no genesis /
# MAWM_latest_genesis needed for training; the demos are already recorded.
#
# Usage:
#   sbatch [--export=ALL,ACT_TARGET_EPISODES=80] \
#     scripts/hri/train_1p2.sbatch <task> [<raw_data_root>]
#   default task     = place_burger_fries_interrupt
#   default raw root  = data/hri_manicast/vla_1p2_burger

#SBATCH --job-name=hri_act_train_1p2
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=job_logs/hri_act_train_1p2_%j.log
#SBATCH --error=job_logs/hri_act_train_1p2_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASK="${1:-place_burger_fries_interrupt}"

# pi0 venv ships lerobot v0.1.0 (the qpos_target converter + train entry point).
export LEROBOT_PY="/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python"
# MAWM_PY runs only the RLDS build step (build_genesis_hr_bench_rlds.py) which
# just re-reads the already-recorded steps.h5 + images into a tfds dataset -- it
# needs tensorflow + tensorflow_datasets + transforms3d + h5py but NOT genesis,
# so use the MAWM benchmark venv (has TF 2.21 + all of them). MAWM_latest_genesis
# lacks tensorflow, which is why the env.sh-free first attempt died at RLDS build.
export MAWM_PY="${MAWM_PY:-/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM/bin/python}"
export ACT_RAW_DATA_ROOT="${ACT_RAW_DATA_ROOT:-$PWD/${2:-data/hri_manicast/vla_1p2_burger}}"
export ACT_RUN_ROOT="${ACT_RUN_ROOT:-$PWD/runs/act/hri_1p2_burger}"
export ACT_RUN_VERSION="${ACT_RUN_VERSION:-v1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

n=$(ls "$ACT_RAW_DATA_ROOT/$TASK/rasterizer" 2>/dev/null | wc -l)
echo "=== node $(hostname) job ${SLURM_JOB_ID:-local} task=$TASK ==="
echo "raw_data_root: $ACT_RAW_DATA_ROOT"
echo "run_root:      $ACT_RUN_ROOT"
echo "episodes available: $n (target ${ACT_TARGET_EPISODES:-300})"

bash scripts/train_act_qpos_target_20260524_one_task.sh "$TASK"
echo "=== training done ==="
