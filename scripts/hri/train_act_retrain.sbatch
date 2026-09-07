#!/bin/bash
# Train one per-task ACT on the freshly collected v2 demos using the proven
# qpos_target recipe (300 eps, chunk 100, 25k steps, novae).
# Usage: sbatch [--export=ALL,ACT_TARGET_EPISODES=200] \
#   scripts/hri/train_act_retrain.sbatch <task>

#SBATCH --job-name=hri_act_train
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=job_logs/hri_act_train_%j.log
#SBATCH --error=job_logs/hri_act_train_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/env.sh

TASK="${1:?task}"
# The one_task script's converter + trainer are lerobot v0.1.0 code; it
# resolves its python as ${LEROBOT_PY:-${PI0_PY:-pi0 venv}} and scripts/env.sh
# sets LEROBOT_PY to the v0.5.2 venv — force the pi0 (v0.1.0) interpreter.
export LEROBOT_PY="/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python"
export ACT_RAW_DATA_ROOT="${ACT_RAW_DATA_ROOT:-$PWD/data/hri_manicast/vla_data_retrain}"
export ACT_RUN_ROOT="${ACT_RUN_ROOT:-$PWD/runs/act/hri_retrain_20260611}"
export ACT_RUN_VERSION="${ACT_RUN_VERSION:-v2}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== node $(hostname) job $SLURM_JOB_ID task=$TASK ==="
n=$(ls "$ACT_RAW_DATA_ROOT/$TASK/rasterizer" 2>/dev/null | wc -l)
echo "episodes available: $n (target ${ACT_TARGET_EPISODES:-300})"

bash scripts/train_act_qpos_target_20260524_one_task.sh "$TASK"
echo "=== training done ==="
