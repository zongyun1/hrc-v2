#!/bin/bash
# Train ACT on fresh 1.2.0 demos, bypassing the migration-broken scripts/env.sh.
# The one_task pipeline (RLDS build + LeRobot convert + ACT train) is all
# lerobot v0.1.0 + tf; the pi0 env provides both. No genesis needed.
#
# Usage: sbatch --export=ALL,ACT_RAW_DATA_ROOT=<dir>,ACT_TARGET_EPISODES=80,\
#   ACT_RUN_ROOT=<dir>,ACT_RUN_VERSION=hri1p2 scripts/hri/train_act_1p2.sbatch <task>
#SBATCH --job-name=hri_act1p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=job_logs/hri_act1p2_%j.log
#SBATCH --error=job_logs/hri_act1p2_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

PI0=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python
export MAWM_PY="$PI0"          # RLDS builder (needs tf/tfds) — pi0 has both
export LEROBOT_PY="$PI0"       # convert + train (lerobot v0.1.0)
export PI0_PY="$PI0"
export WANDB_MODE=offline
TASK="${1:?task}"
export ACT_TARGET_EPISODES="${ACT_TARGET_EPISODES:-80}"
export ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-8}"   # serial loader = ~3s/step; parallelize
export ACT_STEPS="${ACT_STEPS:-15000}"           # BC on ~80 demos converges well before 25k

echo "=== node $(hostname) job $SLURM_JOB_ID task=$TASK target=$ACT_TARGET_EPISODES ==="
n=$(ls "$ACT_RAW_DATA_ROOT/$TASK/rasterizer" 2>/dev/null | grep -c seed_ || true)
echo "raw episodes available: $n"
bash scripts/train_act_qpos_target_20260524_one_task.sh "$TASK"
echo "=== training done ==="
