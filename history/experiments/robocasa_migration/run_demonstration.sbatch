#!/usr/bin/env bash
#SBATCH --job-name=tea-demo-native
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/tea_demo_%A_%a.log
set -euo pipefail
cd /scratch/jiabenchen_umass/yz/hrc-v2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
episode="${ROBOCASA_DEMO_EPISODE:-${SLURM_ARRAY_TASK_ID:-0}}"
printf -v episode_dir 'episode_%06d' "$episode"
extra=()
if [ "${ROBOCASA_DEMO_VIDEO:-0}" = 1 ]; then extra+=(--video); fi
.venv-robocasa/bin/python robocasa_migration/run_demonstration.py \
    --episode "$episode" --export "outputs/robocasa_migration/demonstrations/$episode_dir" "${extra[@]}"
