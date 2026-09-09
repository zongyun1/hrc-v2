#!/usr/bin/env bash
#SBATCH --job-name=robocasa-native
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/robocasa_native_%j.log
set -euo pipefail
cd /scratch/jiabenchen_umass/yz/hrc-v2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
tasks=("${@}")
if [ ${#tasks[@]} -eq 0 ]; then tasks=(NavigateKitchen); fi
extra=()
if [ "${ROBOCASA_NATIVE_VIDEO:-0}" = 1 ]; then extra+=(--video); fi
for task in "${tasks[@]}"; do
    .venv-robocasa/bin/python robocasa_migration/run_native.py --task "$task" \
        --output "outputs/robocasa_migration/native/${SLURM_JOB_ID}" \
        --speed-cap "${ROBOCASA_NATIVE_SPEED_CAP:-.6}" --waypoint-tolerance "${ROBOCASA_NATIVE_WAYPOINT_TOLERANCE:-.12}" \
        --steps "${ROBOCASA_NATIVE_STEPS:-840}" --policy "${ROBOCASA_NATIVE_POLICY:-navigate}" "${extra[@]}"
done
