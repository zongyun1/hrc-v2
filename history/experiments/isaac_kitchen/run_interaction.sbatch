#!/usr/bin/env bash
#SBATCH --job-name=avatar-c1
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/avatar_c1_%j.log
set -euo pipefail
PROJECT="${KITCHEN_PROJECT_DIR:-/scratch/jiabenchen_umass/yz/hrc-v2}"
export KITCHEN_PROJECT_DIR="$PROJECT"
RESULTS="$PROJECT/outputs/avatar_c1/${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULTS"
failed=0
for condition in legacy_normal c1_normal c1_delay c1_pause; do
    route=c1
    delay=0
    pause=0
    case "$condition" in
        legacy_normal) route=legacy ;;
        c1_delay) delay=180 ;;
        c1_pause) pause=180 ;;
    esac
    export KITCHEN_OUTPUT_DIR="$RESULTS/$condition"
    if ! bash "$PROJECT/isaac_kitchen/run_remote.sbatch" --interaction_route "$route" \
        --robot_delay_steps "$delay" --robot_pause_steps "$pause" \
        > "$RESULTS/$condition.log" 2>&1; then
        failed=1
    fi
    echo "CONDITION_FINISHED $condition"
done
python3 "$PROJECT/isaac_kitchen/summarize_interaction.py" "$RESULTS"
exit "$failed"
