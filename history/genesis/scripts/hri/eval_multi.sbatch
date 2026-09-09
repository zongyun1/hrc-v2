#!/bin/bash
# Run several ACT evals sequentially in one GPU job (server restart per eval).
# Usage:
#   sbatch scripts/hri/eval_multi.sbatch \
#     "<task>|<ckpt>|<out_dir>|<extra eval_vla flags>" [more specs...]
# Common knobs come from env: EVAL_EPISODES (3), EVAL_MAX_STEPS (600).

#SBATCH --job-name=hri_meval
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=job_logs/hri_meval_%j.log
#SBATCH --error=job_logs/hri_meval_%j.err

set -o pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_ROOT"
mkdir -p job_logs
export SERVER_LOG_DIR="runs/act/eval/server_logs"
export GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"
export ACT_SERVER_ENV="${ACT_SERVER_ENV:-lerobot}"
EPISODES="${EVAL_EPISODES:-3}"
MAX_STEPS="${EVAL_MAX_STEPS:-600}"
COMMON_FLAGS="${EVAL_COMMON_FLAGS:---interpolate-qpos-actions --qpos-gripper-max-delta 0.08 --qpos-gripper-close-hold-steps 25}"
PORT=$(( 8900 + SLURM_JOB_ID % 50 ))

echo "=== node $(hostname) job $SLURM_JOB_ID specs: $# ==="
i=0
for spec in "$@"; do
  i=$((i+1))
  IFS='|' read -r task ckpt outdir extra <<< "$spec"
  echo "--- [$i] task=$task ckpt=$ckpt out=$outdir extra=$extra"
  mkdir -p "$outdir"
  # shellcheck disable=SC2086
  scripts/run_vla_eval.sh \
    --model act --checkpoint "$ckpt" \
    --task "$task" --episodes "$EPISODES" --max-steps "$MAX_STEPS" \
    --action-type "${EVAL_ACTION_TYPE:-qpos_abs}" --port $((PORT + i)) \
    --video-dir "$outdir/videos" \
    -- \
    --output-json "$outdir/results.json" \
    --trace-actions-dir "$outdir/traces" \
    --video-limit "${EVAL_VIDEO_LIMIT:-1}" --video-stride 10 \
    $COMMON_FLAGS \
    --success-hold-steps 1 \
    $extra || echo "[warn] spec $i exited nonzero"
done
echo "=== all specs done ==="
