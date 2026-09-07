#!/bin/bash
# HRI VLA eval wrapper for this checkout.  Mirrors scripts/eval.sbatch, but
# resolves REPO_ROOT from the script path instead of using a hardcoded checkout.
#
# Usage:
#   sbatch scripts/hri/eval_hri_vla.sbatch <model> [<checkpoint>] [run_vla_eval.sh args...]

#SBATCH --job-name=hri_eval
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="a100|h100|l40s|a40"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=job_logs/hri_eval_%j.log
#SBATCH --error=job_logs/hri_eval_%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/scripts/run_vla_eval.sh" ]]; then
  echo "Could not resolve repo root from SLURM_SUBMIT_DIR=$REPO_ROOT" >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p job_logs

MODEL="${1:?usage: sbatch $0 <model> [<checkpoint>] [extra run_vla_eval.sh args...]}"
shift

CKPT_ARG=()
if [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; then
  CKPT_ARG=(--checkpoint "$1")
  shift
fi

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -5
echo "repo:     $REPO_ROOT"
echo "model:    $MODEL"
echo "ckpt:     ${CKPT_ARG[*]:-(default)}"
echo "extra:    $*"
echo

OUT="runs/${MODEL}/eval/job_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export SERVER_LOG_DIR="runs/${MODEL}/eval/server_logs"
export GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"
export ACT_SERVER_ENV="${ACT_SERVER_ENV:-lerobot}"

exec scripts/run_vla_eval.sh \
    --model "$MODEL" "${CKPT_ARG[@]}" \
    --video-dir "$OUT" \
    "$@"
