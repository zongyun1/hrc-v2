#!/bin/bash
# Generic fine-tuning driver for LeRobot imitation policies (ACT,
# Diffusion, VQ-BeT) on the genesis-hr-bench LeRobot dataset (built by
# scripts/convert_genesis_hr_bench_to_lerobot.py --lerobot-naming).
#
# Usage:
#   scripts/finetune_lerobot.sh --policy-type act      --sanity
#   scripts/finetune_lerobot.sh --policy-type vqbet    --full
#   scripts/finetune_lerobot.sh --policy-type diffusion --full --exp-name my_run -- \
#       --batch_size=16
#
# What this wraps:
#   * python -m lerobot.scripts.train --policy.type=<type> \
#       --dataset.repo_id=genesis-hr-bench/genesis_hr_bench_lerobot \
#       --output_dir=runs/<baseline>/finetune/<exp>
#
# Requirements:
#   * pi0 venv at yz/env/pi0 (already ships lerobot==0.1.0 with all three
#     policy types). Override with LEROBOT_PY env var if needed.
#   * Dataset materialized at
#     $HF_LEROBOT_HOME/genesis-hr-bench/genesis_hr_bench_lerobot/.
#   * GPU recommended.
#
# Output:
#   * runs/<baseline>/finetune/<exp-name>/checkpoints/<step>/pretrained_model/
#   * Latest symlinked to runs/<baseline>/finetune/<exp-name>/checkpoints/last/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_PY_DEFAULT="/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python"
ENV_PY="${LEROBOT_PY:-$ENV_PY_DEFAULT}"

POLICY_TYPE=""
MODE=""
EXP_NAME=""
EXTRA=()

usage() { sed -n '2,28p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-type) POLICY_TYPE="$2"; shift 2 ;;
    --sanity)      MODE="sanity"; shift ;;
    --full)        MODE="full"; shift ;;
    --exp-name)    EXP_NAME="$2"; shift 2 ;;
    --)            shift; EXTRA+=("$@"); break ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to lerobot train)" >&2; exit 2 ;;
  esac
done

case "$POLICY_TYPE" in
  act|diffusion|vqbet) ;;
  *) echo "required: --policy-type {act|diffusion|vqbet}" >&2; exit 2 ;;
esac

case "$MODE" in
  sanity|full) ;;
  *) echo "required: --sanity or --full" >&2; exit 2 ;;
esac

if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

if [[ "$MODE" == "sanity" ]]; then
  STEPS=200
  BATCH=8
  SAVE_FREQ=200
  LOG_FREQ=10
  WANDB_FLAGS=(--wandb.enable=false)
else
  STEPS=100000
  BATCH=32
  SAVE_FREQ=5000
  LOG_FREQ=100
  WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"
  WANDB_PROJECT="${WANDB_PROJECT:-${POLICY_TYPE}_genesis_hr_bench}"
  WANDB_FLAGS=(--wandb.enable=true
               --wandb.entity="$WANDB_ENTITY"
               --wandb.project="$WANDB_PROJECT")
fi

# Map lerobot's --policy-type back to the baseline name used in the rest of
# the repo. lerobot calls its Diffusion-Policy baseline "diffusion", but the
# eval registry (run_vla_eval.sh, VLA_REGISTRY) names it "lerobot_diffusion"
# to disambiguate from Chi et al.'s standalone diffusion_policy. Keep both
# names aligned by using the registry name for the output dir.
case "$POLICY_TYPE" in
  diffusion) BASELINE_NAME=lerobot_diffusion ;;
  *)         BASELINE_NAME="$POLICY_TYPE" ;;
esac
OUTPUT_DIR="$REPO_ROOT/runs/${BASELINE_NAME}/finetune/$EXP_NAME"
# Per-policy datasets can deviate from the default 2-camera dataset (e.g.
# VQ-BeT needs a 1-camera variant). Override via DATASET_REPO_ID env var.
DATASET_REPO_ID="${DATASET_REPO_ID:-genesis-hr-bench/genesis_hr_bench_lerobot}"

echo "==============================================================="
echo "[finetune_lerobot] policy   = $POLICY_TYPE"
echo "[finetune_lerobot] mode     = $MODE  (steps=$STEPS, batch=$BATCH)"
echo "[finetune_lerobot] exp-name = $EXP_NAME"
echo "[finetune_lerobot] python   = $ENV_PY"
echo "[finetune_lerobot] output   = $OUTPUT_DIR"
echo "==============================================================="

if "$ENV_PY" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("lerobot.scripts.train") else 1)
PY
then
  TRAIN_MODULE="lerobot.scripts.train"
else
  TRAIN_MODULE="lerobot.scripts.lerobot_train"
fi

RESUME_FLAG=()
if [[ -d "$OUTPUT_DIR/checkpoints/last" ]]; then
  RESUME_FLAG=(--resume=true)
  echo "[finetune_lerobot] existing checkpoint detected; resuming."
fi

HAS_POLICY_PATH=0
for arg in "${EXTRA[@]}"; do
  case "$arg" in
    --policy.path|--policy.path=*) HAS_POLICY_PATH=1 ;;
  esac
done

TRAIN_ARGS=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --output_dir="$OUTPUT_DIR"
  --steps="$STEPS"
  --batch_size="$BATCH"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --save_checkpoint=true
  --policy.device=cuda
  "${WANDB_FLAGS[@]}"
  "${RESUME_FLAG[@]}"
)
if [[ "$HAS_POLICY_PATH" != "1" ]]; then
  TRAIN_ARGS=(--policy.type="$POLICY_TYPE" "${TRAIN_ARGS[@]}")
fi
if [[ "$TRAIN_MODULE" == "lerobot.scripts.lerobot_train" ]]; then
  TRAIN_ARGS+=(--policy.push_to_hub=false)
fi
TRAIN_ARGS+=("${EXTRA[@]}")

echo "[finetune_lerobot] launching: $TRAIN_MODULE ${TRAIN_ARGS[*]}"
exec "$ENV_PY" -m "$TRAIN_MODULE" "${TRAIN_ARGS[@]}"
