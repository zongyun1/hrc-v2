#!/bin/bash
# Train 3D Diffusion Policy (Ze et al. 2024) from scratch on a per-task
# genesis_hr_bench point-cloud zarr.
#
# Like the original Diffusion Policy, DP3 is per-task — there is no
# foundation pretraining to fine-tune from. This launcher just wires the
# upstream `train.py` to our task yaml.
#
# Usage:
#   # 1. Convert collected episodes to a DP3 zarr (auto-built below if missing)
#   scripts/build_dp3_zarr.py is run automatically below if needed.
#
#   # 2. Sanity train (5 epochs, batch 8, wandb disabled).
#   scripts/finetune_dp3.sh --task pour_water --sanity
#
#   # 3. Real train (3000 epochs, batch 128 — DP3 paper defaults).
#   scripts/finetune_dp3.sh --task pour_water --full
#
#   # 4. Forward extra hydra overrides after `--`:
#   scripts/finetune_dp3.sh --task pour_water --full -- training.num_epochs=2000
#
# Output:
#   runs/dp3/finetune/<task>/<timestamp>_<exp_name>/{checkpoints/, .hydra/, logs.json.txt}
#   Zarr point-cloud buffer stays at runs/dp3/<task>/<task>.zarr (data path,
#   not changed by this layout — rebuild costs ~minutes per task).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_PY="${DP3_PY:-/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/dp3/bin/python}"
TRAIN_PY="$REPO_ROOT/baseline/3d_diffusion_policy/3D-Diffusion-Policy/train.py"
TRAIN_DIR="$REPO_ROOT/baseline/3d_diffusion_policy/3D-Diffusion-Policy"

TASK=""
MODE=""
RENDERER="rasterizer"
N_POINTS=1024
USE_COLOR=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)       TASK="$2"; shift 2 ;;
    --renderer)   RENDERER="$2"; shift 2 ;;
    --n-points)   N_POINTS="$2"; shift 2 ;;
    --use-color)  USE_COLOR=1; shift ;;
    --sanity)     MODE="sanity"; shift ;;
    --full)       MODE="full"; shift ;;
    --)           shift; EXTRA+=("$@"); break ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to train.py)" >&2; exit 2 ;;
  esac
done

if [[ -z "$TASK" || -z "$MODE" ]]; then
  echo "required: --task NAME and one of {--sanity, --full}" >&2; exit 2
fi
if [[ ! -x "$ENV_PY" ]]; then
  echo "[finetune_dp3] cannot find dp3 python at $ENV_PY" >&2; exit 1
fi

# Build zarr if missing.
ZARR_DIR="$REPO_ROOT/runs/dp3/$TASK/$TASK.zarr"
if [[ ! -d "$ZARR_DIR" ]]; then
  echo "[finetune_dp3] zarr not found at $ZARR_DIR — building from vla_data/$TASK/$RENDERER/"
  COLOR_FLAG=""; [[ "$USE_COLOR" == "1" ]] && COLOR_FLAG="--use-color"
  "$ENV_PY" scripts/build_dp3_zarr.py \
      --task "$TASK" --renderer "$RENDERER" --n-points "$N_POINTS" $COLOR_FLAG
fi

if [[ "$MODE" == "sanity" ]]; then
  NUM_EPOCHS=5
  BATCH_SIZE=8
  CHECKPOINT_EVERY=5
  VAL_EVERY=1
  WANDB_MODE=disabled
  EXP_NAME=sanity
  ES_PATIENCE=0       # disabled — sanity is too short to benefit
  ES_MIN_DELTA=0.0
else
  NUM_EPOCHS=3000
  BATCH_SIZE=128
  CHECKPOINT_EVERY=200
  VAL_EVERY=1
  WANDB_MODE=${WANDB_MODE:-disabled}
  EXP_NAME=full
  # Early stopping: 0 disables. min_delta=0 means "any new val_loss low counts
  # as improvement" — calibration-free choice for first runs. Stops only after
  # `patience` consecutive epochs with no new low. Override at submit time via
  # `-- training.early_stopping_patience=N training.early_stopping_min_delta=X`.
  ES_PATIENCE=${ES_PATIENCE:-100}
  ES_MIN_DELTA=${ES_MIN_DELTA:-0.0}
fi
WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"

# DP3's `task=genesis_hr_bench` resolves to our vendored
# diffusion_policy_3d/config/task/genesis_hr_bench.yaml. Output dir is
# rooted under runs/dp3/<task>/ so all artefacts stay inside the repo
# (no /tmp; matches feedback memory).
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$REPO_ROOT/runs/dp3/finetune/$TASK/${TIMESTAMP}_${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "==============================================================="
echo "[finetune_dp3] task            = $TASK"
echo "[finetune_dp3] mode            = $MODE"
echo "[finetune_dp3] zarr            = $ZARR_DIR"
echo "[finetune_dp3] output_dir      = $OUTPUT_DIR"
echo "[finetune_dp3] num_epochs      = $NUM_EPOCHS"
echo "[finetune_dp3] batch_size      = $BATCH_SIZE"
echo "[finetune_dp3] checkpoint_every= $CHECKPOINT_EVERY"
echo "[finetune_dp3] wandb_mode      = $WANDB_MODE"
echo "==============================================================="

export WANDB_MODE
export HYDRA_FULL_ERROR=1

# train.py expects to run from the 3D-Diffusion-Policy/ dir (it sys.path
# inserts the parent of __file__). cd there explicitly.
cd "$TRAIN_DIR"

"$ENV_PY" train.py --config-name=dp3.yaml \
    task=genesis_hr_bench \
    task.dataset.zarr_path="$ZARR_DIR" \
    hydra.run.dir="$OUTPUT_DIR" \
    training.num_epochs="$NUM_EPOCHS" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    training.val_every="$VAL_EVERY" \
    checkpoint.save_ckpt=True \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    "checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt'" \
    "+training.early_stopping_patience=$ES_PATIENCE" \
    "+training.early_stopping_min_delta=$ES_MIN_DELTA" \
    exp_name="$EXP_NAME" \
    logging.mode="$WANDB_MODE" \
    "++logging.entity=$WANDB_ENTITY" \
    "${EXTRA[@]}"
