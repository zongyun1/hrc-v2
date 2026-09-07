#!/bin/bash
# Train Diffusion Policy (Chi et al. 2023) from scratch on a per-task
# genesis_hr_bench zarr replay buffer.
#
# DP is per-task — there is no foundation pretraining to fine-tune from.
# This launcher just wires the upstream `train.py` to our task yaml + workspace
# yaml.
#
# Usage:
#   # 1. Convert collected episodes to a zarr replay buffer
#   scripts/build_diffusion_policy_zarr.py is run automatically below if the
#   zarr does not yet exist.
#
#   # 2. Sanity train (5 epochs, batch 8, wandb disabled) — confirms the data
#   #    path + GPU + workspace all wire up before committing to a long run.
#   scripts/finetune_diffusion_policy.sh --task pour_water --sanity
#
#   # 3. Real train (1000 epochs, batch 64) — rule of thumb 4-12 h on L40S.
#   scripts/finetune_diffusion_policy.sh --task pour_water --full
#
#   # 4. Forward extra hydra overrides after `--`:
#   scripts/finetune_diffusion_policy.sh --task pour_water --full -- \
#       training.num_epochs=2000 dataloader.batch_size=128
#
# Output:
#   runs/diffusion_policy/finetune/<task>/<timestamp>_<exp_name>/{checkpoints/, ...}
#   Zarr replay buffer stays at runs/diffusion_policy/<task>/<task>.zarr (data
#   path, not changed by this layout — rebuild costs ~minutes per task).
#
# Requirements:
#   * Ampere+ GPU (>= 16 GB) — sanity fits on 12 GB.
#   * vla_data/<task>/<renderer>/seed_*/ already collected.
#   * yz/env/diffusion_policy populated (see docs/diffusion-policy-finetune-portable.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Path to the diffusion_policy env's python. Override on a different machine
# with `DP_PY=/path/to/python scripts/finetune_diffusion_policy.sh ...`. See
# docs/diffusion-policy-baseline.md for env setup.
ENV_PY="${DP_PY:-/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/diffusion_policy/bin/python}"
TRAIN_PY="$REPO_ROOT/baseline/diffusion_policy/train.py"

TASK=""
MODE=""
RENDERER="rasterizer"
IMAGE_SIZE=96
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)       TASK="$2"; shift 2 ;;
    --renderer)   RENDERER="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --sanity)     MODE="sanity"; shift ;;
    --full)       MODE="full"; shift ;;
    --)           shift; EXTRA+=("$@"); break ;;
    -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to train.py)" >&2; exit 2 ;;
  esac
done

if [[ -z "$TASK" || -z "$MODE" ]]; then
  echo "required: --task NAME and one of {--sanity, --full}" >&2; exit 2
fi
if [[ ! -x "$ENV_PY" ]]; then
  echo "[finetune] cannot find diffusion_policy python at $ENV_PY" >&2; exit 1
fi

# Build zarr if missing.
ZARR_DIR="$REPO_ROOT/runs/diffusion_policy/$TASK/$TASK.zarr"
if [[ ! -d "$ZARR_DIR" ]]; then
  echo "[finetune] zarr not found at $ZARR_DIR — building from vla_data/$TASK/$RENDERER/"
  "$ENV_PY" scripts/build_diffusion_policy_zarr.py \
    --task "$TASK" --renderer "$RENDERER" --image-size "$IMAGE_SIZE"
fi

# Mode-specific defaults — extras after `--` override these.
if [[ "$MODE" == "sanity" ]]; then
  NUM_EPOCHS=5
  BATCH_SIZE=8
  CHECKPOINT_EVERY=5
  VAL_EVERY=1
  WANDB_MODE=disabled
  EXP_NAME=sanity
else
  NUM_EPOCHS=1000
  BATCH_SIZE=64
  CHECKPOINT_EVERY=50
  VAL_EVERY=1
  WANDB_MODE=${WANDB_MODE:-disabled}
  EXP_NAME=full
fi

# wandb entity: default to the lab team that this machine's WANDB_API_KEY
# (zhouqqhh) belongs to. Override with `WANDB_ENTITY=<team>` for runs from a
# different account — note that personal entities are disabled on at least
# zong0043, so the override must be a team slug.
WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"

echo "==============================================================="
echo "[finetune] task            = $TASK"
echo "[finetune] mode            = $MODE"
echo "[finetune] zarr            = $ZARR_DIR"
echo "[finetune] num_epochs      = $NUM_EPOCHS"
echo "[finetune] batch_size      = $BATCH_SIZE"
echo "[finetune] checkpoint_every= $CHECKPOINT_EVERY"
echo "[finetune] wandb_mode      = $WANDB_MODE"
echo "[finetune] wandb_entity    = $WANDB_ENTITY"
echo "==============================================================="

export WANDB_MODE
# train.py uses Python's tee to /dev/null buffering tricks at line 9; nothing
# else env-wise to set.

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$REPO_ROOT/runs/diffusion_policy/finetune/$TASK/${TIMESTAMP}_${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "[finetune] output_dir      = $OUTPUT_DIR"

"$ENV_PY" "$TRAIN_PY" \
    --config-name=train_diffusion_unet_image_genesis_hr_bench_workspace \
    task.dataset.zarr_path="$ZARR_DIR" \
    hydra.run.dir="$OUTPUT_DIR" \
    training.num_epochs="$NUM_EPOCHS" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    training.val_every="$VAL_EVERY" \
    exp_name="$EXP_NAME" \
    ++logging.entity="$WANDB_ENTITY" \
    "${EXTRA[@]}"
