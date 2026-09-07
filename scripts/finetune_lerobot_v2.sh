#!/bin/bash
# Generic fine-tuning driver for lerobot v0.5.2 VLA policies on the
# genesis-hr-bench LeRobot dataset:  SmolVLA, pi0, pi0.5, pi0-FAST.
#
# All four are HuggingFace lerobot policies trained by the same Accelerator-
# wrapped trainer (lerobot.scripts.lerobot_train), so multi-node / multi-GPU
# is one uniform knob pair: NNODES x GPUS_PER_NODE (see scripts/_dist.sh).
#
# Usage:
#   scripts/finetune_lerobot_v2.sh --policy-type smolvla --sanity
#   scripts/finetune_lerobot_v2.sh --policy-type pi0     --full --exp-name run1
#   NNODES=2 GPUS_PER_NODE=4 scripts/finetune_lerobot_v2.sh --policy-type pi05 --full
#   scripts/finetune_lerobot_v2.sh --policy-type pi0 --full \
#       --pretrained lerobot/pi0 -- --batch_size=4
#
# Knobs:
#   --policy-type    smolvla | pi0 | pi05 | pi0_fast        (required)
#   --sanity|--full  200-step smoke  |  100k-step real run  (required)
#   --exp-name NAME  run name; default <mode>_<jobid|timestamp>
#   --pretrained P   finetune a pretrained checkpoint (--policy.path=P) instead
#                    of a fresh policy (--policy.type=<policy>, lerobot default)
#   --               everything after is forwarded verbatim to lerobot_train
#
# Env:
#   NNODES, GPUS_PER_NODE  distributed shape (default 1x1; see scripts/_dist.sh)
#   DEVICE                 cuda (default) | cpu — set cpu for a no-GPU smoke test
#   DATASET_REPO_ID        override the LeRobot dataset repo id
#   LEROBOT_PY             lerobot venv python (resolved by scripts/env.sh)
#   WANDB_ENTITY/PROJECT   override the W&B target
#
# Multi-GPU note: lerobot's trainer treats --batch_size as PER-PROCESS, so the
# effective global batch is batch_size * NNODES * GPUS_PER_NODE.
#
# Output: runs/<policy>/finetune/<exp-name>/checkpoints/<step>/pretrained_model/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=_dist.sh
source "$HERE/_dist.sh"
cd "$REPO_ROOT"

ENV_PY="$LEROBOT_PY"
DEVICE="${DEVICE:-cuda}"

POLICY=""
MODE=""
EXP_NAME=""
PRETRAINED=""
EXTRA=()

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-type) POLICY="$2"; shift 2 ;;
    --sanity)      MODE="sanity"; shift ;;
    --full)        MODE="full"; shift ;;
    --exp-name)    EXP_NAME="$2"; shift 2 ;;
    --pretrained)  PRETRAINED="$2"; shift 2 ;;
    --)            shift; EXTRA+=("$@"); break ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to lerobot_train)" >&2; exit 2 ;;
  esac
done

case "$POLICY" in
  smolvla|pi0|pi05|pi0_fast) ;;
  *) echo "required: --policy-type {smolvla|pi0|pi05|pi0_fast}" >&2; exit 2 ;;
esac
case "$MODE" in
  sanity|full) ;;
  *) echo "required: --sanity or --full" >&2; exit 2 ;;
esac

[[ -x "$ENV_PY" ]] || { echo "lerobot venv python not found: $ENV_PY" >&2; exit 1; }

if [[ -z "$EXP_NAME" ]]; then
  # Under sbatch, $SLURM_JOB_ID is stable across requeues and identical on
  # every node — same job-id = same output_dir = lerobot --resume works.
  EXP_NAME="${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

# Per-mode hyperparameters. pi0-family backbones (PaliGemma-3B) are far larger
# than SmolVLA (~450M), so they get smaller per-process batches by default.
if [[ "$MODE" == "sanity" ]]; then
  STEPS=200; SAVE_FREQ=200; LOG_FREQ=10
  case "$POLICY" in smolvla) BATCH=4 ;; *) BATCH=2 ;; esac
  WANDB_FLAGS=(--wandb.enable=false)
else
  STEPS=100000; SAVE_FREQ=5000; LOG_FREQ=100
  BATCH=8
  WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"
  # pi0 / pi05 must use separate W&B projects — ${POLICY}_genesis_hr_bench
  # gives each policy its own.
  WANDB_PROJECT="${WANDB_PROJECT:-${POLICY}_genesis_hr_bench}"
  WANDB_FLAGS=(--wandb.enable=true
               --wandb.entity="$WANDB_ENTITY"
               --wandb.project="$WANDB_PROJECT")
fi

OUTPUT_DIR="$REPO_ROOT/runs/${POLICY}/finetune/$EXP_NAME"
# v3.0-format dataset — lerobot v0.5.2 rejects the older v2.1 dataset
# (genesis_hr_bench_lerobot) that the v0.1.0 baselines still use.
DATASET_REPO_ID="${DATASET_REPO_ID:-genesis-hr-bench/genesis_hr_bench_lerobot_v30}"

# Policy source: pretrained checkpoint (--policy.path) or a fresh policy of the
# given type (--policy.type, the lerobot README default).
if [[ -n "$PRETRAINED" ]]; then
  POLICY_ARGS=(--policy.path="$PRETRAINED")
else
  POLICY_ARGS=(--policy.type="$POLICY")
fi
# pi0-family configs default to float32 — impractical for a 3B backbone.
# bf16 is the sane default; override via `-- --policy.dtype=float32`.
case "$POLICY" in
  pi0|pi05|pi0_fast) POLICY_ARGS+=(--policy.dtype=bfloat16) ;;
esac

# Distributed launch prefix (accelerate family — see scripts/_dist.sh).
ACCELERATE_BIN="${ENV_PY%/python}/accelerate"
dist_setup accelerate "$ACCELERATE_BIN"
# DEVICE=cpu -> force accelerate off CUDA (no-GPU smoke testing).
if [[ "$DEVICE" == cpu ]]; then
  DIST_LAUNCHER+=(--cpu)
fi

echo "==============================================================="
echo "[finetune_lerobot_v2] policy   = $POLICY"
echo "[finetune_lerobot_v2] mode     = $MODE  (steps=$STEPS, batch=$BATCH/proc)"
echo "[finetune_lerobot_v2] exp-name = $EXP_NAME"
echo "[finetune_lerobot_v2] python   = $ENV_PY"
echo "[finetune_lerobot_v2] dataset  = $DATASET_REPO_ID"
echo "[finetune_lerobot_v2] output   = $OUTPUT_DIR"
echo "[finetune_lerobot_v2] dist     = $DIST_SUMMARY"
[[ -n "$PRETRAINED" ]] && echo "[finetune_lerobot_v2] pretrained = $PRETRAINED"
echo "==============================================================="

RESUME_FLAG=()
if [[ -d "$OUTPUT_DIR/checkpoints/last" ]]; then
  # lerobot v0.5.2 contract: when --resume=true, --config_path is also required
  # and must point at the train_config.json saved in the most recent checkpoint.
  # Without it, lerobot/configs/train.py:150-168 raises in cfg.validate() with
  # "A config_path is expected when resuming a run."
  RESUME_FLAG=(
    --resume=true
    --config_path="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  )
  echo "[finetune_lerobot_v2] existing checkpoint detected; resuming."
fi

# Detect whether the passthrough EXTRA[] already supplies --dataset.repo_id.
# draccus / lerobot v0.5.2 keeps the FIRST occurrence of a CLI flag, so the
# script-side default below would silently override any user passthrough. This
# bit us with the pi0_base_qpos_* runs being trained on the EE dataset despite
# `-- --dataset.repo_id=...lerobot_qpos_v30` passthrough.
# Workaround: omit the script default when the passthrough supplies its own.
# Users can also set $DATASET_REPO_ID env var to override the default cleanly.
EXTRA_HAS_DATASET=0
for arg in "${EXTRA[@]}"; do
  case "$arg" in --dataset.repo_id=*|--dataset.repo_id) EXTRA_HAS_DATASET=1; break ;; esac
done

TRAIN_ARGS=(
  "${POLICY_ARGS[@]}"
)
(( EXTRA_HAS_DATASET == 0 )) && TRAIN_ARGS+=(--dataset.repo_id="$DATASET_REPO_ID")
TRAIN_ARGS+=(
  --output_dir="$OUTPUT_DIR"
  --steps="$STEPS"
  --batch_size="$BATCH"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --save_checkpoint=true
  # lerobot v0.5.2 validate() aborts unless push_to_hub is off (or a hub
  # repo_id is given) — we save locally, so disable it.
  --policy.push_to_hub=false
  --policy.device="$DEVICE"
  "${WANDB_FLAGS[@]}"
  "${RESUME_FLAG[@]}"
)
TRAIN_ARGS+=("${EXTRA[@]}")

echo "[finetune_lerobot_v2] launching: ${DIST_LAUNCHER[*]} -m lerobot.scripts.lerobot_train ${TRAIN_ARGS[*]}"
exec "${DIST_LAUNCHER[@]}" -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
