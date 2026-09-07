#!/bin/bash
# Fine-tune OpenVLA-7B with OFT on the genesis_hr_bench TFDS dataset.
#
# Usage:
#   scripts/finetune_openvla_oft.sh --sanity                 # 2000-step smoke
#   scripts/finetune_openvla_oft.sh --full                   # 50000-step run
#   GPUS_PER_NODE=4 scripts/finetune_openvla_oft.sh --full    # 4-GPU, one node
#   NNODES=2 GPUS_PER_NODE=8 scripts/finetune_openvla_oft.sh --full   # 16-GPU
#   scripts/finetune_openvla_oft.sh --sanity -- --batch_size 1   # forward flags
#
# Distributed shape comes from the NNODES x GPUS_PER_NODE env contract
# (see scripts/_dist.sh); launched via torchrun.
#
# Requirements:
#   * Ampere+ GPU (A100 / H100 / L40S / A40); TITAN X dev node will not work.
#   * Cold start downloads openvla/openvla-7b (~14 GB) into checkpoints/.
#   * Dataset at openvla/data/genesis_hr_bench/1.0.0/
#     (rebuild with: scripts/build_genesis_hr_bench_rlds.py).
#
# Output:
#   * Checkpoints + adapter at runs/openvla_oft/finetune/<run_id>/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=_dist.sh
source "$HERE/_dist.sh"
cd "$REPO_ROOT"

ENV_PY="$OFT_PY"
FINETUNE_PY="$REPO_ROOT/baseline/openvla-oft/vla-scripts/finetune.py"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/openvla/data}"
RUN_ROOT="$REPO_ROOT/runs/openvla_oft/finetune"

[[ -x "$ENV_PY" ]]      || { echo "[finetune] cannot find OFT python at $ENV_PY" >&2; exit 1; }
[[ -f "$FINETUNE_PY" ]] || { echo "[finetune] cannot find finetune.py at $FINETUNE_PY" >&2; exit 1; }
if [[ ! -d "$DATA_ROOT/genesis_hr_bench/1.0.0" ]]; then
  echo "[finetune] dataset missing: $DATA_ROOT/genesis_hr_bench/1.0.0" >&2
  echo "[finetune] rebuild with: scripts/build_genesis_hr_bench_rlds.py" >&2
  exit 1
fi

MODE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sanity) MODE="sanity"; shift ;;
    --full)   MODE="full";   shift ;;
    --)       shift; EXTRA+=("$@"); break ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to finetune.py)" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "must pass --sanity or --full" >&2; exit 2; }

if [[ "$MODE" == "sanity" ]]; then
  MAX_STEPS=2000;  WARMUP_DECAY=1000;  SAVE_FREQ=1000; SHUFFLE=5000
  RUN_NOTE="${RUN_NOTE:-sanity_pour_water}"
else
  MAX_STEPS=50000; WARMUP_DECAY=30000; SAVE_FREQ=5000; SHUFFLE=20000
  RUN_NOTE="${RUN_NOTE:-genesis_hr_bench_pour_water}"
fi

mkdir -p "$RUN_ROOT"

# torchrun launch prefix from the NNODES x GPUS_PER_NODE contract.
TORCHRUN_BIN="${ENV_PY%/python}/torchrun"
dist_setup torchrun "$TORCHRUN_BIN"

echo "==============================================================="
echo "[finetune] mode      = $MODE"
echo "[finetune] dist      = $DIST_SUMMARY"
echo "[finetune] max_steps = $MAX_STEPS"
echo "[finetune] save_freq = $SAVE_FREQ"
echo "[finetune] data_root = $DATA_ROOT"
echo "[finetune] run_root  = $RUN_ROOT"
echo "[finetune] note      = $RUN_NOTE"
echo "==============================================================="

# HF_HUB_CACHE: keep the openvla-7b download next to other baselines' weights.
# TOKENIZERS_PARALLELISM: silence noisy fork warnings under torchrun.
export HF_HUB_CACHE="$REPO_ROOT/checkpoints"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

# Wandb: finetune.py calls wandb.init() unconditionally on the main process
# (vla-scripts/finetune.py:792-793, no guard), so empty entity/project would
# fail unless WANDB_MODE=disabled silences it. Defaults here:
#   - sanity: WANDB_MODE=disabled (smoke runs aren't worth tracking).
#   - full:   WANDB_MODE=online with the shared lab team + per-baseline project.
# Override either by exporting WANDB_MODE / WANDB_ENTITY / WANDB_PROJECT
# before invoking this script. (Mirror of finetune_rdt.sh:77-82 / finetune_lerobot_v2.sh:91-101.)
if [[ "$MODE" == "sanity" ]]; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  : "${WANDB_ENTITY:=}"
  : "${WANDB_PROJECT:=}"
else
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"
  export WANDB_PROJECT="${WANDB_PROJECT:-openvla_oft_genesis_hr_bench}"
fi
echo "[finetune_openvla_oft] wandb_mode    = $WANDB_MODE"
echo "[finetune_openvla_oft] wandb_entity  = ${WANDB_ENTITY:-(none)}"
echo "[finetune_openvla_oft] wandb_project = ${WANDB_PROJECT:-(none)}"

# finetune.py's check_model_logic_mismatch() does os.walk("./prismatic/") to
# sync modeling_prismatic.py into the HF snapshot. That path only resolves
# from baseline/openvla-oft/, so cd there before launching.
cd "$REPO_ROOT/baseline/openvla-oft"

exec "${DIST_LAUNCHER[@]}" \
    "$FINETUNE_PY" \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name genesis_hr_bench \
    --run_root_dir "$RUN_ROOT" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay "$WARMUP_DECAY" \
    --max_steps "$MAX_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --shuffle_buffer_size "$SHUFFLE" \
    --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" \
    --run_id_note "$RUN_NOTE" \
    "${EXTRA[@]}"
