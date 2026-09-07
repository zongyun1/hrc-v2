#!/usr/bin/env bash
# Build one qpos_target_abs LeRobot dataset from vla_data_qpos_target_20260524
# and train one ACT checkpoint for that task.

set -euo pipefail

TASK="${1:?usage: scripts/train_act_qpos_target_20260524_one_task.sh <task>}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAWM_PY="${MAWM_PY:-$REPO_ROOT/.venv/bin/python}"
PI0_PY="${LEROBOT_PY:-${PI0_PY:-/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python}}"
if [[ ! -x "$MAWM_PY" ]]; then
  MAWM_PY="${MAWM_PY_FALLBACK:-python}"
fi
if [[ ! -x "$PI0_PY" ]]; then
  PI0_PY="${PI0_PY_FALLBACK:-python}"
fi

cd "$REPO_ROOT"

RAW_DATA_ROOT="${ACT_RAW_DATA_ROOT:-$REPO_ROOT/vla_data_qpos_target_20260524}"
RUN_ROOT="${ACT_RUN_ROOT:-$REPO_ROOT/runs/act/qpos_target_20260524}"
ACT_CONTROL_MODE="${ACT_CONTROL_MODE:-qpos_target_abs}"
ACT_TARGET_EPISODES="${ACT_TARGET_EPISODES:-300}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-100}"
ACT_TEMPORAL_ENSEMBLE_COEFF="${ACT_TEMPORAL_ENSEMBLE_COEFF:-0.01}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-1}"
ACT_STEPS="${ACT_STEPS:-25000}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-128}"
ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-5000}"
ACT_LOG_FREQ="${ACT_LOG_FREQ:-100}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-0}"
ACT_RUN_ID="${ACT_RUN_ID:-full300_c100_s25000_novae}"
ACT_RUN_VERSION="${ACT_RUN_VERSION:-v1}"
ACT_DATASET_PREFIX="${ACT_DATASET_PREFIX:-ghb}"
ACT_FORCE_REBUILD_DATA="${ACT_FORCE_REBUILD_DATA:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-multi-agent-world-model}"
WANDB_PROJECT="${WANDB_PROJECT:-act_genesis_hr_bench_qpos_target_20260524}"

RUN_SUFFIX="${ACT_CONTROL_MODE}_${ACT_TARGET_EPISODES}ep_${ACT_RUN_ID}"
RAW_ROOT="$RUN_ROOT/raw/${TASK}_${RUN_SUFFIX}"
OPENVLA_DIR="$RUN_ROOT/data/${TASK}/${RUN_SUFFIX}/openvla"
LEROBOT_HOME="$RUN_ROOT/data/${TASK}/${RUN_SUFFIX}/lerobot"
DATASET_REPO_ID="${ACT_DATASET_PREFIX}/${TASK}_qt24_${ACT_TARGET_EPISODES}_c${ACT_CHUNK_SIZE}"
EXP_NAME="act_${TASK}_20260524_${RUN_SUFFIX}_${ACT_RUN_VERSION}"
TRAIN_OUTPUT_DIR="$RUN_ROOT/finetune/$EXP_NAME"

mkdir -p "$RUN_ROOT/raw" "$RUN_ROOT/data/$TASK" "$RUN_ROOT/finetune" "$RUN_ROOT/logs"

echo "=== ACT qpos-target train task=$TASK host=$(hostname) pid=$$ ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -16 || true
fi
echo "raw_data_root:             $RAW_DATA_ROOT"
echo "target_episodes:           $ACT_TARGET_EPISODES"
echo "control_mode:              $ACT_CONTROL_MODE"
echo "chunk_size:                $ACT_CHUNK_SIZE"
echo "n_action_steps:            $ACT_N_ACTION_STEPS"
echo "temporal_ensemble_coeff:   $ACT_TEMPORAL_ENSEMBLE_COEFF"
echo "steps:                     $ACT_STEPS"
echo "batch_size:                $ACT_BATCH_SIZE"
echo "num_workers:               $ACT_NUM_WORKERS"
echo "dataset_repo_id:           $DATASET_REPO_ID"
echo "output:                    $TRAIN_OUTPUT_DIR"
echo

SRC_TASK_DIR="$RAW_DATA_ROOT/$TASK/rasterizer"
if [[ ! -d "$SRC_TASK_DIR" ]]; then
  echo "missing raw task dir: $SRC_TASK_DIR" >&2
  exit 1
fi

mapfile -t SEEDS < <(
  for seed_dir in "$SRC_TASK_DIR"/seed_*; do
    [[ -f "$seed_dir/steps.h5" && -f "$seed_dir/meta.json" ]] || continue
    basename "$seed_dir" | sed -E 's/^seed_//'
  done | sort -n | head -n "$ACT_TARGET_EPISODES"
)
if [[ "${#SEEDS[@]}" -lt "$ACT_TARGET_EPISODES" ]]; then
  echo "expected at least $ACT_TARGET_EPISODES episodes for $TASK, found ${#SEEDS[@]}" >&2
  exit 1
fi

if [[ "$ACT_FORCE_REBUILD_DATA" == "1" || ! -f "$RAW_ROOT/seeds.txt" ]]; then
  rm -rf "$RAW_ROOT"
  mkdir -p "$RAW_ROOT/$TASK/rasterizer"
  for seed in "${SEEDS[@]}"; do
    ln -s "$SRC_TASK_DIR/seed_${seed}" "$RAW_ROOT/$TASK/rasterizer/seed_${seed}"
  done
  printf "%s\n" "${SEEDS[@]}" > "$RAW_ROOT/seeds.txt"
fi

REBUILD_RLDS="$ACT_FORCE_REBUILD_DATA"
if [[ "$REBUILD_RLDS" != "1" ]]; then
  if [[ ! -f "$OPENVLA_DIR/genesis_hr_bench/1.0.0/dataset_statistics.json" ]]; then
    REBUILD_RLDS=1
  elif ! "$MAWM_PY" - "$OPENVLA_DIR/genesis_hr_bench/1.0.0/dataset_statistics.json" "$ACT_TARGET_EPISODES" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
raise SystemExit(0 if int(stats.get("num_trajectories", -1)) == expected else 1)
PY
  then
    REBUILD_RLDS=1
  fi
fi
if [[ "$REBUILD_RLDS" == "1" ]]; then
  rm -rf "$OPENVLA_DIR/genesis_hr_bench" "$OPENVLA_DIR/downloads"
  "$MAWM_PY" scripts/build_genesis_hr_bench_rlds.py \
    --raw-root "$RAW_ROOT" \
    --renderer rasterizer \
    --data-dir "$OPENVLA_DIR" \
    --image-size 256 \
    --control-mode "$ACT_CONTROL_MODE"
else
  echo "[act_train_one] reusing RLDS dataset at $OPENVLA_DIR"
fi

REBUILD_LEROBOT="$ACT_FORCE_REBUILD_DATA"
if [[ "$REBUILD_LEROBOT" != "1" ]]; then
  if [[ ! -f "$LEROBOT_HOME/$DATASET_REPO_ID/meta/info.json" ]]; then
    REBUILD_LEROBOT=1
  elif ! "$PI0_PY" - "$LEROBOT_HOME/$DATASET_REPO_ID/meta/info.json" "$ACT_TARGET_EPISODES" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
value = None
for key in ("total_episodes", "num_episodes", "episodes"):
    if key in info:
        value = info[key]
        break
raise SystemExit(0 if int(value) == expected else 1)
PY
  then
    REBUILD_LEROBOT=1
  fi
fi
if [[ "$REBUILD_LEROBOT" == "1" ]]; then
  rm -rf "$LEROBOT_HOME/$DATASET_REPO_ID"
  HF_LEROBOT_HOME="$LEROBOT_HOME" \
  "$PI0_PY" scripts/convert_genesis_hr_bench_to_lerobot.py \
    --tfds-data-dir "$OPENVLA_DIR" \
    --repo-id "$DATASET_REPO_ID" \
    --control-mode "$ACT_CONTROL_MODE" \
    --lerobot-naming \
    --overwrite
else
  echo "[act_train_one] reusing LeRobot dataset at $LEROBOT_HOME/$DATASET_REPO_ID"
fi

TRAIN_ARGS=(
  --policy.type=act
  --dataset.repo_id="$DATASET_REPO_ID"
  --output_dir="$TRAIN_OUTPUT_DIR"
  --job_name="$EXP_NAME"
  --steps="$ACT_STEPS"
  --batch_size="$ACT_BATCH_SIZE"
  --num_workers="$ACT_NUM_WORKERS"
  --policy.chunk_size="$ACT_CHUNK_SIZE"
  --policy.n_action_steps="$ACT_N_ACTION_STEPS"
  --policy.temporal_ensemble_coeff="$ACT_TEMPORAL_ENSEMBLE_COEFF"
  --policy.use_vae="${ACT_USE_VAE:-false}"
  --policy.dropout=0.0
  --save_freq="$ACT_SAVE_FREQ"
  --log_freq="$ACT_LOG_FREQ"
  --save_checkpoint=true
  --policy.device=cuda
  --wandb.enable=true
  --wandb.entity="$WANDB_ENTITY"
  --wandb.project="$WANDB_PROJECT"
  --wandb.notes="act qpos_target 20260524 ${TASK} ${RUN_SUFFIX}"
)
if [[ -d "$TRAIN_OUTPUT_DIR/checkpoints/last" ]]; then
  TRAIN_ARGS+=(--resume=true)
fi

HF_LEROBOT_HOME="$LEROBOT_HOME" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_PROJECT="$WANDB_PROJECT" \
"$PI0_PY" -m lerobot.scripts.train "${TRAIN_ARGS[@]}"

echo
echo "ready_to_eval_checkpoint=$TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"
