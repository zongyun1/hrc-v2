#!/bin/bash
# Evaluate HF/local checkpoints for pi0, pi05, openvla_oft, and rdt.
# Starts each model server through scripts/run_vla_eval.sh, writes videos and
# JSON results into one timestamped run directory, and keeps server logs there.
#
# Example:
#   scripts/eval_hf_vla_suite.sh --task pour_water --episodes 3 --max-steps 300 \
#     --checkpoint pi0=zhouqh/pi0-pour-water \
#     --checkpoint pi05=zhouqh/pi05-pour-water \
#     --checkpoint openvla_oft=zhouqh/openvla-oft-pour-water \
#     --checkpoint rdt=zhouqh/rdt-pour-water
#
# If your pi checkpoints were trained on qpos_target_abs data, keep the default
# pi0/pi05 action type qpos_abs. Override with --action-type pi0=ee if needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cd "$REPO_ROOT"

TASK="pour_water"
EPISODES=1
MAX_STEPS=300
START_SEED=0
MODELS="pi0,pi05,openvla_oft,rdt"
OUT_ROOT="runs/vla_eval_hf"
VIDEO_STRIDE=""
PULL=0
MIN_FREE_GB=80
HF_CACHE_DIR="${HF_HOME:-$REPO_ROOT/checkpoints/hf_cache}"

declare -A CKPT
declare -A ACTION_TYPE=(
  [pi0]="qpos_abs"
  [pi05]="qpos_abs"
  [openvla_oft]="ee"
  [rdt]="qpos"
)
declare -A PORT=(
  [pi0]="8767"
  [pi05]="8768"
  [openvla_oft]="8769"
  [rdt]="8772"
)

usage() { sed -n '2,28p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --video-stride) VIDEO_STRIDE="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --pull) PULL=1; shift ;;
    --checkpoint)
      kv="$2"; shift 2
      model="${kv%%=*}"; val="${kv#*=}"
      [[ "$model" != "$val" ]] || { echo "--checkpoint must be model=repo_or_path" >&2; exit 2; }
      CKPT[$model]="$val"
      ;;
    --action-type)
      kv="$2"; shift 2
      model="${kv%%=*}"; val="${kv#*=}"
      [[ "$model" != "$val" ]] || { echo "--action-type must be model=ee|qpos|qpos_abs" >&2; exit 2; }
      ACTION_TYPE[$model]="$val"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_ROOT/${TASK}_${TS}"
mkdir -p "$RUN_DIR"
SERVER_LOG_DIR="$RUN_DIR/server_logs"
mkdir -p "$SERVER_LOG_DIR"

free_gb=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G", "", $4); print $4}')
{
  echo "task=$TASK"
  echo "episodes=$EPISODES"
  echo "max_steps=$MAX_STEPS"
  echo "start_seed=$START_SEED"
  echo "models=$MODELS"
  echo "hf_cache_dir=$HF_CACHE_DIR"
  echo "free_gb=$free_gb"
  echo "run_dir=$RUN_DIR"
} | tee "$RUN_DIR/manifest.txt"

if (( free_gb < MIN_FREE_GB )); then
  cat >&2 <<MSG
[eval_hf_vla_suite] Not enough free disk for checkpoint pulls/eval.
  free: ${free_gb}G
  required: ${MIN_FREE_GB}G
Set --hf-cache-dir to a larger mounted volume, lower --min-free-gb if you know
checkpoints are already cached, or free space under /root/.cache/huggingface.
MSG
  exit 1
fi

check_python_module() {
  local py="$1" mod="$2"
  "$py" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$mod")
PY
}

preflight_model() {
  local model="$1"
  case "$model" in
    pi0|pi05)
      [[ -x "$LEROBOT_PY" ]] || { echo "missing LEROBOT_PY=$LEROBOT_PY" >&2; return 1; }
      check_python_module "$LEROBOT_PY" flask || { echo "$model: lerobot env missing flask" >&2; return 1; }
      ;;
    openvla_oft)
      [[ -x "$OFT_PY" ]] || { echo "missing OFT_PY=$OFT_PY" >&2; return 1; }
      check_python_module "$OFT_PY" flask || { echo "openvla_oft env missing flask" >&2; return 1; }
      ;;
    rdt)
      [[ -x "$RDT_PY" ]] || { echo "missing RDT_PY=$RDT_PY" >&2; return 1; }
      check_python_module "$RDT_PY" flask || { echo "rdt env missing flask; install flask in .venvs/rdt" >&2; return 1; }
      ;;
    *) echo "unknown model $model" >&2; return 1 ;;
  esac
}

pull_if_requested() {
  local model="$1" ckpt="$2"
  [[ "$PULL" == 1 ]] || return 0
  [[ -n "$ckpt" ]] || return 0
  [[ -e "$ckpt" ]] && return 0
  mkdir -p "$HF_CACHE_DIR"
  local py="$LEROBOT_PY"
  [[ "$model" == "openvla_oft" ]] && py="$OFT_PY"
  [[ "$model" == "rdt" ]] && py="$RDT_PY"
  echo "[eval_hf_vla_suite] pulling $model checkpoint: $ckpt"
  HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub" "$py" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="$ckpt", repo_type="model"))
PY
}

for model in "${MODEL_LIST[@]}"; do
  model="${model//[[:space:]]/}"
  [[ -n "$model" ]] || continue
  echo "===============================================================" | tee -a "$RUN_DIR/manifest.txt"
  echo "[eval_hf_vla_suite] model=$model" | tee -a "$RUN_DIR/manifest.txt"

  preflight_model "$model" || { echo "[eval_hf_vla_suite] preflight failed for $model" >&2; exit 1; }

  ckpt="${CKPT[$model]:-}"
  if [[ -z "$ckpt" && ( "$model" == pi0 || "$model" == pi05 ) ]]; then
    echo "[eval_hf_vla_suite] --checkpoint $model=... is required" >&2
    exit 2
  fi
  pull_if_requested "$model" "$ckpt"

  model_dir="$RUN_DIR/$model"
  video_dir="$model_dir/videos"
  mkdir -p "$video_dir"
  result_json="$model_dir/results.json"
  eval_log="$model_dir/eval.log"

  args=(
    --model "$model"
    --task "$TASK"
    --episodes "$EPISODES"
    --max-steps "$MAX_STEPS"
    --port "${PORT[$model]}"
    --action-type "${ACTION_TYPE[$model]}"
    --video-dir "$video_dir"
  )
  [[ -n "$ckpt" ]] && args+=(--checkpoint "$ckpt")
  args+=(-- --start-seed "$START_SEED" --output-json "$result_json")
  [[ -n "$VIDEO_STRIDE" ]] && args+=(--video-stride "$VIDEO_STRIDE")

  echo "[eval_hf_vla_suite] command: SERVER_LOG_DIR=$SERVER_LOG_DIR scripts/run_vla_eval.sh ${args[*]}" | tee "$model_dir/command.txt"
  HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub" SERVER_LOG_DIR="$SERVER_LOG_DIR" \
    scripts/run_vla_eval.sh "${args[@]}" 2>&1 | tee "$eval_log"
done

echo "[eval_hf_vla_suite] done: $RUN_DIR"
