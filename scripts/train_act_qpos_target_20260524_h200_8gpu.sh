#!/usr/bin/env bash
# Launch separate per-task ACT training processes across an 8-GPU H200 node.
# Run inside an allocation, or directly on a machine with visible GPUs.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

RAW_DATA_ROOT="${ACT_RAW_DATA_ROOT:-$REPO_ROOT/vla_data_qpos_target_20260524}"
TASK_FILE="${ACT_TASK_FILE:-}"
TASKS_ENV="${ACT_TASKS:-}"
GPUS="${ACT_GPUS:-0,1,2,3,4,5,6,7}"
JOBS_PER_GPU="${ACT_JOBS_PER_GPU:-1}"
SKIP_EXISTING="${ACT_SKIP_EXISTING:-1}"
AUTO_BATCH="${ACT_AUTO_BATCH:-1}"
DEFAULT_BATCH="${ACT_BATCH_SIZE:-128}"
LOG_ROOT="${ACT_LOG_ROOT:-$REPO_ROOT/runs/act/qpos_target_20260524/h200_logs}"
RUN_ROOT="${ACT_RUN_ROOT:-$REPO_ROOT/runs/act/qpos_target_20260524}"
ACT_TARGET_EPISODES="${ACT_TARGET_EPISODES:-300}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-100}"
ACT_RUN_ID="${ACT_RUN_ID:-full300_c100_s25000_novae}"
ACT_RUN_VERSION="${ACT_RUN_VERSION:-v1}"
DRY_RUN="${ACT_DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Launch separate per-task ACT training processes across an 8-GPU H200 node.
Run inside an allocation, or directly on a machine with visible GPUs.

Environment knobs:
  ACT_TASKS="task_a task_b"          explicit task list
  ACT_TASK_FILE=path.txt             one task per line or comma-separated
  ACT_USE_TASK_POOL=0                set to 1 to default to raw_root/task_pool.txt
  ACT_GPUS=0,1,2,3,4,5,6,7           visible GPU ids to use
  ACT_JOBS_PER_GPU=1                 increase to 2 if one ACT run underutilizes H200
  ACT_BATCH_SIZE=128                 override batch size
  ACT_AUTO_BATCH=1                   auto-scale batch from GPU memory if ACT_BATCH_SIZE is unset
  ACT_SKIP_EXISTING=1                skip tasks with final checkpoint already present
  ACT_DRY_RUN=0                      set to 1 to print the planned tasks and exit
  ACT_STEPS=25000                    forwarded to one-task trainer
  ACT_RUN_ID=full300_c100_s25000_novae
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

split_tasks() {
  if [[ -n "$TASKS_ENV" ]]; then
    printf "%s\n" "$TASKS_ENV" | tr ',[:space:]' '\n' | sed '/^$/d'
  elif [[ -n "$TASK_FILE" ]]; then
    tr ',[:space:]' '\n' < "$TASK_FILE" | sed '/^$/d'
  elif [[ "${ACT_USE_TASK_POOL:-0}" == "1" && -f "$RAW_DATA_ROOT/task_pool.txt" ]]; then
    tr ',[:space:]' '\n' < "$RAW_DATA_ROOT/task_pool.txt" | sed '/^$/d'
  else
    find "$RAW_DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name rasterizer \
      | sed -E "s#^$RAW_DATA_ROOT/([^/]+)/rasterizer#\\1#" \
      | sort
  fi
}

checkpoint_path_for() {
  local task="$1"
  local suffix="qpos_target_abs_${ACT_TARGET_EPISODES}ep_${ACT_RUN_ID}_${ACT_RUN_VERSION}"
  printf "%s/finetune/act_%s_20260524_%s/checkpoints/last/pretrained_model/model.safetensors" \
    "$RUN_ROOT" "$task" "$suffix"
}

auto_batch_for_gpu() {
  local gpu="$1"
  local fallback="$DEFAULT_BATCH"
  if [[ "$AUTO_BATCH" != "1" || -n "${ACT_BATCH_SIZE:-}" ]]; then
    echo "$fallback"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$fallback"
    return
  fi
  local mem_mib
  mem_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "$mem_mib" ]]; then
    echo "$fallback"
    return
  fi
  python - "$mem_mib" "$JOBS_PER_GPU" <<'PY'
import math, sys
mem_mib = int(float(sys.argv[1]))
jobs = max(1, int(sys.argv[2]))
# Batch 16 was validated on an 11 GB 2080 Ti. Use about 70% of linear
# memory scaling and divide by colocated jobs; cap to keep optimizer updates
# frequent on 300-episode datasets.
raw = 16 * (mem_mib / 11264.0) * 0.70 / jobs
batch = max(16, min(256, int(raw // 16) * 16))
print(batch)
PY
}

mapfile -t ALL_TASKS < <(split_tasks)
TASKS=()
for task in "${ALL_TASKS[@]}"; do
  [[ -d "$RAW_DATA_ROOT/$task/rasterizer" ]] || continue
  ckpt="$(checkpoint_path_for "$task")"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
    echo "[h200_act] skip existing $task -> $ckpt"
    continue
  fi
  TASKS+=("$task")
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "[h200_act] no remaining tasks to train"
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "no GPUs selected via ACT_GPUS=$GPUS" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
echo "[h200_act] tasks=${#TASKS[@]} gpus=${GPUS} jobs_per_gpu=${JOBS_PER_GPU} log_root=$LOG_ROOT"
echo "[h200_act] first tasks: ${TASKS[*]:0:12}"
if [[ "$DRY_RUN" == "1" ]]; then
  printf "[h200_act] remaining task list:\n"
  printf "  %s\n" "${TASKS[@]}"
  exit 0
fi

declare -a SLOT_GPUS=()
declare -a SLOT_PIDS=()
declare -a SLOT_TASKS=()
for gpu in "${GPU_LIST[@]}"; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do
    SLOT_GPUS+=("$gpu")
    SLOT_PIDS+=(0)
    SLOT_TASKS+=("")
  done
done

next_task=0
running=0
failures=0

launch_slot() {
  local slot="$1"
  local task="$2"
  local gpu="$3"
  local batch
  batch="$(auto_batch_for_gpu "$gpu")"
  local stamp log
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  log="$LOG_ROOT/${task}_${stamp}_gpu${gpu}.log"
  echo "[h200_act] launch task=$task gpu=$gpu batch=$batch log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export ACT_RAW_DATA_ROOT="$RAW_DATA_ROOT"
    export ACT_RUN_ROOT="$RUN_ROOT"
    export ACT_TARGET_EPISODES="$ACT_TARGET_EPISODES"
    export ACT_CHUNK_SIZE="$ACT_CHUNK_SIZE"
    export ACT_RUN_ID="$ACT_RUN_ID"
    export ACT_RUN_VERSION="$ACT_RUN_VERSION"
    export ACT_BATCH_SIZE="$batch"
    exec scripts/train_act_qpos_target_20260524_one_task.sh "$task"
  ) >"$log" 2>&1 &
  SLOT_PIDS[$slot]="$!"
  SLOT_TASKS[$slot]="$task"
  running=$((running + 1))
}

for slot in "${!SLOT_GPUS[@]}"; do
  [[ "$next_task" -lt "${#TASKS[@]}" ]] || break
  launch_slot "$slot" "${TASKS[$next_task]}" "${SLOT_GPUS[$slot]}"
  next_task=$((next_task + 1))
done

while [[ "$running" -gt 0 ]]; do
  slept=0
  while [[ "$slept" -lt 30 ]]; do
    sleep 1
    slept=$((slept + 1))
    for i in "${!SLOT_PIDS[@]}"; do
      pid="${SLOT_PIDS[$i]}"
      [[ "$pid" == "0" ]] && continue
      if ! kill -0 "$pid" 2>/dev/null; then
        task="${SLOT_TASKS[$i]}"
        if wait "$pid"; then
          echo "[h200_act] done task=$task"
        else
          status=$?
          failures=$((failures + 1))
          echo "[h200_act] FAILED task=$task status=$status"
        fi
        SLOT_PIDS[$i]=0
        SLOT_TASKS[$i]=""
        running=$((running - 1))
        if [[ "$next_task" -lt "${#TASKS[@]}" ]]; then
          launch_slot "$i" "${TASKS[$next_task]}" "${SLOT_GPUS[$i]}"
          next_task=$((next_task + 1))
        fi
      fi
    done
  done
  echo "[h200_act] progress launched=$next_task/${#TASKS[@]} running=$running failures=$failures"
done

if [[ "$failures" -gt 0 ]]; then
  echo "[h200_act] completed with failures=$failures" >&2
  exit 1
fi
echo "[h200_act] all tasks complete"
