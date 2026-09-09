#!/bin/bash
# One-command VLA data collection. Wraps scripts/collect.py with the env,
# config, and output layout used by the OpenVLA fine-tuning dataset
# (`openvla/data/genesis_hr_bench_raw/<task>/seed_*.pkl`).
#
# Usage:
#   scripts/collect_vla.sh --task pour_water --episodes 50 --start-seed 0
#   scripts/collect_vla.sh --task pour_water --episodes 50 --start-seed 100  # shard 2
#   scripts/collect_vla.sh --all --episodes 50 --start-seed 0                # default Franka task pool
#
# Flags:
#   --task NAME           Task to collect. One of the default Franka tasks (see TASKS).
#                         Mutually exclusive with --all.
#   --all                 Collect every Franka task in TASKS sequentially.
#   --episodes N          Episodes to attempt per task (default 50).
#                         Only successful episodes are saved.
#   --start-seed K        Starting seed (default 0). Use distinct, non-overlapping
#                         ranges across collaborators to avoid clobbering files.
#   --record-stride N     Record one obs frame every N sim steps (default 50, ≈ 10 Hz).
#   --config PATH         Override the collect config (default config/openvla_collect.yml).
#   --                    Pass remaining args through to scripts/collect.py.
#
# Output layout (per task):
#   openvla/data/genesis_hr_bench_raw/<task>/seed_<N>.pkl
#   openvla/data/genesis_hr_bench_raw/<task>/video/seed_<N>.mp4
#
# Once enough seeds are collected across all tasks, rebuild the TFDS dataset:
#   python scripts/build_genesis_hr_bench_rlds.py
# This reads every seed_*.pkl across every task subdir and writes
# openvla/data/genesis_hr_bench/1.0.0/ + dataset_statistics.json.

set -euo pipefail

# Source per-host config (REPO_ROOT, MAWM_PY, ...). To override, create
# scripts/env.local.sh — see scripts/env.local.sh.example.
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"

# Default Franka tasks pooled into the genesis_hr_bench TFDS dataset.
TASKS=(
  pour_water
  categorize_cooperative
  categorize_interrupt
  take_from_human_easy
)

PY="$MAWM_PY"

TASK=""
ALL=0
EPISODES=50
START_SEED=0
RECORD_STRIDE=50
CONFIG="config/openvla_collect.yml"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --all) ALL=1; shift ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --record-stride) RECORD_STRIDE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to forward to collect.py)" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PY" ]]; then
  echo "[collect_vla] cannot find MAWM python at $PY" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[collect_vla] config not found: $CONFIG" >&2
  exit 1
fi

run_one() {
  local t="$1"
  echo
  echo "==============================================================="
  echo "[collect_vla] task=$t  episodes=$EPISODES  start_seed=$START_SEED"
  echo "[collect_vla] config=$CONFIG  record_stride=$RECORD_STRIDE"
  echo "==============================================================="
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    SETUPTOOLS_USE_DISTUTILS=stdlib GENESIS_BACKEND=gpu "$PY" scripts/collect.py \
      --task "$t" --config "$CONFIG" --episodes "$EPISODES" \
      --start-seed "$START_SEED" --record-stride "$RECORD_STRIDE" \
      "${EXTRA[@]}"
  else
    SETUPTOOLS_USE_DISTUTILS=stdlib GENESIS_BACKEND=gpu "$PY" scripts/collect.py \
      --task "$t" --config "$CONFIG" --episodes "$EPISODES" \
      --start-seed "$START_SEED" --record-stride "$RECORD_STRIDE"
  fi
}

if [[ $ALL -eq 1 ]]; then
  if [[ -n "$TASK" ]]; then
    echo "--all and --task are mutually exclusive" >&2; exit 2
  fi
  for t in "${TASKS[@]}"; do
    run_one "$t"
  done
elif [[ -n "$TASK" ]]; then
  # Sanity-check that the task is one of the Franka tasks the dataset expects.
  ok=0
  for t in "${TASKS[@]}"; do
    if [[ "$t" == "$TASK" ]]; then ok=1; break; fi
  done
  if [[ $ok -eq 0 ]]; then
    echo "[collect_vla] WARNING: '$TASK' is not in the canonical Franka task list:" >&2
    printf '  %s\n' "${TASKS[@]}" >&2
    echo "[collect_vla] continuing anyway — the TFDS builder will pick up any task subdir under openvla/data/genesis_hr_bench_raw/." >&2
  fi
  run_one "$TASK"
else
  echo "must pass --task NAME or --all" >&2
  exit 2
fi

echo
echo "[collect_vla] done. Raw pickles under openvla/data/genesis_hr_bench_raw/."
echo "[collect_vla] Rebuild TFDS with:  $PY scripts/build_genesis_hr_bench_rlds.py"
