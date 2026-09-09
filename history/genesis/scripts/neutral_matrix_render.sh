#!/bin/bash
# Render one video per (neutral task x neutral motion) combo, throttled to a
# fixed number of concurrent SLURM jobs (this session's quota).  Each job runs
# one scripted-rollout episode and saves a video + collision summary.
#
# Usage: scripts/neutral_matrix_render.sh [max_concurrent]
set -u
REPO="/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench"
PY="/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python"
cd "$REPO"
MAXJ="${1:-10}"
PREFIX="nmx"

TASKS=(blocks_ranking_rgb blocks_ranking_size categorize dump_bin
       place_bread_in_basket place_burger_fries place_dual_shoes
       place_food_in_skillet put_object_cabinet stack_bowls_three)
MOTIONS=(wipe01 wipe05 cans mouse bottle laptop writing)

mine() { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^${PREFIX}_"; }

submit() {
  local task="$1" mot="$2"
  local name="${PREFIX}_${task}_${mot}"
  local out="data/neutral_matrix/${task}/${mot}"
  mkdir -p "$out"
  sbatch -J "$name" -p gpu-preempt --gres=gpu:1 --mem=48G -c 4 -t 1:30:00 \
    --exclude=gpu051 -o "job_logs/${name}_%j.log" \
    --wrap "cd $REPO && export PYOPENGL_PLATFORM=egl && $PY scripts/collect.py \
      --task ${task}_neutral --config config/neutral_motion_${mot}.yml \
      --episodes 1 --start-seed 0 --retry-reset-until-success \
      --max-reset-attempts 6 --save-dir $out --video-stride 25 \
      --track-avatar-collision" >/dev/null 2>&1
}

total=0
for t in "${TASKS[@]}"; do for m in "${MOTIONS[@]}"; do total=$((total+1)); done; done
echo "[matrix] submitting $total combos, max ${MAXJ} concurrent"

n=0
for t in "${TASKS[@]}"; do
  for m in "${MOTIONS[@]}"; do
    while [ "$(mine)" -ge "$MAXJ" ]; do sleep 20; done
    submit "$t" "$m"
    n=$((n+1))
    echo "[matrix] submitted ${n}/${total}: ${t} x ${m}  (in flight: $(mine))"
    sleep 3
  done
done

echo "[matrix] all submitted; waiting for completion..."
while [ "$(mine)" -gt 0 ]; do sleep 30; done

echo "[matrix] === RESULTS ==="
for t in "${TASKS[@]}"; do
  for m in "${MOTIONS[@]}"; do
    vid="data/neutral_matrix/${t}/${m}/video/seed_0.mp4"
    log=$(ls -t job_logs/${PREFIX}_${t}_${m}_*.log 2>/dev/null | head -1)
    res=$(grep -hoE "SUCCESS|FAIL" "$log" 2>/dev/null | tail -1)
    coll=$(grep -hE "avatar-collision:" "$log" 2>/dev/null | tail -1 | sed 's/^ *//')
    have=$([ -f "$vid" ] && echo "video" || echo "no-video")
    echo "${t} x ${m}: ${res:-?} | ${have} | ${coll:-}"
  done
done
echo "[matrix] done"
