#!/usr/bin/env bash
set -u

ROOT=eval_sets/eval_v2_100_scene_init
PY=/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python
MAX_SCINIT_JOBS=${MAX_SCINIT_JOBS:-9}
STATE="$ROOT/logs/submitted_tasks.txt"
mkdir -p "$ROOT/per_task" "$ROOT/logs" job_logs
touch "$STATE"

mapfile -t TASKS < <("$PY" - <<'PY'
import json
seen = []
for line in open("eval_sets/eval_v2_100_scene_init/manifest.jsonl"):
    task = json.loads(line)["task"]
    if task not in seen:
        seen.append(task)
print("\n".join(seen))
PY
)

while :; do
  running=$(squeue -u "$USER" -h -o "%j" | awk '/^scinit_/ {n++} END {print n+0}')
  launched=0
  for task in "${TASKS[@]}"; do
    [[ "$running" -lt "$MAX_SCINIT_JOBS" ]] || break
    [[ -f "$ROOT/per_task/${task}.yaml" ]] && continue
    grep -qxF "$task" "$STATE" && continue

    sbatch -p gpu --gres=gpu:1 --mem=32G -c 4 -t 4:00:00 \
      -J "scinit_${task:0:20}" \
      -o "job_logs/scinit_${task}_%j.log" \
      --wrap "cd '$PWD' && '$PY' scripts/generate_eval_scene_init.py --manifest '$ROOT' --task '$task' --keep-failures --output '$ROOT/per_task/${task}.yaml'"
    echo "$task" >> "$STATE"
    running=$((running + 1))
    launched=$((launched + 1))
  done

  remaining=$("$PY" - <<'PY'
import json
from pathlib import Path
root = Path("eval_sets/eval_v2_100_scene_init")
seen = []
for line in open(root / "manifest.jsonl"):
    task = json.loads(line)["task"]
    if task not in seen:
        seen.append(task)
missing = [task for task in seen if not (root / "per_task" / f"{task}.yaml").exists()]
print(len(missing))
PY
)
  active=$(squeue -u "$USER" -h -o "%j" | awk '/^scinit_/ {n++} END {print n+0}')
  date -u +"%Y-%m-%dT%H:%M:%SZ remaining=$remaining active=$active launched=$launched" \
    | tee -a "$ROOT/logs/submitter.log"
  if [[ "$remaining" == "0" && "$active" == "0" ]]; then
    "$PY" "$ROOT/finalize_scene_init_set.py" 2>&1 | tee -a "$ROOT/logs/finalize.log"
    break
  fi
  if [[ "$remaining" != "0" && "$active" == "0" && "$launched" == "0" ]]; then
    "$PY" - <<'PY' > "$STATE"
import json
from pathlib import Path
root = Path("eval_sets/eval_v2_100_scene_init")
seen = []
for line in open(root / "manifest.jsonl"):
    task = json.loads(line)["task"]
    if task not in seen:
        seen.append(task)
for task in seen:
    if (root / "per_task" / f"{task}.yaml").exists():
        print(task)
PY
    echo "reset submitted state for missing tasks" | tee -a "$ROOT/logs/submitter.log"
  fi
  sleep 120
done
