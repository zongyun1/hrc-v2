# Neutral Avatar Parallel Handoff

This document splits the neutral-avatar work across four agents.  The goal is
to add a neutral variation to each task that already has interrupt/assist
coverage: the avatar performs unrelated work on the same table, never touches
the robot task objects, and the robot rollout must avoid avatar collision.

## Non-Negotiable Rules

- Use scripted robot rollout, not random policy.  Validation must call
  `task.play_once()` for the real task policy.
- The neutral avatar may use its own props only.  It must not attach, move, or
  release the robot task objects.
- Before placing neutral props/avatar in a task, use the neutral motion
  footprint to avoid the task's live objects, robot target regions, and robot
  workspace corridors.
- Validation video must include both the robot rollout and the neutral avatar
  motion.
- Passing compatibility means:
  - `rollout_ok == true`
  - `finished == true`
  - `avatar_collision.n_collisions == 0`
  - `metrics.success == true` unless the task owner documents why the parent
    metric is not appropriate for the neutral variant.
- Avoid post-robot neutral playback unless no concurrent region exists.  If
  used, document it in the task file and in the summary; the target behavior is
  concurrent same-table neutral work.

## Shared Infrastructure

Shared implementation lives in:

- `envs/tasks/neutral_avatar_table_work.py`
- `scripts/render_neutral_task_motion.py`
- `scripts/run_neutral_task_motion_matrix.sh`

The mixin already computes a swept neutral-object footprint by dry-running the
selected animation with its props.  It records the footprint in:

- `self._neutral_avatar_motion_footprint`
- `type(self).NEUTRAL_MOTION_FOOTPRINTS[motion_name]`

Each task should define candidate work positions instead of hard-coding one
position:

```python
NEUTRAL_WORK_XY = np.array([...], dtype=np.float64)
NEUTRAL_MOUSE_WORK_XY = np.array([...], dtype=np.float64)
NEUTRAL_WORK_CANDIDATES = [
    np.array([...], dtype=np.float64),
    ...
]
NEUTRAL_MOUSE_WORK_CANDIDATES = [
    np.array([...], dtype=np.float64),
    ...
]
```

The selector checks the neutral swept AABB against task objects/targets through
`_neutral_avatar_forbidden_regions()`.  If a task has important areas not
covered by that helper, extend the helper or override it in that task.

## Four Work Sets

Set 1 is owned by `neut1` / agent 3.

| Set | Owner | Tasks | Primary files |
|---|---|---|---|
| 1 | `neut1` | `blocks_ranking_rgb_neutral`, `blocks_ranking_size_neutral` | `envs/tasks/neutral_avatar_table_work.py`, `envs/tasks/blocks_ranking_rgb_neutral.py`, `envs/tasks/blocks_ranking_size_neutral.py`, render scripts |
| 2 | agent B | `place_bread_in_basket_neutral`, `place_dual_shoes_neutral`, `stack_bowls_three_neutral` | corresponding three `*_neutral.py` files |
| 3 | agent C | `categorize_neutral`, `dump_bin_neutral` | corresponding two `*_neutral.py` files |
| 4 | agent D | `place_burger_fries_neutral`, `place_food_in_skillet_neutral`, `put_object_cabinet_neutral` | corresponding three `*_neutral.py` files |

Agents B-D should not edit shared mixin/render scripts unless they coordinate
with set 1.  If a task needs custom geometry, prefer overriding task-local
candidate lists or `_neutral_avatar_forbidden_regions()` in that task file.

## Validation Commands

Print available motions:

```bash
/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
  scripts/render_neutral_task_motion.py --print-motions
```

Render one task/motion with scripted robot rollout:

```bash
sbatch -J neutral_one -p gpu --gres=gpu:1 --mem=32G -c 4 -t 1:00:00 \
  -o job_logs/neutral_one_%j.log -e job_logs/neutral_one_%j.log \
  --wrap='cd /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench && \
  /work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
  scripts/render_neutral_task_motion.py \
  --task blocks_ranking_rgb_neutral \
  --motion 01_KIT_wipe_table_wiping_the_table01_stageii \
  --seed 0 \
  --output-dir data/neutral_task_motion_agent_smoke \
  --video-stride 25 --collision-stride 30 --max-steps 22000 --tail-steps 120'
```

Summarize outputs:

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("data/neutral_task_motion_agent_smoke")
for p in sorted(root.glob("**/*_summary.json")):
    d = json.load(open(p))
    c = d.get("avatar_collision") or {}
    m = d.get("metrics") or {}
    print(
        p.relative_to(root),
        "rollout_ok=", d.get("rollout_ok"),
        "success=", m.get("success"),
        "coll=", c.get("n_collisions"), "/", c.get("n_checks"),
        "finished=", d.get("finished"),
        "target=", d.get("neutral_avatar_target"),
    )
PY
```

## Known Notes From Set 1 Exploration

- The two block-ranking neutral variants rendered cleanly for all five neutral
  motions in `data/neutral_task_motion_set1_neut1_20260509/` with SLURM array
  `57470742`; see `docs/neutral_avatar_set1_status.md`.
- `categorize_neutral` had zero avatar collisions in early tests, but inherited
  success metrics can fail if the parent checks avatar-side cooperative objects.
  Agent C should decide whether neutral needs a custom `check_success()`.
- `dump_bin_neutral` and `place_burger_fries_neutral` are hard layouts.  Earlier
  attempts showed zero avatar collision can still make scripted planning fail
  if the avatar obstacle cloud blocks a narrow robot path.  Agents C/D should
  tune candidate regions and only use post-robot neutral playback if explicitly
  accepted.

## Registration

The current scaffold already registers the neutral tasks in `envs/tasks/__init__.py`.
If an agent adds a new neutral class, update both the import section and
`TASK_MAP`.  Avoid unrelated edits in `__init__.py` because all agents may touch
that file.

## Agent Status

Each agent should update only its own row in `AGENT_STATUS.md` following
`docs/general/claude_agent_status_howto.md`.  Use ET timestamps.
