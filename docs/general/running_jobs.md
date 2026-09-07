# Running jobs on the cluster

Operational notes for any agent running code in this repo. Applies to all
tasks and scripts unless a doc explicitly overrides.

## The login node is SHARED — keep it light

You run on a **login node shared with every other cluster user**. Do **not**
run risky or heavy commands directly here. Anything that burns CPU, memory, or
I/O belongs in a SLURM allocation (`srun`/`sbatch`), never on the login node.

Forbidden on the login node:

- Filesystem-wide scans: `find /`, `find /scratch ...`, `du -sh /<big>`,
  `grep -r` over huge trees, `locate`-style sweeps. Scope every search to a
  specific small directory, or run it inside an `srun` step.
- Heavy compute / builds: training, data collection, Genesis scene builds,
  CUDA/NVRTC compiles, `pip`/`uv` installs of large stacks, mesh decomposition.
- Anything long-running, many-process, or high-memory (no `xargs -P`, no big
  parallel loops, no multi-GB reads).

Safe on the login node: light orchestration only — `git`, `squeue`/`sbatch`/
`scancel`/`sinfo`, editing files, tailing logs, small scoped `grep`/`ls`, and
`sbatch --wrap "..."` to push the real work onto a compute node.

When in doubt, wrap it in `srun -p gpu --gres=gpu:1 ...` (see below) rather than
running it here. A slow login node degrades the cluster for everyone.

## SLURM: use `srun`, always request a GPU

All Python runs go through `srun`. Typical allocation for **non-LuisaRender**
jobs (rasterizer, unit tests, debug scripts, data collection):

```
srun -p gpu --gres=gpu:1 --mem=32G -c 4 -t <time>
```

- `-p gpu` + `--gres=gpu:1` — even when using CPU Genesis, the GPU partition
  is the default compute pool here.
- `--mem=32G` / `-c 4` — enough for a single-episode rollout. Bump only if a
  specific task OOMs.
- `-t <time>` — set explicitly (e.g. `-t 1:00:00`). Default limits may be
  shorter than the run needs.

For **LuisaRender** jobs use the template in
[`docs/luisa_render.md`](../luisa_render.md) (more memory, `-t 4:00:00`,
`sm_70` constraint, MAWM Python).

## Genesis backend: CPU by default

Non-LuisaRender runs should use `gs.cpu` — `BaseTask._setup_scene` already
does this unless `GENESIS_BACKEND=gpu` is set in the environment. Do **not**
export `GENESIS_BACKEND=gpu` for rasterizer/debug runs; only set it for
raytracer jobs (the `--renderer raytracer` flag handles this automatically).

## Python environment

> **CRITICAL — use the Genesis 1.0.0 env, not the old fork.** The benchmark now
> runs on **Genesis 1.0.0**. Use:
>
> ```
> /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python
> ```
>
> The old `conda/robotwin/bin/python` is the editable **Genesis 0.3.3** fork and
> behaves DIFFERENTLY — in particular its gripper/contact physics is more
> forgiving, so grasp fixes that "work" on 0.3.3 are **false positives** that slip
> on 1.0.0. **Every grasp / contact / physics fix MUST be verified on
> `MAWM_latest_genesis` (1.0.0).** Sanity check: `is_latest_genesis()` must return
> `True` (genesis-world >= 1.0.0). See memory `project_genesis_1_0_migration`.

Example:

```bash
srun -p gpu --gres=gpu:1 --mem=32G -c 4 -t 1:00:00 \
    /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_latest_genesis/bin/python \
    scripts/collect.py --task pour_water --episodes 1
```

### Scoring vs. debugging harness

`scripts/collect.py` (random seeds, default config) is fine for **debugging**
crashes / grasps, but it is **not** the benchmark score. The actual benchmark is
`scripts/eval_expert_set.py` over a fixed eval set
(`eval_sets/eval_v1_100/`, `eval_sets/eval_v2_100_scene_init/`) with the eval
set's `config_overrides` — notably `eval_mode: true`,
`track_avatar_collision: true`, and `neutral_avatar_strict_layout: true`, which
change success/failure vs. a bare `collect.py` run. Confirm fixes against
`eval_expert_set.py` (launched via `scripts/eval_expert_set.sbatch`, which uses
`${MAWM_PY}` = the 1.0.0 env) before claiming a task is fixed.

The legacy `conda/robotwin` (0.3.3) env is retained only for backward-compat
regression checks. For LuisaRender use the MAWM env
(`/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM/bin/python`);
see [`docs/luisa_render.md`](../luisa_render.md).

## Where to write logs

**All `*.log` files go under `job_logs/`** — never at the repo root, never
inside `data/`, never in CWD-relative locations that depend on where a script
is invoked from.

- sbatch directives: `#SBATCH -o job_logs/<job_name>_%j.log` and `-e` matching.
- Python scripts that write logs: open
  `job_logs/<scriptname>_<obj>.log` (anchored to the repo root), not
  `<save_dir>/iter.log` or `<save_dir>/run.log`.
- Default `slurm-<id>.out` at the repo root is wrong — always set the
  `-o` / `-e` directives explicitly so logs land under `job_logs/`.
- The `job_logs/` directory is gitignored apart from its `.gitkeep`. Clean
  it out periodically; nothing there is load-bearing.

## Where to write review outputs

Coding agents should keep generated outputs easy to inspect and easy to
delete.  Do not scatter videos, images, traces, or scratch files across the
repo root or many one-off subdirectories.

- Videos/images that the user should review go under `data/<agent-or-task>/`.
  Use one stable folder per agent/task, for example
  `data/git_place_bread_review/`, and reuse it across iterations.
- When producing a new review set, clean up the previous obsolete outputs in
  that same folder first.  Keep only the latest useful videos/images and any
  small result files needed to interpret them.
- Avoid deep folder trees.  Prefer
  `data/<agent-or-task>/<short_run_name>/` only when there is a real need to
  compare multiple runs side by side.
- Temporary outputs that are not meant for review can go under `data/tmp/`.
  Treat `data/tmp/` as disposable scratch and clean it frequently.
- Do not create new top-level output folders such as `output/`, `videos/`,
  `tmp_*`, or ad-hoc script-local frame directories.  Use `data/` for media
  and `job_logs/` for logs.

## Task implementation rules

These rules apply to every task class added under `envs/tasks/`. They exist
because past attempts to bypass them silently corrupted other tasks' eval.

### Robot arm: PD control only — no kinematic shortcuts

The robot arm must reach poses through **real physics** — the planned
trajectory plays through `set_arm_joints` (PD position targets) and the
simulator's contact dynamics. Do **not**:

- Call `entity.set_qpos(...)` or `teleport_arm_joints(...)` to jump the
  arm to a target during normal task execution. (One exception: setting
  the arm's *initial* qpos at scene reset, before `play_once` begins, is
  fine — that's initialization, not execution.)
- Set `USE_SETQPOS=1` / `DEBUG_TELEPORT=1` in production launchers — those
  envs are for debugging only.
- Use `attach_to_gripper` / `gripper_tether` / any kinematic constraint
  that pins a held object to the gripper without contact dynamics.

If your task only works because the arm teleports through obstacles or
through a held object, the task is broken. Find a feasible plan / grasp
geometry / orientation instead. (The avatar's hand-attached items, e.g. a
pan being held by a moving avatar, are a separate system and not subject
to this rule.)

### Path planning + IK: mplib only — never Genesis IK

Use **mplib** for both motion planning and inverse kinematics:

- `arm.planner.plan_path(current_qpos, target_pose7)` — RRT-Connect
  through `mplib.plan_pose`.
- `arm.planner.solve_ik(...)` — mplib IK (Pinocchio under the hood).

Do **not** use `arm.planner.solve_ik_cartesian(...)` or any wrapper that
calls `entity.inverse_kinematics(...)`. Genesis's damped-least-squares IK
silently returns the closest reachable qpos with **no convergence gate**,
so a "Success" result can put the EE 25–60 cm off the requested target
(see `memory/project_mplib_silent_ik_relax.md`). It also has no notion of
the planner obstacle pcd, so its solutions can drive the arm through the
avatar / counter / cabinet at execution time.

mplib's `plan_pose` itself can sometimes silently relax the IK (same memory
file). The mitigation is FK-verifying the trajectory's end qpos and
retrying the plan with a different RRT seed — not falling back to
Genesis IK.

### Avatar must be in the planner's obstacle pcd

The robot must avoid colliding with the avatar. Pass the avatar's current
capsule pcd into `arm.planner.update_obstacles(...)` before each plan via
`_refresh_planner_obstacles`. If a particular snapshot of the frying /
chopping motion blocks every IK candidate, time-snapshot search (step the
sim forward, refresh obstacles, retry) — don't drop the avatar from the
pcd to make the planner happy.

## Reminders

- Save videos/images/data under `data/<agent-or-task>/...`, never under
  `output/<task>/...` or repo-root scratch folders.
- Never `scancel` all jobs — Claude Code itself runs as a SLURM job.
- `AGENT_STATUS.md` "Task" column is **1–2 sentences max** — current
  state, not a per-version chronicle. Detail belongs in commits / memory /
  per-task docs.
