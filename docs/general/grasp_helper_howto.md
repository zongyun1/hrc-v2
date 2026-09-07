# Using the grasp helper in tasks

`BaseTask.select_and_execute_grasp` is the **single grasp entry point** for
new task implementations. Use it. Do not roll your own pre-grasp + RRT +
screw plumbing — that path was unified in 2026-05 and other variants
(`select_grasp` + manual `move_and_execute`, ad-hoc `arm.planner.plan_*`
calls, etc.) produce silent failures we've already paid for.

## Why one helper

Earlier tasks each rebuilt the grasp pipeline inline. The pattern that
actually works is:

1. **Rank** annotated grasps by approach-axis alignment + manual-first
   preference, after filtering grasps that would put the gripper below
   the table or whose pre-grasp is closer to the object than the grasp.
2. **Plan curr→pre via RRT** (`mplib.plan_pose`). Curved transit is fine
   here — we're nowhere near the object yet.
3. **Plan pre→grasp via `plan_screw_path`** (constant TCP-twist linear
   motion). NO RRT fallback — the whole point of this leg is the
   gripper approaches *exactly* along the grasp z-axis. If screw fails,
   the candidate is dropped. RRT-fallback hides bad grasps as
   "successful" plans through the object.
4. **FK-quality check** the planned grasp endpoint (>3 cm error → drop).
   This catches mplib's silent IK relaxation (memory:
   `project_mplib_silent_ik_relax`).
5. Execute both legs with `execute_plan`.

`select_and_execute_grasp` does all of this for you. If a candidate
fails any step, the loop advances to the next ranked candidate
automatically.

## Calling it

```python
result = self.select_and_execute_grasp(
    object_name="048_stapler",
    object_pose=obj_pose,         # current world pose of the object
    arm_tag="right",              # "left" / "right" — single-arm tasks always pass "right"
    model_id=0,                   # match the model_id you loaded
    robot_type=None,              # auto-pulled from config["robot_type"]
    max_candidates=10,            # bound on # candidates to attempt
    object_scale=1.0,             # see "Object scale" below — REQUIRED for rescaled objects
    categories=None,              # filter to e.g. ["handle"] for knife/screwdriver
    pre_dist=None,                # override the YAML's per-grasp pre_distance (m)
    table_z=None,                 # default = self.TABLE_TOP_Z
)

if result is None:
    # Every candidate failed — give up or fail the trial.
    return False

grasp_link, pre_link, grasp = result
# grasp_link / pre_link: link-frame Poses you can re-use for lift planning.
# grasp: the GraspPose object that was used; .name is the YAML entry name.

# Continue with: close gripper, lift along TCP +z, etc.
```

## Object scale (DO NOT skip)

If the asset's `model_data{N}.json` specifies a `scale` field (most do),
`load_object` applies it at load time. **The grasp helper transforms YAML
grasps using `object_scale` you pass in.** Loader-vs-helper mismatch
silently produces wrong-size grasps that miss by `mesh_extent × (1 -
scale)`.

The standard pattern:

```python
import json
from envs.utils import ASSETS_PATH

def _read_object_scale(name, mid=0):
    p = ASSETS_PATH / "objects" / name / f"model_data{mid}.json"
    if not p.exists():
        return 1.0
    raw = json.load(open(p)).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)

obj_scale = _read_object_scale("029_olive-oil", 0)
self.select_and_execute_grasp(..., object_scale=obj_scale)
```

If your task computes its own scale (e.g. `_mesh_fit_scale` to a target
dimension), pass that same number — never `1.0` by default.

## Forcing one specific grasp (testing / debug only)

```python
result = self.try_grasp_by_name(
    object_name="048_stapler",
    object_pose=obj_pose,
    arm_tag="right",
    grasp_name="manual_000",      # YAML entry name
    model_id=0, object_scale=1.0,
    execute=True,                 # default; runs the two legs in sequence
)
```

Use this only for debug / annotation testing
(`scripts/debug_grasp_helper_grid.py`). **Production tasks should call
`select_and_execute_grasp`** so the ranker can fall back to a different
grasp when the forced one is unreachable from the current pose.

When `execute=False`, you get the planned trajectories back without
execution: `(result_pre, result_grasp, grasp_link, pre_link, grasp)`.
This is the hook the grid sweep uses to interleave snapshots between
the pre and grasp legs — task code shouldn't need it.

## What NOT to call

These exist but are either obsolete or internal:

- `select_grasp` — the old "rank only, no FK / no execute" helper. Still
  used by some pre-2026-05 tasks. Don't write new callers; migrating
  inherited callers to `select_and_execute_grasp` is welcome.
- `arm.planner.plan_pose` / `plan_screw_path` — call these directly only
  if you're adding a *non-grasp* motion (e.g. a place / pour
  trajectory). For grasping, go through the helper.
- `_plan_one_grasp` — private; the inner per-candidate body of
  `select_and_execute_grasp` and `try_grasp_by_name`. Don't call it.
- The old `move_and_execute(grasp_link.to_pose7())` pattern after a
  manual `select_grasp` — drops the screw straight-line guarantee and
  RRTs around obstacles, which is precisely what the helper was
  written to prevent.

## Lift / transport after the grasp

The helper leaves the robot at the grasp pose with the gripper still
open. Standard follow-up (copied from `pick_and_place` / `pour_water`):

```python
grasp_link, pre_link, grasp = self.select_and_execute_grasp(...)
self.set_gripper(0.0, arm_tag)               # close
for _ in range(200): self.step_sim()         # let contact build

# Lift along world +z — recompute target in TCP frame from the grasp pose.
grasp_tcp = grasp.to_world(obj_pose, object_scale)
lift_tcp  = Pose(grasp_tcp.p + np.array([0, 0, lift_height_m]), grasp_tcp.q)
lift_link = tcp_to_link_pose(lift_tcp, arm.tcp_offset)
self.move_and_execute(lift_link.to_pose7(), arm_tag)
```

Don't full-close on round / smooth meshes — see memory
`project_gripper_close_target`. Use `set_gripper(~0.55)` for those.

## When the helper returns None

Every candidate failed. Most common reasons (with the log line that
flags them):

- `pre RRT failed` — pre-grasp pose is unreachable / past joint limits.
  Object is too far from base, or the candidate's approach axis points
  at a joint singularity.
- `screw failed — skipping` — the linear approach from pre to grasp
  passes through the object or an obstacle. The grasp annotation may be
  wrong-side / wrong-axis, OR the object's pose puts the approach axis
  through the table.
- `FK err {x}m > 3cm` — mplib silently relaxed IK and returned a "valid"
  plan whose endpoint isn't actually at the goal. Almost always means
  the grasp is unreachable in this orientation.

If you see persistent `None` returns on an object you expect to work,
run `scripts/launch/debug_grasp_helper_one.sh <object>` against it —
that grid sweep enumerates every annotated grasp × N seeds and writes
6 PNGs (pre / grasp / lifted, top-down + side view) per trial so you
can see which annotation is bad.  The side view is critical for small
or dark objects (e.g. `047_mouse`) that disappear under the gripper in
the top-down view.  Outputs: `data/grasp_helper_grid/<object>.json`
(per-grasp success rates + a `blacklist` field for grasps with rate ≤
20%) and
`data/grasp_helper_grid/snap/<object>/<grasp>/seed*__{pre_grasp,grasp,lifted}{,_side}.png`.

## Reference

- Helper source: `envs/base_task.py:select_and_execute_grasp` (and
  the inner `_plan_one_grasp`).
- Public force-by-name: `envs/base_task.py:try_grasp_by_name`.
- Grasp YAML format: `envs/grasp.py:GraspPose` (note `scale_frame:
  mesh_unit` vs `runtime` — the helper handles both, but if you author
  a new manual grasp, prefer `mesh_unit` so it auto-rescales when the
  asset's `scale` is changed).
- Example callers (good): `envs/tasks/cutting_board_clutter_clearing.py`,
  `envs/tasks/oil_bottle_recovery.py`, `envs/tasks/pour_water.py`.
- Debug driver: `scripts/debug_grasp_helper_grid.py` and the launcher
  `scripts/launch/debug_grasp_helper_one.sh`.
- Per-object validation status files were removed during task documentation
  cleanup; record reusable validation notes in the relevant task or grasp doc.
