# Factory human-robot collaboration tasks

Five new HRC tasks set in a shared **factory work cell** (an
Amazon-style packing station). A human worker and a robot arm work at
the same industrial bench. Each task is meant to be implemented by one
agent, independently; all five reuse the same scene.

Status legend: each task section has an **Owner** line — claim a task by
writing your agent name there (and your `AGENT_STATUS.md` row) before
starting.

## Getting started (per agent)

1. Read `docs/general/README.md` and everything it links (cluster rules,
   `AGENT_STATUS.md` upkeep, grasp helper conventions).
2. Claim your task: write your agent name on its **Owner** line below and
   add your row to `AGENT_STATUS.md`.
3. Implement the task class under `envs/tasks/factory_<name>.py` as
   `class Factory<Name>(FactorySceneMixin, BaseTask)` (MRO order
   matters), register it in `envs/tasks/__init__.py` `TASK_MAP`.
4. Do NOT modify shared infra (`BaseTask`, `AvatarController`,
   `FactorySceneMixin`, motion modules…) — subclass/compose instead. If
   the scene needs a knob the mixin lacks, override the class attribute
   in your task; only extend `envs/scenes/factory.py` if it's generic and
   additive.
5. Verify with
   `srun -p gpu --gres=gpu:1 --mem=32G -c 4 -t 1:00:00 <robotwin python> scripts/collect.py --task factory_<name> --episodes 1`
   until you have a successful episode with a watchable video under
   `data/factory_<name>/video/`.
6. Rendering stills of the scene (not videos): the avatar skin shows a
   T-pose on a script's first render — see `scripts/preview_factory_scene.py`
   for the play-idle + render/step/render workaround.

## Shared scene: `FactorySceneMixin`

`envs/scenes/factory.py` — mirror of `KitchenSceneMixin`
(`envs/scenes/kitchen.py`). Provides:

- **Packing bench** 1.70 × 0.95 m, steel-gray top, bench top z = 0.765
  (`TABLE_TOP_Z`). Replaces the default BaseTask table via
  `_create_table` override.
- **Robot**: Franka mounted ON the bench at `(+0.60, -0.30, 0.765)`
  (kitchen-proven pose, injected via `FACTORY_ROBOT_KWARGS`).
- **Worker (avatar)**: stands at the left half of the bench at
  `(-0.75, -0.90)` facing +y (`avatar_init_pos` / `avatar_init_rot`
  class attrs; tasks may move them).
- **Backdrop**: warehouse storage rack (orange uprights, 3 gray
  shelves, static cardboard stock boxes) behind the bench at y ≥ 0.85 —
  outside the robot's workspace.
- **Floor**: yellow safety-tape border around the cell (render-only),
  wooden pallet with sealed boxes parked at `(+1.55, +0.45)`.
- All furniture is **static**; the scene spawns **no graspable
  objects**. Tasks add their own objects in `load_actors()` after
  calling `self.build_factory()`.

Usage:

```python
from envs.base_task import BaseTask
from envs.scenes.factory import FactorySceneMixin

class FactoryKitPacking(FactorySceneMixin, BaseTask):   # MRO order matters
    use_avatar = True
    def load_actors(self):
        self.build_factory()
        # spawn task objects here
```

Preview renders: `scripts/preview_factory_scene.py` →
`data/factory_scene_preview/*.png`.

Reachability notes (from existing tasks):
- Avatar FABRIK reach ≈ 0.75 m from shoulder — keep avatar pick/place
  targets within that, or `avatar.reset(...)` closer per episode
  (see `memory/project_avatar_fabrik_reach.md`).
- Robot reachable bench area ≈ x ∈ [-0.2, 0.8] around its base at
  (+0.60, -0.30); keep robot targets on the right/center of the bench.
- Follow `docs/general/running_jobs.md` task implementation rules:
  PD-only arm motion, mplib planning/IK, avatar in the planner obstacle
  pcd. Avatar may use `attach_obj` kinematic holds; the robot may not.

Suggested shared conventions for all five tasks:
- Task names: `factory_<name>` registered in `envs/tasks/__init__.py`
  TASK_MAP.
- Task objects come from `assets/objects/` with verified
  `grasp_poses_franka.yml` (e.g. `001_bottle`, `071_can`, `031_jam-jar`,
  `038_milk-box`, `113_coffee-box`, `024_scanner`, `032_screwdriver`,
  `020_hammer`, `008_tray`, `110_basket`).
- "Shipping box" containers: either an open-top box built from 5 static
  primitives (cheap, robust — see the kitchen cabinet pattern in
  `build_kitchen_backdrop`) or a basket/tray asset.

---

## Task 1 — `factory_kit_packing` (assist)

**Owner:** kit (tmux 5)

Human and robot pack one shared shipping kit in parallel.

- **Scene**: open shipping box (center bench, ~(0.0, 0.15)); 4 kit items
  on the bench — 2 on the robot's side (e.g. `001_bottle`, `071_can`),
  2 on the worker's side (e.g. `031_jam-jar`, `113_coffee-box`).
- **Robot**: picks its 2 items and places them into the box
  (top-down pick-place; copy the flow of
  `envs/task_bases/place_burger_fries.py`).
- **Human**: concurrently `pick_and_place()`s (FABRIK) its 2 items into
  the same box. Pacing per `stack_bowls_three_assist`
  (80/120/80 frames).
- **Success**: all 4 items inside the box AABB; no avatar collision.
- **Precedent**: `envs/tasks/place_burger_fries_assist.py`,
  `envs/tasks/stack_bowls_three_assist.py`.
- **Risk**: low. Watch concurrent access to the box — stagger robot vs
  avatar placement XY so they never reach the same spot at once.

## Task 2 — `factory_line_feeding` (handover)

**Owner:** feed (tmux 6)

Robot feeds parts to a worker busy at their station.

- **Scene**: worker at the left bench doing a looping work animation
  (e.g. `Use_mouse` or a wipe-style motion as "assembly"); parts tray
  zone on the robot's side with 2–3 parts (`032_screwdriver`,
  `020_hammer`, or small boxes); a marked handover zone within the
  worker's reach (~0.3 m right of the avatar).
- **Robot**: picks the next part and places it in the handover zone;
  waits; repeats for each part.
- **Human**: on each delivery, interrupts the work loop, FABRIK-reaches
  to the handover zone, `attach_obj`s the part, places it at their
  station, resumes the loop.
- **Success**: all parts delivered AND taken (each part ends at the
  worker's station side); no collision.
- **Precedent**: `envs/tasks/deliver_to_human_easy.py` (robot→human
  delivery + take motion), `envs/avatar/inspect_motion_mixin.py`
  (interleaving avatar motions).
- **Risk**: medium-low. Timing handshake robot-place → avatar-take is
  scripted sequencing, no new physics.

## Task 3 — `factory_inspect_pack` (sequential cooperative)

**Owner:** qc (tmux 7) · **Eval category:** `intent` (cueing) — registered
in the `intent` group of `scripts/generate_eval_set.py`.

Quality-control pipeline: human inspects upstream, robot packs
downstream — the robot may only pack inspected items.  The worker's
inspect gesture *cues* which product is ready; the robot acts on that
intent.

- **Scene**: input zone on the worker's side with 3 products
  (e.g. `001_bottle`, `071_can`, `038_milk-box`); a marked "inspected"
  staging zone mid-bench; shipping box on the robot's side.
- **Human**: per item — pick (FABRIK), play an `Inspect*` animation
  (pool exists, see `inspect_motion_mixin.py`), place into staging.
- **Robot**: watches staging; once an item lands there, picks it and
  packs it into the box, while the human inspects the next item.
- **Success**: all items in the box AND every item passed through
  staging (track per-item state machine in the task).
- **Precedent**: `envs/tasks/categorize_cooperative.py` (avatar
  pick-place + robot sort), inspect pool from interrupt tasks.
- **Risk**: medium. The interesting part is the gating logic; keep the
  staging zone in both reach envelopes (~x ∈ [-0.1, 0.2]).

## Task 4 — `factory_sort_interrupt` (interrupt / safety)

**Owner:** _unclaimed_

Robot sorts a shared bin while the worker's arm periodically intrudes.

- **Scene**: shared input zone mid-bench with 4 mixed parts (2 cans +
  2 bottles, or colored cubes); two output zones/baskets on the robot's
  side (one per category).
- **Robot**: repeatedly picks a part and sorts it into the right
  output; refreshes planner obstacles before every plan
  (`_refresh_planner_obstacles`).
- **Human**: standing at the bench, periodically reaches into the
  shared input zone (inspect/touch motion from the interrupt pool) —
  the robot must wait or re-plan around the arm.
- **Success**: all parts sorted correctly; zero robot-avatar collision
  (use `--track-avatar-collision` metrics).
- **Precedent**: `envs/tasks/dump_bin_interrupt.py`,
  `envs/tasks/categorize_interrupt.py`, `InspectMotionMixin`.
- **Risk**: medium-low — the interrupt machinery is proven; tuning the
  intrusion timing vs robot cycle is the main work.

## Task 5 — `factory_pack_relay` (two-stage relay)

**Owner:** _unclaimed_

Mini production line: robot fills, human carries to outbound.

- **Scene**: one open-top tray/box at a fill station mid-bench; 2–3
  products on the robot's side; outbound zone = the pallet at
  `(+1.55, +0.45)` or a marked floor/bench zone on the worker's side.
- **Robot**: places all products into the tray.
- **Human**: once the tray is full (task-tracked condition), walks to
  the bench, two-hand carries the tray
  (`attach_object_to_two_hands`, kinematic hold is avatar-legal),
  walks it to the outbound zone, sets it down.
- **Success**: tray at the outbound zone with all products still inside
  (re-check product-in-tray AABB after set-down).
- **Precedent**: avatar walk + carry in `envs/avatar/controller.py`
  (walk/turn modules), two-hand attach in `envs/avatar/robot.py`.
- **Risk**: highest of the five — items must stay in the kinematically
  carried tray (settle before pickup; carry slowly, keep the tray
  level). Verify item-retention early with a smoke video before
  polishing.

---

## Verification expectations (all tasks)

1. `scripts/collect.py --task factory_<name> --episodes 1` succeeds and
   produces a watchable video under `data/factory_<name>/video/`.
2. `check_success()` is a real geometric check (AABB containment /
   distance), not a scripted flag.
3. Respect `docs/general/running_jobs.md`: srun + robotwin python, CPU
   Genesis backend, logs under `job_logs/`, review media under
   `data/<agent-or-task>/`.
