# Kitchen scene — reusable builder for tabletop cooking tasks

`envs/scenes/kitchen.py` is a drop-in scene builder that dresses a `BaseTask`
tabletop into a kitchen counter: 15 canonical items on a 1.70 × 0.95 m
walnut counter, a cooktop with skillet on the left, a wall cabinet with
stocked shelves above, a Franka arm mounted on the right, and the avatar
standing next to the cooktop.

It is built so new cooking tasks don't have to re-derive asset scales,
positions, or robot mounting — inherit one mixin, call one function, and
customize with class attributes.

## Minimal task

```python
from envs.base_task import BaseTask
from envs.scenes import KitchenSceneMixin

class CookEgg(KitchenSceneMixin, BaseTask):
    use_avatar = True

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

    def load_actors(self):
        self.build_kitchen()

        # reach into pre-spawned fixtures
        self.pan     = self.kitchen_items["skillet"]
        self.cooktop = self.kitchen_cooktop["body"]
        self.cabinet = self.kitchen_backdrop["cabinet"]   # dict of 5 panels

        # spawn any task-specific extras here
        # ...

    def play_once(self):
        ...

    def check_success(self) -> bool:
        ...
```

Register it in `envs/tasks/__init__.py` (import + `TASK_MAP` entry) and it
runs under `scripts/collect.py` / `scripts/eval.py` like any other task.

See `envs/tasks/kitchen_scene_preview.py` for a runnable end-to-end example
(`KitchenScenePreview` renders headless; `KitchenScenePreviewFull` enables the
robot and avatar).

## What `build_kitchen()` does

One call, three steps:

| Step | Helper | Attribute set |
|------|--------|---------------|
| 1 | `build_kitchen_backdrop` | `self.kitchen_backdrop` — `{"wall", "cabinet": {...5 panels}, "cabinet_items": {role: entity}, "floor"?}` |
| 2 | `build_kitchen_cooktop`  | `self.kitchen_cooktop`  — `{"body", "burners": [cyl, cyl]}` |
| 3 | `build_kitchen_table`    | `self.kitchen_items`    — `{role: entity}` for each spawned `KitchenItem` |

Robot mounting, counter geometry, avatar pose are applied automatically
through mixin overrides of `_load_robot`, `_create_table`, and the
`avatar_init_pos` / `avatar_init_rot` class attributes. The mixin also
swaps the recording / side cameras to a 3/4 view framed on the full
counter.

## Class-attribute knobs

All of these are `KitchenSceneMixin` class attributes you can override in
your subclass without touching the helpers.

| Attribute | Purpose | Default |
|-----------|---------|---------|
| `KITCHEN_LAYOUT` | list of `KitchenItem` for the table | `DEFAULT_KITCHEN_LAYOUT` (15 items) |
| `KITCHEN_BACKDROP` | `KitchenBackdrop(...)` dataclass, or `None` to skip wall+cabinet | wall + cabinet + 8 default items |
| `KITCHEN_COOKTOP` | dict of kwargs for `build_kitchen_cooktop`, or `None` to skip | `{"xy": (-0.55, 0.28)}` |
| `KITCHEN_ROBOT_KWARGS` | merged into `config["robot_kwargs"]` before Franka loads | `{"pos": [0.60, -0.30, 0.765]}` |
| `KITCHEN_TABLE_HALF` | counter half-size `(hx, hy)` | `(0.85, 0.475)` → 1.70 × 0.95 m |
| `KITCHEN_TABLE_LEG_XY` | 4 leg `(x, y)` positions | corners at `(±0.80, ±0.43)` |
| `KITCHEN_TABLE_COLOR` | top-panel RGB tuple, or `None` for BaseTask default | `(0.56, 0.38, 0.22)` (walnut) |
| `avatar_init_pos` / `avatar_init_rot` | avatar spawn pose | `(-0.75, -0.90, -0.18)`, facing +y |
| `recording_camera_pos` / `_lookat` | main camera | front-left, `lookat=(0,0,0.90)` |
| `side_camera_pos` / `_lookat` | side camera | front-right, same lookat |

## Customizing items on the table

`KitchenItem` is a dataclass — one per mesh on the counter. The builder
measures each mesh with trimesh, applies your `euler_deg`, scales so the
longest world-axis dimension matches `target_dim_m`, and z-places so the
world-AABB bottom sits exactly at `table_top + extra_z - sink_z`.

```python
from envs.scenes import KitchenItem, DEFAULT_KITCHEN_LAYOUT

class CutBread(KitchenSceneMixin, BaseTask):
    KITCHEN_LAYOUT = [
        # drop the cutboard + knife + bread + apple from the default,
        # and keep just the dressing items:
        *[it for it in DEFAULT_KITCHEN_LAYOUT
            if it.role not in ("cutboard", "knife", "bread")],

        # your own, using your own xy + scale:
        KitchenItem(
            role="cutboard", asset="104_board", model_id=0,
            target_dim_m=0.32, euler_deg=(90, 0, 0),
            xy=(-0.55, -0.05), is_static=True, convex=True,
        ),
        KitchenItem(
            role="knife", asset="034_knife", model_id=0,
            target_dim_m=0.22, euler_deg=(0, -90, 90),
            xy=(-0.55, -0.15), extra_z=0.025,
            is_static=False, convex=True,
        ),
    ]
```

Conventions worth knowing:
- `xy` is the **world-AABB center** of the rotated+scaled mesh, not the
  mesh-origin — you can place two items at the same `xy` and they stack
  visually, regardless of where the original mesh's origin lives.
- `is_static=True` skips physics; use it for things the robot won't move
  (cutboard, skillet-on-cooktop).
- `extra_z` is for stacking (bread on plate, skillet on cooktop); `sink_z`
  is for embedding (rarely needed).
- `euler_deg` is extrinsic XYZ (transforms3d `axes="sxyz"`).
  `_UP_FROM_LOCAL_Y = (90, 0, 0)` is the default for most Genesis assets
  (local +Y is up); `_FLAT_KNIFE = (0, -90, 90)` lays the knife flat.

## Customizing the backdrop / cabinet

`KitchenBackdrop` is a dataclass with every wall + cabinet dimension
exposed. The most useful overrides:

```python
from envs.scenes import KitchenBackdrop, KitchenShelfItem, DEFAULT_CABINET_ITEMS

class TaskWithoutCabinet(KitchenSceneMixin, BaseTask):
    KITCHEN_BACKDROP = KitchenBackdrop(cabinet=False)  # wall only

class TaskWithCustomShelf(KitchenSceneMixin, BaseTask):
    KITCHEN_BACKDROP = KitchenBackdrop(
        cabinet_width=2.0, cabinet_height=0.40, cabinet_clearance=0.60,
        cabinet_items=(
            *DEFAULT_CABINET_ITEMS,
            KitchenShelfItem(role="extra", asset="031_jam-jar",
                             target_dim_m=0.14, x=+0.80),
        ),
    )
```

All cabinet items are loaded static + convex, so physics can't launch
them even if something bumps the cabinet shelf.

## Moving the robot or the avatar

The Franka position is injected at load time via `KITCHEN_ROBOT_KWARGS`.
Change it by setting the class attribute or by passing
`robot_kwargs` in the config:

```python
class TaskWithBackwardRobot(KitchenSceneMixin, BaseTask):
    KITCHEN_ROBOT_KWARGS = {"pos": [0.70, -0.20, 0.765],
                             "quat": [0.707, 0, 0, 0.707]}
```

Avatar pose is standard `BaseTask` — just override `avatar_init_pos` /
`avatar_init_rot` at the class level. The mixin's defaults put the avatar
on the left (cooking) side facing the counter; flip the rotation to
`[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]` if you want them on the opposite
side.

## Functional form (no BaseTask)

All three builders are exported from `envs.scenes` and take any object
with `.scene` and `.TABLE_TOP_Z` set. Useful for one-off rendering
scripts:

```python
from envs.scenes import (
    build_kitchen_backdrop, build_kitchen_cooktop, build_kitchen_table,
)

build_kitchen_backdrop(task=self)
build_kitchen_cooktop(task=self, xy=(-0.55, 0.28))
items = build_kitchen_table(task=self)   # dict {role: entity}
```

Call them from `load_actors()` **before** `scene.build()`, same as any
other entity in this repo.

## Previewing

Two debug tasks are wired up for visual iteration:

```bash
bash scripts/launch/kitchen_scene_preview.sh 0         # no robot / no avatar, fast
bash scripts/launch/kitchen_scene_preview_full.sh 0    # full stack, slower
```

Outputs land at `data/kitchen_scene/preview{,_full}/video/seed_0.mp4`.

## Where things live

| File | Contents |
|------|----------|
| `envs/scenes/kitchen.py` | `KitchenSceneMixin`, `KitchenItem`, `KitchenShelfItem`, `KitchenBackdrop`, `DEFAULT_KITCHEN_LAYOUT`, `DEFAULT_CABINET_ITEMS`, `build_kitchen_backdrop`, `build_kitchen_cooktop`, `build_kitchen_table`, `COOKTOP_XY`, `COOKTOP_HEIGHT_M` |
| `envs/scenes/__init__.py` | re-exports the above |
| `envs/tasks/kitchen_scene_preview.py` | `KitchenScenePreview` (headless) and `KitchenScenePreviewFull` (robot+avatar) reference implementations |
| `scripts/launch/kitchen_scene_preview.sh` | sbatch wrapper, no robot |
| `scripts/launch/kitchen_scene_preview_full.sh` | sbatch wrapper, full stack |

## Gotchas

- **Asset `model_data*.json` is unreliable.** Several SAPIEN-derived
  assets have stale `scale` / `extents` / `center` values. The builder
  sidesteps this by measuring each mesh directly with trimesh, so
  `target_dim_m` and z-placement are correct even when the JSON isn't.
- **Convex hulls hide concavities.** Bowls, cups, pans load as convex —
  dropping an apple *into* the bowl won't work without a non-convex
  collision hull. Place inside-concavity items on the table next to the
  bowl instead, and treat the bowl as visual.
- **Robot base mounts on the counter.** Default `pos=(0.60, -0.30, 0.765)`
  puts the Franka's base flange on top of the counter. If you move the
  robot below the counter height, add a visible pedestal or it'll float.
- **Cabinet items are static.** They cannot be picked up as-is. If you
  need a task where the robot grabs something from the shelf, respawn
  that specific item as a dynamic `KitchenItem` (or regular
  `load_object`) at the shelf location with `is_static=False`.
