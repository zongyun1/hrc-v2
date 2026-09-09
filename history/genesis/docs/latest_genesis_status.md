# Latest Genesis Status

Date: 2026-06-21

## Runtime

Use the repo latest environment and GPU backend:

```bash
source scripts/env.sh
export GENESIS_BACKEND=gpu
```

Validated package versions:

- `genesis-world==1.1.2`
- `gs-nyx==0.1.2`
- `gs-nyx-plugin==0.1.3`

## API Migration State

- Mesh loading is centralized through `envs.genesis_compat.mesh_frame_kwargs`.
- Benchmark mesh frames are preserved with `align=False`.
- `file_meshes_are_zup` defaults to `True` for every mesh type so task-authored
  asset frames are preserved without an extra Genesis axis conversion.
- DOF gain and force-range writes go through compatibility helpers in
  `envs/genesis_compat.py`.
- Runtime setup no longer forces stale OSMesa defaults. Use
  `GENESIS_BACKEND=gpu` for normal latest-Genesis validation.

## Validation Commands

Mesh loader tests:

```bash
GENESIS_BACKEND=gpu GENESIS_SOFTWARE_RENDER=0 \
  .venv-genesis-latest/bin/python -m pytest -q tests/test_mesh_loading_defaults.py
```

Robot/task smoke:

```bash
GENESIS_BACKEND=gpu GENESIS_SOFTWARE_RENDER=0 \
  .venv-genesis-latest/bin/python scripts/check_latest_genesis_integration.py \
  --python .venv-genesis-latest/bin/python \
  --robots franka \
  --tasks blocks_ranking_rgb \
  --steps 1
```

Pour-water no-avatar:

```bash
GENESIS_BACKEND=gpu GENESIS_SOFTWARE_RENDER=0 \
  .venv-genesis-latest/bin/python scripts/collect.py \
  --task pour_water \
  --config config/latest_genesis/pour_water_no_avatar.yml \
  --episodes 1 \
  --save-dir outputs/latest_genesis_current_validation/pour_water_no_avatar_gpu_pass \
  --start-seed 0
```

Pour-water static-avatar:

```bash
GENESIS_BACKEND=gpu GENESIS_SOFTWARE_RENDER=0 \
  .venv-genesis-latest/bin/python scripts/collect.py \
  --task pour_water \
  --config config/latest_genesis/pour_water_static_avatar.yml \
  --episodes 1 \
  --save-dir outputs/latest_genesis_current_validation/pour_water_static_avatar_gpu_pass \
  --start-seed 0
```

## Current Results

- GPU Genesis initialization passes.
- `tests/test_mesh_loading_defaults.py`: `3 passed`.
- Direct Franka GPU smoke passes.
- `blocks_ranking_rgb` GPU reset/step smoke passes.
- `pour_water` no-avatar GPU passes.
- `pour_water` static-avatar GPU passes with `0/93` avatar collisions.
- Animated avatar GLB loading and moving render works. The latest animated
  pour-water artifact renders correctly but still fails the task collision gate
  because the robot path intersects the avatar palm.

## Known Remaining Behavior Issue

`put_object_cabinet` reaches latest-Genesis GPU scene construction but can hit:

```text
Invalid constraint forces causing 'nan'. Please decrease Rigid simulation timestep.
```

Treat this as a task physics-stability issue, not an API-loading issue.
