# Pre-Grasp Direction Sampling System

## Overview

All tasks use a shared pre-grasp direction sampling method (`BaseTask.find_best_pregrasp`) to find the best approach direction before grasping or pressing an object. Instead of using a single hard-coded pre-grasp direction, the system samples many candidate directions, filters them by task-specific constraints, solves IK for each, and picks the one with the smallest end-effector position error.

## API

```python
BaseTask.find_best_pregrasp(
    grasp_tcp: Pose,       # TCP pose at the grasp/contact point (world frame)
    arm_tag: str,          # "right" or "left"
    pre_dist: float,       # backup distance from grasp point (meters)
    constraints: list,     # half-space constraints [(normal_vec, min_dot), ...]
    n_az: int = 12,        # azimuth samples
    n_el: int = 8,         # elevation samples
    bar_axis: np.ndarray = None,  # optional TCP +X axis hint (e.g. handle bar direction)
) -> Pose  # returns link-frame Pose ready for move_and_execute
```

## Constraint Interface

Constraints are specified as a list of `(normal_vec, min_dot)` tuples defining half-space filters:

```python
constraints = [
    ([1, 0, 0], 0.1),    # direction must have positive X component (dot > 0.1)
    ([0, 0, -1], -0.1),  # direction must not come from below
]
```

A candidate direction `d` passes **only if** `dot(d, normal) > min_dot` for **all** constraints.

### Interpretation

| Constraint | Meaning |
|---|---|
| `([1,0,0], 0.1)` | Approach has +X component (e.g., don't penetrate microwave behind) |
| `([0,0,-1], -0.1)` | Not approaching from below table |
| `([0,0,1], 0.3)` | Strong downward approach (for top-down press) |
| `([0,-1,0], 0.0)` | Approach has -Y component |

## Sampling Strategy

1. **Direction generation**: Samples `n_az × n_el` directions in spherical coordinates around the grasp TCP's +Z axis (the default approach direction). Azimuth spans ±80°, elevation spans -60° to +30°.

2. **Constraint filtering**: Each direction is checked against all half-space constraints. Invalid directions are discarded.

3. **IK evaluation**: For each valid direction, the system:
   - Builds a pre-grasp TCP pose at `grasp_pos - direction * pre_dist`
   - Constructs a proper TCP rotation frame (using `bar_axis` for +X if provided)
   - Converts TCP → link pose via `tcp_to_link_pose`
   - Solves IK from current joint configuration
   - Teleports arm to IK solution and measures actual EE position error

4. **Selection**: Returns the link pose with minimum EE position error.

5. **Fallback**: If no valid direction passes both constraints and IK, falls back to the default approach direction (grasp TCP +Z).

## Per-Task Configurations

### open_microwave
```python
self.find_best_pregrasp(
    grasp_tcp, arm_tag, pre_dist=0.12,
    constraints=[
        ([1, 0, 0], 0.1),    # +X: don't penetrate microwave body
        ([0, 0, -1], -0.1),  # not from below table
    ],
    bar_axis=self._handle_bar_world(),  # handle bar direction for TCP +X
)
```

### take_from_human_safety (knife grasp)
```python
self.find_best_pregrasp(
    grasp_tcp_world, arm_tag, pre_dist=grasp.pre_distance,
    constraints=[
        ([0, 0, -1], -0.1),  # not from below
    ],
)
```

### pour_water (bottle grasp)
```python
self.find_best_pregrasp(
    grasp_tcp_world, arm_tag, pre_dist=grasp.pre_distance,
    constraints=[
        ([0, 0, -1], -0.1),  # not from below
    ],
)
```

### hold_object_steady (top-down press)
```python
self.find_best_pregrasp(
    press_tcp, arm_tag, pre_dist=0.12,
    constraints=[
        ([0, 0, 1], 0.3),   # must come from above
    ],
)
```

## Implementation Location

- Method: `envs/base_task.py` → `BaseTask.find_best_pregrasp()` (line ~395)
- TCP conversion: `envs/grasp.py` → `tcp_to_link_pose()`
