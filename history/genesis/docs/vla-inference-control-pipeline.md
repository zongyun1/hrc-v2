# VLA Inference and Robot Control Pipeline Notes

Date: 2026-05-27

This note records the current Genesis HR Bench inference wiring for OpenVLA-OFT,
pi0, and pi0.5, plus the robot-control issue found while debugging the
OpenVLA-OFT failures.

## Summary

- OpenVLA-OFT now runs through `baseline/servers/openvla_oft_server.py` with the
  Genesis-specific 7-D proprio/action shape from the OpenVLA-OFT constants patch.
- pi0 and pi0.5 now run through the LeRobot v0.5.2 server path
  (`baseline/servers/lerobot_server.py`) instead of the older OpenPI/JAX server
  path.
- The remote EE clients now integrate predicted EE deltas against the last
  commanded EE target by default, not the measured TCP pose at the next sim step.
- `BaseTask` now skips IK when the requested absolute EE pose is already equal to
  the current EE pose within tolerance. This prevents no-op labels from causing
  small null-space IK drift.
- The baseline submodule edits are recorded as tracked patch files:
  `patches/openvla-oft-genesis-hr-bench.patch` and
  `patches/lerobot-genesis-hr-bench.patch`.

## OpenVLA-OFT Inference

The OpenVLA-OFT server changes are in `baseline/servers/openvla_oft_server.py`.
The server now:

- Resolves the OpenVLA-OFT submodule path correctly from `baseline/servers/` to
  `baseline/openvla-oft`.
- Imports `PROPRIO_DIM` from `prismatic.vla.constants` instead of assuming the
  Libero-style 8-D proprio vector.
- Decodes PNG payloads into RGB numpy arrays, matching the OpenVLA-OFT robot
  helper input path.
- Caches the 8-step action chunk returned by OpenVLA-OFT and serves one action
  per `/predict` call until the chunk is exhausted.
- Clears the action chunk when the client sends `reset=true` at episode start.
- Reports `chunk_size` and `unnorm_key` from `/health` for easier server-log
  debugging.

The OpenVLA-OFT policy client changes are in `scripts/vla_client.py`.
For EE eval it now sends Genesis proprio as:

```text
[x, y, z, roll, pitch, yaw, gripper_closed]
```

The current benchmark EE pose is still stored as quaternion pose for control:

```text
[x, y, z, qw, qx, qy, qz]
```

The policy converts the quaternion to `sxyz` Euler only for model proprio. The
returned model action is treated as Genesis EE delta:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper_closed]
```

That delta is composed into the benchmark's absolute EE command format:

```text
[x, y, z, qw, qx, qy, qz, gripper_open]
```

The default `config/vla.yml` `unnorm_key` is now `null`. `scripts/run_vla_eval.sh`
passes `--unnorm-key genesis_hr_bench` automatically when the OpenVLA-OFT
checkpoint name contains `genesis`; otherwise the caller can pass it explicitly.

## pi0 and pi0.5 Inference

pi0 and pi0.5 are served by `baseline/servers/lerobot_server.py` using the
LeRobot v0.5.2 policy classes:

```text
pi0  -> lerobot.policies.pi0.modeling_pi0.PI0Policy
pi05 -> lerobot.policies.pi05.modeling_pi05.PI05Policy
```

The LeRobot server now:

- Loads saved `policy_preprocessor.json` and `policy_postprocessor.json` when
  they exist.
- Uses the saved preprocessor/postprocessor path for pi0/pi0.5 so the real
  PaliGemma tokenizer and observation rename processors are used at inference.
- Refuses to silently run pi0/pi0.5 with dummy language tokens if the tokenizer
  or saved processors cannot load.
- Reads the saved `RenameObservationsProcessor` map and feeds raw training-time
  image keys (`observation.image`, `observation.wrist_image`) so LeRobot can map
  them into policy slots itself.
- Resets policy and processors when the client sends `reset=true`.

`scripts/vla_client.py` also avoids duplicating the gripper value when
`joint_state` already contains the gripper as its final element. This matters for
qpos/qpos_abs pi checkpoints, where an accidental extra gripper element shifts
the action/proprio schema.

The current `scripts/run_vla_eval.sh` defaults are:

```text
openvla_oft -> action_type=ee,      port=8769
pi0         -> action_type=qpos_abs, port=8767
pi05        -> action_type=qpos_abs, port=8768
```

Use `--action-type ee` for pi0/pi0.5 only when evaluating an EE-delta checkpoint.
The current qpos_target_abs checkpoints should stay on `qpos_abs`.

## Delta-EE vs Absolute-EE Finding

The Genesis EE dataset uses absolute EE state but delta EE actions:

```text
state  = [x, y, z, roll, pitch, yaw, gripper_closed]
action = [dx, dy, dz, droll, dpitch, dyaw, gripper_closed]
```

The benchmark robot controller consumes absolute EE targets:

```text
[x, y, z, qw, qx, qy, qz, gripper_open]
```

The OpenVLA-OFT failure was caused by two control-path mismatches interacting:

1. EE deltas were being composed against measured TCP feedback at each step. The
   simulated robot does not instantaneously reach the previous command, so the
   next delta was integrated from a lagged pose rather than from the target pose
   the policy logically commanded.
2. During stationary/no-op prefixes, repeatedly sending an absolute target equal
   to the current TCP still forced a fresh IK solve. IK can choose a slightly
   different null-space solution for the same TCP, so no-op labels produced small
   joint drift before the real manipulation began.

The fix is split across the policy client and robot control layer:

- `scripts/vla_client.py` stores `_ee_command_pose` and composes EE deltas on
  that commanded pose by default. Set `ee_delta_integration: measured` in config
  only when deliberate feedback integration is needed.
- OpenVLA-OFT and LeRobot-family clients use the same
  commanded-target integration path for Genesis EE-delta policies.
- `envs/base_task.py` skips `_move_to_pose()` when the requested absolute EE pose
  is already within 1 mm and 1 mrad of the current pose, while still applying the
  gripper command.

This keeps delta-action rollout consistent with training data and removes the
control drift that made OpenVLA-OFT fail early in tasks with a stationary prefix.

## Baseline Patch Files

`patches/openvla-oft-genesis-hr-bench.patch` now records the OpenVLA-OFT
submodule changes needed for Genesis:

- Genesis platform constants: 8-step chunks, 7-D actions, 7-D proprio.
- `genesis_hr_bench` OXE dataset config.
- Genesis RLDS transform preserving `1=closed` gripper convention.
- Dynamic HF component lookup for custom action head and proprio projector
  checkpoint filenames.

`patches/lerobot-genesis-hr-bench.patch` now records the LeRobot submodule
changes needed for selected-task diffusion/pi training:

- Longer distributed init timeout for cold-cache dataset construction.
- Selected-episode normalization stats when `--dataset.episodes` is used.
- Episode-aware sampler boundaries rebuilt in local selected-dataset index space.

## Eval Runner Changes

`scripts/run_vla_eval.sh` now:

- Accepts and forwards `--unnorm-key`.
- Defaults pi0/pi0.5 to `qpos_abs`.
- Auto-selects `genesis_hr_bench` unnorm for Genesis OpenVLA-OFT checkpoints.
- Uses the LeRobot venv for `lerobot_diffusion` serving.
- Forces `GENESIS_BACKEND=gpu` unless the caller overrides it.

`scripts/eval_hf_vla_suite.sh` now forwards `--start-seed` and `--output-json`
after `--`, so they are passed to `scripts/eval_vla.py` rather than being
consumed by `run_vla_eval.sh`.
