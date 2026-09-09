# Bug report: `stamp_documents` task has 60 source episodes with 6-DOF qpos, breaking the RDT pipeline

**Summary:** running 4-baseline VLA pipeline on FAIR Cloud h200 from `zhouqh/hrbench` HF dataset
**Date:** 2026-05-21
**Severity:** RDT-only blocker. pi0 / pi05 / openvla_oft are unaffected.
**Affected dataset:** `huggingface.co/datasets/zhouqh/hrbench` @ main, downloaded via `scripts/sync_hf_assets.py`

## TL;DR

In `vla_data/stamp_documents/rasterizer/`, **60 of 401 seed episodes have `qpos` / `qpos_delta_action` of shape `(T, 6)` instead of the usual `(T, 7)`** (i.e., 6-DOF arm, not 7-DOF Franka). Every other task in the dataset (~25 tasks, all 17k+ episodes I sampled) has `(T, 7)`. The RDT converter doesn't validate dim, silently passes 6-DOF through, and the resulting `episode.hdf5` files have `qpos` / `action` of shape `(T, 7)` instead of `(T, 8)`. The RDT loader (your `baseline/rdt_overrides/data/hdf5_vla_dataset.py`) hardcodes 8 `UNI_STATE_INDICES` (7 Franka joints + 1 gripper), so it crashes on the first 6-DOF episode it iterates.

The OFT (RLDS) and LeRobot pipelines aren't affected because they default to `control_mode="ee"` (EE state is always 7-dim regardless of arm DOF).

## How it surfaces

`scripts/finetune.sh --model rdt --slurm --gpus-per-node 1 --sanity` → sbatch jobs 1370654, 1370659, 1370948 (latest). Fails in ~3-4 min during `compute_dataset_stat_hdf5.py`:

```
File "baseline/rdt/data/hdf5_vla_dataset.py", line 208, in parse_hdf5_file_state_only
    state = self._fill_in_state(qpos[first_idx - 1:])
File "baseline/rdt/data/hdf5_vla_dataset.py", line 104, in _fill_in_state
    uni_vec[..., self.UNI_STATE_INDICES] = values
ValueError: shape mismatch: value array of shape (545,7) could not be broadcast to indexing result of shape (545,8)
```

Triggering file (alphabetically first 6-DOF in iteration order): `runs/rdt/data/genesis_hr_bench/stamp_documents/stamp_documents_seed_0/episode.hdf5` (545 timesteps).

## Root cause — source-data inconsistency in `vla_data/stamp_documents/`

Across the full `vla_data/`:

| Task | qpos.shape[-1] = 6 | qpos.shape[-1] = 7 |
|---|---|---|
| **`stamp_documents`** | **60 seeds** | **341 seeds** |
| every other task (~25) | 0 | all |

Concrete schema dump of one **6-DOF** seed (`vla_data/stamp_documents/rasterizer/seed_0/steps.h5`):
```
ee                  shape=(622, 7)   float32         ← EE stays 7-dim (xyz + rpy + ???)
ee_delta_action     shape=(622, 6)   float32
qpos                shape=(622, 6)   float32         ← arm is 6-DOF here
qpos_delta_action   shape=(622, 6)   float32
gripper_meas        shape=(622,)     float32
gripper_action      shape=(622,)     float32
images/head_camera  shape=(622,)     object (JPEG)
images/right_wrist  shape=(622,)     object (JPEG)
depth/head_camera   shape=(622, 480, 640) float16
depth/right_wrist   shape=(622, 480, 640) float16
```

Concrete schema dump of one **7-DOF** seed (`vla_data/blocks_ranking_rgb_assist/rasterizer/seed_10/steps.h5`):
```
ee                  shape=(361, 7)   float32
ee_delta_action     shape=(361, 6)   float32
qpos                shape=(361, 7)   float32         ← arm is 7-DOF (Franka)
qpos_delta_action   shape=(361, 7)   float32
gripper_meas        shape=(361,)     float32
gripper_action      shape=(361,)     float32
(images/depth identical)
```

Even the `meta.json` doesn't flag the embodiment — it just says `"task": "stamp_documents"` / `"renderer": "rasterizer"`. So this looks like `stamp_documents` was collected against **two different robot embodiments** at different times — 60 episodes on a 6-DOF arm (UR5/UR10/some custom?) and 341 on the Franka 7-DOF.

## Where each piece of code is implicated

### Source data
- `vla_data/stamp_documents/rasterizer/seed_*/steps.h5` — 60 of 401 seeds have 6-dim qpos. Generated upstream by your collection pipeline on UMass gypsum (per `_task_restart_backups/oil_bottle_recovery_20260517_080606_ET/rasterizer/reset_loop_multi_57813144.log`: `node=gypsum-gpu050`).

### Converter — silently passes through
`scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`, lines 51-67:
```python
qpos        = np.asarray(f["qpos"][:], dtype=np.float32)              # docstring says (T, 7)
qpos_delta  = np.asarray(f["qpos_delta_action"][:], dtype=np.float32) # docstring says (T, 7)
grip_meas   = np.asarray(f["gripper_meas"][:], dtype=np.float32)      # (T,)
grip_act    = np.asarray(f["gripper_action"][:], dtype=np.float32)    # (T,)
...
state  = np.concatenate([qpos[:t],       grip_meas[:t, None]], axis=1)  # (T, 8) ← actually (T, 7) for stamp_documents
action = np.concatenate([qpos_delta[:t], grip_act[:t,  None]], axis=1)  # (T, 8) ← same
```
No assertion that `qpos.shape[-1] == 7`. So for 6-DOF seeds, the output is silently `(T, 7)` not `(T, 8)`.

### RDT loader override — hard 8-dim assumption
`baseline/rdt_overrides/data/hdf5_vla_dataset.py`, lines 40-42:
```python
# 7 Franka arm joints + 1 gripper -> slots in RDT's 128-D unified vector.
UNI_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
```
8 indices. Line 104 (`_fill_in_state`):
```python
uni_vec[..., self.UNI_STATE_INDICES] = values    # values=(T,7) -> shape mismatch
```

### OFT (RLDS) + LeRobot — UNAFFECTED
`scripts/build_genesis_hr_bench_rlds.py` defaults `control_mode="ee"` (line 68), and EE state is `[x, y, z, roll, pitch, yaw, gripper]` (7,) regardless of arm DOF. The qpos path inside the same builder (`control_mode in {"qpos", "qpos_abs"}`, lines 137-156) would have the same 6-vs-7 problem, but that path isn't exercised by the standard build. LeRobot reads from TFDS output, so it inherits the EE-mode 7-dim state — also unaffected.

## Scope of impact

| Pipeline | Reads | Affected? |
|---|---|---|
| OFT (RLDS) | `vla_data/` directly, EE mode | ❌ no — EE state is always (7,) |
| pi0 / pi05 / pi0_fast / smolvla | LeRobot v3.0 (derived from RLDS) | ❌ no — inherits EE state |
| **rdt** | RDT HDF5 (from `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`), joint-space | **✅ yes** — 60 of 14794 files break the loader |

For rdt, the 60 broken files are 0.41% of the converted dataset. All in `stamp_documents/`. If we drop them, rdt loses 60 / 14794 = 0.4% of training data, but `stamp_documents` itself loses 60 / 401 = 15% of its task data.

## Open questions for you (zhouqh) to decide

1. **Was the 6-DOF subset of `stamp_documents` intentional?** I.e., was this task deliberately collected on two embodiments (UR5+Franka)? Or did one collection job use the wrong robot config? The split is 60 vs 341, and EE stays 7-dim throughout — feels accidental, but I can't tell.

2. **If accidental, which side wins?**
   - **Drop the 60 6-DOF seeds** from the dataset upstream (re-push to HF without them). Cleanest. Affects all downstream users.
   - **Re-collect those 60 seeds on Franka** so the whole task is consistent at 7-DOF. Preserves data volume.
   - **Drop `stamp_documents` from the rdt baseline only** (keep for OFT/pi0/pi05 which don't care). Quick local workaround.

3. **If intentional (mixed-embodiment dataset by design), how should the RDT loader handle it?**
   - Pad 6-DOF to 7-DOF with zero (treat as "joint 7 = 0 fixed")? Semantically dishonest for downstream training.
   - Separate datasets per embodiment, two RDT runs? Probably the right answer if mixed-embodiment is the goal.
   - Skip 6-DOF in the loader (silent loss; would prefer not, per your preference for fail-loud).

4. **Should the converter `assert qpos.shape[-1] == 7`?** Right now it silently produces the wrong-dim output and the failure happens 17h later in the loader. A converter-level assert (or skip-and-log) would surface this immediately at conversion time, regardless of which fix path you pick above.

## What I'm doing right now (pending your call)

- 3 other baselines (openvla_oft, pi0, pi05) are unblocked — sanity jobs are queued and auto-start when the build (1370941, ~5h ETA) finishes. They're not affected by this bug.
- For rdt specifically: **the failed sanity ran, the 14794 converted .hdf5 files are on disk**, and I'm holding off on a re-submit until you say which fix you want.

If you want a quick local unblock to validate rdt today: I can `mv` the 60 broken `runs/rdt/data/genesis_hr_bench/stamp_documents/*_seed_<N>/` dirs aside and re-submit ft_rdt. That gives you a working sanity in <30 s + ~2 h job time. We can still apply the upstream fix later.

## Files referenced

- Source data (broken): `vla_data/stamp_documents/rasterizer/seed_{0..400}/steps.h5` (60 with qpos=6, 341 with qpos=7)
- Converter (no shape assert): `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py:51-67`
- Loader override (assumes 8-dim): `baseline/rdt_overrides/data/hdf5_vla_dataset.py:40-42, 101-104`
- Failing sanity logs: `runs/rdt/finetune/slurm_1370948.{out,err}`
- Pre-existing report (different error, same root cause now revealed): `docs/rdt-loader-schema-issue.md`
