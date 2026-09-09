# RDT dataloader is not adapted for genesis-hr-bench — schema mismatch blocks sanity runs

**Setting:** running the 4-baseline pipeline (`pi0`, `pi05`, `openvla_oft`, `rdt`) on FAIR Cloud H200 cluster (`shared-aws-usw1-1`).

**Status (2026-05-20): RESOLVED in parent repo at `baseline/rdt_overrides/`.** The schema-adapted `hdf5_vla_dataset.py` + the three config files now live under `baseline/rdt_overrides/` and are layered onto the submodule by `scripts/finetune_rdt.sh` before each run (see `baseline/rdt_overrides/README.md`). After `git pull` (and the next `scripts/finetune_rdt.sh` invocation), the `ValueError: operands could not be broadcast together with shapes (208,8) (1,14)` should be gone.

The original report follows for posterity.

---

**Original status:** rdt sanity job cannot start. pi0/pi05/openvla_oft are unaffected. Their datasets are still building.

**TL;DR:** `baseline/rdt/data/hdf5_vla_dataset.py` in the genesis-hr-bench repo is still the upstream bimanual ALOHA template (all `[Modify]` markers intact). It hardcodes a 14-dim state vector (2× 7-DOF arms), 3 cameras, and reads `expanded_instruction_gpt-4-turbo.json`. Our actual genesis-hr-bench HDF5 (produced by `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`) has 8 dims (1× Franka + gripper), 2 cameras, and `instruction.json`. The converter's docstring claims `"baseline/rdt/data/hdf5_vla_dataset.py, adapted for genesis-hr-bench"`, but **the adaptation was never written into the repo**.

---

## 1. Concrete evidence — schema mismatch

### 1.1 What the data actually looks like

Sampled `runs/rdt/data/genesis_hr_bench/put_object_cabinet_assist/put_object_cabinet_assist_seed_239/episode.hdf5`:

```
action                                 shape=(208, 8)    dtype=float32
observations/qpos                      shape=(208, 8)    dtype=float32
observations/images/cam_high           shape=(208,)      dtype=object   (vlen JPEG)
observations/images/cam_right_wrist    shape=(208,)      dtype=object   (vlen JPEG)
```

Sibling files in the episode dir: `episode.hdf5`, `instruction.json` (no expanded variant).

`instruction.json`:
```json
{"instruction": "place the tissue box inside the drawer opened by the human"}
```

qpos per-dim ranges (one episode):
- dims 0..6: Franka joint angles in radians (typical [-2.8, 3.0] range)
- dim 7: gripper, observed range **0.596 → 1.0** in this episode — **NOT the ALOHA 4.79 scale**

action per-dim ranges (one episode):
- dims 0..6: joint deltas (small, ~±0.1 rad/step)
- dim 7: gripper command, **0.596 → 1.0** in this episode

### 1.2 What the upstream RDT bimanual loader assumes

`baseline/rdt/data/hdf5_vla_dataset.py:164-169` (in both `parse_hdf5_file` and `parse_hdf5_file_state_only`):

```python
qpos = qpos / np.array(
   [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]
)
target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE] / np.array(
   [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
)
```

14-dim normalization (positions 6 and 13 are the two ALOHA grippers).

`baseline/rdt/data/hdf5_vla_dataset.py:188-196`:
```python
UNI_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
```

Maps **14 source dims** into the 128-dim unified vector across 6 left-arm joints + left gripper + 6 right-arm joints + right gripper. Note `range(6)` — only 6 joints per arm, not 7. Genesis HR Bench has **7 Franka joints**, so even the 7-DOF mapping needs adjustment.

`baseline/rdt/data/hdf5_vla_dataset.py:142`:
```python
with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
```

Expects an instruction file that doesn't exist in our conversion.

`baseline/rdt/data/hdf5_vla_dataset.py:230-233`:
```python
cam_left_wrist = parse_img('cam_left_wrist')   # KeyError on our data
cam_left_wrist_mask = cam_high_mask.copy()
cam_right_wrist = parse_img('cam_right_wrist')
cam_right_wrist_mask = cam_high_mask.copy()
```

Reads `cam_left_wrist`; our converter never wrote it (single-arm setup). The docstring on line 237 mentions returning a zero-shape array for unavailable cameras, but the active code path doesn't.

### 1.3 The failing trace (sbatch job 1370659, `find -L` + `RDT_HDF5_DIR` already applied)

```
File "baseline/rdt/data/compute_dataset_stat_hdf5.py", line 81, in <module>
    vla_dataset = HDF5VLADataset()
File "baseline/rdt/data/hdf5_vla_dataset.py", line 45, in __init__
    valid, res = self.parse_hdf5_file_state_only(file_path)
File "baseline/rdt/data/hdf5_vla_dataset.py", line 288, in parse_hdf5_file_state_only
    qpos = qpos / np.array(
ValueError: operands could not be broadcast together with shapes (208,8) (1,14)
```

---

## 2. Mismatch summary

| Property | genesis-hr-bench HDF5 (actual) | RDT bimanual ALOHA loader (assumes) |
|---|---|---|
| qpos dims | **8** (7 Franka joints + 1 gripper) | 14 (2 arms × 7, grippers at idx 6, 13) |
| action dims | **8** | 14 |
| Gripper normalization | ~[0, 1] range (Franka, observed 0.596→1.0) | divide by `4.7908`/`4.7888` (ALOHA) |
| Action gripper normalization | ~[0, 1] | divide by `11.8997`/`13.9231` (ALOHA) |
| Cameras | `cam_high` + `cam_right_wrist` | `cam_high` + `cam_left_wrist` + `cam_right_wrist` |
| Instruction file | `instruction.json` (`{"instruction": "..."}`) | `expanded_instruction_gpt-4-turbo.json` (with `instruction`/`simplified_instruction`/`expanded_instruction` keys) |
| Unified-vec slot count | 7 joints + 1 gripper = 8 slots | 6+1+6+1 = 14 slots |
| Joints per arm | 7 (Franka) | 6 (per ALOHA's `range(6)`) |

---

## 3. What I've already patched (uncommitted; pure plumbing, no semantic changes)

Three small fixes applied to expose the schema bug. None of these affect data semantics — they just let the loader find the right files.

### 3.1 `scripts/finetune_rdt.sh` — `find` → `find -L`

Line 67. The dataset-existence guard used `find $RDT_HDF5_DIR` which doesn't follow symlinks. Our dataset lives at `runs/rdt/data/genesis_hr_bench`, which is a symlink to `/lustre-storage/datasets/zengh/genesis-hr-bench/rdt_data/genesis_hr_bench`.

### 3.2 `baseline/rdt/data/hdf5_vla_dataset.py` — honor `RDT_HDF5_DIR` env var + follow symlinks

The class hardcodes `HDF5_DIR = "data/datasets/agilex/rdt_data/"` and `os.walk(HDF5_DIR)` (no `followlinks`), so it never sees our data regardless of where it lives. Patched to:

```python
HDF5_DIR = os.environ.get("RDT_HDF5_DIR", "data/datasets/agilex/rdt_data/")
...
for root, _, files in os.walk(HDF5_DIR, followlinks=True):
```

`finetune_rdt.sh` already sets and exports `RDT_HDF5_DIR`, so this is just wiring the existing contract through.

### 3.3 `scripts/env.local.sh` — add `CUDA_HOME` + `HF_HOME` (separate cluster-bringup work)

```bash
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/lustre-storage/datasets/zengh/huggingface
```

CUDA_HOME satisfies DeepSpeed's import-time probe (see `finetune_rdt.sh:94-110`). HF_HOME routes the ~20 GB T5-XXL + SigLIP + RDT-1B downloads to lustre instead of FSX. These are cluster-port concerns, mentioned for completeness; not RDT-loader related.

None of the above changes data semantics. They are not the fix.

---

## 4. What's missing — the actual schema adaptation

`baseline/rdt/data/hdf5_vla_dataset.py` needs to be rewritten for single-arm Franka. Both `parse_hdf5_file` (lines 81-253) and `parse_hdf5_file_state_only` (lines 255-322) need parallel changes. The upstream code has `[Modify]` markers on both functions exactly to flag this work.

Open decisions only the benchmark author can make:

1. **Gripper normalization constants.** What's the open/closed range of the Franka gripper in `vla_data`? The `4.7908` / `11.8997` ALOHA constants are clearly wrong but I don't know the right values without digging into the data collection code. Observed range in one episode is `[0.596, 1.0]` for both state and action, which suggests the data may already be normalized into `[0, 1]` — in which case the divisor should be `1.0`.

2. **Which unified-vec arm slot?** Should our 7 Franka joints map to `right_arm_joint_{0..6}_pos` (and leave left-arm slots zero) or `left_arm_joint_{0..6}_pos`? Note RDT's `STATE_VEC_IDX_MAPPING` originally only goes 0..5 per arm — does it have slots for `*_joint_6_pos` (7th joint), or does Franka's extra joint need a different slot like `*_wrist_joint_pos` or similar?

3. **Instruction file format.** Two options:
   - Patch the loader to read `instruction.json` and use the single instruction string for all 3 sampling paths (`instruction`/`simplified_instruction`/`expanded_instruction`).
   - Re-run `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py` to also emit `expanded_instruction_gpt-4-turbo.json` (with all 3 fields, perhaps reusing the same text or calling an LLM for variants).

4. **Missing `cam_left_wrist`.** Return zero-shape `np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))` per the docstring hint on line 237, and set `cam_left_wrist_mask` to all-False? Or duplicate `cam_right_wrist`?

5. **Dataset stats.** Once the parse functions work, `baseline/rdt/configs/dataset_stat.json` should be regenerated from genesis-hr-bench. Right now it ships agilex/fractal/etc. stats. The loader currently has `self.DATASET_NAME = "agilex"` — if we change it to e.g. `"genesis_hr_bench"`, the stats lookup will miss and `compute_dataset_stat_hdf5 --skip_exist` will recompute. Without the rename, the agilex stats get used and normalization is wrong for full training.

---

## 5. Repro

Environment already prepared at `~/envs/rdt/` (Python 3.10, torch 2.1.0+cu121, deepspeed 0.14.2, diffusers 0.27.2, transformers 4.41.0, numpy 1.26.4). Dataset already converted (14,794 episodes from 17,863, on lustre via symlink). All preconditions for sanity are met except the schema adaptation.

From a login node (`zengh-login-0`) shell with `tmux`:

```bash
cd /storage/home/zengh/projects/genesis-hr-bench
scripts/finetune.sh --model rdt --slurm --gpus-per-node 1 --sanity --time 3:00:00
```

Returns an sbatch jobid. Currently fails in ~1 minute with the ValueError above.

Latest failed run log: `runs/rdt/finetune/slurm_1370659.{out,err}`.

---

## 6. References

- Converter (writes the HDF5s, schema is documented in its docstring): `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`
- Dataloader (needs adaptation): `baseline/rdt/data/hdf5_vla_dataset.py`
- Wrapper script (working, sets `RDT_HDF5_DIR`): `scripts/finetune_rdt.sh`
- Cluster pipeline doc: `docs/vla-finetune-pipeline.md`
- Setup plans: `~/.llms/plans/vla_pi_family_h200_setup.plan.md`, `~/.llms/plans/vla_4baseline_h200_setup.plan.md`
- Upstream RDT (for comparing how agilex/aloha did it): `baseline/rdt/data/agilex/` and `baseline/rdt/data/aloha/`
- Unified state vec mapping: `baseline/rdt/configs/state_vec.py` (`STATE_VEC_IDX_MAPPING`)

Other 3 baselines (`pi0`, `pi05`, `openvla_oft`) are blocked on the LeRobot v3.0 / RLDS dataset builds (sbatch `1370536`, ~8h remaining) and are unrelated to this report.
