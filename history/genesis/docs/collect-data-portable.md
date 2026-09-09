# Collecting `genesis_hr_bench` data — portable, no-shell-wrapper walkthrough

Step-by-step instructions for **producing** the raw rollouts and RLDS /
LeRobot datasets consumed by the current VLA fine-tune pipeline. You run the
Genesis simulator yourself, save the rollouts to disk, and convert them into
the on-disk format your chosen baseline expects. **No dataset is shipped to
you** — this doc and the fine-tune pipeline doc together cover the full path
from a fresh checkout to a trained checkpoint.

Pipeline overview:

```
1. Setup Genesis env + assets                                      §2
2. Collect raw rollouts (per-seed H5 + JPEG)                       §3, §4
   └─ output: vla_data/<task>/<renderer>/seed_<N>/{steps.h5, meta.json}
3. Build the RLDS dataset                                          §5
   └─ output: openvla/data/genesis_hr_bench/1.0.0/  (used by OpenVLA-OFT and as converter input)
4. Verify the RLDS dataset                                         §6
5. (LeRobot models) Convert RLDS → LeRobot                         §7
   └─ output: $HF_LEROBOT_HOME/genesis-hr-bench/genesis_hr_bench_lerobot/
6. Hand off to the fine-tune doc for your chosen baseline          §8
```

You only need step 5 if you plan to fine-tune **pi0, pi0.5, pi0-FAST, or
SmolVLA**. OpenVLA-OFT reads the RLDS copy from step 3 directly. RDT uses an
HDF5 conversion path covered in [`vla-finetune-pipeline.md`](vla-finetune-pipeline.md).

Every step is a raw command — **no `scripts/*.sh` wrappers are used**. Copy
the commands as-is on a fresh box. The shell wrappers
(`scripts/collect_vla.sh`, `scripts/vla_data/sbatch_*.sh`,
`scripts/vla_data/scheduler.py`) are tuned to the maintainers' SLURM cluster;
ignore them when reproducing on a different site.

All commands assume your cwd is the genesis-hr-bench checkout root. `cd`
there once at the start:

```bash
cd /path/to/genesis-hr-bench
```

---

## 1. What you produce

Up to three on-disk artefacts, in this order:

| Stage | Path | Format | Size (≈400 ep) | Required if you fine-tune… |
|---|---|---|---|---|
| **Raw rollouts** | `vla_data/<task>/<renderer>/seed_<N>/{steps.h5, meta.json, verdict.json}` | per-episode H5 + JSON | ~5–10 GB | always (intermediate) |
| **RLDS dataset** | `openvla/data/genesis_hr_bench/1.0.0/{features.json, dataset_info.json, dataset_statistics.json, *.tfrecord-*}` | TFDS / RLDS | ~300 MB | OpenVLA-OFT, LeRobot conversion source |
| **LeRobot dataset** | `$HF_LEROBOT_HOME/genesis-hr-bench/genesis_hr_bench_lerobot/` | LeRobot | ~300 MB | pi0, pi0.5, pi0-FAST, SmolVLA |

You can keep all three on the same box. The raw H5 dir is the largest
artifact but can be deleted once the RLDS / LeRobot copies are built and
verified.

The dataset pools **4 Franka tasks** into a single `train` split. Each
episode carries a per-episode `language_instruction` so models can learn
all four in one fine-tune:

```
pour_water                Grasp the bottle and pour water into the cup.
categorize_cooperative    "
categorize_interrupt      "
take_from_human_easy      "
```

(Source of truth: `scripts/vla_client.py`.)

Per-step features (Bridge-V2-shaped):

```
observation.image          (256, 256, 3) uint8     # head_camera RGB
observation.wrist_image    (256, 256, 3) uint8     # right_wrist RGB
observation.state          (7,) float32            # [x, y, z, roll, pitch, yaw, gripper]
action                     (7,) float32            # [dx, dy, dz, droll, dpitch, dyaw, gripper]
language_instruction       str
discount / reward / is_first / is_last / is_terminal: standard RLDS
```

Gripper polarity in the RLDS dataset: `1 = closed`, `0 = open` (Bridge
convention). The repo's runtime convention is the opposite (`1 = open`); the
RLDS builder flips it at write time. **Don't flip it again downstream.**

---

## 2. Setup

### 2.1 Hardware

| Stage | Requirement |
|---|---|
| Rasterizer collection (default) | CPU only. ~16 cores, ~60 GB RAM per process. Each seed = 5–20 min wall-clock. |
| Raytracer / LuisaRender (optional) | One Ampere+ GPU (cc ≥ 8.0), ~80 GB RAM. ~30–60 min/seed. Photo-realistic frames; rasterizer is the default for VLA training. |
| RLDS build (one-shot) | CPU only. ~10 min for ~400 episodes. |

You don't need a GPU to produce the dataset that the four fine-tune docs
consume — the **rasterizer** path is fine, and that's what the dataset
shipped to the receiving group has been built from.

### 2.2 Build the Genesis environment

The same `.venv-genesis-latest` env that runs the benchmark (see the README
Installation section). On a fresh box:

```bash
# Genesis runtime (genesis-world 1.2.0 + NYX; h5py for VLA recording is pinned here).
python3.10 -m venv .venv-genesis-latest
.venv-genesis-latest/bin/python -m pip install -U pip wheel
.venv-genesis-latest/bin/python -m pip install -r requirements-genesis-latest.txt

# Extra, only for RLDS/TFDS export (not needed for raw steps.h5 collection).
.venv-genesis-latest/bin/python -m pip install \
    "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" "protobuf<5"
```

Pin `tensorflow==2.15.0` exactly — newer 2.x versions break TFDS metadata
reading on uniform `dataset_info.proto` (`AttributeError: 'FieldDescriptor'
object has no attribute 'label'`).

### 2.3 Download assets

```bash
# Assets live separately from the code (URDFs, meshes, grasp YAMLs).
# Download from the OneDrive link the maintainers shared, unpack into:
ls assets/
# expect: objects/  robots/  scenes/  ...
```

Without `assets/` populated, every task `__init__` raises `FileNotFoundError`
on its URDF/MJCF. There is no auto-download — you must rsync from the
upstream box.

### 2.4 Headless rendering

Genesis defaults to EGL on Linux for the rasterizer. **Do not set
`PYOPENGL_PLATFORM=osmesa`** — PyOpenGL's osmesa binding silently fails to
import, and `cam.render` returns a 32×32 constant frame, producing a useless
H5. Leave the variable unset and let Genesis pick EGL.

If your box has no EGL, fall back to `xvfb-run`:

```bash
xvfb-run -s "-screen 0 1280x1024x24" python scripts/vla_data/collect_vla.py ...
```

### 2.5 Sanity-check imports

```bash
python -c "
from envs.tasks import TASK_MAP
from scripts.vla_client import get_task_instruction
print(sorted(TASK_MAP.keys())[:5], '...')
print(get_task_instruction('pour_water'))
"
```

---

## 3. Collect a single seed (smoke test)

Always run this before kicking off a multi-task batch — it verifies that
the env, assets, and renderer are all wired correctly on this box.

```bash
GENESIS_BACKEND=cpu \
python scripts/vla_data/collect_vla.py \
    --task pour_water \
    --seed 0 \
    --out-root vla_data/pour_water/rasterizer \
    --mode record \
    --renderer rasterizer \
    --cams head_camera,right_wrist
```

Wall-clock: 5–15 min on 16 CPU cores. On success you'll see:

```
vla_data/pour_water/rasterizer/seed_0/
├── steps.h5         # per-step proprio + JPEG bytes
├── meta.json        # task / seed / instruction / success / wall_time
├── verdict.json     # same as meta.json + infra_failure_reason
└── run.log          # full stdout
```

Verify the H5 is non-empty and has ACT target labels:

```bash
python -c "
import h5py
f = h5py.File('vla_data/pour_water/rasterizer/seed_0/steps.h5')
print('steps:', f['t'].shape[0])
print('schema:', f.attrs.get('schema_version', 'unknown'))
print('keys:', sorted(f.keys()))
print('cams:', list(f['images'].keys()))
print('qpos_target_action:', f['qpos_target_action'].shape)
print('gripper_target_action:', f['gripper_target_action'].shape)
if 'qpos_target_action_valid' in f:
    print('valid target actions:', bool(f['qpos_target_action_valid'][:].all()))
print('image bytes (first frame):', len(bytes(f['images/head_camera'][0])))
"
# expect: steps: ~150-400, qpos_target_action + gripper_target_action present,
#         cams: ['head_camera', 'right_wrist'], image bytes > 1000
```

If the image-bytes count is ~0 you hit the EGL/osmesa issue (§2.4); if
`steps` is < 2 the recorder dropped the episode (re-run with a different
seed, the scripted policy fails ~10–30% of seeds depending on the task).
New collections must include `qpos_target_action` and `gripper_target_action`;
old H5 files without those keys need recollection before ACT target-action
training.

### 3.1 Idempotency

`collect_vla.py` is idempotent on the output dir: if `meta.json` already
exists for that seed, the script exits 0 without re-running. To re-collect
a seed, delete its `seed_<N>/` directory first.

### 3.2 Mode = `verdict` (no H5)

If you only want to triage which seeds the scripted policy succeeds on
without paying the H5 write cost, pass `--mode verdict`. It runs
`play_once()` and writes only `verdict.json`. Useful when you want to
pre-filter seeds before committing to a raytracer rerun.

---

## 4. Collect the full dataset (per-task batch)

For each of the 4 Franka tasks, loop over a seed range. The simplest
portable form — one seed at a time, sequential, no scheduler:

```bash
TASKS="pour_water categorize_cooperative
       categorize_interrupt take_from_human_easy"

for task in $TASKS; do
  for seed in $(seq 0 99); do
    GENESIS_BACKEND=cpu \
    python scripts/vla_data/collect_vla.py \
        --task "$task" \
        --seed "$seed" \
        --out-root "vla_data/$task/rasterizer" \
        --mode record \
        --renderer rasterizer \
        --cams head_camera,right_wrist || true
  done
done
```

`|| true` keeps the loop going across the 10–30% of seeds where the
scripted policy fails (the RLDS builder filters those out by default).

Wall-clock estimate: 100 seeds × 4 tasks × ~10 min/seed ÷ 1 process ≈ 67
hours of CPU. **Run this in parallel** if you have the cores.

### 4.1 Parallel — GNU `parallel` (single host, no SLURM)

```bash
TASKS="pour_water categorize_cooperative
       categorize_interrupt take_from_human_easy"

# Generate the (task, seed) work list.
for task in $TASKS; do
  for seed in $(seq 0 99); do
    echo "$task $seed"
  done
done | parallel --colsep ' ' -j 8 \
    'GENESIS_BACKEND=cpu python scripts/vla_data/collect_vla.py \
        --task {1} --seed {2} \
        --out-root vla_data/{1}/rasterizer \
        --mode record --renderer rasterizer \
        --cams head_camera,right_wrist || true'
```

Tune `-j 8` to fit your RAM — each process holds ~5–8 GB of Genesis state
plus image buffers. On a 256-GB host, 24 parallel workers is a safe ceiling.

### 4.2 Parallel — SLURM (cluster)

There is a daemonised scheduler at `scripts/vla_data/scheduler.py` that
submits SLURM jobs and tracks `(task, seed)` state across restarts. It is
**hard-coded for the maintainers' partition / sbatch wrapper** and won't
work as-is on a different site, but the source is short (~300 lines) and a
useful template if you have SLURM. See the docstring at the top of
`scripts/vla_data/scheduler.py` for the state machine, and edit
`scripts/vla_data/sbatch_raster.sh` to match your partition / memory /
time-limit. The portable alternative is §4.1.

### 4.3 How many seeds per task?

The RLDS builder drops episodes with `meta.success == false`. Scripted
success rates vary per task (typical range 70–95%). Aim for **~100
successes per task** (~600 total) — that's what the four fine-tune docs
assume in their wall-clock estimates. To get 100 successes from a 90%
success rate, allocate ~115 seeds per task; for tasks at 70%, ~145 seeds.

The scheduler at `scripts/vla_data/scheduler.py` has a `--keep-trying`
mode that auto-allocates fresh seeds until each task hits a target success
count, capped by `--max-seeds-per-task`. If you're rolling your own loop,
just oversubscribe — collecting 150 seeds and discarding 30 failures is
cheaper than discovering you have 78 successes after the fact and
restarting.

---

## 5. Build the RLDS dataset

After the raw H5 directories are populated, fold them into a single TFDS
dataset:

```bash
TF_CPP_MIN_LOG_LEVEL=2 \
python scripts/build_genesis_hr_bench_rlds.py \
    --raw-root vla_data \
    --renderer rasterizer \
    --data-dir openvla/data \
    --image-size 256
```

Wall-clock: ~5–10 min for ~400 episodes (CPU only — TF resizes the JPEGs
and re-encodes; no GPU is touched).

Output:

```
openvla/data/genesis_hr_bench/1.0.0/
├── features.json                                   # TFDS schema
├── dataset_info.json                               # episode counts, splits
├── dataset_statistics.json                         # 1st/99th-percentile per action dim (OpenVLA normalization)
└── genesis_hr_bench-train.tfrecord-00000-of-00004  # ... -00003
```

### 5.1 Tunable flags

| Flag | Default | Notes |
|---|---|---|
| `--raw-root` | `vla_data` | Root of `<task>/<renderer>/seed_*/` tree |
| `--renderer` | `rasterizer` | Subdir to read; switch to `raytracer` if collecting LuisaRender frames |
| `--data-dir` | `openvla/data` | Output root; final dataset is `<data-dir>/genesis_hr_bench/1.0.0/` |
| `--image-size` | `256` | Resize H×W; OpenVLA-OFT trains at 256 |
| `--include-failures` | False | Default behaviour: drop episodes with `meta.success == false`. Pass to keep them |
| `--primary-cam` | `head_camera` | Renamed to `observation.image` in the output |
| `--wrist-cam` | `right_wrist` | Renamed to `observation.wrist_image` |
| `--skip-stats` | False | Skip computing `dataset_statistics.json` (you almost never want this) |

### 5.2 What's in `dataset_statistics.json`

OpenVLA / OpenVLA-OFT normalize actions to `[-1, +1]` using **1st/99th
percentile** per action dim. The gripper dim (index 6) is masked off so it
stays in `[0, 1]`. The builder writes this file automatically; **do not
edit it by hand** — the receiving group's fine-tuner reads it verbatim.

---

## 6. Verify the dataset

```bash
ls openvla/data/genesis_hr_bench/1.0.0/
# expect: features.json  dataset_info.json  dataset_statistics.json  *.tfrecord-*
```

Count episodes and frames:

```bash
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('genesis_hr_bench', data_dir='openvla/data', split='train')
n_eps = 0
n_frames = 0
tasks = set()
for ep in ds:
    n_eps += 1
    n_frames += int(ep['steps'].cardinality())
    tasks.add(ep['episode_metadata']['task_name'].numpy().decode())
print('episodes:', n_eps)
print('frames:  ', n_frames)
print('tasks:   ', sorted(tasks))
"
```

A healthy dataset has ~400 episodes, ~150–400 frames each (~100 k frames
total), and exactly the 6 Franka tasks listed in §1.

Spot-check a single frame:

```bash
python -c "
import tensorflow_datasets as tfds
import numpy as np
ds = tfds.load('genesis_hr_bench', data_dir='openvla/data', split='train')
for ep in ds.take(1):
    for step in ep['steps'].take(1):
        img = step['observation']['image'].numpy()
        st = step['observation']['state'].numpy()
        ac = step['action'].numpy()
        print('image:', img.shape, img.dtype, 'mean=', img.mean())
        print('state:', st)
        print('action:', ac)
        print('lang: ', step['language_instruction'].numpy().decode())
"
```

A black image (`mean ≈ 0`) means the rasterizer didn't render — see §2.4.

---

## 7. Convert RLDS → LeRobot

OpenVLA-OFT trains directly on the RLDS dataset built in §5. **Skip this
section** for OpenVLA-OFT.

pi0, pi0.5, pi0-FAST, and SmolVLA read from the LeRobot dataset format. The
repo ships a converter that walks the RLDS shards and writes a LeRobot tree on
your filesystem.

```bash
# JAX_PLATFORMS=cpu silences the "no GPU" warnings; conversion is CPU-only.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=2 \
  python scripts/convert_genesis_hr_bench_to_lerobot.py \
    --tfds-data-dir openvla/data \
    --lerobot-naming \
    --repo-id genesis-hr-bench/genesis_hr_bench_lerobot
```

Wall-clock: a few minutes for ~400 episodes. Output lives under
`$HF_LEROBOT_HOME/genesis-hr-bench/genesis_hr_bench_lerobot/` (default
`~/.cache/huggingface/lerobot/...`). To put it on a faster filesystem,
`export HF_LEROBOT_HOME=/path/to/scratch/lerobot` before running.

For a smoke conversion (5 episodes), pass `--max-episodes 5 --overwrite`.

Schema written (per frame):

```
image         (256, 256, 3) uint8     head_camera RGB
wrist_image   (256, 256, 3) uint8     right_wrist RGB
state         (7,)         float32   [x, y, z, roll, pitch, yaw, gripper]
actions       (7,)         float32   [dx, dy, dz, droll, dpitch, dyaw, gripper]
task          str                    language instruction
```

Current pi0 / pi0.5 / pi0-FAST / SmolVLA training uses LeRobot v0.5.2 and a
v3.0 dataset with quantile stats. After this conversion, follow the LeRobot
dataset migration steps in [`vla-finetune-pipeline.md`](vla-finetune-pipeline.md).

You can drop the raw H5 dir at this point — both the RLDS and LeRobot
copies are derived from it and self-contained.

---

## 8. Next: pick a baseline and fine-tune

The dataset is now ready. To finetune any VLA baseline on it, see
[`vla-finetune-pipeline.md`](vla-finetune-pipeline.md) — the unified launcher
covers openvla_oft (RLDS), pi0 / pi05 / pi0_fast / smolvla (LeRobot dataset),
and rdt (HDF5), and builds whichever dataset a baseline needs.

For the fine-tuning pipeline (portable launcher, multi-node, per-baseline
venvs and datasets), see [`vla-finetune-pipeline.md`](vla-finetune-pipeline.md).

### 8.1 (Optional) Sharing the dataset with another box

If you do want to move the dataset to a different fine-tune box (e.g. you
collected on a CPU node and want to train on a GPU node), rsync the
**RLDS** copy:

```bash
rsync -av --progress \
    openvla/data/genesis_hr_bench/1.0.0/ \
    user@gpu.host:/path/to/genesis-hr-bench/openvla/data/genesis_hr_bench/1.0.0/
```

Don't ship the LeRobot copy — its on-disk layout includes
`HF_LEROBOT_HOME`-relative paths and the receiver should rebuild it via
`scripts/convert_genesis_hr_bench_to_lerobot.py` (§6). The raw H5 dir does
not need to be transferred at all.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All `image_bytes` ≈ 0, frames are constant 32×32 | `PYOPENGL_PLATFORM=osmesa` set in the env | Unset it; let Genesis pick EGL (§2.4). On a node with no EGL, prefix with `xvfb-run -s "-screen 0 1280x1024x24"` |
| `FileNotFoundError: assets/objects/...` at task `__init__` | `assets/` not populated | Re-rsync the OneDrive bundle (§2.3) |
| `meta.success == false` for >50% of seeds on a task | Scripted policy regression, or wrong avatar-collision settings | Inspect `verdict.json:evaluate` for the task's failure mode; check `infra_failure_reason` (None = task-logic failure, set = retry-worthy) |
| Per-seed wall-clock > 30 min on rasterizer | CPU oversubscription | Drop `parallel -j` count, or set `OMP_NUM_THREADS=4` per process |
| `AttributeError: 'FieldDescriptor' object has no attribute 'label'` from `build_genesis_hr_bench_rlds.py` | TF 2.16+ / protobuf 5+ broke TFDS metadata | Pin `tensorflow==2.15.0 protobuf<5 tensorflow-datasets==4.9.3` (§2.2) |
| `dataset_statistics.json` missing after build | Used `--skip-stats`, or build crashed mid-stats | Re-run `build_genesis_hr_bench_rlds.py` (idempotent on the TFRecord shards if they're present) |
| Fine-tune raises `KeyError: 'genesis_hr_bench'` on the receiving box | dataset not registered / not built on that box | rebuild the dataset there — see `vla-finetune-pipeline.md` §4 |
| Receiving group's `tfds.load` returns 0 episodes | rsync truncated mid-transfer; some shard is shorter than `dataset_info.json` claims | Re-rsync; verify file sizes match on both sides |

---

## 10. Reference: where each file lives

```
genesis-hr-bench/
├── envs/
│   ├── tasks/                                  # task implementations (play_once, evaluate)
│   └── vla_recorder.py                         # streams JPEGs + proprio into seed_<N>/steps.h5
├── scripts/vla_client.py                       # per-task language instruction (string)
├── config/openvla_collect.yml                  # collect-time config (cameras, renderer)
├── scripts/
│   ├── vla_data/
│   │   ├── collect_vla.py                      # PER-SEED collector (§3, §4)
│   │   ├── scheduler.py                        # SLURM daemon (cluster only — see §4.2)
│   │   ├── sbatch_raster.sh                    # SLURM rasterizer wrapper (cluster only)
│   │   └── sbatch_luisa.sh                     # SLURM raytracer wrapper (cluster only)
│   ├── build_genesis_hr_bench_rlds.py          # RAW H5 -> RLDS (§5)
│   ├── convert_genesis_hr_bench_to_lerobot.py  # RLDS -> LeRobot (run on receiver, NOT here)
│   └── collect_vla.sh                          # legacy wrapper around scripts/collect.py
├── vla_data/<task>/<renderer>/seed_<N>/        # raw H5 + meta.json + verdict.json (large; stays here)
└── openvla/data/genesis_hr_bench/1.0.0/        # OUTPUT RLDS dataset (ships to receiver)
```
