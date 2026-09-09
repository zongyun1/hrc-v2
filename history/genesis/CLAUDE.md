# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genesis HR Bench is a humanoid robot manipulation benchmark built on the Genesis physics simulator. It provides 15 manipulation tasks, multiple robot embodiments (Piper, Franka Panda), pluggable motion planning backends (mplib, cuRobo, Genesis native), and avatar support for human-robot interaction tasks.

## Setup

Requires a custom Genesis fork:
```bash
git clone git@github.com:UMass-Embodied-AGI/Genesis.git
cd Genesis && git checkout vico
pip install -e .[dev]
pip install setuptools==65.7.0 trimesh==4.6.3
pip install -r requirements.txt
```
Assets must be downloaded separately from OneDrive and placed in `assets/`.

## Common Commands

```bash
# Collect data for a task (scripted rollout)
python scripts/collect.py --task pour_water --episodes 1

# Evaluate a policy on a task
python scripts/eval.py --task <task_name> --episodes 10
```

Task names are registered in `envs/tasks/__init__.py` TASK_MAP (e.g., `pour_water`, `pick_and_place`, `handover_block`, `open_jar_lid`, `open_cabinet_door`, `tool_delivery`).

There is no automated test suite. Debug/validation scripts exist as `test_setup_demo.py`, `test_fk.py`, `test_fk2.py` in the repo root.

## Architecture

### Task System
- **`envs/base_task.py`** — `BaseTask` is the central class. Lifecycle: `__init__(config)` → `reset(seed)` → `play_once()` (scripted) or `take_action(action)` (policy). Each task implements `load_actors()`, `play_once()`, and `check_success()`.
- **`envs/tasks/`** — 15 concrete task subclasses. Tasks with `use_avatar = True` involve human-robot interaction.
- **`scripts/collect.py`** — Loops episodes calling `reset()` → `play_once()` → `save_trajectory()`.
- **`scripts/eval.py`** — Loops episodes calling `reset()` → `take_action(policy_fn(obs))` per step.

### Robot Abstraction (`envs/robot/`)
- **`base.py`** — `Robot` base class and `Arm` class (wraps entity + joints + gripper + planner). Arm handles joint PD control, EE tracking, gripper open/close, and TCP offset transforms.
- **`franka_robot.py`** — Franka Panda (7-DOF, MJCF). **`piper_robot.py`** — Piper dual-arm (URDF).

### Motion Planning (`envs/planning/`)
- **`base.py`** — `Planner` ABC with `plan_path()`, `solve_ik()`, `plan_batch()`. Returns `PlanResult`.
- Backends: `mplib_planner.py` (default, OMPL-based), `curobo_planner.py`, `genesis_planner.py`, `genesis_ik.py`.

### Grasp System
- **`envs/grasp.py`** — `GraspPose` class. Loads grasp poses from YAML, transforms object-local → world frame → IK target via TCP offset.

### Avatar System (`envs/avatar/`)
- **`robot.py`** — `AvatarRobot` with collision box + SMPLX skin mesh.
- **`controller.py`** — Drives avatar via motion modules (walk, turn, play animation, replay).

### Core Utilities
- **`envs/utils.py`** — `Pose` class (position + quaternion w,x,y,z) with matrix conversion, composition (`__mul__`), and inverse. Also `to_numpy()`, `TABLE_HEIGHT = 0.74`, path constants (`ROOT_PATH`, `ASSETS_PATH`).
- **`envs/camera.py`** — Multi-camera management (static + wrist-mounted), RGB/depth rendering.
- **`config/default.yml`** — Default configuration (viewer, camera, avatar, collect/eval settings).

## Key Conventions

- Quaternions use `(w, x, y, z)` ordering throughout (`envs/utils.py` Pose class).
- Grasp transforms chain as: `T_world_tcp = T_world_object @ T_object_grasp`, then `T_world_link = T_world_tcp @ inv(T_tcp_offset)`.
- Robot configs are YAML files referenced by task classes (e.g., `piper/config.yml` under assets).
- Genesis scene is built inside `reset()`, not `__init__()`. Each `reset()` call rebuilds the scene from scratch.

## VLA Baseline

**Architecture (as of 2026-04-22):** every VLA baseline is an out-of-process HTTP server. The benchmark runner (`scripts/eval_vla.py`, MAWM env) talks to the model server via a thin client policy. This keeps Genesis out of model-specific dep hells and keeps the model weights warm across episodes.

```
  MAWM env                                          dedicated env (per model)
  ┌─────────────────────────────┐                   ┌──────────────────────────────┐
  │ scripts/eval_vla.py         │                   │ baseline/servers/            │
  │   └─ <Model>Policy ──HTTP──→│── POST /predict ─→│      <model>_server.py       │
  │       (RemoteVLAPolicy)     │←── action ────────│   Flask                      │
  └─────────────────────────────┘                   └──────────────────────────────┘
```

Current baselines:

| Model | Server (`baseline/servers/`) | Env | Port | action |
|---|---|---|---|---|
| openvla_oft | openvla_oft_server.py | yz/env/openvla_oft | 8769 | ee |
| pi0 / pi05 / pi0_fast | lerobot_server.py --policy-type {pi0,pi05,pi0_fast} | yz/env/lerobot | 8767/8768/8779 | ee |
| smolvla | lerobot_server.py --policy-type smolvla | yz/env/lerobot | 8778 | ee |
| act / vqbet / lerobot_diffusion | lerobot_server.py --policy-type {act,vqbet,diffusion} | yz/env/pi0 | 8775/8776/8777 | ee |
| rdt | rdt_server.py | yz/env/rdt | 8772 | qpos |
| diffusion_policy / dp3 | {diffusion_policy,dp3}_server.py | yz/env/{diffusion_policy,dp3} | 8773/8774 | qpos |

`baseline/` layout: `baseline/servers/` holds the 6 `*_server.py`; the baseline
source repos are git submodules at `baseline/{lerobot,openvla-oft,rdt,diffusion_policy,3d_diffusion_policy}`.

Files:
- `scripts/vla_client.py` — benchmark-side VLA HTTP clients, task instruction lookup, EE-delta composition, and DP3 point-cloud helpers.
- `baseline/servers/` — the per-model HTTP servers. `lerobot_server.py` is generic: `--policy-type {act,diffusion,vqbet,smolvla,pi0,pi05,pi0_fast}` → the matching lerobot policy class (tries v0.1.0 `lerobot.common.policies.*` then v0.5.2 `lerobot.policies.*`). `act_server.py` is a back-compat shim. `openvla_oft_server.py`, `rdt_server.py`, `diffusion_policy_server.py`, `dp3_server.py` each load one model.
- `config/vla.yml` — `server_url`, `server_timeout`, `wrist_camera`, `secondary_camera`. Each client class declares its own `IMG_WRIST` / `IMG_PRIMARY`.

Protocol: `POST /predict` body `{instruction, images{image_primary[,image_wrist]}, proprio?, reset?}`; reply `{action: list[float]}`. Images are base64-PNG uint8. `proprio` is consumed by the lerobot + rdt servers. `reset` flushes stateful servers' per-episode caches.

`scripts/eval_vla.py --server-url http://127.0.0.1:8765` passes the URL down to the client policy. The `--checkpoint` arg is now optional (kept for backwards-compat with local policies — remote policies ignore it; the server is what holds the checkpoint).

The `template` policy + `scripts/eval_vla.py --task pick_and_place --model template --action-type qpos` path is verified end-to-end and needs no server.

### Envs (per model server)

Per-baseline venvs live under `yz/env/<name>/`. Override paths on a new
cluster via `scripts/env.local.sh` (see "Unified VLA finetune launcher").

- **lerobot venv** → `yz/env/lerobot`, Python 3.12 (uv). `uv pip install -e baseline/lerobot[smolvla,training] flask` — lerobot v0.5.2. Trains **and** serves pi0 / pi05 / pi0_fast / smolvla. Module layout `lerobot.policies.*`; train entry point `lerobot.scripts.lerobot_train`.
- **pi0 venv** → `yz/env/pi0`, Python 3.11 (uv). Ships lerobot v0.1.0 — used for **act / vqbet / lerobot_diffusion only** (training + serving). Despite the name it no longer runs pi0; pi0 moved to the lerobot venv.
- **openvla_oft** → `yz/env/openvla_oft`, Python 3.10. Prismatic stack (torch + TF + flash-attn). `HF_HUB_CACHE=$PWD/checkpoints`; Ampere+.
- **rdt** → `yz/env/rdt`, Python 3.10. RDT-1B; needs T5-XXL + SigLIP encoders, plus `deepspeed` + `accelerate` for training. Pin `huggingface_hub<0.26`.
- **diffusion_policy / dp3** → `yz/env/{diffusion_policy,dp3}`.
- **MAWM env** (`yz/env/MAWM`) — benchmark runner; just `requests` + genesis deps, the eval client is HTTP-only.

### OpenVLA-OFT and RDT-1B detail

- **OpenVLA-OFT** (`yz/env/openvla_oft`) — 7B Prismatic backbone + L1-regression MLP action head + proprio projector + 8-step action chunk. Server uses `experiments.robot.openvla_utils.*`. Default checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial`. Output 7-D bridge EE-delta; client `delta_to_absolute_ee`. Port 8769, `--action-type ee`. Finetune: `finetune_openvla_oft.sh` (torchrun, multi-node).
- **RDT-1B** (`yz/env/rdt`) — 1B robotics diffusion transformer. Server uses the ManiSkill single-arm loader (`rdt/scripts/maniskill_model.py`, `MANISKILL_INDICES` → unified 128-D vector). Needs T5-XXL + SigLIP-so400m encoders at load time (~20 GB download). Output 8-D `[7 joint, 1 gripper]` native; client passes through. Port 8772, `--action-type qpos`. Finetune: `finetune_rdt.sh` (accelerate + DeepSpeed ZeRO-2) on the joint-space HDF5 dataset built by `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`.

### LeRobot baselines: ACT, LeRobot-Diffusion, VQ-BeT, SmolVLA

All four are HuggingFace's `lerobot` repo's policies, served by a single generic server `baseline/lerobot_server.py` rather than one server module per baseline. Source submodule at `baseline/lerobot/` (v0.5.2). The dispatch tries both v0.1.0 (`lerobot.common.policies.<x>`) and v0.5.2 (`lerobot.policies.<x>`) import paths so the same server runs in either venv.

Two venvs back the four baselines:
- **pi0 venv** (`yz/env/pi0`, Python 3.11) ships `lerobot==0.1.0`. Used for ACT / LeRobot-Diffusion / VQ-BeT. No extra install.
- **lerobot venv** (`yz/env/lerobot`, Python 3.12) installed from the submodule via `uv pip install -e baseline/lerobot[smolvla,training] flask`. Used for SmolVLA only (the v0.1.0 in pi0 venv doesn't ship SmolVLA, and SmolVLA requires Python ≥ 3.12).

LeRobot-family clients in `scripts/vla_client.py` ship `proprio + images + instruction` per call, set `reset: true` on the first call after an episode reset so the server clears its action-chunk cache, and compose 7-D bridge EE-delta onto current proprio when `action_type == "ee"`.

Training: all four go through the unified driver. Under the hood:
- `scripts/finetune_lerobot.sh --policy-type {act,diffusion,vqbet}` is the generic driver wrapping `python -m lerobot.scripts.train`. `--sanity` = 200 steps batch=8, `--full` = 100k steps batch=32. Resume-on-requeue via stable `$SLURM_JOB_ID` exp name + lerobot's `--resume`. Override dataset via `DATASET_REPO_ID` env var (VQ-BeT uses this to pick its single-cam variant). Output → `runs/<baseline>/finetune/<exp>/` where `<baseline>` maps lerobot's `--policy-type diffusion` back to the registry name `lerobot_diffusion`.
- `scripts/finetune_{act,vqbet,lerobot_diffusion}.sh` are 1-line shims; `scripts/finetune_smolvla.sh` is separate because it needs the lerobot venv (v0.5.2 entry point `lerobot.scripts.lerobot_train`) instead of the pi0 venv's v0.1.0.

Per-baseline notes:

- **ACT** (Action Chunking Transformer) — Zhao et al. 2023, ~50M params. Single-cam + dual-cam both work. Action: 7-D bridge EE-delta with `chunk_size=100`. Port 8775.
- **LeRobot Diffusion** — LeRobot's port of Chi et al.'s Diffusion Policy, ~250M params + DDPM head, `n_action_steps=8`. Different from Chi et al.'s standalone repo at `baseline/diffusion_policy/`, so registry key is `lerobot_diffusion`. Port 8777.
- **VQ-BeT** — Lee et al. 2024, ~~`validate_features()` requires *exactly one* image input.~~ Our 2-cam `genesis_hr_bench_lerobot` dataset breaks the check, and lerobot's `factory.make_policy` rebuilds `cfg.input_features` from dataset metadata so CLI overrides can't dodge it. `scripts/finetune_vqbet.sh` defaults to `DATASET_REPO_ID=genesis-hr-bench/genesis_hr_bench_lerobot_singlecam` and refuses to run until that dataset exists (build via `scripts/convert_genesis_hr_bench_to_lerobot.py --single-image --lerobot-naming`). Port 8776.
- **SmolVLA** — HF's small (~450M) VLM-backed VLA. Always-on language conditioning (the server forces `obs["task"]` for `policy_type=smolvla` regardless of `cfg.use_language_conditioning`). Needs the dedicated lerobot venv. Port 8778.

### Unified train + eval drivers

Every baseline launches the same way. Two dispatchers mirror each other:

| Stage | Driver | Per-baseline knowledge |
|---|---|---|
| Finetune | `scripts/finetune.sh --model <name> [--slurm] [--sanity\|--full] [--task TASK] [--exp-name NAME] -- <passthrough>` | `case` block delegates to `scripts/finetune_<name>.sh` (each knows its framework: torchrun, accelerate, hydra workspaces, LeRobot's `lerobot_train`, RDT's `main.py`, etc.) |
| Eval   | `scripts/run_vla_eval.sh --model <name> [--checkpoint PATH] --task TASK [more args] -- <eval_vla.py passthrough>` | `case` block spawns the matching `baseline/<name>_server.py` in its env, waits for `/health`, runs `scripts/eval_vla.py`, kills the server on exit |

Trainable/evaluable today: `openvla_oft | pi0 | pi05 | pi0_fast | smolvla | rdt | diffusion_policy | dp3 | act | vqbet | lerobot_diffusion`. `template` is the in-process zero-action smoke policy.

**VLA finetuning is portable + multi-node** (see "Unified VLA finetune launcher" below): `openvla_oft | pi0 | pi05 | pi0_fast | smolvla | rdt` go through `finetune.sh` with a uniform `--nodes N --gpus-per-node M` contract; `pi0/pi05/pi0_fast` train via lerobot v0.5.2 (`lerobot_train`, accelerate).

```bash
# Finetune locally (sanity = 200-step smoke, full = real run)
scripts/finetune.sh --model smolvla            --sanity
scripts/finetune.sh --model pi0 --gpus-per-node 2 --full --exp-name my_run
scripts/finetune.sh --model diffusion_policy   --task pour_water --full
scripts/finetune.sh --model dp3 --task pour_water --full -- training.num_epochs=2000

# Finetune on SLURM — multi-node / multi-GPU as variables (VLA models only)
scripts/finetune.sh --model openvla_oft --slurm --nodes 2 --gpus-per-node 4 --full
scripts/finetune.sh --model rdt --slurm --gpus-per-node 4 --full

# Eval (pretrained: omit checkpoint; finetuned/per-task: include it)
scripts/run_vla_eval.sh --model openvla_oft --task pour_water
scripts/run_vla_eval.sh --model pi0 --checkpoint runs/pi0/finetune/<exp>/checkpoints/last/pretrained_model --task pour_water
scripts/run_vla_eval.sh --model act --checkpoint runs/act/finetune/<exp>/checkpoints/last/pretrained_model

# SLURM submit
sbatch scripts/eval.sbatch <model> [<checkpoint>] -- --task TASK     # generic eval sbatch
```

### Unified VLA finetune launcher

`finetune.sh` is portable to any SLURM cluster and exposes multi-node/multi-GPU
as variables. Key pieces:

- **`scripts/env.sh`** — single source of truth for venv pythons (`MAWM_PY`,
  `OFT_PY`, `LEROBOT_PY`, `RDT_PY`, derived from `VLA_ENV_ROOT`) and SLURM knobs
  (`FT_PARTITION`/`FT_CONSTRAINT`/`FT_ACCOUNT`/`FT_GRES`). Override everything in
  gitignored `scripts/env.local.sh` (see `env.local.sh.example`) — no hardcoded
  cluster paths anywhere.
- **`scripts/_dist.sh`** — `dist_setup` turns the `NNODES` × `GPUS_PER_NODE`
  contract into a `torchrun` or `accelerate launch` prefix (reads SLURM
  rendezvous env).
- **`scripts/finetune.sh --slurm`** — builds the `sbatch` command (per-model
  mem/cpus/time defaults; partition/constraint/account/gres overridable) and
  submits the generic **`scripts/finetune.sbatch`** (no static `#SBATCH` cluster
  names; `srun`-per-node for multi-node).
- Drivers: `finetune_openvla_oft.sh` (torchrun), `finetune_lerobot_v2.sh`
  (accelerate — `pi0|pi05|pi0_fast|smolvla` via `lerobot_train`),
  `finetune_rdt.sh` (accelerate + DeepSpeed ZeRO-2). `finetune_smolvla.sh` is a
  shim into `finetune_lerobot_v2.sh`.
- **RDT data**: `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py` builds the
  joint-space HDF5 dataset (raw `vla_data/*/steps.h5` → RDT schema);
  `baseline/rdt/data/hdf5_vla_dataset.py` is adapted for genesis-hr-bench
  (Franka 8-D); set `RDT_HDF5_DIR` to point RDT at it.

Legacy per-model training paths were retired; the per-VLA `.sbatch` files were
replaced by the generic `finetune.sbatch`.

### Layout under `runs/`

Each trainable baseline has the same shape:

```
runs/<baseline>/
├── finetune/                          ← training outputs (new layout, all baselines)
│   └── <exp_name>/                    ← or <task>/<exp_name>/ for DP / DP3 (per-task baselines)
│       └── checkpoints/
└── eval/                              ← eval outputs
    ├── job_<jobid>/                  ← per-run video + results.json (eval.sbatch convention)
    └── server_logs/                  ← persistent server logs (see SERVER_LOG_DIR below)
        └── <model>_server.<jobid>.log
```

Symlinks at old run paths may exist to keep stale references resolving, but new
training writes under `runs/<baseline>/finetune/`.

### Persistent server logs

`scripts/run_vla_eval.sh` defaults the server log to `mktemp` (lost when the compute-node `/tmp` is wiped at job end — so silent server hangs become un-diagnosable post-hoc). Set `SERVER_LOG_DIR=<path>` to redirect:

```bash
export SERVER_LOG_DIR=runs/<baseline>/eval/server_logs
scripts/run_vla_eval.sh --model openvla_oft ...
# server log lands at runs/<baseline>/eval/server_logs/openvla_oft_server.${SLURM_JOB_ID}.log
```

`scripts/eval.sbatch` sets this automatically.

### Smoke-test status

- Current smoke status is tracked in the baseline-specific docs and run
  outputs under `runs/<baseline>/eval/`. The policy registry in
  `scripts/vla_client.py` is the source of truth for supported model keys.

### Planner override

`scripts/eval_vla.py` sets `config["planner_override"] = "genesis_ik"` on every run. `BaseTask._load_robot` honors this config field by:
- forwarding it to `FrankaRobot(planner_type=...)` (Franka takes its planner through `__init__` kwargs), and
- patching `self.robot.config["planner"]` before `add_to_scene` for robots whose `Arm` reads the planner name from the embodiment YAML dict (Piper, Stretch, …).

Why: the benchmark's PD loop is driven at ~20 Hz by single-step EE targets from the VLA. cuRobo and mplib both do global IK — on every step they can jump to a joint branch far from current qpos, and the PD controller then tries to close that gap in 50 ms, producing runaway. `genesis_ik` uses damped least-squares seeded from current qpos, so its solutions stay local.

`scripts/collect.py` does **not** set `planner_override`. Data collection keeps using whatever each embodiment YAML declares (curobo for Piper, etc.). Verified: `grep planner_override scripts/collect.py` is empty.

Generic `Arm.init_planner` (`envs/robot/base.py`) grew a `genesis_ik` branch next to the existing `mplib`/`curobo` ones so non-Franka robots can use it.

### Gripper in `ee` mode

`BaseTask.take_action(action_type="ee")` expects 8-D per arm = `[xyz, qw, qx, qy, qz, gripper]`. The EE-delta helpers in `scripts/vla_client.py` append `1 - bridge_gripper` (polarity flip: OpenVLA/Bridge/Octo: 1=closed; benchmark `set_gripper`: 1=open). Dual-arm ee is 16-D (`[left_8, right_8]`).

### Open questions / TODOs

- **pi0 action semantics**: `Pi0Policy.predict_raw` returns the server's raw action pass-through. `BaseTask.take_action(action_type="qpos")` treats the first 7 dims as **joint deltas** and adds them to current qpos. If pi05_droid's action head outputs absolute joint positions instead, this produces drift; client-side delta composition (or server-side delta/abs toggle) will be needed. Verify on first real smoke test.
- **pi0 dual-arm / ALOHA**: `Pi0Policy` raises on `--dual-arm`. pi05_aloha emits 14-D bimanual actions; wiring it needs a separate client branch and a new `build_obs` mapping in `pi0_server.py`.
- **Action chunking**: only the first step of some model action chunks is used. Enabling chunk replay would cut inference cost but requires either ee-deltas re-composed every step against the current proprio, or switching to qpos-delta mode.
