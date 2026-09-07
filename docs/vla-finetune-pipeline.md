# VLA Finetuning Pipeline — Running on a New Cluster

A handoff guide for running the genesis-hr-bench VLA finetuning pipeline on a
server other than the one it was developed on.

The pipeline finetunes 6 VLA baselines through one launcher (`scripts/finetune.sh`)
with multi-node / multi-GPU exposed as variables. **Every cluster-specific path
and SLURM name lives in one gitignored file** — `scripts/env.local.sh` — so
porting to a new cluster is mostly: build the venvs, write that file, build the
datasets.

| Baseline | Framework | Launcher family | Multi-node |
|---|---|---|---|
| `openvla_oft` | PyTorch | torchrun | ✅ |
| `pi0`, `pi05`, `pi0_fast` | lerobot v0.5.2 | accelerate | ✅ |
| `smolvla` | lerobot v0.5.2 | accelerate | ✅ |
| `rdt` | RDT + DeepSpeed | accelerate | ✅ |

(Non-VLA imitation baselines `act` / `vqbet` / `lerobot_diffusion` / `diffusion_policy` / `dp3`
also launch through `finetune.sh` but are single-GPU and out of scope here.)

---

## 0. TL;DR

```bash
git clone --recurse-submodules <repo-url> genesis-hr-bench
cd genesis-hr-bench
cp scripts/env.local.sh.example scripts/env.local.sh   # <-- edit for your cluster
$EDITOR scripts/env.local.sh
# ... build venvs + datasets (sections 2 and 4) ...
scripts/finetune.sh --model smolvla --slurm --gpus-per-node 1 --sanity   # smoke test
scripts/finetune.sh --model pi0 --slurm --nodes 2 --gpus-per-node 4 --full
```

---

## 1. Get the code

```bash
git clone --recurse-submodules <repo-url> genesis-hr-bench
cd genesis-hr-bench
git submodule update --init --recursive       # if you forgot --recurse-submodules
```

5 submodules land under `baseline/`: `lerobot`, `openvla-oft`, `rdt`,
`diffusion_policy`, `3d_diffusion_policy`. The per-model HTTP eval servers are
tracked in `baseline/servers/`.

---

## 2. Set up the environments (venvs)

This is the most involved step. Each baseline family runs in its **own** venv
— their `torch` / `transformers` / `jax` pins conflict, so they cannot share
one. For the 6 VLA baselines you need **4 venvs**:

| venv | Python | Used by |
|---|---|---|
| `MAWM` | 3.10 | dataset converters + the HTTP eval client |
| `lerobot` | 3.12 | `pi0`, `pi05`, `pi0_fast`, `smolvla` |
| `openvla_oft` | 3.10 | `openvla_oft` |
| `rdt` | 3.10 | `rdt` |

All four are built with [`uv`](https://docs.astral.sh/uv/). Put them under one
parent dir and point `VLA_ENV_ROOT` at it (section 3) — `env.sh` derives
`MAWM_PY` / `LEROBOT_PY` / `OFT_PY` / `RDT_PY` from it. Below, `$ENVS` is that
parent dir and commands run from the repo root.

### 2a. `lerobot` venv — pi0 / pi05 / pi0_fast / smolvla

```bash
uv venv --python 3.12 "$ENVS/lerobot"
uv pip install --python "$ENVS/lerobot/bin/python" \
    -e "baseline/lerobot[smolvla,pi,training]" flask
```

The **`pi` extra is required** — it pulls `scipy`, which pi0-FAST's tokenizer
needs. `[smolvla,training]` alone is not enough (pi0_fast then dies with
`ImportError: 'scipy' is required`).

### 2b. `rdt` venv — rdt

```bash
uv venv --python 3.10 "$ENVS/rdt"
RDT_PY="$ENVS/rdt/bin/python"
# torch first — RDT pins the cu121 build
uv pip install --python "$RDT_PY" torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$RDT_PY" -r baseline/rdt/requirements.txt
uv pip install --python "$RDT_PY" 'setuptools<81' 'huggingface_hub<0.26'
```

Two non-obvious pins:
- **`setuptools<81`** — setuptools ≥81 dropped `pkg_resources`, which the RDT
  stack imports (`ModuleNotFoundError: No module named 'pkg_resources'` otherwise).
- **`huggingface_hub<0.26`** — diffusers 0.27 imports `cached_download`, removed
  in newer hf_hub.

RDT also needs the T5-XXL + SigLIP encoders — auto-downloaded from HF on the
first training launch (~20 GB; the first run is slow).

RDT's DeepSpeed also needs a **CUDA toolkit** (`CUDA_HOME`) at import time.
`finetune_rdt.sh` `module load`s `$CUDA_MODULE` (default `cuda/12.1`) when
`CUDA_HOME` isn't already set — set `CUDA_MODULE` (your cluster's module name)
or `CUDA_HOME` directly in `scripts/env.local.sh`.

### 2c. `openvla_oft` venv — openvla_oft

```bash
uv venv --python 3.10 "$ENVS/openvla_oft"
```

Then install the Prismatic stack (torch + TF + flash-attn) into it following
`baseline/openvla-oft/SETUP.md`. **flash-attn must be installed last**, with
`--no-build-isolation` (its build backend needs torch already present).

### 2d. `MAWM` venv — dataset tooling + eval client

The Genesis benchmark env. Build it per the repo's top-level `README.md` and
`requirements.txt`.
For *this* pipeline it only runs the dataset converters and the HTTP eval
client, so it additionally needs `requests`, `h5py`, and `tensorflow` +
`tensorflow_datasets` (the RLDS / LeRobot converters use TFDS).

### Verify the envs

```bash
"$ENVS/lerobot/bin/python" -c "import lerobot, scipy, accelerate; print('lerobot venv OK')"
"$ENVS/rdt/bin/python"     -c "import torch, deepspeed, pkg_resources; print('rdt venv OK')"
"$ENVS/openvla_oft/bin/python" -c "import torch; print('oft venv OK')"
```

**Shortcut:** if the new server shares a filesystem with the original (or you
can `rsync` the venv trees to the **identical absolute path**), copy the venvs
and skip the rebuild — point `scripts/env.local.sh` at them. venvs are not
relocatable to a different path (shebangs/configs hardcode it).

---

## 3. Configure for your cluster — `scripts/env.local.sh`

This is **the** portability file. It is gitignored, so each person/cluster keeps
their own. `scripts/env.sh` reads it and falls back to defaults for anything
left unset.

```bash
cp scripts/env.local.sh.example scripts/env.local.sh
$EDITOR scripts/env.local.sh
```

Set the venv location — either the parent dir (paths are derived from it):

```bash
VLA_ENV_ROOT=/path/to/envs        # holds MAWM/ lerobot/ openvla_oft/ rdt/
```

…or each interpreter individually if they are not siblings:

```bash
MAWM_PY=/path/to/MAWM/bin/python
LEROBOT_PY=/path/to/lerobot/bin/python
OFT_PY=/path/to/openvla_oft/bin/python
RDT_PY=/path/to/rdt/bin/python
```

And the SLURM knobs for **your** cluster — these become the `sbatch` flags:

```bash
FT_PARTITION=gpu                  # sbatch -p
FT_CONSTRAINT="a100|h100"         # sbatch --constraint  (empty -> flag omitted)
FT_ACCOUNT=my_account             # sbatch -A            (empty -> flag omitted)
FT_GRES=gpu                       # gres name; sbatch --gres=${FT_GRES}:<gpus-per-node>
```

`scripts/finetune.sbatch` carries **no** static `#SBATCH` cluster names — they
all come from these variables, so nothing else needs editing to port clusters.

---

## 4. Build the datasets

Each baseline reads a different dataset format, all derived from the collected
`vla_data/` episodes. **Either** transfer the prebuilt datasets from the
original server **or** rebuild from raw `vla_data/` (44 GB, must be present):

| Baseline(s) | Dataset | Location | Build command |
|---|---|---|---|
| pi0 / pi05 / pi0_fast / smolvla | LeRobot v3.0 dataset | `$HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`) | see "LeRobot dataset" below — 3 steps |
| openvla_oft | RLDS / TFDS | `openvla/data/genesis_hr_bench/1.0.0/` | `scripts/build_genesis_hr_bench_rlds.py --raw-root vla_data` |
| rdt | RDT HDF5 | `runs/rdt/data/genesis_hr_bench/` | `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py --out-dir runs/rdt/data/genesis_hr_bench` |

**LeRobot dataset — must be v3.0 with quantile stats.** The converter emits a
v2.1 dataset, but lerobot v0.5.2 (the `lerobot` venv) requires **v3.0**, and
pi0.5 additionally needs quantile (`q01`/`q99`) stats. Three steps:

```bash
# 1. build the v2.1 dataset (MAWM venv — needs tensorflow + tensorflow_datasets)
$MAWM_PY scripts/convert_genesis_hr_bench_to_lerobot.py \
    --lerobot-naming --repo-id genesis-hr-bench/genesis_hr_bench_lerobot
DS=~/.cache/huggingface/lerobot/genesis-hr-bench
# 2. migrate v2.1 -> v3.0 (lerobot venv)
cp -r "$DS/genesis_hr_bench_lerobot" "$DS/genesis_hr_bench_lerobot_v30"
$LEROBOT_PY -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    --root "$DS/genesis_hr_bench_lerobot_v30" --push-to-hub=false
rm -rf "$DS/genesis_hr_bench_lerobot_v30_old"
# 3. add quantile stats (lerobot venv; the trailing 403 hub-push error is harmless)
$LEROBOT_PY -m lerobot.scripts.augment_dataset_quantile_stats \
    --repo-id genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    --root "$DS/genesis_hr_bench_lerobot_v30" || true
```

`finetune_lerobot_v2.sh` defaults `DATASET_REPO_ID` to
`genesis-hr-bench/genesis_hr_bench_lerobot_v30`.

Notes:
- Run the LeRobot + RLDS converters from the `MAWM` venv; the RLDS dataset
  must exist before step 1 (the LeRobot converter reads the RLDS source).
- The RDT converter reads raw `vla_data/` directly, from the `MAWM` venv.
- For rdt, point the trainer at the dataset with `RDT_HDF5_DIR=<path>` (the
  default already matches the table above).

---

## 5. Run finetuning — `scripts/finetune.sh`

One entry point, two modes.

**Local** (run in-process on the current box):

```bash
scripts/finetune.sh --model smolvla --sanity                 # 1 GPU
scripts/finetune.sh --model pi0 --gpus-per-node 2 --full      # 2 GPUs, this node
```

**SLURM** (`--slurm` builds the `sbatch` command and submits `finetune.sbatch`):

```bash
scripts/finetune.sh --model <name> --slurm \
    --nodes N --gpus-per-node M \
    [--time HH:MM:SS] [--mem 80G] [--cpus 8] \
    [--partition P] [--constraint C] [--account A] \
    [--sanity | --full] [--exp-name NAME] \
    [-- <extra framework args>]
```

- `--nodes` × `--gpus-per-node` is the **multi-node / multi-GPU** knob. e.g.
  `--nodes 2 --gpus-per-node 4` = 8 GPUs across 2 nodes.
- `--sanity` = short smoke run (200 steps; 2000 for openvla_oft). `--full` = real run.
- Resource flags default per-model; `--partition`/`--constraint`/`--account`
  default to the `FT_*` values from `env.local.sh`.
- Anything after `--` is forwarded verbatim to the underlying trainer.

Examples:

```bash
# smoke-test every VLA baseline (1 GPU each)
for m in openvla_oft pi0 pi05 pi0_fast smolvla rdt; do
  scripts/finetune.sh --model $m --slurm --gpus-per-node 1 --sanity --time 2:00:00
done

# a real multi-node run
scripts/finetune.sh --model pi05 --slurm --nodes 4 --gpus-per-node 8 --full --exp-name pi05_v1
```

How multi-node works: `finetune.sbatch` derives `MASTER_ADDR` from
`SLURM_JOB_NODELIST`, exports `NNODES`/`GPUS_PER_NODE`, and runs the per-baseline
driver under `srun` (one task per node). `scripts/_dist.sh` turns that into the
right `torchrun --rdzv-endpoint …` / `accelerate launch --num_machines …` prefix.

**Non-SLURM multi-node:** run `finetune.sh` (no `--slurm`) on each node with
`NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK` set in the
environment.

---

## 6. Outputs

```
runs/<model>/finetune/<exp-name>/        training checkpoints
runs/<model>/finetune/slurm_<jobid>.out  stdout
runs/<model>/finetune/slurm_<jobid>.err  stderr
```

`<exp-name>` defaults to `<mode>_<slurm_job_id>`. Because the job id is stable
across SLURM requeues, a preempted job resumes from its last checkpoint on
restart (lerobot `--resume`, RDT `--resume_from_checkpoint`).

Monitor: `squeue -u $USER` ; `tail -f runs/<model>/finetune/slurm_<jobid>.out`.

---

## 7. Evaluation (optional)

Each baseline evaluates as an out-of-process HTTP server spawned by
`scripts/run_vla_eval.sh`:

```bash
scripts/run_vla_eval.sh --model pi0 \
    --checkpoint runs/pi0/finetune/<exp>/checkpoints/last/pretrained_model \
    --task pour_water --episodes 10
# on SLURM:
sbatch scripts/eval.sbatch <model> <checkpoint> -- --task pour_water
```

The server runs in the baseline's own venv; the benchmark runner talks to it
over HTTP from the `MAWM` venv.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot find … python at …` | venv path wrong — fix `VLA_ENV_ROOT` / `*_PY` in `env.local.sh` |
| `sbatch: invalid partition` / constraint errors | fix `FT_PARTITION` / `FT_CONSTRAINT` / `FT_GRES` for your cluster |
| `dataset missing` / `no *.hdf5 under RDT_HDF5_DIR` | build the dataset (section 4) |
| lerobot `--dataset.repo_id` not found | the LeRobot dataset isn't materialized under `$HF_LEROBOT_HOME` |
| job preempted | it auto-resumes on requeue — no action needed |
| multi-node job hangs at startup | check the nodes can reach `MASTER_ADDR:MASTER_PORT`; check NCCL / fabric env vars for your cluster |

---

## Quick reference

- Launcher: `scripts/finetune.sh` — local + `--slurm`.
- Portability config: `scripts/env.local.sh` (copy from `.example`).
- Distributed contract: `NNODES` × `GPUS_PER_NODE` (`scripts/_dist.sh`).
- Generic SLURM wrapper: `scripts/finetune.sbatch` (no hardcoded cluster names).
- Per-baseline drivers: `finetune_openvla_oft.sh`, `finetune_lerobot_v2.sh`
  (pi0/pi05/pi0_fast/smolvla), `finetune_rdt.sh`.
- Repo-wide setup: `README.md`; VLA details are in this document.
