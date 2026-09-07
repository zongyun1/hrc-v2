# Diffusion Policy fine-tuning — portable, no-shell-wrapper walkthrough

Step-by-step instructions for training **Diffusion Policy** (Chi et al. 2023)
on a `genesis_hr_bench` task on **any** machine. Every step is a raw command
— `scripts/finetune_diffusion_policy.sh` is the convenience wrapper, and we
break out exactly what it does so you can run it on a fresh box.

Diffusion Policy is **per-task** (not a foundation model). There is no
pretrained DP checkpoint to fine-tune from — "fine-tuning" here is just
"supervised training from scratch on one task's collected rollouts." Plan
on one DP checkpoint per task you want to evaluate.

This doc covers training only. The dataset itself must already exist at
`vla_data/<task>/<renderer>/seed_*/{steps.h5, meta.json}` on this box. If
it does not, build it now by following
[`collect-data-portable.md`](collect-data-portable.md) — that walks you
through running the Genesis simulator and collecting scripted rollouts.
**Come back here once the `vla_data/<task>/` tree exists.**

All commands assume your cwd is the genesis-hr-bench checkout root:

```bash
cd /path/to/genesis-hr-bench
```

---

## 1. Setup

### 1.1 Hardware

One CUDA GPU with **≥ 16 GB VRAM** — A100, H100, L40S, A40 all work; a
3090 / 4090 / RTX 6000 also works for the default 277 M-parameter UNet.
Pre-Ampere cards (TITAN X, V100) will technically run but are too slow for
the full 1000-epoch schedule.

Wall-clock for 1000 epochs on one L40S with the default `pour_water` zarr
(~150 episodes / ~18 k steps): ~6–10 h depending on batch size and worker
count. The 5-epoch sanity run finishes in ~2 min after the first compile.

CPU-only training works for a smoke test (the full UNet does ~3–5 it/s on
modern CPUs) but is impractical for any real run.

### 1.2 Source

Diffusion Policy upstream lives at `baseline/diffusion_policy/`, vendored
in this repo. Three files inside that tree are genesis_hr_bench-specific
and **already committed alongside the upstream code** — no patch step:

| File | Purpose |
|---|---|
| `diffusion_policy/dataset/genesis_hr_bench_image_dataset.py` | Reads the zarr replay buffer; keeps the full 8-D `[7 qpos, 1 gripper]` state as `agent_pos` instead of slicing to 2-D like the pusht reference. |
| `diffusion_policy/env_runner/genesis_hr_bench_noop_runner.py` | Returns `{}` from `run(policy)`. Genesis-based eval is **out of process** via `scripts/run_vla_eval.sh`, so we skip the in-trainer rollout. |
| `diffusion_policy/config/task/genesis_hr_bench_image.yaml` + `diffusion_policy/config/train_diffusion_unet_image_genesis_hr_bench_workspace.yaml` | The Hydra task + workspace configs that tie those two together. The workspace flips `topk.monitor_key` from `test_mean_score` (which the no-op runner never logs) to `val_loss`. |

If you ever pull a fresh upstream copy of `diffusion_policy/`, copy those
four files back in.

### 1.3 Build the Diffusion Policy environment

DP's dependency stack (`diffusers`, `hydra-core`, `dill`, `robomimic`,
`zarr`, `numba`, …) collides with the OpenVLA / Octo / pi0 stacks — **use
a dedicated env**. Conda is the upstream-recommended path; a plain venv
works too.

#### 1.3a Conda (recommended)

```bash
conda create -n diffusion_policy python=3.10 -y
conda activate diffusion_policy

# Torch + CUDA. Pick the cu wheel that matches your driver — cu121 is the
# version we run on; cu118 / cu124 also work as long as torch and the
# install you do later are consistent.
pip install "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# Diffusion Policy stack — pinned where the upstream repo or our patches
# require a specific version.
pip install \
    "diffusers==0.27.2" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "dill==0.4.1" \
    "robomimic==0.2.0" \
    "huggingface_hub<0.26"        # pin: diffusers 0.27 imports `cached_download` removed in 0.26+

# Upstream-required deps for replay buffer + dataloading + media I/O.
pip install \
    "zarr==2.18.3" \
    "numba" \
    "pandas" \
    "scikit-image" \
    "scikit-video" \
    "imageio" "imageio-ffmpeg" \
    "av" \
    "einops" \
    "shapely" \
    "imagecodecs" \
    "accelerate" \
    "datasets" \
    "h5py" \
    "Pillow" \
    "threadpoolctl" \
    "tqdm" \
    "wandb"

# Server (used by scripts/run_vla_eval.sh once the checkpoint exists).
pip install flask
```

#### 1.3b venv alternative

```bash
python3.10 -m venv /path/to/dp_env
source /path/to/dp_env/bin/activate
# ... then run the same pip install ... commands as 1.3a.
```

The shell wrapper `scripts/finetune_diffusion_policy.sh` defaults to a
hardcoded path under `/scratch4/...`. On any other machine, point it at
your env via the `DP_PY` environment variable:

```bash
export DP_PY=/path/to/dp_env/bin/python
```

(or edit the `ENV_PY=` line near the top of that script). The same env
python is used by `baseline/diffusion_policy_server.py` at eval time.

#### 1.3c Sanity-check the env

```bash
$DP_PY -c "
import sys, os
sys.path.insert(0, 'baseline/diffusion_policy')
import torch, diffusers, hydra, zarr, dill
from diffusion_policy.workspace import train_diffusion_unet_image_workspace
from diffusion_policy.dataset.genesis_hr_bench_image_dataset import GenesisHrBenchImageDataset
from diffusion_policy.env_runner.genesis_hr_bench_noop_runner import GenesisHrBenchNoopRunner
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('diffusers', diffusers.__version__)
print('zarr', zarr.__version__)
print('genesis_hr_bench dataset + noop runner: importable')
"
```

If `cuda available: False` and you have an NVIDIA GPU, you installed the
CPU torch wheel — redo §1.3a with the `--index-url ...` line.

### 1.4 Verify the dataset is in place

```bash
ls vla_data/<task>/rasterizer/ | head      # expect seed_0, seed_1, ...
ls vla_data/<task>/rasterizer/seed_0/      # expect steps.h5, meta.json
```

If empty, follow [`collect-data-portable.md`](collect-data-portable.md)
first. ~150 successful episodes is a reasonable lower bound for a useful
checkpoint; ~500 is comfortable. The build step below skips episodes with
`meta.success != true` by default.

---

## 2. Build the zarr replay buffer

Diffusion Policy ingests data as a pusht-style zarr replay buffer
(`data/{img,state,action}` + `meta/episode_ends`). The conversion is a
single command:

```bash
$DP_PY scripts/build_diffusion_policy_zarr.py \
    --task <task> \
    --renderer rasterizer \
    --image-size 96
```

Output: `runs/diffusion_policy/<task>/<task>.zarr/` with

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `data/img` | `(N, 96, 96, 3)` | `uint8` | head_camera primary stream, resized |
| `data/state` | `(N, 8)` | `float32` | `[7 qpos, 1 gripper]` — `agent_pos` |
| `data/action` | `(N, 8)` | `float32` | `[7 qpos_delta, 1 gripper_action]` |
| `meta/episode_ends` | `(n_episodes,)` | `int64` | cumulative end indices |

**Action / proprio convention** is wired to match
`BaseTask.take_action(action_type="qpos")` at inference time (first 7 dims
are joint deltas added to current qpos; dim 7 is the absolute gripper
target with `1.0 == open`). If you change the action space (e.g.
end-effector instead of qpos, or a 6-DOF Piper instead of 7-DOF Franka),
update both the builder and the task yaml's `shape_meta` — they must agree.

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--image-size` | `96` | Square resize of the primary cam. Match the workspace yaml's `image_shape`. |
| `--primary-cam` | `head_camera` | Switch if your collection used a different camera key. |
| `--include-failures` | off | Off by default — we drop episodes with `meta.success != true`. |
| `--out-dir` | `runs/diffusion_policy/<task>/<task>.zarr` | Override only if you want multiple zarrs per task. |

The script refuses to overwrite an existing zarr — `rm -rf` it first if
you want to rebuild.

---

## 3. Train

### 3.1 Required env vars

```bash
# Same path you set in §1.3b.
export DP_PY=/path/to/dp_env/bin/python

# Disable wandb until you want it. Drop this line for real wandb logging.
export WANDB_MODE=disabled

# Hydra writes log files via a tee inside train.py. Nothing else env-wise.
```

### 3.2 Sanity run (5 epochs, batch 8)

Confirms the data path + GPU + workspace all wire up before committing
to a long run.

```bash
$DP_PY baseline/diffusion_policy/train.py \
    --config-name=train_diffusion_unet_image_genesis_hr_bench_workspace \
    task.dataset.zarr_path=runs/diffusion_policy/<task>/<task>.zarr \
    training.num_epochs=5 \
    dataloader.batch_size=8 \
    val_dataloader.batch_size=8 \
    training.checkpoint_every=5 \
    training.val_every=1 \
    exp_name=sanity
```

Cold start instantiates the 277 M-parameter UNet (~5 s) + first epoch is
~30–60 s of dataloading-warmup, then steady-state ~3–5 it/s on CPU,
30–80 it/s on a modern GPU. Output:

```
runs/diffusion_policy/<task>/<timestamp>_sanity/
├── checkpoints/
│   ├── epoch=0000-val_loss=0.0XXX.ckpt
│   └── latest.ckpt
├── .hydra/{config,hydra,overrides}.yaml
├── logs.json.txt
└── train.log
```

`latest.ckpt` is ~4.6 GB (params + EMA + optimizer + scheduler). The
top-k checkpoint format `epoch=NNNN-val_loss=X.XXXX.ckpt` is keyed off
`val_loss` because our env_runner is a no-op (see §1.2).

### 3.3 Full run (1000 epochs, batch 64)

```bash
$DP_PY baseline/diffusion_policy/train.py \
    --config-name=train_diffusion_unet_image_genesis_hr_bench_workspace \
    task.dataset.zarr_path=runs/diffusion_policy/<task>/<task>.zarr \
    training.num_epochs=1000 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    training.checkpoint_every=50 \
    training.val_every=1 \
    exp_name=full
```

Wall-clock on one L40S with ~18 k transitions: ~6–10 h. Final checkpoint
at `runs/diffusion_policy/<task>/<timestamp>_full/checkpoints/latest.ckpt`.

If you would rather use the shell wrapper for the standard schedule:

```bash
DP_PY=/path/to/dp_env/bin/python \
    scripts/finetune_diffusion_policy.sh --task <task> --full
```

(it auto-builds the zarr first if missing).

### 3.4 Tunable knobs

All are Hydra overrides on the `train.py` command line.

| Knob | Default | Notes |
|---|---|---|
| `training.num_epochs` | 1000 | Lower if val loss plateaus early. |
| `dataloader.batch_size` | 64 | Drop to 32 / 16 on tight VRAM (12–16 GB). |
| `dataloader.num_workers` | 4 | Bump on systems with fast disks; drop to 0 if you hit dataloader hangs. |
| `policy.num_inference_steps` | 100 | Used during val sampling — at *inference* the server overrides this to 16 (DDIM) for latency. |
| `horizon` | 16 | Action chunk window the diffusion model predicts. |
| `n_obs_steps` | 2 | Observation history fed to the model. |
| `n_action_steps` | 8 | How many of the 16 predicted actions get executed before re-planning. |
| `policy.obs_encoder.crop_shape` | `[76, 76]` | Random-crop augmentation; must be `<= image_shape`. |
| `training.use_ema` | true | EMA-averaged params are what `predict_action` uses; turn off only for ablations. |
| `logging.mode` | `online` | Set `WANDB_MODE=disabled` to skip without code edits (§3.1). |

### 3.5 Resume after a crash / preemption

`training.resume: True` is on by default. Re-run the **same command** from
§3.2 / §3.3 (same `exp_name`, same hydra run dir). The workspace's
`load_payload` rehydrates model + EMA + optimizer + LR scheduler from
`checkpoints/latest.ckpt`.

```bash
$DP_PY baseline/diffusion_policy/train.py \
    --config-name=train_diffusion_unet_image_genesis_hr_bench_workspace \
    hydra.run.dir=runs/diffusion_policy/<task>/<timestamp>_full \
    task.dataset.zarr_path=runs/diffusion_policy/<task>/<task>.zarr \
    training.num_epochs=1000 \
    dataloader.batch_size=64 \
    exp_name=full
```

The `hydra.run.dir=...` override is the part that matters — it points
Hydra at the existing run directory so it picks up `checkpoints/latest.ckpt`
instead of starting a fresh run under a new timestamp.

---

## 4. Serve the checkpoint for benchmark eval

DP eval is **out of process**: a Flask server in the DP env, talking to
the benchmark runner over HTTP. The wrapper that orchestrates both is
`scripts/run_vla_eval.sh`:

```bash
DP_PY=/path/to/dp_env/bin/python \
    scripts/run_vla_eval.sh \
        --model diffusion_policy \
        --task <task> \
        --checkpoint runs/diffusion_policy/<task>/<timestamp>_full/checkpoints/latest.ckpt \
        --episodes 10
```

(`run_vla_eval.sh` reads `$DP_PY` via the env-var override added to
`scripts/finetune_diffusion_policy.sh` in §1.3b — point them at the same
env.)

`--checkpoint` is **required** for `diffusion_policy` (no default —
checkpoints are per-task). DP's HTTP server defaults to port 8773 and
emits 8-D `[7 qpos_delta, 1 gripper]` actions, so the eval client uses
`--action-type qpos`. Inference uses DDIM with 16 steps by default; pass
`--num-inference-steps N` to the server (edit `run_vla_eval.sh`'s
`SERVER_ARGS=(...)` for the `diffusion_policy` case to add it) for a
quality/latency trade-off.

If your machine is CPU-only (no GPU), the server's default `--device cuda`
will fail. Manual launch:

```bash
$DP_PY baseline/diffusion_policy_server.py \
    --checkpoint runs/diffusion_policy/<task>/<timestamp>_full/checkpoints/latest.ckpt \
    --device cpu \
    --num-inference-steps 16 \
    --port 8773 \
    > runs/dp_server.log 2>&1 &
```

then run `scripts/eval_vla.py --model diffusion_policy --server-url http://127.0.0.1:8773 ...`
directly against it.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` | hf_hub ≥ 0.26 dropped the symbol that diffusers 0.27 still imports | `pip install "huggingface_hub<0.26"` (§1.3a) |
| `KeyError: 'genesis_hr_bench_image'` at hydra config compose | task yaml not present in the upstream tree | re-add the four files listed in §1.2 |
| `RuntimeError: cannot initialize CUDA` / no CUDA visible | wrong torch wheel for the driver, or pre-Ampere card | reinstall torch with the right `--index-url`; verify with `nvidia-smi` and the §1.3c snippet |
| OOM on 16 GB GPU | batch too large or workers leaking RAM | `dataloader.batch_size=32 dataloader.num_workers=2` |
| `refuse to overwrite existing <task>.zarr` | builder is conservative on rebuild | `rm -rf runs/diffusion_policy/<task>/<task>.zarr` first |
| `monitor_key=test_mean_score not found in run logs` | upstream workspace yaml; ours overrides to `val_loss` | use our `train_diffusion_unet_image_genesis_hr_bench_workspace.yaml` (§1.2) |
| Trainer never writes a `top-k` checkpoint, only `latest.ckpt` | val loss never improved (e.g. only 1 episode in zarr → val split is empty) | collect more rollouts; check `n_episodes` in the build log |
| Server returns 500 / `cuda` error | server defaulted to GPU on a CPU box | launch manually with `--device cpu` (§4) |
| Sanity train hangs at "loading workspace" for > 60 s | dill is unpickling the EMA module against the wrong source path | confirm `baseline/diffusion_policy` is on `sys.path` (the server prepends it; manual python sessions need to do the same) |

---

## 6. Reference: where each file lives

```
genesis-hr-bench/
├── vla_data/<task>/<renderer>/seed_*/{steps.h5, meta.json}   # input (assumed present)
├── baseline/
│   ├── diffusion_policy/                                     # vendored upstream
│   │   ├── train.py                                          # Hydra entrypoint (§3)
│   │   └── diffusion_policy/
│   │       ├── dataset/genesis_hr_bench_image_dataset.py     # genesis-hr-bench specific (§1.2)
│   │       ├── env_runner/genesis_hr_bench_noop_runner.py    # no-op runner (§1.2)
│   │       └── config/
│   │           ├── task/genesis_hr_bench_image.yaml          # task config (§1.2)
│   │           └── train_diffusion_unet_image_genesis_hr_bench_workspace.yaml  # workspace (§1.2)
│   └── diffusion_policy_server.py                            # HTTP server for eval (§4)
├── scripts/
│   ├── build_diffusion_policy_zarr.py                        # h5 → zarr (§2)
│   ├── finetune_diffusion_policy.sh                          # train wrapper (§3)
│   ├── finetune_diffusion_policy.sbatch                      # SLURM wrapper (§3)
│   └── run_vla_eval.sh                                       # eval wrapper (§4)
├── scripts/vla_client.py                                     # client policy (§4)
├── runs/diffusion_policy/<task>/                             # outputs (zarr + checkpoints)
│   ├── <task>.zarr/
│   └── <timestamp>_<exp>/checkpoints/{latest.ckpt, epoch=*.ckpt}
```
