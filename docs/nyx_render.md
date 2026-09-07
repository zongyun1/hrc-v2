# Running Tasks with NYX Render

NYX is available as a user-selectable renderer for task collection and
evaluation. Use it when you want NYX sensor output without changing task logic,
robot motion, or grasp selection.

## TL;DR

1. Use a Python environment with `genesis-world`, `gs-nyx`, and
   `gs-nyx-plugin` installed.
2. Run on a GPU node and set `GENESIS_BACKEND=gpu`.
3. Select NYX with `--renderer nyx` or `renderer: nyx` in the task config.
4. Keep the same task config/grasp settings you would use for the regular
   render unless you intentionally want to test a different behavior.

Example:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export GENESIS_BACKEND=gpu

python scripts/collect.py \
  --task pour_water \
  --config config/latest_genesis/pour_water_nyx.yml \
  --renderer nyx \
  --episodes 1 \
  --start-seed 0 \
  --save-dir outputs/nyx_pour_water
```

## How it is wired

The task-facing renderer switch is `renderer: nyx`. Internally, NYX is a camera
sensor backend rather than a `gs.renderers` scene renderer. The scene is still
created with `gs.renderers.Rasterizer()`, and `NyxCameraManager` creates cameras
with:

```python
scene.add_sensor(NyxCameraOptions(...))
```

This preserves the existing task camera API:

- `render_all()`
- `get_rgb(name)`
- `get_depth(name)`
- `get_all_rgb()`
- `get_all_depth()`
- `get_intrinsic(name)`
- `get_extrinsic(name)`

The integration lives in:

- `envs/base_task.py`: selects GPU backend for `renderer: nyx` and installs
  `NyxCameraManager`.
- `envs/camera.py`: implements the NYX sensor-backed camera manager.
- `scripts/collect.py`, `scripts/eval.py`, and
  `scripts/vla_data/collect_vla.py`: accept `--renderer nyx`.

## Config

Minimal config:

```yaml
renderer: nyx
camera_config:
  default:
    w: 640
    h: 480
    fovy: 60
nyx:
  spp: 1
  denoise: false
  open_window: false
```

`scripts/collect.py --renderer nyx` also sets a default NYX block if the config
does not define one:

```yaml
nyx:
  spp: 1
  denoise: false
  open_window: false
```

Any keys under `nyx:` are passed to `NyxCameraOptions`, so keep them compatible
with the installed `gs-nyx-plugin` version.

The current `pour_water` example is:

```bash
config/latest_genesis/pour_water_nyx.yml
```

## Slurm

Use a modern GPU. The checked-in helper requests A100, H100, L40S, or A40:

```bash
sbatch scripts/render_pour_water_nyx.sbatch
```

That script writes:

- video: `outputs/nyx_pour_water/video/seed_0.mp4`
- trajectory: `outputs/nyx_pour_water/seed_0.pkl`
- log: `job_logs/nyx_pour_<jobid>.log`

For other tasks, use the same command shape and change `--task`, `--config`,
and `--save-dir`.

## Assets

NYX uses the same asset loading path as the regular task render. If this repo
does not contain a full local `assets/` directory, create a symlink to the
shared asset tree before running tasks:

```bash
ln -s /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench/assets assets
```

Do this only when `assets` does not already exist.

## Grasp Poses

Selecting `--renderer nyx` does not choose a different grasp pose. Grasp
selection still comes from the normal task and object config path. For example,
the validated `pour_water` NYX run used:

```text
manual_body_edge_topdown_posx_posz
```

If NYX and regular renders appear to use different grasps, compare the task
config, object config, random seed, and any forced-grasp setting first. The
renderer should not be the reason.

## Does NYX Recompile Every Run?

NYX itself does not need to be rebuilt from source for every run. However, the
runtime stack can JIT-compile GPU kernels or rebuild local caches when the cache
directory is empty, the environment changes, or the job lands on a different GPU
architecture.

The Slurm helper intentionally points caches at `${TMPDIR:-/tmp}`:

```bash
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/genesis_hr_bench_numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/genesis_hr_bench_mpl_cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/genesis_hr_bench_xdg_cache"
```

That keeps jobs isolated and avoids stale shared-cache issues, but it can make
each fresh job pay some compile/cache warmup cost. This is cache generation, not
manual NYX recompilation.

## Environment Check

Before running a full task, verify the NYX install:

```bash
export GENESIS_BACKEND=gpu
python scripts/check_genesis_nyx.py
```

Expected output includes Genesis, `gs-nyx`, `gs-nyx-plugin`, CUDA availability,
and a small `nyx_rgb` tensor summary.

## Troubleshooting

- `renderer='nyx' requires the CUDA backend`: run on a GPU node and set
  `GENESIS_BACKEND=gpu`.
- `ModuleNotFoundError: gs_nyx_plugin`: use a Python environment with
  `gs-nyx-plugin` installed.
- Video is missing or has only a few frames: check the Slurm log for an earlier
  task failure before video encoding finished.
- Unicode decode errors while reading object config: ensure YAML object/grasp
  files are opened as UTF-8. The current grasp loader does this.
- Black or unexpectedly dark output: start with `spp: 1`, `denoise: false`,
  `open_window: false`, and a simple directional light, then tune `nyx:` options
  for quality.
