# Running Tasks with LuisaRender (Raytracer)

How to render any task with the photorealistic LuisaRender backend on the Unity SLURM cluster.

## TL;DR

1. Use the **MAWM** Python env (it has `LuisaRenderPy.so` built in its Genesis copy).
2. Put **this project** first on `PYTHONPATH` so our task code is used, but let MAWM's Genesis resolve `LuisaRenderPy`.
3. Use `--renderer raytracer` or a raytracer config under `config/`.
4. Allocate a GPU and enough wall time (see timing section below).

## Why this setup

LuisaRender ships as a C++ extension (`LuisaRenderPy.cpython-310-x86_64-linux-gnu.so`) that is **only built inside MAWM's Genesis**. Our project's code imports Genesis at runtime; if Genesis has a LuisaRender `.so` next to it, raytracer rendering works — otherwise you get an `ImportError` and the task falls back to rasterizer.

The trick: run with MAWM's Python (so its Genesis + LuisaRender are importable), but prepend this repo to `PYTHONPATH` so task definitions in `envs/` come from here.

Path to the `.so` (for reference, no action needed):
```
/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/MAWM_Genesis/Genesis/genesis/ext/LuisaRender/build/bin/
```

## Paths and constants

- **MAWM Python**: `/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM/bin/python`
- **Project root**: `/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench`
- **Base raytracer config**: `config/raytracer.yml` (`spp=128`, `tracing_depth=32`, EXR environment + lights, no explicit denoise setting)

## Recommended Settings

Use LuisaRender at `spp=128` as the default quality/speed tradeoff. Increase
samples only for small final-quality checks or specific scenes where the
default remains too noisy.

Recommended command shape:

```bash
python scripts/collect.py --task <task> --episodes 1 --renderer raytracer \
  --config config/raytracer.yml
```

Notes:

- `spp=128` is the current proposed default. Use higher values only for a small
  final-quality ablation or a specific scene where the 128-spp output is still
  too noisy.
- Keep randomized table geometry/materials enabled with `--table-surface asset` unless you are specifically debugging table material issues.

## Submitting a job

Use the generic launcher — it wires up the MAWM Python env, the
a100|l40s|a40 GPU constraint, 12h wall time, and the `data/<task>/luisa/`
save layout for you:

```bash
sbatch scripts/launch/luisa.sh <task> [config.yml]
```

- `<task>` is the task name registered in `envs/tasks/__init__.py`
  (e.g. `deliver_to_human_easy`, `pour_water`, `categorize_cooperative`).
- `[config.yml]` is optional — defaults to `config/raytracer_fast.yml`.
  Pass `config/raytracer.yml` for the base raytracer settings, or a per-run
  sweep config under `config/sweep/`.

For the rasterizer equivalent (robotwin env, no LuisaRender), use
`scripts/launch/rasterizer.sh` the same way.  Both launchers redirect
output to `data/<task>/<renderer>/run.log`.

## Timing and resources

- **Wall time**: use `-t 4:00:00` for review videos. The default limit can kill Luisa jobs before video encoding finishes.
- **GPU**: required. Raytracer sets `GENESIS_BACKEND=gpu` automatically.
- **Memory**: 64G is enough. Cluster warns about overhead but has worked fine.
- **CPUs**: 8 is enough; LuisaRender is GPU-bound.

Measured review-video timings from the `spp=128/256/512` ablation:

| task | frames | spp128 | spp256 | spp512 |
|---|---:|---:|---:|---:|
| `dump_bin` | 1427 | 10:21 | 31:07 | 52:32 |
| `put_object_cabinet` | 3168 | 25:02 | 1:24:38 | 2:25:02 |

The `spp=128` runs above used L40S GPUs, while the `spp=256/512` runs used A100 GPUs, so the 128-to-256 ratio is not a perfectly controlled comparison. The clean comparison from the A100 runs is that `spp=512` cost about 1.7x as much wall time as `spp=256`. All runs used 1 GPU, 8 CPUs, 64G memory, `1280x720`, `video-stride 3`, denoise enabled, `crf=14`, and `preset=slow`.

## Outputs

After success:
- Video: `data/<task>/luisa/video/seed_0.mp4`
- Trajectory: `data/<task>/luisa/seed_0.pkl`
- Log: `data/<task>/luisa/run.log`

**Do not** save under `output/` — videos/data should always live under `data/<task>/...`.

## Fast iteration: rasterizer first

LuisaRender is too slow for debugging task logic. For iteration, use the rasterizer (robotwin Python env, no LuisaRender needed):

```bash
sbatch scripts/launch/rasterizer.sh <task> [config.yml]
```

Rasterizer finishes in ~5–10 min per episode. Only submit the LuisaRender job once the task logic looks correct in the rasterizer video.

## Troubleshooting

- **`ImportError: LuisaRenderPy`** — You're using a Python env without LuisaRender. Switch to the MAWM Python.
- **Job killed at ~1h** — Missing `#SBATCH -t 4:00:00`; the default limit is too short.
- **Imports find the wrong task code** — Make sure `PYTHONPATH` prepends this repo so our `envs/` is found before any stale copy in MAWM.
- **Rasterizer output identical to a prior run** — `seed_0.mp4` is overwritten in place. Check the file mtime (`ls -la`) to confirm the new render.
