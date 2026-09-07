# Genesis HR Bench

This workspace is organized around the latest Genesis World runtime. The active
integration target is `genesis-world==1.2.0` with the NYX GPU renderer, in the
repo-local `.venv-genesis-latest` environment (see Installation).

## Environment

### Intention

| Task | Description |
|---|---|
| `take_from_human_easy` | Take the requested compact object from the person and place it on the table. |
| `deliver_to_human_easy` | Pick up the requested object from the table and deliver it to the human. |
| `take_from_human_safety` | Take the kitchen knife offered by the chopping human and place it in the knife box. |
| `stamp_documents` | Push each document into the stamp zone so the human can stamp it. |
| `pour_water` | Pour water from the mug into the cup. |
| `oil_bottle_recovery` | Recover the knocked-over oil bottle from the cooking area and place it upright. |
| `open_microwave` | Open the microwave door for the human carrying a plate. |

### Interrupt / Assist / Neutral

| Family | Assist / Cooperative | Interrupt | Neutral |
|---|---|---|---|
| Categorize | `categorize_cooperative`: Sort the objects into matching baskets. | `categorize_interrupt`: Sort objects into matching baskets while yielding when the human inspects one. | `categorize_neutral`: Sort objects while the human works nearby. |
| Put object cabinet | `put_object_cabinet_assist`: Place the object in a human-opened cabinet drawer. | `put_object_cabinet_interrupt`: Place the object after the human inspects it. | `put_object_cabinet_neutral`: Put the object in the cabinet while the human works nearby. |
| Stack bowls three | `stack_bowls_three_assist`: Stack one bowl while the human stacks another. | `stack_bowls_three_interrupt`: Stack bowls while waiting if the human inspects one. | `stack_bowls_three_neutral`: Stack bowls while the human works nearby. |
| Place bread in basket | `place_bread_in_basket_assist`: Place one bread while the human places the other. | `place_bread_in_basket_interrupt`: Place breads while waiting if the human inspects one. | `place_bread_in_basket_neutral`: Place breads while the human works nearby. |
| Dump bin | `dump_bin_assist`: Empty the table cubes while the human tosses another cube into the bin. | `dump_bin_interrupt`: Place each cube into the trash can, waiting if the human inspects one. | `dump_bin_neutral`: Put table objects into the bin while the human works nearby. |
| Place burger fries | `place_burger_fries_assist`: Place meal items while the human adds one item. | `place_burger_fries_interrupt`: Place meal items while waiting if the human inspects one. | `place_burger_fries_neutral`: Place meal items while the human works nearby. |
| Place dual shoes | `place_dual_shoes_assist`: Put one shoe in the shoebox while the human puts the other. | `place_dual_shoes_interrupt`: Put shoes in the shoebox while waiting if the human inspects one. | `place_dual_shoes_neutral`: Place shoes in the shoebox while the human works nearby. |
| Place food in skillet | `place_food_in_skillet_assist`: Place the skillet on the stove and put foods in it while the human helps place one food. | `place_food_in_skillet_interrupt`: Place the skillet and foods while waiting if the human inspects one. | `place_food_in_skillet_neutral`: Place food in the skillet while the human works nearby. |
| Blocks ranking RGB | `blocks_ranking_rgb_assist`: Place red, green, and blue blocks in left-to-right order while the human helps place one. | `blocks_ranking_rgb_interrupt`: Order RGB blocks while waiting if the human inspects one. | `blocks_ranking_rgb_neutral`: Order RGB blocks while the human works nearby. |
| Blocks ranking size | `blocks_ranking_size_assist`: Place blocks from largest to smallest while the human helps place one. | `blocks_ranking_size_interrupt`: Order blocks by size while waiting if the human inspects one. | `blocks_ranking_size_neutral`: Order blocks by size while the human works nearby. |

A human-robot manipulation benchmark built on the
[Genesis](https://github.com/UMass-Embodied-AGI/Genesis) physics simulator.
Ships 44 registered manipulation tasks, multiple robot embodiments
(Franka Panda, Piper, ARX X5, UR5+WSG, xArm7), pluggable motion planners
(mplib, cuRobo, Genesis IK), avatar support for human-robot interaction
tasks.

## Installation

Requires **Python 3.10** and a CUDA GPU. The default **NYX** renderer needs a
**Vulkan-capable** GPU (`2080_ti`, `l40s`, `a40`, `a16`, `rtx_8000`, `a4000`, …)
— **not** `a100`/`h100` (compute-only, no Vulkan; see
[`docs/general/gpu_resources.md`](docs/general/gpu_resources.md)).

```bash
# 1. Create the Genesis runtime env (genesis-world 1.2.0 + NYX, all from PyPI).
python3.10 -m venv .venv-genesis-latest
.venv-genesis-latest/bin/python -m pip install -U pip wheel
.venv-genesis-latest/bin/python -m pip install -r requirements-genesis-latest.txt

# 2. (VLA baselines only) baseline source submodules.
git submodule update --init --recursive   # baseline/{lerobot,openvla-oft,rdt,diffusion_policy,3d_diffusion_policy}
```

Then download `assets/` from the
[OneDrive bundle](https://umass-my.sharepoint.com/:f:/g/personal/qinhongzhou_umass_edu/IgAv1Rf-uN_AQruMkgNuIfhZAQuAIwcvN5yFBne8wFNUXGY?e=31tltm)
and unpack into the repo root (or sync via `scripts/sync_assets.sh pull`).

`assets/objects/` has 13k+ items — OneDrive's own "Download as Zip" button on
that folder is past its per-download item ceiling and will silently hand you
a truncated/corrupted archive. Use `scripts/fetch_objects_zip.sh` instead (or
download the single `objects.zip` file from the bundle by hand); it fetches
and unpacks a pre-built archive, sidestepping OneDrive's folder-zip feature
entirely.

`source scripts/env.sh` activates `.venv-genesis-latest` and sets `PYTHONPATH`;
the examples below call the venv python directly so they work without it.

## Quick start

```bash
export GENESIS_BACKEND=gpu   # required for the NYX renderer

# Collect one scripted demo of a task (writes data/ + a video).
# Note: scene builds are memory-heavy — allocate at least 64GB RAM (32GB can OOM mid-run).
.venv-genesis-latest/bin/python scripts/collect.py --task pour_water --episodes 1

# Run a task with the privileged scripted expert policy.
# eval_isolated.py runs each episode in its own process; it takes the same
# arguments as scripts/eval.py and aggregates the per-episode results.
.venv-genesis-latest/bin/python scripts/eval_isolated.py --task pour_water \
    --policy expert_full_state --episodes 10
```

Tasks are registered in `envs/tasks/__init__.py` (`TASK_MAP`).

## Documentation

| Topic | Doc |
|---|---|
| **Collect VLA training data** (Franka tasks → RLDS / LeRobot) | [`docs/collect-data-portable.md`](docs/collect-data-portable.md) |
| **Fine-tune VLA baselines** — portable launcher, multi-node (start here) | [`docs/vla-finetune-pipeline.md`](docs/vla-finetune-pipeline.md) |
| Fine-tune **Diffusion Policy** (Chi et al., per-task) | [`docs/diffusion-policy-finetune-portable.md`](docs/diffusion-policy-finetune-portable.md) |
| Raytracer (LuisaRender) setup | [`docs/luisa_render.md`](docs/luisa_render.md) |
| Avatar / Mixamo motion conversion | [`docs/mixamo-motion-conversion.md`](docs/mixamo-motion-conversion.md) |

For coding conventions, project layout, and architecture notes
(used by Claude Code when working in this repo), see
[`CLAUDE.md`](CLAUDE.md).

## VLA baselines

Vision-Language-Action baselines run as out-of-process HTTP servers under
`baseline/servers/`. The benchmark talks to them through thin client shims in
`scripts/vla_client.py`. Supported baselines:
OpenVLA-OFT, the pi0 family (pi0 / pi0.5 / pi0-FAST), SmolVLA, RDT,
Diffusion Policy, 3D Diffusion Policy, ACT, LeRobot Diffusion, and VQ-BeT —
see [`docs/vla-finetune-pipeline.md`](docs/vla-finetune-pipeline.md) for the
portable train + multi-node launcher, and `CLAUDE.md` for per-baseline detail.

Use the latest Genesis environment from the repo root:

```bash
source scripts/env.sh
export GENESIS_BACKEND=gpu
```

The local convenience link is:

```bash
.venv-genesis-latest/bin/python
```

Core versions (pinned in `requirements-genesis-latest.txt`, verified end-to-end
— task eval + VLA data collection with the NYX renderer):

- Python 3.10
- `torch==2.4.1+cu121`
- `genesis-world==1.2.0`
- `gs-nyx==0.1.3` / `gs-nyx-plugin==0.1.4`
- `numpy==1.26.4` for `mplib==0.2.1` compatibility
- `h5py==3.16.0` for VLA data recording

## Verify

```bash
source scripts/env.sh
export GENESIS_BACKEND=gpu
python scripts/check_latest_genesis_integration.py \
  --python .venv-genesis-latest/bin/python \
  --robots franka \
  --tasks blocks_ranking_rgb \
  --steps 1
```

If this times out, a cold first-time kernel/JIT compile can exceed the 120s
default — retry with larger `--robot-timeout`/`--task-timeout` values (e.g. `300`).

For the current latest-Genesis status, validation commands, and known remaining
behavior issues, see `docs/latest_genesis_status.md`.

## Rendering

Genesis provides three camera/render paths: **NYX** (default — GPU-only sensor
renderer, needs a Vulkan GPU + `GENESIS_BACKEND=gpu`; transparently falls back
to the rasterizer when unavailable), **Rasterizer** (fast, CPU-friendly), and
**LuisaRender** (raytraced, photo-realistic, GPU-only). Switch via `--renderer`;
see [`docs/nyx_render.md`](docs/nyx_render.md) for NYX setup and
[`docs/luisa_render.md`](docs/luisa_render.md) for the raytracer.
