# Which GPUs can we actually use?

Measured 2026-07-13 by running `scripts/check_genesis_nyx.py` on one node of each
GPU type on the cluster. Read this before sizing any GPU run (collection, eval,
render sweeps) — the usable pool is **not** what you'd guess, and the two biggest
surprises are load-bearing.

## TL;DR

```
--constraint='2080_ti|l40s|a16|a40|l4|rtx_8000|a4000'
--exclude=uri-gpu[001-017],umd-cscdr-gpu[001-002],gpu013,gpu014,gpu015,gpu017,gpu018,gpu019,gpu020,gpu021,gpu051,gypsum-gpu151,gypsum-gpu055
```

- **578 usable GPUs.** Plenty — the cluster is not the bottleneck.
- **Do NOT ask for `a100` / `h100` for NYX.** They are the biggest, most-idle pool
  and they **do not work** (see below). This is the trap.
- **`2080_ti` DOES work** (352 GPUs, the largest usable pool). Older notes calling
  it "PTX-broken" are **stale** — that was the pre-1.2.0 Genesis fork.

## Two independent constraints

A job needs to satisfy BOTH:

**1. Genesis GPU backend — needs a modern arch.**
Rules out `1080_ti` (sm_61), `titan_x` / `m40` (sm_52) — ~570 GPUs. Anything
sm_75+ is fine.

**2. NYX needs a GRAPHICS-capable GPU (Vulkan).**
This is the non-obvious one. NYX renders through Vulkan, and the datacenter
compute parts (**A100, H100, H200**) are compute-only — no graphics/display
engine — so Vulkan cannot initialise. NYX dies at scene build with:

```
[ERROR][NYX][ASSERT_FAILURE] Vulkan API: Failed to create the device.
[Vulkan Loader] ERROR: vkGetDeviceQueue: Invalid device
  ...or...
[ERROR][NYX][ASSERT_FAILURE] Vulkan API: Failed to allocate buffer memory
```

Compute capability is **not** the criterion — RTX 8000 (sm_75) and 2080 Ti (sm_75)
render fine, while A100 (sm_80) and H100 (sm_90) fail. What matters is whether the
card has a render engine.

> The checked-in `scripts/render_pour_water_nyx.sbatch` requests
> `--constraint="a100|h100|l40s|a40|a16"`. The `a100|h100` part is **wrong** and
> will fail (or silently waste a good node's slot). Don't copy it.

## Measured results

| type | GPUs | cc | NYX? | note |
|---|---|---|---|---|
| **2080_ti** | **352** | 7.5 | ✅ OK | largest usable pool; the "PTX-broken" note is stale |
| **l40s** | 88 | 8.9 | ✅ OK | |
| **a16** | 64 | 8.6 | ✅ OK | |
| v100 | 25 | 7.0 | ✅ (expected) | at the sm_70 floor |
| l4 | 18 | 8.9 | ✅ OK | |
| rtx_8000 | 17 | 7.5 | ✅ OK | |
| a40 | 12 | 8.6 | ✅ OK | |
| a4000 | 2 | 8.6 | ✅ OK | |
| **a100** | **157** | 8.0 | ❌ **FAIL** | compute-only → no Vulkan. Failed on **4/4** nodes across 3 families: `gpu001`, `gpu004`, `gpu016` (good `gpu*` family, bad nodes excluded) and `uri-gpu007` |
| **h100** | 8 | 9.0 | ❌ **FAIL** | compute-only → no Vulkan |
| h200_nvl | 8 | 9.0 | ❌ (assumed) | compute-only |
| 1080_ti | 303 | 6.1 | ❌ | arch too old for Genesis |
| titan_x | 188 | 5.2 | ❌ | arch too old |
| m40 | 76 | 5.2 | ❌ | arch too old |

**Usable total: 578 GPUs** (2080_ti 352 + l40s 88 + a16 64 + v100 25 + l4 18 +
rtx_8000 17 + a40 12 + a4000 2).

## Bad node families (exclude regardless of GPU model)

Some nodes fail Genesis/NYX no matter which card they hold — driver/PTX issues,
not GPU-model issues:

- `uri-gpu[001-017]`, `umd-cscdr-gpu[001-002]` — PTX + Vulkan broken
- `gpu013-015,017-021,051`, `gypsum-gpu151`, `gypsum-gpu055` — PTX / SIGABRT

Note these families hold *all* the `h100` and most `l4`/`rtx_8000` nodes, so
excluding them removes those types from scheduling entirely. Not a loss — the
`2080_ti` + `l40s` + `a16` pool (504 GPUs) is what you'll actually run on.

See `memory/project_genesis_ptx_bad_nodes` for the live bad-node list.

## Don't need NYX? Then you don't need any of this

Renderer choice does **not** affect physics, planning, grasps, or success — only
the images. For debug/physics-only work use the rasterizer on CPU
(`GENESIS_BACKEND=cpu`), which runs anywhere and sidesteps every constraint above.
NYX is a *smart* default: it auto-falls back to the rasterizer when it can't run
(CPU backend, no CUDA, incompatible plugin), so a CPU job silently does the right
thing. See [`nyx_render_howto.md`](nyx_render_howto.md).

## How to re-verify

One node per type, ~2 min each:

```bash
sbatch -p gpu-preempt --gres=gpu:1 --constraint=<type> --mem=32G -c 4 -t 0:20:00 \
  --wrap 'GENESIS_BACKEND=gpu yz/env/MAWM_nyx_clean/bin/python scripts/check_genesis_nyx.py'
```
Success prints an `nyx_rgb` tensor summary.
