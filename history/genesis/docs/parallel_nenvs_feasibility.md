# Genesis n_envs parallelism feasibility — avatar focus

**Verdict (2026-07-10): the avatar is NOT the blocker.** Per-env avatar state
is fully supported by Genesis 1.2.0, and per-env avatar *images* are obtainable
today via a sequential broadcast-render recipe (no Genesis patch). True
one-call batched avatar rendering needs a small upstream rasterizer patch.

Probe: `scripts/probe_avatar_nenvs.py` (jobs 61599171→61604001), outputs under
`data/parallel/avatar_nenvs_probe_sp{0,8}/` (`env0_seq0.png` / `env1_seq1.png`
are the money shots: 2 envs, 2 different avatar poses, per-env rigid state).

## What was proven

| Check | Result |
|---|---|
| Scene with full avatar stack (AvatarController + KinematicAvatarSkin + box) builds at `n_envs=2` | PASS |
| Per-env skin poses: `entity.set_vverts(pose, envs_idx=[i])` | PASS — exact round-trip, 0 error |
| Rigid entities per env (`set_pos(..., envs_idx=)`) | PASS |
| Per-env vvert buffers survive `scene.step()` | PASS |
| Today's `update_mesh()` (no envs_idx) broadcasts one pose to all envs | confirmed (the integration change needed) |
| One batched render call returning different avatar poses per env | **FAIL — stock 1.2.0 rasterizer cannot** (see below) |
| Sequential broadcast recipe (per-env pose broadcast → step → render slice i) | **PASS** at env_spacing 0 and 8 |

## Genesis 1.2.0 batched-rasterizer model (established empirically + source)

- One shared pyrender scene; per-env passes swap **rigid** node poses
  (env-local coordinates — the camera transform deliberately gets no env
  offset when `env_separate_rigid=True`).
- Custom-vverts entities (our avatar skin) get one node **per env**
  (`ctx.vverts_nodes[(env, geom.uid)]`), with correct per-env vertex data, but
  the nodes are (a) placed at `vverts + envs_offset` (inconsistent with the
  env-local camera) and (b) **visible in every env's pass** — never toggled
  per pass. Hence every frame shows every env's skin (spacing 0) or one
  arbitrary in-frustum skin (spacing > 0), identical across frames.
- Node GL buffers refresh only when the scene has stepped
  (`update_scene` no-ops when sim time is unchanged; `force_render=True`
  should bypass — untested).
- Two 1.2.0 camera bugs hit along the way: `env_idx`-bound cameras crash at
  `camera.build()` in batched `env_separate_rigid` scenes (use one batched
  `env_idx=None` camera instead → returns `(n_envs, H, W, 3)`), and
  `get_vverts` returns CUDA tensors.

## Working recipe (no Genesis changes)

Physics fully batched; rendering serialized per env at capture time only:

```python
for i in range(n_envs):
    skin.update_mesh(<env i's pose>)   # today's broadcast path, unchanged
    # refresh renderer (scene.step() proven; force_render=True likely cleaner)
    frames = cam.render()[0]           # (n_envs, H, W, 3)
    keep(frames[i])                    # env i's avatar pose + env i's rigid state
```

Use `env_spacing=(0,0)` so all envs' skin nodes overlap (each frame shows one
clean figure). Rigid state per slice is already correct. Cost: N renders per
capture step — acceptable since capture is strided (`record_stride` ≥ 25) and
physics, not rendering, dominates collection wall-clock.

Proper fix upstream (small): in the per-env render pass, hide
`vverts_nodes[(j, ·)]` for j ≠ current env and drop the `+ envs_offset`
placement (or add the same offset to the camera). Then one render call
returns correct per-env avatars.

## Remaining engineering for batched HRI collection (all ordinary, no walls)

- Thread `envs_idx` through `AvatarRobot.update()` / `update_mesh` /
  attachments; run N motion-controller state machines (CPU numpy, ~N× cost).
- Batch robot control (`Arm` uses single-env `get_qpos`/`control_dofs_position`
  — both have batched APIs) and task success checks.
- mplib planning stays serial per env; scene morphology (object models, table
  variant, avatar GLB) must be identical across a batch — batch over layouts,
  not variants.
- GPU-backend physics re-validation (grasp recipes were tuned on CPU).

## Recommendation

Feasible goals, in increasing ambition: (1) `collect.py --workers` process
parallelism (zero risk, immediate); (2) batched robot-only eval
(`take_action` loop is nearly branch-free; genesis_ik batches natively);
(3) batched HRI collection using the recipe above. The avatar no longer
rules out (3).
