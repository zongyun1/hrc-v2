# Target-object catalog & unified selection

`envs/object_catalog.py` is the **single source of truth** for the manipulable
*target* objects tasks pick/place (not the random tabletop clutter, which lives
in `utils.table_object_pool`).

## The mechanism

Every task declares an `OBJECT_SET` class attribute — either form:

- **white-list** — an inline list of object keys, often size 1:
  `OBJECT_SET = ["041_shoe"]` (the shoe task only ever uses a shoe).
- **category** — a named, shared list resolved from `CATEGORIES`:
  `OBJECT_SET = "tabletop_pick_pool"`.

Both are just lists of object keys; a white-list is inline, a category is named
and shared. A `@model` suffix pins a non-default model variant
(`"035_apple@1"`).

The task picks its target via `BaseTask.resolve_target_object(randomize=...)`:

- `config["object_name"]` forces a specific key (validated against the set);
- else a key is sampled (`randomize=True`) or the first key is used
  (`randomize=False`, for tasks with a stable default like dump_bin's cube);
- returns the catalog `ObjectEntry`, stores it on `self._target_entry`, and
  re-templates `self.instruction` (fills any `{object}` slot with the object's
  display name).

`ObjectEntry.as_spec()` returns the legacy per-task `spec` dict shape
(`object_id, model_id, quat, friction, kind`, plus any set manipulation hints)
so downstream spawn code is unchanged after migration.

Multi-object tasks can use `BaseTask.resolve_target_objects(count,
replace=False)` for seeded distinct draws. An explicit `object_name` override
intentionally repeats that object, which is useful for per-object sweeps.
Tasks that need raw pairs can consume the **whole** set via
`object_catalog.object_set_pairs(set_name)` → `[(object_id, model_id), ...]`.

## Catalog entry fields

Identity + physics + manipulation hints, keyed by object key:
`kind (mesh|primitive)`, `object_id`, `model_id`, `display_name`, `spawn_quat`,
`friction`, `scale`, and optional hints `close_value`, `grasp_z_offset`,
`topdown_grasp_z_offset`, `topdown_approach_xy_offset`,
`topdown_preopen_steps`, `topdown_close_value`, `pre_dist`,
`spawn_z_clearance`, `grasp_categories`, `finger_pd`. Hints are
emitted by `as_spec()` only when set, under the keys tasks already read.

## Instruction templating

`INSTRUCTION` may contain a `{object}` slot. Per reset, the resolved object's
`display_name` fills it into `self.instruction` (collected into training data).
For VLA eval the policy instruction is refreshed each episode from
`task.instruction` (`scripts/eval.py` calls `policy.set_instruction(...)`).
Static instruction lookups (`get_task_instruction`, `eval_vla`, `collect_vla`)
use `BaseTask.default_instruction()`, which fills `{object}` with a
representative object so a raw placeholder is never surfaced. `eval_vla` /
`collect_vla` only force `config["instruction"]` on an explicit `--instruction`
override — otherwise the task templates per-episode.

## Migrated tasks

**Shared pool** — `tabletop_pick_pool` = `[048_stapler, 031_jam-jar, cube,
038_milk-box, 035_apple@1, 086_woodenblock@1, 100_seal@1, 073_rubikscube@1]`.
All variants of a family are **identical** (point at the same set).

| Family | OBJECT_SET | How it draws |
|---|---|---|
| put_object_cabinet (+assist/interrupt/neutral) | `tabletop_pick_pool` | samples 1 / forced |
| dump_bin (+assist/interrupt/neutral) | `tabletop_pick_pool` | distinct samples per episode / forced |
| categorize_cooperative / _interrupt / _neutral | `tabletop_pick_pool` | samples `NUM_CATEGORIES`(3) distinct/episode, one per basket |
| place_dual_shoes (+variants) | `["041_shoe"]` | shoe |
| stack_bowls_three (+variants) | `["002_bowl"]` | bowl (model 1) |
| place_bread_in_basket (+variants) | `["075_bread"]` | bread |
| oil_bottle_recovery | `["029_olive-oil"]` | oil bottle |
| take_from_human_easy | `human_transfer_pool` | apple or Rubik's cube |
| pour_water | `["039_mug"]` | mug (grasped target) |

Additional shared pools:

| Pool | Tasks | Members |
|---|---|---|
| `human_transfer_pool` | `deliver_to_human_easy`, `take_from_human_easy` | `035_apple@1`, `073_rubikscube@1` |
| `factory_product_pool` | `factory_inspect_pack`, `factory_kit_packing` | coffee box, milk box, tea box, tissue box, Rubik's cube, wooden block |

Dump-bin, categorize, and factory multi-object draws are without replacement
unless an explicit debug object override is supplied.

For single-object tasks the module ID constants are **derived from the
catalog** (`_SHOE_ID = get_entry("041_shoe").object_id`) — one source of truth.

**Primitive cube** is supported in cabinet + categorize via `create_primitive`
+ the radius-based pick path (no mesh / grasp YAML needed); `ObjectEntry.prim_half`
sizes it.

## Pool feasibility (single-seed validation — noisy)

The pool *resolves* and *runs* uniformly. After the manipulation fixes below,
**every object succeeds at least sometimes** in dump + cabinet (incl. variants);
success rates still vary per object (the harder ones — apple, woodenblock — are
~1/3), which is acceptable for data collection (failed one-shot picks are
dropped) but means eval rates differ per object.

**Four manipulation fixes** (cabinet + dump share
`TopDownPickPlaceMixin.pick_and_place`, so the grasp is identical — the
differences were the place strategy, success criterion, and per-object grip):

1. **dump `_read_center` scale bug** — returned the mesh-origin offset
   **unscaled**, so offset-centre objects (100_seal, centre 0.947 mesh-units)
   spawned ~1 m off the table. Now `center * scale`.
2. **dump place generalized** — dump released at the 0.94 m transit altitude
   and let objects free-fall into the floor bin (tuned for a symmetric cube).
   Now hover-releases at `rim + DROP_HOVER_Z_ABOVE_RIM` (0.20 → ≈0.42 m, the
   lowest Franka-reachable height). Recovered seal + stapler.
3. **Seeded descent** (`PlaceSpec.descent_seeded`, opt-in; dump sets it) — the
   floor bin is at the Franka reach edge where the straight vertical screw
   crosses a wrist singularity and the fallback misplaces (round objects roll
   out). The seeded IK + joint-space-RRT descent preserves the branch. Made
   stapler/rubikscube reliable; default-off keeps the screw for table baskets.
4. **Per-task grip overrides** — cabinet keeps its catalog close values, while
   dump-bin removes apple's 0.35 override (uses radius adaptation) and gives
   milk box a centered 0.55 close. Applying the cabinet's deeper apple/milk
   closes in dump-bin displaced both objects in the 2026-07 pool sweep.
5. **Dump pick-center fix** — mesh targets now use the scaled, rotated model
   center instead of the actor/mesh origin. The old origin was 3.3 cm off on
   the milk carton, so every close was asymmetric regardless of jaw width.
6. **Narrow-clearance approach fix** — tea box, tissue box, seal, and jam jar
   wait for the Franka fingers to reach their fully open command and compensate
   the measured +1 cm commanded-Y tracking bias during top-down approach.
7. **Tall-object finger clearance** — jam jar grasps 1.5 cm higher; coffee box
   and wooden block grasp 3 cm higher. Milk box no longer uses top-down: it
   uses the upper-body horizontal `grasp_065`, then must survive a 100-step
   post-lift hold. Jam also uses a task-local 0.55 close so the higher body
   contact has sufficient squeeze.

`close_value` is read by dump + cabinet, while the `topdown_*` hints are
consumed by the shared Cartesian dump picker. Categorize reads
`model_data:grasp_close_target_m`, so tuning it here doesn't affect categorize.

## Out of scope / open

- **Primitive-target tasks** (`blocks_ranking_*`, `stamp_documents` push docs)
  and **multi-object trays** (`place_burger_fries`, `place_food_in_skillet`) are
  not catalog-driven — their "target" is a primitive or a fixed multi-item set.
- **close_value is task-dependent for some objects** — e.g. factory tasks tune
  `038_milk-box` locally while dump-bin applies its own shallow close;
  `023_tissue-box`
  has no close in the cabinet but 0.46 in factory packing. The catalog holds a
  canonical value; tasks whose tuning genuinely differs (`factory_inspect_pack`,
  `take_from_human_safety`) were left
  un-migrated pending a decision on canonicalizing those per-object values vs.
  allowing per-task overrides.

## Validation

- `scripts/_validate_catalog_resolver.py` — fast, no-scene resolver checks
  (sampling, forcing, alias, bad-key raise, set membership, hint flow, derived
  categorize lists). Run via sbatch GPU.
- `scripts/_smoke_rollout.py <task>` — single-episode reset+play_once+success
  for a robot-only base task (end-to-end integration).
