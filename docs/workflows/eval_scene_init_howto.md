# Deterministic Eval Scene Init

Eval should not rely on seeds as the only episode definition.  Seeds are still
useful for training-data generation and for rebuilding task topology, but eval
episodes can now carry a concrete scene-init spec with table/avatar/object
state.

## Generate a scene-init file

From an existing eval manifest:

```bash
srun -p gpu --gres=gpu:1 --mem=32G -c 4 -t 1:00:00 \
  /work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
  scripts/generate_eval_scene_init.py \
    --manifest eval_sets/eval_v1_100 \
    --task pour_water \
    --output eval_sets/eval_v1_100/pour_water_scene_init.yaml
```

Or directly from task/seeds:

```bash
srun -p gpu --gres=gpu:1 --mem=32G -c 4 -t 1:00:00 \
  /work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
  scripts/generate_eval_scene_init.py \
    --task pour_water \
    --episodes 100 \
    --start-seed 0 \
    --output eval_sets/pour_water_scene_init.yaml
```

Use `--randomize-avatar`, `--randomize-table`, `--table-variant-name`, and
`--task-irrelevant-objects` when those should be part of the fixed eval
distribution.  The output records the chosen avatar skin/pose, table variant,
named entity poses/qpos, and simple task init attributes.

## Run eval with scene-init

```bash
scripts/run_vla_eval.sh \
  --model act \
  --checkpoint <checkpoint> \
  --task pour_water \
  --episodes 10 \
  --max-steps 300 \
  -- \
  --scene-init-file eval_sets/eval_v1_100/pour_water_scene_init.yaml \
  --scene-init-start-index 0 \
  --output-json runs/act/eval/pour_water_scene_init/results_0_9.json
```

`scripts/eval.py` accepts the same `--scene-init-file` and
`--scene-init-start-index` flags.  Output JSON includes `scene_init_id`,
`scene_init_source_seed`, and `scene_init_replay`; `missing_entities` should be
empty for a clean replay.
