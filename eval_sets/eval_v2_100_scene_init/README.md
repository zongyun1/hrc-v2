# eval_v2_100_scene_init

This eval set is the scene-init version of `eval_v1_100`: 100 deterministic
initialization specs for each formal task.

Files:

- `manifest.jsonl`: conventional eval-set rows, one per episode.
- `per_task/<task>.yaml`: concrete scene-init YAML for one task.
- `scene_init.yaml`: combined scene-init YAML for all tasks, written by
  `finalize_scene_init_set.py` after all per-task files validate.
- `validation_summary.json` / `validation_errors.json`: final validation
  outputs.

Run eval with both the task and matching scene-init file. Example:

```bash
python scripts/eval.py \
  --task pour_water \
  --scene-init-file eval_sets/eval_v2_100_scene_init/per_task/pour_water.yaml \
  --episodes 100
```

The seed remains as source metadata for rebuilding task topology, but final
avatar/table/object poses are replayed from the YAML scene-init spec.
