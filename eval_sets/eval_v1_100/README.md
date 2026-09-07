# eval_v1_100

This eval set stores deterministic episode initializations only. It does not
contain expert rollouts, videos, actions, observations, or success labels.

It contains 100 initialization specs per task.

Each episode is defined by:

- `task`: registered task name resolved by `envs.tasks.resolve_task_class`
- `seed`: seed passed to `task.reset(seed=seed)`
- `category`: high-level grouping for reporting
- defaults plus any episode-local fields: deterministic environment knobs that
  should be applied during inference

The policy under evaluation should run from these initial states and produce
its own trajectory. Any reference videos outside this directory are visual
debug artifacts, not part of the eval set.
