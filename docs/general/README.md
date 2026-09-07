# General Agent Rules

`docs/general/` is the shared rulebook for coding agents working in this
repository.  Read these docs before changing code, running jobs, collecting
data, or writing outputs.

Required reading for every agent:

- [`running_jobs.md`](running_jobs.md) — cluster execution, log locations,
  review-output locations, cleanup expectations, and task implementation rules.
- [`claude_agent_status_howto.md`](claude_agent_status_howto.md) — how to keep
  `AGENT_STATUS.md` accurate.
- [`grasp_helper_howto.md`](grasp_helper_howto.md) — required grasp helper
  conventions for task code.
- [`topdown_direction_debug_howto.md`](topdown_direction_debug_howto.md) —
  image-plane direction conventions for placement/debug discussions.
- [`nyx_render_howto.md`](nyx_render_howto.md) — the default NYX (`gs-nyx`)
  renderer: which env to use (`MAWM_nyx_clean`), smart-default/fallback
  behavior, cameras, speed, bad-node excludes, and troubleshooting.
- [`gpu_resources.md`](gpu_resources.md) — which GPU types actually work
  (**A100/H100 do NOT run NYX**; `2080_ti` does), the usable 578-GPU pool, and
  the exact `--constraint` / `--exclude` to use. Read before sizing a GPU run.

Keep general, repo-wide rules here.  Task-specific implementation notes belong
in `docs/<task-or-feature>.md`; workflow how-tos belong in
[`../workflows/`](../workflows/); temporary run state belongs in
`AGENT_STATUS.md` or generated files under `data/` and `job_logs/`.
