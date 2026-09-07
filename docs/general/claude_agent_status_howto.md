# Maintaining `AGENT_STATUS.md`

`AGENT_STATUS.md` (repo root) is the **shared, actively maintained log** of
Claude Code agents running in this repo.  The user reads it to know who's
doing what and when a result will land — keep it accurate.

## When to update

- **Session start** — add (or refresh) your row + detail section.
- **Switching tasks** — update your row's "Task" and "ETA" fields.
- **Job milestones** — after submitting a SLURM job, update ETA and
  `Current job`; after the job finishes, update "Result path" (or remove
  the row if the session ends).
- **Every ~30 min while waiting** — bump "Updated" so readers know the
  entry isn't stale.  Stale entries (>2 h without update) are presumed
  dead and can be removed by any other agent.

## Format

Top of the file is a single status table, one row per tmux window:

```
| Tmux | Name | Task (one sentence) | ETA | Result path | Updated |
```

- **Tmux** — the tmux window ID the user uses to reach you (e.g. `0`, `1`).
- **Name** — short handle for the agent/task (lowercase, no spaces).
- **Task** — one sentence, present-tense, describing what you're doing.
  Good: "Iterate deliver_to_human_easy sweep with raytracer.".
  Bad: "Working on stuff."
- **ETA** — absolute timestamp `YYYY-MM-DD HH:MM` when a reviewable result
  should be ready (e.g. a finished video, a PR, a benchmark run).
  Leave `—` if the work is open-ended with no single deliverable.
- **Result path** — relative path where the user should look when ETA
  passes.  Must be a real file/dir path, not a description.
- **Updated** — timestamp of last edit to this row.

Below the table, a single `## Recover` section lists the exact command
needed to re-attach each agent.  Keep it to a table — no per-agent
prose, no "Current job", no "Log tail", no "Notes".  Those details
belong in:

- the agent's own running context (scrollback, tmux buffer)
- a task-specific doc (`docs/<task>.md`) if the context is worth
  preserving across sessions

Rationale: the status log is read at-a-glance by the user to see who is
doing what and when results land.  Long per-agent prose blocks make the
file tedious to skim and go stale within minutes.  The one-sentence
"Task" column is the only narrative; everything else is data.

## Conventions

- **Timezone — US Eastern (the user's zone).**  The SLURM cluster runs in
  UTC, so whenever you record a time convert to Eastern before writing.
  In April–October that's EDT (UTC−4); in November–March it's EST
  (UTC−5).  Timestamp format: `YYYY-MM-DD HH:MM ET` (the `ET` suffix is
  required so readers don't have to guess).  Convert relative terms
  ("in an hour") to absolute ET timestamps in this file.

  Quick conversion: `date -d "$(date -u) -4 hours" +"%Y-%m-%d %H:%M ET"`
  during EDT (and `-5 hours` during EST).
- Don't invent tmux IDs — only edit a row when you know that's your
  window.
- When you finish a task for good, remove your row and section.  Don't
  leave "DONE" entries around.
- One row per agent, not per job.  If you're running multiple jobs in
  one session, mention them in `Notes` but keep the table row singular.
- Keep edits small and atomic — don't rewrite other agents' sections.
