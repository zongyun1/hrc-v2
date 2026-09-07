"""VLA-data SLURM scheduler / queue daemon.

Drives the data pipeline. The set of stages run depends on --renderer:

  --renderer both (default):  legacy 2-stage pipeline.
      Stage 1 (rasterizer, verdict-only): fast triage, no H5.
      Stage 2 (raytracer, recording):     auto-enqueued for stage-1 wins.
      H5 written only by stage 2.

  --renderer rasterizer:  single stage 1, mode=record.
      sbatch_raster.sh ... record  → writes steps.h5 from rasterized images.
      No stage 2.

  --renderer raytracer:   single stage 2, mode=record, no stage-1 gating.
      sbatch_luisa.sh ... record   → writes steps.h5 from raytraced images.
      Every seed goes straight to recording (no cheap verdict pre-filter).

Caps: --max-raster (default 25) and --max-luisa (default 20). Either is
ignored when the corresponding stage is disabled.

State per (task, seed, stage):
  pending        no verdict.json / meta.json, not currently running
  running        squeue shows a job whose name encodes (task, seed, stage)
  done           stage's expected output present (see _refresh_state)
  failed_task    verdict shows success=False AND infra_failure_reason=None
                 (does not auto-promote to stage 2; does not retry)
  failed_infra   verdict shows infra_failure_reason set OR job exited
                 nonzero with no verdict at all (retry up to --max-retries)

Loop cadence: poll every --poll-interval seconds, dispatch up to caps,
write _status.csv.  Blocking; runs in tmux foreground.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
RASTER_SBATCH = REPO / "scripts/vla_data/sbatch_raster.sh"
LUISA_SBATCH = REPO / "scripts/vla_data/sbatch_luisa.sh"

# SLURM job names encode (stage, task, seed) so we can recognize our own
# jobs in squeue without reading job files.  Keep them <= 30 chars (slurm
# truncates, but readable): vlad_r_<task_abbr>_<hash>_<seed>.
def _job_name(stage: str, task: str, seed: int) -> str:
    abbr = task.replace("_", "")[:10]
    digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:4]
    tag = "r" if stage == "raster" else "l"
    return f"vlad_{tag}_{abbr}_{digest}_{seed}"


# Seed-level state record.
@dataclass
class SeedState:
    task: str
    seed: int
    stage: str  # "raster" or "luisa"
    out_root: Path
    status: str = "pending"
    job_id: str | None = None
    retries: int = 0

    @property
    def episode_dir(self) -> Path:
        return self.out_root / f"seed_{self.seed}"

    @property
    def verdict_path(self) -> Path:
        return self.episode_dir / "verdict.json"

    @property
    def meta_path(self) -> Path:
        return self.episode_dir / "meta.json"


def _read_verdict(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _squeue_jobs(user: str) -> dict[str, str]:
    """Return {job_name -> job_state} for all of `user`'s jobs."""
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-u", user, "-o", "%j|%T"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        if "|" not in line:
            continue
        name, state = line.split("|", 1)
        res[name.strip()] = state.strip()
    return res


def _ensure_dirs(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)


def _list_seed_states(
    task: str, seeds: Iterable[int], stage: str, out_root: Path
) -> list[SeedState]:
    return [SeedState(task=task, seed=s, stage=stage, out_root=out_root) for s in seeds]


def _refresh_state(state: SeedState, live_jobs: dict[str, str],
                   stage_modes: dict[str, str]) -> None:
    """Update `state.status` based on disk state and live SLURM jobs.

    `stage_modes` maps stage name ("raster"/"luisa") to its mode
    ("verdict" or "record"). When a stage is in record mode, the seed is
    only "done" once meta.json (written by VLARecorder.close) appears.
    """
    name = _job_name(state.stage, state.task, state.seed)
    mode = stage_modes.get(state.stage, "verdict" if state.stage == "raster" else "record")
    # Done?
    if mode == "record":
        if state.meta_path.exists():
            state.status = "done"
            return
        v = _read_verdict(state.verdict_path)
        if v is not None and v.get("infra_failure_reason"):
            state.status = "failed_infra"
            return
        if v is not None and not v.get("success", False):
            state.status = "failed_task"
            return
    else:  # verdict-only stage
        v = _read_verdict(state.verdict_path)
        if v is not None:
            if v.get("infra_failure_reason"):
                state.status = "failed_infra"
            elif not v.get("success", False):
                state.status = "failed_task"
            else:
                state.status = "done"
            return

    # Running?
    if name in live_jobs:
        state.status = "running"
        return

    # Was running last we checked, but slurm no longer lists it AND no
    # verdict landed?  Treat as infra failure (timeout, OOM, node death).
    if state.status == "running":
        state.status = "failed_infra"
        return

    # Otherwise: still pending (or fresh).
    if state.status not in ("done", "failed_task", "failed_infra"):
        state.status = "pending"


def _submit(stage: str, mode: str, task: str, seed: int, out_root: Path,
            extra_flags: list[str], dry_run: bool) -> str | None:
    sbatch_path = RASTER_SBATCH if stage == "raster" else LUISA_SBATCH
    cmd = [
        "sbatch", "--parsable",
        "-J", _job_name(stage, task, seed),
        str(sbatch_path),
        task, str(seed), str(out_root), mode,
        *extra_flags,
    ]
    if dry_run:
        print(f"[dry] {' '.join(shlex.quote(c) for c in cmd)}")
        return "DRYRUN"
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out.strip().split(";")[0]  # job_id; ignore cluster suffix
    except subprocess.CalledProcessError as e:
        print(f"[scheduler] sbatch failed for {stage}/{task}/{seed}: {e.output}",
              file=sys.stderr)
        return None


def _write_status_csv(state_dir: Path, all_states: list[SeedState]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / "_status.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "stage", "seed", "status", "job_id", "retries"])
        for s in all_states:
            w.writerow([s.task, s.stage, s.seed, s.status, s.job_id or "", s.retries])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True,
                   help="Comma-separated task names (Wave A: pour_water,deliver_to_human_easy,categorize_interrupt)")
    p.add_argument("--seeds", default="0-49",
                   help="Range like '0-49' or comma list")
    p.add_argument("--root", required=True, type=Path,
                   help="Top-level output root, e.g. /scratch4/.../vla_data")
    p.add_argument("--renderer", choices=["rasterizer", "raytracer", "both"],
                   default="rasterizer",
                   help="rasterizer (default) / raytracer = single-stage record "
                        "(writes H5); both = legacy verdict→record pipeline "
                        "(H5 from stage 2 only).")
    p.add_argument("--max-raster", type=int, default=25)
    p.add_argument("--max-luisa", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=2,
                   help="Max retries for infra failures (per stage per seed)")
    p.add_argument("--keep-trying", action="store_true",
                   help="Auto-allocate fresh seeds (beyond --seeds) to replace "
                        "task failures, until --target-successes successes per "
                        "task land. Capped by --max-seeds-per-task.")
    p.add_argument("--target-successes", type=int, default=None,
                   help="With --keep-trying, target this many H5 episodes per "
                        "task. Default = number of seeds in --seeds.")
    p.add_argument("--max-seeds-per-task", type=int, default=None,
                   help="With --keep-trying, hard cap on total seeds tried per "
                        "task (safety brake). Default = 5 × target.")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--user", default=os.environ.get("USER", ""))
    p.add_argument("--randomize-avatar", action="store_true")
    p.add_argument("--randomize-table", action="store_true",
                   help="Forward --randomize-table to collect_vla.py. "
                        "This also enables the table debug gate there.")
    p.add_argument("--random-object", action="store_true",
                   help="Forward --random-object to collect_vla.py for tasks "
                        "that support object identity randomization.")
    p.add_argument("--random-interrupt-target", action="store_true",
                   help="Forwarded to categorize_interrupt episodes only")
    p.add_argument("--once", action="store_true",
                   help="One pass then exit (for testing)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    # Renderer → which stages run, and in what mode.
    if args.renderer == "rasterizer":
        stage_modes = {"raster": "record"}
    elif args.renderer == "raytracer":
        stage_modes = {"luisa": "record"}
    else:  # "both"
        stage_modes = {"raster": "verdict", "luisa": "record"}

    raster_root = lambda task: args.root / task / "rasterizer"
    luisa_root = lambda task: args.root / task / "raytracer"

    # Build initial state. For "both", only stage-1 entries are seeded
    # up-front; stage-2 entries materialize from stage-1 wins. For
    # single-stage runs we seed that stage directly.
    all_states: list[SeedState] = []
    for task in tasks:
        if "raster" in stage_modes:
            _ensure_dirs(raster_root(task))
            all_states.extend(_list_seed_states(task, seeds, "raster", raster_root(task)))
        if "luisa" in stage_modes and "raster" not in stage_modes:
            _ensure_dirs(luisa_root(task))
            all_states.extend(_list_seed_states(task, seeds, "luisa", luisa_root(task)))
        elif "luisa" in stage_modes:
            # Pipeline mode: ensure dir exists; entries materialize lazily.
            _ensure_dirs(luisa_root(task))

    # --keep-trying bookkeeping: per-task seed allocator + targets.
    target = args.target_successes if args.target_successes is not None else len(seeds)
    max_seeds = args.max_seeds_per_task if args.max_seeds_per_task is not None else 5 * target
    allocated: dict[str, set[int]] = {t: set(seeds) for t in tasks}
    next_seed: dict[str, int] = {t: max(seeds) + 1 for t in tasks}
    # In pipeline mode an episode is "complete" when the luisa stage is done.
    # In single-stage runs the only stage is the terminal one.
    terminal_stage = "luisa" if "luisa" in stage_modes else "raster"
    initial_stage = "raster" if "raster" in stage_modes else "luisa"
    initial_root = raster_root if initial_stage == "raster" else luisa_root

    print(f"[scheduler] tasks={tasks} seeds={len(seeds)} root={args.root}")
    print(f"[scheduler] renderer={args.renderer} stage_modes={stage_modes}")
    print(f"[scheduler] caps: raster={args.max_raster}, luisa={args.max_luisa}")
    if args.keep_trying:
        print(f"[scheduler] keep-trying: target={target}/task max_seeds={max_seeds}/task")
    print(f"[scheduler] user={args.user} poll={args.poll_interval}s")

    stop_flag = {"stop": False}

    def _handle_signal(signum, frame):
        print(f"[scheduler] got signal {signum}, finishing current pass and exiting")
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        live_jobs = _squeue_jobs(args.user) if not args.dry_run else {}

        # Materialize stage-2 entries for any newly-successful stage-1 seeds.
        # Only relevant in pipeline ("both") mode where both stages are active.
        if "raster" in stage_modes and "luisa" in stage_modes:
            existing_keys = {(s.task, s.seed, s.stage) for s in all_states}
            for s in list(all_states):
                if s.stage != "raster":
                    continue
                v = _read_verdict(s.verdict_path)
                if v is None or not v.get("success", False):
                    continue
                key = (s.task, s.seed, "luisa")
                if key in existing_keys:
                    continue
                all_states.append(SeedState(
                    task=s.task, seed=s.seed, stage="luisa",
                    out_root=luisa_root(s.task),
                ))
                existing_keys.add(key)

        # Refresh status for everything.
        for s in all_states:
            _refresh_state(s, live_jobs, stage_modes)

        # Reset failed_infra → pending if under retry budget.
        for s in all_states:
            if s.status == "failed_infra" and s.retries < args.max_retries:
                # Wipe stale verdict so the next run gets a clean slate.
                if s.verdict_path.exists():
                    try:
                        s.verdict_path.unlink()
                    except Exception:
                        pass
                s.retries += 1
                s.status = "pending"
                s.job_id = None

        # --keep-trying: allocate fresh seeds to replace failures, until each
        # task has enough "alive" candidates to potentially hit the target.
        # A seed is "alive" if it could still produce a terminal-stage `done`:
        # pending/running, retryable failed_infra, or a non-terminal `done`
        # waiting to promote (stage-1 success in pipeline mode).
        if args.keep_trying:
            per_task_alive: dict[str, int] = defaultdict(int)
            per_task_complete: dict[str, int] = defaultdict(int)
            for s in all_states:
                if s.stage == terminal_stage and s.status == "done":
                    per_task_complete[s.task] += 1
                elif s.status in ("pending", "running"):
                    per_task_alive[s.task] += 1
                elif s.status == "failed_infra" and s.retries < args.max_retries:
                    per_task_alive[s.task] += 1
                elif s.stage != terminal_stage and s.status == "done":
                    # Stage-1 done (pipeline): will promote next iter.
                    per_task_alive[s.task] += 1
            for task in tasks:
                gap = target - per_task_complete[task] - per_task_alive[task]
                while gap > 0 and len(allocated[task]) < max_seeds:
                    new_seed = next_seed[task]
                    next_seed[task] += 1
                    allocated[task].add(new_seed)
                    all_states.append(SeedState(
                        task=task, seed=new_seed, stage=initial_stage,
                        out_root=initial_root(task),
                    ))
                    gap -= 1
                if gap > 0 and len(allocated[task]) >= max_seeds:
                    print(f"[scheduler] {task}: hit max_seeds={max_seeds} cap "
                          f"(complete={per_task_complete[task]}, "
                          f"target={target}); not allocating more.")

        # Count running per stage.
        running_count: dict[str, int] = defaultdict(int)
        for s in all_states:
            if s.status == "running":
                running_count[s.stage] += 1

        # Dispatch.
        flags_for_task = {}
        for task in tasks:
            extras: list[str] = []
            if args.randomize_avatar:
                extras.append("--randomize-avatar")
            if args.randomize_table:
                extras.append("--randomize-table")
            if args.random_object:
                extras.append("--random-object")
            if args.random_interrupt_target and task == "categorize_interrupt":
                extras.append("--random-interrupt-target")
            flags_for_task[task] = extras

        # Round-robin across tasks so all 3 tasks make progress in parallel
        # rather than draining task[0] first.  Group by task within stage,
        # then interleave.
        def _interleave_by_task(stage_name: str) -> list[SeedState]:
            buckets: dict[str, list[SeedState]] = defaultdict(list)
            for s in all_states:
                if s.stage == stage_name and s.status == "pending":
                    buckets[s.task].append(s)
            for v in buckets.values():
                v.sort(key=lambda x: x.seed)
            out = []
            i = 0
            while True:
                progressed = False
                for task in tasks:
                    if i < len(buckets.get(task, [])):
                        out.append(buckets[task][i])
                        progressed = True
                i += 1
                if not progressed:
                    break
            return out

        # Stage 1 first.
        for s in _interleave_by_task("raster"):
            if running_count["raster"] >= args.max_raster:
                break
            jid = _submit("raster", stage_modes["raster"], s.task, s.seed,
                          s.out_root, flags_for_task[s.task], args.dry_run)
            if jid is None:
                continue
            s.job_id = jid
            s.status = "running"
            running_count["raster"] += 1

        # Stage 2 (luisa). In pipeline mode, only seeds with stage-1 success
        # exist here (materialized above). In raytracer-only mode, every
        # seed is a luisa entry from the start.
        for s in _interleave_by_task("luisa"):
            if running_count["luisa"] >= args.max_luisa:
                break
            jid = _submit("luisa", stage_modes["luisa"], s.task, s.seed,
                          s.out_root, flags_for_task[s.task], args.dry_run)
            if jid is None:
                continue
            s.job_id = jid
            s.status = "running"
            running_count["luisa"] += 1

        # Per-task status breakdown for the log line.
        per_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in all_states:
            per_task[(s.task, s.stage)][s.status] += 1
        ts = time.strftime("%H:%M:%S")
        print(f"--- [{ts}] running raster={running_count['raster']}/{args.max_raster} "
              f"luisa={running_count['luisa']}/{args.max_luisa} ---")
        for (task, stage), counts in sorted(per_task.items()):
            line = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  {task} [{stage}]  {line}")

        # Persist per-task status csv.
        for task in tasks:
            for stage, root_fn in (("raster", raster_root), ("luisa", luisa_root)):
                subset = [s for s in all_states if s.task == task and s.stage == stage]
                if subset:
                    _write_status_csv(root_fn(task), subset)

        # Are we done?
        terminal = {"done", "failed_task", "failed_infra"}
        all_terminal = all(
            s.status in terminal for s in all_states
            if not (s.status == "failed_infra" and s.retries < args.max_retries)
        )
        # Stage-1 seeds whose verdict didn't land yet still need a wait.
        pending_or_running = [s for s in all_states if s.status in ("pending", "running")]

        if args.keep_trying:
            # Exit only when every task has hit target OR exhausted its
            # max-seeds budget. The next loop iteration will allocate any
            # missing seeds, so we only stop when there's truly nothing
            # left to do.
            n_complete_per_task = defaultdict(int)
            for s in all_states:
                if s.stage == terminal_stage and s.status == "done":
                    n_complete_per_task[s.task] += 1
            target_met = all(n_complete_per_task[t] >= target for t in tasks)
            budget_exhausted = all(
                len(allocated[t]) >= max_seeds
                and n_complete_per_task[t] < target
                for t in tasks
            )
            if (target_met or budget_exhausted) and not pending_or_running:
                if target_met:
                    print(f"[scheduler] all tasks hit target={target} successes, exiting")
                else:
                    print(f"[scheduler] all tasks exhausted max_seeds={max_seeds} "
                          f"without reaching target, exiting")
                return 0
        else:
            if all_terminal and not pending_or_running:
                print("[scheduler] all seeds in terminal state, exiting")
                return 0

        if args.once or stop_flag["stop"]:
            return 0

        # Sleep with periodic wake to check stop flag.
        for _ in range(args.poll_interval):
            if stop_flag["stop"]:
                break
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
