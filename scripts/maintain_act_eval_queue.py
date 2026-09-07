#!/usr/bin/env python3
"""Keep ACT eval jobs filled and mirror current results into table.txt."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "runs" / "act" / "eval"
FINETUNE_ROOT = REPO_ROOT / "runs" / "act" / "qpos_target_20260524" / "finetune"
TABLE_PATH = REPO_ROOT / "table.txt"
MONITOR_PATH = EVAL_ROOT / "success_rate_latest.md"
MAINTAINER_LOG = EVAL_ROOT / "act_eval_maintainer.log"
EVAL_SCRIPT = "scripts/launch/eval_act_seed_ranges_multi_fixed_out.sbatch"
EVAL_CONFIG = "runs/act/eval/eval_v1_100_act_config.yml"
OPEN_MICROWAVE_EVAL_CONFIG = "runs/act/eval/eval_v1_100_act_open_microwave_config.yml"
TRAIN_DEFAULT_EVAL_CONFIG = "runs/act/eval/eval_v1_100_act_train_defaults_config.yml"
RUN_SUFFIX = "qpos_target_abs_300ep_full300_c100_s25000_novae_v1"
EXCLUDED_TASK_POOL = REPO_ROOT / "vla_data_qpos_target_20260524" / "task_pool.txt"
MAX_EVAL_JOBS = 20
DEFAULT_EVAL_STEPS = 1200
TASK_EVAL_STEPS = {
    # 4x mean successful rollout length in vla_data_qpos_target_20260524.
    "dump_bin": 744,
    "place_bread_in_basket": 478,
    "place_burger_fries": 692,
    "place_dual_shoes": 674,
    "place_food_in_skillet": 708,
    "put_object_cabinet": 1530,
    "stack_bowls_three": 420,
    "take_from_human_easy": 390,
    "deliver_to_human_easy": 544,
    "take_from_human_safety": 800,
    "stamp_documents": 1846,
    "pour_water": 406,
    "oil_bottle_recovery": 1078,
    "open_microwave": 842,
    "put_object_cabinet_assist": 824,
}
PROCS_PER_JOB = 2
EPISODES_PER_PROC = 5
SHARD_SIZE = PROCS_PER_JOB * EPISODES_PER_PROC
MAX_SEEDS = 100
PORT_BASE = 9000
TRAIN_DEFAULT_EVAL_TASKS = {"deliver_to_human_easy"}
SKIP_EVAL_TASKS = {
    # Paused by request: data/eval quality needs separate debugging.
    "take_from_human_easy",
}
BASE_EVAL_TASKS = [
    "dump_bin",
    "place_bread_in_basket",
    "place_burger_fries",
    "place_dual_shoes",
    "place_food_in_skillet",
    "put_object_cabinet",
    "stack_bowls_three",
]
BASE_TABLE_TASKS = {
    "put_object_cabinet_assist": "put_object_cabinet",
    "stack_bowls_three_assist": "stack_bowls_three",
    "place_bread_in_basket_assist": "place_bread_in_basket",
    "dump_bin_assist": "dump_bin",
    "place_burger_fries_assist": "place_burger_fries",
    "place_dual_shoes_assist": "place_dual_shoes",
    "place_food_in_skillet_assist": "place_food_in_skillet",
    "blocks_ranking_rgb_assist": "blocks_ranking_rgb",
    "blocks_ranking_size_assist": "blocks_ranking_size",
}

JOB_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+act_eval\s+(\S+)")
RESULT_RE = re.compile(r"results_seed_(\d+)_(\d+)\.json$")
LOG_RUN_RE = re.compile(r"\[run_vla_eval\] running eval: (.*)")
SUBMIT_RE = re.compile(r"(\d+):([^:\s]+):(\d+)-(\d+):(\S+)")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def active_eval_jobs() -> list[dict]:
    proc = run(["squeue", "-u", subprocess.getoutput("whoami"), "-h", "-o", "%i %P %j %T"], check=False)
    jobs = []
    for line in proc.stdout.splitlines():
        m = JOB_RE.match(line)
        if not m:
            continue
        jobs.append({"id": m.group(1), "partition": m.group(2), "state": m.group(3)})
    return jobs


def table_tasks() -> list[str]:
    lines = TABLE_PATH.read_text().splitlines()
    excluded = excluded_tasks() | SKIP_EVAL_TASKS
    table = [x for x in lines[1].split("\t")[1:] if x and x not in excluded]
    tasks = list(BASE_EVAL_TASKS)
    tasks.extend(task for task in table if task not in tasks)
    return tasks


def excluded_tasks() -> set[str]:
    if not EXCLUDED_TASK_POOL.exists():
        return set()
    text = EXCLUDED_TASK_POOL.read_text()
    return {item.strip() for item in re.split(r"[\s,]+", text) if item.strip()}


def checkpoint_for(task: str) -> Path:
    return FINETUNE_ROOT / f"act_{task}_20260524_{RUN_SUFFIX}" / "checkpoints" / "025000" / "pretrained_model"


def eval_steps_for(task: str) -> int:
    return int(TASK_EVAL_STEPS.get(task, DEFAULT_EVAL_STEPS))


def latest_or_new_output(task: str) -> Path:
    eval_steps = eval_steps_for(task)
    eval_tag = (
        f"evalv1train_s{eval_steps}_100"
        if task in TRAIN_DEFAULT_EVAL_TASKS
        else f"evalv1_s{eval_steps}_100"
    )
    matches = sorted(EVAL_ROOT.glob(f"{task}_{eval_tag}_full300_ckpt025000_*"))
    if matches:
        return matches[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return EVAL_ROOT / f"{task}_{eval_tag}_full300_ckpt025000_{stamp}"


def eval_args_for(task: str) -> list[str]:
    eval_config = OPEN_MICROWAVE_EVAL_CONFIG if task == "open_microwave" else EVAL_CONFIG
    if task == "place_bread_in_basket":
        return [
            "--config", eval_config,
            "--vla-config", "runs/act/eval/act_place_bread_gripper_vla_config.yml",
            "--no-vla-train-defaults",
            "--max-videos-per-dir", "3",
            "--act-gripper-closed-cmd", "0.82",
            "--interpolate-qpos-actions",
            "--qpos-gripper-max-delta", "0.08",
            "--qpos-gripper-close-hold-steps", "25",
            "--success-hold-steps", "1",
        ]
    if task in TRAIN_DEFAULT_EVAL_TASKS:
        return [
            "--config", TRAIN_DEFAULT_EVAL_CONFIG,
            "--max-videos-per-dir", "3",
            "--act-gripper-closed-cmd", "0.0",
            "--interpolate-qpos-actions",
            "--qpos-gripper-max-delta", "0.08",
            "--qpos-gripper-close-hold-steps", "25",
            "--success-hold-steps", "1",
        ]
    return [
        "--config", eval_config,
        "--no-vla-train-defaults",
        "--max-videos-per-dir", "3",
        "--act-gripper-closed-cmd", "0.0",
        "--interpolate-qpos-actions",
        "--qpos-gripper-max-delta", "0.08",
        "--qpos-gripper-close-hold-steps", "25",
        "--success-hold-steps", "1",
    ]


def completed_seed_ranges(out: Path) -> set[int]:
    seeds = set()
    for path in out.glob("results_seed_*.json"):
        m = RESULT_RE.match(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "episode_results" in data:
            for item in data.get("episode_results") or []:
                if "seed" in item:
                    seeds.add(int(item["seed"]))
            continue
        start, end = int(m.group(1)), int(m.group(2))
        seeds.update(range(start, end + 1))
    return seeds


def active_seed_ranges() -> dict[str, set[int]]:
    active_ids = {job["id"] for job in active_eval_jobs()}
    by_task: dict[str, set[int]] = {}
    if MAINTAINER_LOG.exists():
        text = MAINTAINER_LOG.read_text(errors="replace")
        for match in SUBMIT_RE.finditer(text):
            job_id, task, start, end, _part = match.groups()
            if job_id in active_ids:
                by_task.setdefault(task, set()).update(range(int(start), int(end) + 1))
    for job_id in active_ids:
        path = REPO_ROOT / "job_logs" / f"act_eval_{job_id}.log"
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            m = LOG_RUN_RE.search(line)
            if not m:
                continue
            argv = m.group(1).split()
            task = None
            start = None
            episodes = None
            for i, token in enumerate(argv):
                if token == "--task" and i + 1 < len(argv):
                    task = argv[i + 1]
                elif token == "--start-seed" and i + 1 < len(argv):
                    start = int(argv[i + 1])
                elif token == "--episodes" and i + 1 < len(argv):
                    episodes = int(argv[i + 1])
            if task is not None and start is not None and episodes is not None:
                by_task.setdefault(task, set()).update(range(start, start + episodes))
    return by_task


def update_table_from_monitor() -> None:
    run(["python", "scripts/monitor_act_eval_success.py"], check=False)
    if not MONITOR_PATH.exists() or not TABLE_PATH.exists():
        return
    rows = {}
    in_table = False
    for line in MONITOR_PATH.read_text().splitlines():
        if line.startswith("| Job | Task |"):
            in_table = True
            continue
        if in_table and (not line.startswith("|") or line.startswith("|---")):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if len(cells) < 5 or cells[1] == "-":
                continue
            task = cells[1]
            completed = cells[2]
            rate = cells[4].replace("%", "")
            try:
                done, total = (int(x) for x in completed.split("/", 1))
            except ValueError:
                done = total = 0
            value = rate if done >= 100 or done >= total else f"{rate} (running)"
            rows[task] = value

    rows.update({task: value for task, value in completed_eval_rows().items() if task not in rows})

    lines = TABLE_PATH.read_text().splitlines()
    main_headers = lines[1].split("\t")[1:]
    new_lines = []
    base_section = False
    for idx, line in enumerate(lines):
        parts = line.split("\t")
        if idx == 6:
            base_section = True
        if parts and parts[0] == "ACT" and not base_section:
            vals = {h: "" for h in main_headers}
            vals.update(rows)
            new_lines.append("\t".join(["ACT"] + [vals[h] for h in main_headers]))
        elif parts and parts[0] == "ACT" and base_section:
            base_headers = lines[6].split("\t")[1:]
            vals = {}
            for header in base_headers:
                base_task = BASE_TABLE_TASKS.get(header, header)
                vals[header] = rows.get(base_task, "")
            new_lines.append("\t".join(["ACT"] + [vals[h] for h in base_headers]))
        else:
            new_lines.append(line)
    TABLE_PATH.write_text("\n".join(new_lines) + "\n")


def completed_eval_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for out in sorted(EVAL_ROOT.glob("*_full300_ckpt025000_*")):
        parts = out.name.split("_evalv1", 1)
        if len(parts) != 2:
            continue
        task = parts[0]
        completed = 0
        successes = 0
        for path in out.glob("results_seed_*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            eps = data.get("episode_results") or []
            completed += len(eps)
            successes += sum(1 for item in eps if bool(item.get("success")))
        if not completed:
            continue
        rate = 100.0 * successes / completed
        value = f"{rate:.1f}" if completed >= MAX_SEEDS else f"{rate:.1f} (running)"
        rows[task] = value
    return rows


def submit_next_shards() -> list[str]:
    jobs = active_eval_jobs()
    free = max(0, MAX_EVAL_JOBS - len(jobs))
    if free <= 0:
        return []
    active = active_seed_ranges()
    submitted = []
    used_ports = len(jobs) * PROCS_PER_JOB
    for task in table_tasks():
        ckpt = checkpoint_for(task)
        if not ckpt.is_dir():
            continue
        out = latest_or_new_output(task)
        scheduled = completed_seed_ranges(out) | active.get(task, set())
        if len(scheduled) >= MAX_SEEDS:
            continue
        for start in range(0, MAX_SEEDS, SHARD_SIZE):
            seeds = set(range(start, start + SHARD_SIZE))
            if seeds & scheduled:
                continue
            part = "gpu" if len([j for j in jobs if j["partition"] == "gpu"]) < 5 else "gpu-preempt"
            port = str(PORT_BASE + used_ports)
            eval_steps = eval_steps_for(task)
            cmd = [
                "sbatch", "--parsable", "-p", part, EVAL_SCRIPT,
                task, str(ckpt), port, str(eval_steps), str(EPISODES_PER_PROC),
                str(start), str(out), str(PROCS_PER_JOB),
                *eval_args_for(task),
            ]
            proc = run(cmd)
            job_id = proc.stdout.strip()
            submitted.append(f"{job_id}:{task}:{start}-{start + SHARD_SIZE - 1}:{part}")
            jobs.append({"id": job_id, "partition": part, "state": "PENDING"})
            active.setdefault(task, set()).update(seeds)
            used_ports += PROCS_PER_JOB
            free -= 1
            break
        if free <= 0:
            break
    return submitted


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    update_table_from_monitor()
    submitted = submit_next_shards()
    update_table_from_monitor()
    if submitted:
        print("submitted", " ".join(submitted))
    else:
        print("submitted none")


if __name__ == "__main__":
    main()
