"""Probe robot-only task reset/step behavior under latest Genesis.

The direct robot probe verifies wrappers in isolation. This script exercises the
task layer: task construction, scene setup, object loading, reset, robot joint
initialization, settle steps, and one post-reset scene step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


DEFAULT_TASKS = [
    "simple_cube_pick_place",
    "put_object_cabinet",
    "stack_bowls_three",
    "place_bread_in_basket",
    "dump_bin",
    "place_burger_fries",
    "place_dual_shoes",
    "place_food_in_skillet",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "dump_bin_xarm_calibrated",
]


def configure_env(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    tmp = Path(env.get("TMPDIR", "/tmp"))
    env.setdefault("GENESIS_BACKEND", "cpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "1")
    env.setdefault("NUMBA_CACHE_DIR", str(tmp / "genesis_hr_bench_numba_cache"))
    env.setdefault("MPLCONFIGDIR", str(tmp / "genesis_hr_bench_mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(tmp / "genesis_hr_bench_xdg_cache"))
    return env


def child_probe(task_name: str, seed: int, steps: int, settle_steps: int | None) -> None:
    from envs.genesis_compat import configure_genesis_runtime

    configure_genesis_runtime()

    from envs.tasks import resolve_task_class

    started = time.monotonic()
    resolved, task_cls = resolve_task_class(task_name)
    cfg = {
        "skip_reset_obs": True,
        "show_viewer": False,
        "side_video": False,
        "use_avatar": False,
    }
    task = task_cls(cfg)
    if settle_steps is not None:
        task.SETTLE_STEPS = int(settle_steps)
    obs = task.reset(seed=seed)
    for _ in range(max(0, int(steps))):
        task.scene.step()
    robot = getattr(task, "robot", None)
    arm = robot.get_arm("right") if robot is not None else None
    qpos_shape = None
    ee_xyz = None
    if arm is not None:
        qpos = arm.get_arm_qpos()
        qpos_shape = tuple(getattr(qpos, "shape", ()))
        ee_xyz = [round(float(x), 5) for x in arm.get_ee_pose()[:3]]
    elapsed = time.monotonic() - started
    print(
        f"RESULT task={task_name} resolved={resolved} class={task_cls.__name__} "
        f"obs={type(obs).__name__} qpos_shape={qpos_shape} ee_xyz={ee_xyz} "
        f"settle_steps={task.SETTLE_STEPS} elapsed={elapsed:.2f}s",
        flush=True,
    )


def run_parent(args) -> int:
    failures = []
    for task_name in args.tasks:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--task",
            task_name,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
        ]
        if args.settle_steps is not None:
            cmd.extend(["--settle-steps", str(args.settle_steps)])
        print(f"=== {task_name} ===", flush=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                env=configure_env(os.environ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append(task_name)
            print(f"TIMEOUT task={task_name} timeout={args.timeout}s", flush=True)
            if exc.stdout:
                print(exc.stdout, flush=True)
            continue
        elapsed = time.monotonic() - started
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
        if proc.returncode != 0:
            failures.append(task_name)
            print(f"FAIL task={task_name} returncode={proc.returncode} elapsed={elapsed:.2f}s", flush=True)
        else:
            print(f"PASS task={task_name} elapsed={elapsed:.2f}s", flush=True)

    passed = len(args.tasks) - len(failures)
    print(f"SUMMARY passed={passed} failed={len(failures)} total={len(args.tasks)}", flush=True)
    if failures:
        print(f"FAILED_TASKS {' '.join(failures)}", flush=True)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--settle-steps", type=int)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()

    if args.child:
        if not args.task:
            raise SystemExit("--child requires --task")
        try:
            child_probe(args.task, seed=args.seed, steps=args.steps, settle_steps=args.settle_steps)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
