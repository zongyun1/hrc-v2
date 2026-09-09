"""Compare pour_water cup/mug reset behavior in old vs latest Genesis.

This intentionally uses the regular task path, not the NYX render-only wrapper:
no virtual mug, no USE_SETQPOS, and no static cup unless supplied by config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_OLD_PY = Path(
    os.environ.get(
        "OLD_GENESIS_PY",
        REPO.parent / "genesis-hr-bench" / ".venv-bench" / "bin" / "python",
    )
)
DEFAULT_LATEST_PY = Path(
    os.environ.get(
        "LATEST_GENESIS_PY",
        REPO / ".venv-genesis-latest" / "bin" / "python",
    )
)
DEFAULT_OUTPUT = REPO / "outputs" / "pour_water_old_latest_reset_compare" / "reset_compare.json"


def configured_env() -> dict[str, str]:
    env = dict(os.environ)
    tmp = Path(env.get("TMPDIR", "/tmp"))
    env.setdefault("GENESIS_BACKEND", "cpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "1")
    env.setdefault("NUMBA_CACHE_DIR", str(tmp / "genesis_hr_bench_numba_cache"))
    env.setdefault("MPLCONFIGDIR", str(tmp / "genesis_hr_bench_mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(tmp / "genesis_hr_bench_xdg_cache"))
    env.setdefault("GS_CACHE_FILE_PATH", str(tmp / "genesis_hr_bench_gs_cache"))
    env.pop("USE_SETQPOS", None)
    env.pop("DEBUG_TELEPORT", None)
    return env


def _jsonify(x):
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, (float, int, str, bool)):
        return x
    try:
        return float(x)
    except Exception:
        return str(x)


def child_main(args) -> int:
    started = time.monotonic()
    report = {
        "label": args.label,
        "python": sys.executable,
        "repo": str(REPO),
        "seed": args.seed,
        "extra_steps": args.extra_steps,
    }
    try:
        import importlib.metadata as md
        import numpy as np
        import transforms3d as t3d

        from envs.genesis_compat import configure_genesis_runtime

        configure_genesis_runtime()
        import genesis as gs
        from envs.tasks import TASK_MAP
        from envs.utils import to_numpy

        versions = {}
        for name in ("genesis-world", "genesis", "vico", "gs-nyx", "gs-nyx-plugin"):
            try:
                versions[name] = md.version(name)
            except Exception:
                versions[name] = None
        report["versions"] = versions
        report["genesis_file"] = getattr(gs, "__file__", None)

        def unwrap(obj):
            return getattr(obj, "entity", obj)

        def safe(fn):
            try:
                return _jsonify(fn())
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        def record_entity(entity, initial_quat=None):
            entity = unwrap(entity)
            pos = safe(lambda: to_numpy(entity.get_pos()).ravel()[:3])
            quat = safe(lambda: to_numpy(entity.get_quat()).ravel()[:4])
            aabb = safe(lambda: to_numpy(entity.get_AABB()))
            out = {
                "type": type(entity).__name__,
                "pos": pos,
                "quat": quat,
                "aabb": aabb,
                "qpos": safe(lambda: to_numpy(entity.get_qpos()).ravel()),
            }
            if isinstance(quat, list) and len(quat) == 4:
                R = t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64))
                out["world_x_axis"] = _jsonify(R[:, 0])
                out["world_y_axis"] = _jsonify(R[:, 1])
                out["world_z_axis"] = _jsonify(R[:, 2])
                if initial_quat is not None:
                    R0 = t3d.quaternions.quat2mat(np.asarray(initial_quat, dtype=np.float64))
                    dot = float(np.clip((np.trace(R0.T @ R) - 1.0) / 2.0, -1.0, 1.0))
                    out["angle_from_spawn_deg"] = float(np.degrees(np.arccos(dot)))
            if isinstance(aabb, list) and len(aabb) == 2:
                lo = np.asarray(aabb[0], dtype=np.float64)
                hi = np.asarray(aabb[1], dtype=np.float64)
                ext = hi - lo
                out["aabb_extent"] = _jsonify(ext)
                out["aabb_center"] = _jsonify((lo + hi) * 0.5)
                out["bottom_z"] = float(lo[2])
                out["top_z"] = float(hi[2])
            return out

        def record_task(task, label):
            cup_entity = task.cup_actor.entity if getattr(task, "cup_actor", None) is not None else None
            mug_entity = getattr(task, "mug_entity", None)
            return {
                "label": label,
                "table_top_z": float(getattr(task, "TABLE_TOP_Z", float("nan"))),
                "cup_spawn_pose": _jsonify(getattr(task, "cup_pose", None).to_pose7()),
                "mug_spawn_pose": _jsonify(getattr(task, "mug_pose", None).to_pose7()),
                "mug_physics_spawn_pose": _jsonify(
                    getattr(task, "_mug_physics_pose", getattr(task, "mug_pose", None)).to_pose7()
                ),
                "cup": record_entity(cup_entity, getattr(task, "cup_pose", None).q) if cup_entity is not None else None,
                "mug": record_entity(
                    mug_entity,
                    getattr(task, "_mug_physics_pose", getattr(task, "mug_pose", None)).q,
                ) if mug_entity is not None else None,
            }

        cfg = {
            "no_avatar": True,
            "skip_reset_obs": True,
            "side_video": False,
            "renderer": "rasterizer",
        }
        if not args.no_kinematic_mug_grasp:
            cfg["kinematic_mug_grasp"] = True
        if args.stable_tableware_spawn:
            cfg["stable_tableware_spawn"] = True
        if args.static_cup:
            cfg["static_cup"] = True
        Task = TASK_MAP["pour_water"]
        task = Task(cfg)
        report["config"] = cfg
        task.reset(seed=args.seed)
        report["after_reset"] = record_task(task, "after_reset")
        for _ in range(int(args.extra_steps)):
            task.step_sim()
        report["after_extra_settle"] = record_task(task, "after_extra_settle")

    except Exception:
        report["child_error"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1

    report["elapsed_sec"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def summarize(report: dict) -> str:
    lines = ["pour_water reset/settle comparison"]
    for label, run in report["runs"].items():
        versions = run.get("versions", {})
        lines.append(
            f"- {label}: genesis-world={versions.get('genesis-world')} "
            f"genesis={versions.get('genesis')} python={run.get('python')}"
        )
    for stage in ("after_reset", "after_extra_settle"):
        lines.append(f"\n{stage}")
        for label, run in report["runs"].items():
            snap = run.get(stage, {})
            for name in ("cup", "mug"):
                ent = snap.get(name) or {}
                pos = ent.get("pos")
                ext = ent.get("aabb_extent")
                angle = ent.get("angle_from_spawn_deg")
                bottom = ent.get("bottom_z")
                lines.append(
                    f"- {label} {name}: pos={pos} extent={ext} "
                    f"bottom_z={bottom} angle_from_spawn_deg={angle}"
                )
    return "\n".join(lines)


def parent_main(args) -> int:
    runs = {}
    failures = []
    for label, py in (("old", args.old_python), ("latest", args.latest_python)):
        cmd = [
            str(py),
            str(Path(__file__).resolve()),
            "--child",
            "--label",
            label,
            "--seed",
            str(args.seed),
            "--extra-steps",
            str(args.extra_steps),
        ]
        if args.stable_tableware_spawn:
            cmd.append("--stable-tableware-spawn")
        if args.static_cup:
            cmd.append("--static-cup")
        if args.no_kinematic_mug_grasp:
            cmd.append("--no-kinematic-mug-grasp")
        print(f"=== {label} ===", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=configured_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        print(proc.stdout, flush=True)
        try:
            runs[label] = json.loads(proc.stdout[proc.stdout.find("{"):])
        except Exception:
            runs[label] = {"parse_error": proc.stdout}
        if proc.returncode != 0:
            failures.append(label)

    report = {
        "old_python": str(args.old_python),
        "latest_python": str(args.latest_python),
        "seed": args.seed,
        "extra_steps": args.extra_steps,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    summary = summarize(report)
    summary_path = args.output.with_suffix(".summary.txt")
    summary_path.write_text(summary + "\n")
    print(summary, flush=True)
    print(f"WROTE_JSON {args.output}", flush=True)
    print(f"WROTE_SUMMARY {summary_path}", flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-python", type=Path, default=DEFAULT_OLD_PY)
    parser.add_argument("--latest-python", type=Path, default=DEFAULT_LATEST_PY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-steps", type=int, default=200)
    parser.add_argument("--stable-tableware-spawn", action="store_true")
    parser.add_argument("--static-cup", action="store_true")
    parser.add_argument("--no-kinematic-mug-grasp", action="store_true")
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--label", default="child")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
