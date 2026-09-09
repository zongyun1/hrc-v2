"""Compare asset loading between legacy Genesis and latest Genesis.

The parent mode runs this script in two Python environments and writes a
side-by-side JSON report. Child mode imports Genesis in the selected
environment, loads a small asset set, and records link/joint/qpos/AABB data plus
short-step stability.
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
DEFAULT_OUTPUT = REPO / "outputs" / "latest_genesis_asset_compare" / "old_vs_latest_asset_loading.json"

URDF_ASSETS = {
    "036_cabinet_46653_original_urdf": {
        "path": REPO / "assets" / "objects" / "036_cabinet" / "46653" / "mobility.urdf",
        "pos": (0.30, -0.65, 0.927),
        "quat": (0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
        "scale": 0.20,
    },
    "044_microwave_7167_original_urdf": {
        "path": REPO / "assets" / "objects" / "044_microwave" / "7167" / "mobility.urdf",
        "pos": (0.0, 0.0, 0.85),
        "quat": (1.0, 0.0, 0.0, 0.0),
        "scale": 0.20,
    },
    "015_laptop_10040_original_urdf": {
        "path": REPO / "assets" / "objects" / "015_laptop" / "10040" / "mobility.urdf",
        "pos": (0.0, 0.0, 0.85),
        "quat": (1.0, 0.0, 0.0, 0.0),
        "scale": 0.20,
    },
}


def configured_env() -> dict[str, str]:
    env = dict(os.environ)
    tmp = Path(env.get("TMPDIR", "/tmp"))
    env.setdefault("GENESIS_BACKEND", "cpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "1")
    env.setdefault("NUMBA_CACHE_DIR", str(tmp / "genesis_hr_bench_numba_cache"))
    env.setdefault("MPLCONFIGDIR", str(tmp / "genesis_hr_bench_mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(tmp / "genesis_hr_bench_xdg_cache"))
    env.setdefault("GS_CACHE_FILE_PATH", str(tmp / "genesis_hr_bench_gs_cache"))
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
    report: dict = {
        "label": args.label,
        "python": sys.executable,
        "repo": str(REPO),
        "assets": {},
    }
    try:
        import importlib.metadata as md
        import numpy as np

        from envs.genesis_compat import configure_genesis_runtime

        configure_genesis_runtime()
        import genesis as gs

        report["genesis_file"] = getattr(gs, "__file__", None)
        versions = {}
        for name in ("genesis-world", "genesis", "vico", "gs-nyx", "gs-nyx-plugin"):
            try:
                versions[name] = md.version(name)
            except Exception:
                versions[name] = None
        report["versions"] = versions

        try:
            gs.init(backend=gs.cpu, logging_level="error")
            report["init"] = "ok"
        except Exception as exc:
            report["init"] = f"already_or_failed:{type(exc).__name__}:{exc}"

        def to_numpy(x):
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()
            return np.asarray(x)

        def safe_call(fn):
            try:
                return _jsonify(fn())
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        def unwrap_entity(obj):
            return getattr(obj, "entity", obj)

        def get_aabb(entity):
            entity = unwrap_entity(entity)
            return to_numpy(entity.get_AABB()).astype(float)

        def add_plane(scene):
            try:
                scene.add_entity(gs.morphs.Plane())
            except Exception:
                pass

        def new_scene():
            kwargs = {
                "show_viewer": False,
                "sim_options": gs.options.SimOptions(dt=0.002),
            }
            try:
                kwargs["renderer"] = gs.renderers.Rasterizer()
            except Exception:
                pass
            return gs.Scene(**kwargs)

        def record_steps(scene, entity, max_steps: int):
            entity = unwrap_entity(entity)
            steps = []
            for step in range(1, max_steps + 1):
                try:
                    scene.step()
                    steps.append(
                        {
                            "step": step,
                            "status": "ok",
                            "qpos": safe_call(lambda: to_numpy(entity.get_qpos()).ravel()),
                            "aabb": safe_call(lambda: get_aabb(entity)),
                        }
                    )
                except Exception as exc:
                    steps.append(
                        {
                            "step": step,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    break
            return steps

        def record_entity_common(entity):
            entity = unwrap_entity(entity)
            data = {
                "entity_type": type(entity).__name__,
                "aabb": safe_call(lambda: get_aabb(entity)),
                "pos": safe_call(lambda: to_numpy(entity.get_pos()).ravel()[:3]),
                "quat": safe_call(lambda: to_numpy(entity.get_quat()).ravel()[:4]),
                "qpos": safe_call(lambda: to_numpy(entity.get_qpos()).ravel()),
                "dof_limits": safe_call(lambda: entity.get_dofs_limit()),
            }
            links = []
            for link in getattr(entity, "links", []):
                links.append(
                    {
                        "name": str(getattr(link, "name", "")),
                        "pos": safe_call(lambda link=link: to_numpy(link.get_pos()).ravel()[:3]),
                        "quat": safe_call(lambda link=link: to_numpy(link.get_quat()).ravel()[:4]),
                        "aabb": safe_call(lambda link=link: get_aabb(link)),
                    }
                )
            data["links"] = links
            joints = []
            for joint in getattr(entity, "joints", []):
                item = {"name": str(getattr(joint, "name", ""))}
                for attr in ("type", "dof_idx_local", "dof_start", "n_dofs"):
                    if hasattr(joint, attr):
                        item[attr] = _jsonify(getattr(joint, attr))
                joints.append(item)
            data["joints"] = joints
            return data

        def load_urdf_asset(spec: dict):
            urdf = Path(spec["path"]).resolve()
            scene = new_scene()
            add_plane(scene)
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf),
                    pos=tuple(spec["pos"]),
                    quat=tuple(spec["quat"]),
                    scale=float(spec["scale"]),
                    fixed=True,
                )
            )
            scene.build()
            data = record_entity_common(entity)
            data["build_status"] = "ok"
            data["steps"] = record_steps(scene, entity, args.steps)
            return data

        def load_mesh_asset(name: str, model_id: int):
            from envs.utils import Pose, load_object

            scene = new_scene()
            add_plane(scene)
            entity = load_object(
                scene,
                Pose([0.0, 0.0, 0.85], [1.0, 0.0, 0.0, 0.0]),
                name,
                model_id=model_id,
                convex=True,
                is_static=False,
                friction=4.0,
            )
            scene.build()
            data = record_entity_common(entity)
            data["build_status"] = "ok"
            data["steps"] = record_steps(scene, entity, min(args.steps, 15))
            return data

        loaders = {
            "021_cup_model0": lambda: load_mesh_asset("021_cup", 0),
            "039_mug_model0": lambda: load_mesh_asset("039_mug", 0),
            "039_mug_model8": lambda: load_mesh_asset("039_mug", 8),
            "048_stapler_model0": lambda: load_mesh_asset("048_stapler", 0),
            "023_tissue_box_model0": lambda: load_mesh_asset("023_tissue-box", 0),
        }
        for asset_name, spec in URDF_ASSETS.items():
            loaders[asset_name] = lambda spec=spec: load_urdf_asset(spec)
        for name, loader in loaders.items():
            asset_started = time.monotonic()
            try:
                data = loader()
                data["elapsed_sec"] = round(time.monotonic() - asset_started, 3)
                report["assets"][name] = data
            except Exception:
                report["assets"][name] = {
                    "build_status": "failed",
                    "error": traceback.format_exc(),
                    "elapsed_sec": round(time.monotonic() - asset_started, 3),
                }

    except Exception:
        report["child_error"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1

    report["elapsed_sec"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def summarize(report: dict) -> str:
    lines = []
    labels = list(report["runs"].keys())
    lines.append("Asset comparison summary")
    for label in labels:
        run = report["runs"][label]
        versions = run.get("versions", {})
        lines.append(f"- {label}: python={run.get('python')} genesis-world={versions.get('genesis-world')} genesis={versions.get('genesis')}")
    asset_names = sorted({name for run in report["runs"].values() for name in run.get("assets", {})})
    for asset in asset_names:
        lines.append(f"\n{asset}")
        for label in labels:
            data = report["runs"][label].get("assets", {}).get(asset, {})
            status = data.get("build_status")
            step_status = "n/a"
            steps = data.get("steps") or []
            if steps:
                failed = next((s for s in steps if s.get("status") == "failed"), None)
                if failed:
                    step_status = f"failed_step_{failed.get('step')}: {failed.get('error')}"
                else:
                    step_status = f"ok_{len(steps)}_steps"
            links = len(data.get("links") or [])
            qpos = data.get("qpos")
            lines.append(f"- {label}: build={status} links={links} qpos={qpos} steps={step_status}")
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
            "--steps",
            str(args.steps),
        ]
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
        "steps": args.steps,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    summary_path = args.output.with_suffix(".summary.txt")
    summary = summarize(report)
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
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=240.0)
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
