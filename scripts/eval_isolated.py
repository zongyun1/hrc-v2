"""Run scripts/eval.py one episode per subprocess.

Why: the NYX renderer's native singletons/material assets are never torn
down between in-process scene rebuilds (each ``reset()`` calls
``gs.init()``/rebuilds a fresh ``NyxCameraManager`` without releasing the
previous one's renderer -- see envs/camera.py). Looping ``--episodes N>1``
in one process reliably segfaults around episode 3 once NYX's own internal
leak-detector trips. Every single-episode run (``--episodes 1``, the same
path collect.py uses) has been reliable, so this wrapper gets identical
per-episode behavior/output by giving each episode a fresh process -- which
guarantees clean native renderer state every time, at the cost of repeating
each episode's scene-build overhead (which already happens on every
in-process reset() anyway).

Usage: same flags as scripts/eval.py. ``--episodes``/``--start-seed``
control how many isolated subprocess invocations run; every other flag
(``--task``, ``--policy``, ``--renderer``, ``--video-dir``, ...) passes
through unchanged to each ``eval.py --episodes 1`` call.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PY = os.path.join(REPO_ROOT, "scripts", "eval.py")


def main():
    parser = argparse.ArgumentParser(
        description="Run scripts/eval.py one episode per subprocess (avoids the "
                     "NYX renderer's cross-episode native leak/segfault)."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None,
                         help="If set, write the combined aggregate + per-episode "
                              "results here (same schema as eval.py --output-json).")
    parser.add_argument("--python", type=str, default=sys.executable,
                         help="Python interpreter to invoke eval.py with.")
    args, passthrough = parser.parse_known_args()

    episode_results = []
    success_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(args.episodes):
            seed = args.start_seed + i
            ep_json = os.path.join(tmpdir, f"episode_{i}.json")
            cmd = [
                args.python, EVAL_PY,
                *passthrough,
                "--episodes", "1",
                "--start-seed", str(seed),
                "--output-json", ep_json,
            ]
            print(f"\n=== isolated episode {i + 1}/{args.episodes} (seed={seed}) ===",
                  flush=True)
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"  subprocess exited with code {result.returncode} "
                      f"(seed={seed}) -- treating episode as failed", flush=True)
                episode_results.append({
                    "seed": seed,
                    "success": False,
                    "subprocess_returncode": result.returncode,
                })
                continue

            if not os.path.exists(ep_json):
                print(f"  WARNING: {ep_json} not written -- treating episode as failed",
                      flush=True)
                episode_results.append({"seed": seed, "success": False})
                continue

            with open(ep_json) as f:
                data = json.load(f)
            ep_detail_list = data.get("episode_results") or []
            if ep_detail_list:
                detail = ep_detail_list[0]
                episode_results.append(detail)
                if detail.get("success"):
                    success_count += 1
            else:
                episode_results.append({"seed": seed, "success": False})

    success_rate = success_count / args.episodes if args.episodes > 0 else 0.0
    print(f"\nSuccess rate: {success_count}/{args.episodes} = {success_rate:.1%}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "success_rate": success_rate,
                    "successes": success_count,
                    "episodes": args.episodes,
                    "episode_results": episode_results,
                },
                f, indent=2, sort_keys=True,
            )
        print(f"Results JSON: {args.output_json}")


if __name__ == "__main__":
    main()
