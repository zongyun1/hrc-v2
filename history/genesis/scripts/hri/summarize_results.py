"""Summarize HRI eval results.json files into one markdown table.

Usage:
  python scripts/hri/summarize_results.py data/hri_manicast/<batch_dir> ...

Walks the given roots for ``results.json`` produced by scripts/eval_vla.py
and prints per-run rows: task, model, condition (dir name), episodes,
success rate, generic-yield trigger stats, collision / safe-distance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    eps = d.get("episode_results") or []
    gp = [e.get("generic_pace") or {} for e in eps]
    fired = [bool(g.get("fired")) for g in gp]
    coll = []
    safe = []
    for e in eps:
        ac = e.get("avatar_collision") or {}
        if "any_collision" in ac:
            coll.append(bool(ac["any_collision"]))
        sd = e.get("safe_distance") or {}
        if sd.get("safe_distance_m") is not None:
            safe.append(float(sd["safe_distance_m"]))
    return {
        "dir": path.parent.name,
        "task": d.get("task"),
        "model": d.get("model"),
        "episodes": d.get("episodes", len(eps)),
        "success_rate": d.get("success_rate"),
        "trigger_rate": _mean([1.0 if f else 0.0 for f in fired]) if gp else None,
        "mean_trigger_step": _mean([g.get("trigger_step") for g in gp]),
        "mean_wait": _mean([g.get("wait_elapsed") for g in gp if g.get("fired")]),
        "return_conv": _mean([
            1.0 if g.get("return_converged") else 0.0
            for g in gp if g.get("fired")]),
        "collision_rate": _mean([1.0 if c else 0.0 for c in coll]),
        "min_safe_dist": min(safe) if safe else None,
    }


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("data/hri_manicast")]
    rows = []
    for root in roots:
        for p in sorted(root.rglob("results.json")):
            r = summarize(p)
            if r:
                rows.append(r)
    if not rows:
        print("no results found")
        return

    def fmt(v, pct=False):
        if v is None:
            return "—"
        if pct:
            return f"{v:.0%}"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    cols = ["dir", "task", "model", "episodes", "success_rate",
            "trigger_rate", "mean_trigger_step", "mean_wait",
            "return_conv", "collision_rate", "min_safe_dist"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(
            fmt(r[c], pct=c in ("success_rate", "trigger_rate",
                                "return_conv", "collision_rate"))
            for c in cols) + " |")


if __name__ == "__main__":
    main()
