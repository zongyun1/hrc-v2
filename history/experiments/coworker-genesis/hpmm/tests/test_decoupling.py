"""Invariant #2 of the plan: ``hpmm_motion`` must never import ``hpmm_sim`` or Genesis.

The human motion layer has to stay developable and verifiable on its own — it runs in
the TRUMANS venv, where Genesis is not even installed. This is a *static* scan rather
than an import test on purpose: it holds regardless of which venv runs the suite, and
it catches a violation buried in a branch that no test happens to execute.

Run with:  PYTHONPATH=hpmm python -m pytest hpmm/tests/ -q
"""

from __future__ import annotations

import ast
import pathlib

MOTION_PKG = pathlib.Path(__file__).resolve().parents[1] / "hpmm_motion"

#: Top-level module names the motion layer is not allowed to reach for.
FORBIDDEN = {"genesis", "hpmm_sim", "taichi", "mujoco", "robosuite"}


def _imported_roots(tree: ast.AST) -> set[str]:
    """Root module name of every import in the file, absolute imports only."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_motion_layer_does_not_import_simulator():
    violations = []
    for path in sorted(MOTION_PKG.rglob("*.py")):
        roots = _imported_roots(ast.parse(path.read_text(), filename=str(path)))
        for bad in sorted(roots & FORBIDDEN):
            violations.append(f"{path.relative_to(MOTION_PKG.parent)} imports {bad!r}")
    assert not violations, "hpmm_motion must stay simulator-agnostic:\n  " + "\n  ".join(violations)


def test_schema_imports_without_torch_or_genesis():
    """The contract itself must load anywhere — it is numpy-only by design."""
    from hpmm_motion.io import schema

    seq = schema.empty(2, 5, fps=30.0, gender="male").validate()
    assert seq.n_envs == 2 and seq.n_frames == 5
    assert len(schema.BODY_JOINT_NAMES) == schema.N_BODY_JOINTS + 1
    assert len(schema.BODY_JOINT_PARENTS) == schema.N_BODY_JOINTS + 1
