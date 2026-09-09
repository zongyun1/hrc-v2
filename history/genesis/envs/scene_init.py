"""Capture and replay deterministic task initialization specs.

Scene-init specs are an eval-facing layer on top of task reset.  A source
seed may still be stored to rebuild task topology, but the reviewable eval
definition names concrete table/avatar/object state instead of relying on
the seed as the only contract.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .utils import ASSETS_PATH, Actor, ArticulationActor, to_numpy


SCHEMA_VERSION = "genesis_hr_bench.scene_init.v1"

_EXCLUDED_ROOT_ATTRS = {
    "avatar",
    "avatar_collider",
    "avatar_collision_checker",
    "avatar_retreat_checker",
    "avatar_safety_checker",
    "cameras",
    "config",
    "floor",
    "robot",
    "scene",
    "table",
    "table_variant_entity",
    "target",
    "vla_recorder",
}

_EXCLUDED_STATE_ATTRS = _EXCLUDED_ROOT_ATTRS | {
    "left_joint_path",
    "right_joint_path",
    "traj_data",
    "avatar_collision_log",
    "avatar_safety_log",
    "avatar_safety_intervention_log",
}

_INFRA_PREFIXES = (
    "_avatar_collision",
    "_avatar_retreat",
    "_gripper_attached",
    "_record",
    "_video",
)

_MAX_ENTITY_CONTAINER_ITEMS = 64


def jsonable(value: Any) -> Any:
    """Convert numpy/scalar containers into JSON/YAML-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def _simple_state(value: Any, *, depth: int = 0) -> Any:
    if depth > 2:
        raise TypeError
    if isinstance(value, np.ndarray):
        if value.size > 128:
            raise TypeError
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise TypeError
        return [_simple_state(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        if len(value) > 64:
            raise TypeError
        out = {}
        for k, v in value.items():
            if isinstance(k, (str, int, float, bool)):
                out[str(k)] = _simple_state(v, depth=depth + 1)
            else:
                raise TypeError
        return out
    raise TypeError


def _entity_from_value(value: Any):
    if isinstance(value, (Actor, ArticulationActor)):
        return value.entity, getattr(value, "name", None)
    if hasattr(value, "get_pos") and hasattr(value, "get_quat"):
        return value, getattr(value, "name", None)
    return None, None


def _entity_pose(entity) -> dict:
    pos = to_numpy(entity.get_pos()).ravel()[:3].astype(float).tolist()
    quat = to_numpy(entity.get_quat()).ravel()[:4].astype(float).tolist()
    out = {"pose": [*pos, *quat]}
    try:
        qpos = to_numpy(entity.get_qpos()).ravel().astype(float).tolist()
        if qpos:
            out["qpos"] = qpos
    except Exception:
        pass
    return out


def _iter_named_entities(task):
    seen: set[int] = set()

    def rec(path: str, value: Any, depth: int):
        entity, actor_name = _entity_from_value(value)
        if entity is not None:
            ident = id(entity)
            if ident not in seen:
                seen.add(ident)
                record = _entity_pose(entity)
                if actor_name is not None:
                    record["actor_name"] = str(actor_name)
                yield path, record
            return
        if depth >= 2:
            return
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_ENTITY_CONTAINER_ITEMS:
                return
            for i, item in enumerate(value):
                yield from rec(f"{path}[{i}]", item, depth + 1)
        elif isinstance(value, dict):
            if len(value) > _MAX_ENTITY_CONTAINER_ITEMS:
                return
            for key, item in value.items():
                if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    yield from rec(f"{path}.{key}", item, depth + 1)

    for name, value in vars(task).items():
        if name.startswith("_") or name in _EXCLUDED_ROOT_ATTRS:
            continue
        yield from rec(name, value, 0)


def _capture_task_attrs(task) -> dict:
    attrs = {}
    for name, value in vars(task).items():
        if name in _EXCLUDED_STATE_ATTRS:
            continue
        if any(name.startswith(prefix) for prefix in _INFRA_PREFIXES):
            continue
        try:
            attrs[name] = _simple_state(value)
        except TypeError:
            continue
    return attrs


def capture_scene_init(task, *, episode_id: str | None = None, seed: int | None = None) -> dict:
    """Capture concrete eval initialization state from a reset task."""
    init = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if episode_id is not None:
        init["id"] = str(episode_id)
    if seed is not None:
        init["source_seed"] = int(seed)

    table = {
        "variant_name": getattr(task, "table_variant_name", None),
        "top_z": float(getattr(task, "TABLE_TOP_Z", 0.0)),
    }
    if getattr(task, "table", None) is not None:
        table.update(_entity_pose(task.table))
    init["table"] = jsonable(table)

    avatar = {}
    if hasattr(task, "avatar_init_pos"):
        avatar["init_pos"] = np.asarray(task.avatar_init_pos, dtype=float).ravel()[:3].tolist()
    if hasattr(task, "avatar_init_rot"):
        avatar["init_rot"] = np.asarray(task.avatar_init_rot, dtype=float).reshape(3, 3).tolist()
    skin = ((task.config or {}).get("avatar", {}) or {}).get("skin_options", {}) or {}
    glb = skin.get("glb_path")
    if getattr(task, "avatar", None) is not None:
        robot = getattr(task.avatar, "robot", None)
        skin_entity = getattr(robot, "skin", None)
        morph = getattr(skin_entity, "morph", None) if skin_entity is not None else None
        glb = getattr(morph, "file", glb)
    if glb:
        try:
            root = Path(__file__).resolve().parent.parent
            avatar["skin_glb_path"] = os.path.relpath(str(glb), root)
        except Exception:
            avatar["skin_glb_path"] = str(glb)
    if avatar:
        init["avatar"] = jsonable(avatar)

    entities = {}
    for path, record in _iter_named_entities(task):
        if path in {"table", "table_variant_entity"}:
            continue
        entities[path] = jsonable(record)
    init["entities"] = entities
    init["task_attrs"] = jsonable(_capture_task_attrs(task))
    return init


def prepare_config_for_scene_init(config: dict, scene_init: dict | None) -> dict:
    """Return a config copy with pre-build forced choices from scene_init."""
    cfg = deepcopy(config or {})
    if not scene_init:
        return cfg
    cfg["scene_init"] = deepcopy(scene_init)

    table = scene_init.get("table") or {}
    if table.get("variant_name"):
        cfg["randomize_table"] = True
        cfg["debug_table_randomization"] = True
        cfg["table_variant_name"] = str(table["variant_name"])

    avatar = scene_init.get("avatar") or {}
    if avatar.get("skin_glb_path"):
        avatar_cfg = dict(cfg.get("avatar", {}) or {})
        skin = dict(avatar_cfg.get("skin_options", {}) or {})
        glb_path = Path(str(avatar["skin_glb_path"]))
        if not glb_path.is_absolute():
            root = Path(__file__).resolve().parent.parent
            root_path = root / glb_path
            assets_path = ASSETS_PATH / glb_path
            if root_path.exists():
                glb_path = root_path
            elif assets_path.exists():
                glb_path = assets_path
        skin["glb_path"] = str(glb_path)
        avatar_cfg["skin_options"] = skin
        cfg["avatar"] = avatar_cfg
        cfg["randomize_avatar"] = False
    return cfg


def apply_scene_init_pre_avatar(task) -> None:
    scene_init = (task.config or {}).get("scene_init") or {}
    avatar = scene_init.get("avatar") or {}
    if "init_pos" in avatar:
        task.avatar_init_pos = np.asarray(avatar["init_pos"], dtype=np.float64)
    if "init_rot" in avatar:
        task.avatar_init_rot = np.asarray(avatar["init_rot"], dtype=np.float64).reshape(3, 3)


def _resolve_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", part)
        if match is None:
            raise KeyError(path)
        name, idx = match.group(1), match.group(2)
        if isinstance(current, dict):
            current = current[name]
        else:
            current = getattr(current, name)
        if idx is not None:
            current = current[int(idx)]
    return current


def _apply_entity_record(task, path: str, record: dict) -> bool:
    try:
        value = _resolve_path(task, path)
        entity, _actor_name = _entity_from_value(value)
    except Exception:
        return False
    if entity is None:
        return False
    pose = record.get("pose")
    if pose is not None:
        arr = np.asarray(pose, dtype=np.float64).ravel()
        if arr.size >= 3 and hasattr(entity, "set_pos"):
            entity.set_pos(arr[:3])
        if arr.size >= 7 and hasattr(entity, "set_quat"):
            entity.set_quat(arr[3:7])
    if "qpos" in record and hasattr(entity, "set_qpos"):
        try:
            entity.set_qpos(np.asarray(record["qpos"], dtype=np.float64))
        except Exception:
            pass
    return True


def apply_scene_init_post_settle(task) -> dict:
    """Replay entity states after reset settling and return replay diagnostics."""
    scene_init = (task.config or {}).get("scene_init") or {}
    if not scene_init:
        return {"enabled": False}

    applied_attrs = 0
    for name, value in (scene_init.get("task_attrs") or {}).items():
        if name in _EXCLUDED_STATE_ATTRS:
            continue
        try:
            current = getattr(task, name, None)
            restored = jsonable(value)
            if isinstance(current, np.ndarray):
                restored = np.asarray(restored, dtype=current.dtype)
            setattr(task, name, restored)
            applied_attrs += 1
        except Exception:
            pass

    applied_entities = []
    missing_entities = []
    for path, record in (scene_init.get("entities") or {}).items():
        if _apply_entity_record(task, path, record):
            applied_entities.append(path)
        else:
            missing_entities.append(path)

    table = scene_init.get("table") or {}
    if table.get("pose") and getattr(task, "table", None) is not None:
        _apply_entity_record(task, "table", table)
    if "top_z" in table:
        try:
            task.TABLE_TOP_Z = float(table["top_z"])
            task._table_top_z_fixed = True
        except Exception:
            pass

    if getattr(task, "avatar", None) is not None:
        avatar = scene_init.get("avatar") or {}
        if "init_pos" in avatar or "init_rot" in avatar:
            default_pos = getattr(task, "avatar_init_pos", [0.0, 0.0, 0.0])
            default_rot = getattr(task, "avatar_init_rot", np.eye(3, dtype=np.float64))
            pos = np.asarray(avatar.get("init_pos", default_pos), dtype=np.float64)
            rot = np.asarray(avatar.get("init_rot", default_rot), dtype=np.float64).reshape(3, 3)
            task.avatar_init_pos = pos
            task.avatar_init_rot = rot
            task.avatar.reset(pos, rot)
            if getattr(task, "avatar_collider", None) is not None:
                task.avatar_collider.update()

    return {
        "enabled": True,
        "id": scene_init.get("id"),
        "source_seed": scene_init.get("source_seed"),
        "applied_attrs": applied_attrs,
        "applied_entities": len(applied_entities),
        "missing_entities": missing_entities,
    }


def load_scene_init_file(path: str | os.PathLike) -> list[dict]:
    """Load a scene-init YAML/JSON/JSONL file and return episode entries."""
    p = Path(path)
    if p.suffix == ".jsonl":
        episodes = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if p.suffix in {".yaml", ".yml"} else json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "episodes" in data:
        return list(data["episodes"] or [])
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported scene-init file shape: {p}")


def episode_scene_init(entry: dict) -> dict:
    """Return the nested init spec from an episode entry."""
    if "init" in entry and isinstance(entry["init"], dict):
        init = deepcopy(entry["init"])
        init.setdefault("id", entry.get("id"))
        if "source_seed" not in init and "seed" in entry:
            init["source_seed"] = entry["seed"]
        return init
    return deepcopy(entry)
