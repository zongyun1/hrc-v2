"""Capture an eval-side observation and diff it against the training image.

Builds the task with the SAME config eval_vla.py uses (vla_head_camera,
planner_override, action_substeps, randomize_* defaults) and dumps the
head_camera + right_wrist RGB at reset. Then loads the matching training
steps.h5 frame 0 for the same seed. Saves a side-by-side for visual check.
"""
from __future__ import annotations
import sys, io, json
import numpy as np
sys.path.insert(0, ".")
from PIL import Image
from envs.tasks import resolve_task_class
from scripts.vla_client import get_task_instruction

TASK = sys.argv[1] if len(sys.argv) > 1 else "dump_bin_interrupt"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
RANDO_TABLE = (sys.argv[3] == "rt") if len(sys.argv) > 3 else True

name, TaskClass = resolve_task_class(TASK, no_human=False)
cfg = {
    "vla_head_camera": True,
    "randomize_avatar": True,
    "randomize_table": RANDO_TABLE,
    "track_avatar_collision": True,
    "action_substeps": 25,
    "planner_override": "genesis_ik",
    "instruction": get_task_instruction(name),
}
task = TaskClass(cfg)
obs = task.reset(seed=SEED)

def to_uint8(img):
    a = np.asarray(img)
    if a.ndim == 4: a = a[0]
    if a.dtype != np.uint8:
        a = np.clip(a*255 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
    return a[..., :3]

eval_head = to_uint8(obs["rgb"]["head_camera"])
eval_wrist = to_uint8(obs["rgb"]["right_wrist"]) if "right_wrist" in obs["rgb"] else None
print("eval head:", eval_head.shape, "cams:", list(obs["rgb"].keys()))

# training image frame 0 for same seed
import h5py, glob
trp = f"data/hri_manicast/vla_data_retrain/{TASK}/rasterizer/seed_{SEED}/steps.h5"
train_head = None
try:
    with h5py.File(trp, "r") as f:
        buf = bytes(f["images/head_camera"][0])
        train_head = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
        print("train head:", train_head.shape)
except Exception as e:
    print("no training image:", e)

# build side-by-side at common height
def rs(a, h=256):
    import cv2
    return cv2.resize(a, (int(a.shape[1]*h/a.shape[0]), h))
panels = [rs(eval_head)]
if train_head is not None: panels.insert(0, rs(train_head))
if eval_wrist is not None: panels.append(rs(eval_wrist))
strip = np.concatenate(panels, axis=1)
Image.fromarray(strip).save("/tmp/obs_compare.jpg")
print("saved /tmp/obs_compare.jpg  (train | eval_head | eval_wrist)")

# numeric: mean abs diff between train and eval head (resized to match)
if train_head is not None:
    import cv2
    e = cv2.resize(eval_head, (train_head.shape[1], train_head.shape[0]))
    print("train-vs-eval head MAD:", float(np.mean(np.abs(e.astype(float)-train_head.astype(float)))))
