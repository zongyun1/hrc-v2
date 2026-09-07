"""Diffusion Policy (Chi et al. 2023) model server. Run from the
diffusion_policy venv:

    /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/diffusion_policy/bin/python \
        baseline/diffusion_policy_server.py \
        --checkpoint runs/diffusion_policy/genesis_hr_bench/latest.ckpt \
        --port 8773

Then from MAWM env:
    python scripts/eval_vla.py --model diffusion_policy \
        --server-url http://127.0.0.1:8773 ...

Diffusion Policy is trained per-task, not a foundation model. The expected
checkpoint is a Hydra-saved workspace produced by `train.py` in the
`real-stanford/diffusion_policy` repo (`baseline/diffusion_policy/`). The
workspace embeds the policy class, normalizer, image/proprio shapes,
n_obs_steps, n_action_steps, and the dataset's action space — we just load
and sample.

Conventions:
  - `shape_meta.obs` keys are task-specific upstream (pusht uses
    `image`+`agent_pos`, robomimic uses `agentview_image`+`robot0_eef_pos`+
    ...). On startup we scan `cfg.policy.shape_meta.obs` and pick the first
    `rgb` entry as the image slot and the first `low_dim` entry as the
    proprio slot. Multi-image / multi-low_dim configs are not supported by
    this minimal server — extend `build_obs_dict` if needed.
  - `obs_dict` shape follows the policy's `predict_action` contract:
      <rgb_key>     : (B=1, To, C, H, W) float32 in [0,1]
      <low_dim_key> : (B=1, To, D_proprio) float32
    where `To = policy.n_obs_steps`. We have no history on the server, so
    we tile the current frame/proprio across the To window.
  - Output: the policy returns `{'action': (B, Ta, D_action)}`. We take the
    first step of the chunk and return it as a flat list. Action space
    matches whatever the trained checkpoint emits — for genesis_hr_bench
    Franka qpos data that's 8-D `[7 joint, 1 gripper]`; the client
    (DiffusionPolicyPolicy) passes through and uses `--action-type qpos`.

Notes:
  - The full Diffusion Policy stack pulls in `diffusers`, `hydra-core`,
    `omegaconf`, `dill`, `robomimic`, and `wandb`. These conflict with
    OpenVLA/Octo so it lives in its own env.
  - First-token latency is dominated by DDIM sampling steps (default 100
    in the original repo, often dropped to 16 at inference). Override via
    `--num-inference-steps`.
"""
import argparse
import base64
import io
import os
import sys
import time
import traceback
from collections import deque

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion_policy")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def decode_png(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))), dtype=np.uint8)


def load_workspace(checkpoint: str, device: str):
    """Reconstruct a Diffusion Policy workspace from a `.ckpt` saved by the
    upstream `train.py`. The workspace pickles its own class via dill; we
    rehydrate it, pull out the EMA policy if present (otherwise the live
    policy), and move to device.
    """
    import dill
    import hydra

    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = getattr(workspace, "ema_model", None) or workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def discover_obs_keys(shape_meta):
    """Return (rgb_key, image_shape, low_dim_key, proprio_dim) from
    `shape_meta.obs`. Picks the first rgb entry and the first low_dim entry.
    Defaults to type='low_dim' when an entry has no explicit type (the
    upstream convention).
    """
    rgb_key = None
    image_shape = None
    low_dim_key = None
    proprio_dim = None
    for key, attr in shape_meta["obs"].items():
        kind = attr.get("type", "low_dim")
        shape = tuple(attr["shape"])
        if kind == "rgb" and rgb_key is None:
            rgb_key, image_shape = key, shape
        elif kind == "low_dim" and low_dim_key is None:
            low_dim_key, proprio_dim = key, int(shape[0])
    if rgb_key is None or low_dim_key is None:
        raise ValueError(
            f"shape_meta.obs must contain at least one rgb and one low_dim key; "
            f"got {list(shape_meta['obs'].keys())}"
        )
    return rgb_key, image_shape, low_dim_key, proprio_dim


def make_app(checkpoint: str, device: str, num_inference_steps: int | None):
    print(f"[diffusion_policy_server] loading {checkpoint} on {device} ...", flush=True)
    policy, cfg = load_workspace(checkpoint, device)

    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = num_inference_steps

    n_obs_steps = int(cfg.policy.get("n_obs_steps", cfg.get("n_obs_steps", 2)))
    shape_meta = cfg.policy.shape_meta
    rgb_key, image_shape, low_dim_key, proprio_dim = discover_obs_keys(shape_meta)
    action_dim = int(shape_meta["action"]["shape"][0])
    print(
        f"[diffusion_policy_server] loaded. n_obs_steps={n_obs_steps} "
        f"rgb_key={rgb_key} image={image_shape} "
        f"low_dim_key={low_dim_key} proprio={proprio_dim} action={action_dim}",
        flush=True,
    )

    # Rolling per-episode observation history. Server is single-threaded
    # (`threaded=False` below) and serves one eval at a time, so a single
    # shared deque is fine — the client must send {"reset": true} on the
    # first call of each episode so we don't bleed obs across resets.
    img_history: deque = deque(maxlen=n_obs_steps)
    proprio_history: deque = deque(maxlen=n_obs_steps)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "model": "diffusion_policy",
            "n_obs_steps": n_obs_steps,
            "rgb_key": rgb_key,
            "image_shape": list(image_shape),
            "low_dim_key": low_dim_key,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
        })

    request_counter = [0]

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            request_counter[0] += 1
            req_id = request_counter[0]
            t_start = time.time()
            data = request.get_json(force=True)

            if data.get("reset", False):
                img_history.clear()
                proprio_history.clear()
            print(f"[dp_server] req#{req_id} reset={data.get('reset', False)} "
                  f"hist_len={len(img_history)}", flush=True)

            img = decode_png(data["images"]["image_primary"])  # (H, W, 3) uint8
            _, H, W = image_shape
            if img.shape[0] != H or img.shape[1] != W:
                img = np.asarray(
                    Image.fromarray(img).resize((W, H), Image.LANCZOS), dtype=np.uint8
                )
            img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_chw).to(device)

            proprio_list = data.get("proprio") or []
            if len(proprio_list) < proprio_dim:
                proprio_list = list(proprio_list) + [0.0] * (proprio_dim - len(proprio_list))
            proprio_list = proprio_list[:proprio_dim]
            proprio_t = torch.tensor(proprio_list, dtype=torch.float32, device=device)

            # Append current obs, then stack last n_obs_steps. When the deque
            # is shorter than n_obs_steps (e.g. the first call), edge-pad
            # with the oldest available frame — matches the canonical
            # MultiStepWrapper / DPRunner stack_last_n_obs behaviour.
            img_history.append(img_t)
            proprio_history.append(proprio_t)

            def _stack(hist):
                items = list(hist)
                # Edge-pad with cloned copies of the oldest frame so torch.stack
                # never receives the same tensor twice (defensive: shared refs
                # have surprised PyTorch on rare codepaths before).
                while len(items) < n_obs_steps:
                    items.insert(0, items[0].clone())
                return torch.stack(items, dim=0)

            image = _stack(img_history)[None].contiguous()       # (1, To, C, H, W)
            agent_pos = _stack(proprio_history)[None].contiguous()  # (1, To, D)

            obs_dict = {rgb_key: image, low_dim_key: agent_pos}

            t_pre = time.time()
            with torch.inference_mode():
                result = policy.predict_action(obs_dict)
            t_infer = time.time()
            actions = result["action"]  # (B=1, Ta, action_dim)
            first = actions[0, 0].detach().cpu().numpy().astype(np.float64).tolist()
            print(f"[dp_server] req#{req_id} done "
                  f"pre={t_pre - t_start:.3f}s infer={t_infer - t_pre:.3f}s "
                  f"total={time.time() - t_start:.3f}s", flush=True)
            return jsonify({"action": first, "unnorm_key": "diffusion_policy"})
        except Exception as e:
            print(f"[dp_server] req#{req_id} EXCEPTION after "
                  f"{time.time() - t_start:.3f}s", flush=True)
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a Hydra-saved Diffusion Policy `.ckpt`.")
    parser.add_argument("--num-inference-steps", type=int, default=16,
                        help="DDIM sampling steps. Lower = faster, fewer = noisier. "
                             "Set to None to keep the workspace's default.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    app = make_app(args.checkpoint, args.device, args.num_inference_steps)
    print(f"[diffusion_policy_server] listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)
