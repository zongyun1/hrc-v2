"""3D Diffusion Policy (Ze et al. 2024) model server. Run from the dp3 venv:

    /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/dp3/bin/python \
        baseline/dp3_server.py \
        --checkpoint runs/dp3/<task>/<run>/checkpoints/latest.ckpt \
        --port 8774

Then from MAWM env:
    python scripts/eval_vla.py --model dp3 --server-url http://127.0.0.1:8774 ...

Like the original Diffusion Policy, DP3 is per-task — there is no foundation
checkpoint. The expected file is a Hydra-saved workspace produced by the
upstream `train.py` (`baseline/3d_diffusion_policy/3D-Diffusion-Policy/`):
the payload bundles cfg + state_dicts + pickles, with the policy class instantiated
from `cfg.policy._target_` (`diffusion_policy_3d.policy.dp3.DP3`).

Wire format:
  POST /predict  body:
      { "instruction": str (ignored — DP3 has no language conditioning),
        "point_cloud": (N, 3) or (N, 6) list of floats,
        "proprio":     (D_proprio,) list of floats }
  reply:
      { "action": list[float] of length action_dim,
        "unnorm_key": "dp3" }

Server-side:
  - Tile the (N, 3) cloud and (D,) proprio across the policy's `n_obs_steps`.
  - Call policy.predict_action({"point_cloud": ..., "agent_pos": ...}) — keys
    must match cfg.task.shape_meta.obs.
  - Return the first action of result["action"] (shape (B, n_action_steps, Da)).

The DP3 stack pulls in pytorch3d, diffusers, hydra, omegaconf, dill, and the
upstream's third_party (gym 0.21, mujoco-py, mj_envs/mjrl/dexart/metaworld).
That conflicts with every other baseline env, hence the dedicated dp3 venv.
"""
import argparse
import os
import sys
import traceback

import numpy as np
from flask import Flask, request, jsonify
import torch

REPO_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "3d_diffusion_policy", "3D-Diffusion-Policy",
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_workspace(checkpoint: str, device: str):
    """Reconstruct a TrainDP3Workspace from a `.ckpt` saved by the upstream
    train.py. The cfg's policy._target_ is hydra-instantiated, then the
    state_dicts are loaded back. EMA weights win if present.
    """
    import dill
    import hydra
    # Importing train brings TrainDP3Workspace into scope so dill.loads can
    # find it when the payload's pickles reference workspace-internal classes.
    import train as dp3_train  # noqa: F401

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_) if "_target_" in cfg else dp3_train.TrainDP3Workspace
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if getattr(workspace, "ema_model", None) is not None else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def discover_obs_keys(shape_meta):
    """Return (pc_key, pc_shape, low_dim_key, proprio_dim, action_dim) from
    cfg.task.shape_meta. Picks the first point_cloud entry and the first
    low_dim entry. DP3 task configs use keys 'point_cloud' and 'agent_pos',
    but we discover them so subclassed configs (e.g. multi-arm) also work.
    """
    obs = shape_meta["obs"]
    pc_key = pc_shape = low_dim_key = proprio_dim = None
    for k, attr in obs.items():
        kind = attr.get("type", "low_dim")
        shape = tuple(attr["shape"])
        if kind == "point_cloud" and pc_key is None:
            pc_key, pc_shape = k, shape
        elif kind == "low_dim" and low_dim_key is None:
            low_dim_key, proprio_dim = k, int(shape[0])
    if pc_key is None or low_dim_key is None:
        raise ValueError(
            f"shape_meta.obs needs one point_cloud and one low_dim key; got {list(obs.keys())}"
        )
    action_dim = int(shape_meta["action"]["shape"][0])
    return pc_key, pc_shape, low_dim_key, proprio_dim, action_dim


def make_app(checkpoint: str, device: str, num_inference_steps: int | None):
    print(f"[dp3_server] loading {checkpoint} on {device} ...", flush=True)
    policy, cfg = load_workspace(checkpoint, device)

    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = num_inference_steps

    n_obs_steps = int(cfg.policy.get("n_obs_steps", cfg.get("n_obs_steps", 2)))
    shape_meta = cfg.task.shape_meta if "task" in cfg else cfg.shape_meta
    pc_key, pc_shape, low_dim_key, proprio_dim, action_dim = discover_obs_keys(shape_meta)
    n_points = int(pc_shape[0])
    pc_channels = int(pc_shape[1]) if len(pc_shape) > 1 else 3
    print(
        f"[dp3_server] loaded. n_obs_steps={n_obs_steps} "
        f"pc_key={pc_key} n_points={n_points} pc_channels={pc_channels} "
        f"low_dim_key={low_dim_key} proprio={proprio_dim} action={action_dim}",
        flush=True,
    )

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "model": "dp3",
            "n_obs_steps": n_obs_steps,
            "pc_key": pc_key,
            "n_points": n_points,
            "pc_channels": pc_channels,
            "use_color": pc_channels == 6,
            "low_dim_key": low_dim_key,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            pc = np.asarray(data["point_cloud"], dtype=np.float32)
            if pc.ndim != 2 or pc.shape[0] != n_points:
                raise ValueError(
                    f"point_cloud shape {pc.shape} != expected ({n_points}, {pc_channels})"
                )
            if pc.shape[1] != pc_channels:
                # Caller sent xyz but model expects xyzrgb (or vice versa).
                # Pad with zeros / drop colour as needed — model is the source of truth.
                if pc.shape[1] < pc_channels:
                    pad = np.zeros((pc.shape[0], pc_channels - pc.shape[1]), dtype=np.float32)
                    pc = np.concatenate([pc, pad], axis=1)
                else:
                    pc = pc[:, :pc_channels]
            pc_t = torch.from_numpy(pc).to(device)
            # (B=1, To, N, C) — tile current frame across the obs window.
            pc_in = pc_t[None, None].expand(1, n_obs_steps, -1, -1).contiguous()

            proprio_list = data.get("proprio") or []
            if len(proprio_list) < proprio_dim:
                proprio_list = list(proprio_list) + [0.0] * (proprio_dim - len(proprio_list))
            proprio_list = proprio_list[:proprio_dim]
            proprio_t = torch.tensor(proprio_list, dtype=torch.float32, device=device)
            agent_pos = proprio_t[None, None].expand(1, n_obs_steps, -1).contiguous()

            obs_dict = {pc_key: pc_in, low_dim_key: agent_pos}

            with torch.inference_mode():
                result = policy.predict_action(obs_dict)
            actions = result["action"]  # (B=1, n_action_steps, action_dim)
            first = actions[0, 0].detach().cpu().numpy().astype(np.float64).tolist()
            return jsonify({"action": first, "unnorm_key": "dp3"})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a Hydra-saved DP3 `.ckpt`.")
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="DDIM sampling steps. Default matches upstream dp3.yaml.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    app = make_app(args.checkpoint, args.device, args.num_inference_steps)
    print(f"[dp3_server] listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)
