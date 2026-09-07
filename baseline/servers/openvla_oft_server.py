"""OpenVLA-OFT model server. Run from the openvla_oft venv:

    HF_HUB_CACHE=$PWD/checkpoints \
        /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/openvla_oft/bin/python \
        baseline/openvla_oft_server.py \
        --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 8769

Then from MAWM env:
    python scripts/eval_vla.py --model openvla_oft --server-url http://127.0.0.1:8769 ...

OFT replaces OpenVLA's autoregressive action-token head with an L1 regression
MLP (faster + more precise), and predicts an 8-step chunk per forward pass.
The loader is non-trivial — OFT splits weights into (a) the VLA backbone,
(b) a `proprio_projector` side-network, (c) the `action_head` MLP. All three
live in the checkpoint repo and must be loaded in tandem.

The HF checkpoints only ship the LIBERO-finetuned flavors:
    moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10,spatial-object-goal-10}

`unnorm_key` defaults to `libero_spatial_no_noops`; pass `--unnorm-key` to
match a different checkpoint (the OFT loader's `check_unnorm_key` helper
does a `_no_noops` suffix fallback).
"""
import argparse
import base64
import io
import sys
import traceback
from collections import deque
from types import SimpleNamespace

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch

# openvla-oft's inference helpers are under `experiments.robot`; put the repo
# root on sys.path so we can import them without cloning into site-packages.
import os
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openvla-oft")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)
from prismatic.vla.constants import PROPRIO_DIM  # noqa: E402


def decode_png_rgb(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)


def build_cfg(checkpoint: str, unnorm_key: str, num_images: int) -> SimpleNamespace:
    """The OFT helpers expect a `GenerateConfig` dataclass, but only read a
    handful of attributes at inference time. A SimpleNamespace with those
    fields is enough and lets us avoid pulling in the LIBERO eval script.
    """
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=num_images,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        lora_rank=32,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=unnorm_key,
    )


def make_app(checkpoint: str, unnorm_key: str, num_images: int, device: str):
    print(f"[openvla_oft_server] loading {checkpoint} ...", flush=True)
    cfg = build_cfg(checkpoint, unnorm_key, num_images)

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=PROPRIO_DIM)
    action_head = get_action_head(cfg, vla.llm_dim)
    print("[openvla_oft_server] loaded (vla + proprio_projector + action_head)", flush=True)

    app = Flask(__name__)
    action_chunk = deque()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "ok": True,
                "model": "openvla_oft",
                "unnorm_key": unnorm_key,
                "chunk_size": cfg.num_open_loop_steps,
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            instruction = data["instruction"]
            cfg.unnorm_key = data.get("unnorm_key") or unnorm_key
            if data.get("reset", False):
                action_chunk.clear()

            primary = decode_png_rgb(data["images"]["image_primary"])
            wrist = None
            if "image_wrist" in data["images"] and num_images > 1:
                wrist = decode_png_rgb(data["images"]["image_wrist"])

            proprio = np.asarray(data.get("proprio") or [0.0] * PROPRIO_DIM, dtype=np.float32)
            if proprio.size < PROPRIO_DIM:
                pad = np.zeros(PROPRIO_DIM - proprio.size, dtype=np.float32)
                proprio = np.concatenate([proprio, pad])
            proprio = proprio[:PROPRIO_DIM]

            obs = {"full_image": primary, "state": proprio}
            if wrist is not None:
                obs["wrist_image"] = wrist

            if not action_chunk:
                with torch.inference_mode():
                    actions = get_vla_action(
                        cfg,
                        vla,
                        processor,
                        obs,
                        task_label=instruction,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                action_chunk.extend(
                    np.asarray(action, dtype=np.float64).flatten()
                    for action in actions
                )

            first = action_chunk.popleft()
            return jsonify(
                {
                    "action": first.tolist()[:7],
                    "unnorm_key": cfg.unnorm_key,
                    "remaining_chunk": len(action_chunk),
                }
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
        help="HF Hub path of an OFT-finetuned checkpoint "
             "(see openvla-oft README for supported IDs).",
    )
    parser.add_argument(
        "--unnorm-key",
        default="libero_spatial_no_noops",
        help="Action un-normalization key; check_unnorm_key in OFT's loader "
             "falls back to `<key>_no_noops` if the plain key is missing.",
    )
    parser.add_argument("--num-images", type=int, default=2,
                        help="1 = primary-only, 2 = primary + wrist")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    app = make_app(args.checkpoint, args.unnorm_key, args.num_images, args.device)
    print(
        f"[openvla_oft_server] listening on http://{args.host}:{args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=False)
