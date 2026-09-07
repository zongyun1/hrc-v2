"""RDT-1B (Robotics Diffusion Transformer) model server. Run from the rdt venv:

    HF_HUB_CACHE=$PWD/checkpoints \
        /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/rdt/bin/python \
        baseline/rdt_server.py \
        --checkpoint robotics-diffusion-transformer/rdt-1b \
        --text-encoder google/t5-v1_1-xxl \
        --vision-encoder google/siglip-so400m-patch14-384 \
        --port 8772

Then from MAWM env:
    python scripts/eval_vla.py --model rdt --server-url http://127.0.0.1:8772 ...

RDT is a 1B-parameter diffusion transformer. It predicts 64-step chunks in a
128-D unified action vector (covers up to 14-DOF bimanual). This server uses
the ManiSkill-style single-arm loader (`rdt/scripts/maniskill_model.py`)
which maps Franka 7-joint + gripper into slots via `MANISKILL_INDICES`.

Conventions:
  Image order expected by `step()`: [ext_{t-1}, right_wrist_{t-1},
  left_wrist_{t-1}, ext_t, right_wrist_t, left_wrist_t]. On the first call
  (no history) we duplicate the current frame into the t-1 slots and leave
  left_wrist as None (the model fills that with a neutral background).

  Action output: 8-D per step after `_unformat_action_to_joint` — 7 joint
  positions + 1 gripper (ManiSkill convention). Returned as-is; the client
  passes through (Pi0Policy-style) since this is NOT a bridge delta.

First-run caveat: T5-XXL (~20 GB) and SigLIP-so400m (~3 GB) download into
$HF_HUB_CACHE — size the deadline accordingly.
"""
import argparse
import base64
import io
import os
import sys
import traceback

import numpy as np
import yaml
from PIL import Image
from flask import Flask, request, jsonify
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdt")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.maniskill_model import (  # noqa: E402
    RoboticDiffusionTransformerModel,
    create_model,
)


def decode_png_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def make_app(
    checkpoint: str,
    text_encoder: str,
    vision_encoder: str,
    config_path: str,
    device: str,
):
    with open(config_path) as f:
        args = yaml.safe_load(f)

    print(
        f"[rdt_server] loading RDT checkpoint={checkpoint} "
        f"text={text_encoder} vision={vision_encoder} ...",
        flush=True,
    )
    model: RoboticDiffusionTransformerModel = create_model(
        args=args,
        device=device,
        dtype=torch.bfloat16,
        pretrained=checkpoint,
        pretrained_text_encoder_name_or_path=text_encoder,
        pretrained_vision_encoder_name_or_path=vision_encoder,
    )
    print("[rdt_server] loaded", flush=True)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {"ok": True, "model": "rdt", "chunk_size": args["common"]["action_chunk_size"]}
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            instruction = data["instruction"]

            primary = decode_png_pil(data["images"]["image_primary"])
            wrist = (
                decode_png_pil(data["images"]["image_wrist"])
                if "image_wrist" in data["images"]
                else None
            )
            # Image order: [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
            #               ext_t,     right_wrist_t,     left_wrist_t]
            # We don't keep history on the server — duplicate the current
            # frame into the t-1 slots.
            images = [primary, wrist, None, primary, wrist, None]

            # 8-D Franka proprio: [7 joint_pos, 1 gripper_open]. If the client
            # only sent 7 (no gripper proprio), pad with 0.
            proprio_list = data.get("proprio") or []
            if len(proprio_list) < 8:
                proprio_list = list(proprio_list) + [0.0] * (8 - len(proprio_list))
            proprio_list = proprio_list[:8]
            # The ManiSkill step expects shape (B=1, N=1, 14) in _format_joint_to_state,
            # but MANISKILL_INDICES only indexes 8 (7 joint + 1 gripper). Looking at
            # the upstream: the (1, 1, 14) call site must be for bimanual; for
            # single-arm we need (1, 1, 8) matching the 8 MANISKILL_INDICES slots.
            proprio = torch.tensor(proprio_list, dtype=torch.float32).reshape(1, 8)

            with torch.inference_mode():
                text_embeds = model.encode_instruction(instruction, device=device)
                # step returns raw action in model's action space; maniskill_model's
                # _unformat_action_to_joint pulls out the 8 MANISKILL_INDICES slots.
                action_chunk = model.step(proprio, images, text_embeds)
            a = np.asarray(action_chunk)
            # Chunk is (horizon, 8) or similar — take first step.
            first = a.reshape(-1, a.shape[-1])[0]
            return jsonify(
                {"action": first.astype(np.float64).tolist()[:8], "unnorm_key": "maniskill"}
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="robotics-diffusion-transformer/rdt-1b")
    parser.add_argument("--text-encoder", default="google/t5-v1_1-xxl")
    parser.add_argument("--vision-encoder", default="google/siglip-so400m-patch14-384")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "base.yaml"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    app = make_app(
        args.checkpoint, args.text_encoder, args.vision_encoder, args.config, args.device
    )
    print(f"[rdt_server] listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)
