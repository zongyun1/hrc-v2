"""Generic LeRobot policy server. Dispatches by ``--policy-type`` to one of
the imitation policies that ship with ``lerobot==0.1.0`` (already installed
in the pi0 venv):

    --policy-type act       → lerobot.common.policies.act.modeling_act.ACTPolicy
    --policy-type diffusion → lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy
    --policy-type vqbet     → lerobot.common.policies.vqbet.modeling_vqbet.VQBeTPolicy

All three share the same surface:
    * ``Class.from_pretrained(ckpt)`` builds the policy from a LeRobot-style
      checkpoint dir.
    * ``policy.select_action(batch)`` consumes a dict with keys matching
      ``cfg.image_features`` (+ ``observation.state`` when
      ``cfg.robot_state_feature`` is set) and emits one action at a time.
      The policy internally caches a chunk and pops from it per call.
    * ``policy.reset()`` clears that cache between episodes.

Usage (from the pi0 venv):

    /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/pi0/bin/python \
        baseline/lerobot_server.py \
        --policy-type {act,diffusion,vqbet} \
        --checkpoint runs/<name>_finetune/<exp>/checkpoints/last/pretrained_model \
        --port {8775,8776,8777}

Protocol (same as the other VLA servers):
    GET  /health   → {"ok": true, "model": <type>, "action_dim": int,
                      "image_keys": [...], "chunk_size": int|None,
                      "n_action_steps": int|None}
    POST /predict  → body {instruction, images{image_primary[,image_wrist]},
                           proprio: list[float], reset: bool}
                      reply {action: list[float]}

For the genesis-hr-bench LeRobot dataset (built by
``scripts/convert_genesis_hr_bench_to_lerobot.py --lerobot-naming``) the
action space is either 7-D bridge EE-delta
``[dx,dy,dz,droll,dpitch,dyaw,gripper]`` (gripper 1=closed) or 8-D qpos
``[7 joint deltas, gripper]`` (gripper 1=open), depending on the dataset used
for training. The client selects the matching ``--action-type`` at eval time.
"""
import argparse
import base64
import importlib
import io
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify


# Each policy_type maps to a list of `(module_path, class_name)` candidates
# tried in order. LeRobot v0.1.0 (pi0 venv) uses ``lerobot.common.policies.*``;
# v0.5.2 (lerobot venv, where SmolVLA lives) reorganized to
# ``lerobot.policies.*``. The shared server module works in either venv: it
# tries v0.1.0 paths first, then v0.5.2 paths, and uses whichever imports.
POLICY_REGISTRY = {
    "pi0": [
        ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    ],
    "pi05": [
        ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    ],
    "act": [
        ("lerobot.common.policies.act.modeling_act", "ACTPolicy"),
        ("lerobot.policies.act.modeling_act", "ACTPolicy"),
    ],
    "diffusion": [
        ("lerobot.common.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
        ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
    ],
    # NOTE: VQ-BeT requires a 1-camera checkpoint — see
    # scripts/finetune_vqbet.sh for the dataset-rebuild flow.
    "vqbet": [
        ("lerobot.common.policies.vqbet.modeling_vqbet", "VQBeTPolicy"),
        ("lerobot.policies.vqbet.modeling_vqbet", "VQBeTPolicy"),
    ],
    # SmolVLA only exists in lerobot>=0.4 (v0.5.2 in baseline/lerobot/), so
    # only the new-layout path is valid. Run against the lerobot venv at
    # yz/env/lerobot, not the pi0 venv.
    "smolvla": [
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    ],
}


def load_policy_class(policy_type: str):
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"unknown --policy-type {policy_type!r}; "
            f"expected one of {sorted(POLICY_REGISTRY)}"
        )
    candidates = POLICY_REGISTRY[policy_type]
    last_err: Exception | None = None
    for module_path, class_name in candidates:
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            last_err = e
            continue
    raise ImportError(
        f"could not import any of {candidates} for policy_type={policy_type!r}; "
        f"last error: {last_err}"
    )


def decode_png(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))), dtype=np.uint8)


def _to_chw_float(rgb: np.ndarray) -> torch.Tensor:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def make_app(policy_type: str, checkpoint: str, device: str):
    Policy = load_policy_class(policy_type)
    print(f"[lerobot_server] policy_type={policy_type} loading {checkpoint} on {device} ...",
          flush=True)
    policy = Policy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    cfg = policy.config
    # Optional inference-time override: ACT trained with
    # temporal_ensemble_coeff averages ~chunk_size near-uniformly-weighted
    # chunk predictions, which can wash out decisive low-frequency actions
    # (e.g. a gripper close). LEROBOT_TEMPORAL_ENSEMBLE=off switches to the
    # open-loop action-queue path; LEROBOT_N_ACTION_STEPS sets the replan
    # horizon (actions popped per chunk).
    te_env = os.environ.get("LEROBOT_TEMPORAL_ENSEMBLE", "")
    if te_env.lower() == "off":
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            cfg.temporal_ensemble_coeff = None
            if hasattr(policy, "temporal_ensembler"):
                policy.temporal_ensembler = None
            print("[lerobot_server] temporal ensembling DISABLED via env", flush=True)
    elif te_env:
        # Numeric override: keep temporal ensembling but at a different
        # coefficient. weight = exp(-coeff * age); LARGER coeff => recent
        # predictions dominate (decisive actions like a grasp survive, while
        # the approach stays smoothed/reactive). Rebuild the ensembler so the
        # new coeff takes effect.
        coeff = float(te_env)
        cfg.temporal_ensemble_coeff = coeff
        ACTTemporalEnsembler = None
        for modpath in (
            "lerobot.common.policies.act.modeling_act",  # v0.1.0 (pi0 env)
            "lerobot.policies.act.modeling_act",          # v0.5.2 (lerobot env)
        ):
            try:
                ACTTemporalEnsembler = importlib.import_module(modpath).ACTTemporalEnsembler
                break
            except Exception:
                continue
        if ACTTemporalEnsembler is not None:
            policy.temporal_ensembler = ACTTemporalEnsembler(coeff, cfg.chunk_size)
        else:
            print("[lerobot_server] WARN ACTTemporalEnsembler not found; "
                  "coeff set on cfg only", flush=True)
        print(f"[lerobot_server] temporal_ensemble_coeff override -> {coeff}", flush=True)
    nas_override = os.environ.get("LEROBOT_N_ACTION_STEPS")
    if nas_override:
        cfg.n_action_steps = int(nas_override)
        print(f"[lerobot_server] n_action_steps override -> {cfg.n_action_steps}", flush=True)
    if hasattr(policy, "reset"):
        policy.reset()
    preprocessor = None
    postprocessor = None
    processor_files_exist = (
        (Path(checkpoint) / "policy_preprocessor.json").exists()
        and (Path(checkpoint) / "policy_postprocessor.json").exists()
    )
    if processor_files_exist:
        try:
            from lerobot.policies.factory import make_pre_post_processors
            # The saved processor pipeline hardcodes a device_processor step
            # (often device='cuda'). Override it to the requested device so
            # CPU serving works when the env's torch can't run on the node GPU
            # (cudaErrorNoKernelImageForDevice). Harmless no-op on v0.1.0.
            _dev_ovr = {"device_processor": {"device": device}}
            try:
                preprocessor, postprocessor = make_pre_post_processors(
                    cfg, pretrained_path=checkpoint,
                    preprocessor_overrides=_dev_ovr,
                    postprocessor_overrides=_dev_ovr,
                )
            except TypeError:
                preprocessor, postprocessor = make_pre_post_processors(
                    cfg, pretrained_path=checkpoint)
        except ImportError as exc:
            if policy_type in {"pi0", "pi05", "smolvla"}:
                raise RuntimeError(
                    "Saved policy processors are required for language-conditioned "
                    f"{policy_type} inference, but this LeRobot environment does "
                    "not provide lerobot.policies.factory."
                ) from exc
            print(
                "[lerobot_server] saved processor files detected, but "
                "lerobot.policies.factory is unavailable; continuing with "
                "legacy policy-native preprocessing.",
                flush=True,
            )
        except Exception as exc:
            if policy_type in {"pi0", "pi05"}:
                raise RuntimeError(
                    "Failed to load the PI0/PI05 saved policy processors. These "
                    "processors include the real PaliGemma tokenizer required for "
                    "language-conditioned inference. Ensure this environment has "
                    "Hugging Face access to google/paligemma-3b-pt-224, for example "
                    "by logging in with a token approved for that gated repo, or by "
                    "pre-populating the tokenizer files in the HF cache. Refusing to "
                    "run with dummy language tokens."
                ) from exc
            raise

    processor_rename_map = {}
    if preprocessor is not None:
        proc_path = Path(checkpoint) / "policy_preprocessor.json"
        try:
            with proc_path.open() as f:
                proc_cfg = json.load(f)
            for step in proc_cfg.get("steps", []):
                if step.get("registry_name") == "rename_observations_processor":
                    processor_rename_map = dict(step.get("config", {}).get("rename_map", {}) or {})
                    break
        except Exception:
            processor_rename_map = {}

    action_dim = int(cfg.output_features["action"].shape[0])
    # Different policies use different attr names for the chunk; report all,
    # default to None so JSON serializes cleanly.
    chunk_size = getattr(cfg, "chunk_size", None) or getattr(cfg, "action_chunk_size", None)
    n_action_steps = getattr(cfg, "n_action_steps", None)
    image_keys = list(cfg.image_features.keys())
    state_key = "observation.state" if cfg.robot_state_feature is not None else None
    state_dim = None
    if state_key is not None:
        input_features = getattr(cfg, "input_features", None) or {}
        state_feature = input_features.get(state_key)
        if state_feature is not None and getattr(state_feature, "shape", None):
            state_dim = int(state_feature.shape[0])
        else:
            state_dim = action_dim
    # These VLA policies are always language-conditioned; their configs do not
    # all expose use_language_conditioning, so force the flag on here.
    use_language = bool(getattr(cfg, "use_language_conditioning", False)) \
        or policy_type in {"pi0", "pi05", "smolvla"}
    print(f"[lerobot_server] loaded. action_dim={action_dim} chunk_size={chunk_size} "
          f"n_action_steps={n_action_steps} image_keys={image_keys} "
          f"state_key={state_key} state_dim={state_dim} use_language={use_language} "
          f"rename_map={processor_rename_map}",
          flush=True)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "model": policy_type,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "image_keys": image_keys,
            "state_dim": state_dim,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            images = data["images"]
            proprio = np.asarray(data.get("proprio") or [], dtype=np.float32)

            primary = decode_png(images["image_primary"])
            wrist = (
                decode_png(images["image_wrist"])
                if "image_wrist" in images
                else None
            )

            obs = {}
            use_processors = preprocessor is not None and postprocessor is not None
            if use_processors and processor_rename_map:
                # Feed the same raw dataset keys used at training time. The
                # saved RenameObservationsProcessor maps these to the policy's
                # image slots, and any configured-but-absent camera (for
                # example left_wrist_0_rgb in 2-camera PI0 runs) remains
                # missing so PI0 masks it as an empty camera.
                raw_images = {
                    "observation.image": primary,
                    "image": primary,
                }
                if wrist is not None:
                    raw_images["observation.wrist_image"] = wrist
                    raw_images["wrist_image"] = wrist
                for raw_key, final_key in processor_rename_map.items():
                    if final_key not in image_keys or raw_key not in raw_images:
                        continue
                    obs[raw_key] = _to_chw_float(raw_images[raw_key]).unsqueeze(0)
            else:
                for key in image_keys:
                    # Convention: any key containing "wrist" maps to the wrist
                    # camera; everything else maps to the primary head camera.
                    if "wrist" in key:
                        img = wrist if wrist is not None else primary
                    else:
                        img = primary
                    tensor = _to_chw_float(img)
                    obs[key] = tensor.unsqueeze(0) if use_processors else tensor.unsqueeze(0).to(device)

            if state_key is not None:
                if use_processors:
                    # Saved normalizer stats are for the policy state dimension.
                    # Keep inputs batched so LeRobot processors treat all
                    # observation keys consistently.
                    target_dim = int(proprio.size)
                    state = torch.from_numpy(proprio[:target_dim])
                    obs[state_key] = state.unsqueeze(0)
                else:
                    target_dim = int(state_dim or proprio.size)
                    if proprio.size < target_dim:
                        proprio = np.concatenate(
                            [proprio, np.zeros(target_dim - proprio.size, dtype=np.float32)]
                        )
                    state = torch.from_numpy(proprio[:target_dim])
                    obs[state_key] = state.unsqueeze(0).to(device)

            if use_language:
                obs["task"] = data.get("instruction", "") if use_processors else [data.get("instruction", "")]

            if data.get("reset", False):
                policy.reset()
                if preprocessor is not None:
                    preprocessor.reset()
                if postprocessor is not None:
                    postprocessor.reset()

            with torch.inference_mode():
                if use_processors:
                    obs = preprocessor(obs)
                    action = policy.select_action(obs)
                    action = postprocessor(action)
                else:
                    action = policy.select_action(obs)
            action = action.squeeze(0).detach().cpu().numpy().astype(np.float64)

            return jsonify({"action": action.tolist()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-type", required=True, choices=sorted(POLICY_REGISTRY))
    p.add_argument("--checkpoint", required=True,
                   help="path to a saved LeRobot pretrained_model dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()

    app = make_app(args.policy_type, args.checkpoint, args.device)
    print(f"[lerobot_server] listening on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
