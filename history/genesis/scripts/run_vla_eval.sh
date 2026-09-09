#!/bin/bash
# One-command VLA baseline eval. Spawns the matching model server in its own
# env, waits for /health, runs scripts/eval_vla.py in the MAWM env, and kills
# the server on exit (normal, error, or Ctrl-C).
#
# Usage:
#   scripts/run_vla_eval.sh --model {openvla_oft|pi0|pi05|pi0_fast|rdt|diffusion_policy|dp3|act|vqbet|lerobot_diffusion|smolvla}
#                           [--task NAME] [--episodes N] [--max-steps N]
#                           [--port P] [--action-type ee|qpos|qpos_abs]
#                           [--checkpoint PATH] [--unnorm-key KEY] [--video-dir DIR]
#                           [-- extra eval_vla.py args...]
#
# Defaults per model (override with flags above):
#   openvla_oft:       port 8769, action-type ee    (serves openvla-7b-oft finetune)
#   pi0:               port 8767, action-type qpos_abs (--checkpoint REQUIRED, lerobot_server, lerobot venv)
#   pi05:              port 8768, action-type qpos_abs (--checkpoint REQUIRED, lerobot_server, lerobot venv)
#   pi0_fast:          port 8779, action-type ee    (--checkpoint REQUIRED, lerobot_server, lerobot venv)
#   rdt:               port 8772, action-type qpos  (serves rdt-1b + T5-XXL + SigLIP)
#   diffusion_policy:  port 8773, action-type qpos  (--checkpoint REQUIRED)
#   dp3:               port 8774, action-type qpos  (--checkpoint REQUIRED)
#   act:               port 8775, action-type ee    (--checkpoint REQUIRED, lerobot_server, pi0 venv)
#   vqbet:             port 8776, action-type ee    (--checkpoint REQUIRED, lerobot_server, pi0 venv)
#   lerobot_diffusion: port 8777, action-type ee    (--checkpoint REQUIRED, lerobot_server, lerobot venv)
#   smolvla:           port 8778, action-type ee    (--checkpoint REQUIRED, lerobot_server, lerobot venv)
#
# Shared defaults: --task pour_water --episodes 1 --max-steps 50
#
# Args after `--` are forwarded verbatim to eval_vla.py.

set -euo pipefail

MODEL=""
TASK="pour_water"
EPISODES=1
MAX_STEPS=50
PORT=""
ACTION_TYPE="${ACTION_TYPE:-}"
VIDEO_DIR=""
CHECKPOINT=""
UNNORM_KEY=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --unnorm-key) UNNORM_KEY="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (use -- to pass through to eval_vla.py)" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "required: --model {openvla_oft|pi0|pi05|pi0_fast|rdt|diffusion_policy|dp3|act|vqbet|lerobot_diffusion|smolvla}" >&2
  exit 2
fi

# Source per-host config (REPO_ROOT, MAWM_PY, OFT_PY, VLA_ENV_ROOT, ...).
# To override, create scripts/env.local.sh — see scripts/env.local.sh.example.
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ENV_ROOT="$VLA_ENV_ROOT"
cd "$REPO_ROOT"

# Per-model dispatch table. Keep in sync with the corresponding baseline/*_server.py
# scripts — these fields are load-bearing:
#   SERVER_PY          env python that can import the model
#   SERVER_SCRIPT      the server module to run
#   DEFAULT_PORT       baseline port (non-overlapping so you can run side-by-side)
#   SERVER_ARGS        any non-port CLI flags the server needs (config/checkpoint)
#   SERVER_ENV         env-vars prefixed to the server launch
#   DEFAULT_ACTION_TYPE  matches what the model was trained to emit
SERVER_ENV=()
case "$MODEL" in
  pi0|pi05|pi0_fast)
    # pi0 / pi0.5 / pi0-FAST are finetuned via lerobot v0.5.2 (see
    # scripts/finetune_lerobot_v2.sh) and served by the generic
    # lerobot_server.py from the lerobot venv — same path as smolvla.
    # (This replaced the openpi/JAX pi0_server.py + openpi submodule, both
    # removed: openpi could not train multi-node, lerobot's accelerate can.)
    SERVER_PY="$ENV_ROOT/lerobot/bin/python"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    case "$MODEL" in
      pi0)      DEFAULT_PORT=8767 ;;
      pi05)     DEFAULT_PORT=8768 ;;
      pi0_fast) DEFAULT_PORT=8779 ;;
    esac
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for $MODEL (finetuned via lerobot, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type "$MODEL" --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos_abs"
    ;;
  openvla_oft)
    SERVER_PY="$ENV_ROOT/openvla_oft/bin/python"
    SERVER_SCRIPT="baseline/servers/openvla_oft_server.py"
    DEFAULT_PORT=8769
    OPENVLA_OFT_CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
    if [[ -z "$UNNORM_KEY" && "$OPENVLA_OFT_CHECKPOINT" == *genesis* ]]; then
      UNNORM_KEY="genesis_hr_bench"
    fi
    SERVER_ARGS=(--checkpoint "$OPENVLA_OFT_CHECKPOINT")
    [[ -n "$UNNORM_KEY" ]] && SERVER_ARGS+=(--unnorm-key "$UNNORM_KEY")
    SERVER_ENV=("HF_HUB_CACHE=$PWD/checkpoints")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  rdt)
    SERVER_PY="$ENV_ROOT/rdt/bin/python"
    SERVER_SCRIPT="baseline/servers/rdt_server.py"
    DEFAULT_PORT=8772
    SERVER_ARGS=(--checkpoint "${CHECKPOINT:-robotics-diffusion-transformer/rdt-1b}")
    SERVER_ENV=("HF_HUB_CACHE=$PWD/checkpoints")
    DEFAULT_ACTION_TYPE="qpos"
    ;;
  diffusion_policy)
    SERVER_PY="$ENV_ROOT/diffusion_policy/bin/python"
    SERVER_SCRIPT="baseline/servers/diffusion_policy_server.py"
    DEFAULT_PORT=8773
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for diffusion_policy (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos"
    ;;
  dp3)
    SERVER_PY="$ENV_ROOT/dp3/bin/python"
    SERVER_SCRIPT="baseline/servers/dp3_server.py"
    DEFAULT_PORT=8774
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for dp3 (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos"
    ;;
  act)
    # All three LeRobot imitation baselines (act/vqbet/lerobot_diffusion) run
    # through baseline/servers/lerobot_server.py with a different --policy-type.
    # Reuse the legacy pi0 venv for older ACT checkpoints. Newer LeRobot
    # checkpoints with saved policy processors can opt into the v0.5 env with
    # ACT_SERVER_ENV=lerobot.
    if [[ "${ACT_SERVER_ENV:-pi0}" == "lerobot" ]]; then
      SERVER_PY="$ENV_ROOT/lerobot/bin/python"
    else
      SERVER_PY="$ENV_ROOT/pi0/bin/python"
    fi
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8775
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for act (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type act --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="qpos_abs"
    ;;
  vqbet)
    SERVER_PY="$ENV_ROOT/pi0/bin/python"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8776
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for vqbet (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type vqbet --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  lerobot_diffusion)
    SERVER_PY="$ENV_ROOT/lerobot/bin/python"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8777
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for lerobot_diffusion (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type diffusion --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  smolvla)
    # SmolVLA only exists in lerobot >= 0.4 (submodule v0.5.2); the pi0
    # venv's v0.1.0 doesn't have it. Use the dedicated lerobot venv.
    SERVER_PY="$ENV_ROOT/lerobot/bin/python"
    SERVER_SCRIPT="baseline/servers/lerobot_server.py"
    DEFAULT_PORT=8778
    if [[ -z "$CHECKPOINT" ]]; then
      echo "--checkpoint PATH required for smolvla (per-task checkpoint, no default)" >&2
      exit 2
    fi
    SERVER_ARGS=(--policy-type smolvla --checkpoint "$CHECKPOINT")
    DEFAULT_ACTION_TYPE="ee"
    ;;
  *) echo "--model must be openvla_oft|pi0|pi05|pi0_fast|rdt|diffusion_policy|dp3|act|vqbet|lerobot_diffusion|smolvla, got: $MODEL" >&2; exit 2 ;;
esac

PORT="${PORT:-$DEFAULT_PORT}"
ACTION_TYPE="${ACTION_TYPE:-$DEFAULT_ACTION_TYPE}"

# MAWM_PY already set by env.sh; nothing to do.
# Server log: defaults to /tmp/mktemp (ephemeral on compute nodes — gone when
# the job ends). For diagnosable failures, set SERVER_LOG_DIR to a persistent
# directory (e.g. SERVER_LOG_DIR=runs/<baseline>/eval/server_logs in your
# sbatch) and the log lands under <dir>/${MODEL}_server.${SLURM_JOB_ID}.log.
if [[ -n "${SERVER_LOG_DIR:-}" ]]; then
  mkdir -p "$SERVER_LOG_DIR"
  SERVER_LOG="${SERVER_LOG_DIR}/${MODEL}_server.${SLURM_JOB_ID:-$$}.log"
else
  SERVER_LOG="$(mktemp -t "${MODEL}_server.XXXXXX.log")"
fi

echo "[run_vla_eval] model=$MODEL port=$PORT action_type=$ACTION_TYPE"
echo "[run_vla_eval] server log: $SERVER_LOG"

# When SERVER_ENV is empty, `"${arr[@]:-}"` expands to a single empty
# positional — which `env` then tries to exec as a command. Skip the `env`
# prefix entirely in that case. bash ≥ 4.4 is fine with `"${arr[@]}"` on an
# empty (but declared) array under `set -u`, so no `:-` needed here.
if [[ ${#SERVER_ENV[@]} -gt 0 ]]; then
  env "${SERVER_ENV[@]}" "$SERVER_PY" "$SERVER_SCRIPT" \
      "${SERVER_ARGS[@]}" --port "$PORT" \
      > "$SERVER_LOG" 2>&1 &
else
  "$SERVER_PY" "$SERVER_SCRIPT" \
      "${SERVER_ARGS[@]}" --port "$PORT" \
      > "$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_vla_eval] stopping $MODEL server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Cold-cache first runs download multi-GB checkpoints — allow 30 min. Warm
# reruns come up in ~30-60s.
DEADLINE=1800
for ((i=1; i<=DEADLINE; i++)); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[run_vla_eval] server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_vla_eval] $MODEL server died during startup. Log tail:" >&2
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
  fi
  if (( i % 30 == 0 )); then
    echo "[run_vla_eval] still waiting ... (${i}s) — last log line:"
    tail -n 1 "$SERVER_LOG" || true
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[run_vla_eval] server did not come up within ${DEADLINE}s. Log tail:" >&2
  tail -n 50 "$SERVER_LOG" >&2
  exit 1
fi

EVAL_ARGS=(
  --task "$TASK" --model "$MODEL"
  --server-url "http://127.0.0.1:$PORT"
  --action-type "$ACTION_TYPE"
  --episodes "$EPISODES" --max-steps "$MAX_STEPS"
)
[[ -n "$UNNORM_KEY" ]] && EVAL_ARGS+=(--unnorm-key "$UNNORM_KEY")
[[ -n "$VIDEO_DIR" ]] && EVAL_ARGS+=(--video-dir "$VIDEO_DIR")
EVAL_ARGS+=("${EXTRA_ARGS[@]}")

echo "[run_vla_eval] running eval: ${EVAL_ARGS[*]}"
# VLA eval is simulation/render heavy; use Genesis GPU backend unless the
# caller explicitly overrides it (e.g. GENESIS_BACKEND=cpu for debugging).
export GENESIS_BACKEND="${GENESIS_BACKEND:-gpu}"
echo "[run_vla_eval] GENESIS_BACKEND=$GENESIS_BACKEND"
SETUPTOOLS_USE_DISTUTILS=stdlib "$MAWM_PY" scripts/eval_vla.py "${EVAL_ARGS[@]}"
