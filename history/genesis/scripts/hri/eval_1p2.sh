#!/bin/bash
# Self-contained ManiCast-ACT eval on Genesis 1.2.0, bypassing the
# migration-broken scripts/env.sh (which activates a nonexistent
# .venv-genesis-latest and no longer sets MAWM_PY).
#
# Starts the ACT model server in its own venv, waits for /health, runs
# scripts/eval_vla.py under MAWM_latest_genesis (genesis 1.2.0), kills server.
#
# Usage:
#   SERVER_ENV=lerobot CKPT=<path> PORT=8931 TE_COEFF=0.3 \
#     scripts/hri/eval_1p2.sh <eval_vla.py args...>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

ENV_ROOT="/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env"
SIM_PY="$ENV_ROOT/MAWM_latest_genesis/bin/python"
SRV_ENV="${SERVER_ENV:-lerobot}"          # lerobot (Miiche v0.5.2) or pi0 (v0.1.0)
SRV_PY="$ENV_ROOT/$SRV_ENV/bin/python"
CKPT="${CKPT:?set CKPT}"
PORT="${PORT:-8931}"
TE_COEFF="${TE_COEFF:-0.3}"
export GENESIS_BACKEND="${GENESIS_BACKEND:-cpu}"
mkdir -p job_logs
SRVLOG="job_logs/eval1p2_server_${SLURM_JOB_ID:-local}.log"

# The lerobot-env torch is built for a different GPU arch than some nodes
# (cudaErrorNoKernelImageForDevice) — and it bites even with --device cpu
# because the saved normalizer stats live on CUDA. ACT is small, so default
# to PURE CPU by hiding the GPU from the server process entirely; override
# with SERVER_DEVICE=cuda (+ a matching node) when the arch lines up.
SRV_DEVICE="${SERVER_DEVICE:-cpu}"
SRV_GPU_ENV=()
[[ "$SRV_DEVICE" == "cpu" ]] && SRV_GPU_ENV=(CUDA_VISIBLE_DEVICES="")
POLICY_TYPE="${POLICY_TYPE:-act}"       # act | diffusion | ...
EVAL_MODEL="${EVAL_MODEL:-act}"         # client registry key: act | lerobot_diffusion | ...
# temporal-ensemble knob only applies to ACT; skip for diffusion.
TE_ENV=()
[[ "$POLICY_TYPE" == "act" ]] && TE_ENV=(LEROBOT_TEMPORAL_ENSEMBLE="$TE_COEFF")
echo "[eval_1p2] policy=$POLICY_TYPE model=$EVAL_MODEL server_env=$SRV_ENV ckpt=$CKPT port=$PORT device=$SRV_DEVICE sim=1.2.0"
env "${SRV_GPU_ENV[@]}" "${TE_ENV[@]}" "$SRV_PY" baseline/servers/lerobot_server.py \
    --policy-type "$POLICY_TYPE" --checkpoint "$CKPT" --device "$SRV_DEVICE" --port "$PORT" > "$SRVLOG" 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

# wait for /health
for s in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "[eval_1p2] server up (${s}0s)"; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[eval_1p2] server DIED; tail:"; tail -20 "$SRVLOG"; exit 1; fi
  sleep 10
done

SETUPTOOLS_USE_DISTUTILS=stdlib "$SIM_PY" scripts/eval_vla.py \
    --model "$EVAL_MODEL" --server-url "http://127.0.0.1:$PORT" --action-type qpos_abs "$@"
RC=$?
echo "[eval_1p2] eval rc=$RC"
exit $RC
