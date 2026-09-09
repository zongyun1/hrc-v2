#!/usr/bin/env bash
#SBATCH --job-name=ridgeback-warehouse
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/ridgeback_warehouse_%j.log
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1
PROJECT="$ROOT/hrc-v2"
OUT_DIR="${WAREHOUSE_OUTPUT_DIR:-$PROJECT/outputs/isaac_warehouse/${SLURM_JOB_ID:-manual}}"
mkdir -p "$ROOT/.cache/kitchen-home" "$ROOT/.cache/kitchen-kit-cache" \
    "$ROOT/.cache/kitchen-kit-logs" "$OUT_DIR"
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_ISAAC_PATH=/isaac-sim
export APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONUNBUFFERED=1
cd "$PROJECT"
apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
    --home "$ROOT/.cache/kitchen-home:/root" \
    --bind "$ROOT/.cache/kitchen-kit-cache:/isaac-sim/kit/cache" \
    --bind "$ROOT/.cache/kitchen-kit-logs:/isaac-sim/kit/logs" \
    "$RUNTIME/isaac-lab_3.0.0-beta2-post1_amd64.sif" \
    /isaac-sim/python.sh "$PROJECT/isaac_kitchen/run_kitchen.py" \
    --headless --enable_cameras --scene warehouse --steps 2700 --robot_xy 0 0 \
    --camera_eye -3.0 -4.0 2.8 --camera_target 0.0 0.8 0.9 \
    "$@" --output_dir "$OUT_DIR"
test -s "$OUT_DIR/result.json"
test -s "$OUT_DIR/preview.mp4"

python3 - "$OUT_DIR/result.json" <<'PY_CHECK'
import json, sys
with open(sys.argv[1]) as f:
    assert json.load(f)["status"] == "passed", "Grasp verification failed"
PY_CHECK
