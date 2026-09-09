#!/usr/bin/env bash
#SBATCH --job-name=human-kitchen-cosim
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/human_kitchen_%j.log
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz
PROJECT="$ROOT/hrc-v2"
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
OUT_DIR="$PROJECT/outputs/isaac_human/kitchen_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT_DIR" "$ROOT/.cache/robocasa-kit-home" "$ROOT/.cache/robocasa-kit-cache" "$ROOT/.cache/robocasa-kit-logs"
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_PRIVACY_CONSENT=Y APPTAINERENV_ISAAC_PATH=/isaac-sim APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONUNBUFFERED=1
cd "$PROJECT"
apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
    --home "$ROOT/.cache/robocasa-kit-home:/root" \
    --bind "$ROOT/.cache/robocasa-kit-cache:/isaac-sim/kit/cache" \
    --bind "$ROOT/.cache/robocasa-kit-logs:/isaac-sim/kit/logs" \
    "$RUNTIME" /isaac-sim/python.sh -m isaac_human.run_kitchen_cosim \
    --headless --enable_cameras "$@" --output_dir "$OUT_DIR"
test -s "$OUT_DIR/preview.mp4"
python3 - "$OUT_DIR/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result['status'] in ('passed', 'diagnostic_passed'), result
assert all(result['checks'].values()), result
print(result['status'])
PY
