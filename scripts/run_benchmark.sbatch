#!/usr/bin/env bash
#SBATCH --job-name=hrc-lightwheel
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/hrc_benchmark_%j.log
set -euo pipefail
PROJECT=${HRC_PROJECT_ROOT:-/scratch/jiabenchen_umass/yz/hrc-v2}
LW_ROOT="$PROJECT/external/isaaclab3"
mkdir -p "$LW_ROOT/home" "$LW_ROOT/cache" "$LW_ROOT/logs"
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_PRIVACY_CONSENT=Y APPTAINERENV_ISAAC_PATH=/isaac-sim
export APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1 APPTAINERENV_PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "$PROJECT"
OUT_DIR="$PROJECT/outputs/benchmark/$SLURM_JOB_ID"
apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
    --home "$LW_ROOT/home:/root" --bind "$LW_ROOT/cache:/isaac-sim/kit/cache" \
    --bind "$LW_ROOT/logs:/isaac-sim/kit/logs" \
    "$RUNTIME" /isaac-sim/python.sh -m hrc_bench --headless --enable_cameras \
    --lightwheel-root "$PROJECT" --output "$OUT_DIR" "$@"
python3 - "$OUT_DIR/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['status'] == 'passed', r
PY
