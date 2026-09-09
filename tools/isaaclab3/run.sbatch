#!/usr/bin/env bash
#SBATCH --job-name=lw-lab3-camera
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/lw-benchhub-lab3/gpu_%j.log
set -euo pipefail
ROOT=${HRC_PROJECT_ROOT:-/scratch/jiabenchen_umass/yz/hrc-v2}
STATE="$ROOT/external/isaaclab3"
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
mkdir -p "$STATE/home" "$STATE/cache" "$STATE/logs"
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_PRIVACY_CONSENT=Y APPTAINERENV_ISAAC_PATH=/isaac-sim
export APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1 APPTAINERENV_PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "$ROOT"
nvidia-smi --query-gpu=name,driver_version --format=csv
apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
    --home "$STATE/home:/root" --bind "$STATE/cache:/isaac-sim/kit/cache" \
    --bind "$STATE/logs:/isaac-sim/kit/logs" \
    "$RUNTIME" /isaac-sim/python.sh "$ROOT/tools/isaaclab3/smoke.py" \
    --headless --enable_cameras --device cuda:0 --output "$ROOT/outputs/lightwheel/$SLURM_JOB_ID" "$@"
