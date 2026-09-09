#!/usr/bin/env bash
#SBATCH --job-name=lw-lab3-camera
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/lw-benchhub-lab3/gpu_%j.log
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz/lw-benchhub-lab3
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
mkdir -p "$ROOT/home" "$ROOT/cache" "$ROOT/logs"
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_PRIVACY_CONSENT=Y APPTAINERENV_ISAAC_PATH=/isaac-sim
export APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1 APPTAINERENV_PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "$ROOT/source"
nvidia-smi --query-gpu=name,driver_version --format=csv
apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
    --home "$ROOT/home:/root" --bind "$ROOT/cache:/isaac-sim/kit/cache" \
    --bind "$ROOT/logs:/isaac-sim/kit/logs" \
    "$RUNTIME" /isaac-sim/python.sh "$ROOT/smoke.py" \
    --headless --enable_cameras --device cuda:0 --output "$ROOT/results/$SLURM_JOB_ID" "$@"
