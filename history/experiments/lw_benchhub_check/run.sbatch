#!/usr/bin/env bash
#SBATCH --job-name=lw-source-check
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/lw-benchhub-check/gpu_%j.log
set -euo pipefail
ulimit -c 0
CHECK_ROOT=/scratch/jiabenchen_umass/yz/lw-benchhub-check
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1
export ISAACLAB_PATH="$CHECK_ROOT/source/third_party/IsaacLab-Arena/submodules/IsaacLab"
export LD_LIBRARY_PATH="$CHECK_ROOT/runtime/lib/python3.11/site-packages/cmeel.prefix/lib:${LD_LIBRARY_PATH:-}"
CAMERA_ARGS=()
if [[ "${LW_ENABLE_CAMERAS:-1}" == 1 ]]; then CAMERA_ARGS+=(--enable_cameras); fi
cd "$CHECK_ROOT/source"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
apptainer exec --cleanenv --nv --bind /scratch:/scratch --bind /work:/work \
    /scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif \
    env -u PYTHONPATH -u PYTHONHOME -u ISAAC_PATH -u EXP_PATH -u CARB_APP_PATH -u CONDA_PREFIX \
    ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1 \
    ISAACLAB_PATH="$ISAACLAB_PATH" \
    LD_LIBRARY_PATH="$CHECK_ROOT/runtime/lib/python3.11/site-packages/cmeel.prefix/lib:/.singularity.d/libs" \
    "$CHECK_ROOT/runtime/bin/python" "$CHECK_ROOT/smoke.py" --headless "${CAMERA_ARGS[@]}" \
    --device cuda:0 --output "$CHECK_ROOT/results/$SLURM_JOB_ID" "$@"
