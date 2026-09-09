#!/usr/bin/env bash
#SBATCH --job-name=robocasa-probe
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/robocasa_probe_%j.log
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz
PROJECT="$ROOT/hrc-v2"
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
mkdir -p "$ROOT/.cache/robocasa-kit-home" "$ROOT/.cache/robocasa-kit-cache" "$ROOT/.cache/robocasa-kit-logs"
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_ISAAC_PATH=/isaac-sim APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1
export PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "$PROJECT"
output="outputs/robocasa_migration/probes/${SLURM_JOB_ID}"
mode="${ROBOCASA_PROBE_MODE:-asset}"
tasks=("${@}")
if [ ${#tasks[@]} -eq 0 ]; then tasks=(PickPlaceCounterToSink OpenCabinet OpenMicrowave); fi
failed=0
for task in "${tasks[@]}"; do
    extra=()
    extra+=(--migration_root "${ROBOCASA_MIGRATION_ROOT:-outputs/robocasa_migration}")
    if [ -n "${ROBOCASA_DEMONSTRATION:-}" ]; then extra+=(--demonstration "$ROBOCASA_DEMONSTRATION"); fi
    if [ "${ROBOCASA_DEMONSTRATION_FEEDBACK:-0}" = 1 ]; then extra+=(--demonstration_feedback); fi
    if [ "${ROBOCASA_SOURCE_CONTACT_FRICTION:-0}" = 1 ]; then extra+=(--source_contact_friction); fi
    camera_args=(--enable_cameras)
    if [ "${ROBOCASA_NO_CAMERA:-0}" = 1 ]; then camera_args=(--no_camera); fi
    extra+=(--sample_every "${ROBOCASA_SAMPLE_EVERY:-8}")
    extra+=(--mobile_friction_scale "${ROBOCASA_MOBILE_FRICTION_SCALE:-1}")
    if [ "${ROBOCASA_SERVE_TEA_CONTACT_PROBE:-0}" = 1 ]; then extra+=(--serve_tea_contact_probe); fi
    if [ "$mode" = robot ]; then
        extra+=(--robot_attempt --steps "${ROBOCASA_ROBOT_STEPS:-1800}" --hold_steps "${ROBOCASA_HOLD_STEPS:-0}")
    elif [ "$task" != PickPlaceCounterToSink ] && [ "$task" != ServeTea ]; then
        extra+=(--joint_probe)
    fi
    apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
        --home "$ROOT/.cache/robocasa-kit-home:/root" \
        --bind "$ROOT/.cache/robocasa-kit-cache:/isaac-sim/kit/cache" \
        --bind "$ROOT/.cache/robocasa-kit-logs:/isaac-sim/kit/logs" \
        "$RUNTIME" /isaac-sim/python.sh robocasa_migration/simulate_scene.py \
        --headless "${camera_args[@]}" --task "$task" --output "$output" "${extra[@]}" || failed=1
    python3 - "$output/$task/result.json" <<'PY' || failed=1
import json,sys
assert json.load(open(sys.argv[1]))['status'] == 'simulated'
PY
done
exit "$failed"
